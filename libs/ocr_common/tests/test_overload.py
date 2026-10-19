"""Overload protection (ocr_common.overload) and where it is wired in: the bulkhead of a stage pipeline, the
concurrency limit, total timeout and circuit breaker of RemoteModelClient, jittered retries, and the Retry-After
header of the error envelope."""

import asyncio
import random

import httpx
import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from ocr_common.clients.remote import RemoteClientError, RemoteModelClient
from ocr_common.config import BaseServiceSettings
from ocr_common.errors import (
    SERVICE_BUSY,
    InternalError,
    ServiceBusy,
    TooManyRequests,
    UpstreamTimeout,
    UpstreamUnavailable,
)
from ocr_common.overload import Bulkhead, CircuitBreaker, full_jitter, retry_after_header, spread
from ocr_common.pipeline import STAGE_OCR, InMemoryJobRepository, StagePipeline
from ocr_common.pipeline.callbacks import not_ready_delay, with_retry
from ocr_common.testing import RecordingCallback
from ocr_common.web.app import create_app

# --- jitter ----------------------------------------------------------------------------------------------------


def test_full_jitter_stays_in_the_window_and_spreads():
    rng = random.Random(7)
    waits = [full_jitter(4.0, rng) for _ in range(500)]
    assert all(0.0 <= w <= 4.0 for w in waits)
    assert min(waits) < 0.5 and max(waits) > 3.5


def test_spread_is_around_the_interval():
    rng = random.Random(7)
    sleeps = [spread(10.0, 0.2, rng) for _ in range(500)]
    assert all(8.0 <= s <= 12.0 for s in sleeps)


def test_not_ready_delay_is_one_to_two_seconds():
    """The central orchestrator's contract: send a RESULT_NOT_READY callback again after 1-2 s."""
    assert all(1.0 <= not_ready_delay() <= 2.0 for _ in range(200))


async def test_with_retry_waits_a_jittered_time(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    calls = 0

    async def flaky():
        nonlocal calls
        calls += 1
        raise UpstreamUnavailable("down")

    with pytest.raises(UpstreamUnavailable):
        await with_retry(flaky, attempts=4, delay=0.5)
    assert calls == 4
    assert len(slept) == 3
    assert all(0.0 <= s <= cap for s, cap in zip(slept, (0.5, 1.0, 2.0), strict=True))


def test_retry_after_header_rounds_up_to_whole_seconds():
    assert retry_after_header(None) is None
    assert retry_after_header(0.2) == "1"
    assert retry_after_header(4.1) == "5"


# --- bulkhead --------------------------------------------------------------------------------------------------


async def test_bulkhead_runs_limit_queues_the_rest_and_refuses_beyond():
    bulkhead = Bulkhead("t", limit=2, queue=1, retry_after=3.0)
    gate = asyncio.Event()
    running = 0
    peak = 0

    async def job(ticket):
        nonlocal running, peak
        async with ticket:
            running += 1
            peak = max(peak, running)
            await gate.wait()
            running -= 1

    tasks = [asyncio.create_task(job(bulkhead.reserve())) for _ in range(3)]
    await asyncio.sleep(0)
    assert (bulkhead.running, bulkhead.reserved) == (2, 3)
    with pytest.raises(ServiceBusy) as caught:
        bulkhead.reserve()
    assert caught.value.status_code == 503
    assert caught.value.code == SERVICE_BUSY
    assert caught.value.retry_after == 3.0
    # A job that is already claimed is never refused.
    forced = bulkhead.reserve(force=True)
    forced.release()

    gate.set()
    await asyncio.gather(*tasks)
    assert peak == 2
    assert (bulkhead.running, bulkhead.reserved) == (0, 0)


async def test_bulkhead_without_limit_never_refuses():
    bulkhead = Bulkhead("t", limit=0)
    tickets = [bulkhead.reserve() for _ in range(1000)]
    for ticket in tickets:
        async with ticket:
            pass
    assert bulkhead.reserved == 0


async def test_released_ticket_frees_its_place_once():
    bulkhead = Bulkhead("t", limit=1, queue=0)
    ticket = bulkhead.reserve()
    ticket.release()
    ticket.release()
    assert bulkhead.reserved == 0
    bulkhead.reserve().release()


async def test_bulkhead_error_can_be_429():
    bulkhead = Bulkhead(
        "admission", limit=1, queue=0, retry_after=2.0, error=lambda m, after: TooManyRequests(m, retry_after=after)
    )
    bulkhead.reserve()
    with pytest.raises(TooManyRequests) as caught:
        bulkhead.reserve()
    assert caught.value.status_code == 429


async def test_cancelled_while_waiting_gives_the_place_back():
    bulkhead = Bulkhead("t", limit=1, queue=5)
    holder = bulkhead.reserve()
    await holder.__aenter__()
    waiter = bulkhead.reserve()

    async def wait():
        async with waiter:
            pass

    task = asyncio.create_task(wait())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert bulkhead.reserved == 1
    await holder.__aexit__(None, None, None)
    assert bulkhead.reserved == 0


# --- stage pipeline with a bulkhead -----------------------------------------------------------------------------


async def test_pipeline_bulkhead_queues_then_refuses_before_claiming():
    repository = InMemoryJobRepository()
    pipeline = StagePipeline(
        stage=STAGE_OCR,
        repository=repository,
        callback=RecordingCallback(),
        callbacks=False,
        bulkhead=Bulkhead("OCR", limit=1, queue=1, retry_after=4.0),
    )
    gate = asyncio.Event()
    started: list[str] = []

    def work_for(rid):
        async def work():
            started.append(rid)
            await gate.wait()
            return {"ok": rid}

        return work

    await pipeline.submit("REQ_a", work_for("REQ_a"))
    await pipeline.submit("REQ_b", work_for("REQ_b"))
    await asyncio.sleep(0.05)
    assert started == ["REQ_a"]  # REQ_b waits for the one slot

    with pytest.raises(ServiceBusy) as caught:
        await pipeline.submit("REQ_c", work_for("REQ_c"))
    assert caught.value.retry_after == 4.0
    assert await repository.get("REQ_c") is None  # refused before the claim: nothing stored

    gate.set()
    await pipeline.runner.drain(5)
    assert started == ["REQ_a", "REQ_b"]
    record = await repository.get("REQ_b")
    assert record is not None and record["status"] == "DONE"
    assert pipeline.bulkhead.reserved == 0


async def test_pipeline_duplicate_submit_gives_its_place_back():
    repository = InMemoryJobRepository()
    pipeline = StagePipeline(
        stage=STAGE_OCR,
        repository=repository,
        callback=RecordingCallback(),
        callbacks=False,
        bulkhead=Bulkhead("OCR", limit=1, queue=0),
    )

    async def work():
        return {}

    await pipeline.submit("REQ_dup", work)
    await pipeline.runner.drain(5)
    answer = await pipeline.submit("REQ_dup", work)
    assert answer["duplicate"] is True
    assert pipeline.bulkhead.reserved == 0


# --- circuit breaker -------------------------------------------------------------------------------------------


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_breaker_opens_after_failures_and_probes_after_reset():
    clock = Clock()
    breaker = CircuitBreaker("ocr", failures=3, reset_seconds=30, clock=clock)
    for _ in range(2):
        breaker.before_call()
        breaker.record(UpstreamTimeout("slow"))
    assert breaker.state == "closed"
    breaker.before_call()
    breaker.record(UpstreamTimeout("slow"))
    assert breaker.state == "open"

    with pytest.raises(UpstreamUnavailable) as caught:
        breaker.before_call()
    retry_after = caught.value.retry_after
    assert retry_after is not None and 0 < retry_after <= 30

    clock.now += 31
    breaker.before_call()  # the probe
    with pytest.raises(UpstreamUnavailable):
        breaker.before_call()  # a second caller while probing
    breaker.record(InternalError("still broken"))
    assert breaker.state == "open"

    clock.now += 31
    breaker.before_call()
    breaker.record(None)
    assert breaker.state == "closed"
    breaker.before_call()


def test_breaker_counts_only_5xx_and_a_success_resets():
    breaker = CircuitBreaker("ocr", failures=2, reset_seconds=30, clock=Clock())
    breaker.record(UpstreamTimeout("slow"))
    breaker.record(RemoteClientError(400, "bad file"))  # the remote answered properly
    breaker.record(UpstreamTimeout("slow"))
    assert breaker.state == "closed"


def test_breaker_off_never_refuses():
    breaker = CircuitBreaker("ocr", failures=0)
    for _ in range(10):
        breaker.before_call()
        breaker.record(UpstreamTimeout("slow"))
    breaker.before_call()


# --- RemoteModelClient -------------------------------------------------------------------------------------------


async def test_remote_client_limits_calls_in_flight():
    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return httpx.Response(200, json={"ok": True})

    client = RemoteModelClient(
        "http://model.test", 5, name="ocr", transport=httpx.MockTransport(handler), max_concurrency=2
    )
    await asyncio.gather(*(client.get_json("/x") for _ in range(8)))
    assert peak == 2


async def test_remote_client_total_timeout_is_504():
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(200, json={})

    client = RemoteModelClient(
        "http://model.test", 30, name="ocr", transport=httpx.MockTransport(handler), total_timeout=0.05
    )
    with pytest.raises(UpstreamTimeout) as caught:
        await client.get_json("/slow")
    assert "within 0.05s" in caught.value.message


async def test_remote_client_breaker_stops_calling_a_failing_model():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(502, text="bad gateway")

    client = RemoteModelClient(
        "http://model.test",
        5,
        name="ocr",
        transport=httpx.MockTransport(handler),
        breaker=CircuitBreaker("ocr-test", failures=2, reset_seconds=60),
    )
    for _ in range(2):
        with pytest.raises(InternalError):
            await client.get_json("/x")
    with pytest.raises(UpstreamUnavailable) as caught:
        await client.get_json("/x")
    assert "circuit open" in caught.value.message
    assert calls == 2


async def test_remote_client_keeps_retry_after_of_the_remote():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"errors": "SERVICE_BUSY", "message": "busy"}, headers={"Retry-After": "7"})

    client = RemoteModelClient(
        "http://stage.test", 5, name="stage", transport=httpx.MockTransport(handler), passthrough_statuses=(503,)
    )
    with pytest.raises(RemoteClientError) as caught:
        await client.get_json("/x")
    assert (caught.value.status_code, caught.value.retry_after, caught.value.remote_code) == (503, 7.0, "SERVICE_BUSY")


# --- error envelope ----------------------------------------------------------------------------------------------


def test_error_envelope_carries_retry_after():
    router = APIRouter()

    @router.get("/v1/busy")
    async def busy():
        raise ServiceBusy("too much", retry_after=4.2)

    @router.get("/v1/limited")
    async def limited():
        raise TooManyRequests("slow down", retry_after=2)

    app = create_app(
        settings=BaseServiceSettings(api_key="k", _env_file=None), title="Demo", description="demo", routers=[router]
    )
    client = TestClient(app, raise_server_exceptions=False)
    busy_response = client.get("/v1/busy", headers={"X-API-Key": "k"})
    assert busy_response.status_code == 503
    assert busy_response.headers["Retry-After"] == "5"
    assert busy_response.json()["errors"] == SERVICE_BUSY
    limited_response = client.get("/v1/limited", headers={"X-API-Key": "k"})
    assert (limited_response.status_code, limited_response.headers["Retry-After"]) == (429, "2")
    assert limited_response.json()["errors"] == "TOO_MANY_REQUESTS"
