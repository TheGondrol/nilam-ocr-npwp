"""Overload protection in the orchestrator: admission control on POST extract-ocr (429 with Retry-After), a busy
stage's Retry-After passed on to the caller, and the polling of the wait backing off."""

import pytest

from ocr_common.errors import ServiceBusy, TooManyRequests
from ocr_common.overload import Bulkhead
from ocr_common.testing import image_upload

from app.dependencies import get_admission
from app.main import app
from app.services import pipeline_waiter
from app.services.pipeline_waiter import PipelineWaiter
from tests.conftest import JPEG


def _submit(client, auth):
    return client.post(
        "/v1/extract-ocr",
        data={"request_id": "OCR_1"},
        files=image_upload("npwp.jpg", JPEG, "image/jpeg"),
        headers=auth,
    )


@pytest.fixture
def admission():
    gate = Bulkhead(
        "orchestrator-admission-test",
        1,
        0,
        retry_after=2.0,
        error=lambda message, after: TooManyRequests(message, retry_after=after),
    )
    app.dependency_overrides[get_admission] = lambda: gate
    yield gate
    app.dependency_overrides.pop(get_admission, None)


def test_beyond_max_in_flight_is_429_before_anything_starts(client, auth, admission, stub_guardrails, stub_extraction):
    held = admission.reserve()  # one request already held: the limit is 1

    response = _submit(client, auth)

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "2"
    assert response.json()["errors"] == "TOO_MANY_REQUESTS"
    assert stub_guardrails.checked == [] and stub_extraction.submitted == []

    held.release()
    assert _submit(client, auth).status_code == 200
    assert admission.reserved == 0


def test_a_busy_extraction_is_503_with_its_retry_after(client, auth, stub_extraction):
    stub_extraction.error = ServiceBusy("OCR is busy", retry_after=7)

    response = _submit(client, auth)

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "7"
    body = response.json()
    assert (body["errors"], body["pipeline_last_stage"]) == ("DOWNSTREAM_UNAVAILABLE", "extraction")


class SlowStage:
    """A stage whose job stays PROCESSING for `polls` reads, then is DONE."""

    stage = "OCR"

    def __init__(self, polls: int):
        self.reads = 0
        self.polls = polls

    async def get(self, request_id):
        self.reads += 1
        if self.reads > self.polls:
            return {"status": "DONE", "result": {}}
        return {"status": "PROCESSING"}


async def test_wait_polls_less_and_less_often(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(pipeline_waiter.asyncio, "sleep", fake_sleep)
    stage = SlowStage(polls=6)
    waiter = PipelineWaiter([stage], poll_interval=0.25, max_poll_interval=2.0)

    outcome = await waiter.wait("REQ_1", 60)

    assert outcome.status == "DONE"
    assert stage.reads == 7
    expected = [0.25, 0.5, 1.0, 2.0, 2.0, 2.0]
    assert len(slept) == len(expected)
    assert all(0.9 * want <= got <= 1.1 * want for got, want in zip(slept, expected, strict=True))


async def test_wait_without_a_maximum_keeps_a_fixed_interval(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(pipeline_waiter.asyncio, "sleep", fake_sleep)
    await PipelineWaiter([SlowStage(polls=3)], poll_interval=0.5).wait("REQ_1", 60)
    assert all(0.45 <= got <= 0.55 for got in slept)
