"""
Mesin tahap pipeline (ocr_common/jobs.py). Kedua repository diuji dengan
kontrak yang sama; yang SQL terhadap SQLite (aiosqlite) sungguhan supaya
ON CONFLICT, upsert, dan join-nya benar-benar jalan.
"""

import httpx
import pytest

from ocr_common import database
from ocr_common.errors import ServiceError
from ocr_common.jobs import (
    STAGE_OCR,
    STAGE_STRUCTURING,
    InMemoryJobRepository,
    NextStageClient,
    OrchestrationCallback,
    StagePipeline,
)
from ocr_common.jobs_sql import SqlJobRepository
from ocr_common.remote import RemoteModelClient
from ocr_common.testing import RecordingCallback


@pytest.fixture(params=["memory", "sql"])
async def repository(request, tmp_path):
    if request.param == "memory":
        yield InMemoryJobRepository()
        return
    await database.dispose_engines()
    repo = SqlJobRepository(f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}", "ocr")
    async with repo.engine.begin() as conn:
        await conn.run_sync(repo.metadata.create_all)
    yield repo
    await database.dispose_engines()


async def test_claim_is_idempotent(repository):
    assert await repository.claim("REQ_1") is True
    assert await repository.claim("REQ_1") is False
    record = await repository.get("REQ_1")
    assert record["status"] == "PROCESSING"
    assert record["result"] is None
    assert record["created_at"].endswith("+00:00")


async def test_complete_stores_result_and_is_not_reclaimable(repository):
    await repository.claim("REQ_2")
    await repository.complete("REQ_2", {"blocks": [{"text": "NPWP"}]})
    record = await repository.get("REQ_2")
    assert record["status"] == "DONE"
    assert record["result"] == {"blocks": [{"text": "NPWP"}]}
    assert await repository.claim("REQ_2") is False


async def test_failed_job_can_be_claimed_again(repository):
    await repository.claim("REQ_3")
    await repository.fail("REQ_3", "ekstraksi OCR model is unavailable")
    record = await repository.get("REQ_3")
    assert (record["status"], record["error_message"]) == ("FAILED", "ekstraksi OCR model is unavailable")

    assert await repository.claim("REQ_3") is True
    record = await repository.get("REQ_3")
    assert (record["status"], record["error_message"]) == ("PROCESSING", None)
    # Hasil percobaan kedua menimpa (UPSERT), bukan bentrok primary key.
    await repository.complete("REQ_3", {"attempt": 2})
    assert (await repository.get("REQ_3"))["result"] == {"attempt": 2}


async def test_get_unknown_returns_none(repository):
    assert await repository.get("REQ_missing") is None


def _pipeline(repository) -> tuple[StagePipeline, RecordingCallback]:
    callback = RecordingCallback()
    return StagePipeline(stage=STAGE_OCR, repository=repository, callback=callback), callback


async def test_pipeline_success_writes_result_then_callback_then_handoff(repository):
    pipeline, callback = _pipeline(repository)
    events: list[str] = []

    async def work():
        return {"full_text": "NPWP"}

    async def handoff(result):
        # Urutan diagram: hasil sudah tersimpan dan callback sudah terkirim sebelum handoff.
        assert (await repository.get("REQ_10"))["status"] == "DONE"
        assert [c["status"] for c in callback.calls] == ["DONE"]
        events.append(result["full_text"])

    accepted = await pipeline.submit("REQ_10", work, handoff=handoff, next_stage=STAGE_STRUCTURING)
    assert accepted == {"request_id": "REQ_10", "stage": "OCR", "status": "PROCESSING", "duplicate": False}
    await pipeline.runner.drain(5)

    assert events == ["NPWP"]
    assert callback.calls == [
        {"request_id": "REQ_10", "stage": "OCR", "status": "DONE", "result": None, "error_message": None}
    ]


async def test_pipeline_duplicate_does_not_run_work_twice(repository):
    pipeline, callback = _pipeline(repository)
    runs = 0

    async def work():
        nonlocal runs
        runs += 1
        return {}

    await pipeline.submit("REQ_11", work)
    await pipeline.runner.drain(5)
    second = await pipeline.submit("REQ_11", work)
    await pipeline.runner.drain(5)

    assert second["duplicate"] is True
    assert second["status"] == "DONE"
    assert runs == 1
    assert len(callback.calls) == 1


async def test_pipeline_work_failure_is_recorded_and_reported(repository):
    pipeline, callback = _pipeline(repository)
    handed_off = False

    async def work():
        raise ServiceError(503, "ekstraksi OCR model is unavailable")

    async def handoff(result):
        nonlocal handed_off
        handed_off = True

    await pipeline.submit("REQ_12", work, handoff=handoff, next_stage=STAGE_STRUCTURING)
    await pipeline.runner.drain(5)

    assert handed_off is False
    assert (await repository.get("REQ_12"))["status"] == "FAILED"
    assert callback.calls[0]["stage"] == "OCR"
    assert callback.calls[0]["status"] == "FAILED"
    assert callback.calls[0]["error_message"] == "ekstraksi OCR model is unavailable"


async def test_pipeline_unexpected_exception_does_not_leak_details(repository):
    pipeline, callback = _pipeline(repository)

    async def work():
        raise RuntimeError("secret internals")

    await pipeline.submit("REQ_13", work)
    await pipeline.runner.drain(5)
    assert callback.calls[0]["error_message"] == "Internal error in OCR stage"


async def test_pipeline_handoff_failure_reports_next_stage_failed(repository):
    pipeline, callback = _pipeline(repository)

    async def work():
        return {}

    async def handoff(result):
        raise ServiceError(503, "structuring service is unavailable")

    await pipeline.submit("REQ_14", work, handoff=handoff, next_stage=STAGE_STRUCTURING)
    await pipeline.runner.drain(5)

    # Tahap ini tetap DONE; yang gagal adalah tahap berikutnya.
    assert (await repository.get("REQ_14"))["status"] == "DONE"
    assert [(c["stage"], c["status"]) for c in callback.calls] == [("OCR", "DONE"), ("STRUCTURING", "FAILED")]
    assert "structuring service is unavailable" in callback.calls[1]["error_message"]


async def test_pipeline_callback_result_is_sent_for_final_stage(repository):
    pipeline, callback = _pipeline(repository)

    async def work():
        return {"score": 0.9}

    await pipeline.submit("REQ_15", work, callback_result=lambda result: {"final": result["score"]})
    await pipeline.runner.drain(5)
    assert callback.calls[0]["result"] == {"final": 0.9}


async def test_pipeline_get_unknown_is_404(repository):
    pipeline, _ = _pipeline(repository)
    with pytest.raises(ServiceError) as exc:
        await pipeline.get("REQ_missing")
    assert exc.value.status_code == 404


def _client(handler) -> RemoteModelClient:
    return RemoteModelClient(
        "http://orkestrasi",
        1.0,
        name="orchestration callback",
        passthrough_client_errors=True,
        transport=httpx.MockTransport(handler),
    )


async def test_callback_posts_payload_and_retries_5xx():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(request.content))
        return httpx.Response(503 if len(seen) == 1 else 200, json={})

    callback = OrchestrationCallback(_client(handler), "/v1/callbacks/stage", attempts=3, delay=0)
    assert await callback.notify("REQ_20", "OCR", "DONE") is True
    assert len(seen) == 2
    assert seen[0] == {
        "request_id": "REQ_20",
        "stage": "OCR",
        "status": "DONE",
        "result": None,
        "error_message": None,
    }


async def test_callback_failure_is_swallowed_and_4xx_not_retried():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, json={"message": "unknown request_id"})

    callback = OrchestrationCallback(_client(handler), "/v1/callbacks/stage", attempts=3, delay=0)
    assert await callback.notify("REQ_21", "OCR", "DONE") is False
    assert calls == 1


async def test_callback_without_url_is_skipped():
    assert await OrchestrationCallback(None, "/v1/callbacks/stage").notify("REQ_22", "OCR", "DONE") is False


async def test_next_stage_gives_up_after_attempts():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, json={"message": "boom"})

    next_stage = NextStageClient(_client(handler), "/v1/structuring/jobs", attempts=3, delay=0)
    with pytest.raises(ServiceError):
        await next_stage.submit({"request_id": "REQ_23"})
    assert calls == 3
