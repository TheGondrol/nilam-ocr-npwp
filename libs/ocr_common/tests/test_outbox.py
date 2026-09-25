import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, update
from sqlalchemy.exc import OperationalError

from ocr_common.config import PipelineSettings
from ocr_common.errors import ServiceError
from ocr_common.pipeline import STAGE_OCR, STAGE_STRUCTURING, InMemoryJobRepository, StagePipeline, database
from ocr_common.pipeline.outbox import KIND_CALLBACK, KIND_HANDOFF, OutboxRelay, callback_message
from ocr_common.pipeline.outbox_sql import SqlOutbox
from ocr_common.pipeline.outbox_status import OutboxStatusResponse, outbox_status, outbox_status_responses
from ocr_common.pipeline.repository_sql import SqlJobRepository
from ocr_common.testing import RecordingCallback, make_client
from ocr_common.web.app import create_app
from ocr_common.web.envelope import envelope
from ocr_common.web.security import verify_api_key

RID = "REQ_outbox"
PAYLOAD = {"request_id": RID, "ocr": {"full_text": "NPWP"}}


class Sink:
    def __init__(self, error: Exception | None = None):
        self.bodies: list[dict] = []
        self.error = error

    async def send(self, body: dict) -> None:
        if self.error is not None:
            raise self.error
        self.bodies.append(body)


@pytest.fixture
async def pipeline(tmp_path):
    await database.dispose_engines()
    url = f"sqlite+aiosqlite:///{tmp_path / 'outbox.db'}"
    outbox = SqlOutbox(url)
    repository = SqlJobRepository(url, "ocr", outbox=outbox, stage=STAGE_OCR)
    async with repository.engine.begin() as conn:
        await conn.run_sync(repository.metadata.create_all)
        await conn.run_sync(outbox.table.metadata.create_all)
    callback = RecordingCallback()
    yield StagePipeline(stage=STAGE_OCR, repository=repository, callback=callback, outbox=outbox), callback
    await database.dispose_engines()


async def _rows(outbox: SqlOutbox) -> list[dict]:
    async with database.get_engine(outbox._url).connect() as conn:
        rows = (await conn.execute(select(outbox.table).order_by(outbox.table.c.id))).mappings().all()
    return [dict(row) for row in rows]


async def _age_all(outbox: SqlOutbox, seconds: float) -> None:
    created = datetime.now(UTC) - timedelta(seconds=seconds)
    async with database.get_engine(outbox._url).begin() as conn:
        await conn.execute(update(outbox.table).values(created_at=created))


async def _run(pipeline, *, fails: bool = False) -> None:
    async def work():
        if fails:
            raise ServiceError(503, "extraction OCR model is unavailable")
        return {"full_text": "NPWP"}

    await pipeline.submit(
        RID,
        work,
        handoff_payload=lambda result: PAYLOAD,
        next_stage=STAGE_STRUCTURING,
        callback_result=lambda result: {"final": True},
    )
    await pipeline.runner.drain(5)


def _relay(stage, *, callback=None, next_stage=None, **kwargs) -> OutboxRelay:
    return OutboxRelay(
        stage.outbox, stage=STAGE_OCR, callback=callback or Sink(), next_stage=next_stage or Sink(), **kwargs
    )


async def test_finishing_a_job_queues_the_callback_and_the_handoff(pipeline):
    stage, callback = pipeline
    await _run(stage)

    assert callback.calls == []
    rows = await _rows(stage.outbox)
    assert [(row["kind"], row["stage"], row["attempts"], row["failed_at"]) for row in rows] == [
        (KIND_CALLBACK, STAGE_OCR, 0, None),
        (KIND_HANDOFF, STAGE_OCR, 0, None),
    ]
    assert rows[0]["payload"] == {
        "request_id": RID,
        "stage": STAGE_OCR,
        "status": "DONE",
        "result": {"final": True},
        "error_message": None,
    }
    assert rows[1]["payload"] == {"next_stage": STAGE_STRUCTURING, "body": PAYLOAD}


async def test_a_failed_job_queues_a_failed_callback(pipeline):
    stage, callback = pipeline
    await _run(stage, fails=True)

    assert callback.calls == []
    [row] = await _rows(stage.outbox)
    assert row["kind"] == KIND_CALLBACK
    assert (row["payload"]["status"], row["payload"]["error_message"]) == (
        "FAILED",
        "extraction OCR model is unavailable",
    )
    assert (await stage.repository.get(RID))["status"] == "FAILED"


async def test_the_relay_is_woken_after_the_job_transaction_committed(pipeline):
    stage, _ = pipeline
    assert not stage.outbox.pending.is_set()

    async with stage.repository.engine.begin() as conn:
        await stage.outbox.add(conn, RID, STAGE_OCR, [callback_message(RID, STAGE_OCR, "DONE")])
        assert not stage.outbox.pending.is_set(), "add must not wake the relay before the commit"

    await _run(stage)
    assert stage.outbox.pending.is_set()


async def test_the_relay_delivers_the_handoff_first_and_clears_the_rows(pipeline):
    stage, _ = pipeline
    await _run(stage)
    orchestration, next_stage = Sink(), Sink()
    relay = _relay(stage, callback=orchestration, next_stage=next_stage)

    assert await relay.deliver_due() == 2

    assert orchestration.bodies[0]["status"] == "DONE"
    assert next_stage.bodies == [PAYLOAD]
    assert await _rows(stage.outbox) == []


async def test_a_callback_the_orchestrator_cannot_take_is_retried_without_holding_up_the_handoff(pipeline):
    stage, _ = pipeline
    await _run(stage)
    next_stage = Sink()
    relay = _relay(stage, callback=Sink(ServiceError(503, "orchestration is unavailable")), next_stage=next_stage)

    assert await relay.deliver_due() == 1

    [row] = await _rows(stage.outbox)
    assert (row["kind"], row["attempts"], row["failed_at"]) == (KIND_CALLBACK, 1, None)
    assert row["last_error"] == "orchestration is unavailable"
    assert row["next_attempt_at"] > datetime.now(UTC).replace(tzinfo=None)
    assert next_stage.bodies == [PAYLOAD]


async def test_a_callback_the_orchestrator_rejects_becomes_a_dead_letter_that_stays(pipeline):
    stage, _ = pipeline
    await _run(stage, fails=True)
    relay = _relay(stage, callback=Sink(ServiceError(422, "stage: unexpected value")))

    assert await relay.deliver_due() == 0

    [row] = await _rows(stage.outbox)
    assert row["failed_at"] is not None
    assert row["last_error"] == "stage: unexpected value"
    assert row["payload"]["status"] == "FAILED"

    assert await relay.deliver_due() == 0, "a dead letter is never claimed again"
    assert (await _rows(stage.outbox))[0]["attempts"] == 1


async def test_a_callback_that_keeps_failing_is_retried_until_it_is_too_old(pipeline):
    stage, _ = pipeline
    await _run(stage, fails=True)
    relay = _relay(stage, callback=Sink(ServiceError(503, "orchestration is unavailable")), max_age_seconds=3600)

    await relay.deliver_due()
    [row] = await _rows(stage.outbox)
    assert row["failed_at"] is None

    await _age_all(stage.outbox, 3601)
    async with database.get_engine(stage.outbox._url).begin() as conn:
        await conn.execute(update(stage.outbox.table).values(next_attempt_at=datetime.now(UTC)))
    await relay.deliver_due()
    [row] = await _rows(stage.outbox)
    assert row["failed_at"] is not None
    assert row["attempts"] == 2


def test_backoff_grows_exponentially_up_to_the_cap():
    relay = OutboxRelay(SqlOutbox("sqlite+aiosqlite://"), stage=STAGE_OCR, callback=Sink(), retry_delay_seconds=0.5)
    assert [relay._backoff(n) for n in (1, 2, 3, 4)] == [0.5, 1.0, 2.0, 4.0]
    assert relay._backoff(30) == 300.0


async def test_a_handoff_the_next_stage_refuses_becomes_a_failed_callback(pipeline):
    stage, _ = pipeline
    await _run(stage)
    orchestration = Sink()
    relay = _relay(stage, callback=orchestration, next_stage=Sink(ServiceError(400, "Unknown document_type")))

    await relay.deliver_due()

    dead, [replacement] = (
        [row for row in await _rows(stage.outbox) if row["failed_at"]],
        [row for row in await _rows(stage.outbox) if not row["failed_at"]],
    )
    assert [row["kind"] for row in dead] == [KIND_HANDOFF]
    assert dead[0]["last_error"] == "Unknown document_type"
    assert replacement["kind"] == KIND_CALLBACK
    assert replacement["payload"]["stage"] == STAGE_STRUCTURING
    assert replacement["payload"]["status"] == "FAILED"
    assert replacement["payload"]["error_message"] == "Handoff to STRUCTURING failed: Unknown document_type"

    assert await relay.deliver_due() == 1
    assert [row["kind"] for row in await _rows(stage.outbox)] == [KIND_HANDOFF]


async def test_a_relay_only_claims_the_messages_of_its_own_stage(pipeline):
    stage, _ = pipeline
    await _run(stage)
    async with database.get_engine(stage.outbox._url).begin() as conn:
        await stage.outbox.add(conn, RID, STAGE_STRUCTURING, [callback_message(RID, STAGE_STRUCTURING, "DONE")])
    next_stage = Sink()
    relay = _relay(stage, next_stage=next_stage)

    await relay.deliver_due()

    assert [row["stage"] for row in await _rows(stage.outbox)] == [STAGE_STRUCTURING]
    assert next_stage.bodies == [PAYLOAD]


async def test_the_messages_and_the_job_are_written_in_one_transaction(pipeline):
    stage, _ = pipeline
    async with stage.repository.engine.begin() as conn:
        await conn.run_sync(stage.outbox.table.drop)

    with pytest.raises(OperationalError):
        await stage.repository.complete(
            RID, {"full_text": "NPWP"}, messages=stage._messages(RID, None, PAYLOAD, STAGE_STRUCTURING)
        )

    assert await stage.repository.get(RID) is None


async def test_stats_count_pending_retrying_and_dead_letters_per_stage(pipeline):
    stage, _ = pipeline
    empty = await stage.outbox.stats(STAGE_OCR)
    assert (empty.pending, empty.retrying, empty.oldest_pending_seconds, empty.dead_letters) == (0, 0, None, 0)

    await _run(stage)
    async with database.get_engine(stage.outbox._url).begin() as conn:
        await stage.outbox.add(conn, RID, STAGE_STRUCTURING, [callback_message(RID, STAGE_STRUCTURING, "DONE")])
    relay = _relay(
        stage,
        callback=Sink(ServiceError(503, "orchestration is unavailable")),
        next_stage=Sink(ServiceError(400, "Unknown document_type")),
    )
    await relay.deliver_due()
    await _age_all(stage.outbox, 120)

    stats = await stage.outbox.stats(STAGE_OCR)
    assert stats.stage == STAGE_OCR
    assert (stats.pending, stats.retrying, stats.dead_letters) == (2, 1, 1)
    assert stats.oldest_pending_seconds is not None and 119 < stats.oldest_pending_seconds < 130
    other = await stage.outbox.stats(STAGE_STRUCTURING)
    assert (other.pending, other.dead_letters) == (1, 0)


async def test_the_relay_warns_about_a_stale_backlog_and_dead_letters(pipeline, caplog):
    stage, _ = pipeline
    await _run(stage)
    relay = _relay(stage, stale_after_seconds=60, watch_interval_seconds=0)

    with caplog.at_level("WARNING", logger="ocr_common.pipeline.outbox"):
        stats = await relay.watch()
        assert stats is not None and stats.pending == 2
        assert "backlog" not in caplog.text
        await _age_all(stage.outbox, 61)
        await relay.watch()
    assert "outbox OCR backlog: 2 pending (0 retrying, oldest 61s), 0 dead letters" in caplog.text


async def test_stop_delivers_what_the_drained_jobs_queued(pipeline):
    stage, _ = pipeline
    orchestration, next_stage = Sink(), Sink()
    relay = _relay(stage, callback=orchestration, next_stage=next_stage, interval_seconds=60)
    relay.start()
    await asyncio.sleep(0)
    await _run(stage)

    await stage.aclose(1, relay=relay)

    assert relay._task is None
    assert next_stage.bodies == [PAYLOAD]
    assert [body["status"] for body in orchestration.bodies] == ["DONE"]
    assert await _rows(stage.outbox) == []


async def test_the_running_relay_delivers_a_new_message_without_waiting_for_the_poll(pipeline):
    stage, _ = pipeline
    orchestration = Sink()
    relay = _relay(stage, callback=orchestration, interval_seconds=60)
    relay.start()
    await asyncio.sleep(0)
    await _run(stage, fails=True)

    for _ in range(50):
        if orchestration.bodies:
            break
        await asyncio.sleep(0.02)
    await relay.stop()
    assert [body["status"] for body in orchestration.bodies] == ["FAILED"]


def _status_app(pipeline_factory):
    """The route as each service declares it in app/api/jobs.py, on top of the shared schema and handler."""
    from fastapi import APIRouter, Depends

    router = APIRouter(dependencies=[Depends(verify_api_key)])

    @router.get("/v1/ocr/outbox", response_model=OutboxStatusResponse, responses=outbox_status_responses("OCR"))
    async def status():
        return envelope(200, "Success", await outbox_status(pipeline_factory()), RID)

    settings = PipelineSettings(api_key="k", environment="local", _env_file=None)
    app = create_app(settings=settings, title="Demo", description="demo", routers=[router])
    return make_client(app)


async def test_outbox_status_reports_the_backlog(pipeline):
    stage, _ = pipeline
    await _run(stage)
    await _relay(stage, callback=Sink(ServiceError(503, "orchestration is unavailable"))).deliver_due()
    client = _status_app(lambda: stage)

    assert client.get("/v1/ocr/outbox").status_code == 401
    body = client.get("/v1/ocr/outbox", headers={"X-API-Key": "k"}).json()
    assert body["status_code"] == 200
    data = body["data"]
    assert (data["enabled"], data["stage"], data["pending"], data["retrying"], data["dead_letters"]) == (
        True,
        STAGE_OCR,
        1,
        1,
        0,
    )
    assert data["oldest_pending_seconds"] >= 0


def test_outbox_status_says_so_when_the_outbox_is_off():
    stage = StagePipeline(stage=STAGE_OCR, repository=InMemoryJobRepository(), callback=RecordingCallback())
    client = _status_app(lambda: stage)

    data = client.get("/v1/ocr/outbox", headers={"X-API-Key": "k"}).json()["data"]
    assert data == {
        "enabled": False,
        "stage": STAGE_OCR,
        "pending": 0,
        "retrying": 0,
        "oldest_pending_seconds": None,
        "dead_letters": 0,
    }


async def test_without_callbacks_only_the_handoff_is_queued(pipeline):
    stage, _ = pipeline
    stage.callbacks = False
    await _run(stage)

    assert [row["kind"] for row in await _rows(stage.outbox)] == [KIND_HANDOFF]


async def test_without_callbacks_a_dead_handoff_is_reported_through_the_hook_not_a_callback(pipeline):
    stage, _ = pipeline
    stage.callbacks = False
    await _run(stage)
    failures: list[tuple[str, str, str]] = []

    async def handoff_failed(request_id: str, next_stage: str, message: str) -> None:
        failures.append((request_id, next_stage, message))

    relay = _relay(
        stage,
        next_stage=Sink(ServiceError(400, "Unknown document_type")),
        callbacks=False,
        handoff_failed=handoff_failed,
    )
    await relay.deliver_due()

    [row] = await _rows(stage.outbox)
    assert (row["kind"], row["failed_at"] is not None) == (KIND_HANDOFF, True)
    assert failures == [(RID, STAGE_STRUCTURING, "Handoff to STRUCTURING failed: Unknown document_type")]


async def test_released_dead_letters_are_claimed_and_delivered_again(pipeline):
    stage, _ = pipeline
    await _run(stage, fails=True)
    await _relay(stage, callback=Sink(ServiceError(422, "stage: unexpected value"))).deliver_due()
    assert (await _rows(stage.outbox))[0]["failed_at"] is not None

    assert await stage.outbox.release(STAGE_OCR, "REQ_lain") == 0
    assert await stage.outbox.release(STAGE_OCR, RID) == 1
    assert stage.outbox.pending.is_set()
    [row] = await _rows(stage.outbox)
    assert row["failed_at"] is None

    orchestration = Sink()
    assert await _relay(stage, callback=orchestration).deliver_due() == 1
    assert [body["status"] for body in orchestration.bodies] == ["FAILED"]
    assert await _rows(stage.outbox) == []


async def test_the_release_handler_answers_409_without_an_outbox(pipeline):
    from ocr_common.pipeline.outbox_status import outbox_release

    stage, _ = pipeline
    assert await outbox_release(stage, None) == {"stage": STAGE_OCR, "request_id": None, "released": 0}

    off = StagePipeline(stage=STAGE_OCR, repository=InMemoryJobRepository(), callback=RecordingCallback())
    with pytest.raises(ServiceError) as raised:
        await outbox_release(off, None)
    assert raised.value.status_code == 409
