import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from ocr_common import database
from ocr_common.errors import ServiceError
from ocr_common.jobs import STAGE_OCR, STAGE_STRUCTURING, StagePipeline
from ocr_common.jobs_sql import SqlJobRepository
from ocr_common.outbox import KIND_CALLBACK, KIND_HANDOFF, OutboxRelay, SqlOutbox, callback_message
from ocr_common.testing import RecordingCallback

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


async def _run(pipeline, *, fails: bool = False) -> None:
    async def work():
        if fails:
            raise ServiceError(503, "ekstraksi OCR model is unavailable")
        return {"full_text": "NPWP"}

    await pipeline.submit(
        RID,
        work,
        handoff_payload=lambda result: PAYLOAD,
        next_stage=STAGE_STRUCTURING,
        callback_result=lambda result: {"final": True},
    )
    await pipeline.runner.drain(5)


async def test_finishing_a_job_queues_the_callback_and_the_handoff(pipeline):
    stage, callback = pipeline
    await _run(stage)

    assert callback.calls == []
    rows = await _rows(stage.outbox)
    assert [(row["kind"], row["stage"], row["attempts"]) for row in rows] == [
        (KIND_CALLBACK, STAGE_OCR, 0),
        (KIND_HANDOFF, STAGE_OCR, 0),
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
        "ekstraksi OCR model is unavailable",
    )
    assert (await stage.repository.get(RID))["status"] == "FAILED"


async def test_the_relay_delivers_in_order_and_clears_the_rows(pipeline):
    stage, _ = pipeline
    await _run(stage)
    orchestration, next_stage = Sink(), Sink()
    relay = OutboxRelay(stage.outbox, stage=STAGE_OCR, callback=orchestration, next_stage=next_stage)

    assert await relay.deliver_due() == 2

    assert orchestration.bodies[0]["status"] == "DONE"
    assert next_stage.bodies == [PAYLOAD]
    assert await _rows(stage.outbox) == []


async def test_a_callback_the_orchestrator_cannot_take_is_retried_without_holding_up_the_handoff(pipeline):
    stage, _ = pipeline
    await _run(stage)
    next_stage = Sink()
    relay = OutboxRelay(
        stage.outbox,
        stage=STAGE_OCR,
        callback=Sink(ServiceError(503, "orchestration is unavailable")),
        next_stage=next_stage,
    )

    assert await relay.deliver_due() == 1

    rows = await _rows(stage.outbox)
    assert [row["kind"] for row in rows] == [KIND_CALLBACK]
    assert rows[0]["attempts"] == 1
    assert next_stage.bodies == [PAYLOAD]


async def test_a_handoff_the_next_stage_refuses_becomes_a_failed_callback(pipeline):
    stage, _ = pipeline
    await _run(stage)
    orchestration = Sink()
    relay = OutboxRelay(
        stage.outbox,
        stage=STAGE_OCR,
        callback=orchestration,
        next_stage=Sink(ServiceError(400, "Unknown document_type")),
    )

    await relay.deliver_due()

    [row] = await _rows(stage.outbox)
    assert row["kind"] == KIND_CALLBACK
    assert row["payload"]["stage"] == STAGE_STRUCTURING
    assert row["payload"]["status"] == "FAILED"
    assert "Handoff to STRUCTURING failed: Unknown document_type" == row["payload"]["error_message"]

    assert await relay.deliver_due() == 1
    assert await _rows(stage.outbox) == []


async def test_a_relay_only_claims_the_messages_of_its_own_stage(pipeline):
    stage, _ = pipeline
    await _run(stage)
    async with database.get_engine(stage.outbox._url).begin() as conn:
        await stage.outbox.add(conn, RID, STAGE_STRUCTURING, [callback_message(RID, STAGE_STRUCTURING, "DONE")])
    next_stage = Sink()
    relay = OutboxRelay(stage.outbox, stage=STAGE_OCR, callback=Sink(), next_stage=next_stage)

    await relay.deliver_due()

    assert [row["stage"] for row in await _rows(stage.outbox)] == [STAGE_STRUCTURING]
    assert next_stage.bodies == [PAYLOAD]


async def test_the_messages_and_the_job_are_written_in_one_transaction(pipeline):
    stage, _ = pipeline
    async with stage.repository.engine.begin() as conn:
        await conn.run_sync(stage.outbox.table.drop)

    async def work():
        return {"full_text": "NPWP"}

    with pytest.raises(OperationalError):
        await stage.repository.complete(
            RID, {"full_text": "NPWP"}, messages=stage._messages(RID, None, PAYLOAD, STAGE_STRUCTURING)
        )

    assert await stage.repository.get(RID) is None
