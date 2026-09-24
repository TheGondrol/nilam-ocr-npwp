import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from ocr_common.config import PipelineSettings
from ocr_common.pipeline import STAGE_OCR, STAGE_SCORING, StagePipeline, database
from ocr_common.pipeline.outcomes import (
    ApiEventOutcome,
    CompositeOutcome,
    OrchestrationOutcome,
    build_stage_outcome,
)
from ocr_common.pipeline.repository_sql import SqlJobRepository
from ocr_common.pipeline.tables import orchestration_api_events_table, orchestration_outcome_table
from ocr_common.testing import RecordingCallback

TABLE = "orchestration_api_events"
RID = "OCR_api_events"
DATA = {
    "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 1},
    "nama": {"value": "BUDI SANTOSO", "confidence": 0},
}
REASON = "Kode provinsi pada NPWP tidak valid, mohon dicek kembali"


async def _repository(tmp_path, stage: str, outcome_factory):
    await database.dispose_engines()
    table = orchestration_api_events_table(TABLE)
    repo = SqlJobRepository(
        f"sqlite+aiosqlite:///{tmp_path / 'events.db'}",
        stage.lower(),
        outcome=outcome_factory(table, stage),
        stage=stage,
    )
    async with repo.engine.begin() as conn:
        await conn.run_sync(repo.metadata.create_all)
        await conn.run_sync(table.metadata.create_all)
    return repo, table


@pytest.fixture
async def scoring(tmp_path):
    yield await _repository(tmp_path, STAGE_SCORING, lambda table, stage: ApiEventOutcome(table, stage=stage))
    await database.dispose_engines()


async def _rows(repository) -> list[dict]:
    repo, table = repository
    async with repo.engine.connect() as conn:
        return [dict(row) for row in (await conn.execute(select(table).order_by(table.c.id))).mappings().all()]


async def test_claiming_writes_nothing(scoring):
    repo, _ = scoring
    assert await repo.claim(RID) is True
    assert await _rows(scoring) == []


async def test_completing_appends_a_completed_get_ocr_result_row(scoring):
    repo, _ = scoring
    await repo.claim(RID)
    await repo.complete(RID, {"npwp_confidence": 0.98}, outcome_data=DATA)

    [row] = await _rows(scoring)
    assert (row["endpoint"], row["request_id"], row["status_code"]) == ("GET_OCR_RESULT", RID, 200)
    assert (row["downstream_status"], row["downstream_stage"], row["document_type"]) == ("COMPLETED", "SCORING", "npwp")
    assert row["error_code"] is None
    assert row["ds"]
    data = row["result_data"]
    assert data["result"] == {"document_type": "npwp", **DATA, "guardrails": 1}
    assert (data["status"], data["document_type"], data["error_code"], data["error_message"]) == (
        "completed",
        "npwp",
        None,
        None,
    )
    assert data["created_at"] and data["updated_at"]


async def test_completing_without_data_writes_nothing(scoring):
    repo, _ = scoring
    await repo.claim(RID)
    await repo.complete(RID, {"blocks": []})
    assert await _rows(scoring) == []


async def test_a_failed_ocr_stage_is_reported_as_extraction(tmp_path):
    repo, table = await _repository(tmp_path, STAGE_OCR, lambda table, stage: ApiEventOutcome(table, stage=stage))
    try:
        await repo.claim(RID)
        await repo.fail(RID, "ekstraksi OCR model is unavailable")

        [row] = await _rows((repo, table))
        assert (row["status_code"], row["downstream_status"], row["downstream_stage"]) == (422, "FAILED", "EXTRACTION")
        assert row["error_code"] == "OCR_FAILED"
        assert row["result_data"]["result"] is None
        assert row["result_data"]["status"] == "failed"
        assert row["result_data"]["error_message"] == "ekstraksi OCR model is unavailable"
    finally:
        await database.dispose_engines()


async def test_a_failed_handoff_names_the_next_stage(scoring):
    repo, _ = scoring
    await repo.handoff_failed(RID, "STRUCTURING", "Handoff to STRUCTURING failed: structuring service is unavailable")

    [row] = await _rows(scoring)
    assert (row["downstream_status"], row["downstream_stage"], row["error_code"]) == (
        "FAILED",
        "STRUCTURING",
        "STRUCTURING_FAILED",
    )


async def test_a_rejection_appends_a_400_failed_row_with_the_reason(tmp_path):
    repo, table = await _repository(tmp_path, "STRUCTURING", lambda table, stage: ApiEventOutcome(table, stage=stage))
    try:
        await repo.claim(RID)
        await repo.complete(RID, {"reject_reason": REASON}, rejection=REASON)

        [row] = await _rows((repo, table))
        assert (row["status_code"], row["downstream_status"], row["downstream_stage"]) == (400, "FAILED", "STRUCTURING")
        assert row["error_code"] == "DOWNSTREAM_VALIDATION_ERROR"
        assert (row["result_data"]["status"], row["result_data"]["error_message"]) == ("failed", REASON)
        assert row["result_data"]["result"] is None
        record = await repo.get(RID)
        assert record is not None and record["status"] == "DONE"
    finally:
        await database.dispose_engines()


async def test_the_pipeline_records_a_rejection_instead_of_a_completion(tmp_path):
    repo, table = await _repository(tmp_path, "STRUCTURING", lambda table, stage: ApiEventOutcome(table, stage=stage))
    try:
        callback = RecordingCallback()
        pipeline = StagePipeline(stage="STRUCTURING", repository=repo, callback=callback)

        async def work():
            return {"fields": {}, "flag": True, "flag_reason": REASON, "reject_reason": REASON}

        await pipeline.submit(
            RID, work, outcome_data=lambda result: DATA, rejection=lambda result: result.get("reject_reason")
        )
        await pipeline.runner.drain(5)

        [row] = await _rows((repo, table))
        assert (row["status_code"], row["error_code"]) == (400, "DOWNSTREAM_VALIDATION_ERROR")
        assert [(c["status"], c["error_message"], c["error_code"]) for c in callback.calls] == [
            ("FAILED", REASON, "DOWNSTREAM_VALIDATION_ERROR")
        ]
    finally:
        await database.dispose_engines()


async def test_a_second_run_appends_and_the_newest_row_is_the_state(scoring):
    repo, _ = scoring
    await repo.claim(RID)
    await repo.fail(RID, "boom")
    await repo.claim(RID)
    await repo.complete(RID, {"npwp_confidence": 0.98}, outcome_data=DATA)

    rows = await _rows(scoring)
    assert [row["downstream_status"] for row in rows] == ["FAILED", "COMPLETED"]


async def test_the_event_and_the_job_are_written_in_one_transaction(scoring):
    repo, table = scoring
    await repo.claim(RID)
    async with repo.engine.begin() as conn:
        await conn.run_sync(table.drop)

    with pytest.raises(OperationalError):
        await repo.complete(RID, {"npwp_confidence": 0.98}, outcome_data=DATA)

    record = await repo.get(RID)
    assert record is not None
    assert record["status"] == "PROCESSING"
    assert record["result"] is None


async def test_the_pipeline_passes_the_contract_data_of_the_last_stage(scoring):
    repo, _ = scoring
    pipeline = StagePipeline(stage=STAGE_SCORING, repository=repo, callback=RecordingCallback())

    async def work():
        return {"npwp_confidence": 0.98, "name_confidence": 0.41}

    await pipeline.submit(RID, work, outcome_data=lambda result: DATA)
    await pipeline.runner.drain(5)

    [row] = await _rows(scoring)
    assert (row["downstream_status"], row["result_data"]["result"]["nomor_npwp"]) == ("COMPLETED", DATA["nomor_npwp"])


async def test_double_write_fills_the_outcome_row_and_the_event_log(tmp_path):
    outcome_table = orchestration_outcome_table("orchestration_extract_ocr")

    def both(table, stage):
        return CompositeOutcome([OrchestrationOutcome(outcome_table, stage=stage), ApiEventOutcome(table, stage=stage)])

    repo, table = await _repository(tmp_path, STAGE_SCORING, both)
    try:
        async with repo.engine.begin() as conn:
            await conn.run_sync(outcome_table.metadata.create_all)
        await repo.claim(RID)
        await repo.complete(RID, {"npwp_confidence": 0.98}, outcome_data=DATA)

        async with repo.engine.connect() as conn:
            outcome = (await conn.execute(select(outcome_table))).mappings().one()
        [event] = await _rows((repo, table))
        assert (outcome["downstream_status"], outcome["result_data"]) == ("completed", DATA)
        assert event["downstream_status"] == "COMPLETED"
    finally:
        await database.dispose_engines()


def test_writers_follow_the_settings():
    def build(**tables):
        settings = PipelineSettings(api_key="k", environment="local", _env_file=None, **tables)
        return build_stage_outcome(settings, stage=STAGE_SCORING)

    assert build() is None

    events = build(orchestration_api_events_table="ocr.orchestration_api_events")
    assert isinstance(events, ApiEventOutcome)
    assert (events.table.schema, events.table.name) == ("ocr", "orchestration_api_events")

    both = build(
        orchestration_outcome_table="orchestration_extract_ocr",
        orchestration_api_events_table="ocr.orchestration_api_events",
    )
    assert isinstance(both, CompositeOutcome)
    assert [type(writer) for writer in both.writers] == [OrchestrationOutcome, ApiEventOutcome]
