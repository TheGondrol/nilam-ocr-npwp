import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from ocr_common.config import PipelineSettings
from ocr_common.pipeline import STAGE_SCORING, StagePipeline, database
from ocr_common.pipeline.outcomes import OrchestrationOutcome, build_stage_outcome
from ocr_common.pipeline.repository_sql import SqlJobRepository
from ocr_common.pipeline.tables import orchestration_outcome_table
from ocr_common.testing import RecordingCallback

TABLE = "orchestration_extract_ocr"
RID = "REQ_outcome"
DATA = {
    "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 1},
    "nama": {"value": "BUDI SANTOSO", "confidence": 0},
}


@pytest.fixture
async def repository(tmp_path):
    await database.dispose_engines()
    table = orchestration_outcome_table(TABLE)
    repo = SqlJobRepository(
        f"sqlite+aiosqlite:///{tmp_path / 'outcome.db'}",
        "scoring",
        outcome=OrchestrationOutcome(table, stage=STAGE_SCORING),
    )
    async with repo.engine.begin() as conn:
        await conn.run_sync(repo.metadata.create_all)
        await conn.run_sync(table.metadata.create_all)
    yield repo, table
    await database.dispose_engines()


async def _row(repository) -> dict:
    repo, table = repository
    async with repo.engine.connect() as conn:
        rows = (await conn.execute(select(table))).mappings().all()
    assert len(rows) == 1
    return dict(rows[0])


async def test_claiming_a_job_marks_the_request_processing_at_that_stage(repository):
    repo, _ = repository
    assert await repo.claim(RID) is True

    row = await _row(repository)
    assert (row["request_id"], row["document_type"]) == (RID, "npwp")
    assert (row["status_code"], row["downstream_status"], row["downstream_stage"]) == (202, "processing", "SCORING")
    assert (row["result_data"], row["error_code"], row["error_message"]) == (None, None, None)
    assert row["ds"]


async def test_completing_with_data_marks_the_request_completed(repository):
    repo, _ = repository
    await repo.claim(RID)
    await repo.complete(RID, {"npwp_confidence": 0.7}, outcome_data=DATA)

    row = await _row(repository)
    assert (row["status_code"], row["downstream_status"], row["downstream_stage"]) == (200, "completed", "SCORING")
    assert row["result_data"] == DATA
    assert (row["error_code"], row["error_message"]) == (None, None)


async def test_completing_without_data_leaves_the_request_processing(repository):
    repo, _ = repository
    await repo.claim(RID)
    await repo.complete(RID, {"blocks": []})

    row = await _row(repository)
    assert (row["status_code"], row["downstream_status"]) == (202, "processing")
    assert row["result_data"] is None


async def test_failing_records_the_stage_that_failed(repository):
    repo, _ = repository
    await repo.claim(RID)
    await repo.fail(RID, "No text lines to structure")

    row = await _row(repository)
    assert (row["status_code"], row["downstream_status"], row["downstream_stage"]) == (422, "failed", "SCORING")
    assert (row["error_code"], row["error_message"]) == ("SCORING_FAILED", "No text lines to structure")


async def test_a_second_run_of_the_same_request_id_overwrites_its_row(repository):
    repo, _ = repository
    await repo.claim(RID)
    await repo.fail(RID, "boom")
    await repo.claim(RID)
    await repo.complete(RID, {"npwp_confidence": 0.7}, outcome_data=DATA)

    row = await _row(repository)
    assert (row["downstream_status"], row["error_code"]) == ("completed", None)


async def test_the_row_and_the_job_are_written_in_one_transaction(repository):
    repo, table = repository
    await repo.claim(RID)
    async with repo.engine.begin() as conn:
        await conn.run_sync(table.drop)

    with pytest.raises(OperationalError):
        await repo.complete(RID, {"npwp_confidence": 0.7}, outcome_data=DATA)

    record = await repo.get(RID)
    assert record is not None
    assert record["status"] == "PROCESSING"
    assert record["result"] is None


async def test_the_pipeline_passes_the_row_data_of_the_last_stage(repository):
    repo, _ = repository
    callback = RecordingCallback()
    pipeline = StagePipeline(stage=STAGE_SCORING, repository=repo, callback=callback)

    async def work():
        return {"npwp_confidence": 0.7296, "name_confidence": 0.41}

    await pipeline.submit(RID, work, outcome_data=lambda result: DATA)
    await pipeline.runner.drain(5)

    row = await _row(repository)
    assert (row["downstream_status"], row["result_data"]) == ("completed", DATA)


def test_the_outcome_row_is_off_until_the_table_is_configured():
    off = PipelineSettings(api_key="k", environment="local", _env_file=None)
    assert build_stage_outcome(off, stage=STAGE_SCORING) is None

    on = PipelineSettings(api_key="k", environment="local", orchestration_outcome_table=TABLE, _env_file=None)
    writer = build_stage_outcome(on, stage=STAGE_SCORING)
    assert writer is not None
    assert writer.table.name == TABLE


async def test_a_failed_handoff_marks_the_request_failed_at_the_next_stage(repository):
    repo, _ = repository
    await repo.claim(RID)
    await repo.complete(RID, {"npwp_confidence": 0.7}, outcome_data=DATA)

    await repo.handoff_failed(RID, "STRUCTURING", "Handoff to STRUCTURING failed: structuring service is unavailable")

    row = await _row(repository)
    assert (row["status_code"], row["downstream_status"], row["downstream_stage"]) == (422, "failed", "STRUCTURING")
    assert row["error_code"] == "STRUCTURING_FAILED"
    assert row["error_message"] == "Handoff to STRUCTURING failed: structuring service is unavailable"
