"""nilam_ocr_results: the request's final answer, written by the stage the request ends at, in the job's transaction."""

from datetime import datetime

import pytest
from sqlalchemy import MetaData, select
from sqlalchemy.exc import OperationalError

from ocr_common.config import PipelineSettings
from ocr_common.pipeline import (
    STAGE_OCR,
    STAGE_SCORING,
    STAGE_STRUCTURING,
    StagePipeline,
    build_stage_pipeline,
    database,
)
from ocr_common.pipeline.ocr_results_sql import OcrResultsOutcome
from ocr_common.pipeline.outcomes import CompositeOutcome, OrchestrationOutcome
from ocr_common.pipeline.repository_sql import SqlJobRepository
from ocr_common.pipeline.tables import (
    ocr_results_table,
    orchestration_outcome_table,
    pipeline_tables,
    repo_metadata,
)
from ocr_common.testing import RecordingCallback

RID = "REQ_ocr_results"
DATA = {
    "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 1},
    "nama": {"value": "BUDI SANTOSO", "confidence": 0},
}
REASON = "Kode provinsi pada NPWP tidak valid, mohon dicek kembali"
NO_GUARDRAILS = {"pipeline_name_sequence": ["extraction", "structuring", "scoring"]}


@pytest.fixture
async def url(tmp_path):
    await database.dispose_engines()
    url = f"sqlite+aiosqlite:///{tmp_path / 'ocr_results.db'}"
    async with database.get_engine(url).begin() as conn:
        await conn.run_sync(repo_metadata().create_all)
    yield url
    await database.dispose_engines()


def _repository(url: str, table_prefix: str, stage: str) -> SqlJobRepository:
    jobs, _ = pipeline_tables(table_prefix, MetaData())
    outcome = OcrResultsOutcome(ocr_results_table(MetaData()), jobs, stage=stage)
    return SqlJobRepository(url, table_prefix, stage=stage, outcome=outcome)


async def _rows(url: str, table_prefix: str = "") -> list[dict]:
    table = ocr_results_table(MetaData(), table_prefix)
    async with database.get_engine(url).connect() as conn:
        return [dict(row) for row in (await conn.execute(select(table).order_by(table.c.id))).mappings().all()]


async def _row(url: str) -> dict:
    [row] = await _rows(url)
    return row


def _answer(row: dict) -> tuple:
    return (
        row["request_id"],
        row["status_code"],
        row["status_desc"],
        row["message"],
        row["data"],
        row["errors"],
        row["guardrails"],
        row["pipeline_last_stage"],
    )


async def test_the_last_stage_writes_its_data_as_the_answer(url):
    repo = _repository(url, "scoring", STAGE_SCORING)
    await repo.claim(RID)
    await repo.complete(RID, {"npwp_confidence": 0.7}, outcome_data=DATA)

    row = await _row(url)
    assert _answer(row) == (RID, 200, "OK", "OCR extraction completed successfully", DATA, None, 0, "scoring")
    assert isinstance(row["created_at"], datetime) and isinstance(row["update_at"], datetime)


async def test_an_earlier_last_stage_writes_its_result_as_it_is(url):
    repo = _repository(url, "ocr", STAGE_OCR)
    result = {"text": "NPWP", "blocks": []}
    await repo.claim(RID, input={"pipeline_name_sequence": ["guardrails", "extraction"]})
    await repo.complete(RID, result, outcome_data=result)

    assert _answer(await _row(url))[1:] == (
        200,
        "OK",
        "OCR extraction completed successfully",
        result,
        None,
        0,
        "extraction",
    )


async def test_guardrails_is_null_when_the_request_left_it_out(url):
    repo = _repository(url, "scoring", STAGE_SCORING)
    await repo.claim(RID, input=NO_GUARDRAILS)
    await repo.complete(RID, {"npwp_confidence": 0.7}, outcome_data=DATA)

    assert (await _row(url))["guardrails"] is None


async def test_a_stage_that_hands_the_job_on_writes_nothing(url):
    repo = _repository(url, "ocr", STAGE_OCR)
    await repo.claim(RID)
    await repo.complete(RID, {"text": "NPWP"})

    assert await _rows(url) == []


async def test_a_failed_stage_ends_the_request_with_422(url):
    repo = _repository(url, "ocr", STAGE_OCR)
    await repo.claim(RID)
    await repo.fail(RID, "file_url could not be downloaded")

    assert _answer(await _row(url))[1:] == (
        422,
        "Unprocessable Entity",
        "file_url could not be downloaded",
        None,
        "OCR_FAILED",
        0,
        "extraction",
    )


async def test_a_failed_handoff_names_the_next_stage(url):
    repo = _repository(url, "ocr", STAGE_OCR)
    await repo.claim(RID, input=NO_GUARDRAILS)
    await repo.complete(RID, {"text": "NPWP"})
    await repo.handoff_failed(RID, STAGE_STRUCTURING, "Handoff to STRUCTURING failed: unavailable")

    row = await _row(url)
    assert (row["status_code"], row["errors"], row["message"]) == (
        422,
        "STRUCTURING_FAILED",
        "Handoff to STRUCTURING failed: unavailable",
    )
    assert (row["guardrails"], row["pipeline_last_stage"]) == (None, "structuring")


@pytest.mark.parametrize(("input", "guardrails"), [(None, 1), (NO_GUARDRAILS, None)])
async def test_a_rejection_by_the_structuring_rules_ends_the_request_with_400(url, input, guardrails):
    repo = _repository(url, "structuring", STAGE_STRUCTURING)
    await repo.claim(RID, input=input)
    await repo.complete(RID, {"reject_reason": REASON}, rejection=REASON)

    assert _answer(await _row(url))[1:] == (
        400,
        "Bad Request",
        REASON,
        None,
        "DOWNSTREAM_VALIDATION_ERROR",
        guardrails,
        "structuring",
    )


async def test_a_request_id_run_again_gets_another_row_append_only(url):
    repo = _repository(url, "scoring", STAGE_SCORING)
    await repo.claim(RID)
    await repo.fail(RID, "boom")
    await repo.claim(RID)
    await repo.complete(RID, {"npwp_confidence": 0.7}, outcome_data=DATA)

    failed, done = await _rows(url)
    assert (failed["request_id"], failed["status_code"], failed["errors"]) == (RID, 422, "SCORING_FAILED")
    assert (done["request_id"], done["status_code"], done["errors"], done["data"]) == (RID, 200, None, DATA)
    assert done["id"] > failed["id"]
    assert all(row["update_at"] == row["created_at"] for row in (failed, done))


async def test_the_answer_is_written_in_the_job_transaction(url):
    repo = _repository(url, "scoring", STAGE_SCORING)
    await repo.claim(RID)
    async with repo.engine.begin() as conn:
        await conn.run_sync(ocr_results_table(MetaData()).drop)

    with pytest.raises(OperationalError):
        await repo.complete(RID, {"npwp_confidence": 0.7}, outcome_data=DATA)

    record = await repo.get(RID)
    assert record is not None and record["status"] == "PROCESSING"


# --- wiring ----------------------------------------------------------------------------------------


def _settings(**values) -> PipelineSettings:
    return PipelineSettings(api_key="k", environment="local", _env_file=None, **values)


async def _run_last_stage(pipeline: StagePipeline) -> None:
    async def work():
        return {"text": "NPWP"}

    await pipeline.submit(RID, work, callback_result=dict, outcome_data=dict)
    await pipeline.runner.drain(5)


async def test_every_stage_with_a_database_writes_the_answer(url):
    pipeline = build_stage_pipeline(_settings(database_url=url), stage=STAGE_OCR, table_prefix="ocr")
    pipeline.callback = RecordingCallback()
    await _run_last_stage(pipeline)

    assert _answer(await _row(url))[1:] == (
        200,
        "OK",
        "OCR extraction completed successfully",
        {"text": "NPWP"},
        None,
        0,
        "extraction",
    )


async def test_the_testing_pipeline_writes_the_testing_table(url):
    pipeline = build_stage_pipeline(_settings(database_url=url), stage=STAGE_OCR, table_prefix="ocr", testing=True)
    await _run_last_stage(pipeline)

    assert await _rows(url) == []
    [row] = await _rows(url, "testing_")
    assert (row["request_id"], row["status_code"]) == (RID, 200)


async def test_the_answer_goes_next_to_the_orchestrators_outcome_row(url):
    table = "orchestration_extract_ocr"
    async with database.get_engine(url).begin() as conn:
        await conn.run_sync(orchestration_outcome_table(table).metadata.create_all)
    settings = _settings(database_url=url, orchestration_outcome_table=table)
    pipeline = build_stage_pipeline(settings, stage=STAGE_OCR, table_prefix="ocr")
    pipeline.callback = RecordingCallback()
    assert isinstance(pipeline.repository, SqlJobRepository)
    writer = pipeline.repository._outcome
    assert isinstance(writer, CompositeOutcome)
    assert [type(w) for w in writer.writers] == [OcrResultsOutcome, OrchestrationOutcome]

    await _run_last_stage(pipeline)

    assert (await _row(url))["status_code"] == 200
    async with database.get_engine(url).connect() as conn:
        outcome = (await conn.execute(select(orchestration_outcome_table(table)))).mappings().one()
    assert outcome["downstream_status"] == "completed"
