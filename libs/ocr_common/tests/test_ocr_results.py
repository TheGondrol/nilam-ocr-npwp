"""nilam_ocr_results logs every answer the central orchestrator gets; here the stages' part: one row per result
callback, written once it was delivered (LoggedCallback + CallbackResultsLog)."""

from typing import Any, cast

import pytest
from sqlalchemy import MetaData, select

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.config import PipelineSettings
from ocr_common.errors import ServiceError
from ocr_common.npwp import REJECTED_CODE
from ocr_common.pipeline import STAGE_OCR, STAGE_SCORING, STAGE_STRUCTURING, build_stage_pipeline, database
from ocr_common.pipeline.callbacks import (
    LoggedCallback,
    OrchestrationCallback,
    ResultCallback,
    stage_callback_body,
)
from ocr_common.pipeline.ocr_results_sql import CallbackResultsLog, callback_row
from ocr_common.pipeline.outbox import OutboxRelay, callback_message
from ocr_common.pipeline.outbox_sql import SqlOutbox
from ocr_common.pipeline.tables import ocr_results_table, pipeline_tables, repo_metadata

RID = "REQ_ocr_results"
DATA = {
    "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 1},
    "nama": {"value": "BUDI SANTOSO", "confidence": 0},
}
REASON = "Kode provinsi pada NPWP tidak valid, mohon dicek kembali"
DONE = stage_callback_body(RID, STAGE_SCORING, "DONE", result={"npwp_confidence": 0.7}, final=True, answer=DATA)
REJECTED = stage_callback_body(RID, STAGE_STRUCTURING, "FAILED", error_message=REASON, error_code=REJECTED_CODE)
FAILED = stage_callback_body(RID, STAGE_OCR, "FAILED", error_message="file_url could not be downloaded")
NOT_FINAL = stage_callback_body(RID, STAGE_OCR, "DONE", result={"text": "NPWP"})


# --- the row of a callback ----------------------------------------------------------------------


def test_a_completed_callback_is_the_200_answer_with_its_data():
    assert callback_row(DONE) == {
        "status_code": 200,
        "message": "OCR extraction completed successfully",
        "data": DATA,
        "errors": None,
        "guardrails": 0,
        "pipeline_last_stage": "scoring",
    }


def test_a_rejection_callback_is_the_400_answer():
    assert callback_row(REJECTED) == {
        "status_code": 400,
        "message": REASON,
        "data": None,
        "errors": REJECTED_CODE,
        "guardrails": 1,
        "pipeline_last_stage": "structuring",
    }


def test_a_failed_callback_is_the_422_answer_of_the_stage_that_failed():
    row = callback_row(FAILED)
    assert row is not None
    assert (row["status_code"], row["errors"], row["message"], row["pipeline_last_stage"]) == (
        422,
        "OCR_FAILED",
        "file_url could not be downloaded",
        "extraction",
    )


def test_a_callback_that_does_not_end_the_request_has_no_row():
    assert callback_row(NOT_FINAL) is None


# --- LoggedCallback: only what was sent and ends the request -------------------------------------


class Inner:
    """Stands in for OrchestrationCallback / ResultCallback."""

    def __init__(self, sent: bool = True, error: Exception | None = None):
        self.sent = sent
        self.error = error

    async def notify(self, *args: Any, **kwargs: Any) -> bool:
        return self.sent

    async def send(self, body: dict[str, Any]) -> bool:
        if self.error is not None:
            raise self.error
        return self.sent

    async def aclose(self) -> None:
        pass


def _logged(inner: Inner) -> tuple[LoggedCallback, list[dict[str, Any]]]:
    delivered: list[dict[str, Any]] = []

    async def record(body: dict[str, Any]) -> None:
        delivered.append(body)

    return LoggedCallback(cast(ResultCallback, inner), record), delivered


async def test_a_sent_callback_that_ends_the_request_is_logged_once():
    callback, delivered = _logged(Inner())
    assert await callback.send(DONE) is True
    assert delivered == [DONE]


@pytest.mark.parametrize(("inner", "body"), [(Inner(sent=False), DONE), (Inner(), NOT_FINAL)])
async def test_a_skipped_or_not_final_callback_is_not_logged(inner, body):
    callback, delivered = _logged(inner)
    await callback.send(body)
    assert delivered == []


async def test_a_callback_that_failed_to_send_is_not_logged_and_still_raises():
    callback, delivered = _logged(Inner(error=ServiceError(503, "orchestration unavailable")))
    with pytest.raises(ServiceError):
        await callback.send(DONE)
    assert delivered == []


async def test_the_direct_mode_logs_what_notify_sent():
    callback, delivered = _logged(Inner())
    await callback.notify(RID, STAGE_STRUCTURING, "FAILED", error_message=REASON, error_code=REJECTED_CODE)
    assert delivered == [REJECTED]

    quiet, nothing = _logged(Inner(sent=False))
    await quiet.notify(RID, STAGE_STRUCTURING, "FAILED", error_message=REASON, error_code=REJECTED_CODE)
    assert nothing == []


# --- CallbackResultsLog on a database ------------------------------------------------------------


@pytest.fixture
async def url(tmp_path):
    await database.dispose_engines()
    url = f"sqlite+aiosqlite:///{tmp_path / 'ocr_results.db'}"
    async with database.get_engine(url).begin() as conn:
        await conn.run_sync(repo_metadata().create_all)
    yield url
    await database.dispose_engines()


def _log(url: str, table_prefix: str = "scoring") -> CallbackResultsLog:
    jobs, _ = pipeline_tables(table_prefix, MetaData())
    return CallbackResultsLog(url, ocr_results_table(MetaData()), jobs)


async def _rows(url: str) -> list[dict[str, Any]]:
    table = ocr_results_table(MetaData())
    async with database.get_engine(url).connect() as conn:
        return [dict(row) for row in (await conn.execute(select(table).order_by(table.c.id))).mappings().all()]


async def _job(url: str, table_prefix: str, sequence: list[str] | None) -> None:
    jobs, _ = pipeline_tables(table_prefix, MetaData())
    async with database.get_engine(url).begin() as conn:
        await conn.execute(
            jobs.insert().values(
                request_id=RID, status="DONE", input={"pipeline_name_sequence": sequence}, ds="20261007"
            )
        )


async def test_a_delivered_callback_becomes_a_row(url):
    await _log(url)(DONE)

    [row] = await _rows(url)
    assert (row["request_id"], row["status_code"], row["status_desc"], row["data"], row["guardrails"]) == (
        RID,
        200,
        "OK",
        DATA,
        0,
    )
    assert row["pipeline_last_stage"] == "scoring"
    assert "update_at" not in row


async def test_guardrails_is_null_when_the_request_left_it_out(url):
    await _job(url, "scoring", ["extraction", "structuring", "scoring"])
    await _log(url)(DONE)

    [row] = await _rows(url)
    assert row["guardrails"] is None


async def test_every_delivered_callback_is_a_new_row(url):
    log = _log(url)
    await log(FAILED)
    await log(DONE)

    assert [row["status_code"] for row in await _rows(url)] == [422, 200]


async def test_a_write_that_fails_does_not_fail_the_delivery(tmp_path, caplog):
    log = _log(f"sqlite+aiosqlite:///{tmp_path / 'empty.db'}")  # no table

    await log(DONE)

    assert f"callback answer of {RID} not recorded" in caplog.text
    await database.dispose_engines()


# --- through the outbox relay: written when a try succeeds, once -------------------------------


class FlakyOrchestrator:
    """The central orchestrator's callback endpoint: 503 for the first `failures` posts, then 200."""

    def __init__(self, failures: int):
        self.failures = failures
        self.posts: list[dict[str, Any]] = []

    async def post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        self.posts.append(body)
        if len(self.posts) <= self.failures:
            raise ServiceError(503, "orchestration unavailable")
        return {}

    async def aclose(self) -> None:
        pass


async def test_a_callback_retried_by_the_relay_is_logged_once_when_it_gets_through(url):
    orchestrator = FlakyOrchestrator(failures=2)
    inner = ResultCallback(cast(RemoteModelClient, orchestrator), "/v1/ocr-callback")
    outbox = SqlOutbox(url)
    relay = OutboxRelay(outbox, stage=STAGE_SCORING, callback=LoggedCallback(inner, _log(url)), retry_delay_seconds=0)
    async with database.get_engine(url).begin() as conn:
        message = callback_message(RID, STAGE_SCORING, "DONE", result={"npwp_confidence": 0.7}, final=True, answer=DATA)
        await outbox.add(conn, RID, STAGE_SCORING, [message])

    for _ in range(3):
        await relay.deliver_due()
        if len(orchestrator.posts) < 3:
            assert await _rows(url) == []  # 503: not delivered, nothing logged

    assert len(orchestrator.posts) == 3
    [row] = await _rows(url)
    assert (row["status_code"], row["data"]) == (200, DATA)


# --- wiring -------------------------------------------------------------------------------------


def _settings(**values: Any) -> PipelineSettings:
    return PipelineSettings(api_key="k", environment="local", _env_file=None, **values)


def test_every_stage_with_a_database_logs_its_callbacks(url):
    live = build_stage_pipeline(_settings(database_url=url), stage=STAGE_OCR, table_prefix="ocr")
    assert isinstance(live.callback, LoggedCallback)
    testing = build_stage_pipeline(_settings(database_url=url), stage=STAGE_OCR, table_prefix="ocr", testing=True)
    assert isinstance(testing.callback, OrchestrationCallback)  # the testing lane sends no callback at all
    without_db = build_stage_pipeline(_settings(), stage=STAGE_OCR, table_prefix="ocr")
    assert not isinstance(without_db.callback, LoggedCallback)
