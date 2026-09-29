"""Every guardrails verdict is kept in guardrails_results, the rejected documents included, and a write that
fails never fails the request."""

import pytest
from sqlalchemy import MetaData, select

from ocr_common.pipeline.database import dispose_engines, get_engine
from ocr_common.pipeline.tables import guardrails_results_table

from app.services.guardrails_log import SqlGuardrailsLog
from tests.conftest import ACCEPTED_REPORT, JPEG, REJECTED_REPORT

RID = "OCR_guardrails_log"


def _submit(client, auth, filename="npwp.jpg", **form):
    return client.post(
        "/v1/extract-ocr",
        headers=auth,
        data={"request_id": RID, **form},
        files={"file": (filename, JPEG, "image/jpeg")},
    )


# --- what the service records ------------------------------------------------------------------------


def test_an_accepted_document_is_recorded_with_whose_threshold_decided(client, auth, guardrails_log):
    _submit(client, auth, guardrails_confidence_threshold="0.3", guardrails_tendency="accepted")
    _submit(client, auth)

    assert [(r["request_id"], r["report"]["passed"], r["threshold_from_request"]) for r in guardrails_log.records] == [
        (RID, True, True),
        (RID, True, False),
    ]


def test_a_rejected_document_is_recorded_too(client, auth, guardrails_log):
    response = _submit(client, auth, filename="blur.jpg")

    assert response.status_code == 400
    [record] = guardrails_log.records
    assert record["report"] == REJECTED_REPORT


def test_nothing_is_recorded_when_guardrails_is_left_out(client, auth, guardrails_log):
    _submit(client, auth, pipeline_name_sequence='["extraction", "structuring", "scoring"]')

    assert guardrails_log.records == []


# --- the table -------------------------------------------------------------------------------------


@pytest.fixture
async def database(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'guardrails.db'}"
    metadata = MetaData()
    table = guardrails_results_table(metadata)
    async with get_engine(url).begin() as conn:
        await conn.run_sync(metadata.create_all)
    yield url, table
    await dispose_engines()


async def _rows(url, table):
    async with get_engine(url).connect() as conn:
        return (await conn.execute(select(table).order_by(table.c.id))).mappings().all()


async def test_the_verdict_is_written_with_its_threshold_and_the_whole_report(database):
    url, table = database
    document = {
        "verdict": "accepted",
        "confidence": 0.9821,
        "n_pages": 1,
        "threshold": 0.3,
        "threshold_target": "accept",
    }
    report = {**ACCEPTED_REPORT, "document": document}

    await SqlGuardrailsLog(url).record(RID, report, threshold_from_request=True)
    await SqlGuardrailsLog(url).record(RID, REJECTED_REPORT, threshold_from_request=False)

    accepted, rejected = await _rows(url, table)
    assert (accepted["request_id"], accepted["passed"], accepted["verdict"], accepted["confidence"]) == (
        RID,
        True,
        "accepted",
        0.9821,
    )
    assert (accepted["threshold"], accepted["threshold_target"], accepted["threshold_source"]) == (
        0.3,
        "accept",
        "request",
    )
    assert accepted["report"] == report
    assert (rejected["passed"], rejected["threshold_source"], rejected["reason"]) == (
        False,
        "service",
        REJECTED_REPORT["reason"],
    )
    assert len(accepted["ds"]) == 8


async def test_a_write_that_fails_is_logged_and_does_not_raise(tmp_path, caplog):
    url = f"sqlite+aiosqlite:///{tmp_path / 'empty.db'}"  # no table: the insert fails

    await SqlGuardrailsLog(url).record(RID, ACCEPTED_REPORT, threshold_from_request=False)

    assert "guardrails verdict of OCR_guardrails_log not recorded" in caplog.text
    await dispose_engines()


def test_the_testing_endpoints_write_their_own_table():
    assert SqlGuardrailsLog("sqlite+aiosqlite://", table_prefix="testing_")._table.name == "testing_guardrails_results"
