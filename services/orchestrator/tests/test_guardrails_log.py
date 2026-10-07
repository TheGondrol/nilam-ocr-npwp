"""Every guardrails verdict is kept in nilam_guardrails_results, the rejected documents included, and a write that
fails never fails the request. When guardrails ends the request, its final answer goes to nilam_ocr_results."""

import pytest
from sqlalchemy import MetaData, select

from ocr_common.pipeline.database import dispose_engines, get_engine
from ocr_common.pipeline.tables import guardrails_results_table, ocr_results_table

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
    _submit(client, auth, guardrails_confidence_threshold='{"acc_rej": 0.3}')
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
    ocr_results_table(metadata)
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
    assert (accepted["threshold"], accepted["threshold_source"]) == (0.3, "request")
    assert "threshold_target" not in accepted
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
    log = SqlGuardrailsLog("sqlite+aiosqlite://", table_prefix="testing_")
    assert (log._table.name, log._results.name) == ("nilam_testing_guardrails_results", "nilam_testing_ocr_results")


async def _answers(url):
    table = ocr_results_table(MetaData())
    async with get_engine(url).connect() as conn:
        return [dict(row) for row in (await conn.execute(select(table))).mappings().all()]


async def test_a_rejection_writes_the_final_answer(database):
    url, _ = database

    await SqlGuardrailsLog(url).record(RID, REJECTED_REPORT, threshold_from_request=False, sequence=["guardrails"])

    [row] = await _answers(url)
    assert (row["request_id"], row["status_code"], row["status_desc"], row["message"]) == (
        RID,
        400,
        "Bad Request",
        REJECTED_REPORT["reason"],
    )
    assert (row["data"], row["errors"], row["guardrails"]) == (None, "DOWNSTREAM_VALIDATION_ERROR", 1)
    assert row["pipeline_last_stage"] == "guardrails"


async def test_a_guardrails_only_request_writes_the_report_as_data(database):
    url, _ = database

    await SqlGuardrailsLog(url).record(RID, ACCEPTED_REPORT, threshold_from_request=False, sequence=["guardrails"])

    [row] = await _answers(url)
    assert (row["status_code"], row["status_desc"], row["message"]) == (
        200,
        "OK",
        "OCR extraction completed successfully",
    )
    assert (row["data"], row["errors"], row["guardrails"]) == (ACCEPTED_REPORT, None, 0)
    assert row["pipeline_last_stage"] == "guardrails"


@pytest.mark.parametrize("sequence", [None, ["guardrails", "extraction"]])
async def test_a_document_that_goes_on_to_the_stages_writes_no_answer_here(database, sequence):
    url, _ = database

    await SqlGuardrailsLog(url).record(RID, ACCEPTED_REPORT, threshold_from_request=False, sequence=sequence)

    assert await _answers(url) == []


def test_the_sequence_is_recorded_with_the_verdict(client, auth, guardrails_log):
    _submit(client, auth, pipeline_name_sequence='["guardrails"]')
    _submit(client, auth)

    assert [r["sequence"] for r in guardrails_log.records] == [
        ["guardrails"],
        ["guardrails", "extraction", "structuring", "scoring"],
    ]


async def test_the_sequence_is_written_and_the_last_verdict_read_back(database):
    url, table = database
    log = SqlGuardrailsLog(url)

    await log.record(RID, REJECTED_REPORT, threshold_from_request=False, sequence=["guardrails", "extraction"])
    await log.record(RID, ACCEPTED_REPORT, threshold_from_request=False, sequence=["guardrails"])

    first, _ = await _rows(url, table)
    assert first["pipeline_name_sequence"] == ["guardrails", "extraction"]
    assert await log.latest(RID) == {"report": ACCEPTED_REPORT, "sequence": ["guardrails"]}
    assert await log.latest("OCR_never_judged") is None


async def test_a_verdict_that_cannot_be_read_is_none_and_logged(tmp_path, caplog):
    url = f"sqlite+aiosqlite:///{tmp_path / 'empty.db'}"  # no table: the select fails

    assert await SqlGuardrailsLog(url).latest(RID) is None
    assert "guardrails verdict of OCR_guardrails_log not readable" in caplog.text
    await dispose_engines()


# --- the GET of a request no stage has a job for -------------------------------------------------------


def _get(client, auth):
    return client.get(f"/v1/extract-ocr/{RID}", headers=auth)


def test_the_get_of_a_document_rejected_by_guardrails_answers_like_its_post(client, auth, stub_waiter):
    posted = _submit(client, auth, filename="blur.jpg")
    stub_waiter.snapshot_outcome = None  # no stage has a job

    response = _get(client, auth)

    assert response.status_code == 400
    body = response.json()
    assert (body["errors"], body["guardrails"], body["pipeline_last_stage"]) == (
        "DOWNSTREAM_VALIDATION_ERROR",
        1,
        "guardrails",
    )
    assert body["message"] == posted.json()["message"]


def test_the_get_of_a_guardrails_only_request_answers_with_the_report(client, auth, stub_waiter):
    posted = _submit(client, auth, pipeline_name_sequence='["guardrails"]')
    stub_waiter.snapshot_outcome = None

    response = _get(client, auth)

    assert response.status_code == 200
    body = response.json()
    assert (body["status_code"], body["guardrails"], body["pipeline_last_stage"]) == (200, 0, None)
    assert body["data"] == posted.json()["data"] == ACCEPTED_REPORT


def test_a_request_that_passed_but_never_reached_a_stage_is_404(client, auth, stub_waiter):
    _submit(client, auth)  # passed guardrails; say its hand-off to extraction failed
    stub_waiter.snapshot_outcome = None

    assert _get(client, auth).status_code == 404


def test_a_request_never_judged_is_404(client, auth, stub_waiter):
    stub_waiter.snapshot_outcome = None

    assert _get(client, auth).status_code == 404


def test_a_stage_job_wins_over_the_verdict(client, auth, stub_waiter):
    _submit(client, auth, filename="blur.jpg")  # an earlier attempt was rejected; the stub still has a DONE job

    assert _get(client, auth).status_code == 200
