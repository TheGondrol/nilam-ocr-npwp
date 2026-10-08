"""Every answer to POST /v1/extract-ocr is logged in nilam_ocr_results before it is sent (the response log
middleware), whatever it is; the GET is not."""

import pytest
from sqlalchemy import MetaData, select

from ocr_common.errors import ServiceError
from ocr_common.pipeline.database import dispose_engines, get_engine
from ocr_common.pipeline.tables import ocr_results_table
from ocr_common.testing import auth_headers, make_client
from ocr_common.web.app import create_app

from app.api import testing
from app.api.response_log import ResponseLogMiddleware
from app.config import get_settings
from app.dependencies import get_guardrails_client, get_testing_extraction_client, get_testing_pipeline_waiter
from app.services.response_log import SqlResponseLog
from tests.conftest import (
    ACCEPTED_REPORT,
    JPEG,
    WITH_GUARDRAILS_THRESHOLD,
    RecordingResponseLog,
    StubExtraction,
    StubGuardrails,
    StubWaiter,
)

RID = "OCR_response_log"


def _submit(client, auth, filename="npwp.jpg", path="/v1/extract-ocr", **form):
    return client.post(
        path, headers=auth, data={"request_id": RID, **form}, files={"file": (filename, JPEG, "image/jpeg")}
    )


def _logged(response_logs, prefix=""):
    return [
        (r["request_id"], r["status_code"], r["body"]["errors"], r["guardrails"]) for r in response_logs[prefix].records
    ]


def test_a_finished_request_is_logged_with_the_answer_it_got(client, auth, response_logs):
    response = _submit(client, auth)

    assert response.status_code == 200
    [record] = response_logs[""].records
    assert (record["request_id"], record["status_code"], record["guardrails"]) == (RID, 200, 0.9821)
    assert record["body"] == response.json()


def test_a_guardrails_rejection_is_logged(client, auth, response_logs):
    response = _submit(client, auth, filename="blur.jpg", **WITH_GUARDRAILS_THRESHOLD)

    assert response.status_code == 400
    assert _logged(response_logs) == [(RID, 400, "DOWNSTREAM_VALIDATION_ERROR", 1)]
    assert response_logs[""].records[0]["body"]["pipeline_last_stage"] == "guardrails"


def test_a_202_is_logged(client, auth, response_logs, stub_waiter):
    from app.services.pipeline_waiter import WaitOutcome

    stub_waiter.outcome = WaitOutcome("OCR", "PROCESSING")

    assert _submit(client, auth).status_code == 202
    assert _logged(response_logs) == [(RID, 202, None, None)]


def test_guardrails_is_null_when_the_sequence_leaves_it_out(client, auth, response_logs):
    response = _submit(client, auth, pipeline_name_sequence='["extraction", "structuring", "scoring"]')

    assert (response.status_code, response.json()["guardrails"]) == (200, 0)
    assert _logged(response_logs) == [(RID, 200, None, None)]


def test_a_guardrails_only_request_is_logged_with_the_report(client, auth, response_logs):
    _submit(client, auth, pipeline_name_sequence='["guardrails"]')

    [record] = response_logs[""].records
    assert (record["status_code"], record["guardrails"]) == (200, 0.9821)
    assert record["body"]["data"] == {**ACCEPTED_REPORT, "auto_accepted": True, "score": 0.9821}


def test_a_refusal_before_anything_runs_is_logged(client, auth, response_logs):
    response = _submit(client, auth, pipeline_name_sequence='["extraction", "scoring"]')

    assert response.status_code == 422
    assert _logged(response_logs) == [(RID, 422, "INVALID_PIPELINE_SEQUENCE", None)]


def test_an_unreachable_stage_is_logged(client, auth, response_logs, stub_extraction):
    stub_extraction.error = ServiceError(503, "extraction service is unavailable")

    assert _submit(client, auth).status_code == 503
    assert _logged(response_logs) == [(RID, 503, "DOWNSTREAM_UNAVAILABLE", None)]


def test_a_file_refused_by_the_error_handler_is_logged(client, auth, response_logs):
    response = client.post(
        "/v1/extract-ocr", headers=auth, data={"request_id": RID}, files={"file": ("npwp.gif", b"GIF89a", "image/gif")}
    )

    assert response.status_code == 400
    [record] = response_logs[""].records
    assert (record["request_id"], record["status_code"]) == (RID, 400)


def test_the_get_is_not_logged(client, auth, response_logs):
    client.get(f"/v1/extract-ocr/{RID}", headers=auth)

    assert response_logs[""].records == []


def test_the_testing_endpoint_has_its_own_log():
    guardrails, extraction, waiter = StubGuardrails(), StubExtraction(), StubWaiter()
    app = create_app(settings=get_settings(), title="Orchestrator", description="testing", routers=testing.routers)
    app.dependency_overrides[get_guardrails_client] = lambda: guardrails
    app.dependency_overrides[get_testing_extraction_client] = lambda: extraction
    app.dependency_overrides[get_testing_pipeline_waiter] = lambda: waiter
    live, testing_log = RecordingResponseLog(), RecordingResponseLog("testing_")
    app.state.response_logs = {"": live, "testing_": testing_log}
    app.add_middleware(ResponseLogMiddleware)

    response = _submit(make_client(app), auth_headers(), path="/v1/extract-ocr-test")

    assert live.records == []
    assert [r["request_id"] for r in testing_log.records] == [response.json()["request_id"]]


# --- the table -------------------------------------------------------------------------------------


@pytest.fixture
async def database(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'results.db'}"
    table = ocr_results_table(MetaData())
    async with get_engine(url).begin() as conn:
        await conn.run_sync(table.metadata.create_all)
    yield url, table
    await dispose_engines()


async def test_each_answer_is_a_new_row(database):
    url, table = database
    log = SqlResponseLog(url)
    body = {"message": "OCR job accepted; still processing", "data": None, "errors": None, "pipeline_last_stage": None}

    await log.record(RID, 202, body, guardrails=None)
    await log.record(
        RID, 200, {**body, "message": "done", "data": {"nama": {"value": "X", "confidence": 1}}}, guardrails=0
    )
    # Without a guardrails threshold the answer's guardrails is the accepted probability (0019: a float column).
    await log.record(RID, 200, {**body, "message": "done"}, guardrails=0.9821)

    async with get_engine(url).connect() as conn:
        rows = (await conn.execute(select(table).order_by(table.c.id))).mappings().all()
    assert [(r["status_code"], r["status_desc"], r["guardrails"]) for r in rows] == [
        (202, "Accepted", None),
        (200, "OK", 0),
        (200, "OK", 0.9821),
    ]
    assert rows[1]["data"] == {"nama": {"value": "X", "confidence": 1}}
    assert list(rows[0]) == [
        "id",
        "request_id",
        "status_code",
        "status_desc",
        "message",
        "data",
        "errors",
        "pipeline_last_stage",
        "guardrails",
        "created_at",
    ]


async def test_a_write_that_fails_does_not_fail_the_answer(tmp_path, caplog):
    log = SqlResponseLog(f"sqlite+aiosqlite:///{tmp_path / 'empty.db'}")  # no table

    await log.record(RID, 200, {"message": "x"}, guardrails=0)

    assert f"answer 200 to {RID} not recorded" in caplog.text
    await dispose_engines()
