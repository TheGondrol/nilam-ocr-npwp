import json
import logging

import pytest
from elasticapm.traces import execution_context

from ocr_common.config import BaseServiceSettings
from ocr_common.pipeline.callbacks import TRACE_PARENT_KEY
from ocr_common.pipeline.outbox import callback_message, handoff_message
from ocr_common.web import apm
from ocr_common.web.app import create_app
from ocr_common.web.logging import JsonFormatter
from ocr_common.web.request_id import bind_request_id, reset_request_id


def _settings(**overrides) -> BaseServiceSettings:
    return BaseServiceSettings(api_key="k", environment="local", _env_file=None, **overrides)


def _middleware(app) -> list[str]:
    return [m.cls.__name__ for m in app.user_middleware]


def test_without_a_server_url_nothing_is_installed():
    app = create_app(settings=_settings(), title="t", description="d", service_name="scoring")

    assert "ElasticAPM" not in _middleware(app)
    assert apm._client is None
    assert apm.trace_fields() == {}
    with apm.job_transaction("OCR", "REQ_1"):
        apm.job_failed()  # no-ops


def test_with_a_server_url_the_agent_never_captures_bodies_or_headers(monkeypatch):
    seen = {}

    def make_apm_client(config):
        seen.update(config)
        return object()

    monkeypatch.setattr("elasticapm.contrib.starlette.make_apm_client", make_apm_client)
    monkeypatch.setattr(apm, "_client", None)  # restored to None afterwards, not to the fake start() installs
    settings = _settings(
        elastic_apm_server_url="http://apm:8200",
        elastic_apm_secret_token="t",
        elastic_apm_api_key="a",
        elastic_apm_sanitize_field_names="password,*token",
    )
    app = create_app(settings=settings, title="t", description="d", service_name="scoring")

    assert _middleware(app)[0] == "ElasticAPM", "the outermost middleware: it spans the whole request"
    assert seen["SERVICE_NAME"] == "nilam-ocr-npwp-scoring"
    assert seen["ENVIRONMENT"] == "local"
    assert (seen["CAPTURE_BODY"], seen["CAPTURE_HEADERS"]) == ("off", False)
    assert (seen["SECRET_TOKEN"], seen["API_KEY"]) == ("t", "a")
    assert seen["SANITIZE_FIELD_NAMES"] == "password,*token"


def test_a_job_is_a_transaction_labelled_with_its_request_id(apm_client):
    with apm.job_transaction("OCR", "REQ_1"):
        transaction = execution_context.get_transaction()
        assert transaction is not None
        assert transaction.transaction_type == "job"
        assert transaction.labels == {"request_id": "REQ_1", "stage": "OCR"}

    assert (transaction.name, transaction.result, transaction.outcome) == ("OCR job", "done", "success")
    assert execution_context.get_transaction() is None


def test_a_failed_job_is_marked_failed(apm_client):
    with apm.job_transaction("SCORING", "REQ_2"):
        transaction = execution_context.get_transaction()
        apm.job_failed()

    assert (transaction.result, transaction.outcome) == ("failed", "failure")


def test_an_interrupted_job_is_marked_and_the_exception_goes_on(apm_client):
    with pytest.raises(RuntimeError), apm.job_transaction("OCR", "REQ_3"):
        transaction = execution_context.get_transaction()
        raise RuntimeError("shutdown")

    assert (transaction.result, transaction.outcome) == ("interrupted", "failure")


def test_binding_a_request_id_labels_the_current_transaction(apm_client):
    apm_client.begin_transaction("request")
    token = bind_request_id("REQ_adopted")
    try:
        assert execution_context.get_transaction().labels["request_id"] == "REQ_adopted"
    finally:
        reset_request_id(token)
        apm_client.end_transaction("t")


def test_json_log_lines_carry_the_trace_of_the_active_transaction(apm_client):
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello", None, None)
    assert "trace.id" not in json.loads(JsonFormatter(None).format(record))

    apm_client.begin_transaction("request")
    try:
        line = json.loads(JsonFormatter(None).format(record))
        transaction = execution_context.get_transaction()
        assert line["trace.id"] == transaction.trace_parent.trace_id
        assert line["transaction.id"] == transaction.id
    finally:
        apm_client.end_transaction("t")


def test_a_job_continues_the_trace_of_the_request_that_queued_it(apm_client):
    request = apm_client.begin_transaction("request")
    try:
        with apm.job_transaction("OCR", "REQ_4"):
            job = execution_context.get_transaction()
    finally:
        apm_client.end_transaction("POST /v1/ocr/jobs")

    assert job is not request
    assert job.trace_parent.trace_id == request.trace_parent.trace_id
    assert job.trace_parent.span_id == request.id, "a child of the request, not of whoever called the request"


def test_a_job_without_a_request_starts_its_own_trace(apm_client):
    with apm.job_transaction("OCR", "REQ_5"):
        job = execution_context.get_transaction()

    assert job.trace_parent.span_id == job.id, "no parent: the root of a new trace"


def test_the_trace_parent_names_the_active_transaction_and_nothing_without_one(apm_client):
    assert apm.trace_parent() is None
    transaction = apm_client.begin_transaction("job")
    try:
        parent = apm.trace_parent()
    finally:
        apm_client.end_transaction("t")

    assert parent == f"00-{transaction.trace_parent.trace_id}-{transaction.id}-01"


def test_outbox_messages_carry_the_trace_of_the_job_that_wrote_them(apm_client):
    assert TRACE_PARENT_KEY not in callback_message("REQ_6", "OCR", "DONE").payload, "no transaction, no trace"

    with apm.job_transaction("OCR", "REQ_6"):
        parent = apm.trace_parent()
        callback = callback_message("REQ_6", "OCR", "DONE")
        handoff = handoff_message("STRUCTURING", {"request_id": "REQ_6"})

    assert callback.payload[TRACE_PARENT_KEY] == parent
    assert handoff.payload == {"next_stage": "STRUCTURING", "body": {"request_id": "REQ_6"}, TRACE_PARENT_KEY: parent}


def test_without_apm_outbox_messages_carry_no_trace():
    with apm.job_transaction("OCR", "REQ_7"):
        assert TRACE_PARENT_KEY not in handoff_message("STRUCTURING", {}).payload


@pytest.mark.parametrize(
    ("result", "outcome"),
    [(apm.DELIVERED, "success"), (apm.SKIPPED, "success"), (apm.RETRY, "failure"), (apm.DEAD, "failure")],
)
def test_a_delivery_is_a_transaction_in_the_trace_of_its_message(apm_client, result, outcome):
    job = apm_client.begin_transaction("job")
    parent = apm.trace_parent()
    apm_client.end_transaction("OCR job")

    with apm.delivery_transaction("OCR", "handoff", "REQ_8", parent, attempt=2, target="STRUCTURING"):
        delivery = execution_context.get_transaction()
        apm.delivery_result(result)

    assert (delivery.name, delivery.transaction_type) == ("OCR handoff", "outbox")
    assert (delivery.result, delivery.outcome) == (result, outcome)
    assert delivery.labels == {
        "request_id": "REQ_8",
        "stage": "OCR",
        "kind": "handoff",
        "attempt": 2,
        "target": "STRUCTURING",
    }
    assert (delivery.trace_parent.trace_id, delivery.trace_parent.span_id) == (job.trace_parent.trace_id, job.id)


def test_a_delivery_that_raises_is_an_error(apm_client):
    with pytest.raises(RuntimeError), apm.delivery_transaction("OCR", "callback", "REQ_9", None, attempt=1):
        delivery = execution_context.get_transaction()
        raise RuntimeError("database gone")

    assert (delivery.result, delivery.outcome) == ("error", "failure")
    assert delivery.trace_parent.span_id == delivery.id, "no parent: the root of a new trace"
