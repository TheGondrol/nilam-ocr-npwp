import json
import logging

import elasticapm
import pytest
from elasticapm.traces import execution_context

from ocr_common.config import BaseServiceSettings
from ocr_common.web import apm
from ocr_common.web.app import create_app
from ocr_common.web.logging import JsonFormatter
from ocr_common.web.request_id import bind_request_id, reset_request_id


def _settings(**overrides) -> BaseServiceSettings:
    return BaseServiceSettings(api_key="k", environment="local", _env_file=None, **overrides)


def _middleware(app) -> list[str]:
    return [m.cls.__name__ for m in app.user_middleware]


@pytest.fixture(scope="module")
def agent():
    """A real agent client that never sends anything and instruments nothing. One for the module: closing it
    waits for its sender thread."""
    agent = elasticapm.Client(
        service_name="test",
        server_url="http://127.0.0.1:9",
        disable_send=True,
        instrument=False,
        central_config=False,
        cloud_provider="none",
        metrics_interval="0ms",
    )
    yield agent
    agent.close()


@pytest.fixture
def client(agent, monkeypatch):
    """The test agent installed as the APM client."""
    monkeypatch.setattr(apm, "_client", agent)
    return agent


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
        elastic_apm_server_url="http://apm:8200", elastic_apm_secret_token="t", elastic_apm_api_key="a"
    )
    app = create_app(settings=settings, title="t", description="d", service_name="scoring")

    assert _middleware(app)[0] == "ElasticAPM", "the outermost middleware: it spans the whole request"
    assert seen["SERVICE_NAME"] == "nilam-ocr-npwp-scoring"
    assert seen["ENVIRONMENT"] == "local"
    assert (seen["CAPTURE_BODY"], seen["CAPTURE_HEADERS"]) == ("off", False)
    assert (seen["SECRET_TOKEN"], seen["API_KEY"]) == ("t", "a")


def test_a_job_is_a_transaction_labelled_with_its_request_id(client):
    with apm.job_transaction("OCR", "REQ_1"):
        transaction = execution_context.get_transaction()
        assert transaction is not None
        assert transaction.transaction_type == "job"
        assert transaction.labels == {"request_id": "REQ_1", "stage": "OCR"}

    assert (transaction.name, transaction.result, transaction.outcome) == ("OCR job", "done", "success")
    assert execution_context.get_transaction() is None


def test_a_failed_job_is_marked_failed(client):
    with apm.job_transaction("SCORING", "REQ_2"):
        transaction = execution_context.get_transaction()
        apm.job_failed()

    assert (transaction.result, transaction.outcome) == ("failed", "failure")


def test_an_interrupted_job_is_marked_and_the_exception_goes_on(client):
    with pytest.raises(RuntimeError), apm.job_transaction("OCR", "REQ_3"):
        transaction = execution_context.get_transaction()
        raise RuntimeError("shutdown")

    assert (transaction.result, transaction.outcome) == ("interrupted", "failure")


def test_binding_a_request_id_labels_the_current_transaction(client):
    client.begin_transaction("request")
    token = bind_request_id("REQ_adopted")
    try:
        assert execution_context.get_transaction().labels["request_id"] == "REQ_adopted"
    finally:
        reset_request_id(token)
        client.end_transaction("t")


def test_json_log_lines_carry_the_trace_of_the_active_transaction(client):
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello", None, None)
    assert "trace.id" not in json.loads(JsonFormatter(None).format(record))

    client.begin_transaction("request")
    try:
        line = json.loads(JsonFormatter(None).format(record))
        transaction = execution_context.get_transaction()
        assert line["trace.id"] == transaction.trace_parent.trace_id
        assert line["transaction.id"] == transaction.id
    finally:
        client.end_transaction("t")
