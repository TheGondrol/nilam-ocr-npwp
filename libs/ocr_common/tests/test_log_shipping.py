import logging
import sys

import pytest
from elasticapm.traces import execution_context
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter

from ocr_common.config import BaseServiceSettings
from ocr_common.web import apm, log_shipping
from ocr_common.web.app import create_app
from ocr_common.web.logging import log_event
from ocr_common.web.request_id import bind_request_id, reset_request_id


def _settings(**overrides) -> BaseServiceSettings:
    return BaseServiceSettings(api_key="k", environment="local", _env_file=None, **overrides)


def _on(**overrides) -> BaseServiceSettings:
    return _settings(elastic_apm_server_url="http://apm:8200", elastic_apm_log_sending=True, **overrides)


@pytest.fixture
def exported(monkeypatch, caplog):
    """What reaches the APM server from the test's loggers: each provider's exporter replaced by one in memory
    (a provider closes its exporter when replaced); `flush()` sends the batch of the current one. No agent is
    started: `create_app` with a server URL gets no APM client."""
    exporters: list[InMemoryLogRecordExporter] = []

    def exporter(settings):
        exporters.append(InMemoryLogRecordExporter())
        return exporters[-1]

    monkeypatch.setattr(log_shipping, "_exporter", exporter)
    monkeypatch.setattr("elasticapm.contrib.starlette.make_apm_client", lambda config: None)
    caplog.set_level(logging.INFO)

    def flush(*, with_resource: bool = False):
        log_shipping._provider.force_flush()
        sent = [
            data
            for data in exporters[-1].get_finished_logs()
            if not str((data.log_record.attributes or {}).get("logger")).startswith("ocr_common.")
        ]
        return sent if with_resource else [data.log_record for data in sent]

    yield flush
    log_shipping.stop()


def _emit(logger: str, level: int, message: str, *args, **extra) -> None:
    logging.getLogger(logger).log(level, message, *args, extra=extra or None)


def test_off_without_a_server_url_or_the_switch(monkeypatch):
    monkeypatch.delitem(sys.modules, "opentelemetry.sdk._logs", raising=False)

    assert log_shipping.start(_settings(elastic_apm_log_sending=True), "s") is None
    assert log_shipping.start(_settings(elastic_apm_server_url="http://apm:8200"), "s") is None
    create_app(settings=_settings(), title="t", description="d", service_name="scoring")

    assert log_shipping._handler is None
    assert "opentelemetry.sdk._logs" not in sys.modules, "nothing loaded while off"


def test_records_go_under_the_apm_service_name_and_environment(exported):
    create_app(settings=_on(), title="t", description="d", version="1.2.3", service_name="scoring")
    _emit("test.ship", logging.INFO, "hello %s", "world")

    [sent] = exported(with_resource=True)
    record, resource = sent.log_record, sent.resource.attributes
    assert record.body == "hello world"
    assert (record.severity_text, record.severity_number.name) == ("INFO", "INFO")
    assert resource["service.name"] == "nilam-ocr-npwp-scoring", "the APM service's name"
    assert resource["deployment.environment"] == "local"
    assert resource["service.version"] == "1.2.3"
    assert record.attributes["logger"] == "test.ship"


def test_the_service_name_and_environment_follow_the_apm_settings(exported):
    settings = _on(elastic_apm_service_name="ms-bribrain-nilam-ocr-npwp-scoring", elastic_apm_environment="dev")
    create_app(settings=settings, title="t", description="d", service_name="scoring")
    _emit("test.ship", logging.WARNING, "careful")

    [sent] = exported(with_resource=True)
    assert sent.resource.attributes["service.name"] == "ms-bribrain-nilam-ocr-npwp-scoring"
    assert sent.resource.attributes["deployment.environment.name"] == "dev"
    assert (sent.log_record.severity_text, sent.log_record.severity_number.name) == ("WARNING", "WARN")


def test_the_server_credential_is_sent_as_the_agent_sends_it():
    assert log_shipping.auth_headers(_on(elastic_apm_secret_token="t")) == {"Authorization": "Bearer t"}
    assert log_shipping.auth_headers(_on(elastic_apm_secret_token="t", elastic_apm_api_key="a")) == {
        "Authorization": "ApiKey a"
    }
    assert log_shipping.auth_headers(_on()) == {}


def test_a_record_carries_the_trace_and_span_of_the_active_transaction(exported, apm_client):
    log_shipping.start(_on(), "s")
    _emit("test.ship", logging.INFO, "outside")
    with apm.job_transaction("OCR", "REQ_1"):
        job = execution_context.get_transaction()
        _emit("test.ship", logging.INFO, "inside")

    outside, inside = exported()
    assert not outside.trace_id
    assert inside.trace_id == int(job.trace_parent.trace_id, 16)
    assert inside.span_id == int(job.id, 16)


def test_request_id_and_event_fields_are_flat_attributes_that_never_replace_the_fixed_ones(exported):
    log_shipping.start(_on(), "s")
    token = bind_request_id("REQ_2")
    try:
        log_event(
            logging.getLogger("test.events"),
            logging.INFO,
            "sent",
            event={"dataset": "outbox", "action": "retry"},
            outbox={"status_code": 500, "error": None},
            request_id="forged",
        )
    finally:
        reset_request_id(token)

    [record] = exported()
    assert record.attributes["request_id"] == "REQ_2"
    assert record.attributes["event.dataset"] == "outbox"
    assert record.attributes["event.action"] == "retry"
    assert record.attributes["outbox.status_code"] == 500
    assert "outbox.error" not in record.attributes, "None is left out"


def test_an_exception_goes_with_its_stack_trace(exported):
    log_shipping.start(_on(), "s")
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("test.ship").exception("failed")

    [record] = exported()
    assert record.attributes["exception.type"] == "ValueError"
    assert record.attributes["exception.message"] == "boom"
    assert "raise ValueError" in record.attributes["exception.stacktrace"]


def test_no_successful_probe_no_own_logger_and_nothing_below_the_level(exported):
    log_shipping.start(_on(elastic_apm_log_level="warning"), "s")
    access = '%s - "%s %s HTTP/%s" %d'
    _emit("uvicorn.access", logging.WARNING, access, "10.0.0.1:1", "GET", "/ready", "1.1", 200)
    _emit("uvicorn.access", logging.WARNING, access, "10.0.0.1:1", "GET", "/ready", "1.1", 503)
    _emit("opentelemetry.exporter.otlp", logging.WARNING, "export failed")
    _emit("urllib3.connectionpool", logging.WARNING, "retrying")
    _emit("test.ship", logging.INFO, "below the level")

    [record] = exported()
    assert "/ready" in record.body and "503" in record.body, "only the failed probe"


def test_a_new_app_replaces_the_handler_instead_of_adding_one(exported):
    create_app(settings=_on(), title="t", description="d", service_name="scoring")
    create_app(settings=_on(), title="t", description="d", service_name="scoring")
    _emit("test.ship", logging.INFO, "once")

    assert [record.body for record in exported()] == ["once"]
    assert sum(isinstance(h, log_shipping.OtlpLogHandler) for h in logging.getLogger().handlers) == 1
