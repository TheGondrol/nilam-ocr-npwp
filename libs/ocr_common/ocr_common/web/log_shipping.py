"""The application log to Elasticsearch through the APM server: every record also goes, as an OpenTelemetry log
record over OTLP/HTTP, to `ELASTIC_APM_SERVER_URL` + `/v1/logs`, and the APM server writes it to the data stream
`logs-apm.app.<service>-default` that Kibana reads. The central orchestrator's log takes the same route.

The shared cluster collects no workload output (no Filebeat, GKE keeps system logs only), so without this a line
on stderr is only in `kubectl logs`. Off unless `ELASTIC_APM_LOG_SENDING=true` with a server URL, and then the
OpenTelemetry SDK is not even imported; stderr keeps every line either way.

A record goes under the APM service name and with the trace and span of the active APM transaction, so Kibana
shows it on the service's Logs tab and next to its trace; the node the pod runs on is a resource attribute,
`k8s.node.name`. `service` (the component), `logger`, `request_id` and the `log_event` fields are attributes,
nested ones flattened (`outbox.status_code`); the APM server keeps them as `labels.*` / `numeric_labels.*`.
The rules of stderr apply: no record for a successful probe, never document data. Records are sent in batches
from a background thread and dropped when its queue is full, so a request never waits for the log; what is left
at exit is flushed. TLS to the server is always verified (`ELASTIC_APM_VERIFY_SERVER_CERT` is the agent's).
"""

import logging
import traceback
from collections.abc import Iterator, Mapping
from typing import Any

from ocr_common.config import BaseServiceSettings
from ocr_common.web import apm
from ocr_common.web.logging import add_handler, event_fields

logger = logging.getLogger(__name__)

# Loggers never sent: the exporter's own (a failed export would log, be sent, fail again) and the HTTP libraries
# under it. They stay on stderr.
_OWN_LOGGERS = frozenset({"opentelemetry", "elasticapm", "urllib3", "requests"})

# The provider and handler `start` made; None while log sending is off.
_provider: Any = None
_handler: logging.Handler | None = None


def start(
    settings: BaseServiceSettings, service_name: str, version: str | None = None, component: str | None = None
) -> logging.Handler | None:
    """Sends the log to the APM server from now on, under `service_name` (the APM service name): the handler
    added to the root logger, or None (and nothing loaded) without `ELASTIC_APM_SERVER_URL` or with
    `ELASTIC_APM_LOG_SENDING` off. `component` (`scoring`, ...) goes on every record as `service`, like on
    stderr, and the node the pod runs on (`K8S_NODE_NAME`) as `k8s.node.name`. A second call replaces the first."""
    global _provider, _handler
    if not (settings.elastic_apm_server_url and settings.elastic_apm_log_sending):
        return None
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.resources import Resource

    stop()
    environment = apm.environment(settings)
    resource = {
        "service.name": service_name,
        # Both spellings: the semantic conventions renamed it, the APM server reads either.
        "deployment.environment": environment,
        "deployment.environment.name": environment,
    }
    if version:
        resource["service.version"] = version
    if settings.k8s_node_name:
        resource["k8s.node.name"] = settings.k8s_node_name
    provider = LoggerProvider(resource=Resource.create(resource))
    provider.add_log_record_processor(BatchLogRecordProcessor(_exporter(settings)))
    handler = OtlpLogHandler(provider, settings.elastic_apm_log_level.upper(), component)
    add_handler(handler)
    _provider, _handler = provider, handler
    logger.info("Log sending on: service %s, server %s", service_name, settings.elastic_apm_server_url)
    return handler


def stop() -> None:
    """Removes the handler, then flushes and closes the provider, if any (tests; at exit the provider closes
    itself)."""
    global _provider, _handler
    if _handler is not None:
        logging.getLogger().removeHandler(_handler)
        _handler = None
    if _provider is not None:
        _provider.shutdown()
        _provider = None


def _exporter(settings: BaseServiceSettings) -> Any:
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

    return OTLPLogExporter(
        endpoint=f"{str(settings.elastic_apm_server_url).rstrip('/')}/v1/logs", headers=auth_headers(settings)
    )


def auth_headers(settings: BaseServiceSettings) -> dict[str, str]:
    """The APM server's credential, as the agent sends it: the API key, else the secret token."""
    if settings.elastic_apm_api_key:
        return {"Authorization": f"ApiKey {settings.elastic_apm_api_key}"}
    if settings.elastic_apm_secret_token:
        return {"Authorization": f"Bearer {settings.elastic_apm_secret_token}"}
    return {}


class OtlpLogHandler(logging.Handler):
    """A `logging` handler that emits each record as an OpenTelemetry log record of `provider`. Not the SDK's
    `LoggingHandler` (deprecated since 1.45), which takes the trace from OpenTelemetry's own context: that one is
    empty here, the traces come from the Elastic agent."""

    def __init__(self, provider: Any, level: int | str = logging.NOTSET, component: str | None = None) -> None:
        super().__init__(level)
        self._provider = provider
        self._component = component

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.split(".", 1)[0] in _OWN_LOGGERS:
            return
        try:
            self._provider.get_logger(record.name).emit(log_record(record, self._component))
        except Exception:
            self.handleError(record)


def log_record(record: logging.LogRecord, component: str | None = None) -> Any:
    """`record` as an OpenTelemetry log record, in the trace and span of the active APM transaction, if any."""
    from opentelemetry._logs import LogRecord

    # An ABC without abstract methods: the SDK's own handler builds it the same way.
    return LogRecord(  # pyrefly: ignore[direct-abstract-base-instantiation]
        timestamp=int(record.created * 1e9),
        context=_trace_context(),
        severity_text=record.levelname,
        severity_number=_severity(record.levelno),
        body=record.getMessage(),
        attributes=attributes(record, component),
    )


def _trace_context() -> Any:
    """An OpenTelemetry context holding the active APM transaction's trace and current span; None without one."""
    ids = apm.trace_ids()
    if ids is None:
        return None
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, set_span_in_context

    trace_id, span_id = ids
    span = SpanContext(int(trace_id, 16), int(span_id, 16), is_remote=False, trace_flags=TraceFlags(TraceFlags.SAMPLED))
    return set_span_in_context(NonRecordingSpan(span))


# Python's levels on OpenTelemetry's severity numbers, highest first (the log data model's mapping).
_SEVERITIES = (
    (logging.CRITICAL, "FATAL"),
    (logging.ERROR, "ERROR"),
    (logging.WARNING, "WARN"),
    (logging.INFO, "INFO"),
    (logging.DEBUG, "DEBUG"),
)


def _severity(levelno: int) -> Any:
    from opentelemetry._logs import SeverityNumber

    for threshold, name in _SEVERITIES:
        if levelno >= threshold:
            return SeverityNumber[name]
    return SeverityNumber.TRACE


def attributes(record: logging.LogRecord, component: str | None = None) -> dict[str, Any]:
    """The attributes of `record`'s log record: `service` (the component), `logger`, `request_id`, the
    `log_event` fields flattened, and the exception, if any. A field never replaces the first three."""
    fixed = {"logger": record.name, "request_id": getattr(record, "request_id", None) or "-"}
    if component:
        fixed["service"] = component
    result = {key: value for key, value in _flatten(event_fields(record)) if key not in fixed}
    result.update(fixed)
    if record.exc_info and record.exc_info[0] is not None:
        kind, value, _ = record.exc_info
        result["exception.type"] = kind.__name__
        result["exception.message"] = str(value)
        result["exception.stacktrace"] = "".join(traceback.format_exception(*record.exc_info))
    return result


def _flatten(fields: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    """`{"outbox": {"status_code": 500}}` as `("outbox.status_code", 500)`: an attribute holds no mapping. None is
    left out; anything else that is not a str, bool, int or float becomes its str."""
    for key, value in fields.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            yield from _flatten(value, f"{name}.")
        elif value is None:
            continue
        elif isinstance(value, str | bool | int | float):
            yield name, value
        else:
            yield name, str(value)
