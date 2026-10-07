"""Process-wide logging for the services: one line per record, either JSON (for Cloud Logging) or text
(for a terminal), and every record carries the `request_id` of the request or job it belongs to.

`request_id` comes from the contextvar that `RequestIdMiddleware` binds for the duration of a request
and that the pipeline binds for the duration of a background job, so a log line written deep inside a
model client still says which request it was for.

Deployed, the lines go to stderr only: the cluster's Filebeat DaemonSet collects every container's output and
ships it to Elasticsearch (deploy/helm/README.md, "Log ke Elasticsearch"), so no service talks to Elasticsearch
itself. A record may carry structured fields for it (`log_event`), e.g. one event per outbox delivery; the access
log of the probes (`/health`, `/ready`, `/metrics`, every few seconds per pod) is left out unless it failed."""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, Literal

from ocr_common.web import apm
from ocr_common.web.request_id import current_request_id

LogFormat = Literal["json", "text"]

TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s]: %(message)s"
_MARK = "_ocr_common_handler"
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")
# Paths the kubelet and Prometheus call every few seconds; a successful call is not worth a log line.
PROBE_PATHS = frozenset({"/health", "/ready", "/metrics"})
# The record attribute `log_event` puts its fields on.
_FIELDS = "event_fields"


def log_event(logger: logging.Logger, level: int, message: str, /, **fields: Any) -> None:
    """Logs `message` with structured `fields` that the JSON format puts on the line as they are (nested objects
    stay objects, e.g. `event={"dataset": "outbox", "action": "delivered"}`), so they can be searched in
    Elasticsearch; the text format shows only the message. Never put document data (NPWP, names) in them."""
    logger.log(level, message, extra={_FIELDS: fields})


class ProbeAccessFilter(logging.Filter):
    """Drops uvicorn's access line of a successful probe call (`PROBE_PATHS`); a failed one is kept."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "uvicorn.access" or not isinstance(record.args, tuple) or len(record.args) < 5:
            return True
        path, status = record.args[2], record.args[4]
        return not (str(path).split("?", 1)[0] in PROBE_PATHS and isinstance(status, int) and status < 400)


class RequestIdFilter(logging.Filter):
    """Adds `request_id` to every record (`-` outside a request or job) so formats can rely on it."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = current_request_id() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line. `severity` and `message` are the field names Cloud Logging reads; `@timestamp`
    and `log.level` the ones Elasticsearch (ECS) reads."""

    def __init__(self, service: str | None) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        time = datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds")
        entry: dict[str, Any] = {
            "@timestamp": time,
            "time": time,
            "severity": record.levelname,
            "log.level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or "-",
        }
        if self._service:
            entry["service"] = self._service
        entry.update(apm.trace_fields())  # `trace.id`, `transaction.id` while an APM transaction is active
        fields = getattr(record, _FIELDS, None)
        if isinstance(fields, dict):
            # Never over the fixed fields: a search on `message` or `request_id` must mean the same everywhere.
            entry.update({key: value for key, value in fields.items() if key not in entry})
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def configure_logging(*, fmt: LogFormat, level: str, service: str | None = None) -> None:
    """Installs one stderr handler on the root logger (replacing an earlier one of ours, never a
    handler someone else added, such as pytest's) and routes uvicorn's loggers through it, so the
    access log has the same shape as the application log."""
    handler = logging.StreamHandler(sys.stderr)
    setattr(handler, _MARK, True)
    handler.addFilter(RequestIdFilter())
    handler.addFilter(ProbeAccessFilter())
    handler.setFormatter(JsonFormatter(service) if fmt == "json" else logging.Formatter(TEXT_FORMAT))

    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not getattr(h, _MARK, False)] + [handler]
    root.setLevel(level.upper())
    for name in _UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True
