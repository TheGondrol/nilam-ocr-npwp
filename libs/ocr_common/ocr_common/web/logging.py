"""Process-wide logging for the services: one line per record, either JSON (for Cloud Logging) or text
(for a terminal), and every record carries the `request_id` of the request or job it belongs to.

`request_id` comes from the contextvar that `RequestIdMiddleware` binds for the duration of a request
and that the pipeline binds for the duration of a background job, so a log line written deep inside a
model client still says which request it was for."""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, Literal

from ocr_common.web.request_id import current_request_id

LogFormat = Literal["json", "text"]

TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s]: %(message)s"
_MARK = "_ocr_common_handler"
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


class RequestIdFilter(logging.Filter):
    """Adds `request_id` to every record (`-` outside a request or job) so formats can rely on it."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = current_request_id() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line. `severity` and `message` are the field names Cloud Logging reads."""

    def __init__(self, service: str | None) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or "-",
        }
        if self._service:
            entry["service"] = self._service
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
    handler.setFormatter(JsonFormatter(service) if fmt == "json" else logging.Formatter(TEXT_FORMAT))

    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not getattr(h, _MARK, False)] + [handler]
    root.setLevel(level.upper())
    for name in _UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True
