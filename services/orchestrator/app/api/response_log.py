"""ASGI middleware that logs every answer to `POST /v1/extract-ocr` (and its `-test` twin) before it is sent: it
holds the response back, hands its envelope to the response log of the path (`app.state.response_logs`, keyed by
table prefix), then sends it. Middleware rather than code in the endpoint, so the answers made outside it are
logged too: a missing field (422), a wrong API key (401), a file refused by the error handler (400, 413).

`guardrails` is null when the request's pipeline_name_sequence left guardrails out: the endpoint leaves the
parsed sequence in `request.state.pipeline_name_sequence`."""

import json
import logging
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ocr_common.pipeline import GUARDRAILS
from ocr_common.testing_endpoints import TESTING_TABLE_PREFIX

logger = logging.getLogger(__name__)

# The logged paths -> the table prefix of their log.
LOGGED_PATHS = {"/v1/extract-ocr": "", "/v1/extract-ocr-test": TESTING_TABLE_PREFIX}
SEQUENCE_STATE = "pipeline_name_sequence"


class ResponseLogMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in LOGGED_PATHS:
            await self.app(scope, receive, send)
            return
        start: Message | None = None
        chunks: list[bytes] = []

        async def hold_back(message: Message) -> None:
            nonlocal start
            if message["type"] == "http.response.start":
                start = message
                return
            if message["type"] != "http.response.body" or start is None:
                await send(message)
                return
            chunks.append(message.get("body", b""))
            if message.get("more_body", False):
                return
            content = b"".join(chunks)
            await self._record(scope, start["status"], content)
            await send(start)
            await send({"type": "http.response.body", "body": content})

        await self.app(scope, receive, hold_back)

    async def _record(self, scope: Scope, status_code: int, content: bytes) -> None:
        try:
            body = json.loads(content)
        except ValueError:
            body = None
        request_id = body.get("request_id") if isinstance(body, dict) else None
        if not isinstance(body, dict) or not isinstance(request_id, str) or not request_id:
            logger.warning("nilam_ocr_results: answer %d without a request_id, not recorded", status_code)
            return
        logs = getattr(scope["app"].state, "response_logs", None) or {}
        log = logs.get(LOGGED_PATHS[scope["path"]])
        if log is None:
            return
        await log.record(request_id, status_code, body, guardrails=_guardrails(scope, body))


def _guardrails(scope: Scope, body: dict[str, Any]) -> int | float | None:
    """The answer's `guardrails`, or None when the request's sequence left guardrails out."""
    sequence = (scope.get("state") or {}).get(SEQUENCE_STATE)
    if sequence is not None and GUARDRAILS not in sequence:
        return None
    return body.get("guardrails")
