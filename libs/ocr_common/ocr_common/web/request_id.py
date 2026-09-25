"""The request_id of the work in progress, available anywhere without passing it around.

`RequestIdMiddleware` binds it for the duration of an HTTP request (from the `X-Request-ID` header, or
a fresh one) and echoes it in the response. The pipeline binds it for the duration of a background job
and of an outbox delivery. Log lines pick it up through `ocr_common.web.logging`, and
`RemoteModelClient` forwards it as `X-Request-ID` to the next service. The orchestrator binds the
central orchestrator's request_id around its calls, so one id follows a request through the orchestrator,
guardrails, OCR, structuring and scoring."""

import contextvars
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

REQUEST_ID_HEADER = "X-Request-ID"

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


def current_request_id() -> str | None:
    """The request_id bound to the current request or job, if any."""
    return _request_id.get()


def bind_request_id(request_id: str | None) -> contextvars.Token:
    """Binds `request_id` for the current context; pass the token to `reset_request_id` when done."""
    return _request_id.set(request_id)


def reset_request_id(token: contextvars.Token) -> None:
    """Restores what was bound before the matching `bind_request_id`."""
    _request_id.reset(token)


def adopt_request_id(request: Request, request_id: str) -> contextvars.Token:
    """Makes `request_id`, sent in the body (e.g. the form field of `extract-ocr`), the id of this request: the
    error envelope of anything raised afterwards, the `X-Request-ID` response header, log lines and outbound
    calls all carry it instead of the header's or a generated one. Pass the token to `reset_request_id`."""
    request.state.request_id = request_id
    return bind_request_id(request_id)


def get_request_id(request: Request) -> str | None:
    """The request_id to put in a response envelope: the path parameter when the route has one, else
    the one the middleware bound (header or generated)."""
    return request.path_params.get("request_id") or getattr(request.state, "request_id", None)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Reads or mints the request_id, binds it for the request, and echoes it in the response header."""

    async def dispatch(self, request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or f"REQ_{uuid.uuid4()}"
        request.state.request_id = request_id
        token = bind_request_id(request_id)
        try:
            response = await call_next(request)
        finally:
            reset_request_id(token)
        # The handler may have adopted the caller's own request_id (adopt_request_id); echo that one.
        response.headers[REQUEST_ID_HEADER] = getattr(request.state, "request_id", None) or request_id
        return response
