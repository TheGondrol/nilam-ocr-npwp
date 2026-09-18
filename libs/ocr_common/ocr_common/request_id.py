"""
request_id untuk endpoint yang tidak membawa request_id di form/path
(/v1/guardrails/check, /v1/ekstraksi/extract, ...): dipakai dari header
X-Request-ID kalau pemanggil mengirimnya, kalau tidak dibuat baru. Endpoint
kontrak orkestrator (/v1/extract-ocr, /v1/get-ocr-result) memakai request_id
dari form/path, sama seperti ocr-* di nilam-ocr-orchestration.
"""

import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

REQUEST_ID_HEADER = "X-Request-ID"


def get_request_id(request: Request) -> str | None:
    return request.path_params.get("request_id") or getattr(request.state, "request_id", None)


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or f"REQ_{uuid.uuid4()}"
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
