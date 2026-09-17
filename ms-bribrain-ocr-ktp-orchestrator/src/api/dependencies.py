"""
API dependencies for authentication and authorization.
"""

import hmac
import os
import re
from typing import Optional

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from src.api.models import ErrorCode

# Reusable request_id format validator
REQUEST_ID_PATTERN = re.compile(
    r"^OCR_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    expected = os.getenv("ORCHESTRATOR_SERVICE_API", "")
    if not api_key or not expected or not hmac.compare_digest(api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "status_code": status.HTTP_401_UNAUTHORIZED,
                "status_desc": "Unauthorized",
                "message": "Invalid or missing API key",
                "data": "",
                "error_code": ErrorCode.UNAUTHORIZED,
                "errors": None,
                "request_id": None,
            },
        )


async def reject_request_body(request: Request) -> None:
    """Reject GET requests carrying a body (GET bodies have no defined semantics)."""
    content_length = request.headers.get("content-length")
    has_body = (
        (content_length is not None and content_length.isdigit() and int(content_length) > 0)
        or request.headers.get("transfer-encoding") is not None
    )
    if has_body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status_code": status.HTTP_400_BAD_REQUEST,
                "status_desc": "Bad Request",
                "message": "GET requests must not include a request body.",
                "data": "",
                "error_code": ErrorCode.BODY_NOT_ALLOWED,
                "errors": None,
                "request_id": None,
            },
        )


def validate_request_id(request_id: str) -> str:
    """Validate request_id matches OCR_{uuid4} format. Reusable across endpoints."""
    if not REQUEST_ID_PATTERN.match(request_id):
        raise HTTPException(
            status_code=400,
            detail={
                "status_code": 400,
                "status_desc": "Bad Request",
                "message": "Invalid request ID format. Expected format: OCR_{uuid4}.",
                "data": "",
                "error_code": ErrorCode.INVALID_REQUEST_ID,
                "errors": None,
                "request_id": request_id,
            },
        )
    return request_id
