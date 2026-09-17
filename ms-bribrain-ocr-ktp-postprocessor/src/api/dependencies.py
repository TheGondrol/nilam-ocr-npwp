"""
API dependencies for authentication and authorization.
"""

import hmac
import os

from fastapi import Security, HTTPException, status, Request
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")


async def verify_api_key(api_key: str = Security(api_key_header)):
    expected = os.getenv("API_KEY")
    # Constant-time compare to avoid leaking the key via timing. Reject outright
    # if the server has no key configured rather than risk a None == None match.
    if not expected or not hmac.compare_digest(api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


async def require_json_content_type(request: Request) -> None:
    """Reject requests whose Content-Type is not application/json (415).

    FastAPI parses a JSON body regardless of the request's Content-Type, so
    without this guard a JSON endpoint would accept a body sent as text/plain.
    The ``; charset=utf-8`` suffix is tolerated by splitting on ``;``.
    """
    media_type = (request.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Content-Type must be application/json",
        )
