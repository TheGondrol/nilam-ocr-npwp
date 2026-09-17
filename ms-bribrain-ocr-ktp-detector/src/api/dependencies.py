"""
API dependencies for authentication and authorization.
"""

import hmac
import os

from fastapi import Security, HTTPException, status
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
