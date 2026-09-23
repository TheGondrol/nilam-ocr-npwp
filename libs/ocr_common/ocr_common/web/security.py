"""The `X-API-Key` check every endpoint but the probes depends on."""

import secrets

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="API key for this service")

API_KEY_ERROR = "Invalid or missing API key"


async def verify_api_key(request: Request, api_key: str | None = Security(api_key_header)) -> str:
    """Route dependency: 401 unless `X-API-Key` is one of the service's accepted keys (`API_KEY` plus
    `API_KEYS`, so two keys can be live during a rotation). Constant-time comparison per key."""
    settings = request.app.state.settings
    if settings.auth_disabled:
        return api_key or ""
    if not api_key or not any(
        secrets.compare_digest(api_key.encode(), expected.encode()) for expected in settings.accepted_api_keys
    ):
        raise HTTPException(status_code=401, detail=API_KEY_ERROR)
    return api_key
