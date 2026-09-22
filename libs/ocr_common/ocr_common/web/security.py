import secrets

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="API key for this service")

API_KEY_ERROR = "Invalid or missing API key"


async def verify_api_key(request: Request, api_key: str | None = Security(api_key_header)) -> str:
    settings = request.app.state.settings
    if settings.auth_disabled:
        return api_key or ""
    expected = settings.api_key
    if not api_key or not secrets.compare_digest(api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail=API_KEY_ERROR)
    return api_key
