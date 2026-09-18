from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from src.core.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="API key for this service")

# Satu-satunya sumber pesan 401. Diimpor juga oleh router untuk contoh
# response di /docs supaya spec dan jawaban service tidak berbeda kalimat.
API_KEY_ERROR = "Invalid or missing API key"


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    settings = get_settings()
    if not api_key or api_key != settings.api_key:
        raise HTTPException(status_code=401, detail=API_KEY_ERROR)
    return api_key
