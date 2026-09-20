from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="API key for this service")

# Satu-satunya sumber pesan 401. Diimpor juga oleh router untuk contoh
# response di /docs supaya spec dan jawaban service tidak berbeda kalimat.
API_KEY_ERROR = "Invalid or missing API key"


async def verify_api_key(request: Request, api_key: str | None = Security(api_key_header)) -> str:
    # Settings diambil dari app.state (dipasang create_app), bukan di-import:
    # lib ini tidak boleh tahu kelas Settings milik service mana pun.
    settings = request.app.state.settings
    if settings.auth_disabled:
        return api_key or ""
    expected = settings.api_key
    if not api_key or api_key != expected:
        raise HTTPException(status_code=401, detail=API_KEY_ERROR)
    return api_key
