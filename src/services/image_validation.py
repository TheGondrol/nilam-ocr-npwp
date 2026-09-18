"""Validasi upload gambar yang dipakai bersama oleh service guardrails dan ekstraksi."""

from src.core.config import Settings
from src.core.errors import ServiceError


def validate_image(content_type: str | None, content: bytes, settings: Settings) -> None:
    content_type = (content_type or "").lower()
    if content_type not in settings.allowed_content_types:
        raise ServiceError(400, f"Unsupported content type: {content_type or 'unknown'}")
    if not content:
        raise ServiceError(400, "Uploaded file is empty")
    if len(content) > settings.max_upload_bytes:
        raise ServiceError(400, f"File exceeds {settings.max_upload_bytes // (1024 * 1024)}MB limit")
