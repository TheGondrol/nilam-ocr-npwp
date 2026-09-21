from ocr_common.config import BaseServiceSettings
from ocr_common.errors import ServiceError


def validate_image(content_type: str | None, content: bytes, settings: BaseServiceSettings) -> None:
    content_type = (content_type or "").lower()
    if content_type not in settings.allowed_content_types:
        raise ServiceError(400, f"Unsupported content type: {content_type or 'unknown'}")
    if not content:
        raise ServiceError(400, "Uploaded file is empty")
    if len(content) > settings.max_upload_bytes:
        raise ServiceError(400, f"File exceeds {settings.max_upload_bytes // (1024 * 1024)}MB limit")
