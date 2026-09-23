"""The checks every uploaded document goes through before a model sees it."""

from ocr_common.config import BaseServiceSettings
from ocr_common.errors import BadRequest


def validate_image(content_type: str | None, content: bytes, settings: BaseServiceSettings) -> None:
    """Raises `BadRequest` when the content type is not allowed, the upload is empty, or it exceeds
    `MAX_UPLOAD_BYTES`.
    """
    content_type = (content_type or "").lower()
    if content_type not in settings.allowed_content_types:
        raise BadRequest(f"Unsupported content type: {content_type or 'unknown'}")
    if not content:
        raise BadRequest("Uploaded file is empty")
    if len(content) > settings.max_upload_bytes:
        raise BadRequest(f"File exceeds {settings.max_upload_bytes // (1024 * 1024)}MB limit")
