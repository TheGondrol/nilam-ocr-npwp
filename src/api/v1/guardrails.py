"""
API (presentation) layer: parses HTTP input, calls the service layer, and
translates ServiceError into the right HTTPException. No business rules
live here; see src/services/guardrails_service.py for those.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from src.api.v1.intake import FileField, FileUrlField, read_image
from src.core.config import get_settings
from src.core.envelope import envelope
from src.core.errors import ServiceError
from src.core.request_id import get_request_id
from src.core.security import API_KEY_ERROR, verify_api_key
from src.models.guardrails import get_document_classifier, get_quality_assessor
from src.schemas.errors import error
from src.schemas.guardrails import GuardrailCheckResponse
from src.services.guardrails_service import GuardrailsService

router = APIRouter(tags=["Guardrails"], dependencies=[Depends(verify_api_key)])


def get_guardrails_service() -> GuardrailsService:
    return GuardrailsService(get_quality_assessor(), get_document_classifier(), get_settings())


@router.post(
    "/v1/guardrails/check",
    response_model=GuardrailCheckResponse,
    summary="Run guardrail checks on a document image",
    description=(
        "Checks image quality and whether the image is the expected document type. "
        "Always returns 200 with a report; `data.passed` tells whether the image should proceed to OCR. "
        "Send the image as `file` or `file_url`; exactly one of the two."
    ),
    responses={
        400: error(400, "Bad file (empty, too large, unsupported type) or bad intake", "Uploaded file is empty"),
        401: error(401, "Missing or invalid X-API-Key", API_KEY_ERROR),
        422: error(422, "Validation Error", "body.file: Expected UploadFile, received: str", errors="VALIDATION_ERROR"),
    },
)
async def check(
    request: Request,
    file: UploadFile | str | None = FileField,
    file_url: str | None = FileUrlField,
    service: GuardrailsService = Depends(get_guardrails_service),
):
    content, filename, content_type = await read_image(file, file_url, limit=get_settings().max_upload_bytes)
    try:
        data = service.check(filename, content_type, content)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    return envelope(200, "Success", data, get_request_id(request))
