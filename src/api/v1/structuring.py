from fastapi import APIRouter, Depends, HTTPException, Request

from src.core.envelope import envelope
from src.core.errors import ServiceError
from src.core.request_id import get_request_id
from src.core.security import API_KEY_ERROR, verify_api_key
from src.models.structuring import get_structurer
from src.schemas.errors import error
from src.schemas.structuring import StructureRequest, StructureResponse
from src.services.structuring_service import StructuringService

router = APIRouter(tags=["Structuring"], dependencies=[Depends(verify_api_key)])


def get_structuring_service() -> StructuringService:
    return StructuringService(get_structurer())


@router.post(
    "/v1/structuring/structure",
    response_model=StructureResponse,
    summary="Turn raw OCR lines into named fields",
    description=(
        "Maps raw text lines (e.g. from /v1/ekstraksi/extract) to named document fields " "with per-field confidence."
    ),
    responses={
        400: error(400, "No usable text lines", "No text lines to structure"),
        401: error(401, "Missing or invalid X-API-Key", API_KEY_ERROR),
        422: error(422, "Validation Error", "body.lines: Field required", errors="VALIDATION_ERROR"),
    },
)
async def structure(
    request: Request,
    body: StructureRequest,
    service: StructuringService = Depends(get_structuring_service),
):
    try:
        data = service.structure([line.model_dump() for line in body.lines])
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    return envelope(200, "Success", data, get_request_id(request))
