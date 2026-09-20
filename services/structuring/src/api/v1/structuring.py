from fastapi import APIRouter, Depends, HTTPException, Request

from ocr_common.envelope import envelope
from ocr_common.errors import ServiceError
from ocr_common.request_id import get_request_id
from ocr_common.schemas import REQUEST_ID_EXAMPLE, UNAUTHORIZED, error, success_examples
from ocr_common.security import verify_api_key
from src.models.structuring import get_structurer
from src.schemas.structuring import StructureRequest, StructureResponse
from src.services.structuring_service import StructuringService

router = APIRouter(tags=["Structuring"], dependencies=[Depends(verify_api_key)])

# Dipakai juga oleh contoh GET /v1/structuring/jobs/{request_id}: hasil tahap ini = bentuk ini.
STRUCTURED_EXAMPLE = {
    "document_type": "npwp",
    "fields": {
        "nomor_npwp": {
            "value": "12.345.678.9-012.345",
            "confidence": 0.9992,
            "source": "NPWP : 12.345.678.9-012.345",
            "signals": {"has_homoglyph": False, "candidate_count": 1},
        },
        "nama": {
            "value": "BUDI SANTOSO",
            "confidence": 0.9773,
            "source": "BUDI SANTOSO",
            "signals": {"corrected": False},
        },
        "nama_badan": {"value": None, "confidence": 0.0, "source": None, "signals": None},
    },
}


def get_structuring_service() -> StructuringService:
    return StructuringService(get_structurer())


@router.post(
    "/v1/structuring/structure",
    response_model=StructureResponse,
    operation_id="structureLines",
    summary="Turn OCR lines into named fields, synchronous (no job, no callback)",
    description=(
        "Maps raw text lines (e.g. from the ekstraksi service) to named document fields " "with per-field confidence."
    ),
    responses={
        200: success_examples(
            "The fields found in the lines",
            person=("A person's card", envelope(200, "Success", STRUCTURED_EXAMPLE, REQUEST_ID_EXAMPLE)),
        ),
        400: error(
            400,
            (
                "No usable text, or the upload is not a lone NPWP card "
                "(other document, CAPTCHA, lookup screenshot, more than 2 pages)"
            ),
            "No text lines to structure",
        ),
        401: UNAUTHORIZED,
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
