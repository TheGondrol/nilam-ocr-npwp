from typing import cast

from fastapi import APIRouter, Depends, Request

from ocr_common.types import OcrBlock
from ocr_common.web.envelope import envelope
from ocr_common.web.request_id import get_request_id
from ocr_common.web.schemas import REQUEST_ID_EXAMPLE, UNAUTHORIZED, error, success_examples
from ocr_common.web.security import verify_api_key

from app.api.schemas import StructureRequest, StructureResponse
from app.dependencies import get_structuring_service
from app.services.structuring_service import StructuringService

router = APIRouter(tags=["Structuring"], dependencies=[Depends(verify_api_key)])

STRUCTURED_EXAMPLE = {
    "document_type": "npwp",
    "fields": {
        "nomor_npwp": {
            "value": "12.345.678.9-012.345",
            "confidence": 0.9992,
            "source": "NPWP : 12.345.678.9-012.345",
            "signals": {
                "candidate_count": 1,
                "has_homoglyph": False,
                "invalid_province_prefix": False,
                "invalid_kecamatan_prefix": False,
                "invalid_birthdate": False,
                "invalid_kpp_prefix": False,
            },
        },
        "nama": {
            "value": "BUDI SANTOSO",
            "confidence": 0.9773,
            "source": "BUDI SANTOSO",
            "signals": {"name_base": "BUDI SANTOSO", "corrected": False},
        },
        "nama_badan": {"value": None, "confidence": 0.0, "source": None, "signals": None},
    },
    "flag": False,
    "flag_reason": None,
    "reject_reason": None,
}
FLAGGED_EXAMPLE = {
    **STRUCTURED_EXAMPLE,
    "fields": {
        **STRUCTURED_EXAMPLE["fields"],
        "nama": {
            "value": "SUKIRMAN",
            "confidence": 0.9611,
            "source": "SUKIRMAN",
            "signals": {"name_base": "SUKIRMAN", "corrected": False},
        },
    },
    "flag": True,
    "flag_reason": "Nama hanya terdiri dari 1 kata, mohon dicek kembali",
}


@router.post(
    "/v1/structuring/structure",
    response_model=StructureResponse,
    operation_id="structureLines",
    summary="Turn OCR lines into named fields, synchronous (no job, no callback)",
    description=(
        "Maps raw text lines (e.g. from the extraction service) to named document fields with per-field "
        "confidence, using the ML team's rules (`npwp_rules`). Nothing is rejected on content: a bundled "
        "second document, a CAPTCHA or lookup screenshot, a missing or single-word name, a letter in the number, "
        "an invalid Kode Wilayah / birthdate / KPP code or more than 2 pages set `flag` and `flag_reason` for the "
        "reviewer and lower the trust model's confidence."
    ),
    responses={
        200: success_examples(
            "The fields found in the lines",
            person=("A person's card", envelope(200, "Success", STRUCTURED_EXAMPLE, REQUEST_ID_EXAMPLE)),
            flagged=(
                "A single-word name: returned, but flagged for review",
                envelope(200, "Success", FLAGGED_EXAMPLE, REQUEST_ID_EXAMPLE),
            ),
        ),
        400: error(400, "No usable text in `lines`", "No text lines to structure"),
        401: UNAUTHORIZED,
        422: error(422, "Validation Error", "body.lines: Field required", errors="VALIDATION_ERROR"),
    },
)
async def structure(
    request: Request,
    body: StructureRequest,
    service: StructuringService = Depends(get_structuring_service),
):
    lines = [cast(OcrBlock, line.model_dump()) for line in body.lines]
    data = service.structure(lines)
    return envelope(200, "Success", data, get_request_id(request))
