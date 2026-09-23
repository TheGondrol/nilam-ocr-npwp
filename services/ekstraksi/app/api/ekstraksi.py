from fastapi import APIRouter, Depends, Request, UploadFile

from ocr_common.web.envelope import envelope
from ocr_common.web.intake import FileField, FileUrlField, read_image
from ocr_common.web.request_id import get_request_id
from ocr_common.web.schemas import REQUEST_ID_EXAMPLE, UNAUTHORIZED, error, success_examples
from ocr_common.web.security import verify_api_key

from app.api.schemas import ExtractResponse
from app.dependencies import get_ekstraksi_service
from app.services.ekstraksi_service import EkstraksiService

router = APIRouter(tags=["Ekstraksi"], dependencies=[Depends(verify_api_key)])

OCR_RESULT_EXAMPLE = {
    "engine": "paddle",
    "model": "PP-OCRv6_medium_det+PP-OCRv6_medium_rec",
    "elapsed_ms": 412.5,
    "full_text": "DIREKTORAT JENDERAL PAJAK\nNPWP : 12.345.678.9-012.345\nBUDI SANTOSO",
    "blocks": [
        {
            "text": "DIREKTORAT JENDERAL PAJAK",
            "confidence": 0.9985,
            "bbox": {"x1": 175, "y1": 247, "x2": 887, "y2": 308},
            "page": 0,
        },
        {
            "text": "NPWP : 12.345.678.9-012.345",
            "confidence": 0.9992,
            "bbox": {"x1": 35, "y1": 389, "x2": 637, "y2": 437},
            "page": 0,
        },
        {"text": "BUDI SANTOSO", "confidence": 0.9773, "bbox": {"x1": 33, "y1": 480, "x2": 483, "y2": 523}, "page": 0},
    ],
}


@router.post(
    "/v1/ekstraksi/extract",
    response_model=ExtractResponse,
    operation_id="extractText",
    summary="Raw OCR of a document, synchronous (no job, no callback)",
    description=(
        "Runs the OCR engine and returns raw text blocks with confidence and bounding boxes. "
        "Send the image as `file` or `file_url`; exactly one of the two."
    ),
    responses={
        200: success_examples(
            "Text lines found in the document",
            npwp_card=("An NPWP card", envelope(200, "Success", OCR_RESULT_EXAMPLE, REQUEST_ID_EXAMPLE)),
        ),
        400: error(400, "Bad file (empty, too large, unsupported type) or bad intake", "Uploaded file is empty"),
        401: UNAUTHORIZED,
        500: error(500, "OCR engine failed", "ekstraksi OCR model error (500): error: OpenCV ..."),
        503: error(503, "OCR model unreachable", "ekstraksi OCR model is unavailable"),
        504: error(504, "OCR model did not answer in time", "ekstraksi OCR model timed out after 30.0s"),
        422: error(422, "Validation Error", "body.file: Expected UploadFile, received: str", errors="VALIDATION_ERROR"),
    },
)
async def extract(
    request: Request,
    file: UploadFile | str | None = FileField,
    file_url: str | None = FileUrlField,
    service: EkstraksiService = Depends(get_ekstraksi_service),
):
    content, filename, content_type = await read_image(request, file, file_url)
    data = await service.extract(filename, content_type, content)
    return envelope(200, "Success", data, get_request_id(request))
