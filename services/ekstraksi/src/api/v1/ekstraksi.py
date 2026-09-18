from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from ocr_common.envelope import envelope
from ocr_common.errors import ServiceError
from ocr_common.intake import FileField, FileUrlField, read_image
from ocr_common.request_id import get_request_id
from ocr_common.schemas import UNAUTHORIZED, error
from ocr_common.security import verify_api_key
from src.core.config import get_settings
from src.models.ekstraksi import get_ocr_engine
from src.schemas.ekstraksi import ExtractResponse
from src.services.ekstraksi_service import EkstraksiService

router = APIRouter(tags=["Ekstraksi"], dependencies=[Depends(verify_api_key)])


def get_ekstraksi_service() -> EkstraksiService:
    return EkstraksiService(get_ocr_engine(), get_settings())


@router.post(
    "/v1/ekstraksi/extract",
    response_model=ExtractResponse,
    summary="Extract raw text from a document image",
    description=(
        "Runs the OCR engine and returns raw text blocks with confidence and bounding boxes. "
        "Send the image as `file` or `file_url`; exactly one of the two."
    ),
    responses={
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
    try:
        data = await service.extract(filename, content_type, content)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    return envelope(200, "Success", data, get_request_id(request))
