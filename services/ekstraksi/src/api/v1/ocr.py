from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile

from ocr_common.envelope import envelope
from ocr_common.errors import ServiceError
from ocr_common.intake import FileField, FileUrlField, read_image
from ocr_common.schemas import UNAUTHORIZED, error, success_examples
from ocr_common.security import verify_api_key
from src.api.v1.ekstraksi import get_ekstraksi_service
from src.clients.stages import get_stage_clients
from src.repositories.request_repository import get_request_repository
from src.schemas.ocr import RID, ExtractOcrResponse, GenerateRequestIdResponse, GetOcrResultResponse
from src.services.ocr_service import OcrService

router = APIRouter(tags=["NPWP OCR"], dependencies=[Depends(verify_api_key)])

LEGACY_NOTE = (
    "**Legacy synchronous contract**, kept until the orchestrator has moved to the asynchronous pipeline "
    "(`POST /v1/ekstraksi/jobs`). "
)


def get_ocr_service() -> OcrService:
    return OcrService(get_request_repository(), get_ekstraksi_service(), get_stage_clients())


@router.post(
    "/v1/generate-request-id",
    response_model=GenerateRequestIdResponse,
    operation_id="generateRequestId",
    deprecated=True,
    summary="Legacy: mint a request_id",
    description=(
        LEGACY_NOTE
        + "Mints a request_id that must be used once in POST /v1/extract-ocr. In the asynchronous pipeline the "
        "orchestrator mints the request_id itself."
    ),
    responses={
        200: success_examples(
            "A new request_id",
            minted=("Use it once in POST /v1/extract-ocr", envelope(200, "Success", {"request_id": RID}, RID)),
        ),
        401: UNAUTHORIZED,
    },
)
async def generate_request_id(service: OcrService = Depends(get_ocr_service)):
    request_id = await service.generate_request_id()
    return envelope(200, "Success", {"request_id": request_id}, request_id)


@router.post(
    "/v1/extract-ocr",
    response_model=ExtractOcrResponse,
    operation_id="extractOcr",
    deprecated=True,
    summary="Legacy: run the whole chain synchronously",
    description=(
        LEGACY_NOTE + "Runs the full pipeline (guardrails service -> this service's OCR -> structuring service "
        "-> scoring service) on the uploaded NPWP image for the given request_id "
        "(from /v1/generate-request-id). A request_id can only be submitted once.\n\n"
        "Send the image as `file`, or send `file_url` and this service fetches "
        "it itself; exactly one of the two."
    ),
    responses={
        200: success_examples(
            "The document was read",
            person=(
                "A person's card",
                envelope(
                    200,
                    "Success",
                    {
                        "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0.9992},
                        "nama": {"value": "BUDI SANTOSO", "confidence": 0.9773},
                        "nama_badan": {"value": None, "confidence": 0.0},
                    },
                    RID,
                    guardrails=0.9904,
                ),
            ),
        ),
        400: error(
            400,
            "Unknown request_id, bad file, or rejected by guardrails",
            "Document rejected by guardrails: 1/1 page(s) rejected (confidence 0.88)",
        ),
        401: UNAUTHORIZED,
        409: error(409, "request_id was already processed", f"request_id {RID} has already been processed"),
        500: error(500, "OCR engine or a downstream stage failed", "Internal server error while processing OCR"),
        503: error(503, "OCR model or a downstream stage unreachable", "ekstraksi OCR model is unavailable"),
        504: error(
            504, "OCR model or a downstream stage did not answer in time", "ekstraksi OCR model timed out after 30.0s"
        ),
        422: error(422, "Validation Error", "body.request_id: Field required", errors="VALIDATION_ERROR"),
    },
)
async def extract_ocr(
    request: Request,
    request_id: str = Form(..., description="request_id from /v1/generate-request-id"),
    file: UploadFile | str | None = FileField,
    file_url: str | None = FileUrlField,
    service: OcrService = Depends(get_ocr_service),
):
    content, filename, content_type = await read_image(request, file, file_url)
    try:
        data, guardrails = await service.extract(request_id, filename, content_type, content)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    return envelope(200, "Success", data, request_id, guardrails=guardrails)


@router.get(
    "/v1/get-ocr-result/{request_id}",
    response_model=GetOcrResultResponse,
    operation_id="getOcrResult",
    deprecated=True,
    summary="Legacy: status and result of an extract-ocr request",
    description=(
        LEGACY_NOTE + "Current status and, once completed, the result of a request_id minted by this service."
    ),
    responses={
        200: success_examples(
            "The request_id exists",
            completed=(
                "Finished",
                envelope(
                    200,
                    "Success",
                    {
                        "request_id": RID,
                        "status": "completed",
                        "result": {
                            "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0.9992},
                            "nama": {"value": "BUDI SANTOSO", "confidence": 0.9773},
                            "nama_badan": {"value": None, "confidence": 0.0},
                        },
                        "error_message": None,
                        "created_at": "2026-03-06T07:01:59.976912+00:00",
                        "updated_at": "2026-03-06T07:07:04.809697+00:00",
                    },
                    RID,
                    guardrails=0.9904,
                ),
            ),
            pending=(
                "Minted, but extract-ocr has not been called (or is still running)",
                envelope(
                    200,
                    "Success",
                    {
                        "request_id": RID,
                        "status": "pending",
                        "result": None,
                        "error_message": None,
                        "created_at": "2026-03-06T07:01:59.976912+00:00",
                        "updated_at": "2026-03-06T07:01:59.976912+00:00",
                    },
                    RID,
                    guardrails=None,
                ),
            ),
        ),
        401: error(401, "Missing or invalid X-API-Key", "Invalid or missing API key", request_id=RID),
        404: error(
            404,
            "request_id was never generated by this service",
            f"No data found for request_id: {RID}",
            request_id=RID,
        ),
        422: error(
            422, "Validation Error", "path.request_id: Field required", request_id=RID, errors="VALIDATION_ERROR"
        ),
    },
)
async def get_ocr_result(request_id: str, service: OcrService = Depends(get_ocr_service)):
    try:
        data = await service.get_result(request_id)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    return envelope(200, "Success", data, request_id, guardrails=data.pop("guardrails", None))
