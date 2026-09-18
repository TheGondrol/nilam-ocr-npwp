"""
API (presentation) layer tahap STRUCTURING di pipeline async. Dipanggil
ServiceOCR; menjawab 202 segera, hasilnya dikabarkan ke Orkestrasi lewat
callback. Tidak ada aturan bisnis di sini; lihat src/services/job_service.py.
"""

from fastapi import APIRouter, Depends, HTTPException

from ocr_common.envelope import envelope
from ocr_common.errors import ServiceError
from ocr_common.schemas import REQUEST_ID_EXAMPLE, UNAUTHORIZED, JobAcceptedResponse, JobStatusResponse, error
from ocr_common.security import verify_api_key
from src.api.v1.structuring import get_structuring_service
from src.core.pipeline import get_next_stage, get_pipeline
from src.schemas.structuring import StructuringJobRequest
from src.services.job_service import StructuringJobService

router = APIRouter(tags=["Pipeline"], dependencies=[Depends(verify_api_key)])


def get_job_service() -> StructuringJobService:
    return StructuringJobService(get_pipeline(), get_structuring_service(), get_next_stage())


@router.post(
    "/v1/structuring/jobs",
    status_code=202,
    response_model=JobAcceptedResponse,
    summary="Submit OCR output to the async pipeline (structuring stage)",
    description=(
        "Called by the OCR service. Records the job (`structuring.jobs`, idempotent per request_id), answers "
        "**202 immediately**, then in the background: turns the OCR blocks into named fields, stores the "
        "result (`structuring.results`), POSTs the stage callback to the orchestrator, and hands the job "
        "(guardrails + OCR + structuring results) to the scoring service.\n\n"
        "Sending the same request_id again does not run the work twice (`duplicate: true`), unless the "
        "previous attempt FAILED."
    ),
    responses={
        401: UNAUTHORIZED,
        422: error(422, "Validation Error", "body.ocr: Field required", errors="VALIDATION_ERROR"),
    },
)
async def submit_job(body: StructuringJobRequest, service: StructuringJobService = Depends(get_job_service)):
    data = await service.submit(body.request_id, body.document_type, body.guardrails, body.ocr.model_dump())
    return envelope(202, "Accepted", data, body.request_id)


@router.get(
    "/v1/structuring/jobs/{request_id}",
    response_model=JobStatusResponse,
    summary="Get the structuring stage status / result of a request_id",
    description=(
        "Status of this stage only (PROCESSING / DONE / FAILED) and its structured fields. For debugging and "
        "reconciliation; the orchestrator normally learns the status from the callback."
    ),
    responses={
        401: UNAUTHORIZED,
        404: error(
            404,
            "No structuring job for this request_id",
            f"No STRUCTURING job found for request_id: {REQUEST_ID_EXAMPLE}",
            request_id=REQUEST_ID_EXAMPLE,
        ),
        422: error(
            422,
            "Validation Error",
            "path.request_id: Field required",
            request_id=REQUEST_ID_EXAMPLE,
            errors="VALIDATION_ERROR",
        ),
    },
)
async def get_job(request_id: str, service: StructuringJobService = Depends(get_job_service)):
    try:
        data = await service.get(request_id)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    return envelope(200, "Success", data, request_id)
