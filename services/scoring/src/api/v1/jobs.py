"""
API (presentation) layer tahap SCORING di pipeline async. Dipanggil
ServiceStructuring; menjawab 202 segera, hasil akhirnya dikirim ke Orkestrasi
lewat callback. Tidak ada aturan bisnis di sini; lihat src/services/job_service.py.
"""

from fastapi import APIRouter, Depends, HTTPException

from ocr_common.envelope import envelope
from ocr_common.errors import ServiceError
from ocr_common.schemas import REQUEST_ID_EXAMPLE, UNAUTHORIZED, JobAcceptedResponse, JobStatusResponse, error
from ocr_common.security import verify_api_key
from src.api.v1.scoring import get_scoring_service
from src.core.pipeline import get_pipeline
from src.schemas.scoring import ScoringJobRequest
from src.services.job_service import ScoringJobService

router = APIRouter(tags=["Pipeline"], dependencies=[Depends(verify_api_key)])


def get_job_service() -> ScoringJobService:
    return ScoringJobService(get_pipeline(), get_scoring_service())


@router.post(
    "/v1/scoring/jobs",
    status_code=202,
    response_model=JobAcceptedResponse,
    summary="Submit a structured document to the async pipeline (scoring stage, last)",
    description=(
        "Called by the structuring service. Records the job (`scoring.jobs`, idempotent per request_id), "
        "answers **202 immediately**, then in the background: scores the document, stores the result "
        "(`scoring.results`), and POSTs the stage callback to the orchestrator. As the last stage, its "
        "callback carries the **final result** (`result`: document_type, fields, scoring, guardrails).\n\n"
        "Sending the same request_id again does not run the work twice (`duplicate: true`), unless the "
        "previous attempt FAILED."
    ),
    responses={
        401: UNAUTHORIZED,
        422: error(422, "Validation Error", "body.structuring: Field required", errors="VALIDATION_ERROR"),
    },
)
async def submit_job(body: ScoringJobRequest, service: ScoringJobService = Depends(get_job_service)):
    data = await service.submit(body.request_id, body.document_type, body.guardrails, body.structuring.model_dump())
    return envelope(202, "Accepted", data, body.request_id)


@router.get(
    "/v1/scoring/jobs/{request_id}",
    response_model=JobStatusResponse,
    summary="Get the scoring stage status / result of a request_id",
    description=(
        "Status of this stage only (PROCESSING / DONE / FAILED) and its score report. For debugging and "
        "reconciliation; the orchestrator normally learns the status from the callback."
    ),
    responses={
        401: UNAUTHORIZED,
        404: error(
            404,
            "No scoring job for this request_id",
            f"No SCORING job found for request_id: {REQUEST_ID_EXAMPLE}",
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
async def get_job(request_id: str, service: ScoringJobService = Depends(get_job_service)):
    try:
        data = await service.get(request_id)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    return envelope(200, "Success", data, request_id)
