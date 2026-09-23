from fastapi import APIRouter, Depends, Request

from ocr_common.pipeline import StagePipeline
from ocr_common.pipeline.outbox_status import (
    OUTBOX_RELEASE_DESCRIPTION,
    OUTBOX_RELEASE_SUMMARY,
    OUTBOX_STATUS_DESCRIPTION,
    OUTBOX_STATUS_SUMMARY,
    OutboxReleaseResponse,
    OutboxStatusResponse,
    outbox_release,
    outbox_release_responses,
    outbox_status,
    outbox_status_responses,
)
from ocr_common.web.envelope import envelope
from ocr_common.web.request_id import get_request_id
from ocr_common.web.schemas import REQUEST_ID_EXAMPLE, UNAUTHORIZED, JobAcceptedResponse, error, success_examples
from ocr_common.web.security import verify_api_key

from app.api.schemas import ScoringJobRequest, ScoringJobStatusResponse
from app.api.scoring import CONFIDENCE_PAYLOAD_EXAMPLE
from app.dependencies import get_job_service, get_pipeline
from app.services.job_service import ScoringJobService

router = APIRouter(tags=["Pipeline"], dependencies=[Depends(verify_api_key)])

_JOB = {"request_id": REQUEST_ID_EXAMPLE, "stage": "SCORING", "created_at": "2026-09-18T04:00:01+00:00"}


@router.post(
    "/v1/scoring/jobs",
    status_code=202,
    response_model=JobAcceptedResponse,
    operation_id="submitScoringJob",
    summary="Hand a structured document to the scoring stage (last)",
    description=(
        "**Step 4 of the pipeline, asynchronous, the last stage. Called by the structuring service, not by the "
        "orchestrator.**\n\n"
        "Records the job (`scoring_jobs`, idempotent per request_id), answers **202 immediately**, then in the "
        "background: builds the ML team's scoring payload from the chained guardrails + OCR + structuring results, "
        "runs the trust model, stores the result (`scoring_results`), and POSTs the `SCORING` callback, which "
        "carries the **final result** of the request.\n\n"
        "The outcome is two per-field confidences. There is no document-level score and no approve / reject "
        "decision: thresholds belong to the orchestrator."
    ),
    responses={
        202: success_examples(
            "The job was accepted (or already existed)",
            accepted=(
                "New job",
                envelope(
                    202,
                    "Accepted",
                    {"request_id": REQUEST_ID_EXAMPLE, "stage": "SCORING", "status": "PROCESSING", "duplicate": False},
                    REQUEST_ID_EXAMPLE,
                ),
            ),
            duplicate=(
                "Same request_id sent again: nothing is re-run",
                envelope(
                    202,
                    "Accepted",
                    {"request_id": REQUEST_ID_EXAMPLE, "stage": "SCORING", "status": "DONE", "duplicate": True},
                    REQUEST_ID_EXAMPLE,
                ),
            ),
        ),
        401: UNAUTHORIZED,
        422: error(422, "Validation Error", "body.request_id: Field required", errors="VALIDATION_ERROR"),
    },
)
async def submit_job(body: ScoringJobRequest, service: ScoringJobService = Depends(get_job_service)):
    guardrails = body.guardrails.model_dump(exclude_unset=True) if body.guardrails is not None else None
    ocr = body.ocr.model_dump() if body.ocr is not None else None
    structuring = body.structuring.model_dump() if body.structuring is not None else None
    data = await service.submit(body.request_id, body.document_type, guardrails, ocr, structuring)
    return envelope(202, "Accepted", data, body.request_id)


@router.get(
    "/v1/scoring/jobs/{request_id}",
    response_model=ScoringJobStatusResponse,
    operation_id="getScoringJob",
    summary="Status and result of the scoring stage",
    description=(
        "Status of this stage only and, once `DONE`, the two confidences plus the exact payload that was scored. "
        "The orchestrator normally receives the final result in the `SCORING` callback; use this to reconcile "
        "after a missed callback, or to audit a score."
    ),
    responses={
        200: success_examples(
            "The job exists",
            done=(
                "Finished",
                envelope(
                    200,
                    "Success",
                    {
                        **_JOB,
                        "status": "DONE",
                        "error_message": None,
                        "result": {
                            "npwp_confidence": 0.9806,
                            "name_confidence": 0.9948,
                            "payload": CONFIDENCE_PAYLOAD_EXAMPLE,
                        },
                        "updated_at": "2026-09-18T04:00:01+00:00",
                    },
                    REQUEST_ID_EXAMPLE,
                ),
            ),
            failed=(
                "Failed",
                envelope(
                    200,
                    "Success",
                    {
                        **_JOB,
                        "status": "FAILED",
                        "error_message": "Unsupported document_type: ktp. Supported: ['npwp']",
                        "result": None,
                        "updated_at": "2026-09-18T04:00:01+00:00",
                    },
                    REQUEST_ID_EXAMPLE,
                ),
            ),
        ),
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
    data = await service.get(request_id)
    return envelope(200, "Success", data, request_id)


@router.get(
    "/v1/scoring/outbox",
    response_model=OutboxStatusResponse,
    operation_id="getScoringOutboxStatus",
    summary=OUTBOX_STATUS_SUMMARY,
    description=OUTBOX_STATUS_DESCRIPTION,
    responses=outbox_status_responses("SCORING"),
)
async def get_outbox_status(request: Request, pipeline: StagePipeline = Depends(get_pipeline)):
    return envelope(200, "Success", await outbox_status(pipeline), get_request_id(request))


@router.post(
    "/v1/scoring/outbox/release",
    response_model=OutboxReleaseResponse,
    operation_id="releaseScoringOutbox",
    summary=OUTBOX_RELEASE_SUMMARY,
    description=OUTBOX_RELEASE_DESCRIPTION,
    responses=outbox_release_responses("SCORING"),
)
async def release_outbox(
    request: Request, request_id: str | None = None, pipeline: StagePipeline = Depends(get_pipeline)
):
    data = await outbox_release(pipeline, request_id)
    return envelope(200, "Success", data, request_id or get_request_id(request))
