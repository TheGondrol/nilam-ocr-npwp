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

from app.api.schemas import StructuringJobRequest, StructuringJobStatusResponse
from app.api.structuring import STRUCTURED_EXAMPLE
from app.dependencies import get_job_service, get_pipeline
from app.services.job_service import StructuringJobService

router = APIRouter(tags=["Pipeline"], dependencies=[Depends(verify_api_key)])

_JOB = {"request_id": REQUEST_ID_EXAMPLE, "stage": "STRUCTURING", "created_at": "2026-09-18T04:00:01+00:00"}


@router.post(
    "/v1/structuring/jobs",
    status_code=202,
    response_model=JobAcceptedResponse,
    operation_id="submitStructuringJob",
    summary="Hand OCR output to the structuring stage",
    description=(
        "**Step 3 of the pipeline, asynchronous. Called by the OCR service, not by the orchestrator.**\n\n"
        "Records the job (`structuring_jobs`, idempotent per request_id), answers **202 immediately**, then in the "
        "background: turns the OCR lines into named fields, stores the result (`structuring_results`), POSTs the "
        "`STRUCTURING` callback, and hands the job (guardrails + OCR + structuring results) to the scoring service.\n\n"
        "A document that is not a lone NPWP card fails the job with a reason: an upload that also contains a "
        "KTP / KK / marriage certificate, a CAPTCHA page, a screenshot of the DJP NPWP lookup, more than 2 pages, "
        "or no text at all."
    ),
    responses={
        202: success_examples(
            "The job was accepted (or already existed)",
            accepted=(
                "New job",
                envelope(
                    202,
                    "Accepted",
                    {
                        "request_id": REQUEST_ID_EXAMPLE,
                        "stage": "STRUCTURING",
                        "status": "PROCESSING",
                        "duplicate": False,
                    },
                    REQUEST_ID_EXAMPLE,
                ),
            ),
            duplicate=(
                "Same request_id sent again: nothing is re-run",
                envelope(
                    202,
                    "Accepted",
                    {"request_id": REQUEST_ID_EXAMPLE, "stage": "STRUCTURING", "status": "DONE", "duplicate": True},
                    REQUEST_ID_EXAMPLE,
                ),
            ),
        ),
        401: UNAUTHORIZED,
        422: error(422, "Validation Error", "body.request_id: Field required", errors="VALIDATION_ERROR"),
    },
)
async def submit_job(body: StructuringJobRequest, service: StructuringJobService = Depends(get_job_service)):
    guardrails = body.guardrails.model_dump(exclude_unset=True) if body.guardrails is not None else None
    ocr = body.ocr.model_dump(exclude_unset=True) if body.ocr is not None else None
    data = await service.submit(body.request_id, body.document_type, guardrails, ocr)
    return envelope(202, "Accepted", data, body.request_id)


@router.get(
    "/v1/structuring/jobs/{request_id}",
    response_model=StructuringJobStatusResponse,
    operation_id="getStructuringJob",
    summary="Status and result of the structuring stage",
    description=(
        "Status of this stage only, and its structured fields once `DONE`. The orchestrator normally learns the "
        "status from the callback; use this to reconcile after a missed callback, or to debug."
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
                        "result": STRUCTURED_EXAMPLE,
                        "updated_at": "2026-09-18T04:00:01+00:00",
                    },
                    REQUEST_ID_EXAMPLE,
                ),
            ),
            failed=(
                "Failed: the upload was not a lone NPWP card",
                envelope(
                    200,
                    "Success",
                    {
                        **_JOB,
                        "status": "FAILED",
                        "error_message": (
                            "Upload contains another document (KARTU TANDA PENDUDUK); send the NPWP card only"
                        ),
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
    data = await service.get(request_id)
    return envelope(200, "Success", data, request_id)


@router.get(
    "/v1/structuring/outbox",
    response_model=OutboxStatusResponse,
    operation_id="getStructuringOutboxStatus",
    summary=OUTBOX_STATUS_SUMMARY,
    description=OUTBOX_STATUS_DESCRIPTION,
    responses=outbox_status_responses("STRUCTURING"),
)
async def get_outbox_status(request: Request, pipeline: StagePipeline = Depends(get_pipeline)):
    return envelope(200, "Success", await outbox_status(pipeline), get_request_id(request))


@router.post(
    "/v1/structuring/outbox/release",
    response_model=OutboxReleaseResponse,
    operation_id="releaseStructuringOutbox",
    summary=OUTBOX_RELEASE_SUMMARY,
    description=OUTBOX_RELEASE_DESCRIPTION,
    responses=outbox_release_responses("STRUCTURING"),
)
async def release_outbox(
    request: Request, request_id: str | None = None, pipeline: StagePipeline = Depends(get_pipeline)
):
    data = await outbox_release(pipeline, request_id)
    return envelope(200, "Success", data, request_id or get_request_id(request))
