import json
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile

from ocr_common.envelope import envelope
from ocr_common.errors import ServiceError
from ocr_common.intake import FileField, FileUrlField, resolve_intake
from ocr_common.schemas import REQUEST_ID_EXAMPLE, UNAUTHORIZED, JobAcceptedResponse, error, success_examples
from ocr_common.security import verify_api_key
from src.api.v1.ekstraksi import OCR_RESULT_EXAMPLE, get_ekstraksi_service
from src.core.config import get_settings
from src.core.pipeline import get_next_stage, get_pipeline
from src.schemas.ekstraksi import OcrJobStatusResponse
from src.services.job_service import EkstraksiJobService, Source

router = APIRouter(tags=["Pipeline"], dependencies=[Depends(verify_api_key)])

_JOB = {"request_id": REQUEST_ID_EXAMPLE, "stage": "OCR", "created_at": "2026-09-18T04:00:00+00:00"}


def get_job_service() -> EkstraksiJobService:
    return EkstraksiJobService(
        get_pipeline(), get_ekstraksi_service(), get_next_stage(), get_settings().max_upload_bytes
    )


def _parse_guardrails(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        value = None
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="guardrails must be a JSON object")
    return value


@router.post(
    "/v1/ekstraksi/jobs",
    status_code=202,
    response_model=JobAcceptedResponse,
    operation_id="submitOcrJob",
    summary="Start the pipeline for a document (OCR stage)",
    description=(
        "**Step 2 of the pipeline, asynchronous: the entry point of the OCR -> structuring -> scoring chain.** "
        "Call it after `POST /v1/guardrails/check` answered `passed: true`.\n\n"
        "Records the job (`ocr.jobs`, idempotent per request_id), answers **202 immediately**, then in the "
        "background: reads the document (`file`, or downloads `file_url`), runs OCR, stores the result "
        "(`ocr.results`), POSTs the `OCR` callback, and hands the job to the structuring service, which hands it "
        "to scoring. The caller does nothing more: it receives one callback per stage, and the final result in "
        "the `SCORING` callback (see *Webhooks*).\n\n"
        "**Document.** Send `file` (multipart) or `file_url`, exactly one. JPEG, PNG or PDF, at most 5 MB. "
        "`file_url` is downloaded in the background, so make a presigned URL live longer than the worst queueing "
        "time; an expired or unreachable URL becomes a `FAILED` job, not a `4xx`.\n\n"
        "**Idempotency.** The same request_id again answers `202` with `duplicate: true` and does not run OCR "
        "twice, unless the earlier attempt `FAILED`, in which case it is run again."
    ),
    responses={
        202: success_examples(
            "The job was accepted (or already existed)",
            accepted=(
                "New job",
                envelope(
                    202,
                    "Accepted",
                    {"request_id": REQUEST_ID_EXAMPLE, "stage": "OCR", "status": "PROCESSING", "duplicate": False},
                    REQUEST_ID_EXAMPLE,
                ),
            ),
            duplicate=(
                "Same request_id sent again: nothing is re-run, `status` is the existing job's status",
                envelope(
                    202,
                    "Accepted",
                    {"request_id": REQUEST_ID_EXAMPLE, "stage": "OCR", "status": "DONE", "duplicate": True},
                    REQUEST_ID_EXAMPLE,
                ),
            ),
        ),
        400: error(
            400,
            "Neither or both of file / file_url, or `guardrails` is not a JSON object",
            "Send exactly one of file or file_url",
        ),
        401: UNAUTHORIZED,
        422: error(422, "Validation Error", "body.request_id: Field required", errors="VALIDATION_ERROR"),
    },
)
async def submit_job(
    request_id: str = Form(..., description="request_id minted by the orchestrator", examples=[REQUEST_ID_EXAMPLE]),
    document_type: str = Form(
        "npwp", description="Document type chosen by the client. Only `npwp` is supported", examples=["npwp"]
    ),
    guardrails: str | None = Form(
        None,
        description=(
            "The `data` object of `POST /v1/guardrails/check`, serialised as a JSON string. Forwarded down the "
            "chain: scoring uses `document.confidence`, and the final result returns it unchanged. Optional, but "
            "without it the scoring model works with one input missing"
        ),
        examples=[
            '{"passed": true, "reason": null, "document": {"verdict": "accepted", "confidence": 0.9821, '
            '"n_pages": 1, "n_approve": 1, "n_reject": 0}, "pages": [{"page_index": 0, "proba_approve": 0.9821, '
            '"proba_reject": 0.0179, "verdict": "accepted"}]}'
        ],
    ),
    file: UploadFile | str | None = FileField,
    file_url: str | None = FileUrlField,
    service: EkstraksiJobService = Depends(get_job_service),
):
    upload, url = resolve_intake(file, file_url)
    source: Source
    if upload is not None:
        source = (await upload.read(), upload.filename or "", upload.content_type)
    else:
        assert url is not None
        source = url
    data = await service.submit(request_id, document_type, _parse_guardrails(guardrails), source)
    return envelope(202, "Accepted", data, request_id)


@router.get(
    "/v1/ekstraksi/jobs/{request_id}",
    response_model=OcrJobStatusResponse,
    operation_id="getOcrJob",
    summary="Status and result of the OCR stage",
    description=(
        "Status of this stage only, and its raw OCR result once `DONE`. The orchestrator normally learns the "
        "status from the callback; use this to reconcile after a missed callback, or to debug. The later stages "
        "have the same endpoint on their own service (`/v1/structuring/jobs/{request_id}`, "
        "`/v1/scoring/jobs/{request_id}`)."
    ),
    responses={
        200: success_examples(
            "The job exists",
            processing=(
                "Still running",
                envelope(
                    200,
                    "Success",
                    {
                        **_JOB,
                        "status": "PROCESSING",
                        "error_message": None,
                        "result": None,
                        "updated_at": "2026-09-18T04:00:00+00:00",
                    },
                    REQUEST_ID_EXAMPLE,
                ),
            ),
            done=(
                "Finished",
                envelope(
                    200,
                    "Success",
                    {
                        **_JOB,
                        "status": "DONE",
                        "error_message": None,
                        "result": OCR_RESULT_EXAMPLE,
                        "updated_at": "2026-09-18T04:00:01+00:00",
                    },
                    REQUEST_ID_EXAMPLE,
                ),
            ),
            failed=(
                "Failed: same `error_message` as the FAILED callback. Resubmitting the request_id runs it again",
                envelope(
                    200,
                    "Success",
                    {
                        **_JOB,
                        "status": "FAILED",
                        "error_message": "ekstraksi OCR model is unavailable",
                        "result": None,
                        "updated_at": "2026-09-18T04:00:03+00:00",
                    },
                    REQUEST_ID_EXAMPLE,
                ),
            ),
        ),
        401: UNAUTHORIZED,
        404: error(
            404,
            "No OCR job for this request_id",
            f"No OCR job found for request_id: {REQUEST_ID_EXAMPLE}",
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
async def get_job(request_id: str, service: EkstraksiJobService = Depends(get_job_service)):
    try:
        data = await service.get(request_id)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    return envelope(200, "Success", data, request_id)
