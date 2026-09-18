"""
API (presentation) layer tahap OCR di pipeline async. Dipanggil Orkestrasi
setelah guardrails lolos; menjawab 202 segera, hasilnya dikabarkan lewat
callback. Tidak ada aturan bisnis di sini; lihat src/services/job_service.py.
"""

import json
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile

from ocr_common.envelope import envelope
from ocr_common.errors import ServiceError
from ocr_common.intake import FileField, FileUrlField, resolve_intake
from ocr_common.schemas import REQUEST_ID_EXAMPLE, UNAUTHORIZED, JobAcceptedResponse, JobStatusResponse, error
from ocr_common.security import verify_api_key
from src.api.v1.ekstraksi import get_ekstraksi_service
from src.core.config import get_settings
from src.core.pipeline import get_next_stage, get_pipeline
from src.services.job_service import EkstraksiJobService, Source

router = APIRouter(tags=["Pipeline"], dependencies=[Depends(verify_api_key)])


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
    summary="Submit a document to the async pipeline (OCR stage)",
    description=(
        "Called by the orchestrator after guardrails passed. Records the job "
        "(`ocr.jobs`, idempotent per request_id), answers **202 immediately**, then in the background: "
        "reads the document (`file`, or downloads `file_url` from MinIO), runs OCR, stores the result "
        "(`ocr.results`), POSTs the stage callback to the orchestrator, and hands the job to the "
        "structuring service.\n\n"
        "Sending the same request_id again does not run OCR twice (`duplicate: true`), unless the "
        "previous attempt FAILED.\n\n"
        "Send the document as `file` or `file_url`; exactly one of the two."
    ),
    responses={
        400: error(400, "Bad intake or malformed guardrails JSON", "Send exactly one of file or file_url"),
        401: UNAUTHORIZED,
        422: error(422, "Validation Error", "body.request_id: Field required", errors="VALIDATION_ERROR"),
    },
)
async def submit_job(
    request_id: str = Form(..., description="request_id minted by the orchestrator", examples=[REQUEST_ID_EXAMPLE]),
    document_type: str = Form("npwp", description="Document type chosen by the client", examples=["npwp"]),
    guardrails: str | None = Form(
        None,
        description="Guardrails result (`data` of /v1/guardrails/check) as a JSON object; forwarded down the chain",
        examples=['{"passed": true, "reason": null}'],
    ),
    file: UploadFile | str | None = FileField,
    file_url: str | None = FileUrlField,
    service: EkstraksiJobService = Depends(get_job_service),
):
    upload, url = resolve_intake(file, file_url)
    # Upload harus dibaca SEKARANG: berkasnya ditutup begitu response terkirim.
    # file_url sebaliknya diunduh di background, supaya 202 tidak menunggu MinIO.
    source: Source
    if upload is not None:
        source = (await upload.read(), upload.filename or "", upload.content_type)
    else:
        assert url is not None  # dijamin resolve_intake: tepat satu dari keduanya
        source = url
    data = await service.submit(request_id, document_type, _parse_guardrails(guardrails), source)
    return envelope(202, "Accepted", data, request_id)


@router.get(
    "/v1/ekstraksi/jobs/{request_id}",
    response_model=JobStatusResponse,
    summary="Get the OCR stage status / result of a request_id",
    description=(
        "Status of this stage only (PROCESSING / DONE / FAILED) and its raw OCR result. For debugging and "
        "reconciliation; the orchestrator normally learns the status from the callback."
    ),
    responses={
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
