import json
import time
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, UploadFile

from ocr_common.envelope import envelope
from ocr_common.errors import ServiceError
from ocr_common.intake import FileField, FileUrlField, read_image
from ocr_common.npwp import DOCUMENT_TYPE
from ocr_common.request_id import get_request_id
from ocr_common.schemas import UNAUTHORIZED, error, success_examples
from ocr_common.security import verify_api_key
from src.api.v1.extract_contract import (
    COMPLETED_MESSAGE,
    PROCESSING_MESSAGE,
    REJECTED_CODE,
    extract_body,
    extract_response,
)
from src.clients.ekstraksi import EkstraksiJobClient, get_ekstraksi_client
from src.clients.stages import get_stage_status_clients
from src.core.config import Settings, get_settings
from src.models.guardrails import get_page_classifier
from src.schemas.guardrails import ExtractOcrResponse, GuardrailReportResponse
from src.services.guardrails_service import GuardrailsService
from src.services.job_service import GuardrailsJobService
from src.services.pipeline_waiter import PipelineWaiter

router = APIRouter(tags=["Guardrails"], dependencies=[Depends(verify_api_key)])

RID = "OCR_9cb01af2-493d-446d-b191-af120333f6d0"
INVALID_PARAMS_MESSAGE = "params must be valid JSON: an object, or a quoted string"

_PARAMS = {"nik": "3123456711950001", "refno": "PK19039Y8U"}
_COMPLETED = extract_body(
    200,
    COMPLETED_MESSAGE,
    data={
        "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 1},
        "nama": {"value": "BUDI SANTOSO", "confidence": 1},
    },
    job_status="completed",
    guardrails=1,
    request_id=RID,
    document_type="npwp",
    params=_PARAMS,
)
_PROCESSING = extract_body(
    202, PROCESSING_MESSAGE, job_status="processing", request_id=RID, document_type="npwp", params=_PARAMS
)
_REJECTED = extract_body(
    400,
    "Document rejected by guardrails: 1/1 page(s) rejected (confidence 0.88)",
    errors=REJECTED_CODE,
    job_status="failed",
    guardrails=0,
    request_id=RID,
    document_type="npwp",
    params=_PARAMS,
)
_FAILED = extract_body(
    422,
    "No text lines to structure",
    errors="STRUCTURING_FAILED",
    job_status="failed",
    guardrails=1,
    request_id=RID,
    document_type="npwp",
    params=_PARAMS,
)

_ACCEPTED_PAGE = {"page_index": 0, "proba_approve": 0.9821, "proba_reject": 0.0179, "verdict": "accepted"}
_REJECTED_PAGE = {"page_index": 0, "proba_approve": 0.1179, "proba_reject": 0.8821, "verdict": "reject"}
_ACCEPTED_REPORT = {
    "passed": True,
    "reason": None,
    "document": {"verdict": "accepted", "confidence": 0.9821, "n_pages": 1, "n_approve": 1, "n_reject": 0},
    "pages": [_ACCEPTED_PAGE],
}
_REJECTED_REPORT = {
    "passed": False,
    "reason": "Document rejected by guardrails: 1/1 page(s) rejected (confidence 0.88)",
    "document": {"verdict": "reject", "confidence": 0.8821, "n_pages": 1, "n_approve": 0, "n_reject": 1},
    "pages": [_REJECTED_PAGE],
}


class _InvalidParams(Exception):
    pass


def _parse_params(raw: str | None) -> Any:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise _InvalidParams from exc
    if not isinstance(value, dict | str):
        raise _InvalidParams
    return value


def get_guardrails_service() -> GuardrailsService:
    return GuardrailsService(get_page_classifier(), get_settings())


@lru_cache
def get_pipeline_waiter() -> PipelineWaiter:
    return PipelineWaiter(get_stage_status_clients(), poll_interval=get_settings().pipeline_poll_interval_seconds)


def get_job_service(
    guardrails: GuardrailsService = Depends(get_guardrails_service),
    ekstraksi: EkstraksiJobClient = Depends(get_ekstraksi_client),
    waiter: PipelineWaiter = Depends(get_pipeline_waiter),
    settings: Settings = Depends(get_settings),
) -> GuardrailsJobService:
    return GuardrailsJobService(guardrails, ekstraksi, waiter, wait_seconds=settings.pipeline_wait_seconds)


@router.post(
    "/v1/extract-ocr",
    response_model=ExtractOcrResponse,
    operation_id="extractOcr",
    summary="Judge a document, run the pipeline, answer with the OCR result or 202",
    description=(
        "**The only call the orchestrator makes.** Judges the document with the guardrails model, hands it to "
        "the OCR stage when it passes, then waits for OCR -> structuring -> scoring for up to "
        "`PIPELINE_WAIT_SECONDS` (15 s by default), counted from when this request arrived. The response follows "
        "the orchestrator's `extract-ocr` contract:\n\n"
        "| Outcome | HTTP | `job_status` | `data` | `guardrails` | `errors` |\n"
        "|---|---|---|---|---|---|\n"
        "| Finished in time | 200 | `completed` | the fields | `1` | null |\n"
        "| Still running | 202 | `processing` | null | null | null |\n"
        f"| Rejected by the guardrails model | 400 | `failed` | null | `0` | `{REJECTED_CODE}` |\n"
        "| A stage failed in time | 422 | `failed` | null | `1` | `OCR_FAILED`, `STRUCTURING_FAILED` or "
        "`SCORING_FAILED` |\n\n"
        "`data` holds `nomor_npwp` and `nama` as `{value, confidence}`; `nama` is the taxpayer's name, or the "
        "registered name on a company's card. `confidence` is `1` when the ML team's trust model gives the value a "
        "probability of being correct of at least `FIELD_CONFIDENCE_THRESHOLD` (0.5 by default), else `0`. "
        "`params` is returned as sent.\n\n"
        "After a hand-off the `OCR`, `STRUCTURING` and `SCORING` callbacks are sent in every case; on 202 the "
        "result arrives in the SCORING callback. Give this call an HTTP timeout well above "
        "`PIPELINE_WAIT_SECONDS` (e.g. +15 s) to cover a slow guardrails check or hand-off.\n\n"
        "Send `request_id` plus the document as `file`, or as `file_url` (downloaded here once, then forwarded "
        "as a file, so the URL only has to live for this call); exactly one of the two. The raw guardrails report "
        "(per-page probabilities) is not part of this response: it travels down the pipeline, comes back in the "
        "SCORING callback, and `POST /v1/guardrails/check` returns it on its own.\n\n"
        "**Idempotency.** The same request_id again re-runs the guardrails check, but the pipeline does not run "
        "twice unless the earlier attempt `FAILED` or outlived the job lease (`PIPELINE_JOB_LEASE_SECONDS`); a "
        "request_id that already finished answers 200 with its stored result."
    ),
    responses={
        200: success_examples(
            "Accepted and finished within the wait",
            completed=("The OCR result", _COMPLETED),
        ),
        202: {
            **success_examples(
                "Accepted, but still running when the wait ran out: the result follows in the SCORING callback",
                processing=("Still processing", _PROCESSING),
            ),
            "model": ExtractOcrResponse,
        },
        400: {
            "model": ExtractOcrResponse,
            "description": (
                f"Rejected by the guardrails model (`{REJECTED_CODE}`, `guardrails: 0`), unsupported "
                "`document_type` (`UNSUPPORTED_DOCUMENT_TYPE`), or a bad file / intake (empty, too large, "
                "unsupported type, unreadable, `file_url` refused)"
            ),
            "content": {"application/json": {"example": _REJECTED}},
        },
        401: UNAUTHORIZED,
        422: {
            "model": ExtractOcrResponse,
            "description": (
                "A pipeline stage failed within the wait (`OCR_FAILED`, `STRUCTURING_FAILED`, `SCORING_FAILED`; "
                "`message` says why), `params` is not valid JSON (`INVALID_PARAMS`), or a required field is "
                "missing (`VALIDATION_ERROR`)"
            ),
            "content": {"application/json": {"example": _FAILED}},
        },
        500: error(
            500,
            "The guardrails model or the ekstraksi service failed or answered in an unexpected shape",
            "guardrails model returned an unexpected response",
        ),
        503: error(
            503,
            "The ekstraksi service (or the remote guardrails model) is unreachable; nothing was started",
            "ekstraksi service is unavailable",
        ),
        504: error(
            504,
            "The ekstraksi service (or the remote guardrails model) did not answer in time",
            "ekstraksi service timed out after 10.0s",
        ),
    },
)
async def extract_ocr(
    request: Request,
    response: Response,
    request_id: str = Form(..., description="request_id minted by the orchestrator", examples=[RID]),
    document_type: str = Form(
        DOCUMENT_TYPE, description="Document type chosen by the client. Only `npwp` is supported", examples=["npwp"]
    ),
    params: str | None = Form(
        None,
        description=(
            "Client metadata as JSON: an object, or a quoted string. Not interpreted; returned unchanged in " "`params`"
        ),
        examples=['{"nik": "3123456711950001", "refno": "PK19039Y8U"}'],
    ),
    file: UploadFile | str | None = FileField,
    file_url: str | None = FileUrlField,
    service: GuardrailsJobService = Depends(get_job_service),
    settings: Settings = Depends(get_settings),
):
    received_at = time.monotonic()
    try:
        parsed_params = _parse_params(params)
    except _InvalidParams:
        response.status_code = 422
        return extract_body(
            422,
            INVALID_PARAMS_MESSAGE,
            errors="INVALID_PARAMS",
            request_id=request_id,
            document_type=document_type,
            params=None,
        )
    if document_type != DOCUMENT_TYPE:
        response.status_code = 400
        return extract_body(
            400,
            f"Unsupported document_type: {document_type}. Supported: {DOCUMENT_TYPE}",
            errors="UNSUPPORTED_DOCUMENT_TYPE",
            request_id=request_id,
            document_type=document_type,
            params=parsed_params,
        )

    content, filename, content_type = await read_image(request, file, file_url)
    try:
        outcome = await service.submit(
            request_id, document_type, filename, content_type, content, received_at=received_at
        )
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    status_code, body = extract_response(
        outcome,
        request_id=request_id,
        document_type=document_type,
        params=parsed_params,
        threshold=settings.field_confidence_threshold,
    )
    response.status_code = status_code
    return body


@router.post(
    "/v1/guardrails/check",
    response_model=GuardrailReportResponse,
    operation_id="checkDocument",
    summary="Judge a document only (internal)",
    description=(
        "Runs the guardrails model on each page (a PDF is rendered page by page; an image is one page) and "
        "aggregates a document verdict, **without** starting anything: no OCR job, no callback. The model is "
        "either in this process (`efficientnet`) or the ML team's model service (`remote`, POST "
        "/v1/predict/json); the report is the same either way. Used by the legacy synchronous `extract-ocr` on "
        "the ekstraksi service, and for debugging. Always 200 when the document was judged: read `data.passed`."
    ),
    responses={
        200: success_examples(
            "The document was judged",
            accepted=("Accepted", envelope(200, "OK", _ACCEPTED_REPORT, RID)),
            rejected=("Rejected", envelope(200, "OK", _REJECTED_REPORT, RID)),
        ),
        400: error(
            400, "Bad file (empty, too large, unsupported type, unreadable) or bad intake", "Uploaded file is empty"
        ),
        401: UNAUTHORIZED,
        422: error(422, "Validation Error", "body.file: Field required", errors="VALIDATION_ERROR"),
        500: error(500, "The guardrails model failed", "guardrails model returned an unexpected response"),
    },
)
async def check_document(
    request: Request,
    request_id: str | None = Form(None, description="Echoed in the response; optional", examples=[RID]),
    file: UploadFile | str | None = FileField,
    file_url: str | None = FileUrlField,
    service: GuardrailsService = Depends(get_guardrails_service),
):
    content, filename, content_type = await read_image(request, file, file_url)
    try:
        report = await service.check(filename, content_type, content)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    return envelope(200, "OK", report, request_id or get_request_id(request))
