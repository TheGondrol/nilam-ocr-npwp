import json
import time
from typing import Any

from fastapi import APIRouter, Depends, Form, Request, Response, UploadFile

from ocr_common.image_validation import PAYLOAD_TOO_LARGE_MESSAGE
from ocr_common.npwp import DOCUMENT_TYPE
from ocr_common.web.intake import FileField, FileUrlField, read_image
from ocr_common.web.request_id import bind_request_id, reset_request_id
from ocr_common.web.schemas import UNAUTHORIZED, error, success_examples
from ocr_common.web.security import verify_api_key

from app.api.extract_contract import (
    COMPLETED_MESSAGE,
    GUARDRAILS_REJECTED_MESSAGE,
    PASSED,
    PROCESSING_MESSAGE,
    REJECTED,
    extract_body,
    extract_response,
)
from app.api.schemas import ExtractOcrResponse
from app.config import Settings, get_settings
from app.dependencies import get_extract_service
from app.services.extract_service import ExtractOcrService

router = APIRouter(tags=["Extract OCR"], dependencies=[Depends(verify_api_key)])

RID = "OCR_9cb01af2-493d-446d-b191-af120333f6d0"
INVALID_PARAMS_MESSAGE = "params must be valid JSON: an object, or a quoted string"
TOO_MANY_PAGES_MESSAGE = "Jumlah halaman melebihi batas, pastikan hanya mengunggah dokumen NPWP"

_PARAMS = {"nik": "3123456711950001", "refno": "PK19039Y8U"}
_DATA = {
    "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 1},
    "nama": {"value": "BUDI SANTOSO", "confidence": 1},
}
_COMPLETED = extract_body(
    200,
    COMPLETED_MESSAGE,
    data=_DATA,
    job_status="completed",
    guardrails=PASSED,
    errors=None,
    request_id=RID,
    document_type="npwp",
    params=_PARAMS,
)
_PROCESSING = extract_body(
    202,
    PROCESSING_MESSAGE,
    job_status="processing",
    request_id=RID,
    document_type="npwp",
    params=_PARAMS,
)
_REJECTED = extract_body(
    200,
    GUARDRAILS_REJECTED_MESSAGE,
    job_status="failed",
    guardrails=REJECTED,
    request_id=RID,
    document_type="npwp",
    params=_PARAMS,
)
_RULES_REJECTED = extract_body(
    200,
    "Kode provinsi pada NPWP tidak valid, mohon dicek kembali",
    job_status="failed",
    guardrails=REJECTED,
    request_id=RID,
    document_type="npwp",
    params=_PARAMS,
)
_UNSUPPORTED = extract_body(
    400,
    "Unsupported document_type: ktp. Supported: npwp",
    errors="UNSUPPORTED_DOCUMENT_TYPE",
    request_id=RID,
    document_type="ktp",
)
_FAILED = extract_body(
    422,
    "No text lines to structure",
    errors="STRUCTURING_FAILED",
    job_status="failed",
    guardrails=PASSED,
    request_id=RID,
    document_type="npwp",
    params=_PARAMS,
)

_CONTRACT_TABLE = (
    "| Outcome | HTTP | `job_status` | `data` | `guardrails` | `errors` |\n"
    "|---|---|---|---|---|---|\n"
    "| Finished | 200 | `completed` | the fields | `0` | null |\n"
    "| Still running | 202 | `processing` | null | null | null |\n"
    "| Rejected by the guardrails model | 200 | `failed` | null | `1` | null "
    f"(`message`: `{GUARDRAILS_REJECTED_MESSAGE}`) |\n"
    "| Rejected by the structuring rules | 200 | `failed` | null | `1` | null (`message`: the rules' reason) |\n"
    "| A stage failed | 422 | `failed` | null | `0` | `OCR_FAILED`, `STRUCTURING_FAILED` or "
    "`SCORING_FAILED` |\n\n"
    "`guardrails` is `1` when the document was rejected (by the guardrails model or by the structuring rules) "
    "and `0` when it passed. A rejection is an answer, not an error: `200` with `job_status: failed`; read "
    "`guardrails`, not the HTTP status, to tell it from a result.\n\n"
)


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


@router.post(
    "/v1/extract-ocr",
    response_model=ExtractOcrResponse,
    operation_id="extractOcr",
    summary="Judge a document, run the pipeline, answer with the OCR result or 202",
    description=(
        "**The call the central orchestrator makes.** Checks the file, has the guardrails service judge it, "
        "hands it to the OCR stage when it passes, then waits for OCR -> structuring -> scoring for up to "
        "`PIPELINE_WAIT_SECONDS` (15 s by default), counted from when this request arrived. The response follows "
        'the central orchestrator\'s `extract-ocr` contract ("Finished" meaning finished within the wait):\n\n'
        + _CONTRACT_TABLE
        + "`data` holds `nomor_npwp` and `nama` as `{value, confidence}`; `nama` is the taxpayer's name, or the "
        "registered name on a company's card. `confidence` is `1` when the ML team's trust model gives the value a "
        "probability of being correct of at least `FIELD_CONFIDENCE_THRESHOLD` (0.5 by default), else `0`. "
        "`params` is returned as sent.\n\n"
        "**Rejected by the structuring rules**: the ML team's rules reject a document that is blurred or blank, "
        "not in the standard NPWP format, another document or bundled with one, a screenshot of the online NPWP "
        "lookup, longer than the page limit, or whose number carries an invalid birthdate, province, kecamatan "
        "or KPP code. `message` is the rules' Indonesian reason, e.g. `Kode provinsi pada NPWP tidak valid, mohon "
        "dicek kembali`. A single-word name or a letter in the number is tolerated: the fields are returned, and "
        "the trust model's confidence already accounts for it.\n\n"
        "**Refused before anything runs** (plain error envelope, no `job_status`): a document above "
        "`MAX_UPLOAD_BYTES` (2.5 MB by default) answers `413`, one with more than `GUARDRAILS_MAX_DOCUMENT_PAGES` "
        "(2) pages answers `400`, both with an Indonesian `message` the client can show as is.\n\n"
        "On 202 the result arrives by callback (sent by the pipeline stages), and can be read with "
        "`GET /v1/extract-ocr/{request_id}`. Give this call an HTTP timeout well above `PIPELINE_WAIT_SECONDS` "
        "(e.g. +15 s) to cover a slow guardrails check or hand-off.\n\n"
        "Send `request_id` plus the document as `file`, or as `file_url`; exactly one of the two. A `file_url` "
        "is downloaded here for the guardrails check, and the URL itself (not the bytes) is handed to the OCR "
        "stage, which downloads it again, also when it re-runs a job left behind by a dead process: the URL "
        "must stay valid for longer than the job lease. The raw guardrails report (per-page probabilities) is "
        "not part of this response: it travels down the pipeline and comes back in the SCORING callback.\n\n"
        "**Idempotency.** The same request_id again re-runs the guardrails check, but the pipeline does not run "
        "twice unless the earlier OCR attempt `FAILED` or outlived the job lease (`PIPELINE_JOB_LEASE_SECONDS`); "
        "a request_id that already finished answers again with its stored outcome."
    ),
    responses={
        200: success_examples(
            "Finished within the wait, or rejected (`guardrails: 1`)",
            completed=("The OCR result", _COMPLETED),
            rejected_by_guardrails=("Rejected by the guardrails model", _REJECTED),
            rejected_by_rules=("Rejected by the structuring rules", _RULES_REJECTED),
        ),
        202: {
            **success_examples(
                "Accepted, but still running when the wait ran out: the result follows by callback",
                processing=("Still processing", _PROCESSING),
            ),
            "model": ExtractOcrResponse,
        },
        400: {
            "model": ExtractOcrResponse,
            "description": (
                "The request cannot be processed: unsupported `document_type` (`UNSUPPORTED_DOCUMENT_TYPE`), more "
                f"than `GUARDRAILS_MAX_DOCUMENT_PAGES` pages (`{TOO_MANY_PAGES_MESSAGE}`, plain error envelope), or a "
                "bad file / intake (empty, unsupported type, unreadable, `file_url` refused). A rejected document is "
                "a 200, not a 400"
            ),
            "content": {"application/json": {"example": _UNSUPPORTED}},
        },
        401: UNAUTHORIZED,
        413: error(
            413,
            "The document exceeds `MAX_UPLOAD_BYTES` (2.5 MB by default); nothing was started",
            PAYLOAD_TOO_LARGE_MESSAGE.format(limit="2,5 MB"),
        ),
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
            "The guardrails or ekstraksi service failed, or answered in an unexpected shape",
            "guardrails service returned an unexpected response",
        ),
        503: error(
            503,
            "The guardrails service, its model, or the ekstraksi service is unreachable; nothing was started",
            "ekstraksi service is unavailable",
        ),
        504: error(
            504,
            "The guardrails service, its model, or the ekstraksi service did not answer in time",
            "ekstraksi service timed out after 10.0s",
        ),
    },
)
async def extract_ocr(
    request: Request,
    response: Response,
    request_id: str = Form(..., description="request_id minted by the central orchestrator", examples=[RID]),
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
    service: ExtractOcrService = Depends(get_extract_service),
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
        )
    if document_type != DOCUMENT_TYPE:
        response.status_code = 400
        return extract_body(
            400,
            f"Unsupported document_type: {document_type}. Supported: {DOCUMENT_TYPE}",
            errors="UNSUPPORTED_DOCUMENT_TYPE",
            request_id=request_id,
            document_type=document_type,
        )

    # The central orchestrator's request_id becomes the X-Request-ID of every call below and of our log
    # lines, so one id follows the request through guardrails and every stage.
    token = bind_request_id(request_id)
    try:
        content, filename, content_type = await read_image(request, file, file_url)
        outcome = await service.submit(
            request_id, document_type, filename, content_type, content, received_at=received_at, file_url=file_url
        )
    finally:
        reset_request_id(token)
    status_code, body = extract_response(
        outcome,
        request_id=request_id,
        document_type=document_type,
        params=parsed_params,
        threshold=settings.field_confidence_threshold,
    )
    response.status_code = status_code
    return body


@router.get(
    "/v1/extract-ocr/{request_id}",
    response_model=ExtractOcrResponse,
    operation_id="getExtractOcr",
    summary="Where a request is now: the extract-ocr answer, without waiting",
    description=(
        "Reads the OCR, structuring and scoring jobs of `request_id` once each, in pipeline order, and answers in "
        'the same contract as `POST /v1/extract-ocr` ("Finished" meaning finished by now):\n\n'
        + _CONTRACT_TABLE
        + "Use it for a request that was answered `202`, e.g. when a callback did not arrive. `params` is always "
        "null here (it is not stored) and `document_type` is `npwp`.\n\n"
        "**404** means no stage has a job for this request_id: it was refused or rejected by the guardrails model "
        "(the POST's answer, `guardrails: 1`, is final: nothing is stored for it), refused before the check, or its "
        "POST is still being judged by guardrails.\n\n"
        "**Limitation.** A hand-off between two stages that failed for good (its retries ran out, or it became a "
        "dead letter in the outbox) leaves the next stage without a job, so this endpoint keeps answering `202` "
        "for it. The `FAILED` callback and the central orchestrator's own tables carry that final state; this "
        "endpoint only reads the stages' jobs."
    ),
    responses={
        200: success_examples(
            "Finished, or rejected by the structuring rules (`guardrails: 1`)",
            completed=("The OCR result", {**_COMPLETED, "params": None}),
            rejected_by_rules=("Rejected by the structuring rules", {**_RULES_REJECTED, "params": None}),
        ),
        202: {
            **success_examples(
                "Still running",
                processing=("Still processing", {**_PROCESSING, "params": None}),
            ),
            "model": ExtractOcrResponse,
        },
        401: UNAUTHORIZED,
        404: error(
            404,
            "No stage has a job for this request_id (rejected by the guardrails model, or not submitted yet)",
            f"No request found for request_id {RID}",
        ),
        422: {
            "model": ExtractOcrResponse,
            "description": "A pipeline stage failed (`OCR_FAILED`, `STRUCTURING_FAILED`, `SCORING_FAILED`)",
            "content": {"application/json": {"example": {**_FAILED, "params": None}}},
        },
        500: error(
            500,
            "A stage answered in an unexpected shape, or refused this service (e.g. a wrong API key)",
            "structuring service error (401): Invalid or missing API key",
        ),
        503: error(503, "A stage service is unreachable", "structuring service is unavailable"),
        504: error(504, "A stage service did not answer in time", "structuring service timed out after 10.0s"),
    },
)
async def get_extract_ocr(
    request_id: str,
    response: Response,
    service: ExtractOcrService = Depends(get_extract_service),
    settings: Settings = Depends(get_settings),
):
    token = bind_request_id(request_id)
    try:
        outcome = await service.status(request_id)
    finally:
        reset_request_id(token)
    status_code, body = extract_response(
        outcome,
        request_id=request_id,
        document_type=DOCUMENT_TYPE,
        params=None,
        threshold=settings.field_confidence_threshold,
    )
    response.status_code = status_code
    return body
