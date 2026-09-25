import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, Form, Request, Response, UploadFile

from ocr_common.image_validation import PAYLOAD_TOO_LARGE_MESSAGE
from ocr_common.npwp import DOCUMENT_TYPE
from ocr_common.pipeline import DEFAULT_SEQUENCE, InvalidSequence, validate_sequence
from ocr_common.web.intake import FileField, FileUrlField, read_image
from ocr_common.web.request_id import adopt_request_id, reset_request_id
from ocr_common.web.schemas import UNAUTHORIZED, error, success_examples
from ocr_common.web.security import verify_api_key

from app.api.extract_contract import (
    COMPLETED_MESSAGE,
    PROCESSING_MESSAGE,
    REJECTED_CODE,
    extract_body,
    extract_response,
)
from app.api.schemas import ExtractOcrResponse
from app.config import Settings, get_settings
from app.dependencies import get_extract_service
from app.services.document_checks import TOO_MANY_PAGES_MESSAGE
from app.services.extract_service import ExtractOcrService
from app.services.pipeline_waiter import StageError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Extract OCR"], dependencies=[Depends(verify_api_key)])

RID = "OCR_9cb01af2-493d-446d-b191-af120333f6d0"
INVALID_PARAMS_MESSAGE = "params must be valid JSON: an object, or a quoted string"
INVALID_SEQUENCE_CODE = "INVALID_PIPELINE_SEQUENCE"

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
    guardrails=0,
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
    400,
    "Document rejected by guardrails: 1/1 page(s) rejected (confidence 0.88)",
    errors=REJECTED_CODE,
    job_status="failed",
    guardrails=1,
    request_id=RID,
    document_type="npwp",
    params=_PARAMS,
)
_FAILED = extract_body(
    422,
    "No text lines to structure",
    errors="STRUCTURING_FAILED",
    job_status="failed",
    guardrails=0,
    request_id=RID,
    document_type="npwp",
    params=_PARAMS,
)
_GUARDRAILS_REPORT = {
    "passed": True,
    "reason": None,
    "document": {
        "verdict": "accepted",
        "confidence": 0.9821,
        "n_pages": 1,
        "n_approve": 1,
        "n_reject": 0,
        "reject_threshold": 0.5,
    },
    "pages": [{"page_index": 0, "proba_approve": 0.9821, "proba_reject": 0.0179, "verdict": "accepted"}],
}
_GUARDRAILS_ONLY = extract_body(
    200,
    COMPLETED_MESSAGE,
    data=_GUARDRAILS_REPORT,
    job_status="completed",
    guardrails=0,
    request_id=RID,
    document_type="npwp",
    params=_PARAMS,
)

_CONTRACT_TABLE = (
    "| Outcome | HTTP | `job_status` | `data` | `guardrails` | `errors` | `pipeline_last_stage` |\n"
    "|---|---|---|---|---|---|---|\n"
    "| Finished | 200 | `completed` | the fields | `0` | null | the last service of the sequence |\n"
    "| Still running | 202 | `processing` | null | null | null | the service still running |\n"
    f"| Rejected by the guardrails model | 400 | `failed` | null | `1` | `{REJECTED_CODE}` | `guardrails` |\n"
    f"| Rejected by the structuring rules | 400 | `failed` | null | `1` | `{REJECTED_CODE}` | `structuring` |\n"
    "| A stage failed | 422 | `failed` | null | `0` | `OCR_FAILED`, `STRUCTURING_FAILED` or "
    "`SCORING_FAILED` | the service that failed |\n\n"
    "`pipeline_last_stage` names the pipeline service an answer comes from, as `pipeline_name_sequence` names "
    "it (`guardrails`, `extraction`, `structuring`, `scoring`), also on a 400 / 500 / 503 / 504 from calling "
    "one of them. It is null when this service refused the request before calling any (file checks, "
    "`pipeline_name_sequence`, `params`, `document_type`).\n\n"
)


class _InvalidParams(Exception):
    pass


def _stage_error_body(exc: StageError, *, request_id: str, document_type: str, params: Any) -> dict[str, Any]:
    """The answer when calling a pipeline service failed (unreachable, timed out, refused the file, answered
    wrongly): the error envelope's status and message, in the extract-ocr shape, naming that service."""
    if exc.status_code >= 500:
        logger.error("%s -> %d: %s", exc.service, exc.status_code, exc.message)
    return extract_body(
        exc.status_code,
        exc.message,
        errors=exc.message,
        request_id=request_id,
        document_type=document_type,
        params=params,
        pipeline_last_stage=exc.service,
    )


def _stage_error_response(code: int, description: str, service: str, message: str) -> dict[str, Any]:
    example = extract_body(
        code, message, errors=message, request_id=RID, document_type="npwp", pipeline_last_stage=service
    )
    return {
        "model": ExtractOcrResponse,
        "description": description,
        "content": {"application/json": {"example": example}},
    }


def _parse_sequence(values: list[str] | None) -> tuple[str, ...]:
    """`pipeline_name_sequence` as repeated form fields, or as one JSON array string; the full pipeline when
    omitted. Raises `InvalidSequence`."""
    if not values:
        return DEFAULT_SEQUENCE
    if len(values) == 1 and values[0].lstrip().startswith("["):
        try:
            parsed = json.loads(values[0])
        except ValueError:
            parsed = None
        if not isinstance(parsed, list) or not all(isinstance(name, str) for name in parsed):
            raise InvalidSequence("send it as a JSON array of strings, or as repeated form fields")
        values = parsed
    return validate_sequence(values)


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
        "hands it to the OCR stage when it passes, then waits for OCR -> structuring -> scoring (or the services "
        "`pipeline_name_sequence` names, see below) for up to "
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
        "`MAX_UPLOAD_BYTES` (2.5 MB by default) answers `413`, one with more than `MAX_DOCUMENT_PAGES` "
        "(2) pages answers `400`, both with an Indonesian `message` the client can show as is.\n\n"
        "**Which services run: `pipeline_name_sequence`.** The services of this request, in order: `guardrails`, "
        "`extraction`, `structuring`, `scoring`. Guardrails may be left out at the front and the end cut off, "
        "never one skipped in the middle or the order changed (else `422` "
        f"`{INVALID_SEQUENCE_CODE}` and nothing runs). Omitted: all four. The last service ends the request and "
        "its result is `data`, **as it is**: the guardrails report (`{passed, reason, document, pages}`) when "
        "only `guardrails` runs, the OCR result (`{text, blocks, ...}`) after `extraction`, the structuring result "
        "(`{fields, flag, reject_reason, ...}`) after `structuring`, and the fields above only after `scoring`. "
        "The rest of the body is the same. The structuring rules still reject when `structuring` runs. Leaving "
        "`guardrails` out is the central orchestrator's call: the file checks above always run, and the trust "
        "model then works without a guardrails probability. A `guardrails`-only request stores nothing: its "
        "POST answer is final, and `GET /v1/extract-ocr/{request_id}` answers 404 for it.\n\n"
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
            "Accepted and finished within the wait",
            completed=("The OCR result", _COMPLETED),
            guardrails_only=(
                '`pipeline_name_sequence: ["guardrails"]`: the guardrails report as it is',
                _GUARDRAILS_ONLY,
            ),
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
                f"Rejected by the guardrails model or by the structuring rules (`{REJECTED_CODE}`, `guardrails: 1`), "
                "unsupported `document_type` (`UNSUPPORTED_DOCUMENT_TYPE`), more than `MAX_DOCUMENT_PAGES` "
                f"pages (`{TOO_MANY_PAGES_MESSAGE}`), or a bad file / intake (empty, unsupported type, unreadable, "
                "`file_url` refused)"
            ),
            "content": {"application/json": {"example": _REJECTED}},
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
                "`message` says why), `params` is not valid JSON (`INVALID_PARAMS`), `pipeline_name_sequence` breaks "
                f"the order rules (`{INVALID_SEQUENCE_CODE}`), or a required field is missing (`VALIDATION_ERROR`)"
            ),
            "content": {"application/json": {"example": _FAILED}},
        },
        500: _stage_error_response(
            500,
            "The guardrails or extraction service failed, or answered in an unexpected shape; "
            "`pipeline_last_stage` names which",
            "guardrails",
            "guardrails service returned an unexpected response",
        ),
        503: _stage_error_response(
            503,
            "The guardrails service, its model, or the extraction service is unreachable; nothing was started. "
            "`pipeline_last_stage` names which",
            "extraction",
            "extraction service is unavailable",
        ),
        504: _stage_error_response(
            504,
            "The guardrails service, its model, or the extraction service did not answer in time; "
            "`pipeline_last_stage` names which",
            "extraction",
            "extraction service timed out after 10.0s",
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
    pipeline_name_sequence: list[str] | None = Form(
        None,
        description=(
            "The services to run, in order: `guardrails`, `extraction`, `structuring`, `scoring`; guardrails "
            "optional at the front, the end may be cut off, nothing skipped in the middle. Repeated form fields, or "
            "one JSON array string. Omitted: all four. The last one's result is `data`, as it is"
        ),
        examples=[["guardrails", "extraction", "structuring", "scoring"]],
    ),
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
    try:
        sequence = _parse_sequence(pipeline_name_sequence)
    except InvalidSequence as exc:
        response.status_code = 422
        return extract_body(
            422,
            f"Invalid pipeline_name_sequence: {exc}",
            errors=INVALID_SEQUENCE_CODE,
            request_id=request_id,
            document_type=document_type,
        )

    # The central orchestrator's request_id becomes the id of this request: in the envelope of an error raised
    # below (413, a bad file, an unreachable stage), in the X-Request-ID response header and outbound calls, and
    # in our log lines, so one id follows the request through guardrails and every stage.
    token = adopt_request_id(request, request_id)
    try:
        content, filename, content_type = await read_image(request, file, file_url)
        outcome = await service.submit(
            request_id,
            document_type,
            filename,
            content_type,
            content,
            received_at=received_at,
            file_url=file_url,
            sequence=sequence,
        )
    except StageError as exc:
        response.status_code = exc.status_code
        return _stage_error_body(exc, request_id=request_id, document_type=document_type, params=parsed_params)
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
        "(the `400` of the POST is its final answer), refused before the check, or its POST is still being "
        "judged by guardrails.\n\n"
        "**Limitation.** A hand-off between two stages that failed for good (its retries ran out, or it became a "
        "dead letter in the outbox) leaves the next stage without a job, so this endpoint keeps answering `202` "
        "for it. The `FAILED` callback and the central orchestrator's own tables carry that final state; this "
        "endpoint only reads the stages' jobs."
    ),
    responses={
        200: success_examples(
            "Finished",
            completed=("The OCR result", {**_COMPLETED, "params": None}),
        ),
        202: {
            **success_examples(
                "Still running",
                processing=("Still processing", {**_PROCESSING, "params": None}),
            ),
            "model": ExtractOcrResponse,
        },
        400: {
            "model": ExtractOcrResponse,
            "description": f"Rejected by the structuring rules (`{REJECTED_CODE}`, `guardrails: 1`)",
            "content": {
                "application/json": {
                    "example": {
                        **_REJECTED,
                        "message": "Kode provinsi pada NPWP tidak valid, mohon dicek kembali",
                        "params": None,
                    }
                }
            },
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
        500: _stage_error_response(
            500,
            "A stage answered in an unexpected shape, or refused this service (e.g. a wrong API key); "
            "`pipeline_last_stage` names which",
            "structuring",
            "structuring service error (401): Invalid or missing API key",
        ),
        503: _stage_error_response(
            503,
            "A stage service is unreachable; `pipeline_last_stage` names which",
            "structuring",
            "structuring service is unavailable",
        ),
        504: _stage_error_response(
            504,
            "A stage service did not answer in time; `pipeline_last_stage` names which",
            "structuring",
            "structuring service timed out after 10.0s",
        ),
    },
)
async def get_extract_ocr(
    request_id: str,
    request: Request,
    response: Response,
    service: ExtractOcrService = Depends(get_extract_service),
    settings: Settings = Depends(get_settings),
):
    token = adopt_request_id(request, request_id)
    try:
        outcome = await service.status(request_id)
    except StageError as exc:
        response.status_code = exc.status_code
        return _stage_error_body(exc, request_id=request_id, document_type=DOCUMENT_TYPE, params=None)
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
