import time
from functools import lru_cache

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, UploadFile

from ocr_common.envelope import envelope
from ocr_common.errors import ServiceError
from ocr_common.intake import FileField, FileUrlField, read_image
from ocr_common.jobs import STATUS_DONE, STATUS_FAILED
from ocr_common.schemas import UNAUTHORIZED, error, success_examples
from ocr_common.security import verify_api_key
from src.clients.ekstraksi import EkstraksiJobClient, get_ekstraksi_client
from src.clients.stages import get_stage_status_clients
from src.core.config import Settings, get_settings
from src.models.guardrails import get_page_classifier
from src.schemas.guardrails import GuardrailCheckResponse
from src.services.guardrails_service import GuardrailsService
from src.services.job_service import GuardrailsJobService
from src.services.pipeline_waiter import PipelineWaiter

router = APIRouter(tags=["Guardrails"], dependencies=[Depends(verify_api_key)])

RID = "OCR_9cb01af2-493d-446d-b191-af120333f6d0"


_ACCEPTED_PAGE = {"page_index": 0, "proba_approve": 0.9821, "proba_reject": 0.0179, "verdict": "accepted"}
_REJECTED_PAGE = {"page_index": 0, "proba_approve": 0.1179, "proba_reject": 0.8821, "verdict": "reject"}
_JOB = {"request_id": RID, "stage": "OCR", "status": "PROCESSING", "duplicate": False}
_NOTHING_STARTED = {"job": None, "pipeline": None, "result": None}
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
    **_NOTHING_STARTED,
}
_RESULT = {
    "document_type": "npwp",
    "fields": {
        "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0.9762},
        "nama": {"value": "BUDI SANTOSO", "confidence": 0.9931},
        "nama_badan": {"value": None, "confidence": 0.0},
    },
    "scoring": {"npwp_confidence": 0.7296, "name_confidence": 0.9471},
    "guardrails": _ACCEPTED_REPORT,
}
_DONE = {"stage": "SCORING", "status": "DONE", "error_message": None}
_EXAMPLES = success_examples(
    "A final answer: rejected, or accepted and the pipeline finished (DONE or FAILED) within the wait",
    done=(
        "Accepted and finished within the wait: `result` is the final result (same as the SCORING callback)",
        envelope(200, "OK", {**_ACCEPTED_REPORT, "job": _JOB, "pipeline": _DONE, "result": _RESULT}, RID),
    ),
    failed=(
        "Accepted, but a stage failed within the wait: the request ends here, as with a FAILED callback",
        envelope(
            200,
            "OK",
            {
                **_ACCEPTED_REPORT,
                "job": _JOB,
                "pipeline": {"stage": "OCR", "status": "FAILED", "error_message": "ekstraksi OCR model is unavailable"},
                "result": None,
            },
            RID,
        ),
    ),
    duplicate=(
        "Same request_id sent again after it finished: nothing runs twice, the stored result is returned",
        envelope(
            200,
            "OK",
            {
                **_ACCEPTED_REPORT,
                "job": {**_JOB, "status": "DONE", "duplicate": True},
                "pipeline": _DONE,
                "result": _RESULT,
            },
            RID,
        ),
    ),
    rejected=(
        "Rejected: nothing was started; answer the client with `reason`",
        envelope(200, "OK", _REJECTED_REPORT, RID),
    ),
    pdf_one_bad_page=(
        "PDF with one rejected page: the whole document is rejected (policy `all`)",
        envelope(
            200,
            "OK",
            {
                "passed": False,
                "reason": "Document rejected by guardrails: 1/2 page(s) rejected (confidence 0.70)",
                "document": {"verdict": "reject", "confidence": 0.7, "n_pages": 2, "n_approve": 1, "n_reject": 1},
                "pages": [
                    _ACCEPTED_PAGE,
                    {"page_index": 1, "proba_approve": 0.3, "proba_reject": 0.7, "verdict": "reject"},
                ],
                **_NOTHING_STARTED,
            },
            RID,
        ),
    ),
)
_STILL_RUNNING = {
    **success_examples(
        "Accepted, but the pipeline was still running when the wait ran out: the outcome follows in the stage "
        "callbacks",
        processing=(
            "Still running after PIPELINE_WAIT_SECONDS",
            envelope(
                202,
                "Accepted",
                {
                    **_ACCEPTED_REPORT,
                    "job": _JOB,
                    "pipeline": {"stage": "STRUCTURING", "status": "PROCESSING", "error_message": None},
                    "result": None,
                },
                RID,
            ),
        ),
        no_wait=(
            "Waiting disabled (PIPELINE_WAIT_SECONDS=0): answered right after the hand-off",
            envelope(202, "Accepted", {**_ACCEPTED_REPORT, "job": _JOB, "pipeline": None, "result": None}, RID),
        ),
    ),
    "model": GuardrailCheckResponse,
}


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
    response_model=GuardrailCheckResponse,
    operation_id="extractOcr",
    summary="Judge a document and, when it passes, start the pipeline",
    description=(
        "**The only call the orchestrator makes.** Runs the guardrails model on the document synchronously, "
        "then:\n\n"
        "- **rejected** -> **200** with `data.passed: false` and `data.reason`; `job`, `pipeline` and `result` are "
        "null. Nothing else runs and no callback follows; the orchestrator answers the client 422 with `reason`.\n"
        "- **passed** -> sends the same document on to the OCR stage (`POST /v1/ekstraksi/jobs` on the "
        "ekstraksi service; its answer is `data.job`), then waits for OCR -> structuring -> scoring for up to "
        "`PIPELINE_WAIT_SECONDS` (15 by default), counted from when this request arrived:\n"
        "  - finished in time -> **200**: `data.pipeline.status` is `DONE` with the final result in `data.result` "
        "(identical to the `result` of the SCORING callback), or `FAILED` with the failed `stage` and "
        "`error_message`;\n"
        "  - still running -> **202**: `data.pipeline.status` is `PROCESSING`; the outcome follows in the stage "
        "callbacks, or read it with `GET /v1/scoring/jobs/{request_id}`.\n\n"
        "The `OCR`, `STRUCTURING` and `SCORING` callbacks are sent in every case, also when the result is already "
        "in this response. The wait is counted from the request's arrival, so the answer normally comes within "
        "`PIPELINE_WAIT_SECONDS`; give this call an HTTP timeout well above it (e.g. +15 s) to cover a slow "
        "guardrails check or hand-off.\n\n"
        "Runs the guardrails model on each page (a PDF is rendered page by page; an image is one page) "
        "and aggregates a document verdict. The model is either in this process (`efficientnet`) or the "
        "ML team's model service (`remote`, POST /v1/predict/json); the response is the same either way.\n\n"
        "Send `request_id` plus the document as `file`, or as `file_url` (downloaded here once, then forwarded "
        "as a file, so the URL only has to live for this call); exactly one of the two.\n\n"
        "**Idempotency.** The same request_id again re-runs the guardrails check, but the OCR stage answers "
        "`duplicate: true` and does not run twice unless the earlier attempt `FAILED` or outlived the job lease "
        "(`PIPELINE_JOB_LEASE_SECONDS`). A request_id that already finished answers 200 with its stored result."
    ),
    responses={
        200: _EXAMPLES,
        202: _STILL_RUNNING,
        400: error(
            400,
            "Bad file (empty, too large, unsupported type, unreadable), bad intake, or rejected by the OCR stage",
            "Uploaded file is empty",
        ),
        401: UNAUTHORIZED,
        422: error(422, "Validation Error", "body.request_id: Field required", errors="VALIDATION_ERROR"),
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
async def check(
    request: Request,
    response: Response,
    request_id: str = Form(..., description="request_id minted by the orchestrator", examples=[RID]),
    document_type: str = Form(
        "npwp", description="Document type chosen by the client. Only `npwp` is supported", examples=["npwp"]
    ),
    handoff: bool = Form(
        True,
        description=(
            "false: judge only, start nothing, `job` stays null. Used by the legacy `extract-ocr` contract, which "
            "runs the stages itself; the orchestrator leaves it out"
        ),
        examples=[True],
    ),
    file: UploadFile | str | None = FileField,
    file_url: str | None = FileUrlField,
    service: GuardrailsJobService = Depends(get_job_service),
):
    received_at = time.monotonic()
    content, filename, content_type = await read_image(request, file, file_url)
    try:
        data = await service.submit(
            request_id, document_type, filename, content_type, content, handoff=handoff, received_at=received_at
        )
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    if data["job"] is not None and (data["pipeline"] or {}).get("status") not in (STATUS_DONE, STATUS_FAILED):
        response.status_code = 202
        return envelope(202, "Accepted", data, request_id)
    return envelope(200, "OK", data, request_id)
