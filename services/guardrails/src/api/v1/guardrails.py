from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile

from ocr_common.envelope import envelope
from ocr_common.errors import ServiceError
from ocr_common.intake import FileField, FileUrlField, read_image
from ocr_common.schemas import UNAUTHORIZED, error, success_examples
from ocr_common.security import verify_api_key
from src.clients.ekstraksi import EkstraksiJobClient, get_ekstraksi_client
from src.core.config import get_settings
from src.models.guardrails import get_page_classifier
from src.schemas.guardrails import GuardrailCheckResponse
from src.services.guardrails_service import GuardrailsService
from src.services.job_service import GuardrailsJobService

router = APIRouter(tags=["Guardrails"], dependencies=[Depends(verify_api_key)])

RID = "OCR_9cb01af2-493d-446d-b191-af120333f6d0"


_ACCEPTED_PAGE = {"page_index": 0, "proba_approve": 0.9821, "proba_reject": 0.0179, "verdict": "accepted"}
_REJECTED_PAGE = {"page_index": 0, "proba_approve": 0.1179, "proba_reject": 0.8821, "verdict": "reject"}
_JOB = {"request_id": RID, "stage": "OCR", "status": "PROCESSING", "duplicate": False}
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
    "job": None,
}
_EXAMPLES = success_examples(
    "The document was judged. Always 200, accepted or not: read `data.passed`",
    accepted=(
        "Accepted: the OCR stage was started, `job` is its answer",
        envelope(200, "OK", {**_ACCEPTED_REPORT, "job": _JOB}, RID),
    ),
    duplicate=(
        "Same request_id sent again: the OCR stage did not start a second run",
        envelope(200, "OK", {**_ACCEPTED_REPORT, "job": {**_JOB, "status": "DONE", "duplicate": True}}, RID),
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
                "job": None,
            },
            RID,
        ),
    ),
)


def get_guardrails_service() -> GuardrailsService:
    return GuardrailsService(get_page_classifier(), get_settings())


def get_job_service(
    guardrails: GuardrailsService = Depends(get_guardrails_service),
    ekstraksi: EkstraksiJobClient = Depends(get_ekstraksi_client),
) -> GuardrailsJobService:
    return GuardrailsJobService(guardrails, ekstraksi)


@router.post(
    "/v1/extract-ocr",
    response_model=GuardrailCheckResponse,
    operation_id="extractOcr",
    summary="Judge a document and, when it passes, start the pipeline",
    description=(
        "**The only call the orchestrator makes.** Runs the guardrails model on the document synchronously, "
        "then:\n\n"
        "- **passed** -> sends the same document on to the OCR stage (`POST /v1/ekstraksi/jobs` on the "
        "ekstraksi service) and returns the report with `data.job`. OCR -> structuring -> scoring then run in "
        "the background and report through the `OCR`, `STRUCTURING` and `SCORING` stage callbacks; the final "
        "result is in the `SCORING` callback.\n"
        "- **rejected** -> returns the report with `data.passed: false`, `data.reason` and `data.job: null`. "
        "Nothing else runs and no callback follows; the orchestrator answers the client 422 with `reason`.\n\n"
        "Always 200 when the document was judged: read `data.passed`, not the HTTP code.\n\n"
        "Runs the guardrails model on each page (a PDF is rendered page by page; an image is one page) "
        "and aggregates a document verdict. The model is either in this process (`efficientnet`) or the "
        "ML team's model service (`remote`, POST /v1/predict/json); the response is the same either way.\n\n"
        "Send `request_id` plus the document as `file`, or as `file_url` (downloaded here once, then forwarded "
        "as a file, so the URL only has to live for this call); exactly one of the two.\n\n"
        "**Idempotency.** The same request_id again re-runs the guardrails check, but the OCR stage answers "
        "`duplicate: true` and does not run twice unless the earlier attempt `FAILED`."
    ),
    responses={
        200: _EXAMPLES,
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
    content, filename, content_type = await read_image(request, file, file_url)
    try:
        data = await service.submit(request_id, document_type, filename, content_type, content, handoff=handoff)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    return envelope(200, "OK", data, request_id)
