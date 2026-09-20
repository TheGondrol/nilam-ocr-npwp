"""
API (presentation) layer: parses HTTP input, calls the service layer, and
translates ServiceError into the right HTTPException. No business rules
live here; see src/services/guardrails_service.py for those.
"""

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile

from ocr_common.envelope import envelope
from ocr_common.errors import ServiceError
from ocr_common.intake import FileField, FileUrlField, read_image
from ocr_common.schemas import UNAUTHORIZED, error, success_examples
from ocr_common.security import verify_api_key
from src.core.config import get_settings
from src.models.guardrails import get_page_classifier
from src.schemas.guardrails import GuardrailCheckResponse
from src.services.guardrails_service import GuardrailsService

router = APIRouter(tags=["Guardrails"], dependencies=[Depends(verify_api_key)])

RID = "OCR_9cb01af2-493d-446d-b191-af120333f6d0"


_ACCEPTED_PAGE = {"page_index": 0, "proba_approve": 0.9821, "proba_reject": 0.0179, "verdict": "accepted"}
_REJECTED_PAGE = {"page_index": 0, "proba_approve": 0.1179, "proba_reject": 0.8821, "verdict": "reject"}
_EXAMPLES = success_examples(
    "The document was judged. Always 200, accepted or not: read `data.passed`",
    accepted=(
        "Accepted: proceed to OCR",
        envelope(
            200,
            "OK",
            {
                "passed": True,
                "reason": None,
                "document": {"verdict": "accepted", "confidence": 0.9821, "n_pages": 1, "n_approve": 1, "n_reject": 0},
                "pages": [_ACCEPTED_PAGE],
            },
            RID,
        ),
    ),
    rejected=(
        "Rejected: stop and answer the client with `reason`",
        envelope(
            200,
            "OK",
            {
                "passed": False,
                "reason": "Document rejected by guardrails: 1/1 page(s) rejected (confidence 0.88)",
                "document": {"verdict": "reject", "confidence": 0.8821, "n_pages": 1, "n_approve": 0, "n_reject": 1},
                "pages": [_REJECTED_PAGE],
            },
            RID,
        ),
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
            },
            RID,
        ),
    ),
)


def get_guardrails_service() -> GuardrailsService:
    return GuardrailsService(get_page_classifier(), get_settings())


@router.post(
    "/v1/guardrails/check",
    response_model=GuardrailCheckResponse,
    operation_id="checkGuardrails",
    summary="Decide whether a document may proceed to OCR",
    description=(
        "**Step 1 of the pipeline, synchronous.** Call this first; submit the OCR job only when `data.passed` "
        "is true, and pass this response's `data` along as the `guardrails` form field of "
        "`POST /v1/ekstraksi/jobs`.\n\n"
        "Runs the guardrails model on each page (a PDF is rendered page by page; an image is one page) "
        "and aggregates a document verdict. The model is either in this process (`efficientnet`) or the "
        "ML team's model service (`remote`, POST /v1/predict/json); the response is the same either way. "
        "Always returns 200 with a report; `data.passed` tells whether "
        "the document should proceed to OCR, and `data.reason` says why not (the orchestrator answers 422 "
        "with it). The orchestrator forwards `data` to the OCR service as the guardrails result.\n\n"
        "Send `request_id` plus the document as `file`, or as `file_url` and this service fetches it "
        "itself; exactly one of the two."
    ),
    responses={
        200: _EXAMPLES,
        400: error(
            400, "Bad file (empty, too large, unsupported type, unreadable) or bad intake", "Uploaded file is empty"
        ),
        401: UNAUTHORIZED,
        422: error(422, "Validation Error", "body.request_id: Field required", errors="VALIDATION_ERROR"),
        500: error(
            500,
            "Guardrails model service failed or answered in an unexpected shape (backend `remote`)",
            "guardrails model returned an unexpected response",
        ),
        503: error(503, "Guardrails model service unreachable (backend `remote`)", "guardrails model is unavailable"),
        504: error(
            504,
            "Guardrails model service did not answer in time (backend `remote`)",
            "guardrails model timed out after 30.0s",
        ),
    },
)
async def check(
    request: Request,
    request_id: str = Form(..., description="request_id from the orchestrator", examples=[RID]),
    file: UploadFile | str | None = FileField,
    file_url: str | None = FileUrlField,
    service: GuardrailsService = Depends(get_guardrails_service),
):
    content, filename, content_type = await read_image(request, file, file_url)
    try:
        data = await service.check(filename, content_type, content)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    return envelope(200, "OK", data, request_id)
