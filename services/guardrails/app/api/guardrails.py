from fastapi import APIRouter, Depends, Form, Request, UploadFile

from ocr_common.image_validation import PAYLOAD_TOO_LARGE_MESSAGE
from ocr_common.web.envelope import envelope
from ocr_common.web.intake import FileField, FileUrlField, read_image
from ocr_common.web.request_id import get_request_id
from ocr_common.web.schemas import UNAUTHORIZED, error, success_examples
from ocr_common.web.security import verify_api_key

from app.api.schemas import GuardrailReportResponse
from app.dependencies import get_guardrails_service
from app.services.guardrails_service import GuardrailsService
from app.services.pages import TOO_MANY_PAGES_MESSAGE

router = APIRouter(tags=["Guardrails"], dependencies=[Depends(verify_api_key)])

RID = "OCR_9cb01af2-493d-446d-b191-af120333f6d0"

_ACCEPTED_PAGE = {"page_index": 0, "proba_approve": 0.9821, "proba_reject": 0.0179, "verdict": "accepted"}
_REJECTED_PAGE = {"page_index": 0, "proba_approve": 0.1179, "proba_reject": 0.8821, "verdict": "reject"}
_ACCEPTED_REPORT = {
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
    "pages": [_ACCEPTED_PAGE],
}
_REJECTED_REPORT = {
    "passed": False,
    "reason": "Document rejected by guardrails: 1/1 page(s) rejected (confidence 0.88)",
    "document": {
        "verdict": "reject",
        "confidence": 0.8821,
        "n_pages": 1,
        "n_approve": 0,
        "n_reject": 1,
        "reject_threshold": 0.5,
    },
    "pages": [_REJECTED_PAGE],
}


@router.post(
    "/v1/guardrails/check",
    response_model=GuardrailReportResponse,
    operation_id="checkDocument",
    summary="Judge a document (internal: called by the orchestrator NPWP)",
    description=(
        "Runs the guardrails model on each page (a PDF is rendered page by page; an image is one page) and "
        "aggregates a document verdict, **without** starting anything: no OCR job, no callback. The model is "
        "either in this process (`efficientnet`) or the ML team's model service (`remote`, POST "
        "/v1/predict/json); the report is the same either way. The orchestrator NPWP calls it for every "
        "`extract-ocr` and hands `data` on to the OCR stage when `passed`. Always 200 when the document was "
        "judged: read `data.passed`. Refused before the model runs: more than `GUARDRAILS_MAX_DOCUMENT_PAGES` "
        f"pages (400 `{TOO_MANY_PAGES_MESSAGE}`), a document above `MAX_UPLOAD_BYTES` (413), an empty, "
        "unreadable or unsupported file (400)."
    ),
    responses={
        200: success_examples(
            "The document was judged",
            accepted=("Accepted", envelope(200, "OK", _ACCEPTED_REPORT, RID)),
            rejected=("Rejected", envelope(200, "OK", _REJECTED_REPORT, RID)),
        ),
        400: error(
            400,
            "Bad file (empty, unsupported type, unreadable, more than `GUARDRAILS_MAX_DOCUMENT_PAGES` pages) or "
            "bad intake",
            TOO_MANY_PAGES_MESSAGE,
        ),
        401: UNAUTHORIZED,
        413: error(413, "The document exceeds `MAX_UPLOAD_BYTES`", PAYLOAD_TOO_LARGE_MESSAGE.format(limit="2,5 MB")),
        422: error(422, "Validation Error", "body.file: Field required", errors="VALIDATION_ERROR"),
        500: error(500, "The guardrails model failed", "guardrails model returned an unexpected response"),
        503: error(503, "The guardrails model service (`remote`) is unreachable", "guardrails model is unavailable"),
        504: error(
            504,
            "The guardrails model service (`remote`) did not answer in time",
            "guardrails model timed out after 30.0s",
        ),
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
    report = await service.check(filename, content_type, content)
    return envelope(200, "OK", report, request_id or get_request_id(request))
