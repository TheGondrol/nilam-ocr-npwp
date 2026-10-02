from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from ocr_common.errors import BadRequest
from ocr_common.npwp import DOCUMENT_TYPE
from ocr_common.web.envelope import envelope
from ocr_common.web.request_id import get_request_id
from ocr_common.web.schemas import REQUEST_ID_EXAMPLE, UNAUTHORIZED, error, success_examples
from ocr_common.web.security import verify_api_key

from app.api.jobs import STRUCTURED_EXAMPLE
from app.api.schemas import StructuringDirectRequest, StructuringDirectResponse
from app.dependencies import get_structuring_service
from app.services.structuring_service import StructuringService

router = APIRouter(tags=["Direct"], dependencies=[Depends(verify_api_key)])

_REJECTED_EXAMPLE = {
    **STRUCTURED_EXAMPLE,
    "flag": True,
    "flag_reason": "Dokumen bukan NPWP, mohon unggah kartu NPWP",
    "reject_reason": "Dokumen bukan NPWP, mohon unggah kartu NPWP",
}


@router.post(
    "/v1/structuring-direct",
    response_model=StructuringDirectResponse,
    operation_id="structureDirect",
    summary="Structure an OCR result now, synchronous (no job, no callback, no hand-off)",
    description=(
        "**For testing one stage on its own (QC).** Takes the output of the previous stage, the extraction "
        "service's OCR result (`data` of its `POST /v1/extraction/extract`, or the `result` of its job), runs "
        "the same structuring rules the pipeline runs, and answers with the structured document in the body. "
        "Nothing is recorded: no `structuring_jobs` row, no callback, no hand-off to scoring, so it never "
        "touches the orchestrator's data.\n\n"
        "The answer is exactly what `GET /v1/structuring/jobs/{request_id}` reports as `result`, and what the "
        "scoring stage receives as `structuring`: paste it into `POST /v1/scoring-direct` to test the next "
        "stage. Always 200 when the rules ran, also for a document they reject: read `reject_reason` (in the "
        "pipeline that rejection fails the job and gives the client a 400). OCR output without a single "
        "non-empty line is 400."
    ),
    responses={
        200: success_examples(
            "The OCR result was structured",
            accepted=("An NPWP card", envelope(200, "Success", STRUCTURED_EXAMPLE, REQUEST_ID_EXAMPLE)),
            rejected=(
                "A document the rules reject: `reject_reason` says why",
                envelope(200, "Success", _REJECTED_EXAMPLE, REQUEST_ID_EXAMPLE),
            ),
        ),
        400: error(400, "No text lines to structure, or an unsupported `document_type`", "No text lines to structure"),
        401: UNAUTHORIZED,
        422: error(422, "Validation Error", "body.ocr: Field required", errors="VALIDATION_ERROR"),
    },
)
async def structure_direct(
    request: Request,
    body: StructuringDirectRequest,
    service: StructuringService = Depends(get_structuring_service),
):
    if body.document_type != DOCUMENT_TYPE:
        raise BadRequest(f"Unsupported document_type: {body.document_type}. Supported: ['{DOCUMENT_TYPE}']")
    lines = StructuringService.lines_from_ocr(body.ocr.model_dump(exclude_unset=True))
    data = await run_in_threadpool(service.structure, lines)
    return envelope(200, "Success", data, body.request_id or get_request_id(request))
