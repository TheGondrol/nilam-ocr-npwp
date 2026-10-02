from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from ocr_common.web.envelope import envelope
from ocr_common.web.request_id import get_request_id
from ocr_common.web.schemas import REQUEST_ID_EXAMPLE, UNAUTHORIZED, error, success_examples
from ocr_common.web.security import verify_api_key

from app.api.schemas import ScoringDirectRequest, ScoringDirectResponse
from app.api.scoring import CONFIDENCE_PAYLOAD_EXAMPLE
from app.config import Settings, get_settings
from app.dependencies import get_confidence_service
from app.services.confidence_service import ConfidenceService

router = APIRouter(tags=["Direct"], dependencies=[Depends(verify_api_key)])

_RESULT_EXAMPLE = {
    "npwp_confidence": 0.9806,
    "name_confidence": 0.9948,
    "fields": {
        "nomor_npwp": {"value": "12.345.678.9-012.000", "confidence": 1, "threshold": 0.9},
        "nama": {"value": "PT CONTOH INDONESIA", "confidence": 1, "threshold": 0.5},
    },
    "payload": CONFIDENCE_PAYLOAD_EXAMPLE,
}


@router.post(
    "/v1/scoring-direct",
    response_model=ScoringDirectResponse,
    operation_id="scoreDirect",
    summary="Score a structured document now, synchronous (no job, no callback)",
    description=(
        "**For testing one stage on its own (QC).** Takes the output of the previous stages, the structuring "
        "result (`data` of `POST /v1/structuring-direct`, or the `result` of a structuring job) and, like the "
        "pipeline, the OCR result and the guardrails report it depends on, builds the ML team's scoring payload "
        "from them, runs the trust model, and answers with the scoring result in the body. Nothing is recorded: "
        "no `nilam_scoring_jobs` row, no callback, no outcome row, so it never touches the orchestrator's data.\n\n"
        "The answer is exactly what `GET /v1/scoring/jobs/{request_id}` reports as `result`: the two "
        "probabilities, the 0/1 decision per field with the threshold used (`column_confidence_threshold` of "
        "the request, else `FIELD_CONFIDENCE_THRESHOLD`), and the exact payload that was scored. `ocr` left out "
        "gives null document scores, `guardrails` left out gives a null guardrails probability; the model's "
        "imputer fills both."
    ),
    responses={
        200: success_examples(
            "The structured document was scored",
            scored=("Both fields present", envelope(200, "Success", _RESULT_EXAMPLE, REQUEST_ID_EXAMPLE)),
        ),
        400: error(400, "An unsupported `document_type`", "Unsupported document_type: ktp. Supported: ['npwp']"),
        401: UNAUTHORIZED,
        422: error(422, "Validation Error", "body.structuring: Field required", errors="VALIDATION_ERROR"),
    },
)
async def score_direct(
    request: Request,
    body: ScoringDirectRequest,
    service: ConfidenceService = Depends(get_confidence_service),
    settings: Settings = Depends(get_settings),
):
    guardrails = body.guardrails.model_dump(exclude_unset=True) if body.guardrails is not None else None
    ocr = body.ocr.model_dump() if body.ocr is not None else None
    data = await run_in_threadpool(
        service.score,
        body.document_type,
        guardrails,
        ocr,
        body.structuring.model_dump(),
        settings.field_confidence_threshold,
        body.column_confidence_threshold,
    )
    return envelope(200, "Success", data, body.request_id or get_request_id(request))
