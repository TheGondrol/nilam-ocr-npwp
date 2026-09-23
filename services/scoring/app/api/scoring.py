from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from ocr_common.web.envelope import envelope
from ocr_common.web.request_id import get_request_id
from ocr_common.web.schemas import REQUEST_ID_EXAMPLE, UNAUTHORIZED, error, success_examples
from ocr_common.web.security import verify_api_key

from app.api.schemas import ConfidenceRequest, ConfidenceResponse, ScoreRequest, ScoreResponse
from app.dependencies import get_confidence_service, get_scoring_service
from app.services.confidence_service import ConfidenceService
from app.services.scoring_service import ScoringService

router = APIRouter(tags=["Scoring"], dependencies=[Depends(verify_api_key)])

CONFIDENCE_PAYLOAD_EXAMPLE = {
    "npwp": "12.345.678.9-012.000",
    "npwp_score": 0.98,
    "npwp_candidate_count": 1,
    "name_base": "PT CONTOH INDONESIA",
    "name_score": 0.95,
    "avg_doc_score": 0.91,
    "min_doc_score": 0.62,
    "flag": False,
    "guardrail_probability": 0.98,
}


@router.post(
    "/v1/scoring/confidence",
    response_model=ConfidenceResponse,
    operation_id="predictConfidence",
    summary="Per-field confidence from the trust model, synchronous (no job, no callback)",
    description=(
        "The ML team's scoring contract (`POST /v1/score` of their scoring service, model of 21 Sep 2026): the "
        "payload carries the signals of the previous stages, the answer is the probability that each extracted "
        "field is correct (`npwp_confidence`, `name_confidence`), from `weights/trust_model.joblib` (imputer -> "
        "scaler -> logistic regression, one feature row per field: `field_is_npwp`, `field_score`, "
        "`shape_confidence`, `field_multiple_candidates`, `name_max_char_len`, `avg_doc_score`, `min_doc_score`, "
        "`flag`, `guardrail_probability`). There is no document score and no approve/reject decision here; "
        "thresholds are the caller's. A field whose value is null gets a null confidence. Any other null is "
        "treated as missing and filled by the model's own median imputer."
    ),
    responses={
        200: success_examples(
            "Probability that each extracted field is correct",
            both=(
                "Both fields present",
                envelope(200, "Success", {"npwp_confidence": 0.9835, "name_confidence": 0.9806}, REQUEST_ID_EXAMPLE),
            ),
            flagged=(
                "`flag: true` (a review flag of the structuring rules) lowers both confidences",
                envelope(200, "Success", {"npwp_confidence": 0.9757, "name_confidence": 0.9725}, REQUEST_ID_EXAMPLE),
            ),
            no_name=(
                "`name_base` is null: a field that was not found has no confidence",
                envelope(200, "Success", {"npwp_confidence": 0.9835, "name_confidence": None}, REQUEST_ID_EXAMPLE),
            ),
        ),
        401: UNAUTHORIZED,
        422: error(
            422, "Validation Error", "body.npwp_score: Input should be a valid number", errors="VALIDATION_ERROR"
        ),
    },
)
async def confidence(
    request: Request,
    body: ConfidenceRequest,
    service: ConfidenceService = Depends(get_confidence_service),
):
    data = await run_in_threadpool(service.predict, body.model_dump())
    return envelope(200, "Success", data, get_request_id(request))


@router.post(
    "/v1/scoring/score",
    response_model=ScoreResponse,
    operation_id="scoreDocument",
    deprecated=True,
    summary="Legacy: heuristic document score (used only by the legacy extract-ocr contract)",
    description=(
        "Combines per-field confidence and format validation into one document score "
        "(the `guardrails` value of extract-ocr), then applies the configured "
        "thresholds to decide approve / review / reject."
    ),
    responses={
        200: success_examples(
            "The document was scored",
            approve=(
                "A person's card: `nama_badan` is optional, so missing it does not lower the score",
                envelope(
                    200,
                    "Success",
                    {
                        "score": 0.9904,
                        "decision": "approve",
                        "field_scores": [
                            {"name": "nomor_npwp", "score": 0.9992, "issues": []},
                            {"name": "nama", "score": 0.9773, "issues": []},
                            {"name": "nama_badan", "score": 0.0, "issues": ["missing"]},
                        ],
                        "reasons": [],
                    },
                    REQUEST_ID_EXAMPLE,
                ),
            ),
        ),
        400: error(400, "Unsupported document_type or no fields", "No fields to score"),
        401: UNAUTHORIZED,
        422: error(422, "Validation Error", "body.fields: Field required", errors="VALIDATION_ERROR"),
    },
)
async def score(
    request: Request,
    body: ScoreRequest,
    service: ScoringService = Depends(get_scoring_service),
):
    fields = {name: value.model_dump() for name, value in body.fields.items()}
    data = service.score(body.document_type, fields)
    return envelope(200, "Success", data, get_request_id(request))
