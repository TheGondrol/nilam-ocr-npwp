from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from ocr_common.envelope import envelope
from ocr_common.errors import ServiceError
from ocr_common.request_id import get_request_id
from ocr_common.schemas import REQUEST_ID_EXAMPLE, UNAUTHORIZED, error, success_examples
from ocr_common.security import verify_api_key
from src.core.config import get_settings
from src.models.scoring import get_scorer
from src.models.trust_model import TrustModel
from src.schemas.scoring import ConfidenceRequest, ConfidenceResponse, ScoreRequest, ScoreResponse
from src.services.confidence_service import ConfidenceService
from src.services.scoring_service import ScoringService

router = APIRouter(tags=["Scoring"], dependencies=[Depends(verify_api_key)])

# Dipakai juga oleh contoh GET /v1/scoring/jobs/{request_id}: payload yang dinilai model.
CONFIDENCE_PAYLOAD_EXAMPLE = {
    "npwp": "123456789012345",
    "npwp_score": 0.9992,
    "npwp_has_homoglyph": False,
    "npwp_candidate_count": 1,
    "name": "BUDI SANTOSO",
    "name_score": 0.9773,
    "name_corrected": False,
    "n_boxes": 6,
    "num_pages": 1,
    "avg_doc_score": 0.984883,
    "min_doc_score": 0.9747,
    "flag": None,
    "guardrail_probability": 0.9821,
}


def get_scoring_service() -> ScoringService:
    return ScoringService(get_scorer(), get_settings())


@lru_cache
def get_trust_model() -> TrustModel:
    return TrustModel(get_settings().scoring_model_path)


def get_confidence_service() -> ConfidenceService:
    return ConfidenceService(get_trust_model())


@router.post(
    "/v1/scoring/confidence",
    response_model=ConfidenceResponse,
    operation_id="predictConfidence",
    summary="Per-field confidence from the trust model, synchronous (no job, no callback)",
    description=(
        "The ML team's scoring contract: the payload carries the chained results of the previous stages, "
        "the answer is the probability that each extracted field is correct (`npwp_confidence`, "
        "`name_confidence`), from `weights/trust_model.joblib` (imputer -> scaler -> logistic regression, "
        "one feature row per field). There is no document score and no approve/reject decision here; "
        "thresholds are the caller's. A field whose value is null gets a null confidence. Any other null "
        "is treated as missing and filled by the model's own median imputer."
    ),
    responses={
        200: success_examples(
            "Probability that each extracted field is correct",
            both=(
                "Both fields present",
                envelope(200, "Success", {"npwp_confidence": 0.9806, "name_confidence": 0.9948}, REQUEST_ID_EXAMPLE),
            ),
            corrected_name=(
                "`name_corrected: true`: the model was trained on data where a corrected field is almost always wrong",
                envelope(200, "Success", {"npwp_confidence": 0.9737, "name_confidence": 0.001}, REQUEST_ID_EXAMPLE),
            ),
            no_name=(
                "`name` is null: a field that was not found has no confidence",
                envelope(200, "Success", {"npwp_confidence": 0.9737, "name_confidence": None}, REQUEST_ID_EXAMPLE),
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
    try:
        data = service.score(body.document_type, fields)
    except ServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    return envelope(200, "Success", data, get_request_id(request))
