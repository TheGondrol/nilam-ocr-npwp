from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from ocr_common.envelope import envelope
from ocr_common.errors import ServiceError
from ocr_common.request_id import get_request_id
from ocr_common.schemas import UNAUTHORIZED, error
from ocr_common.security import verify_api_key
from src.core.config import get_settings
from src.models.scoring import get_scorer
from src.models.trust_model import TrustModel
from src.schemas.scoring import ConfidenceRequest, ConfidenceResponse, ScoreRequest, ScoreResponse
from src.services.confidence_service import ConfidenceService
from src.services.scoring_service import ScoringService

router = APIRouter(tags=["Scoring"], dependencies=[Depends(verify_api_key)])


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
    summary="Per-field confidence from the ML team's trust model",
    description=(
        "The ML team's scoring contract: the payload carries the chained results of the previous stages, "
        "the answer is the probability that each extracted field is correct (`npwp_confidence`, "
        "`name_confidence`), from `weights/trust_model.joblib` (imputer -> scaler -> logistic regression, "
        "one feature row per field). There is no document score and no approve/reject decision here; "
        "thresholds are the caller's. A field whose value is null gets a null confidence. Any other null "
        "is treated as missing and filled by the model's own median imputer."
    ),
    responses={
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
    summary="Legacy: heuristic document score (used only by the legacy extract-ocr contract)",
    description=(
        "Combines per-field confidence and format validation into one document score "
        "(the `guardrails` value of extract-ocr), then applies the configured "
        "thresholds to decide approve / review / reject."
    ),
    responses={
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
