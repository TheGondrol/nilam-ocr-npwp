from fastapi import APIRouter, Depends, HTTPException, Request

from ocr_common.envelope import envelope
from ocr_common.errors import ServiceError
from ocr_common.request_id import get_request_id
from ocr_common.schemas import UNAUTHORIZED, error
from ocr_common.security import verify_api_key
from src.core.config import get_settings
from src.models.scoring import get_scorer
from src.schemas.scoring import ScoreRequest, ScoreResponse
from src.services.scoring_service import ScoringService

router = APIRouter(tags=["Scoring"], dependencies=[Depends(verify_api_key)])


def get_scoring_service() -> ScoringService:
    return ScoringService(get_scorer(), get_settings())


@router.post(
    "/v1/scoring/score",
    response_model=ScoreResponse,
    summary="Score a structured document",
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
