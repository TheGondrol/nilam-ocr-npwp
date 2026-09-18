from fastapi import APIRouter, Depends, HTTPException, Request

from src.core.config import get_settings
from src.core.envelope import envelope
from src.core.errors import ServiceError
from src.core.request_id import get_request_id
from src.core.security import API_KEY_ERROR, verify_api_key
from src.models.scoring import get_scorer
from src.schemas.errors import error
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
        "(the `guardrails` value of /v1/extract-ocr), then applies the configured "
        "thresholds to decide approve / review / reject."
    ),
    responses={
        400: error(400, "Unsupported document_type or no fields", "No fields to score"),
        401: error(401, "Missing or invalid X-API-Key", API_KEY_ERROR),
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
