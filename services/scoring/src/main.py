from contextlib import asynccontextmanager

from fastapi import FastAPI

from ocr_common.app import add_stage_callback_webhook, create_app, database_readiness
from ocr_common.pipeline_schemas import ScoringStageCallback
from src.api.v1 import jobs, scoring
from src.api.v1.scoring import get_trust_model
from src.core.config import get_settings
from src.core.pipeline import get_pipeline
from src.models.scoring import get_scorer

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_scorer()
    get_trust_model()
    if settings.database_url:
        from ocr_common.database import check_connection

        await check_connection(settings.database_url)
    pipeline = get_pipeline()
    yield
    await pipeline.aclose(settings.pipeline_drain_timeout_seconds)
    if settings.database_url:
        from ocr_common.database import dispose_engines

        await dispose_engines()


app = create_app(
    settings=settings,
    title="OCR NPWP Scoring API",
    service_name="scoring",
    description=(
        "Pipeline step 4 (last) for Indonesian NPWP documents: combines per-field confidence and "
        "format validation into one document score plus an approve/review/reject decision. "
        "**Async pipeline:** the structuring service POSTs /v1/scoring/jobs and gets 202; this service "
        "scores in the background, stores the result, and POSTs the stage callback carrying the final "
        "result to the orchestrator. /v1/scoring/score is the same work, synchronous. "
        "All endpoints except /health require an X-API-Key header."
    ),
    tags=[
        {"name": "Pipeline", "description": "Asynchronous pipeline stage: 202, background work, callback"},
        {"name": "Callbacks", "description": "Requests this service SENDS to the orchestrator (see Webhooks)"},
        {"name": "Scoring", "description": "Document score & approve/review/reject decision, synchronous"},
    ],
    routers=[jobs.router, scoring.router],
    backends={
        "scoring": "trust_model",
        "legacy_score": settings.scoring_backend,
        "storage": "postgres" if settings.database_url else "memory",
    },
    readiness=database_readiness(settings.database_url),
    backends_example={"scoring": "trust_model", "legacy_score": "heuristic", "storage": "postgres"},
    readiness_example={"database": "ok"},
    lifespan=lifespan,
)

add_stage_callback_webhook(
    app,
    body_model=ScoringStageCallback,
    sent=(
        "Once per job of `POST /v1/scoring/jobs`, the last callback of a request: `stage: SCORING` with "
        "`status: DONE` and `result` = the **final result** (fields, per-field confidences, and the guardrails "
        "result that was submitted), or `status: FAILED` with `error_message` and `result: null`."
    ),
)
