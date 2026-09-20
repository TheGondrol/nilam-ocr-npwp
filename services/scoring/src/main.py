"""Composition root service Scoring: rakit FastAPI app dari ocr_common + router service ini."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ocr_common.app import create_app, database_readiness
from src.api.v1 import jobs, scoring
from src.api.v1.scoring import get_trust_model
from src.core.config import get_settings
from src.core.pipeline import get_pipeline
from src.models.scoring import get_scorer

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_scorer()  # gagal saat boot kalau backend salah konfigurasi
    get_trust_model()  # file model hilang / versi scikit-learn beda -> gagal saat boot, bukan di job pertama
    if settings.database_url:
        # Import di sini: sqlalchemy hanya dibutuhkan kalau DATABASE_URL diisi.
        from ocr_common.database import check_connection

        await check_connection(settings.database_url)
    pipeline = get_pipeline()
    yield
    # Job yang masih jalan diberi waktu selesai sebelum klien HTTP dan database ditutup.
    await pipeline.aclose(settings.pipeline_drain_timeout_seconds)
    if settings.database_url:
        from ocr_common.database import dispose_engines

        await dispose_engines()


app = create_app(
    settings=settings,
    title="OCR NPWP Scoring API",
    description=(
        "Pipeline step 4 (last) for Indonesian NPWP documents: combines per-field confidence and "
        "format validation into one document score plus an approve/review/reject decision. "
        "**Async pipeline:** the structuring service POSTs /v1/scoring/jobs and gets 202; this service "
        "scores in the background, stores the result, and POSTs the stage callback carrying the final "
        "result to the orchestrator. /v1/scoring/score is the same work, synchronous. "
        "All endpoints except /health require an X-API-Key header."
    ),
    tags=[
        {"name": "Pipeline", "description": "Async pipeline stage: 202 + background work + callback"},
        {"name": "Scoring", "description": "Document score & approve/review/reject decision, synchronous"},
    ],
    routers=[jobs.router, scoring.router],
    backends={
        "scoring": "trust_model",
        "legacy_score": settings.scoring_backend,
        "storage": "postgres" if settings.database_url else "memory",
    },
    readiness=database_readiness(settings.database_url),
    lifespan=lifespan,
)
