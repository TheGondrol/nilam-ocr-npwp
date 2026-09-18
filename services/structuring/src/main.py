"""Composition root service Structuring: rakit FastAPI app dari ocr_common + router service ini."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ocr_common.app import create_app
from src.api.v1 import jobs, structuring
from src.core.config import get_settings
from src.core.pipeline import get_next_stage, get_pipeline
from src.models.structuring import get_structurer

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_structurer()  # gagal saat boot kalau backend salah konfigurasi
    if settings.database_url:
        # Import di sini: sqlalchemy hanya dibutuhkan kalau DATABASE_URL diisi.
        from ocr_common.database import check_connection

        await check_connection(settings.database_url)
    pipeline = get_pipeline()
    next_stage = get_next_stage()
    yield
    # Job yang masih jalan diberi waktu selesai sebelum klien HTTP dan database ditutup.
    await pipeline.aclose(settings.pipeline_drain_timeout_seconds)
    await next_stage.aclose()
    if settings.database_url:
        from ocr_common.database import dispose_engines

        await dispose_engines()


app = create_app(
    settings=settings,
    title="OCR NPWP Structuring API",
    description=(
        "Pipeline step 3 for Indonesian NPWP documents: turns raw OCR text lines into named "
        "fields (nomor_npwp, nama, nama_badan) with per-field confidence. "
        "**Async pipeline:** the OCR service POSTs /v1/structuring/jobs and gets 202; this service "
        "structures in the background, stores the result, POSTs a stage callback to the orchestrator, and "
        "hands the job to the scoring service. /v1/structuring/structure is the same work, synchronous. "
        "All endpoints except /health require an X-API-Key header."
    ),
    tags=[
        {"name": "Pipeline", "description": "Async pipeline stage: 202 + background work + callback + handoff"},
        {"name": "Structuring", "description": "Raw text -> named fields, synchronous"},
    ],
    routers=[jobs.router, structuring.router],
    backends={
        "structuring": settings.structuring_backend,
        "storage": "postgres" if settings.database_url else "memory",
    },
    lifespan=lifespan,
)
