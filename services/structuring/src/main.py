from contextlib import asynccontextmanager

from fastapi import FastAPI

from ocr_common.app import add_stage_callback_webhook, create_app, database_readiness
from ocr_common.pipeline_schemas import StageCallback
from src.api.v1 import jobs, structuring
from src.core.config import get_settings
from src.core.pipeline import get_next_stage, get_pipeline
from src.models.structuring import get_structurer

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_structurer()
    if settings.database_url:
        from ocr_common.database import check_connection

        await check_connection(settings.database_url)
    pipeline = get_pipeline()
    next_stage = get_next_stage()
    yield
    await pipeline.aclose(settings.pipeline_drain_timeout_seconds)
    await next_stage.aclose()
    if settings.database_url:
        from ocr_common.database import dispose_engines

        await dispose_engines()


app = create_app(
    settings=settings,
    title="OCR NPWP Structuring API",
    service_name="structuring",
    description=(
        "Pipeline step 3 for Indonesian NPWP documents: turns raw OCR text lines into named "
        "fields (nomor_npwp, nama, nama_badan) with per-field confidence. "
        "**Async pipeline:** the OCR service POSTs /v1/structuring/jobs and gets 202; this service "
        "structures in the background, stores the result, POSTs a stage callback to the orchestrator, and "
        "hands the job to the scoring service. /v1/structuring/structure is the same work, synchronous. "
        "All endpoints except /health require an X-API-Key header."
    ),
    tags=[
        {"name": "Pipeline", "description": "Asynchronous pipeline stage: 202, background work, callback, hand-off"},
        {"name": "Callbacks", "description": "Requests this service SENDS to the orchestrator (see Webhooks)"},
        {"name": "Structuring", "description": "Raw text -> named fields, synchronous"},
    ],
    routers=[jobs.router, structuring.router],
    backends={
        "structuring": settings.structuring_backend,
        "storage": "postgres" if settings.database_url else "memory",
    },
    readiness=database_readiness(settings.database_url),
    backends_example={"structuring": "npwp_rules", "storage": "postgres"},
    readiness_example={"database": "ok"},
    lifespan=lifespan,
)

add_stage_callback_webhook(
    app,
    body_model=StageCallback,
    sent=(
        "Once per job of `POST /v1/structuring/jobs`: `stage: STRUCTURING` with `status: DONE` after the fields "
        "are stored and BEFORE the job is handed to scoring, or `status: FAILED` with `error_message` when the "
        "upload is not a lone NPWP card or has no text (the chain stops there). Additionally `stage: SCORING`, "
        "`status: FAILED` when structuring succeeded but the scoring service could not be reached after retries."
    ),
)
