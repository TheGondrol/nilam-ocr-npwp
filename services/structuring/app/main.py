from contextlib import asynccontextmanager

from fastapi import FastAPI

from ocr_common.pipeline.schemas import StageCallback
from ocr_common.web.app import add_stage_callback_webhook, create_app, database_readiness

from app.api import jobs, structuring, testing
from app.config import get_settings
from app.dependencies import (
    get_next_stage,
    get_pipeline,
    get_reaper,
    get_relay,
    get_structurer,
    get_testing_next_stage,
    get_testing_pipeline,
    get_testing_reaper,
    get_testing_relay,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_structurer()
    if settings.database_url:
        from ocr_common.pipeline.database import check_connection

        await check_connection(settings.database_url)
    pipeline = get_pipeline()
    next_stage = get_next_stage()
    relay = get_relay()
    if relay is not None:
        relay.start()
    reaper = get_reaper()
    if reaper is not None:
        reaper.start()
    testing_relay = testing_reaper = None
    if settings.testing_endpoints:
        testing_relay, testing_reaper = get_testing_relay(), get_testing_reaper()
        if testing_relay is not None:
            testing_relay.start()
        if testing_reaper is not None:
            testing_reaper.start()
    yield
    if reaper is not None:
        await reaper.stop()
    await pipeline.aclose(settings.pipeline_drain_timeout_seconds, relay=relay)
    await next_stage.aclose()
    if settings.testing_endpoints:
        if testing_reaper is not None:
            await testing_reaper.stop()
        await get_testing_pipeline().aclose(settings.pipeline_drain_timeout_seconds, relay=testing_relay)
        await get_testing_next_stage().aclose()
    if settings.database_url:
        from ocr_common.pipeline.database import dispose_engines

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
    routers=[jobs.router, structuring.router, *([testing.router] if settings.testing_endpoints else [])],
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
        "Once per job of `POST /v1/structuring/jobs`: `stage: STRUCTURING` with `status: DONE` once the fields "
        "are stored, or `status: FAILED` with `error_message` when the upload is not a lone NPWP card or has no "
        "text (the chain stops there). Additionally `stage: SCORING`, `status: FAILED` when structuring succeeded "
        "but the scoring service could not be reached after retries. Without `PIPELINE_OUTBOX` the `STRUCTURING` "
        "callback is sent before the hand-off to scoring; with it the hand-off goes first, so the `SCORING` "
        "callback may arrive before this one."
    ),
)
