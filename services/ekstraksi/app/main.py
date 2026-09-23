from contextlib import asynccontextmanager

from fastapi import FastAPI

from ocr_common.pipeline.database import check_connection, dispose_engines
from ocr_common.pipeline.schemas import StageCallback
from ocr_common.web.app import add_stage_callback_webhook, create_app, database_readiness

from app.api import ekstraksi, jobs, ocr
from app.config import get_settings
from app.dependencies import (
    get_next_stage,
    get_ocr_engine,
    get_pipeline,
    get_reaper,
    get_relay,
    get_stage_clients,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.database_url:
        await check_connection(settings.database_url)
    ocr_engine = get_ocr_engine()
    stages = get_stage_clients()
    pipeline = get_pipeline()
    next_stage = get_next_stage()
    relay = get_relay()
    if relay is not None:
        relay.start()
    reaper = get_reaper()
    if reaper is not None:
        reaper.start()
    yield
    if reaper is not None:
        await reaper.stop()
    await pipeline.aclose(settings.pipeline_drain_timeout_seconds, relay=relay)
    await next_stage.aclose()
    close = getattr(ocr_engine, "aclose", None)  # only the HTTP-backed models hold a connection
    if close is not None:
        await close()
    await stages.aclose()
    await dispose_engines()


app = create_app(
    settings=settings,
    title="OCR NPWP API",
    service_name="ekstraksi",
    description=(
        "OCR service for Indonesian NPWP (tax ID card) documents; ServiceOCR in the pipeline. "
        "**Async pipeline:** the guardrails service POSTs /v1/ekstraksi/jobs once a document passed and gets 202; "
        "this service runs OCR in the background, stores the result, POSTs a stage callback to the "
        "orchestrator, and hands the job to the structuring service. "
        "**Legacy contract:** generate-request-id -> extract-ocr -> get-ocr-result runs the whole chain "
        "synchronously and stays until the orchestrator has moved to the async flow. "
        "The raw OCR step is also exposed as /v1/ekstraksi/extract. "
        "All endpoints except /health require an X-API-Key header."
    ),
    tags=[
        {"name": "Pipeline", "description": "Asynchronous pipeline stage: 202, background work, callback, hand-off"},
        {"name": "Callbacks", "description": "Requests this service SENDS to the orchestrator (see Webhooks)"},
        {
            "name": "NPWP OCR",
            "description": "Legacy synchronous contract used by ocr-orchestration",
        },
        {"name": "Ekstraksi", "description": "Raw OCR text, synchronous"},
    ],
    routers=[jobs.router, ocr.router, ekstraksi.router],
    backends={
        "ekstraksi": settings.ekstraksi_backend,
        "storage": "postgres" if settings.database_url else "memory",
    },
    readiness=database_readiness(settings.database_url),
    backends_example={"ekstraksi": "paddle", "storage": "postgres"},
    readiness_example={"database": "ok"},
    lifespan=lifespan,
)

add_stage_callback_webhook(
    app,
    body_model=StageCallback,
    sent=(
        "Once per job of `POST /v1/ekstraksi/jobs`: `stage: OCR` with `status: DONE` once the OCR result is "
        "stored, or `status: FAILED` with `error_message` when the document could not be read (the chain stops "
        "there). Additionally `stage: STRUCTURING`, `status: FAILED` when OCR succeeded but the structuring "
        "service could not be reached after retries. Without `PIPELINE_OUTBOX` the `OCR` callback is sent before "
        "the hand-off to structuring; with it the hand-off goes first, so the `STRUCTURING` callback may arrive "
        "before this one."
    ),
)
