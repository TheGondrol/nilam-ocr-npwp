"""
Composition root service Ekstraksi ("ServiceOCR" di sequence diagram):

- alur async (/v1/ekstraksi/jobs): tahap OCR pipeline; 202, kerja di
  background, callback ke Orkestrasi, handoff ke structuring;
- OCR mentah sinkron (/v1/ekstraksi/extract);
- kontrak orkestrator lama (generate-request-id -> extract-ocr ->
  get-ocr-result), dipertahankan sampai orkestrator pindah ke alur async.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ocr_common.app import add_stage_callback_webhook, create_app, database_readiness
from ocr_common.database import check_connection, dispose_engines
from ocr_common.pipeline_schemas import StageCallback
from src.api.v1 import ekstraksi, jobs, ocr
from src.clients.stages import get_stage_clients
from src.core.config import get_settings
from src.core.pipeline import get_next_stage, get_pipeline
from src.models.ekstraksi import get_ocr_engine

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Gagal keras saat startup, bukan di request pertama dengan error yang
    # membingungkan: DATABASE_URL yang tidak terjangkau, atau backend model
    # yang salah konfigurasi (mis. EKSTRAKSI_BACKEND=paddle tanpa URL).
    if settings.database_url:
        await check_connection(settings.database_url)
    ocr_engine = get_ocr_engine()
    stages = get_stage_clients()
    pipeline = get_pipeline()
    next_stage = get_next_stage()
    yield
    # Job yang masih jalan diberi waktu selesai dulu: mereka masih butuh klien
    # HTTP dan database di bawah ini.
    await pipeline.aclose(settings.pipeline_drain_timeout_seconds)
    await next_stage.aclose()
    if hasattr(ocr_engine, "aclose"):
        await ocr_engine.aclose()
    await stages.aclose()
    await dispose_engines()


app = create_app(
    settings=settings,
    title="OCR NPWP API",
    service_name="ekstraksi",
    description=(
        "OCR service for Indonesian NPWP (tax ID card) documents; ServiceOCR in the pipeline. "
        "**Async pipeline:** the orchestrator POSTs /v1/ekstraksi/jobs after guardrails passed and gets 202; "
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
        "Once per job of `POST /v1/ekstraksi/jobs`: `stage: OCR` with `status: DONE` after the OCR result is "
        "stored and BEFORE the job is handed to structuring, or `status: FAILED` with `error_message` when the "
        "document could not be read (the chain stops there). Additionally `stage: STRUCTURING`, "
        "`status: FAILED` when OCR succeeded but the structuring service could not be reached after retries."
    ),
)
