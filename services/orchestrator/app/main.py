from contextlib import asynccontextmanager

from fastapi import FastAPI

from ocr_common.pipeline.database import dispose_engines
from ocr_common.web.app import create_app, database_readiness

from app.api import extract_ocr, testing
from app.config import get_settings
from app.dependencies import (
    get_extraction_client,
    get_guardrails_client,
    get_stage_status_clients,
    get_testing_extraction_client,
    get_testing_stage_status_clients,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # The clients are built on the first request that needs them; close only those that exist.
    for get_client in (get_guardrails_client, get_extraction_client, get_testing_extraction_client):
        if get_client.cache_info().currsize:
            await get_client().aclose()
            get_client.cache_clear()
    for get_stages in (get_stage_status_clients, get_testing_stage_status_clients):
        if get_stages.cache_info().currsize:
            for stage in get_stages():
                await stage.aclose()
            get_stages.cache_clear()
    await dispose_engines()  # the nilam_guardrails_results writer's pool, when DATABASE_URL is set


app = create_app(
    settings=settings,
    title="OCR NPWP Orchestrator API",
    service_name="orchestrator",
    description=(
        "Entry point of the OCR NPWP pipeline, and the only service the central orchestrator / gateway calls. "
        "`POST /v1/extract-ocr` checks the file, has the guardrails service judge it, hands a document that "
        "passes to the OCR stage (extraction, which chains to structuring and scoring), and waits up to "
        "`PIPELINE_WAIT_SECONDS` for the pipeline: 200 with the final result, or 202 while it is still running. "
        "`GET /v1/extract-ocr/{request_id}` answers the same contract for a request at any later time. The "
        "stages keep the jobs and send the result callback; this service only keeps every guardrails verdict "
        "(`nilam_guardrails_results`, the rejected documents included) when `DATABASE_URL` is set. All endpoints "
        "except /health, /ready and /metrics require an X-API-Key header."
    ),
    tags=[
        {"name": "Extract OCR", "description": "Start the pipeline for a document, and read where a request is"},
    ],
    routers=[extract_ocr.router, *(testing.routers if settings.testing_endpoints else [])],
    # Not a readiness dependency: nilam_guardrails_results is best-effort, the entry point must keep serving without it.
    readiness={},
    health=database_readiness(settings.database_url),
    lifespan=lifespan,
    entrypoint=True,
)
