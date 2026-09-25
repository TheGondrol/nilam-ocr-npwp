import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ocr_common.web.app import create_app

from app.api import extract_ocr, testing
from app.config import get_settings
from app.dependencies import (
    get_ekstraksi_client,
    get_guardrails_client,
    get_stage_status_clients,
    get_testing_ekstraksi_client,
    get_testing_stage_status_clients,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.guardrails_skip_allowed:
        logging.getLogger(__name__).warning(
            "GUARDRAILS_SKIP_ALLOWED=true: a request with skip_guardrails=true enters the pipeline without the "
            "guardrails model"
        )
    yield
    # The clients are built on the first request that needs them; close only those that exist.
    for get_client in (get_guardrails_client, get_ekstraksi_client, get_testing_ekstraksi_client):
        if get_client.cache_info().currsize:
            await get_client().aclose()
            get_client.cache_clear()
    for get_stages in (get_stage_status_clients, get_testing_stage_status_clients):
        if get_stages.cache_info().currsize:
            for stage in get_stages():
                await stage.aclose()
            get_stages.cache_clear()


app = create_app(
    settings=settings,
    title="OCR NPWP Orchestrator API",
    service_name="orchestrator",
    description=(
        "Entry point of the OCR NPWP pipeline, and the only service the central orchestrator / gateway calls. "
        "`POST /v1/extract-ocr` checks the file, has the guardrails service judge it, hands a document that "
        "passes to the OCR stage (ekstraksi, which chains to structuring and scoring), and waits up to "
        "`PIPELINE_WAIT_SECONDS` for the pipeline: 200 with the final result, or 202 while it is still running. "
        "`GET /v1/extract-ocr/{request_id}` answers the same contract for a request at any later time. This "
        "service stores nothing: the stages keep the jobs, and they send the result callback. All endpoints "
        "except /health, /ready and /metrics require an X-API-Key header."
    ),
    tags=[
        {"name": "Extract OCR", "description": "Start the pipeline for a document, and read where a request is"},
    ],
    routers=[extract_ocr.router, *(testing.routers if settings.testing_endpoints else [])],
    readiness={},
    lifespan=lifespan,
    entrypoint=True,
)
