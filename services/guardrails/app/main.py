from contextlib import asynccontextmanager

from fastapi import FastAPI

from ocr_common.web.app import create_app

from app.api import guardrails
from app.config import get_settings
from app.dependencies import get_ekstraksi_client, get_page_classifier, get_stage_status_clients

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier = get_page_classifier()
    yield
    if get_ekstraksi_client.cache_info().currsize:
        await get_ekstraksi_client().aclose()
        get_ekstraksi_client.cache_clear()
    if get_stage_status_clients.cache_info().currsize:
        for stage in get_stage_status_clients():
            await stage.aclose()
        get_stage_status_clients.cache_clear()
    close = getattr(classifier, "aclose", None)  # only the HTTP-backed models hold a connection
    if close is not None:
        await close()


app = create_app(
    settings=settings,
    title="OCR NPWP Guardrails API",
    service_name="guardrails",
    description=(
        "Entry point of the OCR NPWP pipeline. `POST /v1/extract-ocr` classifies every page of the "
        "document (image or PDF) as accepted / reject with the guardrails model, aggregates a document "
        "verdict, and when it passes hands the document to the OCR stage (ekstraksi), which chains to "
        "structuring and scoring. The orchestrator makes this one call: it waits up to `PIPELINE_WAIT_SECONDS` "
        "for the pipeline and answers 200 with the final result, or 202 while it is still running; the stage "
        "callbacks follow either way. The model runs in this process "
        "(`efficientnet`) or in the ML team's model service (`remote`). All endpoints except /health "
        "require an X-API-Key header."
    ),
    tags=[
        {"name": "Guardrails", "description": "Single entry point: guardrails check, then the OCR stage"},
    ],
    routers=[guardrails.router],
    backends={"guardrails": settings.guardrails_backend},
    readiness={},
    backends_example={"guardrails": "efficientnet"},
    lifespan=lifespan,
)
