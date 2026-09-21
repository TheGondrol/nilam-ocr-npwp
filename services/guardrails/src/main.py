from contextlib import asynccontextmanager

from fastapi import FastAPI

from ocr_common.app import create_app
from src.api.v1 import guardrails
from src.clients.ekstraksi import get_ekstraksi_client
from src.core.config import get_settings
from src.models.guardrails import get_page_classifier

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier = get_page_classifier()
    yield
    if get_ekstraksi_client.cache_info().currsize:
        await get_ekstraksi_client().aclose()
        get_ekstraksi_client.cache_clear()
    if hasattr(classifier, "aclose"):
        await classifier.aclose()


app = create_app(
    settings=settings,
    title="OCR NPWP Guardrails API",
    service_name="guardrails",
    description=(
        "Entry point of the OCR NPWP pipeline. `POST /v1/extract-ocr` classifies every page of the "
        "document (image or PDF) as accepted / reject with the guardrails model, aggregates a document "
        "verdict, and when it passes hands the document to the OCR stage (ekstraksi), which chains to "
        "structuring and scoring. The orchestrator makes this one call and then receives the stage "
        "callbacks. The model runs in this process "
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
