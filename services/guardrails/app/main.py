from contextlib import asynccontextmanager

from fastapi import FastAPI

from ocr_common.web.app import create_app

from app.api import guardrails
from app.config import get_settings
from app.dependencies import get_page_classifier, get_reject_threshold

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier = get_page_classifier()
    threshold = get_reject_threshold()
    yield
    await threshold.aclose()
    close = getattr(classifier, "aclose", None)  # only the HTTP-backed models hold a connection
    if close is not None:
        await close()


app = create_app(
    settings=settings,
    title="OCR NPWP Guardrails API",
    service_name="guardrails",
    description=(
        "The guardrails model of the OCR NPWP pipeline, internal: the orchestrator NPWP calls "
        "`POST /v1/guardrails/check` for every document it receives, and hands a document that passes to the "
        "OCR stage itself. The check classifies every page of the document (image or PDF) as accepted / reject "
        "and aggregates a document verdict. The model runs in this process (`efficientnet`) or in the ML team's "
        "model service (`remote`). All endpoints except /health, /ready and /metrics require an X-API-Key "
        "header."
    ),
    tags=[
        {"name": "Guardrails", "description": "The guardrails check (internal: called by the orchestrator NPWP)"},
    ],
    routers=[guardrails.router],
    backends={"guardrails": settings.guardrails_backend},
    readiness={},
    backends_example={"guardrails": "efficientnet"},
    lifespan=lifespan,
)
