"""Composition root service Guardrails: rakit FastAPI app dari ocr_common + router service ini."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ocr_common.app import create_app
from src.api.v1 import guardrails
from src.core.config import get_settings
from src.models.guardrails import get_page_classifier

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Muat model saat startup: salah path bobot, backend salah konfigurasi (mis.
    # GUARDRAILS_BACKEND=remote tanpa URL) gagal saat boot, dan request pertama
    # tidak menanggung waktu pemuatan.
    classifier = get_page_classifier()
    yield
    # Backend remote memegang koneksi HTTP persisten ke service model.
    if hasattr(classifier, "aclose"):
        await classifier.aclose()


app = create_app(
    settings=settings,
    title="OCR NPWP Guardrails API",
    description=(
        "Pipeline step 1 for Indonesian NPWP documents: classifies every page of the uploaded "
        "document (image or PDF) as accepted / reject with the guardrails model and aggregates a "
        "document verdict. Called synchronously by ocr-orchestration; always answers 200 with "
        "`data.passed` / `data.reason`. The model runs in this process (`efficientnet`) or in the ML "
        "team's model service (`remote`). All endpoints except /health require an X-API-Key header."
    ),
    tags=[{"name": "Guardrails", "description": "Page-level accepted/reject classification"}],
    routers=[guardrails.router],
    backends={"guardrails": settings.guardrails_backend},
    # Model dimuat di lifespan sebelum server menerima koneksi, jadi pod yang
    # menjawab sudah pasti memegang model: tidak ada dependensi lain untuk diperiksa.
    readiness={},
    lifespan=lifespan,
)
