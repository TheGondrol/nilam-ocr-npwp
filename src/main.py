from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse

from src.api.v1 import ekstraksi, guardrails, health, ocr, scoring, structuring
from src.core.config import get_settings
from src.core.database import check_connection, dispose_engine
from src.core.envelope import envelope
from src.core.request_id import RequestIdMiddleware, get_request_id

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Gagal keras saat startup kalau DATABASE_URL diisi tapi database tidak
    # terjangkau, bukan di request pertama dengan error yang membingungkan.
    if settings.database_url:
        await check_connection()
    yield
    await dispose_engine()


servers = (
    [{"url": settings.service_base_url, "description": "Configured base URL"}] if settings.service_base_url else None
)

app = FastAPI(
    title="OCR NPWP API",
    lifespan=lifespan,
    version="1.0.0",
    description=(
        "OCR service for Indonesian NPWP (tax ID card) documents. "
        "Implements the generate-request-id -> extract-ocr -> get-ocr-result "
        "contract used by ocr-orchestration: mint a request_id, submit a file "
        "against it, then poll for the result. The pipeline behind extract-ocr "
        "is also exposed step by step (guardrails -> ekstraksi -> structuring -> "
        "scoring) for debugging and evaluation. All endpoints except /health "
        "require an X-API-Key header."
    ),
    openapi_tags=[
        {"name": "Health", "description": "Liveness check"},
        {
            "name": "NPWP OCR",
            "description": "Endpoints for extracting data from NPWP images (contract used by ocr-orchestration)",
        },
        {"name": "Guardrails", "description": "Pipeline step 1: image quality & document type"},
        {"name": "Ekstraksi", "description": "Pipeline step 2: raw OCR text"},
        {"name": "Structuring", "description": "Pipeline step 3: raw text -> named fields"},
        {"name": "Scoring", "description": "Pipeline step 4: document score & approve/review/reject"},
    ],
    servers=servers,
)

app.add_middleware(RequestIdMiddleware)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = get_request_id(request)
    return JSONResponse(
        status_code=exc.status_code,
        content=envelope(exc.status_code, str(exc.detail), None, request_id, errors=str(exc.detail)),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Form/path/body yang hilang atau salah bentuk gagal sebelum fungsi
    # endpoint jalan. Tanpa handler ini FastAPI menjawab {"detail": [...]},
    # bentuk yang tidak dikenal konsumen kita. errors diisi kode, bukan pesan:
    # pesannya menyebut field yang berbeda tiap kali, jadi hanya kode yang
    # bisa dicabang oleh pemanggil.
    request_id = get_request_id(request)
    message = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in exc.errors())
    return JSONResponse(
        status_code=422,
        content=envelope(422, message, None, request_id, errors="VALIDATION_ERROR"),
    )


app.include_router(health.router)
app.include_router(ocr.router)
app.include_router(guardrails.router)
app.include_router(ekstraksi.router)
app.include_router(structuring.router)
app.include_router(scoring.router)
