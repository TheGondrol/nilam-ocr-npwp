"""
Pabrik FastAPI app: satu tempat untuk middleware request_id, exception
handler (envelope), dan /health, supaya keempat service berperilaku sama
tanpa menyalin main.py.

Tiap service: `app = create_app(settings=..., title=..., routers=[...],
backends={...}, lifespan=...)` di src/main.py.
"""

import logging
from collections.abc import Callable, Iterable
from contextlib import AbstractAsyncContextManager
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse

from ocr_common.config import BaseServiceSettings
from ocr_common.envelope import envelope
from ocr_common.request_id import RequestIdMiddleware, get_request_id
from ocr_common.schemas import HealthResponse

Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]


def create_app(
    *,
    settings: BaseServiceSettings,
    title: str,
    description: str,
    version: str = "1.0.0",
    tags: Iterable[dict[str, Any]] = (),
    routers: Iterable[APIRouter] = (),
    backends: dict[str, str] | None = None,
    lifespan: Lifespan | None = None,
) -> FastAPI:
    # uvicorn hanya mengonfigurasi logger miliknya sendiri; tanpa ini pesan
    # INFO aplikasi (mis. "guardrails model loaded ...") tidak pernah tampil.
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    servers = (
        [{"url": settings.service_base_url, "description": "Configured base URL"}]
        if settings.service_base_url
        else None
    )
    app = FastAPI(
        title=title,
        version=version,
        description=description,
        openapi_tags=[{"name": "Health", "description": "Liveness check"}, *tags],
        servers=servers,
        lifespan=lifespan,
    )
    # Dibaca oleh security.verify_api_key dan intake.read_image, supaya lib
    # ini tidak perlu meng-import kelas Settings milik service.
    app.state.settings = settings

    app.add_middleware(RequestIdMiddleware)
    _register_exception_handlers(app)
    app.include_router(_health_router(version, backends or {}))
    for router in routers:
        app.include_router(router)
    return app


def _health_router(version: str, backends: dict[str, str]) -> APIRouter:
    router = APIRouter(tags=["Health"])

    @router.get(
        "/health",
        response_model=HealthResponse,
        summary="Health check",
        description="Reports service liveness and the active model backend. Does not require an API key.",
    )
    async def health():
        return {"status": "healthy", "version": version, "device": "cpu", "backends": backends}

    return router


def _register_exception_handlers(app: FastAPI) -> None:
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
        # bentuk yang tidak dikenal konsumen kita. errors diisi kode, bukan
        # pesan: pesannya menyebut field yang berbeda tiap kali, jadi hanya
        # kode yang bisa dicabang oleh pemanggil.
        request_id = get_request_id(request)
        message = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in exc.errors())
        return JSONResponse(
            status_code=422,
            content=envelope(422, message, None, request_id, errors="VALIDATION_ERROR"),
        )
