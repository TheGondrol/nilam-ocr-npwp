"""
Pabrik FastAPI app: satu tempat untuk middleware request_id, exception
handler (envelope), dan /health, supaya keempat service berperilaku sama
tanpa menyalin main.py.

Tiap service: `app = create_app(settings=..., title=..., routers=[...],
backends={...}, lifespan=...)` di src/main.py.
"""

import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse

from ocr_common.config import BaseServiceSettings
from ocr_common.envelope import envelope
from ocr_common.request_id import RequestIdMiddleware, get_request_id
from ocr_common.schemas import HealthResponse, ReadyResponse

logger = logging.getLogger(__name__)

Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]
# Satu pemeriksaan dependensi wajib: selesai tanpa exception = ok.
ReadinessCheck = Callable[[], Awaitable[None]]


def database_readiness(database_url: str | None) -> dict[str, ReadinessCheck]:
    """Readiness untuk service yang menyimpan status. Kosong kalau tanpa DATABASE_URL (dev, in-memory)."""
    if not database_url:
        return {}

    async def check() -> None:
        # Import di sini: sqlalchemy hanya extra `ocr-common[db]`.
        from ocr_common.database import check_connection

        await check_connection(database_url)

    return {"database": check}


def create_app(
    *,
    settings: BaseServiceSettings,
    title: str,
    description: str,
    version: str = "1.0.0",
    tags: Iterable[dict[str, Any]] = (),
    routers: Iterable[APIRouter] = (),
    backends: dict[str, str] | None = None,
    readiness: Mapping[str, ReadinessCheck] | None = None,
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
    if settings.auth_disabled:
        logging.getLogger(__name__).warning(
            "AUTH_DISABLED=true: X-API-Key is NOT checked on this service; only for local development"
        )

    app.add_middleware(RequestIdMiddleware)
    _register_exception_handlers(app)
    app.include_router(_health_router(version, backends or {}, readiness or {}))
    for router in routers:
        app.include_router(router)
    return app


def _health_router(version: str, backends: dict[str, str], readiness: Mapping[str, ReadinessCheck]) -> APIRouter:
    router = APIRouter(tags=["Health"])

    @router.get(
        "/health",
        response_model=HealthResponse,
        summary="Health check",
        description="Reports service liveness and the active model backend. Does not require an API key.",
    )
    async def health():
        return {"status": "healthy", "version": version, "device": "cpu", "backends": backends}

    @router.get(
        "/ready",
        response_model=ReadyResponse,
        summary="Readiness check",
        description=(
            "Liveness vs readiness: /health only says the process is alive and never touches a dependency "
            "(a database blip must not make Kubernetes restart every pod). /ready says whether this pod can "
            "do its job right now: 200 when every REQUIRED dependency of this service answers, 503 otherwise, "
            "so the pod is taken out of the Service until it recovers. Downstream stages and model services "
            "are deliberately not checked: their outage is reported per job, not by refusing traffic. "
            "Does not require an API key."
        ),
        responses={503: {"model": ReadyResponse, "description": "A required dependency is unavailable"}},
    )
    async def ready():
        checks: dict[str, str] = {}
        for name, check in readiness.items():
            try:
                await check()
                checks[name] = "ok"
            except Exception as exc:
                # Hanya jenis error-nya: pesan koneksi bisa memuat host / user database.
                logger.warning("readiness check %r failed: %s", name, type(exc).__name__)
                checks[name] = "failed"
        ok = all(state == "ok" for state in checks.values())
        return JSONResponse(
            status_code=200 if ok else 503, content={"status": "ready" if ok else "not_ready", "checks": checks}
        )

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
