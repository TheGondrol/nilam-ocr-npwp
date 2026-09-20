"""
Pabrik FastAPI app: satu tempat untuk middleware request_id, exception
handler (envelope), dan /health, supaya keempat service berperilaku sama
tanpa menyalin main.py.

Tiap service: `app = create_app(settings=..., title=..., routers=[...],
backends={...}, lifespan=...)` di src/main.py.
"""

import json
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any

from fastapi import APIRouter, FastAPI, Header, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse

from ocr_common.config import BaseServiceSettings
from ocr_common.envelope import envelope
from ocr_common.request_id import RequestIdMiddleware, get_request_id
from ocr_common.schemas import HealthResponse, ReadyResponse, success_examples

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


# Bagian yang sama di Swagger keempat service: aturan yang harus diketahui pemanggil sebelum
# membaca endpoint mana pun. Ditulis sekali di sini supaya tidak pernah berbeda antar service.
API_CONVENTIONS = """

## Conventions (the same for every nilam-ocr service)

**Authentication.** Every endpoint except `/health` and `/ready` requires the header `X-API-Key`.
A missing or wrong key answers `401`. The services are reachable only from inside the cluster.

**Envelope.** Every JSON response, success or error, has the same shape:
`{status_code, status_desc, message, data, errors, request_id}`. `status_code` always equals the HTTP
status. On success `errors` is null; on error `data` is null.

**Errors.** `errors` is machine-readable: `VALIDATION_ERROR` for `422`, otherwise the same text as
`message`. `400` = the request or the document is unusable (do not retry unchanged). `401` = API key.
`404` = unknown request_id. `409` = request_id already processed (legacy contract only). `422` = a
field is missing or has the wrong type. `500` = this service or its model failed. `503` = a
dependency is unreachable, `504` = it did not answer in time (both are safe to retry).

**request_id.** Minted by the orchestrator and carried through every stage. Where an endpoint has no
request_id of its own, the `X-Request-ID` request header is used (and echoed in the response header);
without it the service generates one.

**Asynchronous stages.** `POST .../jobs` answers `202` immediately and does the work in the
background. The outcome is reported by a callback to the orchestrator (see *Webhooks*), and can be
read at any time with `GET .../jobs/{request_id}`. Submitting the same request_id again is idempotent:
`202` with `duplicate: true`, the work is not repeated, unless the earlier attempt `FAILED`. Because
the request was already answered `202`, a problem with the document itself (unreadable file, no text,
unsupported document type) shows up as a `FAILED` job and callback, not as a `4xx`.
"""


def create_app(
    *,
    settings: BaseServiceSettings,
    title: str,
    description: str,
    service_name: str | None = None,
    version: str = "1.0.0",
    tags: Iterable[dict[str, Any]] = (),
    routers: Iterable[APIRouter] = (),
    backends: dict[str, str] | None = None,
    readiness: Mapping[str, ReadinessCheck] | None = None,
    # Contoh untuk Swagger. STATIS, bukan nilai yang sedang aktif: openapi.yaml tidak boleh berubah
    # mengikuti .env tempat ia dibuat (efficientnet di laptop, mock di test).
    backends_example: dict[str, str] | None = None,
    readiness_example: dict[str, str] | None = None,
    lifespan: Lifespan | None = None,
) -> FastAPI:
    # uvicorn hanya mengonfigurasi logger miliknya sendiri; tanpa ini pesan
    # INFO aplikasi (mis. "guardrails model loaded ...") tidak pernah tampil.
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # "/" di urutan pertama: Swagger UI memakai server pertama untuk "Try it out", dan halaman /docs
    # selalu disajikan oleh service itu sendiri. Sisanya adalah alamat yang dipakai pemanggil sungguhan.
    servers: list[dict[str, Any]] = [{"url": "/", "description": "This host (where this page is served)"}]
    if settings.service_base_url:
        servers.append({"url": settings.service_base_url, "description": "Configured base URL"})
    if service_name:
        servers += [
            {
                "url": f"http://{service_name}.nilam-ocr-{{environment}}.svc.cluster.local:{settings.port}",
                "description": "GKE, from another namespace (e.g. the orchestrator / gateway)",
                "variables": {
                    "environment": {
                        "default": "dev",
                        "enum": ["dev", "staging", "production"],
                        "description": "One namespace per environment: nilam-ocr-<environment>",
                    }
                },
            },
            {
                "url": f"http://{service_name}:{settings.port}",
                "description": "GKE from the same namespace, or the Docker Compose network",
            },
            {"url": f"http://127.0.0.1:{settings.port}", "description": "Local development"},
        ]
    app = FastAPI(
        title=title,
        version=version,
        description=description.rstrip() + API_CONVENTIONS,
        openapi_tags=[*tags, {"name": "Health", "description": "Liveness and readiness probes; no API key"}],
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
    app.include_router(
        _health_router(version, backends or {}, readiness or {}, backends_example or {}, readiness_example or {})
    )
    for router in routers:
        app.include_router(router)
    return app


def add_stage_callback_webhook(app: FastAPI, *, body_model: type, sent: str) -> None:
    """Dokumentasikan callback tahap sebagai webhook OpenAPI 3.1: request yang DIKIRIM service ini ke
    Orkestrasi. Webhook, bukan `callbacks` per operasi, karena URL tujuannya tidak datang dari request
    melainkan dari konfigurasi (ORCHESTRATION_URL + ORCHESTRATION_CALLBACK_PATH). Tidak menambah route."""

    @app.webhooks.post(
        "stageCallback",
        operation_id="stageCallback",
        tags=["Callbacks"],
        summary="Stage status callback (implemented by the orchestrator / gateway)",
        description=(
            "**This is a request this service SENDS, not one it accepts.** The orchestrator must expose it.\n\n"
            "`POST {ORCHESTRATION_URL}{ORCHESTRATION_CALLBACK_PATH}` (default path `/v1/callbacks/stage`), "
            "`Content-Type: application/json`.\n\n"
            f"**When.** {sent}\n\n"
            "**Expected answer.** Any `2xx`; the body is ignored. `5xx`, a timeout or an unreachable host are "
            "retried (3 attempts by default, exponential back-off from 0.5 s). A `4xx` is NOT retried. A "
            "callback that still fails is logged and dropped: the job itself stays `DONE` / `FAILED`, so the "
            "orchestrator can reconcile with `GET .../jobs/{request_id}` and should time a stage out on its own.\n\n"
            "**What the receiver must tolerate.** The same callback may arrive more than once (treat it as "
            "idempotent), and callbacks of different stages come from different services, so they may arrive "
            "out of order: only ever move a request's status forward. Reject callbacks whose `X-API-Key` is wrong."
        ),
        responses={200: {"description": "Acknowledged. Any 2xx is accepted and the body is ignored."}},
    )
    def stage_callback(
        body: body_model,  # ty: ignore[invalid-type-form]
        x_api_key: str = Header(
            ...,
            description="`ORCHESTRATION_API_KEY` of the sending service; when unset, that service's own `API_KEY`",
        ),
    ) -> None:
        """Hanya untuk dokumentasi; tidak pernah dipanggil."""

    # FastAPI menambahkan 422 + HTTPValidationError ke setiap operasi yang punya body. Di webhook itu
    # menyesatkan: jawaban endpoint ini ditentukan Orkestrasi, bukan oleh service ini.
    generate = app.openapi

    def openapi() -> dict[str, Any]:
        fresh = app.openapi_schema is None
        schema = generate()
        if fresh:
            schema["webhooks"]["stageCallback"]["post"]["responses"].pop("422", None)
            # Setiap 422 di service ini memakai ErrorResponse, jadi skema bawaan FastAPI tinggal yatim.
            if "#/components/schemas/HTTPValidationError" not in json.dumps(schema["paths"]):
                for orphan in ("HTTPValidationError", "ValidationError"):
                    schema["components"]["schemas"].pop(orphan, None)
        return schema

    app.openapi = openapi  # ty: ignore[invalid-assignment]


def _health_router(
    version: str,
    backends: dict[str, str],
    readiness: Mapping[str, ReadinessCheck],
    backends_example: dict[str, str],
    readiness_example: dict[str, str],
) -> APIRouter:
    router = APIRouter(tags=["Health"])

    @router.get(
        "/health",
        response_model=HealthResponse,
        operation_id="getHealth",
        summary="Liveness probe",
        description=(
            "Says the process is alive and which implementations are active. Never touches a dependency. "
            "Does not require an API key."
        ),
        responses={
            200: success_examples(
                "The process is alive",
                healthy=(
                    "Healthy",
                    {"status": "healthy", "version": version, "device": "cpu", "backends": backends_example},
                ),
            )
        },
    )
    async def health():
        return {"status": "healthy", "version": version, "device": "cpu", "backends": backends}

    @router.get(
        "/ready",
        response_model=ReadyResponse,
        operation_id="getReadiness",
        summary="Readiness probe",
        description=(
            "Liveness vs readiness: /health only says the process is alive and never touches a dependency "
            "(a database blip must not make Kubernetes restart every pod). /ready says whether this pod can "
            "do its job right now: 200 when every REQUIRED dependency of this service answers, 503 otherwise, "
            "so the pod is taken out of the Service until it recovers. Downstream stages and model services "
            "are deliberately not checked: their outage is reported per job, not by refusing traffic. "
            "Does not require an API key."
        ),
        responses={
            200: success_examples(
                "Every required dependency answers",
                ready=("Ready", {"status": "ready", "checks": readiness_example}),
            ),
            503: {
                "model": ReadyResponse,
                "description": "A required dependency is unavailable; the pod is taken out of the Service",
                "content": {
                    "application/json": {
                        "example": {"status": "not_ready", "checks": {name: "failed" for name in readiness_example}}
                    }
                },
            },
        },
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
