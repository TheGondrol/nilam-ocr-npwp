"""The FastAPI application factory shared by the services: logging, request id, metrics, the standard
envelope for every error, the health, readiness and metrics routes, and the callback webhook
documentation.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any

from fastapi import APIRouter, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException

from ocr_common.config import BaseServiceSettings
from ocr_common.errors import ServiceError, error_code
from ocr_common.web.envelope import envelope
from ocr_common.web.logging import configure_logging
from ocr_common.web.metrics import MetricsMiddleware, metrics_response
from ocr_common.web.request_id import RequestIdMiddleware, get_request_id
from ocr_common.web.schemas import HealthResponse, ReadyResponse, success_examples

logger = logging.getLogger(__name__)

Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]
ReadinessCheck = Callable[[], Awaitable[None]]


# A dependency check that does not answer in this time counts as failed (a database behind a dead route can
# otherwise hold the connection attempt for minutes).
CHECK_TIMEOUT_SECONDS = 3.0


async def _run_checks(checks: Mapping[str, ReadinessCheck], kind: str) -> dict[str, str]:
    results: dict[str, str] = {}
    for name, check in checks.items():
        try:
            await asyncio.wait_for(check(), CHECK_TIMEOUT_SECONDS)
            results[name] = "ok"
        except Exception as exc:
            logger.warning("%s check %r failed: %s", kind, name, type(exc).__name__)
            results[name] = "failed"
    return results


def database_readiness(database_url: str | None) -> dict[str, ReadinessCheck]:
    """A readiness check named `database` that runs `SELECT 1`; empty when there is no database."""
    if not database_url:
        return {}

    async def check() -> None:
        from ocr_common.pipeline.database import check_connection

        await check_connection(database_url)

    return {"database": check}


API_CONVENTIONS = """

## Conventions (the same for every nilam-ocr service)

**Authentication.** Every endpoint except `/health`, `/ready` and `/metrics` requires the header `X-API-Key`.
A missing or wrong key answers `401`. The services are reachable only from inside the cluster.

**Envelope.** Every JSON response, success or error, has the same shape:
`{status_code, status_desc, message, data, errors, request_id}`. `status_code` always equals the HTTP
status. On success `errors` is null; on error `data` is null.

**Errors.** `errors` is a stable, machine-readable code; branch on it, not on the wording of `message`.
The document checks: `EMPTY_FILE`, `UNSUPPORTED_FILE_TYPE`, `UNREADABLE_FILE`, `TOO_MANY_PAGES`,
`INVALID_FILE_SOURCE`, `FILE_URL_REJECTED` (400), `FILE_TOO_LARGE` (413). Otherwise the code of the status:
`UNAUTHORIZED` (401), `REQUEST_ID_NOT_FOUND` (404, an unknown request_id), `NOT_FOUND` (404, a path that does
not exist), `METHOD_NOT_ALLOWED` (405), `VALIDATION_ERROR` (422), `INTERNAL_SERVER_ERROR` (500),
`DOWNSTREAM_UNAVAILABLE` (503), `DOWNSTREAM_TIMEOUT` (504), unless an endpoint documents a more precise one.
`400` = the request or the document is unusable (do not retry unchanged). `503` / `504` are safe to retry.

**request_id.** Minted by the central orchestrator and carried through every stage. Where an endpoint has
no request_id of its own, the `X-Request-ID` request header is used (and echoed in the response header);
without it the service generates one.

**Asynchronous stages** (internal: only the orchestrator NPWP and the previous stage call them). `POST
.../jobs` answers `202` immediately and does the work in the background. The outcome is reported by a
callback to the central orchestrator (see *Webhooks*), and can be read at any time with
`GET .../jobs/{request_id}`, which the orchestrator NPWP's `GET /v1/extract-ocr/{request_id}` combines
over the three stages. Submitting the same request_id again is idempotent:
`202` with `duplicate: true`, the work is not repeated, unless the earlier attempt `FAILED` or has been
`PROCESSING` for longer than the job lease (`PIPELINE_JOB_LEASE_SECONDS`, 5 minutes by default: the
process running it died). A job still running when the service shuts down is reported `FAILED`. Because
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
    health: Mapping[str, ReadinessCheck] | None = None,
    backends_example: dict[str, str] | None = None,
    readiness_example: dict[str, str] | None = None,
    lifespan: Lifespan | None = None,
    entrypoint: bool = False,
) -> FastAPI:
    """Build the app of a service with everything every service has in common; see the module docstring.

    `entrypoint=True` marks the one service reachable from other namespaces (the orchestrator NPWP): its
    OpenAPI `servers` then start with the release's entry Service, the address the central orchestrator
    uses.

    `readiness` are the dependencies this pod cannot work without (`/ready`, the readiness probe); `health`
    the ones `/health` reports on, e.g. a database this service only writes to on a best-effort basis."""
    configure_logging(fmt=settings.effective_log_format, level=settings.log_level, service=service_name)

    servers: list[dict[str, Any]] = [{"url": "/", "description": "This host (where this page is served)"}]
    if settings.service_base_url:
        servers.append({"url": settings.service_base_url, "description": "Configured base URL"})
    if service_name:
        servers += _cluster_servers(service_name, settings.port, entrypoint=entrypoint)
        servers += [
            {"url": f"http://{service_name}:{settings.port}", "description": "The Docker Compose network"},
            {"url": f"http://127.0.0.1:{settings.port}", "description": "Local development"},
        ]
    app = FastAPI(
        title=title,
        version=version,
        description=description.rstrip() + API_CONVENTIONS,
        openapi_tags=[*tags, {"name": "Health", "description": "Health and readiness checks; no API key"}],
        servers=servers,
        lifespan=lifespan,
    )
    app.state.settings = settings
    # Named in every error answer (`pipeline_last_stage`), so a caller knows which service the error comes from.
    app.state.service_name = service_name
    if settings.auth_disabled:
        logging.getLogger(__name__).warning(
            "AUTH_DISABLED=true: X-API-Key is NOT checked on this service; only for local development"
        )

    app.add_middleware(MetricsMiddleware, service=service_name or title)
    app.add_middleware(RequestIdMiddleware)
    _register_exception_handlers(app)
    app.include_router(
        _health_router(
            version, backends or {}, readiness or {}, health or {}, backends_example or {}, readiness_example or {}
        )
    )
    for router in routers:
        app.include_router(router)
    if service_name:
        _name_the_service_in_error_examples(app, service_name)
    return app


# The Helm chart's names (deploy/helm): the release and its namespace are both `nilam-ocr-npwp`, each
# service is `<release>-<service>`, and `<release>` alone is the entry Service in front of the entry point.
RELEASE = "nilam-ocr-npwp"
_NAMESPACE = {
    "namespace": {
        "default": RELEASE,
        "description": "Namespace of the Helm release (deploy/helm/deploy.sh: nilam-ocr-npwp)",
    }
}


def _cluster_servers(service_name: str, port: int, *, entrypoint: bool) -> list[dict[str, Any]]:
    same_namespace = {
        "url": f"http://{RELEASE}-{service_name}.{{namespace}}.svc.cluster.local:{port}",
        "description": "GKE, from inside the release's namespace (the other services of this pipeline)",
        "variables": _NAMESPACE,
    }
    if not entrypoint:
        return [same_namespace]
    return [
        {
            "url": f"http://{RELEASE}.{{namespace}}.svc.cluster.local:{port}",
            "description": "GKE, from another namespace: the entry Service (the central orchestrator / gateway)",
            "variables": _NAMESPACE,
        },
        same_namespace,
    ]


def add_stage_callback_webhook(app: FastAPI, *, body_model: type, sent: str) -> None:
    """Document (as an OpenAPI webhook) the callback this service sends to the orchestrator."""

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
            "orchestrator can reconcile with the orchestrator NPWP's `GET /v1/extract-ocr/{request_id}` and "
            "should time a request out on its own. "
            "With `PIPELINE_OUTBOX` the callback is instead queued in the same transaction as the result and "
            "retried with back-off (up to 5 minutes apart) for up to `PIPELINE_OUTBOX_MAX_AGE_SECONDS` (24 h by "
            "default); a `4xx`, or that age, makes it a dead letter that stays in the table and is counted by "
            "`GET .../outbox`.\n\n"
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
        pass

    generate = app.openapi

    def openapi() -> dict[str, Any]:
        fresh = app.openapi_schema is None
        schema = generate()
        if fresh:
            schema["webhooks"]["stageCallback"]["post"]["responses"].pop("422", None)
            if "#/components/schemas/HTTPValidationError" not in json.dumps(schema["paths"]):
                for orphan in ("HTTPValidationError", "ValidationError"):
                    schema["components"]["schemas"].pop(orphan, None)
        return schema

    app.openapi = openapi  # ty: ignore[invalid-assignment]


def _health_router(
    version: str,
    backends: dict[str, str],
    readiness: Mapping[str, ReadinessCheck],
    health_checks: Mapping[str, ReadinessCheck],
    backends_example: dict[str, str],
    readiness_example: dict[str, str],
) -> APIRouter:
    router = APIRouter(tags=["Health"])
    healthy = {"status": "healthy", "version": version, "detail": None, "device": "cpu"}

    @router.get(
        "/health",
        response_model=HealthResponse,
        operation_id="getHealth",
        summary="Health check, including the database",
        description=(
            "Says whether the service can reach what it depends on: 200 `healthy` when every dependency it checks "
            "answers (the database, for a service that has one), 503 `unhealthy` with the cause in `detail` "
            "otherwise. Each check gives up after 3 s. Also lists the active implementations (`backends`). Not "
            "used by the Kubernetes probes: liveness and startup check the port, readiness is `/ready`. Does not "
            "require an API key."
        ),
        responses={
            200: success_examples(
                "Every dependency answers",
                healthy=("Healthy", {**healthy, "backends": backends_example}),
            ),
            503: {
                "model": HealthResponse,
                "description": "A dependency does not answer",
                "content": {
                    "application/json": {
                        "example": {
                            **healthy,
                            "status": "unhealthy",
                            "detail": "database unreachable",
                            "backends": backends_example,
                        }
                    }
                },
            },
        },
    )
    async def health():
        results = await _run_checks(health_checks, "health")
        failed = [name for name, state in results.items() if state != "ok"]
        if not failed:
            return {**healthy, "backends": backends}
        detail = ", ".join(f"{name} unreachable" for name in failed)
        return JSONResponse(
            status_code=503, content={**healthy, "status": "unhealthy", "detail": detail, "backends": backends}
        )

    @router.get(
        "/ready",
        response_model=ReadyResponse,
        operation_id="getReadiness",
        summary="Readiness probe",
        description=(
            "The readiness probe. Liveness is checked on the port, never on a dependency (a database blip "
            "must not make Kubernetes restart every pod). /ready says whether this pod can "
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
        checks = await _run_checks(readiness, "readiness")
        ok = all(state == "ok" for state in checks.values())
        return JSONResponse(
            status_code=200 if ok else 503, content={"status": "ready" if ok else "not_ready", "checks": checks}
        )

    @router.get(
        "/metrics",
        response_class=PlainTextResponse,
        operation_id="getMetrics",
        summary="Prometheus metrics",
        description=(
            "Prometheus text exposition of this process: `http_requests_total` and "
            "`http_request_duration_seconds` per route template, and for the pipeline stages "
            "`pipeline_jobs_total` (by outcome), `pipeline_job_duration_seconds`, "
            "`pipeline_stale_jobs_reclaimed_total`, `pipeline_outbox_deliveries_total` and the outbox backlog "
            "gauges (`pipeline_outbox_pending`, `_retrying`, `_dead_letters`, `_oldest_pending_seconds`). "
            "Scraped inside the cluster; does not require an API key."
        ),
        responses={200: {"description": "Metrics in the Prometheus text format", "content": {"text/plain": {}}}},
    )
    async def metrics(request: Request):
        return metrics_response(request)

    return router


def _error_body(
    request: Request, status_code: int, message: str, code: str, exc: Exception | None = None
) -> dict[str, Any]:
    """The error envelope, plus `pipeline_last_stage`: the service the error comes from. That is the service
    named by the exception (an error of another service this one called), else this service."""
    return {
        **envelope(status_code, message, None, get_request_id(request), errors=code),
        "pipeline_last_stage": getattr(exc, "service", None) or getattr(request.app.state, "service_name", None),
    }


def _name_the_service_in_error_examples(app: FastAPI, service_name: str) -> None:
    """The OpenAPI error examples as the handlers answer: with `pipeline_last_stage`, this service's name when
    the example does not name another one."""
    generate = app.openapi

    def label(example: Any) -> None:
        if isinstance(example, dict) and "status_code" in example and example.get("pipeline_last_stage") is None:
            example["pipeline_last_stage"] = service_name

    def openapi() -> dict[str, Any]:
        fresh = app.openapi_schema is None
        schema = generate()
        if fresh:
            for methods in schema.get("paths", {}).values():
                for operation in methods.values():
                    for code, response in operation.get("responses", {}).items():
                        if not code.startswith(("4", "5")):
                            continue
                        for media in response.get("content", {}).values():
                            label(media.get("example"))
                            for named in media.get("examples", {}).values():
                                label(named.get("value"))
        return schema

    app.openapi = openapi  # ty: ignore[invalid-assignment]


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        """Every domain error carries its HTTP status; this is the one place it becomes a response, so
        routes raise and never translate."""
        if exc.status_code >= 500:
            logger.error("%s %s -> %d: %s", request.method, request.url.path, exc.status_code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(request, exc.status_code, exc.message, error_code(exc.status_code, exc.code), exc),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(request, exc.status_code, str(exc.detail), error_code(exc.status_code)),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        """A bug or an error nobody translated: logged with its traceback, answered in the envelope without
        its details."""
        logger.exception("%s %s crashed", request.method, request.url.path, exc_info=exc)
        return JSONResponse(
            status_code=500, content=_error_body(request, 500, "Internal server error", error_code(500))
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        message = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in exc.errors())
        return JSONResponse(status_code=422, content=_error_body(request, 422, message, "VALIDATION_ERROR"))
