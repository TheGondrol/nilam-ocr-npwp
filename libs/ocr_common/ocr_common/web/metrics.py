"""Prometheus metrics of the HTTP layer, and the `/metrics` endpoint that exposes every metric of the
process (these and the pipeline's, see `ocr_common.pipeline.metrics`).

The labels use the route template (`/v1/extraction/jobs/{request_id}`), never the concrete path, so the
cardinality stays one series per endpoint."""

import time

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests handled, by route template and status code",
    ["service", "method", "path", "status"],
)
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "Time from receiving a request to sending its response",
    ["service", "method", "path"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 15, 30),
)

_UNMEASURED = {"/metrics", "/health", "/ready"}


class MetricsMiddleware(BaseHTTPMiddleware):
    """Counts and times every request except the probes and `/metrics` themselves."""

    def __init__(self, app, service: str) -> None:
        super().__init__(app)
        self._service = service

    async def dispatch(self, request, call_next):
        started = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", None) or request.url.path
            if path not in _UNMEASURED:
                HTTP_REQUESTS.labels(self._service, request.method, path, status).inc()
                HTTP_REQUEST_DURATION.labels(self._service, request.method, path).observe(time.perf_counter() - started)


def metrics_response(request: Request) -> Response:
    """The Prometheus text exposition of every metric registered in this process."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
