"""Middleware for adding request ID to each request."""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.core.logging import request_id_ctx


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware to generate and inject request ID for each request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request and inject request ID.

        Generates a UUID for each request and stores it in context variable
        for access throughout the request lifecycle (including logs).

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            HTTP response with X-Request-ID header
        """
        # Use request ID from header if provided, otherwise generate one
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Set in context variable for logging
        token = request_id_ctx.set(request_id)

        try:
            # Process request
            response = await call_next(request)

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response
        finally:
            # Reset context variable
            request_id_ctx.reset(token)
