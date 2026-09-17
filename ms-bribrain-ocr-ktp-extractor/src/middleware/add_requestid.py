"""Request ID middleware for request tracking"""

import uuid
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.core.logging import request_id_ctx

logger = logging.getLogger(__name__)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware that ensures every request has a unique request ID.

    The request ID is:
    - Read from the 'x-request-id' header if provided
    - Generated as a UUID if not provided
    - Stored in context variable for logging
    - Added to response headers
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process request and inject request ID.

        Args:
            request: Incoming request
            call_next: Next middleware/handler in chain

        Returns:
            Response with request ID header
        """
        # Get or generate request ID
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

        # Set context variable for logging
        token = request_id_ctx.set(request_id)

        logger.debug(f"Processing request: {request.method} {request.url.path}")

        try:
            response = await call_next(request)
            response.headers["x-request-id"] = request_id
            return response
        finally:
            # Reset context variable
            request_id_ctx.reset(token)
