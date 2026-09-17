"""
Request ID Middleware
Adds request ID tracking to all requests for logging and tracing
"""

import uuid
from typing import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.logging import request_id_ctx


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware that ensures every request has a unique request ID.
    
    - Uses x-request-id header if provided by client
    - Generates a new UUID if not provided
    - Sets the request ID in context variable for logging
    - Returns the request ID in response header
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Get existing request ID or generate new one
        request_id = request.headers.get("x-request-id")
        if not request_id:
            request_id = uuid.uuid4().hex
        
        # Set context variable for automatic logging injection
        token = request_id_ctx.set(request_id)
        
        try:
            response = await call_next(request)
            # Add request ID to response headers for client tracking
            response.headers["x-request-id"] = request_id
            return response
        finally:
            # Always reset context to prevent leaks
            request_id_ctx.reset(token)