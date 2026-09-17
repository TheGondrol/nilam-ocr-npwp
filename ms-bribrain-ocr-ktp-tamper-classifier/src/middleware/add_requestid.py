import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.logging import request_id_ctx


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

        # if not request_id:
        #     return JSONResponse(
        #         status_code=400,
        #         content={
        #             "error": "Missing required header: x-request-id"
        #         }
        #     )
        
        token = request_id_ctx.set(request_id)

        try:
            response = await call_next(request)
            response.headers["x-request-id"] = request_id
            return response
        finally:
            request_id_ctx.reset(token)
