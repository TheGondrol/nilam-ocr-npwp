import asyncio
import re

import httpx
from fastapi import FastAPI

from src.core.logging import request_id_ctx
from src.middleware.add_requestid import RequestIdMiddleware


def _create_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/ping")
    def ping() -> dict[str, str]:
        request_id = request_id_ctx.get() or ""
        return {"request_id": request_id}

    return app


async def _async_get(app: FastAPI, path: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.get(path, **kwargs)


def test_request_id_is_preserved_when_provided() -> None:
    app = _create_app()
    response = asyncio.get_event_loop().run_until_complete(
        _async_get(app, "/ping", headers={"x-request-id": "client-id-123"})
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "client-id-123"
    assert response.json()["request_id"] == "client-id-123"


def test_request_id_is_generated_when_missing() -> None:
    app = _create_app()
    response = asyncio.get_event_loop().run_until_complete(
        _async_get(app, "/ping")
    )

    assert response.status_code == 200
    request_id = response.headers["x-request-id"]
    assert re.fullmatch(r"[0-9a-f]{32}", request_id) is not None
    assert response.json()["request_id"] == request_id
