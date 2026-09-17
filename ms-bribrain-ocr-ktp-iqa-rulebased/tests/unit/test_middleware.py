"""Unit tests for src.middleware.add_requestid module."""

import httpx
import pytest
from fastapi import FastAPI

from src.core.logging import request_id_ctx
from src.middleware.add_requestid import RequestIdMiddleware


def _create_app():
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)  # type: ignore[arg-type]

    @app.get("/ping")
    async def ping():
        return {"request_id": request_id_ctx.get()}

    return app


class TestRequestIdMiddleware:
    async def test_generates_request_id(self):
        app = _create_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/ping")
        assert resp.status_code == 200
        assert "x-request-id" in resp.headers
        rid = resp.headers["x-request-id"]
        assert len(rid) == 36  # UUID
        assert resp.json()["request_id"] == rid

    async def test_preserves_provided_request_id(self):
        app = _create_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/ping", headers={"x-request-id": "my-custom-id"})
        assert resp.headers["x-request-id"] == "my-custom-id"
        assert resp.json()["request_id"] == "my-custom-id"

    async def test_different_requests_get_different_ids(self):
        app = _create_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r1 = await client.get("/ping")
            r2 = await client.get("/ping")
        assert r1.headers["x-request-id"] != r2.headers["x-request-id"]
