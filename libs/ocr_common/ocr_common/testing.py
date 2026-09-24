"""Helpers shared by the services' test suites: environment, test client, fixtures, recording doubles
for the orchestrator callback and the next stage, and the OpenAPI freshness checks.
"""

import os
import time

import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocr_common.pipeline.callbacks import stage_callback_body
from ocr_common.web.openapi import spec_text

TEST_API_KEY = "test-key"


def set_test_env(**extra: str) -> None:
    """Point the settings at a local test configuration before the app module is imported."""
    os.environ["API_KEY"] = TEST_API_KEY
    os.environ["ENVIRONMENT"] = "local"
    os.environ.update(extra)


def make_client(app: FastAPI) -> TestClient:
    """A `TestClient` that returns 500 responses instead of raising."""
    return TestClient(app, raise_server_exceptions=False)


def auth_headers() -> dict[str, str]:
    """The `X-API-Key` header of the test key."""
    return {"X-API-Key": TEST_API_KEY}


def image_upload(filename="npwp.jpg", content=b"\xff\xd8fake-jpeg-bytes", content_type="image/jpeg"):
    """A multipart `file` field with a small fake JPEG."""
    return {"file": (filename, content, content_type)}


def wait_for_job(client: TestClient, path: str, *, timeout: float = 5.0) -> dict:
    """Poll `GET path` until the job leaves `PROCESSING` or `timeout` passes; returns its `data`."""
    deadline = time.monotonic() + timeout
    while True:
        data = client.get(path, headers=auth_headers()).json()["data"]
        if data["status"] != "PROCESSING" or time.monotonic() > deadline:
            return data
        time.sleep(0.02)


class RecordingCallback:
    """A `StageCallback` that records every call instead of sending it."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def notify(self, request_id, stage, status, *, result=None, error_message=None, error_code=None) -> bool:
        """Record the callback, in the body shape a stage sends (`error_code` only when set)."""
        self.calls.append(
            stage_callback_body(
                request_id, stage, status, result=result, error_message=error_message, error_code=error_code
            )
        )
        return True

    async def send(self, body: dict) -> None:
        """Record the body."""
        self.calls.append(body)

    async def aclose(self) -> None:
        """Nothing to close."""
        pass


class RecordingNextStage:
    """A `NextStage` that records hand-offs, or raises `error` on each."""

    def __init__(self, error: Exception | None = None) -> None:
        self.payloads: list[dict] = []
        self.error = error

    async def submit(self, payload: dict) -> None:
        """Record the payload, or raise `error`."""
        if self.error is not None:
            raise self.error
        self.payloads.append(payload)

    async def send(self, payload: dict) -> None:
        """Same as `submit`."""
        await self.submit(payload)

    async def aclose(self) -> None:
        """Nothing to close."""
        pass


def assert_openapi_up_to_date(app: FastAPI, path: str = "openapi.yaml") -> None:
    """Fail when `openapi.yaml` on disk differs from the app's live schema."""
    with open(path, encoding="utf-8") as handle:
        disk = yaml.safe_load(handle)
    live = yaml.safe_load(spec_text(app))
    assert (
        live == disk
    ), "openapi.yaml ketinggalan dari kode; jalankan `python -m ocr_common.web.openapi` di folder service"


def assert_error_responses_have_examples(app: FastAPI) -> None:
    """Fail when a documented 4xx/5xx response lacks its own envelope example."""
    spec = yaml.safe_load(spec_text(app))
    for path, methods in spec["paths"].items():
        if path in ("/health", "/ready"):
            continue
        for operation in methods.values():
            for code, response in operation["responses"].items():
                if code.startswith(("4", "5")):
                    example = response.get("content", {}).get("application/json", {}).get("example")
                    assert example, f"{path} {code}: tidak punya contoh sendiri"
                    assert str(example["status_code"]) == code
