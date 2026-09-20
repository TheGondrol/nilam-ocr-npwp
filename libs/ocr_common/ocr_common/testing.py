"""Helper test yang sama untuk semua service (dipakai di tests/conftest.py masing-masing)."""

import os
import time

import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ocr_common.openapi import spec_text

TEST_API_KEY = "test-key"


def set_test_env(**extra: str) -> None:
    """Panggil SEBELUM import src.main: Settings dibaca sekali lewat lru_cache saat import."""
    os.environ["API_KEY"] = TEST_API_KEY
    # Test memakai backend mock dan storage in-memory; keduanya hanya sah di local.
    os.environ["ENVIRONMENT"] = "local"
    os.environ.update(extra)


def make_client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def auth_headers() -> dict[str, str]:
    return {"X-API-Key": TEST_API_KEY}


def image_upload(filename="npwp.jpg", content=b"\xff\xd8fake-jpeg-bytes", content_type="image/jpeg"):
    return {"file": (filename, content, content_type)}


def wait_for_job(client: TestClient, path: str, *, timeout: float = 5.0) -> dict:
    """Polling GET /v1/<tahap>/jobs/{request_id} sampai job keluar dari PROCESSING. -> `data`.

    Job jalan sebagai asyncio task di event loop app, jadi `client` harus
    dipakai sebagai context manager (`with make_client(app) as client`): tanpa
    itu TestClient membuat loop baru per request dan task-nya ikut dibatalkan.
    """
    deadline = time.monotonic() + timeout
    while True:
        data = client.get(path, headers=auth_headers()).json()["data"]
        if data["status"] != "PROCESSING" or time.monotonic() > deadline:
            return data
        time.sleep(0.02)


class RecordingCallback:
    """Pengganti OrchestrationCallback: mencatat notifikasi, tanpa jaringan."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def notify(self, request_id, stage, status, *, result=None, error_message=None) -> bool:
        self.calls.append(
            {
                "request_id": request_id,
                "stage": stage,
                "status": status,
                "result": result,
                "error_message": error_message,
            }
        )
        return True

    async def aclose(self) -> None:
        pass


class RecordingNextStage:
    """Pengganti NextStageClient: mencatat payload handoff; `error` diisi -> handoff gagal."""

    def __init__(self, error: Exception | None = None) -> None:
        self.payloads: list[dict] = []
        self.error = error

    async def submit(self, payload: dict) -> None:
        if self.error is not None:
            raise self.error
        self.payloads.append(payload)

    async def aclose(self) -> None:
        pass


def assert_openapi_up_to_date(app: FastAPI, path: str = "openapi.yaml") -> None:
    with open(path, encoding="utf-8") as handle:
        disk = yaml.safe_load(handle)
    live = yaml.safe_load(spec_text(app))
    assert live == disk, "openapi.yaml ketinggalan dari kode; jalankan `python -m ocr_common.openapi` di folder service"


def assert_error_responses_have_examples(app: FastAPI) -> None:
    spec = yaml.safe_load(spec_text(app))
    for path, methods in spec["paths"].items():
        if path in ("/health", "/ready"):  # bukan envelope: bentuk tetap untuk probe Kubernetes
            continue
        for operation in methods.values():
            for code, response in operation["responses"].items():
                if code.startswith(("4", "5")):
                    example = response.get("content", {}).get("application/json", {}).get("example")
                    assert example, f"{path} {code}: tidak punya contoh sendiri"
                    assert str(example["status_code"]) == code
