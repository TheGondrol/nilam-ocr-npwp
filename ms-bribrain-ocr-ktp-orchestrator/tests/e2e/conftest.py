"""
Pytest configuration for e2e tests.

These tests call the REAL downstream microservices to verify the full
orchestrator pipeline works correctly. Only the database and image
storage are mocked.

Supports two modes:
  1. In-process (default): uses httpx ASGITransport to call the app directly.
  2. Live server: set E2E_BASE_URL env var to hit a running orchestrator
     (e.g. during Docker build where the server is already up).

Requirements:
- All downstream microservices must be running and reachable.
- Service URLs and API keys must be configured via config.yaml or env vars.

Usage:
    # In-process (mocks DB):
    pytest tests/e2e/ -m e2e -v --no-cov

    # Against live server (DB mocking N/A — server handles it):
    E2E_BASE_URL=http://127.0.0.1:8000 pytest tests/e2e/ -m e2e -v --no-cov
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load .env so API_KEY is available to tests (same as server's load_dotenv)
from dotenv import load_dotenv
load_dotenv(project_root / ".env", override=False)

# ruff: noqa: E402

import aiohttp
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"


def _collect_images() -> list[Path]:
    """Collect all image files from tests/data/."""
    exts = {".jpeg", ".jpg", ".png"}
    images = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in exts)
    return images


ALL_IMAGES = _collect_images()


@pytest.fixture(params=ALL_IMAGES, ids=lambda p: p.name)
def image_path(request) -> Path:
    """Parametrised fixture yielding each image path in tests/data/."""
    return request.param


@pytest.fixture
def image_bytes(image_path: Path) -> bytes:
    """Read the image file into bytes."""
    return image_path.read_bytes()


@pytest.fixture
def image_content_type(image_path: Path) -> str:
    """Derive MIME type from file extension."""
    ext = image_path.suffix.lower()
    return {".jpeg": "image/jpeg", ".jpg": "image/jpeg", ".png": "image/png"}[ext]


# ---------------------------------------------------------------------------
# Mock database helpers (only used in in-process mode)
# ---------------------------------------------------------------------------

def _make_mock_record(request_id, status="completed", result=None, error_message=None):
    """Build a mock database record."""
    now = datetime.now(timezone.utc)
    record = MagicMock()
    record.request_id = request_id
    record.status = status
    record.result = result
    record.error_message = error_message
    record.created_at = now
    record.updated_at = now
    return record


# ---------------------------------------------------------------------------
# Client fixture
# ---------------------------------------------------------------------------

E2E_BASE_URL = os.getenv("E2E_BASE_URL", "")


@pytest.fixture
async def e2e_client():
    """
    Create an async httpx client for e2e tests.

    If E2E_BASE_URL is set, connects to a live running orchestrator.
    Otherwise, runs in-process with mocked DB and real downstream calls.
    """
    api_key = os.getenv("ORCHESTRATOR_SERVICE_API", "test-api-key")

    if E2E_BASE_URL:
        # ---- Live server mode ----
        async with AsyncClient(base_url=E2E_BASE_URL) as client:
            client.headers["X-API-Key"] = api_key
            yield client
    else:
        # ---- In-process mode (mock DB, real downstream services) ----
        from src.main import app
        import src.core.http_client as http_client_mod

        stored_results: dict[str, MagicMock] = {}

        async def fake_create(request_id):
            stored_results[request_id] = _make_mock_record(request_id, status="pending")

        async def fake_claim(request_id):
            rec = stored_results.get(request_id)
            if rec and rec.status == "pending":
                rec.status = "processing"
                return True
            stored_results[request_id] = _make_mock_record(request_id, status="processing")
            return True

        async def fake_update(request_id, status=None, result=None, error_message=None):
            stored_results[request_id] = _make_mock_record(
                request_id,
                status=str(status) if status else "completed",
                result=result,
                error_message=error_message,
            )

        async def fake_get(request_id):
            return stored_results.get(request_id)

        async def fake_insert_log(**kwargs):
            pass

        # Create real aiohttp session for downstream calls
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=10)
        aiohttp_session = aiohttp.ClientSession(connector=connector)

        original_client = http_client_mod._http_client
        http_client_mod._http_client = aiohttp_session

        try:
            with (
                patch.dict(os.environ, {"ORCHESTRATOR_SERVICE_API": api_key}),
                patch("src.api.routes.create_ocr_result", side_effect=fake_create),
                patch("src.api.routes.claim_ocr_result", side_effect=fake_claim),
                patch("src.api.routes.update_ocr_result", side_effect=fake_update),
                patch("src.api.routes.get_ocr_result", side_effect=fake_get),
                patch("src.api.routes.insert_log", side_effect=fake_insert_log),
                patch("src.services.manage_service.save_image", new_callable=AsyncMock),
                patch("src.services.manage_service.save_image_gcs", new_callable=AsyncMock),
            ):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    client.headers["X-API-Key"] = api_key
                    yield client
        finally:
            await aiohttp_session.close()
            http_client_mod._http_client = original_client
