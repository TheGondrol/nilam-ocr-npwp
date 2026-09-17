"""
Pytest configuration for e2e tests.

These tests call the REAL IQA rule-based service to verify image quality
checks (blur, glare, rotation, confidence) work correctly.

Supports two modes:
  1. In-process (default): uses httpx ASGITransport with mocked DB.
  2. Live server: set E2E_BASE_URL env var to hit a running service
     (e.g. during Docker build where the server is already up).

Usage:
    # In-process:
    pytest tests/e2e/ -m e2e -v --no-cov

    # Against live server:
    E2E_BASE_URL=http://127.0.0.1:8070 pytest tests/e2e/ -m e2e -v --no-cov
"""

import json
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load .env so API_KEY is available to tests (same as server's load_dotenv)
from dotenv import load_dotenv
load_dotenv(project_root / ".env", override=False)

import pytest
import httpx


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


@pytest.fixture
def sample_ocr_result() -> str:
    """Return a valid OCR result JSON string for testing."""
    data = [
        [[[10, 10], [150, 10], [150, 30], [10, 30]], ("PROVINSI DKI JAKARTA", 0.95)],
        [[[10, 40], [150, 40], [150, 60], [10, 60]], ("KOTA JAKARTA PUSAT", 0.90)],
        [[[10, 70], [80, 70], [80, 90], [10, 90]], ("NIK", 0.85)],
        [[[90, 70], [200, 70], [200, 90], [90, 90]], ("3175012345678901", 0.92)],
        [[[10, 100], [80, 100], [80, 120], [10, 120]], ("Nama", 0.88)],
    ]
    return json.dumps(data)


# ---------------------------------------------------------------------------
# Client fixture
# ---------------------------------------------------------------------------

E2E_BASE_URL = os.getenv("E2E_BASE_URL", "")


@pytest.fixture
async def e2e_client():
    """
    Create an async httpx client for e2e tests.

    If E2E_BASE_URL is set, connects to a live running service.
    Otherwise, runs in-process with mocked DB.
    """
    api_key = os.getenv("API_KEY", "test-api-key")

    if E2E_BASE_URL:
        # ---- Live server mode ----
        async with httpx.AsyncClient(base_url=E2E_BASE_URL) as client:
            client.headers["X-API-Key"] = api_key
            yield client
    else:
        # ---- In-process mode ----
        from unittest.mock import patch, AsyncMock

        mock_quality_result = {
            "low_confidence": False,
            "is_blurry": False,
            "is_glare": False,
            "is_rotated": False,
        }

        with patch.dict(os.environ, {"API_KEY": api_key}):
            with patch("src.services.database_service.init_engine", new_callable=AsyncMock):
                with patch("src.services.database_service.dispose_engine", new_callable=AsyncMock):
                    with patch("src.services.database_service.insert_log", new_callable=AsyncMock):
                        with patch("src.api.routes.insert_log", new_callable=AsyncMock):
                            with patch(
                                "src.api.routes.image_quality",
                                return_value=mock_quality_result,
                            ):
                                from src.main import app
                                transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
                                async with httpx.AsyncClient(
                                    transport=transport, base_url="http://test"
                                ) as client:
                                    client.headers["X-API-Key"] = api_key
                                    yield client
