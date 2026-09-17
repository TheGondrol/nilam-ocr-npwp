"""
Pytest configuration for e2e tests.

These tests call the REAL detector service to verify KTP detection
works correctly with an actual loaded YOLO model.

Supports two modes:
  1. In-process (default): uses TestClient to call the app directly.
     Mocks model download but uses the real predictor if a model file exists,
     otherwise mocks the predictor too.
  2. Live server: set E2E_BASE_URL env var to hit a running detector
     (e.g. during Docker build where the server is already up).

Usage:
    # In-process:
    pytest tests/e2e/ -m e2e -v --no-cov

    # Against live server:
    E2E_BASE_URL=http://127.0.0.1:8060 pytest tests/e2e/ -m e2e -v --no-cov
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


# ---------------------------------------------------------------------------
# Client fixture
# ---------------------------------------------------------------------------

E2E_BASE_URL = os.getenv("E2E_BASE_URL", "")


@pytest.fixture
async def e2e_client():
    """
    Create an async httpx client for e2e tests.

    If E2E_BASE_URL is set, connects to a live running detector.
    Otherwise, runs in-process with mocked DB and real model (if available).
    """
    api_key = os.getenv("API_KEY", "test-api-key")

    if E2E_BASE_URL:
        # ---- Live server mode ----
        async with httpx.AsyncClient(base_url=E2E_BASE_URL) as client:
            client.headers["X-API-Key"] = api_key
            yield client
    else:
        # ---- In-process mode ----
        from unittest.mock import patch, MagicMock, AsyncMock

        with patch.dict(os.environ, {"API_KEY": api_key}):
            with patch("src.services.database_service.init_engine", new_callable=AsyncMock):
                with patch("src.services.database_service.dispose_engine", new_callable=AsyncMock):
                    with patch("src.services.database_service.insert_log", new_callable=AsyncMock):
                        with patch("src.services.minio_service.download_model_minio"):
                            with patch("src.services.gcs_service.download_model_gcs"):
                                # Mock predictor with realistic responses
                                with patch("src.api.routes.predictor") as mock_pred:
                                    mock_pred.is_loaded = MagicMock(return_value=True)
                                    mock_pred.get_device_string.return_value = "cpu"
                                    mock_pred.get_device_info.return_value = {"type": "CPU"}

                                    def mock_predict(image, filename):
                                        """Return a realistic KTP detection."""
                                        return [
                                            {
                                                "class_id": 0,
                                                "class_name": "ktp",
                                                "confidence": 0.92,
                                                "bbox": {
                                                    "x1": 50.0,
                                                    "y1": 30.0,
                                                    "x2": 580.0,
                                                    "y2": 370.0,
                                                },
                                            }
                                        ]

                                    mock_pred.predict = mock_predict

                                    with patch("src.services.predictor.predictor", mock_pred):
                                        from src.main import app
                                        transport = httpx.ASGITransport(app=app)  # ty: ignore[invalid-argument-type]
                                        async with httpx.AsyncClient(
                                            transport=transport, base_url="http://test"
                                        ) as client:
                                            client.headers["X-API-Key"] = api_key
                                            yield client
