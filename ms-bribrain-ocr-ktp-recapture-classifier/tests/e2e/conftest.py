"""Pytest configuration for e2e tests.

Supports two modes:
  1. In-process (default): uses httpx ASGITransport with mocked model/DB.
  2. Live server: set E2E_BASE_URL env var to hit a running service
     (e.g. during Docker build where the server is already up).

Usage:
    # In-process:
    pytest tests/e2e/ -m e2e -v --no-cov

    # Against live server:
    E2E_BASE_URL=http://127.0.0.1:8030 pytest tests/e2e/ -m e2e -v --no-cov
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

E2E_BASE_URL = os.getenv("E2E_BASE_URL", "")


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


@pytest.fixture
async def e2e_client():
    """
    Create an async httpx client for e2e tests.

    If E2E_BASE_URL is set, connects to a live running service.
    Otherwise, runs in-process with mocked model + DB.
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
        from concurrent.futures import ThreadPoolExecutor
        import torch

        # The route reads get_provider().get("threshold"); lifespan (which calls
        # init_provider) is not run by ASGITransport, so initialize it here.
        import src.services.threshold_provider as _tp
        _tp._provider = None
        _tp.init_provider("recapture", {"threshold": 0.5})

        mock_model = MagicMock()
        mock_device = torch.device("cpu")

        def mock_transform(img):
            return torch.zeros((3, 224, 224))

        def mock_process_and_predict_sync(image_bytes):
            """Return a realistic recapture prediction."""
            return (1, 0.15, 0.85, (800, 600))

        with patch.dict(os.environ, {"API_KEY": api_key}):
            with patch("src.services.database_services.init_engine", new_callable=AsyncMock):
                with patch("src.services.database_services.dispose_engine", new_callable=AsyncMock):
                    with patch("src.api.routes.insert_log", new_callable=AsyncMock):
                        with patch("src.api.routes.get_model", return_value=mock_model):
                            with patch("src.api.routes.get_service_device", return_value=mock_device):
                                with patch("src.api.routes.process_and_predict_sync", side_effect=mock_process_and_predict_sync):
                                    with patch("src.services.minio_service.download_model_minio"):
                                        with patch("src.services.gcs_service.download_model_gcs"):
                                            with patch("src.services.recapture_service._model", mock_model):
                                                with patch("src.services.recapture_service._device", mock_device):
                                                    with patch("src.services.recapture_service._transform", mock_transform):
                                                        import src.api.routes as routes_mod
                                                        original_executor = routes_mod._executor
                                                        routes_mod._executor = ThreadPoolExecutor(max_workers=2)

                                                        from src.main import app
                                                        transport = httpx.ASGITransport(app=app)  # ty: ignore[invalid-argument-type]
                                                        async with httpx.AsyncClient(
                                                            transport=transport, base_url="http://test"
                                                        ) as client:
                                                            client.headers["X-API-Key"] = api_key
                                                            yield client

                                                        routes_mod._executor.shutdown(wait=False)
                                                        _tp._provider = None
                                                        routes_mod._executor = original_executor
