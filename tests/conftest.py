import os

import pytest
from fastapi.testclient import TestClient

API_KEY = "test-key"

# Settings dibaca sekali lewat lru_cache saat src.main di-import, jadi env
# harus di-set sebelum import dan cache dibersihkan supaya .env lokal tidak ikut.
os.environ["API_KEY"] = API_KEY

from src.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

from src.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth() -> dict[str, str]:
    return {"X-API-Key": API_KEY}


def image_upload(filename="npwp.jpg", content=b"\xff\xd8fake-jpeg-bytes", content_type="image/jpeg"):
    return {"file": (filename, content, content_type)}
