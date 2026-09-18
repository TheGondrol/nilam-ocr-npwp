import pytest

from ocr_common.testing import auth_headers, make_client, set_test_env

# Settings dibaca sekali lewat lru_cache saat src.main di-import, jadi env
# harus di-set SEBELUM import supaya .env lokal tidak ikut.
set_test_env(GUARDRAILS_BACKEND="mock")  # .env lokal boleh efficientnet; test memakai mock

from src.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    return make_client(app)


@pytest.fixture
def auth() -> dict[str, str]:
    return auth_headers()
