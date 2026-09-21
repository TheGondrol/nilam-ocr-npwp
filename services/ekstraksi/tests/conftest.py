import pytest

from ocr_common.testing import auth_headers, make_client, set_test_env

set_test_env(EKSTRAKSI_BACKEND="mock", DATABASE_URL="", ORCHESTRATION_URL="", AUTH_DISABLED="false")

from src.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    return make_client(app)


@pytest.fixture
def auth() -> dict[str, str]:
    return auth_headers()
