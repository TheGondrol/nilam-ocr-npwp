import pytest

from ocr_common.testing import auth_headers, make_client, set_test_env

set_test_env(EKSTRAKSI_BACKEND="mock", DATABASE_URL="", ORCHESTRATION_URL="", AUTH_DISABLED="false")

from app.config import get_settings  # noqa: E402
from app.dependencies import get_ekstraksi_service  # noqa: E402
from app.main import app  # noqa: E402
from app.services.ekstraksi_service import EkstraksiService  # noqa: E402


@pytest.fixture(scope="session")
def client():
    return make_client(app)


@pytest.fixture
def auth() -> dict[str, str]:
    return auth_headers()


@pytest.fixture
def use_engine():
    """Serve /v1/ekstraksi/extract with this OCR engine instead of the configured backend."""

    def _use(engine):
        app.dependency_overrides[get_ekstraksi_service] = lambda: EkstraksiService(engine, get_settings())

    yield _use
    app.dependency_overrides.pop(get_ekstraksi_service, None)
