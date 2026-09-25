import pytest

from ocr_common.testing import auth_headers, make_client, set_test_env

set_test_env(EXTRACTION_BACKEND="mock", DATABASE_URL="", ORCHESTRATION_URL="", AUTH_DISABLED="false")

from app.config import get_settings  # noqa: E402
from app.dependencies import get_extraction_service  # noqa: E402
from app.main import app  # noqa: E402
from app.services.extraction_service import ExtractionService  # noqa: E402


@pytest.fixture(scope="session")
def client():
    return make_client(app)


@pytest.fixture
def auth() -> dict[str, str]:
    return auth_headers()


@pytest.fixture
def use_engine():
    """Serve /v1/extraction/extract with this OCR engine instead of the configured backend."""

    def _use(engine):
        app.dependency_overrides[get_extraction_service] = lambda: ExtractionService(engine, get_settings())

    yield _use
    app.dependency_overrides.pop(get_extraction_service, None)
