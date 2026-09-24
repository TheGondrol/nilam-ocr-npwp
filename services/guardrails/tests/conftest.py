import pytest

from ocr_common.testing import auth_headers, make_client, set_test_env

set_test_env(GUARDRAILS_BACKEND="mock", AUTH_DISABLED="false")

from app.config import get_settings  # noqa: E402
from app.dependencies import get_guardrails_service  # noqa: E402
from app.main import app  # noqa: E402
from app.services.guardrails_service import GuardrailsService  # noqa: E402


@pytest.fixture(scope="session")
def client():
    return make_client(app)


@pytest.fixture
def auth() -> dict[str, str]:
    return auth_headers()


@pytest.fixture
def use_classifier():
    """Serve the routes with this classifier instead of the configured backend."""

    def _use(classifier, **settings):
        configured = get_settings().model_copy(update=settings) if settings else get_settings()
        app.dependency_overrides[get_guardrails_service] = lambda: GuardrailsService(classifier, configured)

    yield _use
    app.dependency_overrides.pop(get_guardrails_service, None)
