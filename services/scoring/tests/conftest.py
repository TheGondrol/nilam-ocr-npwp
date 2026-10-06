from pathlib import Path

import pytest

from ocr_common.testing import auth_headers, make_client, set_test_env

# The models live in GCS and are baked into the image at build (scripts/fetch_weights.py); the tests use this copy
# of the trust model (21 Sep 2026 retrain), so they run without GCS credentials.
TRUST_MODEL = Path(__file__).parent / "fixtures" / "trust_model.joblib"

set_test_env(DATABASE_URL="", ORCHESTRATION_URL="", AUTH_DISABLED="false", SCORING_MODEL_PATH=str(TRUST_MODEL))

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    return make_client(app)


@pytest.fixture
def auth() -> dict[str, str]:
    return auth_headers()
