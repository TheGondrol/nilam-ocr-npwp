import pytest
from pydantic import ValidationError

from app.config import Settings

GCS_URI = "gs://gc-bribrain-dev-gcs-ocr-nilam-01/nilam-ocr-npwp/scoring/trust_model.joblib"


def test_a_trust_model_from_gcs_needs_no_pinned_sha256_when_deployed():
    settings = Settings(
        api_key="x",
        _env_file=None,
        environment="production",
        orchestration_url="http://orch",
        database_url="postgresql+asyncpg://db.internal/x",
        scoring_model_gcs_uri=GCS_URI,
    )
    assert settings.scoring_model_sha256 is None


def test_a_gcs_model_uri_must_name_an_object():
    with pytest.raises(ValidationError, match="gs://bucket/path/to/file"):
        Settings(api_key="x", _env_file=None, environment="local", scoring_model_gcs_uri="gs://bucket-only")
