import pytest
from pydantic import ValidationError

from app.config import Settings

GCS_URI = "gs://gc-bribrain-dev-gcs-ocr-nilam-01/nilam-ocr-npwp/scoring/20260923/trust_model.joblib"
SHA256 = "1f27c2d90d90587c910f562556d8131bcdb28af010b2f4ec0c6c722950689073"


def _deployed(sha256: str | None) -> Settings:
    return Settings(
        api_key="x",
        _env_file=None,
        environment="production",
        orchestration_url="http://orch",
        database_url="postgresql+asyncpg://db.internal/x",
        scoring_model_gcs_uri=GCS_URI,
        scoring_model_sha256=sha256,
    )


def test_a_trust_model_from_gcs_must_be_pinned_when_deployed():
    with pytest.raises(ValidationError, match="SCORING_MODEL_SHA256 must be set"):
        _deployed(None)
    assert _deployed(SHA256).scoring_model_sha256 == SHA256


def test_a_gcs_model_uri_must_name_an_object():
    with pytest.raises(ValidationError, match="gs://bucket/path/to/file"):
        Settings(api_key="x", _env_file=None, environment="local", scoring_model_gcs_uri="gs://bucket-only")
