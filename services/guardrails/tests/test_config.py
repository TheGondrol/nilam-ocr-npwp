import pytest
from pydantic import ValidationError

from app.config import Settings


def test_mock_classifier_is_refused_outside_local():
    with pytest.raises(ValidationError, match="GUARDRAILS_BACKEND=mock fabricates results"):
        Settings(api_key="x", _env_file=None, environment="production")


def test_real_backends_are_accepted_outside_local():
    assert Settings(api_key="x", _env_file=None, environment="production", guardrails_backend="efficientnet")
    assert Settings(
        api_key="x",
        _env_file=None,
        environment="staging",
        guardrails_backend="remote",
        guardrails_model_url="http://guardrails-model:8081",
    )


def test_localhost_model_service_is_refused_outside_local():
    with pytest.raises(ValidationError, match="GUARDRAILS_MODEL_URL points to localhost"):
        Settings(
            api_key="x",
            _env_file=None,
            environment="production",
            guardrails_backend="remote",
            guardrails_model_url="http://localhost:8081",
        )


def test_settings_of_the_pipeline_are_ignored(monkeypatch):
    """The entry-point settings moved to the orchestrator NPWP; an environment that still sets them (an old
    ConfigMap, a local .env) must not stop guardrails from starting."""
    monkeypatch.setenv("EXTRACTION_SERVICE_URL", "http://127.0.0.1:8030")
    monkeypatch.setenv("PIPELINE_WAIT_SECONDS", "15")
    settings = Settings(api_key="x", _env_file=None, environment="production", guardrails_backend="efficientnet")
    assert not hasattr(settings, "pipeline_wait_seconds")


GCS_URI = "gs://gc-bribrain-dev-gcs-ocr-nilam-01/nilam-ocr-npwp/guardrails/20260923/best_model.pt"


def test_a_checkpoint_from_gcs_must_be_pinned_when_deployed():
    with pytest.raises(ValidationError, match="GUARDRAILS_MODEL_SHA256 must be set"):
        Settings(
            api_key="x",
            _env_file=None,
            environment="production",
            guardrails_backend="efficientnet",
            guardrails_model_gcs_uri=GCS_URI,
        )
    assert Settings(
        api_key="x",
        _env_file=None,
        environment="production",
        guardrails_backend="efficientnet",
        guardrails_model_gcs_uri=GCS_URI,
        guardrails_model_sha256="275ec6f7ec07f091395168fa175193cd1e6e88324f98b8cc21ee370fe101727a",
    )
