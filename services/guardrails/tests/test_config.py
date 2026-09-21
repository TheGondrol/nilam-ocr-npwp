import pytest
from pydantic import ValidationError

from src.core.config import Settings

EKSTRAKSI = "http://nilam-ocr-npwp:8030"


def test_mock_classifier_is_refused_outside_local():
    with pytest.raises(ValidationError, match="GUARDRAILS_BACKEND=mock fabricates results"):
        Settings(api_key="x", _env_file=None, environment="production", ekstraksi_service_url=EKSTRAKSI)


def test_real_backends_are_accepted_outside_local():
    assert Settings(
        api_key="x",
        _env_file=None,
        environment="production",
        guardrails_backend="efficientnet",
        ekstraksi_service_url=EKSTRAKSI,
    )
    assert Settings(
        api_key="x",
        _env_file=None,
        environment="staging",
        guardrails_backend="remote",
        guardrails_model_url="http://guardrails-model:8081",
        ekstraksi_service_url=EKSTRAKSI,
    )


def test_localhost_model_service_is_refused_outside_local():
    with pytest.raises(ValidationError, match="GUARDRAILS_MODEL_URL points to localhost"):
        Settings(
            api_key="x",
            _env_file=None,
            environment="production",
            guardrails_backend="remote",
            guardrails_model_url="http://localhost:8081",
            ekstraksi_service_url=EKSTRAKSI,
        )


def test_localhost_ekstraksi_service_is_refused_outside_local():
    with pytest.raises(ValidationError, match="EKSTRAKSI_SERVICE_URL points to localhost"):
        Settings(api_key="x", _env_file=None, environment="production", guardrails_backend="efficientnet")


def test_localhost_ekstraksi_service_is_fine_locally():
    assert Settings(api_key="x", _env_file=None, environment="local").ekstraksi_service_url == "http://127.0.0.1:8030"
