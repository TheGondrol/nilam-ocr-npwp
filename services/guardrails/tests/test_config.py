import pytest
from pydantic import ValidationError

from src.core.config import Settings

EKSTRAKSI = "http://nilam-ocr-npwp:8030"
STRUCTURING = "http://nilam-ocr-npwp:8032"
SCORING = "http://nilam-ocr-npwp:8033"


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
        structuring_service_url=STRUCTURING,
        scoring_service_url=SCORING,
    )
    assert Settings(
        api_key="x",
        _env_file=None,
        environment="staging",
        guardrails_backend="remote",
        guardrails_model_url="http://guardrails-model:8081",
        ekstraksi_service_url=EKSTRAKSI,
        structuring_service_url=STRUCTURING,
        scoring_service_url=SCORING,
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


def test_pipeline_wait_defaults_to_15_seconds_and_can_be_changed():
    assert Settings(api_key="x", _env_file=None, environment="local").pipeline_wait_seconds == 15.0
    assert (
        Settings(api_key="x", _env_file=None, environment="local", pipeline_wait_seconds=8).pipeline_wait_seconds == 8
    )
    with pytest.raises(ValidationError, match="pipeline_wait_seconds"):
        Settings(api_key="x", _env_file=None, environment="local", **{"pipeline_wait_seconds": -1})


def test_localhost_stage_services_are_refused_outside_local_only_while_waiting():
    with pytest.raises(ValidationError, match="STRUCTURING_SERVICE_URL, SCORING_SERVICE_URL points to localhost"):
        Settings(
            api_key="x",
            _env_file=None,
            environment="production",
            guardrails_backend="efficientnet",
            ekstraksi_service_url=EKSTRAKSI,
        )
    assert Settings(
        api_key="x",
        _env_file=None,
        environment="production",
        guardrails_backend="efficientnet",
        ekstraksi_service_url=EKSTRAKSI,
        pipeline_wait_seconds=0,
    )


def test_localhost_ekstraksi_service_is_fine_locally():
    assert Settings(api_key="x", _env_file=None, environment="local").ekstraksi_service_url == "http://127.0.0.1:8030"
