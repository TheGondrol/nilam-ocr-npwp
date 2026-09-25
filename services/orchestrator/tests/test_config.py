import pytest
from pydantic import ValidationError

from app.config import Settings

GUARDRAILS = "http://nilam-ocr-npwp-guardrails:8031"
EKSTRAKSI = "http://nilam-ocr-npwp-ekstraksi:8030"
STRUCTURING = "http://nilam-ocr-npwp-structuring:8032"
SCORING = "http://nilam-ocr-npwp-scoring:8033"
LOCALHOST = "http://127.0.0.1:9999"


def _deployed(
    guardrails: str = GUARDRAILS, ekstraksi: str = EKSTRAKSI, structuring: str = STRUCTURING, scoring: str = SCORING
) -> Settings:
    return Settings(
        api_key="x",
        _env_file=None,
        environment="production",
        pipeline_wait_seconds=0,
        guardrails_service_url=guardrails,
        ekstraksi_service_url=ekstraksi,
        structuring_service_url=structuring,
        scoring_service_url=scoring,
    )


def test_service_addresses_are_accepted_outside_local():
    assert _deployed()


@pytest.mark.parametrize(
    ("setting", "overrides"),
    [
        ("GUARDRAILS_SERVICE_URL", {"guardrails": LOCALHOST}),
        ("EKSTRAKSI_SERVICE_URL", {"ekstraksi": LOCALHOST}),
        ("STRUCTURING_SERVICE_URL", {"structuring": LOCALHOST}),
        ("SCORING_SERVICE_URL", {"scoring": LOCALHOST}),
    ],
)
def test_localhost_services_are_refused_outside_local(setting, overrides):
    """All four, also with PIPELINE_WAIT_SECONDS=0: GET /v1/extract-ocr/{request_id} reads the stages anyway."""
    with pytest.raises(ValidationError, match=f"{setting} points to localhost"):
        _deployed(**overrides)


def test_localhost_services_are_fine_locally():
    settings = Settings(api_key="x", _env_file=None, environment="local")
    assert settings.guardrails_service_url == "http://127.0.0.1:8031"
    assert settings.port == 8034


def test_skipping_guardrails_is_not_allowed_by_default():
    assert Settings(api_key="x", _env_file=None, environment="local").guardrails_skip_allowed is False


def test_pipeline_wait_defaults_to_15_seconds_and_can_be_changed():
    assert Settings(api_key="x", _env_file=None, environment="local").pipeline_wait_seconds == 15.0
    assert (
        Settings(api_key="x", _env_file=None, environment="local", pipeline_wait_seconds=8).pipeline_wait_seconds == 8
    )
    with pytest.raises(ValidationError, match="pipeline_wait_seconds"):
        # The invalid value is the point of this check.
        # pyrefly: ignore[bad-argument-type]
        Settings(api_key="x", _env_file=None, environment="local", pipeline_wait_seconds=-1)
