"""Pengaman konfigurasi khusus guardrails (yang umum: libs/ocr_common/tests/test_config.py)."""

import pytest
from pydantic import ValidationError

from src.core.config import Settings


def test_mock_classifier_is_refused_outside_local():
    """Default backend adalah mock, yang memvonis dari NAMA FILE. Tidak boleh sampai ke produksi karena lupa env."""
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
