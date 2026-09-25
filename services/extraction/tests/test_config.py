from typing import Any

import pytest
from pydantic import ValidationError

from app.config import Settings

PROD: dict[str, Any] = {
    "environment": "production",
    "database_url": "postgresql+asyncpg://u:p@10.0.0.5:5432/db",
    "orchestration_url": "http://orkestrasi:8000",
    "extraction_backend": "paddle",
    "extraction_ocr_url": "http://10.213.128.67:8070",
    "structuring_service_url": "http://structuring:8032",
}


def settings(**overrides) -> Settings:
    return Settings(api_key="x", _env_file=None, **{**PROD, **overrides})


def test_production_configuration_is_accepted():
    assert settings().environment == "production"


def test_mock_ocr_is_refused_outside_local():
    with pytest.raises(ValidationError, match="EXTRACTION_BACKEND=mock fabricates results"):
        settings(extraction_backend="mock")


def test_default_localhost_next_stage_is_refused_outside_local():
    with pytest.raises(ValidationError, match="STRUCTURING_SERVICE_URL points to localhost"):
        settings(structuring_service_url="http://127.0.0.1:8032")


def test_local_keeps_the_laptop_defaults():
    local = Settings(api_key="x", _env_file=None, environment="local")
    assert local.extraction_backend == "mock"
    assert local.structuring_service_url == "http://127.0.0.1:8032"
