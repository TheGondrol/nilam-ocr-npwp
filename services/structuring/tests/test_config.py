from typing import Any

import pytest
from pydantic import ValidationError

from app.config import Settings

PROD: dict[str, Any] = {
    "environment": "production",
    "database_url": "postgresql+asyncpg://u:p@10.0.0.5:5432/db",
    "orchestration_url": "http://orkestrasi:8000",
    "scoring_service_url": "http://scoring:8033",
}


def test_production_configuration_is_accepted():
    assert Settings(api_key="x", _env_file=None, **PROD).environment == "production"


def test_default_localhost_next_stage_is_refused_outside_local():
    with pytest.raises(ValidationError, match="SCORING_SERVICE_URL points to localhost"):
        local_next_stage: dict[str, Any] = {**PROD, "scoring_service_url": "http://127.0.0.1:8033"}
        Settings(api_key="x", _env_file=None, **local_next_stage)
