"""
Pengaman konfigurasi: di luar ENVIRONMENT=local (yaitu dev / staging / production), salah konfigurasi harus gagal
START dengan pesan jelas, bukan berjalan diam-diam dengan perilaku berbahaya.
"""

import pytest
from pydantic import ValidationError

from ocr_common.config import BaseServiceSettings, PipelineSettings

DB = "postgresql+asyncpg://u:p@10.0.0.5:5432/db"
ORCH = "http://orkestrasi:8000"


def base(**overrides) -> BaseServiceSettings:
    return BaseServiceSettings(api_key="k", _env_file=None, **overrides)


def pipeline(**overrides) -> PipelineSettings:
    return PipelineSettings(api_key="k", _env_file=None, **overrides)


def test_environment_defaults_to_production(monkeypatch):
    """Lupa mengisi ENVIRONMENT harus jatuh ke mode yang paling ketat, bukan yang paling longgar."""
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    assert base().environment == "production"
    assert base().is_local is False


def test_dev_is_a_deployed_environment_not_the_laptop_mode():
    """Cluster GKE dev bernama "dev": ENVIRONMENT=dev di sana harus tetap menyalakan semua pengaman."""
    assert base(environment="dev").is_local is False
    with pytest.raises(ValidationError, match="DATABASE_URL, ORCHESTRATION_URL must be set when ENVIRONMENT=dev"):
        pipeline(environment="dev")


def test_unknown_environment_is_rejected():
    with pytest.raises(ValidationError):
        base(environment="prod")


@pytest.mark.parametrize("environment", ["dev", "staging", "production"])
def test_auth_disabled_is_refused_outside_local(environment):
    with pytest.raises(ValidationError, match="AUTH_DISABLED=true is only allowed with ENVIRONMENT=local"):
        base(environment=environment, auth_disabled=True)


def test_auth_disabled_is_allowed_locally():
    assert base(environment="local", auth_disabled=True).auth_disabled is True


def test_pipeline_locally_needs_nothing():
    settings = pipeline(environment="local")
    assert settings.database_url is None and settings.orchestration_url is None


def test_pipeline_outside_local_requires_database_and_orchestration():
    with pytest.raises(
        ValidationError, match="DATABASE_URL, ORCHESTRATION_URL must be set when ENVIRONMENT=production"
    ):
        pipeline(environment="production")
    with pytest.raises(ValidationError, match="ORCHESTRATION_URL must be set"):
        pipeline(environment="staging", database_url=DB)
    assert pipeline(environment="production", database_url=DB, orchestration_url=ORCH).environment == "production"


def test_empty_string_counts_as_missing():
    """ConfigMap dengan kunci kosong (`DATABASE_URL: ""`) sama berbahayanya dengan tanpa kunci."""
    with pytest.raises(ValidationError, match="DATABASE_URL must be set"):
        pipeline(environment="production", database_url="", orchestration_url=ORCH)


@pytest.mark.parametrize("url", ["http://127.0.0.1:8090", "http://localhost:8090", "http://[::1]:8090"])
def test_localhost_service_address_is_refused_outside_local(url):
    with pytest.raises(ValidationError, match="ORCHESTRATION_URL points to localhost"):
        pipeline(environment="production", database_url=DB, orchestration_url=url)


def test_localhost_database_is_allowed_for_the_sql_auth_proxy_sidecar():
    sidecar = "postgresql+asyncpg://u:p@127.0.0.1:5432/db"
    assert pipeline(environment="production", database_url=sidecar, orchestration_url=ORCH).database_url == sidecar


def test_mock_backend_guard():
    settings = base(environment="production")
    with pytest.raises(ValueError, match="GUARDRAILS_BACKEND=mock fabricates results"):
        settings.reject_mock_backend_outside_local(guardrails_backend="mock")
    settings.reject_mock_backend_outside_local(guardrails_backend="efficientnet")
    base(environment="local").reject_mock_backend_outside_local(guardrails_backend="mock")
