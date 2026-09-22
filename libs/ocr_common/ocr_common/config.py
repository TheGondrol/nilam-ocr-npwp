from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ocr_common.clients.fetch_url import UrlPolicy

Environment = Literal["local", "dev", "staging", "production"]
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
DEFAULT_JOB_LEASE_SECONDS = 300.0


class BaseServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    api_key: str = Field(..., min_length=1, validation_alias=AliasChoices("API_KEY", "MOCK_API_KEY"))
    auth_disabled: bool = False

    environment: Environment = "production"
    service_base_url: str | None = None
    port: int = 8000

    max_upload_bytes: int = 5 * 1024 * 1024
    allowed_content_types: list[str] = ["image/jpeg", "image/jpg", "image/png", "application/pdf"]
    file_url_allowed_hosts: str = ""
    field_confidence_threshold: float = Field(0.5, ge=0, le=1)

    @property
    def is_local(self) -> bool:
        return self.environment == "local"

    @property
    def file_url_policy(self) -> UrlPolicy:
        hosts = tuple(
            host.strip().lower().rstrip(".") for host in self.file_url_allowed_hosts.split(",") if host.strip()
        )
        return UrlPolicy(allowed_hosts=hosts, allow_private=self.is_local)

    def require_outside_local(self, **values: object) -> None:
        if self.is_local:
            return
        missing = [name.upper() for name, value in values.items() if not value]
        if missing:
            raise ValueError(
                f"{', '.join(missing)} must be set when ENVIRONMENT={self.environment} "
                "(set ENVIRONMENT=local for local development)"
            )

    def reject_localhost_outside_local(self, **urls: str | None) -> None:
        if self.is_local:
            return
        local = [name.upper() for name, url in urls.items() if url and urlsplit(url).hostname in _LOCAL_HOSTS]
        if local:
            raise ValueError(
                f"{', '.join(local)} points to localhost, which inside a pod is this service itself; "
                f"set the real address when ENVIRONMENT={self.environment}"
            )

    def reject_mock_backend_outside_local(self, **backends: str) -> None:
        if self.is_local:
            return
        mocked = [name.upper() for name, backend in backends.items() if backend == "mock"]
        if mocked:
            raise ValueError(
                f"{', '.join(mocked)}=mock fabricates results and is only allowed with ENVIRONMENT=local; "
                f"set a real backend when ENVIRONMENT={self.environment}"
            )

    @model_validator(mode="after")
    def _guard_auth(self) -> Self:
        if self.auth_disabled and not self.is_local:
            raise ValueError(
                f"AUTH_DISABLED=true is only allowed with ENVIRONMENT=local (got ENVIRONMENT={self.environment}): "
                "it turns off the X-API-Key check on every endpoint"
            )
        return self


class PipelineSettings(BaseServiceSettings):
    database_url: str | None = None

    orchestration_url: str | None = None
    orchestration_callback_path: str = "/v1/callbacks/stage"
    orchestration_api_key: str | None = None
    orchestration_timeout_seconds: float = 10.0

    pipeline_retry_attempts: int = 3
    pipeline_retry_delay_seconds: float = 0.5
    pipeline_drain_timeout_seconds: float = 30.0
    pipeline_job_lease_seconds: float = Field(DEFAULT_JOB_LEASE_SECONDS, gt=0)
    orchestration_outcome_table: str = ""
    pipeline_outbox: bool = False
    pipeline_outbox_interval_seconds: float = Field(1.0, gt=0)
    pipeline_outbox_batch: int = Field(20, gt=0)
    pipeline_outbox_lease_seconds: float = Field(30.0, gt=0)
    pipeline_outbox_max_backoff_seconds: float = Field(300.0, gt=0)
    pipeline_outbox_max_age_seconds: float = Field(24 * 3600.0, gt=0)
    pipeline_outbox_stale_after_seconds: float = Field(300.0, gt=0)

    @property
    def callbacks_enabled(self) -> bool:
        """Stage callbacks are only sent when the orchestrator exposes an endpoint for them. The other way
        to report the outcome is the orchestrator's own table (ORCHESTRATION_OUTCOME_TABLE)."""
        return bool(self.orchestration_url)

    @model_validator(mode="after")
    def _guard_pipeline(self) -> Self:
        self.require_outside_local(database_url=self.database_url)
        if not self.is_local and not self.orchestration_url and not self.orchestration_outcome_table:
            raise ValueError(
                f"ORCHESTRATION_URL or ORCHESTRATION_OUTCOME_TABLE must be set when ENVIRONMENT={self.environment}: "
                "without either, the orchestrator never learns how a request ended "
                "(set ENVIRONMENT=local for local development)"
            )
        self.reject_localhost_outside_local(orchestration_url=self.orchestration_url)
        return self
