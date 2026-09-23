"""Settings of every service, read from the environment (and `.env` locally) with pydantic-settings.

`BaseServiceSettings` is what all four services share; `PipelineSettings` adds what the three
asynchronous stages need. Guards on `ENVIRONMENT` make a deployed service refuse to start with a
laptop-only configuration (mock backends, auth disabled, localhost addresses).
"""

from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ocr_common.clients.fetch_url import UrlPolicy

Environment = Literal["local", "dev", "staging", "production"]
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
DEFAULT_JOB_LEASE_SECONDS = 300.0
DEFAULT_MAX_UPLOAD_BYTES = int(2.5 * 1024 * 1024)


class BaseServiceSettings(BaseSettings):
    """Settings shared by all services: API keys, environment, upload limits, `file_url` policy, logging."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    api_key: str = Field(..., min_length=1)
    api_keys: str = ""
    auth_disabled: bool = False

    log_format: Literal["json", "text"] | None = None
    log_level: str = "INFO"

    environment: Environment = "production"
    service_base_url: str | None = None
    port: int = 8000

    # 2,5 MB: an NPWP document is 1-2 MB, a few reach 2.1 MB (ML team, 23 Sep 2026); larger uploads are
    # refused with 413 before any model runs.
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    allowed_content_types: list[str] = ["image/jpeg", "image/jpg", "image/png", "application/pdf"]
    file_url_allowed_hosts: str = ""
    field_confidence_threshold: float = Field(0.5, ge=0, le=1)

    @property
    def is_local(self) -> bool:
        """True for `ENVIRONMENT=local`: the laptop mode where the safety guards are off."""
        return self.environment == "local"

    @property
    def accepted_api_keys(self) -> tuple[str, ...]:
        """Keys a caller may present: `API_KEY` (also the key this service sends to the others) plus the
        comma-separated `API_KEYS`. Rotation: add the new key to `API_KEYS` everywhere, move the callers,
        make it `API_KEY`, drop the old one."""
        extra = tuple(key.strip() for key in self.api_keys.split(",") if key.strip())
        return (self.api_key, *(key for key in extra if key != self.api_key))

    @property
    def effective_log_format(self) -> Literal["json", "text"]:
        """`LOG_FORMAT` when set; otherwise text on a laptop and JSON (for Cloud Logging) when deployed."""
        return self.log_format or ("text" if self.is_local else "json")

    @property
    def file_url_policy(self) -> UrlPolicy:
        """The `UrlPolicy` for `file_url` downloads built from `FILE_URL_ALLOWED_HOSTS` and the environment."""
        hosts = tuple(
            host.strip().lower().rstrip(".") for host in self.file_url_allowed_hosts.split(",") if host.strip()
        )
        return UrlPolicy(allowed_hosts=hosts, allow_private=self.is_local)

    def require_outside_local(self, **values: object) -> None:
        """Raises unless every given value is set, when not local; used by the subclasses' validators."""
        if self.is_local:
            return
        missing = [name.upper() for name, value in values.items() if not value]
        if missing:
            raise ValueError(
                f"{', '.join(missing)} must be set when ENVIRONMENT={self.environment} "
                "(set ENVIRONMENT=local for local development)"
            )

    def reject_localhost_outside_local(self, **urls: str | None) -> None:
        """Raises when a service URL points to localhost outside local: inside a pod that is the service itself."""
        if self.is_local:
            return
        local = [name.upper() for name, url in urls.items() if url and urlsplit(url).hostname in _LOCAL_HOSTS]
        if local:
            raise ValueError(
                f"{', '.join(local)} points to localhost, which inside a pod is this service itself; "
                f"set the real address when ENVIRONMENT={self.environment}"
            )

    def reject_mock_backend_outside_local(self, **backends: str) -> None:
        """Raises when a backend is `mock` outside local: a mock fabricates results."""
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
    """Settings of the three stage services: database, orchestrator callback or outcome table, job
    lease, outbox, stale-job reaper, hand-off by reference.
    """

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
    orchestration_api_events_table: str = ""
    pipeline_outbox: bool = False
    pipeline_outbox_interval_seconds: float = Field(1.0, gt=0)
    pipeline_outbox_batch: int = Field(20, gt=0)
    pipeline_outbox_lease_seconds: float = Field(30.0, gt=0)
    pipeline_outbox_max_backoff_seconds: float = Field(300.0, gt=0)
    pipeline_outbox_max_age_seconds: float = Field(24 * 3600.0, gt=0)
    pipeline_outbox_stale_after_seconds: float = Field(300.0, gt=0)
    pipeline_handoff_by_reference: bool = False
    pipeline_stale_jobs: bool = True
    pipeline_stale_job_interval_seconds: float = Field(30.0, gt=0)
    pipeline_stale_job_batch: int = Field(10, gt=0)

    @property
    def callbacks_enabled(self) -> bool:
        """Stage callbacks are only sent when the orchestrator exposes an endpoint for them. The other way
        to report the outcome is the orchestrator's own tables (ORCHESTRATION_OUTCOME_TABLE,
        ORCHESTRATION_API_EVENTS_TABLE)."""
        return bool(self.orchestration_url)

    @model_validator(mode="after")
    def _guard_pipeline(self) -> Self:
        self.require_outside_local(database_url=self.database_url)
        reports_outcome = (
            self.orchestration_url or self.orchestration_outcome_table or self.orchestration_api_events_table
        )
        if not self.is_local and not reports_outcome:
            raise ValueError(
                "ORCHESTRATION_URL, ORCHESTRATION_OUTCOME_TABLE or ORCHESTRATION_API_EVENTS_TABLE must be set when "
                f"ENVIRONMENT={self.environment}: without one, the orchestrator never learns how a request ended "
                "(set ENVIRONMENT=local for local development)"
            )
        self.reject_localhost_outside_local(orchestration_url=self.orchestration_url)
        if self.pipeline_handoff_by_reference and not self.database_url:
            raise ValueError(
                "PIPELINE_HANDOFF_BY_REFERENCE=true needs DATABASE_URL: the next stage reads this stage's "
                "result from the shared database instead of the hand-off body"
            )
        return self
