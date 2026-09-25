from functools import lru_cache
from typing import Self

from pydantic import Field, model_validator

from ocr_common.config import BaseServiceSettings


class Settings(BaseServiceSettings):
    port: int = 8034

    # The guardrails model service: every document is judged there before it enters the pipeline.
    guardrails_service_url: str = "http://127.0.0.1:8031"
    guardrails_api_key: str | None = None
    guardrails_timeout_seconds: float = 20.0
    # Whether a request may skip the guardrails model with `skip_guardrails=true`. Off: such a request is
    # refused with 403, so an API key alone does not bypass guardrails. The file checks below always run.
    guardrails_skip_allowed: bool = False

    # A PDF with more pages is refused with 400 here, before guardrails or any stage runs (a genuine NPWP
    # upload is at most 2 pages, ML team 23 Sep 2026). MAX_UPLOAD_BYTES (413) is checked here too.
    max_document_pages: int = Field(2, ge=1)

    ekstraksi_service_url: str = "http://127.0.0.1:8030"
    ekstraksi_api_key: str | None = None
    ekstraksi_timeout_seconds: float = 10.0
    structuring_service_url: str = "http://127.0.0.1:8032"
    structuring_api_key: str | None = None
    structuring_timeout_seconds: float = 10.0
    scoring_service_url: str = "http://127.0.0.1:8033"
    scoring_api_key: str | None = None
    scoring_timeout_seconds: float = 10.0
    pipeline_retry_attempts: int = 3
    pipeline_retry_delay_seconds: float = 0.5
    pipeline_wait_seconds: float = Field(15.0, ge=0)
    pipeline_poll_interval_seconds: float = Field(0.5, gt=0)

    @model_validator(mode="after")
    def _guard_orchestrator(self) -> Self:
        # All four always: GET /v1/extract-ocr/{request_id} reads the stages even when POST does not wait.
        self.reject_localhost_outside_local(
            guardrails_service_url=self.guardrails_service_url,
            ekstraksi_service_url=self.ekstraksi_service_url,
            structuring_service_url=self.structuring_service_url,
            scoring_service_url=self.scoring_service_url,
        )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
