from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, model_validator

from ocr_common.config import BaseServiceSettings


class Settings(BaseServiceSettings):
    port: int = 8031

    guardrails_backend: str = "mock"

    guardrails_model_url: str | None = None
    guardrails_model_api_key: str | None = None
    guardrails_model_timeout_seconds: float = 30.0

    guardrails_model_path: str = "weights/best_model.pt"
    guardrails_device: str = "cpu"
    guardrails_torch_threads: int | None = None
    guardrails_reject_threshold: float | None = None

    guardrails_document_policy: Literal["all", "majority"] = "all"
    guardrails_pdf_dpi: int = 150
    guardrails_max_pages: int = 20
    # A genuine NPWP upload is at most 2 pages (ML team, 23 Sep 2026); more is refused with 400 before the
    # model runs. Applies to both backends (the remote model reports n_pages).
    guardrails_max_document_pages: int = Field(2, ge=1)

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
    def _guard_guardrails(self) -> Self:
        self.reject_mock_backend_outside_local(guardrails_backend=self.guardrails_backend)
        self.reject_localhost_outside_local(
            guardrails_model_url=self.guardrails_model_url, ekstraksi_service_url=self.ekstraksi_service_url
        )
        if self.pipeline_wait_seconds > 0:
            self.reject_localhost_outside_local(
                structuring_service_url=self.structuring_service_url, scoring_service_url=self.scoring_service_url
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
