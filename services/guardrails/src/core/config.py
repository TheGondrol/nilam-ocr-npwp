from functools import lru_cache
from typing import Literal, Self

from pydantic import model_validator

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

    ekstraksi_service_url: str = "http://127.0.0.1:8030"
    ekstraksi_api_key: str | None = None
    ekstraksi_timeout_seconds: float = 10.0
    pipeline_retry_attempts: int = 3
    pipeline_retry_delay_seconds: float = 0.5

    @model_validator(mode="after")
    def _guard_guardrails(self) -> Self:
        self.reject_mock_backend_outside_local(guardrails_backend=self.guardrails_backend)
        self.reject_localhost_outside_local(
            guardrails_model_url=self.guardrails_model_url, ekstraksi_service_url=self.ekstraksi_service_url
        )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
