from functools import lru_cache
from typing import Self

from pydantic import model_validator

from ocr_common.config import PipelineSettings


class Settings(PipelineSettings):
    port: int = 8030

    ekstraksi_backend: str = "mock"
    ekstraksi_ocr_url: str | None = None
    ekstraksi_ocr_api_key: str | None = None
    ekstraksi_ocr_timeout_seconds: float = 30.0

    guardrails_service_url: str = "http://127.0.0.1:8031"
    guardrails_api_key: str | None = None
    guardrails_timeout_seconds: float = 10.0
    structuring_service_url: str = "http://127.0.0.1:8032"
    structuring_api_key: str | None = None
    structuring_timeout_seconds: float = 10.0
    scoring_service_url: str = "http://127.0.0.1:8033"
    scoring_api_key: str | None = None
    scoring_timeout_seconds: float = 10.0

    @model_validator(mode="after")
    def _guard_ekstraksi(self) -> Self:
        self.reject_mock_backend_outside_local(ekstraksi_backend=self.ekstraksi_backend)
        self.reject_localhost_outside_local(structuring_service_url=self.structuring_service_url)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
