from functools import lru_cache
from typing import Any, Self

from pydantic import model_validator

from ocr_common.config import PipelineSettings


class Settings(PipelineSettings):
    port: int = 8030

    ekstraksi_backend: str = "mock"
    ekstraksi_ocr_url: str | None = None
    ekstraksi_ocr_api_key: str | None = None
    ekstraksi_ocr_timeout_seconds: float = 30.0
    # `remote` only: extra form fields sent with every /v1/predict/json call (JSON object). The ML team's
    # OCR model will serve several document types in production and take its parameters per call.
    ekstraksi_ocr_params: dict[str, Any] = {}

    structuring_service_url: str = "http://127.0.0.1:8032"
    structuring_api_key: str | None = None
    structuring_timeout_seconds: float = 10.0

    @model_validator(mode="after")
    def _guard_ekstraksi(self) -> Self:
        self.reject_mock_backend_outside_local(ekstraksi_backend=self.ekstraksi_backend)
        self.reject_localhost_outside_local(structuring_service_url=self.structuring_service_url)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
