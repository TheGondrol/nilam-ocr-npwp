from functools import lru_cache
from typing import Any, Self

from pydantic import Field, model_validator

from ocr_common.config import PipelineSettings


class Settings(PipelineSettings):
    port: int = 8030

    extraction_backend: str = "mock"
    extraction_ocr_url: str | None = None
    # Sent as X-API-Key by both `remote` and `paddle`.
    extraction_ocr_api_key: str | None = None
    # `paddle` only: /ocr on the old VM (10.213.128.67:8070), /v1/ocr on the PP-OCRv6 service in the shared cluster.
    extraction_ocr_path: str = "/ocr"
    extraction_ocr_timeout_seconds: float = 30.0
    # `remote` only: extra form fields sent with every /v1/predict/json call (JSON object). The ML team's
    # OCR model will serve several document types in production and take its parameters per call.
    extraction_ocr_params: dict[str, Any] = {}
    # Overload protection of the calls to the OCR model (ocr_common.overload), all off by default:
    # - CONCURRENCY: at most that many calls in flight to the model per pod; the jobs beyond wait their turn
    #   (bounded by PIPELINE_MAX_QUEUE). The model VM slows down for everyone when it gets more than it can run.
    # - TOTAL_TIMEOUT_SECONDS: the most one call may take. EXTRACTION_OCR_TIMEOUT_SECONDS is per phase (send, then
    #   answer), so a slow upload plus a slow answer took up to twice as long (120 s at 60 s, stress test 15 Oct).
    # - BREAKER_FAILURES / BREAKER_RESET_SECONDS: after that many failures in a row, stop calling the model for that
    #   long (jobs fail at once with 503) instead of every job waiting for its own timeout.
    extraction_ocr_concurrency: int = Field(0, ge=0)
    extraction_ocr_total_timeout_seconds: float | None = Field(None, gt=0)
    extraction_ocr_breaker_failures: int = Field(0, ge=0)
    extraction_ocr_breaker_reset_seconds: float = Field(30.0, gt=0)

    structuring_service_url: str = "http://127.0.0.1:8032"
    structuring_api_key: str | None = None
    structuring_timeout_seconds: float = 10.0

    @model_validator(mode="after")
    def _guard_extraction(self) -> Self:
        self.reject_mock_backend_outside_local(extraction_backend=self.extraction_backend)
        self.reject_localhost_outside_local(structuring_service_url=self.structuring_service_url)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
