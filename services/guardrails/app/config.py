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
    guardrails_reject_threshold: float | None = Field(None, gt=0, lt=1)

    # The reject threshold is owned by the central orchestrator: GET GUARDRAILS_THRESHOLD_URL +
    # GUARDRAILS_THRESHOLD_PATH answers {"reject_threshold": 0.5}, cached GUARDRAILS_THRESHOLD_CACHE_SECONDS.
    # Unset, unreachable or an invalid answer -> GUARDRAILS_REJECT_THRESHOLD, else the checkpoint's (0.5).
    guardrails_threshold_url: str | None = None
    guardrails_threshold_path: str = "/v1/thresholds/guardrails"
    guardrails_threshold_api_key: str | None = None
    guardrails_threshold_timeout_seconds: float = Field(2.0, gt=0)
    guardrails_threshold_cache_seconds: float = Field(60.0, ge=0)

    guardrails_document_policy: Literal["all", "majority"] = "all"
    guardrails_pdf_dpi: int = 150
    guardrails_max_pages: int = 20
    # A genuine NPWP upload is at most 2 pages (ML team, 23 Sep 2026); more is refused with 400 before the
    # model runs. Applies to both backends (the remote model reports n_pages).
    guardrails_max_document_pages: int = Field(2, ge=1)

    @model_validator(mode="after")
    def _guard_guardrails(self) -> Self:
        self.reject_mock_backend_outside_local(guardrails_backend=self.guardrails_backend)
        self.reject_localhost_outside_local(
            guardrails_model_url=self.guardrails_model_url, guardrails_threshold_url=self.guardrails_threshold_url
        )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
