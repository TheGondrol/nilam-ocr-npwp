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

    @model_validator(mode="after")
    def _guard_guardrails(self) -> Self:
        self.reject_mock_backend_outside_local(guardrails_backend=self.guardrails_backend)
        self.reject_localhost_outside_local(guardrails_model_url=self.guardrails_model_url)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
