from functools import lru_cache
from typing import Self

from pydantic import model_validator

from ocr_common.config import PipelineSettings


class Settings(PipelineSettings):
    port: int = 8033

    scoring_model_path: str = "weights/trust_model.joblib"
    # The trust model from GCS instead (downloaded at start, see ocr_common/clients/models.py), pinned by SHA-256.
    scoring_model_gcs_uri: str | None = None
    scoring_model_sha256: str | None = None

    @model_validator(mode="after")
    def _guard_scoring(self) -> Self:
        self.require_model_pin("scoring_model_gcs_uri", self.scoring_model_gcs_uri, self.scoring_model_sha256)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
