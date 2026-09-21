from functools import lru_cache

from ocr_common.config import PipelineSettings


class Settings(PipelineSettings):
    port: int = 8033

    scoring_model_path: str = "weights/trust_model.joblib"

    scoring_backend: str = "heuristic"

    scoring_approve_threshold: float = 0.8
    scoring_review_threshold: float = 0.5


@lru_cache
def get_settings() -> Settings:
    return Settings()
