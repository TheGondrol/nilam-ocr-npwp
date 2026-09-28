from functools import lru_cache

from ocr_common.config import PipelineSettings


class Settings(PipelineSettings):
    port: int = 8033

    scoring_model_path: str = "weights/trust_model.joblib"


@lru_cache
def get_settings() -> Settings:
    return Settings()
