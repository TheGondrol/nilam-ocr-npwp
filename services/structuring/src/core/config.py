from functools import lru_cache

from ocr_common.config import PipelineSettings


class Settings(PipelineSettings):
    # DATABASE_URL (structuring.jobs/structuring.results), ORCHESTRATION_*,
    # PIPELINE_*: lihat PipelineSettings.
    port: int = 8032

    # Backend (implementasi model). Nama harus terdaftar di src/models/structuring.py.
    structuring_backend: str = "rule_based"

    # Tahap berikutnya di pipeline async. SCORING_API_KEY kosong = pakai API_KEY
    # service ini. Default menunjuk port lokal untuk dev bare; di compose
    # di-override ke nama service.
    scoring_service_url: str = "http://127.0.0.1:8033"
    scoring_api_key: str | None = None
    scoring_timeout_seconds: float = 10.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
