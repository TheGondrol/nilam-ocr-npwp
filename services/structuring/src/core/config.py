from functools import lru_cache
from typing import Self

from pydantic import model_validator

from ocr_common.config import PipelineSettings


class Settings(PipelineSettings):
    # DATABASE_URL (structuring.jobs/structuring.results), ORCHESTRATION_*,
    # PIPELINE_*: lihat PipelineSettings.
    port: int = 8032

    # Backend. Nama harus terdaftar di src/models/structuring.py:
    #   npwp_rules - aturan regex + posisi dari ML engineer (src/vendor/npwp_rules); untuk kartu asli
    #   rule_based - regex berbasis label; hanya untuk teks berlabel (keluaran mock OCR)
    structuring_backend: str = "npwp_rules"
    # Hanya npwp_rules: tolak upload yang memuat dokumen lain (KTP/KK/Akta), CAPTCHA, screenshot
    # "Cek NPWP" DJP, atau lebih dari 2 halaman. Penolakan = 400 (sinkron) / job FAILED (pipeline).
    structuring_page_guardrails: bool = True

    # Tahap berikutnya di pipeline async. SCORING_API_KEY kosong = pakai API_KEY
    # service ini. Default menunjuk port lokal untuk dev bare; di compose
    # di-override ke nama service.
    scoring_service_url: str = "http://127.0.0.1:8033"
    scoring_api_key: str | None = None
    scoring_timeout_seconds: float = 10.0

    @model_validator(mode="after")
    def _guard_structuring(self) -> Self:
        self.reject_localhost_outside_local(scoring_service_url=self.scoring_service_url)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
