from functools import lru_cache
from typing import Self

from pydantic import model_validator

from ocr_common.config import PipelineSettings


class Settings(PipelineSettings):
    # Service ini yang memegang kontrak orkestrator lama (generate-request-id ->
    # extract-ocr -> get-ocr-result), jadi ia menempati slot ocr-npwp: 8030.
    # DATABASE_URL, ORCHESTRATION_*, PIPELINE_*: lihat PipelineSettings. Di
    # service ini DATABASE_URL dipakai dua tabel: ocr.jobs/ocr.results (alur
    # async) dan ocr_npwp_requests (kontrak lama).
    port: int = 8030

    # Backend OCR. Nama harus terdaftar di src/models/ekstraksi.py.
    ekstraksi_backend: str = "mock"
    #   remote - service model ekstraksi milik ML engineer (POST {url}/v1/predict/json, X-API-Key)
    #   paddle - API lama service PaddleOCR (POST {url}/ocr, tanpa auth)
    #   mock   - baris NPWP deterministik, untuk test
    # URL wajib diisi untuk `remote` dan `paddle`. API key hanya dipakai `remote`
    # dan dikirim ke SERVICE MODEL (bukan API_KEY service ini, yang dipakai pemanggil kita).
    ekstraksi_ocr_url: str | None = None
    ekstraksi_ocr_api_key: str | None = None
    ekstraksi_ocr_timeout_seconds: float = 30.0

    # Service lain. Alur async hanya memakai STRUCTURING_* (handoff ke tahap
    # berikutnya); kontrak lama extract-ocr memakai ketiganya.
    # *_API_KEY kosong = pakai API_KEY service ini.
    # Default menunjuk port lokal masing-masing untuk dev bare; di compose
    # di-override ke nama service.
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
        # Hanya tahap berikutnya di pipeline async. GUARDRAILS_/SCORING_SERVICE_URL
        # dipakai kontrak lama saja, jadi default localhost-nya tidak dipersoalkan.
        self.reject_localhost_outside_local(structuring_service_url=self.structuring_service_url)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
