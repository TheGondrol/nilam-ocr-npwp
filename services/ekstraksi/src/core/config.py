from functools import lru_cache

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
    # Backend `paddle`: service PaddleOCR milik ML engineer (POST {url}/ocr).
    # Wajib diisi kalau EKSTRAKSI_BACKEND=paddle.
    ekstraksi_ocr_url: str | None = None
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
