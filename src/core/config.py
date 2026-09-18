from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # min_length=1, bukan sekadar wajib: string kosong lolos sebagai nilai yang
    # sah, service tetap start sehat, lalu MENOLAK SEMUA request dengan 401
    # karena tidak ada key yang bisa menyamainya. Pod hijau, API mati. Lebih
    # baik gagal start. MOCK_API_KEY diterima juga supaya repo ini bisa
    # ditaruh menggantikan mock ocr-npwp tanpa mengubah .env / script di sana.
    api_key: str = Field(..., min_length=1, validation_alias=AliasChoices("API_KEY", "MOCK_API_KEY"))
    service_base_url: str | None = None
    port: int = 8030

    # Opsional. Diisi -> status request_id disimpan di PostgreSQL (skema:
    # db/schema.sql, dipasang manual). Kosong -> in-memory seperti mock ocr-*.
    # Format: postgresql+asyncpg://<user>:<password>@<host>:<port>/<dbname>
    database_url: str | None = None

    max_upload_bytes: int = 5 * 1024 * 1024
    allowed_content_types: list[str] = ["image/jpeg", "image/jpg", "image/png"]

    # Backend (implementasi model) per app. Nama harus terdaftar di src/models/<app>.py.
    weights_dir: str = "weights"
    guardrails_backend: str = "mock"
    ekstraksi_backend: str = "mock"
    structuring_backend: str = "rule_based"
    scoring_backend: str = "heuristic"

    # Ambang batas (aturan bisnis, bukan bagian dari model)
    guardrails_min_quality_score: float = 0.5
    guardrails_expected_document_type: str = "npwp"
    guardrails_min_classification_confidence: float = 0.7
    scoring_approve_threshold: float = 0.8
    scoring_review_threshold: float = 0.5


@lru_cache
def get_settings() -> Settings:
    return Settings()
