from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseServiceSettings(BaseSettings):
    """
    Setting yang dimiliki semua service. Tiap service membuat subclass
    (services/<nama>/src/core/config.py) untuk port default dan setting
    modelnya sendiri, lalu menyediakan get_settings() ber-lru_cache.
    """

    # populate_by_name: api_key dibaca dari env API_KEY/MOCK_API_KEY (alias), tapi
    # di test/kode tetap bisa ditulis Settings(api_key=...).
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    # min_length=1, bukan sekadar wajib: string kosong lolos sebagai nilai yang
    # sah, service tetap start sehat, lalu MENOLAK SEMUA request dengan 401.
    # Pod hijau, API mati. Lebih baik gagal start. MOCK_API_KEY diterima juga
    # supaya .env / script dari mock ocr-* bisa dipakai apa adanya.
    api_key: str = Field(..., min_length=1, validation_alias=AliasChoices("API_KEY", "MOCK_API_KEY"))
    # Saklar eksplisit untuk mematikan pemeriksaan X-API-Key (dev / uji coba
    # lokal). Sengaja bukan "API_KEY kosong = tanpa auth": nilai kosong yang
    # diam-diam mematikan auth adalah jebakan di produksi; saklar bernama
    # mudah dicari, dan service mencatat peringatan saat aktif. API_KEY tetap
    # wajib karena juga dipakai sebagai key KELUAR ke service lain.
    auth_disabled: bool = False
    service_base_url: str | None = None
    port: int = 8000

    max_upload_bytes: int = 5 * 1024 * 1024
    # PDF ikut diterima: model PaddleOCR membaca PDF multi-halaman.
    allowed_content_types: list[str] = ["image/jpeg", "image/jpg", "image/png", "application/pdf"]


class PipelineSettings(BaseServiceSettings):
    """
    Setting tambahan untuk tahap pipeline async (ekstraksi, structuring,
    scoring): tabel jobs/results miliknya, callback ke Orkestrasi, dan
    handoff ke tahap berikutnya. Lihat ocr_common/jobs.py.
    """

    # Opsional. Diisi -> <schema>.jobs / <schema>.results di PostgreSQL (skema:
    # services/<nama>/db/schema.sql, dipasang manual). Kosong -> in-memory:
    # cukup untuk dev satu proses, hilang saat restart, tidak idempoten antar replika.
    # Format: postgresql+asyncpg://<user>:<password>@<host>:<port>/<dbname>
    database_url: str | None = None

    # Callback status tahap ke Orkestrasi: POST {ORCHESTRATION_URL}{ORCHESTRATION_CALLBACK_PATH}.
    # URL kosong -> callback dilewati (dicatat di log), berguna untuk dev tanpa orkestrator.
    orchestration_url: str | None = None
    orchestration_callback_path: str = "/v1/callbacks/stage"
    orchestration_api_key: str | None = None
    orchestration_timeout_seconds: float = 10.0

    # Callback dan handoff dicoba ulang kalau tujuannya tidak terjangkau / 5xx.
    pipeline_retry_attempts: int = 3
    pipeline_retry_delay_seconds: float = 0.5
    # Saat shutdown, tunggu job yang masih jalan paling lama sekian detik.
    pipeline_drain_timeout_seconds: float = 30.0
