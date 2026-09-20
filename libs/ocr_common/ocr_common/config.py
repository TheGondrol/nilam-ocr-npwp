from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "dev", "staging", "production"]
# Di dalam pod / container, alamat ini berarti "diri sendiri", bukan service lain.
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


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

    # local = laptop; dev / staging / production = lingkungan ter-deploy (GKE), semuanya
    # dengan pengaman aktif. `dev` sengaja BUKAN mode longgar: cluster dev bernama "dev",
    # dan ENVIRONMENT=dev di sana tidak boleh mematikan pengaman.
    # Default `production`, BUKAN `local`: konfigurasi yang lupa mengisi ENVIRONMENT
    # (ConfigMap salah salin, env baru) harus gagal start dengan pesan jelas,
    # bukan diam-diam berjalan dengan semua pengaman mati. Laptop: ENVIRONMENT=local.
    # Di luar `local` berlaku: AUTH_DISABLED ditolak, backend mock ditolak,
    # DATABASE_URL + ORCHESTRATION_URL wajib, alamat service tidak boleh localhost.
    environment: Environment = "production"
    service_base_url: str | None = None
    port: int = 8000

    max_upload_bytes: int = 5 * 1024 * 1024
    # PDF ikut diterima: model PaddleOCR membaca PDF multi-halaman.
    allowed_content_types: list[str] = ["image/jpeg", "image/jpg", "image/png", "application/pdf"]

    @property
    def is_local(self) -> bool:
        return self.environment == "local"

    def require_outside_local(self, **values: object) -> None:
        """Setting yang boleh kosong di laptop tapi tidak di cluster. Nama = nama field."""
        if self.is_local:
            return
        missing = [name.upper() for name, value in values.items() if not value]
        if missing:
            raise ValueError(
                f"{', '.join(missing)} must be set when ENVIRONMENT={self.environment} "
                "(set ENVIRONMENT=local for local development)"
            )

    def reject_localhost_outside_local(self, **urls: str | None) -> None:
        """Alamat service LAIN. Bukan untuk DATABASE_URL: Cloud SQL Auth Proxy memang di 127.0.0.1."""
        if self.is_local:
            return
        local = [name.upper() for name, url in urls.items() if url and urlsplit(url).hostname in _LOCAL_HOSTS]
        if local:
            raise ValueError(
                f"{', '.join(local)} points to localhost, which inside a pod is this service itself; "
                f"set the real address when ENVIRONMENT={self.environment}"
            )

    def reject_mock_backend_outside_local(self, **backends: str) -> None:
        """Backend mock mengarang hasil (vonis dari nama file, NPWP acak). Default-nya mock: lupa mengisi = mock."""
        if self.is_local:
            return
        mocked = [name.upper() for name, backend in backends.items() if backend == "mock"]
        if mocked:
            raise ValueError(
                f"{', '.join(mocked)}=mock fabricates results and is only allowed with ENVIRONMENT=local; "
                f"set a real backend when ENVIRONMENT={self.environment}"
            )

    @model_validator(mode="after")
    def _guard_auth(self) -> Self:
        if self.auth_disabled and not self.is_local:
            raise ValueError(
                f"AUTH_DISABLED=true is only allowed with ENVIRONMENT=local (got ENVIRONMENT={self.environment}): "
                "it turns off the X-API-Key check on every endpoint"
            )
        return self


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

    @model_validator(mode="after")
    def _guard_pipeline(self) -> Self:
        # Tanpa DATABASE_URL klaim job hanya idempoten di dalam satu proses; tanpa
        # ORCHESTRATION_URL callback dilewati diam-diam dan hasil pipeline hilang.
        self.require_outside_local(database_url=self.database_url, orchestration_url=self.orchestration_url)
        self.reject_localhost_outside_local(orchestration_url=self.orchestration_url)
        return self
