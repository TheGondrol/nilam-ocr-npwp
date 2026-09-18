from functools import lru_cache
from typing import Literal

from ocr_common.config import BaseServiceSettings


class Settings(BaseServiceSettings):
    port: int = 8031

    # Backend model. Nama harus terdaftar di src/models/guardrails.py:
    #   mock         - vonis dari nama file (blur/invalid/notnpwp -> reject)
    #   efficientnet - checkpoint EfficientNet-B0 (weights/best_model.pt) di proses ini
    #   remote       - service model guardrails milik ML engineer (POST {url}/v1/predict/json)
    guardrails_backend: str = "mock"

    # Backend `remote`. URL wajib diisi kalau GUARDRAILS_BACKEND=remote, mis.
    # http://localhost:8081. API key dikirim sebagai header X-API-Key ke service
    # model (bukan API_KEY service ini, yang dipakai pemanggil KITA).
    guardrails_model_url: str | None = None
    guardrails_model_api_key: str | None = None
    guardrails_model_timeout_seconds: float = 30.0

    # --- Hanya backend lokal (mock, efficientnet). Untuk `remote`, ambang,
    # kebijakan dokumen, dan render PDF diputuskan service model. ---
    guardrails_model_path: str = "weights/best_model.pt"
    # Inference CPU saja. Nilai selain cpu hanya dipakai kalau torch melihat
    # device-nya; kalau tidak, jatuh ke cpu dengan peringatan di log.
    guardrails_device: str = "cpu"
    # Batas thread intra-op torch (kosong = bawaan torch: semua core yang terlihat).
    # Di container dengan limit CPU, samakan dengan limit-nya.
    guardrails_torch_threads: int | None = None
    # Kosong -> pakai reject_threshold yang tersimpan di checkpoint (0.5).
    # Halaman divonis reject kalau proba_reject >= ambang ini.
    guardrails_reject_threshold: float | None = None

    # Vonis dokumen dari vonis halaman: `all` = accepted hanya kalau semua
    # halaman accepted; `majority` = accepted kalau accepted > reject.
    guardrails_document_policy: Literal["all", "majority"] = "all"
    # Render PDF: dpi per halaman dan batas jumlah halaman yang dinilai.
    guardrails_pdf_dpi: int = 150
    guardrails_max_pages: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
