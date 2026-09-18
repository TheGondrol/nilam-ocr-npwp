"""
Pembungkus model untuk Guardrails: penilai kualitas gambar dan pengklasifikasi
jenis dokumen. Service layer (src/services/guardrails_service.py) hanya
memanggil assess()/classify() dan tidak tahu implementasinya.

Mock meniru model berdasarkan nama file (skenario sama dengan mock ocr-npwp:
'blur'/'invalid' -> kualitas rendah, 'notnpwp' -> bukan NPWP). Untuk model
asli, tambahkan kelas baru di sini dengan method yang sama, lalu daftarkan di
QUALITY_BACKENDS / CLASSIFIER_BACKENDS.
"""

from functools import lru_cache

from src.core.config import get_settings
from src.models.registry import Factory, build_backend


class MockImageQualityAssessor:
    name = "mock"

    def assess(self, filename: str, content: bytes) -> dict:
        """Returns {"score": 0..1, "notes": [str]}."""
        name = (filename or "").lower()
        if "blur" in name or "invalid" in name:
            return {"score": 0.15, "notes": ["blur detected"]}
        return {"score": 0.92, "notes": []}


class MockDocumentClassifier:
    name = "mock"

    def classify(self, filename: str, content: bytes) -> dict:
        """Returns {"label": str, "confidence": 0..1}."""
        name = (filename or "").lower()
        if "notnpwp" in name:
            return {"label": "unknown", "confidence": 0.88}
        return {"label": "npwp", "confidence": 0.97}


QUALITY_BACKENDS: dict[str, Factory] = {
    "mock": lambda settings: MockImageQualityAssessor(),
}
CLASSIFIER_BACKENDS: dict[str, Factory] = {
    "mock": lambda settings: MockDocumentClassifier(),
}


@lru_cache
def get_quality_assessor():
    return build_backend(QUALITY_BACKENDS, get_settings().guardrails_backend, "guardrails quality")


@lru_cache
def get_document_classifier():
    return build_backend(CLASSIFIER_BACKENDS, get_settings().guardrails_backend, "guardrails classifier")
