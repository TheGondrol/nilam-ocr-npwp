"""
Business logic Guardrails: menjawab "apakah gambar ini layak dan memang
dokumen yang diharapkan?" sebelum OCR dijalankan. Hasilnya laporan
(passed true/false), bukan error. Keputusan menolak/melanjutkan ada di
pemanggil (orchestrator).

Layer ini tidak tahu HTTP (melempar ServiceError) dan tidak tahu model apa
yang dipakai (menerima quality_assessor & classifier lewat constructor).
"""

from typing import Any

from src.core.config import Settings
from src.services.image_validation import validate_image

CHECK_IMAGE_QUALITY = "image_quality"
CHECK_DOCUMENT_TYPE = "document_type"


class GuardrailsService:
    def __init__(self, quality_assessor, classifier, settings: Settings):
        self._quality_assessor = quality_assessor
        self._classifier = classifier
        self._settings = settings

    def check(self, filename: str, content_type: str | None, content: bytes) -> dict[str, Any]:
        validate_image(content_type, content, self._settings)

        quality = self._quality_assessor.assess(filename, content)
        classification = self._classifier.classify(filename, content)
        checks = [self._check_quality(quality), self._check_document_type(classification)]

        return {
            "passed": all(check["passed"] for check in checks),
            "document_type": classification["label"],
            "confidence": classification["confidence"],
            "checks": checks,
        }

    def _check_quality(self, quality: dict) -> dict:
        minimum = self._settings.guardrails_min_quality_score
        passed = quality["score"] >= minimum
        message = (
            "Image quality is acceptable"
            if passed
            else f"Image quality too low ({quality['score']:.2f} < {minimum:.2f})"
        )
        if quality.get("notes"):
            message = f"{message}: {'; '.join(quality['notes'])}"
        return {"name": CHECK_IMAGE_QUALITY, "passed": passed, "score": quality["score"], "message": message}

    def _check_document_type(self, classification: dict) -> dict:
        expected = self._settings.guardrails_expected_document_type
        label, confidence = classification["label"], classification["confidence"]
        passed = label == expected and confidence >= self._settings.guardrails_min_classification_confidence
        if passed:
            message = f"Document recognized as {expected}"
        else:
            message = f"Document not recognized as {expected} (got {label!r} @ {confidence:.2f})"
        return {"name": CHECK_DOCUMENT_TYPE, "passed": passed, "score": confidence, "message": message}
