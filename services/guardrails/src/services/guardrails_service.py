"""
Business logic Guardrails: nilai tiap halaman dokumen accepted / reject dengan
model, lalu putuskan vonis dokumen. Hasilnya laporan, bukan error; keputusan
melanjutkan ke OCR ada di pemanggil (orkestrator / service ekstraksi).

Bentuk laporan (kontrak dengan orkestrator):

    {"passed": bool, "reason": str | None,
     "document": {"verdict": "accepted" | "reject", "confidence", "n_pages", "n_approve", "n_reject"},
     "pages": [{"page_index", "proba_approve", "proba_reject", "verdict"}, ...]}

`passed` / `reason` adalah "true/false + alasan" di sequence diagram: passed=false
-> orkestrator menjawab 422 ke client dengan `reason`; passed=true -> lanjut ke OCR.

Layer ini tidak tahu HTTP (melempar ServiceError) dan tidak tahu model apa
yang dipakai (menerima classifier lewat constructor). Yang ia tahu hanya dua
bentuk kontrak backend (lihat src/models/guardrails.py): lokal `classify()`
per halaman, atau remote `check_document()` yang sudah membawa vonis dokumen.
"""

from typing import Any

from starlette.concurrency import run_in_threadpool

from ocr_common.image_validation import validate_image
from src.core.config import Settings
from src.services.pages import render_pages

VERDICT_ACCEPTED = "accepted"
VERDICT_REJECT = "reject"


def _reject_reason(document: dict[str, Any]) -> str:
    """Alasan penolakan untuk 422 orkestrator. Modelnya biner (accepted / reject), jadi
    alasannya hanya bisa menyebut berapa halaman yang ditolak dan seberapa yakin."""
    return (
        f"Document rejected by guardrails: {document['n_reject']}/{document['n_pages']} page(s) rejected "
        f"(confidence {document['confidence']:.2f})"
    )


class GuardrailsService:
    def __init__(self, classifier, settings: Settings):
        self._classifier = classifier
        self._settings = settings

    async def check(self, filename: str, content_type: str | None, content: bytes) -> dict[str, Any]:
        # Tipe / kosong / ukuran dinilai di sini untuk SEMUA backend, supaya
        # berkas yang jelas salah tidak sampai dikirim ke service model.
        validate_image(content_type, content, self._settings)

        if hasattr(self._classifier, "check_document"):
            # Backend remote: I/O jaringan, vonis dokumen datang dari service model.
            report = await self._classifier.check_document(filename, content, content_type)
        else:
            # Backend lokal: render halaman + inference itu CPU-bound dan sinkron;
            # di threadpool supaya event loop tetap melayani /health dan request lain.
            report = await run_in_threadpool(self._check_locally, filename, content_type, content)

        passed = report["document"]["verdict"] == VERDICT_ACCEPTED
        return {"passed": passed, "reason": None if passed else _reject_reason(report["document"]), **report}

    def _check_locally(self, filename: str, content_type: str | None, content: bytes) -> dict[str, Any]:
        pages = render_pages(
            content_type,
            content,
            dpi=self._settings.guardrails_pdf_dpi,
            max_pages=self._settings.guardrails_max_pages,
        )
        predictions = self._classifier.classify(filename, pages)

        threshold = self._settings.guardrails_reject_threshold
        if threshold is None:
            threshold = float(getattr(self._classifier, "reject_threshold", 0.5))

        page_results = [
            {
                "page_index": index,
                "proba_approve": proba_approve,
                "proba_reject": proba_reject,
                "verdict": VERDICT_REJECT if proba_reject >= threshold else VERDICT_ACCEPTED,
            }
            for index, (proba_approve, proba_reject) in enumerate(predictions)
        ]
        return {"document": self._aggregate(page_results), "pages": page_results}

    def _aggregate(self, pages: list[dict[str, Any]]) -> dict[str, Any]:
        n_pages = len(pages)
        n_reject = sum(1 for page in pages if page["verdict"] == VERDICT_REJECT)
        n_approve = n_pages - n_reject

        if self._settings.guardrails_document_policy == "majority":
            accepted = n_approve > n_reject
        else:  # all
            accepted = n_pages > 0 and n_reject == 0

        # Keyakinan pada vonis dokumen: kalau accepted, halaman yang paling
        # lemah menentukan; kalau reject, halaman yang paling kuat menolak.
        if accepted:
            confidence = min(page["proba_approve"] for page in pages)
        else:
            rejected = [page["proba_reject"] for page in pages if page["verdict"] == VERDICT_REJECT] or [0.0]
            confidence = max(rejected)

        return {
            "verdict": VERDICT_ACCEPTED if accepted else VERDICT_REJECT,
            "confidence": round(confidence, 4),
            "n_pages": n_pages,
            "n_approve": n_approve,
            "n_reject": n_reject,
        }
