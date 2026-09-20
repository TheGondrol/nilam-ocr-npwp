"""
Business logic confidence per field: payload ML engineer -> {"npwp_confidence", "name_confidence"}
lewat trust model (src/models/trust_model.py), dan penyusunan payload itu dari hasil berantai
tahap-tahap sebelumnya ("value-nya adalah chain result dari service sebelumnya").

Asal tiap kunci payload di pipeline:

    npwp, npwp_score, name, name_score        structuring: fields.nomor_npwp / fields.nama | nama_badan
    npwp_has_homoglyph, npwp_candidate_count  structuring: fields.nomor_npwp.signals
    name_corrected                            structuring: fields.<nama>.signals
    n_boxes, num_pages, avg/min_doc_score     ekstraksi: ocr.blocks (jumlah, halaman, confidence)
    guardrail_probability                     guardrails: document.confidence. Contoh ML engineer memakai
                                              angka yang sama (0.9821) di response guardrails dan di payload ini.
    flag                                      BELUM ADA tahap yang menghasilkannya; tidak disebut di kontrak
                                              service lain yang kami terima. Dikirim kosong (imputer model
                                              mengisinya; pengaruhnya kecil). Tanyakan asalnya ke ML engineer.
"""

import re
from typing import Any

from src.models.trust_model import TrustModel

NAME_FIELDS = ("nama", "nama_badan")


def _has_value(field: dict[str, Any] | None) -> bool:
    return bool(field and field.get("value") is not None and str(field["value"]).strip())


class ConfidenceService:
    def __init__(self, model: TrustModel):
        self._model = model

    def predict(self, payload: dict[str, Any]) -> dict[str, float | None]:
        return self._model.predict(payload)

    @staticmethod
    def payload_from_chain(
        guardrails: dict[str, Any] | None, ocr: dict[str, Any] | None, structuring: dict[str, Any]
    ) -> dict[str, Any]:
        fields = structuring.get("fields") or {}
        number = fields.get("nomor_npwp") if _has_value(fields.get("nomor_npwp")) else None
        name = next((fields[key] for key in NAME_FIELDS if _has_value(fields.get(key))), None)
        number_signals = (number or {}).get("signals") or {}
        name_signals = (name or {}).get("signals") or {}

        blocks = (ocr or {}).get("blocks") or []
        scores = [float(block["confidence"]) for block in blocks if block.get("confidence") is not None]
        pages = {int(block.get("page") or 0) for block in blocks}

        document = (guardrails or {}).get("document") or {}
        return {
            # Digit saja, seperti contoh ML engineer ("123456789012000"); nilai kita bertitik untuk 15 digit.
            "npwp": re.sub(r"\D", "", str(number["value"])) if number else None,
            "npwp_score": number.get("confidence") if number else None,
            "npwp_has_homoglyph": number_signals.get("has_homoglyph"),
            "npwp_candidate_count": number_signals.get("candidate_count"),
            "name": str(name["value"]) if name else None,
            "name_score": name.get("confidence") if name else None,
            "name_corrected": name_signals.get("corrected"),
            "n_boxes": len(blocks) or None,
            "num_pages": len(pages) or None,
            "avg_doc_score": round(sum(scores) / len(scores), 6) if scores else None,
            "min_doc_score": min(scores) if scores else None,
            "flag": None,
            "guardrail_probability": document.get("confidence") if document.get("verdict") == "accepted" else None,
        }
