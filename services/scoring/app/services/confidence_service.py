import re
from typing import Any

from ocr_common.types import FieldConfidences

from app.ml.trust_model import TrustModel

NAME_FIELDS = ("nama", "nama_badan")


def _has_value(field: dict[str, Any] | None) -> bool:
    return bool(field and field.get("value") is not None and str(field["value"]).strip())


class ConfidenceService:
    def __init__(self, model: TrustModel):
        self._model = model

    def predict(self, payload: dict[str, Any]) -> FieldConfidences:
        return self._model.predict(payload)

    @staticmethod
    def payload_from_chain(
        guardrails: dict[str, Any] | None, ocr: dict[str, Any] | None, structuring: dict[str, Any]
    ) -> dict[str, Any]:
        """The ML team's scoring payload from the chained stage results: number and name signals from
        structuring (the name as its base read, before normalisation), the document-level OCR scores from
        the OCR blocks, the flag from the structuring rules, and the guardrails confidence of an accepted
        document."""
        fields = structuring.get("fields") or {}
        number = fields.get("nomor_npwp") if _has_value(fields.get("nomor_npwp")) else None
        name = next((fields[key] for key in NAME_FIELDS if _has_value(fields.get(key))), None)
        number_signals = (number or {}).get("signals") or {}
        name_signals = (name or {}).get("signals") or {}

        blocks = (ocr or {}).get("blocks") or []
        scores = [float(block["confidence"]) for block in blocks if block.get("confidence") is not None]

        document = (guardrails or {}).get("document") or {}
        return {
            "npwp": re.sub(r"\D", "", str(number["value"])) if number else None,
            "npwp_score": number.get("confidence") if number else None,
            "npwp_candidate_count": number_signals.get("candidate_count"),
            "name_base": (name_signals.get("name_base") or str(name["value"])) if name else None,
            "name_score": name.get("confidence") if name else None,
            "avg_doc_score": round(sum(scores) / len(scores), 6) if scores else None,
            "min_doc_score": min(scores) if scores else None,
            "flag": bool(structuring.get("flag", False)),
            "guardrail_probability": document.get("confidence") if document.get("verdict") == "accepted" else None,
        }
