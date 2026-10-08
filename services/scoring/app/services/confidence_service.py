import re
from collections.abc import Mapping
from typing import Any

from ocr_common.npwp import final_result, scored_fields
from ocr_common.types import FieldConfidences, ScoringResult

from app.ml.trust_model import TrustModel

NAME_FIELDS = ("nama", "nama_badan")


def _has_value(field: dict[str, Any] | None) -> bool:
    return bool(field and field.get("value") is not None and str(field["value"]).strip())


class ConfidenceService:
    def __init__(self, model: TrustModel):
        self._model = model

    def predict(self, payload: dict[str, Any]) -> FieldConfidences:
        return self._model.predict(payload)

    def score(
        self,
        guardrails: dict[str, Any] | None,
        ocr: dict[str, Any] | None,
        structuring: dict[str, Any],
        column_thresholds: Mapping[str, float] | None = None,
    ) -> ScoringResult:
        """The scoring stage's result for the chained stage results: the ML team's payload built from them,
        the trust model's two probabilities, and each field's confidence: 0/1 with the field's
        `column_thresholds` entry, else the probability itself (threshold None). The same for a job and for
        `/v1/scoring-direct`."""
        payload = self.payload_from_chain(guardrails, ocr, structuring)
        result = self.predict(payload)
        npwp, name = result["npwp_confidence"], result["name_confidence"]
        # The decision is stored with the probabilities: each field's confidence, and the threshold that decided it.
        decided = final_result(guardrails, structuring, {"npwp_confidence": npwp, "name_confidence": name})
        return {
            "npwp_confidence": npwp,
            "name_confidence": name,
            "fields": scored_fields(decided, None, column_thresholds),
            "payload": payload,
        }

    @staticmethod
    def payload_from_chain(
        guardrails: dict[str, Any] | None, ocr: dict[str, Any] | None, structuring: dict[str, Any]
    ) -> dict[str, Any]:
        """The ML team's scoring payload from the chained stage results: number and name signals from
        structuring (the name as its base read, before normalisation), the document-level OCR scores from
        the OCR blocks, the flag from the structuring rules, and the guardrails confidence of an accepted
        document: the auto-accept score when the request sent no guardrails threshold (the lowest accepted
        probability of its pages, as `document.confidence` of a document the model accepted)."""
        fields = structuring.get("fields") or {}
        number = fields.get("nomor_npwp") if _has_value(fields.get("nomor_npwp")) else None
        name = next((fields[key] for key in NAME_FIELDS if _has_value(fields.get(key))), None)
        number_signals = (number or {}).get("signals") or {}
        name_signals = (name or {}).get("signals") or {}

        blocks = (ocr or {}).get("blocks") or []
        scores = [float(block["confidence"]) for block in blocks if block.get("confidence") is not None]

        report = guardrails or {}
        document = report.get("document") or {}
        if report.get("auto_accepted"):
            guardrail_probability = report["score"]
        else:
            guardrail_probability = document.get("confidence") if document.get("verdict") == "accepted" else None
        return {
            "npwp": re.sub(r"\D", "", str(number["value"])) if number else None,
            "npwp_score": number.get("confidence") if number else None,
            "npwp_candidate_count": number_signals.get("candidate_count"),
            "name_base": (name_signals.get("name_base") or str(name["value"])) if name else None,
            "name_score": name.get("confidence") if name else None,
            "avg_doc_score": round(sum(scores) / len(scores), 6) if scores else None,
            "min_doc_score": min(scores) if scores else None,
            "flag": bool(structuring.get("flag", False)),
            "guardrail_probability": guardrail_probability,
        }
