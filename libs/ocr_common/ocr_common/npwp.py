"""What the NPWP pipeline produces for the orchestrator: the field names, the final result carried by
the SCORING callback, and its mapping to the orchestrator's `extract-ocr` contract."""

from collections.abc import Mapping
from typing import Any

from ocr_common.types import ContractField, FinalField, FinalResult

DOCUMENT_TYPE = "npwp"
NPWP_FIELDS = ("nomor_npwp", "nama", "nama_badan")
TRUST_SCORES = ("npwp_confidence", "name_confidence")


def contract_fields(result: FinalResult, threshold: float) -> dict[str, ContractField]:
    """The `data` of the orchestrator's `extract-ocr` contract: `nomor_npwp` and `nama` (the person's
    name, or the company's registered name) with `confidence` 1 when the trust model's probability
    reaches `threshold`, else 0."""
    fields = result["fields"]
    scoring = result["scoring"]
    name = fields.get("nama") if _has_value(fields.get("nama")) else fields.get("nama_badan")
    return {
        "nomor_npwp": _field(fields.get("nomor_npwp"), scoring.get("npwp_confidence"), threshold),
        "nama": _field(name, scoring.get("name_confidence"), threshold),
    }


def _has_value(field: Mapping[str, Any] | None) -> bool:
    return bool(field and field.get("value") is not None and str(field["value"]).strip())


def _field(field: Mapping[str, Any] | None, score: float | None, threshold: float) -> ContractField:
    value = field["value"] if field and _has_value(field) else None
    confident = value is not None and score is not None and score >= threshold
    return {"value": value, "confidence": 1 if confident else 0}


def final_result(
    document_type: str,
    guardrails: dict[str, Any] | None,
    structuring: Mapping[str, Any],
    scoring: Mapping[str, Any],
) -> FinalResult:
    """Combines the structuring fields, the trust model's confidences and the guardrails report that was
    submitted into the `FinalResult` the SCORING callback carries."""
    fields: dict[str, FinalField] = {
        name: {"value": field.get("value"), "confidence": field.get("confidence", 1.0)}
        for name, field in (structuring.get("fields") or {}).items()
    }
    return {
        "document_type": document_type,
        "fields": fields,
        "scoring": {"npwp_confidence": scoring["npwp_confidence"], "name_confidence": scoring["name_confidence"]},
        "guardrails": guardrails,
    }
