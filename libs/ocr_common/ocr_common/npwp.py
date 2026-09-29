"""What the NPWP pipeline produces for the orchestrator: the field names, the final result carried by
the SCORING callback, and its mapping to the orchestrator's `extract-ocr` contract."""

import json
import math
from collections.abc import Mapping
from typing import Any

from ocr_common.types import ContractData, ContractField, FinalField, FinalResult

DOCUMENT_TYPE = "npwp"
# `errors` of a 400 for a document that is not accepted: by the guardrails model, or by a rejecting
# check of the structuring rules.
REJECTED_CODE = "DOWNSTREAM_VALIDATION_ERROR"
NPWP_FIELDS = ("nomor_npwp", "nama", "nama_badan")
TRUST_SCORES = ("npwp_confidence", "name_confidence")
# The fields of the `extract-ocr` contract's `data`, the keys of a per-field threshold.
CONTRACT_FIELDS = ("nomor_npwp", "nama")
COLUMN_THRESHOLD_DESCRIPTION = (
    "Per field, from the central orchestrator: the trust model's probability that the field's value is correct "
    "must reach it for the field's `confidence` to be `1` (else `0`), always on the accept side. Keys: "
    "`nomor_npwp`, `nama`; a field left out (or the whole map omitted) uses `FIELD_CONFIDENCE_THRESHOLD`"
)


def parse_column_thresholds(value: Any) -> dict[str, float] | None:
    """`column_confidence_threshold` checked: None, or an object whose keys are `CONTRACT_FIELDS` and whose
    values are numbers from 0 to 1. Raises ValueError with the reason otherwise."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('column_confidence_threshold must be a JSON object, e.g. {"nomor_npwp": 0.9, "nama": 0.5}')
    unknown = sorted(set(value) - set(CONTRACT_FIELDS))
    if unknown:
        raise ValueError(
            f"column_confidence_threshold has unknown field(s) {', '.join(unknown)}; "
            f"expected {', '.join(CONTRACT_FIELDS)}"
        )
    thresholds = {}
    for name, threshold in value.items():
        if isinstance(threshold, bool) or not isinstance(threshold, int | float) or not math.isfinite(threshold):
            raise ValueError(f"column_confidence_threshold.{name} must be a number, got {threshold!r}")
        if not 0 <= threshold <= 1:
            raise ValueError(f"column_confidence_threshold.{name} must be between 0 and 1, got {threshold}")
        thresholds[name] = float(threshold)
    return thresholds or None


def column_thresholds_from_json(raw: str | None) -> dict[str, float] | None:
    """`column_confidence_threshold` as a form field carries it (a JSON object string); None when empty.
    Raises ValueError."""
    if raw is None or not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise ValueError("column_confidence_threshold must be valid JSON: an object of field -> threshold") from exc
    return parse_column_thresholds(value)


def contract_fields(
    result: FinalResult, threshold: float, column_thresholds: Mapping[str, float] | None = None
) -> ContractData:
    """The `data` of the orchestrator's `extract-ocr` contract: `nomor_npwp` and `nama` (the person's
    name, or the company's registered name) with `confidence` 1 when the trust model's probability
    reaches the field's threshold, else 0. A field's threshold is its entry in `column_thresholds`
    (the central orchestrator's `column_confidence_threshold`), else `threshold`. The structuring
    rules' flag stays internal: it is already in the trust model's probability, and a document with a
    rejecting flag never gets this far."""
    fields = result["fields"]
    scoring = result["scoring"]
    columns = column_thresholds or {}
    name = fields.get("nama") if _has_value(fields.get("nama")) else fields.get("nama_badan")
    return {
        "nomor_npwp": _field(
            fields.get("nomor_npwp"), scoring.get("npwp_confidence"), columns.get("nomor_npwp", threshold)
        ),
        "nama": _field(name, scoring.get("name_confidence"), columns.get("nama", threshold)),
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
    """Combines the structuring fields and flag, the trust model's confidences and the guardrails report
    that was submitted into the `FinalResult` the SCORING callback carries."""
    fields: dict[str, FinalField] = {
        name: {"value": field.get("value"), "confidence": field.get("confidence", 1.0)}
        for name, field in (structuring.get("fields") or {}).items()
    }
    return {
        "document_type": document_type,
        "fields": fields,
        "scoring": {"npwp_confidence": scoring["npwp_confidence"], "name_confidence": scoring["name_confidence"]},
        "guardrails": guardrails,
        "flag": bool(structuring.get("flag", False)),
        "flag_reason": structuring.get("flag_reason"),
    }
