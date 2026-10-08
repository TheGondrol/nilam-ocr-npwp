"""What the NPWP pipeline produces for the orchestrator: the field names, the final result carried by
the SCORING callback, and its mapping to the orchestrator's `extract-ocr` contract."""

import json
import math
from collections.abc import Mapping
from typing import Any

from ocr_common.types import ContractData, ContractField, FinalField, FinalResult, ScoredField

DOCUMENT_TYPE = "npwp"
# `errors` of a 400 for a document that is not accepted: by the guardrails model, or by a rejecting
# check of the structuring rules.
REJECTED_CODE = "DOWNSTREAM_VALIDATION_ERROR"
# `message` of the extract-ocr 200, also kept in `nilam_ocr_results`.
COMPLETED_MESSAGE = "OCR extraction completed successfully"
NPWP_FIELDS = ("nomor_npwp", "nama", "nama_badan")
TRUST_SCORES = ("npwp_confidence", "name_confidence")
# The fields of the `extract-ocr` contract's `data`, the keys of a per-field threshold.
CONTRACT_FIELDS = ("nomor_npwp", "nama")
# The key of `column_confidence_threshold` that sets every field at once; a field's own key wins over it.
ALL_FIELDS_KEY = "all_field"
COLUMN_THRESHOLD_DESCRIPTION = (
    "Per field, from the central orchestrator: the trust model's probability that the field's value is correct "
    "must reach it for the field's `confidence` to be `1` (else `0`), always on the accept side. Keys: "
    "`all_field` (every field), `nomor_npwp`, `nama`; a field's own key wins over `all_field`. A field left out "
    "(or the whole map omitted) gets that probability itself as `confidence`, a float from 0 to 1"
)
# The guardrails of this pipeline, as the central orchestrator names them in `guardrails_confidence_threshold`.
# This repo runs one guardrails model (accept / reject), keyed `acc_rej`; the object form leaves room for more.
GUARDRAILS_KEYS = ("acc_rej",)
GUARDRAILS_THRESHOLD_DESCRIPTION = (
    "From the central orchestrator: the guardrails threshold for this document, between 0 and 1 (exclusive), "
    "on the model's accepted probability: a page passes when it reaches the threshold, and is rejected below "
    "it. A JSON object keyed by guardrails name; this pipeline has one, `acc_rej`. Omitted: the document is "
    "accepted whatever the model says, and the answer's `guardrails` is the model's accepted probability (the "
    "lowest of its pages) instead of 0 / 1"
)
# `guardrails` of an answer: 0 the document passed the guardrails model with the central orchestrator's threshold,
# 1 it was rejected (by that model, or by the structuring rules).
GUARDRAILS_PASSED = 0
GUARDRAILS_REJECTED = 1
# Floats in an answer (an accepted probability, a field's trust probability) keep this many decimals, like the
# guardrails report's `confidence`.
SCORE_DECIMALS = 4


def parse_column_thresholds(value: Any) -> dict[str, float] | None:
    """`column_confidence_threshold` checked: None, or an object whose keys are `CONTRACT_FIELDS` or
    `ALL_FIELDS_KEY` and whose values are numbers from 0 to 1. `all_field` is spread over every contract
    field, a field's own key winning, so the result only has field names. Raises ValueError with the reason
    otherwise."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(
            "column_confidence_threshold must be a JSON object, "
            'e.g. {"all_field": 0.8} or {"nomor_npwp": 0.9, "nama": 0.5}'
        )
    unknown = sorted(set(value) - set(CONTRACT_FIELDS) - {ALL_FIELDS_KEY})
    if unknown:
        raise ValueError(
            f"column_confidence_threshold has unknown field(s) {', '.join(unknown)}; "
            f"expected {ALL_FIELDS_KEY}, {', '.join(CONTRACT_FIELDS)}"
        )
    given = {
        name: _threshold_number("column_confidence_threshold", name, threshold) for name, threshold in value.items()
    }
    thresholds = {}
    if ALL_FIELDS_KEY in given:
        thresholds.update(dict.fromkeys(CONTRACT_FIELDS, given.pop(ALL_FIELDS_KEY)))
    thresholds.update(given)
    return thresholds or None


def _threshold_number(field: str, name: str, threshold: Any) -> float:
    """`threshold` as a float from 0 to 1 (inclusive); ValueError naming `field.name` otherwise."""
    if isinstance(threshold, bool) or not isinstance(threshold, int | float) or not math.isfinite(threshold):
        raise ValueError(f"{field}.{name} must be a number, got {threshold!r}")
    if not 0 <= threshold <= 1:
        raise ValueError(f"{field}.{name} must be between 0 and 1, got {threshold}")
    return float(threshold)


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


def parse_guardrails_threshold(value: Any) -> float | None:
    """`guardrails_confidence_threshold` checked: None, or an object with exactly the key `acc_rej` (the one
    guardrails of this pipeline) and a number strictly between 0 and 1, which is returned. Raises ValueError
    with the reason otherwise."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('guardrails_confidence_threshold must be a JSON object, e.g. {"acc_rej": 0.8}')
    unknown = sorted(set(value) - set(GUARDRAILS_KEYS))
    if unknown:
        raise ValueError(
            f"guardrails_confidence_threshold has unknown guardrails {', '.join(unknown)}; "
            f"expected {', '.join(GUARDRAILS_KEYS)}"
        )
    if not value:
        return None
    (name,) = GUARDRAILS_KEYS
    threshold = _threshold_number("guardrails_confidence_threshold", name, value[name])
    if not 0 < threshold < 1:
        raise ValueError(
            f"guardrails_confidence_threshold.{name} must be between 0 and 1 (exclusive), got {value[name]}"
        )
    return threshold


def guardrails_threshold_from_json(raw: str | None) -> float | None:
    """`guardrails_confidence_threshold` as a form field carries it (a JSON object string); None when empty.
    Raises ValueError."""
    if raw is None or not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise ValueError(
            "guardrails_confidence_threshold must be valid JSON: an object of guardrails -> threshold, "
            'e.g. {"acc_rej": 0.8}'
        ) from exc
    return parse_guardrails_threshold(value)


def auto_accept(report: Mapping[str, Any]) -> dict[str, Any]:
    """The guardrails report of a request that sent no `guardrails_confidence_threshold`: accepted whatever the
    model said (`auto_accepted`), with `score` the model's accepted probability, the lowest of its pages (both
    guardrails backends report every page). The service's own verdict stays in `document`."""
    score = min((float(page["proba_approve"]) for page in report.get("pages") or []), default=0.0)
    return {**report, "passed": True, "reason": None, "auto_accepted": True, "score": round(score, SCORE_DECIMALS)}


def guardrails_value(report: Mapping[str, Any] | None) -> int | float | None:
    """`guardrails` of an answer for this guardrails report: its `score` (a float) when the request sent no
    threshold (`auto_accept`), else 0 passed / 1 rejected; None without a report (guardrails did not run)."""
    if not report:
        return None
    if report.get("auto_accepted"):
        return report["score"]
    return GUARDRAILS_PASSED if report.get("passed", True) else GUARDRAILS_REJECTED


def contract_fields(
    result: FinalResult, threshold: float | None = None, column_thresholds: Mapping[str, float] | None = None
) -> ContractData:
    """The `data` of the orchestrator's `extract-ocr` contract: `nomor_npwp` and `nama` (the person's
    name, or the company's registered name). A field with a threshold (its entry in `column_thresholds`,
    the central orchestrator's `column_confidence_threshold`, else `threshold`) has `confidence` 1 when the
    trust model's probability reaches it, else 0; a field without one has that probability itself, a float.
    The structuring rules' flag stays internal: it is already in the trust model's probability, and a
    document with a rejecting flag never gets this far."""
    return contract_data(scored_fields(result, threshold, column_thresholds))


def scored_fields(
    result: FinalResult, threshold: float | None = None, column_thresholds: Mapping[str, float] | None = None
) -> dict[str, ScoredField]:
    """`contract_fields` plus the threshold each field was decided with (None: the probability is the
    confidence): what the scoring stage stores."""
    fields = result["fields"]
    scoring = result["scoring"]
    columns = column_thresholds or {}
    name = fields.get("nama") if _has_value(fields.get("nama")) else fields.get("nama_badan")
    nomor_threshold = columns.get("nomor_npwp", threshold)
    nama_threshold = columns.get("nama", threshold)
    return {
        "nomor_npwp": {
            **_field(fields.get("nomor_npwp"), scoring.get("npwp_confidence"), nomor_threshold),
            "threshold": nomor_threshold,
        },
        "nama": {**_field(name, scoring.get("name_confidence"), nama_threshold), "threshold": nama_threshold},
    }


def contract_data(fields: Mapping[str, ScoredField]) -> ContractData:
    """The `extract-ocr` `data` of stored scored fields: value and confidence, without the threshold."""
    return {
        "nomor_npwp": {"value": fields["nomor_npwp"]["value"], "confidence": fields["nomor_npwp"]["confidence"]},
        "nama": {"value": fields["nama"]["value"], "confidence": fields["nama"]["confidence"]},
    }


def _has_value(field: Mapping[str, Any] | None) -> bool:
    return bool(field and field.get("value") is not None and str(field["value"]).strip())


def _field(field: Mapping[str, Any] | None, score: float | None, threshold: float | None) -> ContractField:
    """A field without a value, or without a trust probability, has `confidence` 0 either way."""
    value = field["value"] if field and _has_value(field) else None
    if value is None or score is None:
        return {"value": value, "confidence": 0}
    if threshold is None:
        return {"value": value, "confidence": round(float(score), SCORE_DECIMALS)}
    return {"value": value, "confidence": 1 if score >= threshold else 0}


def final_result(
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
        "fields": fields,
        "scoring": {"npwp_confidence": scoring["npwp_confidence"], "name_confidence": scoring["name_confidence"]},
        "guardrails": guardrails,
        "flag": bool(structuring.get("flag", False)),
        "flag_reason": structuring.get("flag_reason"),
    }
