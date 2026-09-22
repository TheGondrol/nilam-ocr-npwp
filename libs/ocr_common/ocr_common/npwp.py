from typing import Any

DOCUMENT_TYPE = "npwp"
NPWP_FIELDS = ("nomor_npwp", "nama", "nama_badan")
TRUST_SCORES = ("npwp_confidence", "name_confidence")


def contract_fields(result: dict[str, Any], threshold: float) -> dict[str, dict[str, Any]]:
    fields = result["fields"]
    scoring = result["scoring"]
    name = fields.get("nama") if _has_value(fields.get("nama")) else fields.get("nama_badan")
    return {
        "nomor_npwp": _field(fields.get("nomor_npwp"), scoring.get("npwp_confidence"), threshold),
        "nama": _field(name, scoring.get("name_confidence"), threshold),
    }


def _has_value(field: dict[str, Any] | None) -> bool:
    return bool(field and field.get("value") is not None and str(field["value"]).strip())


def _field(field: dict[str, Any] | None, score: float | None, threshold: float) -> dict[str, Any]:
    value = field["value"] if field and _has_value(field) else None
    confident = value is not None and score is not None and score >= threshold
    return {"value": value, "confidence": 1 if confident else 0}


def final_result(
    document_type: str, guardrails: dict[str, Any] | None, structuring: dict[str, Any], scoring: dict[str, Any]
) -> dict[str, Any]:
    fields = {
        name: {"value": field.get("value"), "confidence": field.get("confidence", 1.0)}
        for name, field in (structuring.get("fields") or {}).items()
    }
    return {
        "document_type": document_type,
        "fields": fields,
        "scoring": {key: scoring[key] for key in TRUST_SCORES},
        "guardrails": guardrails,
    }
