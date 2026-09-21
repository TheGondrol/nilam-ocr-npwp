from typing import Any

DOCUMENT_TYPE = "npwp"
NPWP_FIELDS = ("nomor_npwp", "nama", "nama_badan")
TRUST_SCORES = ("npwp_confidence", "name_confidence")


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
