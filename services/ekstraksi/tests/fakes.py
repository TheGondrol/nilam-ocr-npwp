"""
Pengganti service lain (guardrails, structuring, scoring) untuk test kontrak
extract-ocr, tanpa jaringan. Perilakunya meniru mock masing-masing service
secukupnya: skenario nama file, pelabelan "LABEL : NILAI", skor = confidence
terendah. Rantai sungguhan antar container diuji scripts/smoke_e2e.py.
"""

import re
from types import SimpleNamespace
from typing import Any

from ocr_common.errors import ServiceError
from ocr_common.npwp import NPWP_FIELDS

_LABELS = {"nomor_npwp": "NPWP", "nama_badan": "NAMA BADAN", "nama": "NAMA"}


class FakeGuardrails:
    async def check(self, request_id, filename, content_type, content):
        name = (filename or "").lower()
        if not content:
            raise ServiceError(400, "Uploaded file is empty")
        rejected = "blur" in name or "invalid" in name or "notnpwp" in name
        page = {
            "page_index": 0,
            "proba_approve": 0.1179 if rejected else 0.9821,
            "proba_reject": 0.8821 if rejected else 0.0179,
            "verdict": "reject" if rejected else "accepted",
        }
        return {
            "document": {
                "verdict": page["verdict"],
                "confidence": page["proba_reject"] if rejected else page["proba_approve"],
                "n_pages": 1,
                "n_approve": 0 if rejected else 1,
                "n_reject": 1 if rejected else 0,
            },
            "pages": [page],
        }


class FakeStructuring:
    async def structure(self, lines):
        if not any((line.get("text") or "").strip() for line in lines):
            raise ServiceError(400, "No text lines to structure")
        fields: dict[str, dict[str, Any]] = {
            name: {"value": None, "confidence": 0.0, "source": None} for name in NPWP_FIELDS
        }
        for line in lines:
            for name, label in _LABELS.items():
                match = re.match(rf"^\s*{label}\b\s*[:\-]?\s*(.+?)\s*$", line["text"], re.IGNORECASE)
                if match and fields[name]["value"] is None:
                    fields[name] = {"value": match.group(1), "confidence": line["confidence"], "source": line["text"]}
                    break
        return {"document_type": "npwp", "fields": fields}


class FakeScoring:
    async def score(self, document_type, fields):
        present = [f["confidence"] for f in fields.values() if f.get("value")]
        score = round(min(present), 4) if present else 0.0
        return {"score": score, "decision": "approve" if score >= 0.8 else "reject", "field_scores": [], "reasons": []}


def fake_stage_clients():
    return SimpleNamespace(guardrails=FakeGuardrails(), structuring=FakeStructuring(), scoring=FakeScoring())
