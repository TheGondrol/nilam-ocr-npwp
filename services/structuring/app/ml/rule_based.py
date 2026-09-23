"""The first structurer: labelled lines ("NPWP : ...", "NAMA : ...") and a few fallback patterns.
Kept as the `rule_based` backend for OCR output that still carries labels."""

import re

from ocr_common.npwp import NPWP_FIELDS
from ocr_common.types import OcrBlock, StructuredDocument, StructuredField

from app.ml.utils import _BADAN_PREFIX, normalize_npwp

_LABELS: dict[str, tuple[str, ...]] = {
    "nomor_npwp": ("NOMOR NPWP", "NO. NPWP", "NO NPWP", "NPWP"),
    "nama_badan": ("NAMA BADAN USAHA", "NAMA BADAN", "NAMA PERUSAHAAN", "BADAN USAHA"),
    "nama": ("NAMA WAJIB PAJAK", "NAMA"),
}
_LABEL_PATTERNS = [
    (name, re.compile(rf"^\s*{re.escape(label)}\b\s*[:\-]?\s*(?P<value>.+?)\s*$", re.IGNORECASE))
    for name, labels in _LABELS.items()
    for label in labels
]

_NPWP_FORMATTED = re.compile(r"\d{2}\.\d{3}\.\d{3}\.\d-\d{3}\.\d{3}")
_NPWP_DIGITS = re.compile(r"(?<!\d)\d{15}(?!\d)")

_FALLBACKS = (
    ("nomor_npwp", (_NPWP_FORMATTED, _NPWP_DIGITS)),
    ("nama_badan", (_BADAN_PREFIX,)),
)


class RuleBasedNpwpStructurer:
    name = "rule_based"

    def structure(self, lines: list[OcrBlock]) -> StructuredDocument:
        fields: dict[str, StructuredField] = {}
        unmatched: list[OcrBlock] = []

        for line in lines:
            match = self._match_label(line["text"])
            if match is None:
                unmatched.append(line)
                continue
            name, value = match
            fields.setdefault(name, {"value": value, "confidence": line["confidence"], "source": line["text"]})

        for name, patterns in _FALLBACKS:
            if name in fields:
                continue
            for line in unmatched:
                if any(p.search(line["text"]) for p in patterns):
                    value = line["text"].strip()
                    if name == "nomor_npwp":
                        value = next(m for p in patterns if (m := p.search(line["text"]))).group(0)
                    fields[name] = {"value": value, "confidence": line["confidence"], "source": line["text"]}
                    break

        number = fields.get("nomor_npwp")
        if number is not None and number["value"] is not None:
            number["value"] = normalize_npwp(number["value"])

        return {
            "fields": {
                name: fields.get(name, {"value": None, "confidence": 0.0, "source": None}) for name in NPWP_FIELDS
            },
            "flag": False,
            "flag_reason": None,
        }

    @staticmethod
    def _match_label(text: str) -> tuple[str, str] | None:
        for name, pattern in _LABEL_PATTERNS:
            match = pattern.match(text)
            if match:
                return name, match.group("value").strip()
        return None
