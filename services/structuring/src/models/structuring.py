"""
Pembungkus structurer: baris teks mentah -> field bernama.
    structure(lines: [{"text", "confidence"}]) -> {field: {"value", "confidence", "source"}}

Field NPWP mengikuti ocr_common.npwp (sheet "OCR Nilam - Document Type").
Implementasi default berbasis regex; untuk layout berantakan tambahkan
structurer berbasis model (LLM/layout model) dengan method yang sama.
"""

import re
from functools import lru_cache

from ocr_common.npwp import NPWP_FIELDS
from ocr_common.registry import Factory, build_backend
from src.core.config import get_settings

# Alias label per field. Yang lebih panjang / lebih spesifik harus dicoba
# dulu: "NAMA BADAN" sebelum "NAMA", kalau tidak nama badan terbaca sebagai nama.
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
_BADAN_PREFIX = re.compile(r"^\s*(PT|CV|UD|PD|KOPERASI|YAYASAN|FIRMA)\b", re.IGNORECASE)

# Field yang bisa dikenali tanpa label, dari polanya saja.
_FALLBACKS = (
    ("nomor_npwp", (_NPWP_FORMATTED, _NPWP_DIGITS)),
    ("nama_badan", (_BADAN_PREFIX,)),
)


def normalize_npwp(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if len(digits) == 15:
        return f"{digits[0:2]}.{digits[2:5]}.{digits[5:8]}.{digits[8]}-{digits[9:12]}.{digits[12:15]}"
    return value.strip()


class RuleBasedNpwpStructurer:
    name = "rule_based"

    def structure(self, lines: list[dict]) -> dict:
        fields: dict[str, dict] = {}
        unmatched: list[dict] = []

        # 1) Baris berlabel "LABEL : NILAI"; kemunculan pertama menang.
        for line in lines:
            match = self._match_label(line["text"])
            if match is None:
                unmatched.append(line)
                continue
            name, value = match
            fields.setdefault(name, {"value": value, "confidence": line["confidence"], "source": line["text"]})

        # 2) Fallback untuk nilai tanpa label, berdasarkan polanya.
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

        # 3) Normalisasi.
        if "nomor_npwp" in fields:
            fields["nomor_npwp"]["value"] = normalize_npwp(fields["nomor_npwp"]["value"])

        # 4) Bentuk tetap: semua field dokumen selalu ada, yang tidak ketemu null.
        return {name: fields.get(name, {"value": None, "confidence": 0.0, "source": None}) for name in NPWP_FIELDS}

    @staticmethod
    def _match_label(text: str) -> tuple[str, str] | None:
        for name, pattern in _LABEL_PATTERNS:
            match = pattern.match(text)
            if match:
                return name, match.group("value").strip()
        return None


STRUCTURER_BACKENDS: dict[str, Factory] = {
    "rule_based": lambda settings: RuleBasedNpwpStructurer(),
}


@lru_cache
def get_structurer():
    settings = get_settings()
    return build_backend(STRUCTURER_BACKENDS, settings.structuring_backend, settings, "structuring")
