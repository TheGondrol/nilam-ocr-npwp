"""
Pembungkus structurer: baris teks mentah -> field bernama.
    structure(lines: [{"text", "confidence", "bbox"?, "page"?}]) -> {field: {"value", "confidence", "source"}}

Field NPWP mengikuti ocr_common.npwp (sheet "OCR Nilam - Document Type").

Implementasi:
- npwp_rules (default): aturan regex + POSISI dari ML engineer (src/vendor/npwp_rules).
  Nama dicari sebagai baris berbentuk nama yang paling dekat dengan nomor NPWP, karena
  kartu asli mencetak nama TANPA label. Butuh `bbox`; tanpa itu posisi diturunkan dari
  urutan baca.
- rule_based: regex berbasis label ("NAMA : ..."). Hanya cocok untuk teks berlabel
  seperti keluaran mock OCR; pada kartu asli `nama` tidak ketemu.
"""

import re
from functools import lru_cache
from typing import Any

from ocr_common.errors import ServiceError
from ocr_common.npwp import NPWP_FIELDS
from ocr_common.registry import Factory, build_backend
from src.core.config import Settings, get_settings
from src.vendor.npwp_rules import npwp as rules
from src.vendor.npwp_rules.name_extraction import extract_name

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


def _format_npwp(digits: str) -> str:
    """15 digit -> bentuk bertitik yang sudah jadi kontrak kita; 16 digit (format NIK) tidak punya bentuk bertitik."""
    return normalize_npwp(digits) if len(digits) == 15 else digits


def _page_results(lines: list[dict]) -> list[dict[str, Any]]:
    """Baris kita -> satu hasil PaddleOCR per halaman ({rec_texts, rec_scores, rec_polys}), bentuk yang
    diharapkan aturan ML engineer. `bbox` kita adalah kotak tegak; aturannya hanya memakai titik tengah
    poly, jadi empat sudut kotak itu setara. Kalau ADA baris tanpa bbox di sebuah halaman, seluruh
    halaman itu memakai posisi dari urutan baca (mencampur koordinat asli dan buatan akan menyesatkan)."""
    by_page: dict[int, list[dict]] = {}
    for line in lines:
        by_page.setdefault(int(line.get("page") or 0), []).append(line)

    pages = []
    for index in sorted(by_page):
        page_lines = by_page[index]
        positioned = all(line.get("bbox") for line in page_lines)
        polys = []
        for order, line in enumerate(page_lines):
            if positioned:
                b = line["bbox"]
                polys.append([[b["x1"], b["y1"]], [b["x2"], b["y1"]], [b["x2"], b["y2"]], [b["x1"], b["y2"]]])
            else:
                top = order * 10
                polys.append([[0, top], [100, top], [100, top + 8], [0, top + 8]])
        pages.append(
            {
                "rec_texts": [line["text"] for line in page_lines],
                "rec_scores": [float(line.get("confidence", 1.0)) for line in page_lines],
                "rec_polys": polys,
            }
        )
    return pages


class NpwpRulesStructurer:
    """
    Aturan ML engineer (src/vendor/npwp_rules), dirangkai ke kontrak structurer kita. Kedua file
    kiriman tidak menyertakan pemanggilnya, jadi perangkaian di bawah adalah tafsir atas docstring
    mereka; tiap keputusan ditandai supaya mudah dikoreksi begitu modul pemanggil aslinya ada.
    """

    name = "npwp_rules"

    def __init__(self, page_guardrails: bool = True):
        self._page_guardrails = page_guardrails

    def structure(self, lines: list[dict]) -> dict:
        pages = _page_results(lines)
        if self._page_guardrails:
            self._reject_non_npwp(pages)

        fields: dict[str, dict] = {}
        number = self._pick_number(pages)
        if number is not None:
            fields["nomor_npwp"] = number

        # Nama diambil dari halaman pertama yang memilikinya (upload 2 halaman: depan + belakang,
        # atau suami + istri; halaman pertama adalah kartu utamanya).
        for page in pages:
            found = extract_name(page)
            if found is None:
                continue
            value, score = found
            # name_corrected di payload scoring: apakah koreksi name-master mengubah nama. extract_name
            # sendiri membedakan keluaran "modified" (dengan koreksi) dan "base" (tanpa). Selama
            # name_master.py masih pengganti sementara, koreksi tidak pernah terjadi, jadi selalu False.
            base = extract_name(page, apply_master_correction=False)
            corrected = base is not None and base[0] != value
            # PT/CV diprioritaskan extract_name sendiri; pemetaan ke nama_badan memakai daftar
            # penanda badan yang lebih lebar (UD, YAYASAN, ...), sama dengan backend rule_based.
            field = "nama_badan" if _BADAN_PREFIX.match(value) else "nama"
            source = next((t for t in page["rec_texts"] if value.split(",")[0].strip() in t), value)
            fields[field] = {
                "value": value,
                "confidence": round(float(score or 0.0), 4),
                "source": source,
                "signals": {"corrected": corrected},
            }
            break

        empty = {"value": None, "confidence": 0.0, "source": None, "signals": None}
        return {name: fields.get(name, dict(empty)) for name in NPWP_FIELDS}

    @staticmethod
    def _reject_non_npwp(pages: list[dict[str, Any]]) -> None:
        """Guardrail halaman dari npwp.py: docstring-nya menyebut tiap kondisi "must be rejected outright"."""
        if len(pages) > rules.MAX_EXPECTED_PAGES:
            raise ServiceError(
                400, f"Upload has {len(pages)} pages; an NPWP upload is at most {rules.MAX_EXPECTED_PAGES}"
            )
        for page in pages:
            texts = page["rec_texts"]
            keyword = rules.find_other_document_keyword(texts)
            if keyword is not None:
                raise ServiceError(400, f"Upload contains another document ({keyword}); send the NPWP card only")
            if rules.contains_captcha(texts):
                raise ServiceError(400, "Page shows a CAPTCHA challenge, not an NPWP card")
            if rules.contains_web_lookup_screenshot(texts):
                raise ServiceError(400, "Page is a screenshot of the DJP NPWP lookup, not an NPWP card")

    @staticmethod
    def _pick_number(pages: list[dict[str, Any]]) -> dict | None:
        """Semua kandidat berbentuk NPWP di dalam wilayah kartu, dipilih dengan npwp_priority (16 digit
        mengalahkan 15 digit, lalu confidence). Wilayah kartu dipakai supaya NIK milik dokumen lain yang
        ikut terfoto tidak menang; kalau tidak ada kandidat di dalam wilayah, semua kandidat dipakai
        (perilaku yang sama dengan _select_npwp_anchor di name_extraction.py)."""
        inside, anywhere = [], []
        n_matches = 0  # npwp_candidate_count di payload scoring: semua baris/nomor berbentuk NPWP di dokumen
        for page in pages:
            y_min, y_max = rules.find_document_region(page["rec_texts"], page["rec_polys"])
            for text, score, poly in zip(page["rec_texts"], page["rec_scores"], page["rec_polys"], strict=True):
                y = rules.poly_center(poly)[1]
                in_region = (y_min is None or y >= y_min) and (y_max is None or y <= y_max)
                for match in rules.NPWP_PATTERN.finditer(text):
                    raw = match.group()
                    n_matches += 1
                    digits = rules.normalize_npwp(raw)
                    expected = len(rules._DIGIT_LIKE_PATTERN.findall(raw))
                    # "T" yang tidak bisa dipastikan dibuang normalize_npwp; nomor 16 digit yang tinggal
                    # 15 akan LOLOS sebagai nomor 15 digit yang sah. Nomor yang tidak terbaca utuh dilewati.
                    if len(digits) != expected:
                        continue
                    candidate = (rules.npwp_priority(raw, score), digits, score, text, raw)
                    anywhere.append(candidate)
                    if in_region:
                        inside.append(candidate)
        pool = inside or anywhere
        if not pool:
            return None
        _, digits, score, text, raw = max(pool, key=lambda c: c[0])
        return {
            "value": _format_npwp(digits),
            "confidence": round(float(score), 4),
            "source": text,
            # npwp_has_homoglyph: fungsi ML engineer sendiri ("normalize_npwp() had to correct something").
            "signals": {"has_homoglyph": rules.contains_digit_homoglyph(raw), "candidate_count": n_matches},
        }


def _build_npwp_rules(settings: Settings) -> NpwpRulesStructurer:
    return NpwpRulesStructurer(page_guardrails=settings.structuring_page_guardrails)


STRUCTURER_BACKENDS: dict[str, Factory] = {
    "npwp_rules": _build_npwp_rules,
    "rule_based": lambda settings: RuleBasedNpwpStructurer(),
}


@lru_cache
def get_structurer():
    settings = get_settings()
    return build_backend(STRUCTURER_BACKENDS, settings.structuring_backend, settings, "structuring")
