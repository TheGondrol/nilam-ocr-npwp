"""The `npwp_rules` backend: the ML team's rules (app/vendor/npwp_rules) applied to positioned OCR
lines, wired the same way their own regex service wires them (nilamnpwp `regex/main.py`, 23 Sep 2026).

Per page: the NPWP-shaped lines inside the card region compete on `npwp_priority` (16 digits beat
15, then OCR score); the name is the nearest name-shaped line to the number (`extract_name`). Per
document: the first page with a number / name wins, and every check the rules provide raises the
document flag (`flag`, `flag_reason`), a feature of the trust model. Nine of the eleven checks also
reject the document (`reject_reason`, see `TOLERATED`). Digit-homoglyph correction is switched off by the ML team
(`normalize_npwp_raw`: a misread letter is dropped, the flag says so), so a number can come back
shorter than 15 digits; the trust model then gives it a low confidence."""

from typing import Any

from ocr_common.npwp import NPWP_FIELDS
from ocr_common.types import OcrBlock, StructuredDocument, StructuredField

from app.ml.utils import is_badan, normalize_npwp
from app.vendor.npwp_rules import npwp as rules
from app.vendor.npwp_rules.name_extraction import extract_name

MAX_EXPECTED_PAGES = rules.MAX_EXPECTED_PAGES

# Reasons, in the ML team's wording and priority order (the first that applies is reported).
FLAG_TOO_MANY_PAGES = "Dokumen memiliki {n} halaman (lebih dari {max}), kemungkinan ada dokumen lain yang tergabung"
FLAG_OTHER_DOCUMENT = "Dokumen lain terdeteksi: '{keyword}' pada halaman {page}"
FLAG_CAPTCHA = "bukan format standar NPWP"
FLAG_WEB_LOOKUP = "screenshot cek NPWP online, bukan foto fisik kartu"
FLAG_BLANK = "dokumen blur / blank"
FLAG_SINGLE_WORD_NAME = "Nama hanya terdiri dari 1 kata, mohon dicek kembali"
FLAG_HOMOGLYPH = "Nomor NPWP terdeteksi mengandung huruf (bukan angka murni), mohon dicek kembali"
FLAG_INVALID_PROVINCE = "Kode provinsi pada NPWP tidak valid, mohon dicek kembali"
FLAG_INVALID_KECAMATAN = "Kode kecamatan pada NPWP tidak valid, mohon dicek kembali"
FLAG_INVALID_BIRTHDATE = "Tanggal lahir pada NPWP tidak valid, mohon dicek kembali"
FLAG_INVALID_KPP = "Kode KPP pada NPWP tidak valid, mohon dicek kembali"

# The ML team's decision (23 Sep 2026): of the 11 checks, only these two are tolerated, because a
# reviewer checks and fixes them. They still raise `flag` for the trust model. Every other check
# (more pages than allowed, another document, not the standard NPWP format, an online-lookup
# screenshot, blur / blank, invalid province / kecamatan / birthdate / KPP code) rejects the document.
TOLERATED = frozenset({FLAG_SINGLE_WORD_NAME, FLAG_HOMOGLYPH})

NUMBER_SIGNALS = (
    "invalid_province_prefix",
    "invalid_kecamatan_prefix",
    "invalid_birthdate",
    "invalid_kpp_prefix",
)


def _format_npwp(digits: str, raw: str) -> str:
    """The printed form `XX.XXX.XXX.X-XXX.XXX` only for a number OCR read in that legacy shape with all 15
    digits; anything else (a 16-digit NIK-based number, or a number that lost a letter) stays plain digits."""
    return normalize_npwp(digits) if len(digits) == 15 and "." in raw else digits


def _lost_a_character(digits: str, raw: str) -> bool:
    """True when a digit-like character of the match did not survive `normalize_npwp_raw`: an OCR letter
    was dropped (correction is off), so the number is shorter than printed. `contains_digit_homoglyph`
    misses the ambiguous "T", which the rules resolve positionally or drop."""
    return len(digits) != len(rules._DIGIT_LIKE_PATTERN.findall(raw))


def _pages(lines: list[OcrBlock]) -> list[dict[str, Any]]:
    """Regroup the lines per page into the {page_index, rec_texts, rec_scores, rec_polys} shape the rules
    expect. Lines without a bbox get a synthetic top-to-bottom layout."""
    by_page: dict[int, list[OcrBlock]] = {}
    for line in lines:
        by_page.setdefault(int(line.get("page") or 0), []).append(line)

    pages = []
    for index in sorted(by_page):
        page_lines = by_page[index]
        positioned = all(line.get("bbox") for line in page_lines)
        polys = []
        for order, line in enumerate(page_lines):
            b = line.get("bbox")
            if positioned and b is not None:
                polys.append([[b["x1"], b["y1"]], [b["x2"], b["y1"]], [b["x2"], b["y2"]], [b["x1"], b["y2"]]])
            else:
                top = order * 10
                polys.append([[0, top], [100, top], [100, top + 8], [0, top + 8]])
        pages.append(
            {
                "page_index": index,
                "rec_texts": [line["text"] for line in page_lines],
                "rec_scores": [float(line.get("confidence", 1.0)) for line in page_lines],
                "rec_polys": polys,
            }
        )
    return pages


def _process_page(page: dict[str, Any]) -> dict[str, Any]:
    """One page: its NPWP / name candidate, the signals the trust model needs, and the page-level flag
    reasons (other document, CAPTCHA, lookup screenshot), in that order. Port of `_process_page` of
    the ML team's regex service."""
    rec_texts, rec_scores, rec_polys = page["rec_texts"], page["rec_scores"], page["rec_polys"]

    flag_reasons = []
    keyword = rules.find_other_document_keyword(rec_texts)
    if keyword:
        flag_reasons.append(FLAG_OTHER_DOCUMENT.format(keyword=keyword, page=page["page_index"] + 1))
    if rules.contains_captcha(rec_texts):
        flag_reasons.append(FLAG_CAPTCHA)
    if rules.contains_web_lookup_screenshot(rec_texts):
        flag_reasons.append(FLAG_WEB_LOOKUP)

    texts: list[dict[str, Any]] = []
    for text, score, poly in zip(rec_texts, rec_scores, rec_polys, strict=True):
        match = rules.NPWP_PATTERN.search(text)
        if not match:
            continue
        texts.append(
            {
                "text": rules.normalize_npwp_raw(match.group()),
                "raw": match.group(),
                "score": score,
                "poly": poly,
                "line": text,
            }
        )

    # The name is searched with every normalisation the ML team applies (formatting, name master), and
    # once more with none of them: the trust model measures the name on the base read.
    res_like = {"rec_texts": rec_texts, "rec_polys": rec_polys, "rec_scores": rec_scores}
    name_match = extract_name(res_like)
    name_match_base = extract_name(
        res_like,
        apply_balinese_normalization=False,
        apply_master_correction=False,
        apply_formatting_normalization=False,
    )
    name_corrected = name_match is not None and name_match_base is not None and name_match[0] != name_match_base[0]

    y_min, y_max = rules.find_document_region(rec_texts, rec_polys)
    in_region = [
        t
        for t in texts
        if (y_min is None or rules.poly_center(t["poly"])[1] >= y_min)
        and (y_max is None or rules.poly_center(t["poly"])[1] <= y_max)
    ]
    candidates = in_region or texts
    best = max(candidates, key=lambda t: rules.npwp_priority(str(t["text"]), float(t["score"])), default=None)
    raw = best["raw"] if best else ""

    return {
        "npwp": best["text"] if best else None,
        "npwp_raw": raw,
        "npwp_score": float(best["score"]) if best else None,
        "npwp_source": best["line"] if best else None,
        "npwp_candidate_count": len(candidates),
        "has_homoglyph": bool(best) and (rules.contains_digit_homoglyph(raw) or _lost_a_character(best["text"], raw)),
        "invalid_province_prefix": bool(best) and rules.has_invalid_province_prefix(raw),
        "invalid_kecamatan_prefix": bool(best) and rules.has_invalid_kecamatan_prefix(raw),
        "invalid_birthdate": bool(best) and rules.has_invalid_birthdate_digits(raw),
        "invalid_kpp_prefix": bool(best) and rules.has_invalid_kpp_prefix(raw),
        "name": name_match[0] if name_match else None,
        "name_score": float(name_match[1]) if name_match and name_match[1] is not None else None,
        "name_base": name_match_base[0] if name_match_base else None,
        "name_corrected": name_corrected,
        "name_source": _name_source(rec_texts, name_match_base[0] if name_match_base else None),
        "flag_reasons": flag_reasons,
    }


def _name_source(rec_texts: list[str], name_base: str | None) -> str | None:
    if name_base is None:
        return None
    return next((text for text in rec_texts if name_base in text), name_base)


class NpwpRulesStructurer:
    name = "npwp_rules"

    def structure(self, lines: list[OcrBlock]) -> StructuredDocument:
        pages = [_process_page(page) for page in _pages(lines)]

        flag_reason: str | None = None
        # The first rejecting check in the same priority order (see `REJECTING`). Tracked on its own
        # because flag_reason may hold a tolerated reason that outranks it (a single-word name).
        reject_reason: str | None = None
        guardrail_triggered = False
        too_many_pages = len(pages) > MAX_EXPECTED_PAGES
        if too_many_pages:
            flag_reason = reject_reason = FLAG_TOO_MANY_PAGES.format(n=len(pages), max=MAX_EXPECTED_PAGES)

        number: dict[str, Any] | None = None
        name: dict[str, Any] | None = None
        number_signals = dict.fromkeys(("has_homoglyph", *NUMBER_SIGNALS), False)
        candidate_count = 0
        for page in pages:
            if page["flag_reasons"]:
                guardrail_triggered = True
                flag_reason = flag_reason or page["flag_reasons"][0]
                reject_reason = reject_reason or page["flag_reasons"][0]
            for signal in number_signals:
                number_signals[signal] = number_signals[signal] or page[signal]
            # First present wins, so the reported score belongs to the value that is returned.
            if number is None and page["npwp"]:
                number = page
            if name is None and page["name"]:
                name = page
            candidate_count += page["npwp_candidate_count"]

        if number is None or name is None:
            guardrail_triggered = True
            flag_reason = flag_reason or FLAG_BLANK
            reject_reason = reject_reason or FLAG_BLANK

        single_word_name = any(p["name"] and len(p["name"].split()) == 1 for p in pages)
        checks = (
            (single_word_name, FLAG_SINGLE_WORD_NAME),
            (number_signals["has_homoglyph"], FLAG_HOMOGLYPH),
            (number_signals["invalid_province_prefix"], FLAG_INVALID_PROVINCE),
            (number_signals["invalid_kecamatan_prefix"], FLAG_INVALID_KECAMATAN),
            (number_signals["invalid_birthdate"], FLAG_INVALID_BIRTHDATE),
            (number_signals["invalid_kpp_prefix"], FLAG_INVALID_KPP),
        )
        for triggered, reason in checks:
            if triggered and flag_reason is None:
                flag_reason = reason
            if triggered and reject_reason is None and reason not in TOLERATED:
                reject_reason = reason
        flag = guardrail_triggered or too_many_pages or any(triggered for triggered, _ in checks)

        fields: dict[str, StructuredField] = {}
        if number is not None:
            fields["nomor_npwp"] = {
                "value": _format_npwp(number["npwp"], number["npwp_raw"]),
                "confidence": round(number["npwp_score"], 4),
                "source": number["npwp_source"],
                "signals": {"candidate_count": candidate_count, **number_signals},
            }
        if name is not None:
            fields["nama_badan" if is_badan(name["name"]) else "nama"] = {
                "value": name["name"],
                "confidence": round(name["name_score"] or 0.0, 4),
                "source": name["name_source"],
                "signals": {"name_base": name["name_base"], "corrected": name["name_corrected"]},
            }

        return {
            "fields": {
                field: fields.get(field, StructuredField(value=None, confidence=0.0, source=None, signals=None))
                for field in NPWP_FIELDS
            },
            "flag": flag,
            "flag_reason": flag_reason,
            "reject_reason": reject_reason,
        }
