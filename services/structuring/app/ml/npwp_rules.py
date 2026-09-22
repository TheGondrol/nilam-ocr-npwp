"""The `npwp_rules` backend: the ML engineer's rules (app/vendor/npwp_rules) applied to positioned OCR
lines. Real cards print the name without a label, so it is found by its position relative to the
NPWP number."""

from typing import Any

from ocr_common.errors import ServiceError
from ocr_common.npwp import NPWP_FIELDS

from app.ml.utils import is_badan, normalize_npwp
from app.vendor.npwp_rules import npwp as rules
from app.vendor.npwp_rules.name_extraction import extract_name


def _format_npwp(digits: str) -> str:
    return normalize_npwp(digits) if len(digits) == 15 else digits


def _page_results(lines: list[dict]) -> list[dict[str, Any]]:
    """Regroup the lines per page into the {rec_texts, rec_scores, rec_polys} shape the rules expect.
    Lines without a bbox get a synthetic top-to-bottom layout."""
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
    name = "npwp_rules"

    def __init__(self, page_guardrails: bool = True):
        self._page_guardrails = page_guardrails

    def structure(self, lines: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        pages = _page_results(lines)
        if self._page_guardrails:
            self._reject_non_npwp(pages)

        fields: dict[str, dict] = {}
        number = self._pick_number(pages)
        if number is not None:
            fields["nomor_npwp"] = number

        for page in pages:
            found = extract_name(page)
            if found is None:
                continue
            value, score = found
            base = extract_name(page, apply_master_correction=False)
            corrected = base is not None and base[0] != value
            field = "nama_badan" if is_badan(value) else "nama"
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
        inside, anywhere = [], []
        n_matches = 0
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
            "signals": {"has_homoglyph": rules.contains_digit_homoglyph(raw), "candidate_count": n_matches},
        }
