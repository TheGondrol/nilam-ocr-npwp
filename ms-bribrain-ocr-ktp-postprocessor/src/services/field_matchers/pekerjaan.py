"""Pekerjaan (Occupation) matching module."""

import logging
import re

from src.services.text_helpers import clean_colon, fuzz_ratio, searchmapping
from src.services.constants import THRESHOLD_CONFIDENCE, THRESHOLD_RATIO

logger = logging.getLogger(__name__)


def _strip_label_prefix(text: str) -> str:
    """
    Mengekstrak nilai pekerjaan dari teks yang mungkin berisi label "pekerjaan"
    (atau variannya akibat OCR error) di awal string.

    Contoh:
        "Pekerjaan: GURU" -> "GURU"
        "pekerjan:guru"    -> "guru"
        "ekerjaan:guru"    -> "guru"
        "PekerjaanGURU"    -> "GURU"
        "pekerjaanguru"    -> "guru"
        "GURU"             -> "GURU"
    """
    stripped = text.strip()
    if not stripped:
        return ""

    # 1) Pemisah colon (paling umum): "Pekerjaan:VALUE" / "Pekerjaan : VALUE"
    if ":" in stripped or "：" in stripped:
        parts = re.split(r"[：:]", stripped, maxsplit=1)
        prefix = parts[0].strip()
        suffix = parts[1].strip() if len(parts) > 1 else ""
        if suffix and fuzz_ratio(prefix.lower(), "pekerjaan") >= 70:
            return suffix
        # Tidak match label — fall back ke clean_colon biasa
        return clean_colon(stripped)

    # 2) Lowercase prefix lalu uppercase value: "PekerjaanGURU"
    m = re.match(r"^([A-Za-z]+?)([A-Z]{2,}[A-Za-z\s]*)$", stripped)
    if m:
        prefix, suffix = m.group(1), m.group(2).strip()
        if suffix and fuzz_ratio(prefix.lower(), "pekerjaan") >= 70:
            return suffix

    # 3) Tanpa pemisah, semua lowercase: "pekerjaanguru" — strip prefix fuzzy
    lower = stripped.lower()
    if not any(c.isupper() for c in stripped):
        best_n, best_score = 0, 0
        for n in range(5, min(len(lower), 13)):
            score = fuzz_ratio(lower[:n], "pekerjaan")
            if score > best_score:
                best_score, best_n = score, n
        if best_score >= 90:
            remainder = stripped[best_n:].strip()
            if remainder:
                return remainder

    # 4) Tidak terdeteksi prefix label — kembalikan apa adanya
    return stripped


def _looks_like_kewarganegaraan(text: str) -> bool:
    """True jika teks fuzzy-match dengan label "kewarganegaraan" (field setelah pekerjaan)."""
    if not text:
        return False
    return fuzz_ratio(text.lower(), "kewarganegaraan") >= 70


def _resolve(text: str) -> str:
    """Strip label prefix lalu fuzzy-search ke mapping_pekerjaan."""
    clean = _strip_label_prefix(text)
    if not clean:
        return ""
    p, scorep = searchmapping("pekerjaan", clean)
    if scorep >= THRESHOLD_RATIO:
        return p
    return ""


def matching_pekerjaan(key: str, prev_data: list, data: list, next_data: list) -> str:
    """
    Melakukan mapping pekerjaan menggunakan fuzzy matching.

    Args:
        key (str): Keydat yang match. Jika "pekerjaan" → label-mode (value ada di
            next_data / data / prev_data). Selainnya keydat match VALUE pekerjaan
            secara langsung di `data` itu sendiri (mirip status_perkawinan).
        prev_data (list): Data OCR pada cell sebelumnya, format [text, confidence, box].
            Digunakan sebagai fallback ketika value cell muncul SEBELUM label cell.
        data (list): Data OCR pada cell saat ini, format [text, confidence, box].
        next_data (list): Data OCR pada cell berikutnya, format [text, confidence, box].

    Returns:
        str: Hasil mapping pekerjaan, atau string kosong jika tidak ditemukan.
    """
    try:
        if key != "pekerjaan":
            if data and len(data) > 1 and data[1] > THRESHOLD_CONFIDENCE:
                result = _resolve(data[0])
                if result:
                    return result
            return ""

        next_text = next_data[0] if next_data and len(next_data) > 0 else ""

        # 1) Kewarganegaraan di next_data → value ter-merge di data
        if _looks_like_kewarganegaraan(next_text):
            if data and len(data) > 1 and data[1] > THRESHOLD_CONFIDENCE:
                result = _resolve(data[0])
                if result:
                    return result

        # 2) Coba next_data (kasus paling umum)
        if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE:
            result = _resolve(next_data[0])
            if result:
                return result

        # 3) Fallback ke prev_data (value muncul sebelum label)
        if prev_data and len(prev_data) > 1 and prev_data[1] > THRESHOLD_CONFIDENCE:
            result = _resolve(prev_data[0])
            if result:
                return result

        return ""
    except Exception as e:
        logger.error(f"Gagal ekstrak pekerjaan di matching_pekerjaan: {e}")
        return ""
