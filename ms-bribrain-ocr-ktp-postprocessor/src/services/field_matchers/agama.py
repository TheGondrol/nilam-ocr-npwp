"""Agama (Religion) matching module."""

import logging

from src.services.text_helpers import clean_colon, searchmapping
from src.services.constants import THRESHOLD_CONFIDENCE, THRESHOLD_RATIO

logger = logging.getLogger(__name__)



def matching_agama(key: str, data: list, next_data: list) -> str:
    """
    Melakukan mapping agama menggunakan fuzzy matching.

    Args:
        key (str): Key field yang sedang diproses.
        data (list): Data OCR pada baris saat ini, format [text, confidence].
        next_data (list): Data OCR pada baris berikutnya, format [text, confidence].

    Returns:
        str: Hasil mapping agama, atau string kosong jika tidak ditemukan.
    """
    clean = ""
    try:
        if key == "agama":
            if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE:
                clean = clean_colon(next_data[0])
        else:
            if data and len(data) > 1 and data[1] > THRESHOLD_CONFIDENCE:
                clean = clean_colon(data[0])
        if clean:
            a, scorea = searchmapping("agama", clean)
            if scorea >= THRESHOLD_RATIO:
                return a
        return ""
    except Exception as e:
        logger.error(f"Gagal ekstrak agama di matching_agama: {e}")
        return ""
