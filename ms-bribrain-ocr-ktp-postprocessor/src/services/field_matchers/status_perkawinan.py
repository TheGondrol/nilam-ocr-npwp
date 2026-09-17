"""Status Perkawinan (Marital Status) matching module."""

import logging
import re

from src.services.text_helpers import searchmapping
from src.services.constants import THRESHOLD_CONFIDENCE, THRESHOLD_RATIO, REGEX

logger = logging.getLogger(__name__)



def matching_status(key: str, data: list, next_data: list) -> str:
    """
    Melakukan mapping status perkawinan menggunakan fuzzy matching.

    Args:
        key (str): Key field yang sedang diproses.
        data (list): Data OCR pada baris saat ini, format [text, confidence].
        next_data (list): Data OCR pada baris berikutnya, format [text, confidence].

    Returns:
        str: Hasil mapping status perkawinan, atau string kosong jika tidak ditemukan.
    """
    prefix_sp = None
    try:
        if key == "status perkawinan" and data and len(data) > 0 and len(data[0]) < 20:
            if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE:
                prefix_sp = re.search(REGEX.get('AMBIL_KATA_KAPITAL', r'([A-Z]{2,}.*)'), next_data[0])
        else:
            if data and len(data) > 1 and data[1] > THRESHOLD_CONFIDENCE:
                prefix_sp = re.search(REGEX.get('AMBIL_KATA_KAPITAL', r'([A-Z]{2,}.*)'), data[0])
        if prefix_sp:
            match_sp = prefix_sp.group()
            a, scorea = searchmapping("status_perkawinan", match_sp)
            if scorea >= THRESHOLD_RATIO:
                return a
        return ""
    except Exception as e:
        logger.error(f"Gagal ekstrak status perkawinan di matching_status: {e}")
        return ""
