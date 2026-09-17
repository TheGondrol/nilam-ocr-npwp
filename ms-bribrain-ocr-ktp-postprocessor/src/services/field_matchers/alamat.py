"""Alamat (Address) matching module."""

import logging

from src.services.text_helpers import clean_colon, data_verification
from src.services.constants import THRESHOLD_CONFIDENCE

logger = logging.getLogger(__name__)



def matching_alamat(next_data: list) -> str:
    """
    Mengambil dan membersihkan alamat dari data berikutnya jika confidence mencukupi.

    Args:
        next_data (list): Data OCR pada baris berikutnya, format [text, confidence].

    Returns:
        str: Alamat dalam huruf kapital, atau string kosong jika tidak valid.
    """
    try:
        if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE:
            clean = clean_colon(next_data[0])
            clean = data_verification(clean, "normal")
            return clean.upper()
        return ""
    except Exception as e:
        logger.error(f"Gagal ekstrak alamat di matching_alamat: {e}")
        return ""
