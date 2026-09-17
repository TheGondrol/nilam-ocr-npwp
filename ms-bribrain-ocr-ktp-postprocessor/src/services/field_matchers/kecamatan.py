"""Kecamatan (District) matching module."""

import logging
import re

from src.services.text_helpers import clean_colon, data_verification
from src.services.constants import THRESHOLD_CONFIDENCE, REGEX

logger = logging.getLogger(__name__)



def matching_kecamatan(data: list, next_data: list) -> str:
    """
    Mengambil dan membersihkan kecamatan dari data berikutnya jika confidence mencukupi.

    Args:
        data (list): Data OCR pada baris saat ini, format [text, confidence].
        next_data (list): Data OCR pada baris berikutnya, format [text, confidence].

    Returns:
        str: Kecamatan dalam huruf kapital, atau string kosong jika tidak valid.
    """
    try:
        if len(data[0]) > 11 and data[1] > THRESHOLD_CONFIDENCE:
            process = data[0]
        else:
            process = next_data[0] if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE else ""
        clean = clean_colon(process)
        clean = data_verification(clean, "normal")
        match = re.findall(REGEX.get('AMBIL_DUA_KAPITAL_LALU_HURUF', r'[A-Z]{2}[A-Za-z]+'), clean)
        if match:
            kecamatan = " ".join(match)
            return kecamatan.upper()
        return ""
    except Exception as e:
        logger.error(f"Gagal ekstrak kecamatan di matching_kecamatan: {e}")
        return ""
