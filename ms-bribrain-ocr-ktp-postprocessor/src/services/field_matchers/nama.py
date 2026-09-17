"""Nama (Name) matching module."""

import logging
import re

from src.services.text_helpers import clean_colon, data_verification
from src.services.constants import THRESHOLD_CONFIDENCE, REGEX

logger = logging.getLogger(__name__)



def matching_nama(data: list) -> str:
    """
    Mengambil dan membersihkan nama dari data jika confidence mencukupi.

    Args:
        data (list): Data OCR pada baris saat ini, format [text, confidence].

    Returns:
        str: Nama dalam huruf kapital, atau string kosong jika tidak valid.
    """
    ocr_nama = data
    try:
        if ocr_nama and len(ocr_nama[0]) > 1 and ocr_nama[1] > THRESHOLD_CONFIDENCE:
            nama = clean_colon(ocr_nama[0])
            nama = data_verification(nama, "normal")
            # Ganti karakter spesial dengan spasi
            nama = re.sub(REGEX.get('AMBIL_NON_ALFANUMERIK_SPASI', r'[^A-Za-z0-9\s]'), ' ', nama)

            # Hilangkan spasi berlebih
            nama = re.sub(REGEX.get('AMBIL_SPASI', r'\s+'), ' ', nama).strip()
            return nama.upper()
        return ""
    except Exception as e:
        logger.error(f"Gagal ekstrak nama: {e}")
        return ""
