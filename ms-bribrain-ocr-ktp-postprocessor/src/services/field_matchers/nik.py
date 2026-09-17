"""NIK (National ID Number) matching module."""

import logging

from src.services.text_helpers import clean_colon, correct_alphabets_to_digits, getdigitonly
from src.services.constants import THRESHOLD_CONFIDENCE

logger = logging.getLogger(__name__)


def _is_plausible_nik(nik: str) -> bool:
    """Reject 16-digit strings that cannot be a real NIK.

    A KTP NIK encodes a province code (digits 1-2) and a birth day (digits 7-8,
    with +40 for women). This catches non-NIK 16-digit numbers (serials,
    barcodes, merged fields) without a full checksum (BUG-11). The month segment
    is intentionally left unvalidated to tolerate OCR noise there.
    """
    if len(nik) != 16 or not nik.isdigit():
        return False
    province = int(nik[0:2])
    if not (11 <= province <= 94):  # valid Indonesian province-code range
        return False
    day = int(nik[6:8])
    if day > 40:  # women: birth day + 40
        day -= 40
    return 1 <= day <= 31


def matching_nik(data: list) -> str:
    """
    Mengambil dan membersihkan NIK dari data jika confidence mencukupi.

    Args:
        data (list): Data OCR pada baris saat ini, format [text, confidence].

    Returns:
        str: NIK yang sudah dibersihkan, hanya digit. Jika tidak valid, return string kosong.
    """
    nik = data
    try:
        if nik and len(nik[0]) > 1 and nik[1] > THRESHOLD_CONFIDENCE:
            clean = clean_colon(nik[0])
            dat = correct_alphabets_to_digits(clean)
            dat = getdigitonly(dat)
            if len(dat) == 16 and _is_plausible_nik(dat):
                return dat
        return ""
    except Exception as e:
        logger.error(f"Gagal ekstrak NIK: {e}")
        return ""
