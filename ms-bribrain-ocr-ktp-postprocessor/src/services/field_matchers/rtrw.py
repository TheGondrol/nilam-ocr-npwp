"""RT/RW matching module."""

import logging
import re
from typing import Tuple

from src.services.constants import THRESHOLD_CONFIDENCE, REGEX

logger = logging.getLogger(__name__)



def rtrw_tidak_sesuai_format(rtrw: str) -> Tuple[str, str]:
    """Extract RT/RW from non-standard format."""
    final: dict[str, str] = {}
    index = 0
    before = 0
    try:
        for a in rtrw:
            before = int(a)
            if before == 0 and final != {}:
                index += 1
            if int(a) > 1:
                if final.get(f"{index}"):
                    final[f"{index}"] += a
                else:
                    final[f"{index}"] = a
            rt = final.get("0", "")
            rw = final.get("1", "")
        if rt.isdigit() and rw.isdigit():
            rt = rt.zfill(3)
            rw = rw.zfill(3)
            return rt, rw
    except Exception as e:
        logger.error(f"Gagal ekstrak rtrw: {e}")
        return "", ""
    return "", ""


def rtrw_sesuai_format(rtrw: str) -> Tuple[str, str]:
    """Extract RT/RW from standard format."""
    try:
        if "/" in rtrw:
            rt, rw = rtrw.split("/")
            rt = re.sub(REGEX.get("AMBIL_NON_DIGIT", r"\D"), '', rt)
            rw = re.sub(REGEX.get("AMBIL_NON_DIGIT", r"\D"), '', rw)
        else:
            rt, rw = rtrw[:3], rtrw[3:]
        if rt.isdigit() and rw.isdigit():
            rt = rt.zfill(3)
            rw = rw.zfill(3)
            return rt, rw
    except Exception as e:
        logger.error(f"Gagal ekstrak rtrw: {e}")
        return "", ""
    return "", ""


def matching_rtrw(next_data: list) -> Tuple[str, str]:
    """
    Mengekstrak RT dan RW dari data berikutnya jika confidence mencukupi.

    Args:
        next_data (list): Data OCR pada baris berikutnya, format [text, confidence].

    Returns:
        tuple: (rt, rw) dalam format string, atau ('', '') jika tidak valid.
    """
    if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE:
        dat = re.sub(r'[^0-9/](?=\d)', '', next_data[0])
        if len(dat) >= 6:
            return rtrw_sesuai_format(dat)
        else:
            return rtrw_tidak_sesuai_format(dat)
    return "", ""
