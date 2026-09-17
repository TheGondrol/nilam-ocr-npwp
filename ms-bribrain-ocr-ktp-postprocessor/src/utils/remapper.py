"""Remapping utilities for OCR field completion.

This module contains functions to fill in missing OCR fields
using additional fuzzy matching passes.
"""

import logging
import re

from src.services.constants import (
    keydata_pekerjaan,
    keydata_repeat,
    keydata_status,
    THRESHOLD_CONFIDENCE,
    THRESHOLD_PARTIAL,
    THRESHOLD_RATIO,
)
from src.services.field_matchers.kecamatan import matching_kecamatan
from src.services.field_matchers.keldesa import matching_keldesa
from src.services.field_matchers.pekerjaan import matching_pekerjaan
from src.services.field_matchers.rtrw import matching_rtrw
from src.services.field_matchers.status_perkawinan import matching_status
from src.services.field_matchers.ttl import matching_tempatlahir, matching_tempatlahir_new
from src.services.text_helpers import fuzz_partial

logger = logging.getLogger(__name__)


def remapping(data_cleaned: list, result: dict) -> dict:
    """
    Melengkapi hasil mapping field KTP yang belum terisi dengan fuzzy matching tambahan.

    Args:
        data_cleaned (list): List hasil OCR yang sudah dibersihkan.
        result (dict): Hasil mapping sementara.

    Returns:
        dict: Hasil mapping field KTP yang sudah dilengkapi.
    """
    result = result.copy()

    def update_result(field, value, score):
        # value != "" must guard ALL inserts, not just overwrites (BUG-14).
        if value != "" and (field not in result or result[field][1] <= score):
            result[field] = [value, score]

    for i, dat in enumerate(data_cleaned):
        next_data: list = data_cleaned[i + 1] if i + 1 < len(data_cleaned) else ["", 0, []]
        prev_data: list = data_cleaned[i - 1] if i > 0 else ["", 0, []]
        for keydat in keydata_repeat:
            score = fuzz_partial(dat[0], keydat)
            meet_threshold = score >= THRESHOLD_RATIO and dat[1] > THRESHOLD_CONFIDENCE
            meet_threshold_partial = score >= THRESHOLD_PARTIAL and dat[1] > THRESHOLD_CONFIDENCE

            if keydat == "kel/desa" and meet_threshold:
                update_result("kel_desa", matching_keldesa(next_data), score)
            elif keydat == "kecamatan" and meet_threshold:
                update_result("kecamatan", matching_kecamatan(dat, next_data), score)
            elif keydat in keydata_status and meet_threshold_partial:
                update_result("status_perkawinan", matching_status(keydat, dat, next_data), score)
            elif keydat in keydata_pekerjaan and meet_threshold_partial:
                update_result("pekerjaan", matching_pekerjaan(keydat, prev_data, dat, next_data), score)
            elif keydat == "tempat/tgl lahir" and meet_threshold_partial:
                angka_indat = ''.join(re.findall(r"\d+", next_data[0]))
                if i + 2 < len(data_cleaned):
                    if not any(word in data_cleaned[i + 2][0].lower() for word in ["jenis", "kelamin", "laki", "perempuan", "perem"]) and data_cleaned[i + 2][1] >= THRESHOLD_CONFIDENCE and len(angka_indat) < 5:
                        next_data = [f"{next_data[0] + ' ' + data_cleaned[i + 2][0]}", next_data[1], next_data[2]]
                tempat_lahir, tanggal_lahir = matching_tempatlahir(dat, next_data)
                if tempat_lahir == "" or tanggal_lahir == "":
                    tempat_lahir, tanggal_lahir = matching_tempatlahir_new(dat, next_data)
                update_result("tempat_lahir", tempat_lahir, score)
                update_result("tanggal_lahir", tanggal_lahir, score)
            elif keydat == "rt/rw" and meet_threshold_partial:
                rt, rw = matching_rtrw(next_data)
                update_result("rt", rt, score)
                update_result("rw", rw, score)
    return result
