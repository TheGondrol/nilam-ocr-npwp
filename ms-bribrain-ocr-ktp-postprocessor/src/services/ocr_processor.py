"""OCR processing service.

This module contains the main OCR processing logic for mapping
OCR text to KTP fields.
"""

import logging
import re
from typing import Any

from src.services.constants import (
    keydata,
    keydata_agama,
    keydata_jeniskelamin,
    keydata_status,
    THRESHOLD_CONFIDENCE,
    THRESHOLD_RATIO,
)
from src.services.field_matchers.agama import matching_agama
from src.services.field_matchers.alamat import matching_alamat
from src.services.field_matchers.jenis_kelamin import matching_jeniskelamin
from src.services.field_matchers.kecamatan import matching_kecamatan
from src.services.field_matchers.keldesa import matching_keldesa
from src.services.field_matchers.nama import matching_nama
from src.services.field_matchers.nik import matching_nik
from src.services.field_matchers.pekerjaan import matching_pekerjaan
from src.services.field_matchers.rtrw import matching_rtrw
from src.services.field_matchers.status_perkawinan import matching_status
from src.services.field_matchers.ttl import matching_tempatlahir, matching_tempatlahir_new
from src.services.text_helpers import fuzz_ratio
from src.utils.remapper import remapping

logger = logging.getLogger(__name__)


def mappingnext(data: list) -> tuple[dict, Any]:
    """
    Memetakan hasil OCR ke field KTP dengan fuzzy matching dan validasi confidence.

    Args:
        data (list): List hasil OCR, setiap elemen [box, (text, confidence)].

    Returns:
        tuple: (dict hasil mapping field KTP, nik_image_box)
    """
    # Clean and prepare data
    data_cleaned = [
        [text, score, box]
        for box, (text, score) in data
        if text.strip() not in {"-", ""}
    ]
    
    logger.debug(f"Processing {len(data_cleaned)} cleaned OCR items")
    
    result: dict[str, list[Any]] = {}
    nik_image_box = None

    def update_result(field, value, score):
        # value != "" must guard ALL inserts, not just overwrites: 'and' binds
        # tighter than 'or', so the original let an empty first-seen value be
        # stored, inflating len(result) and suppressing the remapping fallback (BUG-14).
        if value != "" and (field not in result or result[field][1] <= score):
            result[field] = [value, score]
            logger.debug(f"Updated field '{field}' with value '{value}' (score: {score})")

    for i, dat in enumerate(data_cleaned):
        next_data: list = data_cleaned[i + 1] if i + 1 < len(data_cleaned) else ["", 0, []]
        for keydat in keydata:
            score = fuzz_ratio(dat[0], keydat)
            meet_threshold = score >= THRESHOLD_RATIO and dat[1] > THRESHOLD_CONFIDENCE
            if meet_threshold:
                if keydat == "nik":
                    if len(next_data[0]) < 14 and i + 2 < len(data_cleaned):
                        next_data = data_cleaned[i + 2]
                    update_result("nik", matching_nik(next_data), score)
                    nik_image_box = next_data[2]
                elif keydat == "nama":
                    if i + 2 < len(data_cleaned) and not any(word in data_cleaned[i + 2][0].lower() for word in ["tempat", "tgl", "lahir"]) and data_cleaned[i + 2][1] >= THRESHOLD_CONFIDENCE:
                        next_data = [f"{next_data[0] + ' ' + data_cleaned[i + 2][0]}", next_data[1], next_data[2]]
                    update_result("nama", matching_nama(next_data), score)
                elif keydat == "tempat/tgl lahir":
                    angka_indat = ''.join(re.findall(r"\d+", next_data[0]))
                    if i + 2 < len(data_cleaned) and not any(word in data_cleaned[i + 2][0].lower() for word in ["jenis", "kelamin", "laki", "perempuan", "perem", "gol", "darah"]) and data_cleaned[i + 2][1] >= THRESHOLD_CONFIDENCE and len(angka_indat) < 5:
                        next_data = [f"{next_data[0] + ' ' + data_cleaned[i + 2][0]}", next_data[1], next_data[2]]
                    tempat_lahir, tanggal_lahir = matching_tempatlahir(dat, next_data)
                    if tempat_lahir == "" or tanggal_lahir == "":
                        tempat_lahir, tanggal_lahir = matching_tempatlahir_new(dat, next_data)
                    update_result("tempat_lahir", tempat_lahir, score)
                    update_result("tanggal_lahir", tanggal_lahir, score)
                elif keydat in keydata_jeniskelamin:
                    update_result("jenis_kelamin", matching_jeniskelamin(keydat, dat, next_data), score)
                elif keydat == "alamat":
                    if i + 2 < len(data_cleaned):
                        if not any(word in data_cleaned[i + 2][0].lower() for word in ["rt", "rw", "rt/rw"]) and data_cleaned[i + 2][1] >= THRESHOLD_CONFIDENCE:
                            if not any(word in next_data[0].lower() for word in ["gol", "darah", "gol.darah"]):
                                next_data = [f"{next_data[0] + ' ' + data_cleaned[i + 2][0]}", next_data[1], next_data[2]]
                            else:
                                next_data = data_cleaned[i + 2]
                    update_result("alamat", matching_alamat(next_data), score)
                elif keydat == "rt/rw":
                    rt, rw = matching_rtrw(next_data)
                    update_result("rt", rt, score)
                    update_result("rw", rw, score)
                elif keydat == "kel/desa":
                    update_result("kel_desa", matching_keldesa(next_data), score)
                elif keydat == "kecamatan":
                    update_result("kecamatan", matching_kecamatan(dat, next_data), score)
                elif keydat in keydata_agama:
                    update_result("agama", matching_agama(keydat, dat, next_data), score)
                elif keydat in keydata_status:
                    update_result("status_perkawinan", matching_status(keydat, dat, next_data), score)
                elif keydat == "pekerjaan":
                    # Skip cell yang hanya berisi colon/whitespace, lompat ke i+2
                    actual_next = next_data
                    next_text = next_data[0] if next_data and len(next_data) > 0 else ""
                    if not next_text.strip().strip(":：").strip() and i + 2 < len(data_cleaned):
                        actual_next = data_cleaned[i + 2]
                    prev_data = data_cleaned[i - 1] if i > 0 else ["", 0, []]
                    update_result("pekerjaan", matching_pekerjaan(keydat, prev_data, dat, actual_next), score)

    # Remapping for missing fields
    if len(result) < 13:
        logger.debug(f"Only {len(result)} fields found, running remapping...")
        result = remapping(data_cleaned, result)

    # Fallback: detect NIK by its 16-digit pattern when the "NIK" label was
    # missed by OCR, so the label-anchored pass above never fired. Scan every
    # cell and reuse matching_nik, which enforces the confidence gate and that
    # the cleaned value is exactly 16 digits.
    if not result.get("nik", [""])[0]:
        for cell in data_cleaned:
            candidate = matching_nik(cell)
            if candidate:
                result["nik"] = [candidate, cell[1]]
                nik_image_box = cell[2]
                logger.debug(f"NIK recovered by 16-digit detection: {candidate}")
                break

    # Only return the value, not the score
    final_result = {k: v[0] for k, v in result.items()}
    # logger.info(f"Extracted {len(final_result)} fields: {list(final_result.keys())}")
    
    return final_result, nik_image_box
