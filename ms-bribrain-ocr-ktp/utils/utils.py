from utils.helper import (
    fuzz_ratio, fuzz_partial,
    matching_nik, matching_nama, matching_agama, matching_tempatlahir_new, matching_tempatlahir,
    matching_alamat, matching_jeniskelamin, matching_rtrw, matching_status,
    matching_keldesa, matching_kecamatan
)
from utils.prefix import keydata, keydata_agama, keydata_jeniskelamin, keydata_status, keydata_repeat
from dotenv import load_dotenv
import os

load_dotenv()
threshold_ratio = float(os.environ["THRESHOLD_RATIO"])
threshold_partial = float(os.environ["THRESHOLD_PARTIAL"])
threshold_confidence = float(os.environ["THRESHOLD_CONFIDENCE"])

def mappingnext(data: list, image_bytes:bytes) -> dict:
    """
    Memetakan hasil OCR ke field KTP dengan fuzzy matching dan validasi confidence.

    Args:
        data (list): List hasil OCR, setiap elemen [text, confidence].

    Returns:
        dict: Hasil mapping field KTP.
    """
    # data = [line[1] for line in data]
    data_cleaned = [
    [text, score, box]
    for box, (text, score) in data
    if text.strip() != '-' and len(text.strip()) > 2
]
    result = {}

    def update_result(field, value, score):
        if field not in result or (result[field][1] <= score and value != ""):
            result[field] = [value, score]

    for i, dat in enumerate(data_cleaned):
        next_data = data_cleaned[i+1] if i+1 < len(data_cleaned) else ["", 0]
        for keydat in keydata:
            score = fuzz_ratio(dat[0], keydat)
            meet_threshold = score >= threshold_ratio and dat[1] > threshold_confidence
            if meet_threshold:
                if keydat == "nik":
                    update_result("nik", matching_nik(next_data), score)
                elif keydat == "nama":
                    if not any(word in data_cleaned[i+2][0].lower() for word in ["tempat", "tgl", "lahir"]) and data_cleaned[i+2][1] >= threshold_confidence:
                        next_data = [f"{next_data[0] + ' ' + data_cleaned[i+2][0]}", next_data[1], next_data[2]]
                    update_result("nama", matching_nama(next_data), score)
                elif keydat == "tempat/tgl lahir":
                    if not any(word in data_cleaned[i+2][0].lower() for word in ["jenis", "kelamin", "laki", "perempuan", "perem", "gol", "darah"]) and data_cleaned[i+2][1] >= threshold_confidence:
                        next_data = [f"{next_data[0] + ' ' + data_cleaned[i+2][0]}", next_data[1], next_data[2]]
                    tempat_lahir, tanggal_lahir = matching_tempatlahir(dat, next_data)
                    if tempat_lahir == "" or tanggal_lahir == "":
                        tempat_lahir, tanggal_lahir = matching_tempatlahir_new(dat, next_data)
                    update_result("tempat_lahir", tempat_lahir, score)
                    update_result("tanggal_lahir", tanggal_lahir, score)
                elif keydat in keydata_jeniskelamin:
                    update_result("jenis_kelamin", matching_jeniskelamin(keydat, dat, next_data), score)
                elif keydat == "alamat":
                    if not any(word in data_cleaned[i+2][0].lower() for word in ["rt", "rw", "rt/rw"]) and data_cleaned[i+2][1] >= threshold_confidence:
                        if not any(word in next_data[0].lower() for word in ["gol", "darah", "gol.darah"]):
                            next_data = [f"{next_data[0] + ' ' + data_cleaned[i+2][0]}", next_data[1], next_data[2]]
                        else:
                            next_data = data_cleaned[i+2]
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

    # Remapping for missing fields
    if len(result) < 12:
        result = remapping(data_cleaned, result)

    # Only return the value, not the score
    return {k: v[0] for k, v in result.items()}

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

    print("Remapping initiated...")

    def update_result(field, value, score):
        if field not in result or (result[field][1] <= score and value != ""):
            result[field] = [value, score]

    for i, dat in enumerate(data_cleaned):
        next_data = data_cleaned[i+1] if i+1 < len(data_cleaned) else ["", 0]
        for keydat in keydata_repeat:
            score = fuzz_partial(dat[0], keydat)
            meet_threshold = score >= threshold_ratio and dat[1] > threshold_confidence
            meet_threshold_partial = score >= threshold_partial and dat[1] > threshold_confidence

            if keydat == "kel/desa" and meet_threshold:
                update_result("kel_desa", matching_keldesa(next_data), score)
            elif keydat == "kecamatan" and meet_threshold:
                update_result("kecamatan", matching_kecamatan(dat, next_data), score)
            elif keydat in keydata_status and meet_threshold_partial:
                update_result("status_perkawinan", matching_status(keydat, dat, next_data), score)
            elif keydat == "tempat/tgl lahir" and meet_threshold_partial:
                if not any(word in data_cleaned[i+2][0].lower() for word in ["jenis", "kelamin", "laki", "perempuan", "perem"]) and data_cleaned[i+2][1] >= threshold_confidence:
                    next_data = [f"{next_data[0] + ' ' + data_cleaned[i+2][0]}", next_data[1], next_data[2]]
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