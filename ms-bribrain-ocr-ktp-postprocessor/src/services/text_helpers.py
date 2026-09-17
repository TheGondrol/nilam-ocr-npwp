"""Text processing helper functions.

This module contains utility functions for text processing, fuzzy matching,
character correction, and text cleaning operations.
"""

import re
from typing import Tuple

from rapidfuzz import fuzz, process

from src.services.constants import (
    alphabet_mapping,
    digit_mapping,
    keydata,
    keyid,
    mapping_agama,
    mapping_jeniskelamin,
    mapping_pekerjaan,
    mapping_status,
    message_blur,
    message_glare,
    message_rotated,
    symbol_mapping,
    REGEX,
)


def filter_result(result, width):
    """
    Memfilter hasil OCR berdasarkan posisi X agar tidak melewati ambang batas lebar dokumen.

    Args:
        result (list): Hasil OCR berupa list berisi koordinat dan teks.
        width (int | float): Lebar dokumen atau gambar.

    Returns:
        list: Baris hasil OCR yang semua titik X-nya berada di dalam ambang batas.
    """
    threshold = (width / 5) * 4
    filtered_line = []
    for line in result:
        cordinate = line[0]
        if all(float(c[0]) <= threshold for c in cordinate):  # semua titik X harus <= threshold
            filtered_line.append(line)
    return filtered_line


def fuzz_ratio(query, data):
    """
    Menghitung skor kemiripan (similarity score) antara dua string
    menggunakan metode rasio fuzzy (fuzz.ratio) dari RapidFuzz.

    Args:
        query (str): String pertama yang akan dibandingkan.
        data (str): String kedua yang akan dibandingkan.

    Returns:
        int: Skor kemiripan antara 0-100, semakin tinggi semakin mirip.
    """
    score = fuzz.ratio(query.lower(), data.lower())
    return score


def fuzz_partial(query, data):
    """
    Menghitung skor kemiripan (similarity score) antara dua string
    menggunakan metode partial_ratio dari RapidFuzz.

    Args:
        query (str): String pertama yang akan dibandingkan.
        data (str): String kedua yang akan dibandingkan.

    Returns:
        int: Skor kemiripan antara 0-100, semakin tinggi semakin mirip.
    """
    score = fuzz.partial_ratio(query.lower(), data.lower())
    return score


def correct_alphabets_to_digits(input_string):
    """
    Corrects characters in the input string using the provided correction mapping.
    A character is corrected only if it is between digits.

    Parameters:
        input_string (str): The string to process.

    Returns:
        str: The corrected string with replacements applied.
    """
    corrected = []
    n = len(input_string)

    for i, char in enumerate(input_string):
        # Check if the current character is in symbol_mapping
        if char in symbol_mapping:
            # Check if the character is between digits
            if (
                i > 0
                and i < n - 1
                and input_string[i - 1].isdigit()
                and input_string[i + 1].isdigit()
            ):
                corrected.append(str(symbol_mapping[char]))
            else:
                corrected.append(char)  # Skip correction

        elif char in alphabet_mapping:
            # Check if the character is between digits
            if (
                i > 0
                and i < n - 1
                and input_string[i - 1].isdigit()
                and input_string[i + 1].isdigit()
            ):
                corrected.append(str(alphabet_mapping[char]))
            else:
                corrected.append(char)  # Skip correction
        else:
            corrected.append(char)  # Append as is if not in correction_mapping

    return "".join(corrected)


def correct_digits_to_alphabets(input_string):
    """
    Corrects characters in the input string using the provided correction mapping.
    A character is corrected only if it is between alphabets.

    Parameters:
        input_string (str): The string to process.

    Returns:
        str: The corrected string with replacements applied.
    """
    corrected = []
    n = len(input_string)

    for i, char in enumerate(input_string):
        # Check if the current character is in digit_mapping
        if char in digit_mapping:
            # Check if the character is between alphabets
            if (
                i > 0
                and i < n - 1
                and input_string[i - 1].isalpha()
                and input_string[i + 1].isalpha()
            ):
                corrected.append(str(digit_mapping[char]))
            else:
                corrected.append(char)  # Skip correction
        else:
            corrected.append(char)  # Append as is if not in correction_mapping

    return "".join(corrected)


def searchmapping(field: str, query: str) -> Tuple:
    """
    Melakukan pencarian fuzzy pada mapping field tertentu.

    Args:
        field (str): Nama field yang akan dicari ("pekerjaan", "agama", "status_perkawinan", "jenis_kelamin").
        query (str): Query string yang akan dicocokkan.

    Returns:
        tuple: (hasil_mapping, skor_kemiripan) jika ditemukan, jika tidak ditemukan (None, 0).
    """
    mapping_dict = {
        "pekerjaan": mapping_pekerjaan,
        "agama": mapping_agama,
        "status_perkawinan": mapping_status,
        "jenis_kelamin": mapping_jeniskelamin,
    }
    listmap = mapping_dict.get(field)
    if not listmap:
        return None, 0

    match = process.extractOne(query.upper(), listmap, scorer=fuzz.ratio)
    if match:
        return match[0], match[1]
    return None, 0


def clean_colon(text: str) -> str:
    """
    Menghapus karakter ':' dan spasi di awal string.

    Args:
        text (str): Teks yang akan dibersihkan.

    Returns:
        str: Teks tanpa ':' dan spasi di awal.
    """
    if not text:
        return ""
    clean = re.sub(REGEX.get("AMBIL_COLON_SPASI_AWAL", r"^[:\s]+"), "", text)
    return clean


def getdigitonly(text: str) -> str:
    """
    Mengambil hanya digit dari string (menghapus spasi dan karakter non-digit).

    Args:
        text (str): String yang akan diekstrak digitnya.

    Returns:
        str: String hanya berisi digit.
    """
    if not text:
        return ""
    cleaned = re.sub(REGEX.get("AMBIL_NON_DIGIT", r"\D"), "", text)
    return cleaned


def data_verification(data, valuetype):
    """
    Verifikasi data untuk memastikan tidak mengandung kata kunci field.

    Args:
        data: Data yang akan diverifikasi
        valuetype: Tipe nilai ("normal" atau lainnya)

    Returns:
        str: Data yang sudah diverifikasi atau string kosong jika tidak valid
    """
    if valuetype == "normal":
        if data.lower() in keydata:
            return ""
        else:
            return data
    else:
        if data.lower() in keyid:
            return ""
        else:
            return data


def create_err_message(is_blurry, is_glare, is_rotated):
    """
    Membuat pesan error gabungan berdasarkan kondisi kualitas gambar.

    Args:
        is_blurry (bool): True jika gambar terdeteksi blur.
        is_glare (bool): True jika gambar terdeteksi glare (pantulan cahaya).
        is_rotated (bool): True jika gambar terdeteksi terbalik/rotasi.

    Returns:
        str: Pesan error yang menggabungkan semua kondisi yang terdeteksi.
    """
    message = []
    if is_blurry:
        message.append(message_blur)
    if is_glare:
        message.append(message_glare)
    if is_rotated:
        message.append(message_rotated)
    if len(message) > 1:
        result = " ".join(message[:-1]) + " dan " + message[-1]
    else:
        result = message[0]

    return result
