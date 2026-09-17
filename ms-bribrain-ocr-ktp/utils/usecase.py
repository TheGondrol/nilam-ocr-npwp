import numpy as np
from dotenv import load_dotenv
import re
import os
import time
import json
from rapidfuzz import fuzz, process
import statistics
# from ppocr.utils.logging import get_logger
import logging
from utils.prefix import mapping_pekerjaan, mapping_status, mapping_agama, mapping_jeniskelamin, symbol_mapping, alphabet_mapping, digit_mapping, bulan_dict, keydata, keyid, message_blur, message_glare, message_rotated, message_median
# from pymilvus import model, MilvusClient

load_dotenv()
THRESHOLD_RATIO = float(os.environ["THRESHOLD_RATIO"])
THRESHOLD_PARTIAL = float(os.environ["THRESHOLD_PARTIAL"])
THRESHOLD_CONFIDENCE = float(os.environ["THRESHOLD_CONFIDENCE"])
THRESHOLD_BLUR = float(os.environ["THRESHOLD_BLUR"])
THRESHOLD_MEDIAN = float(os.environ["THRESHOLD_MEDIAN"])
THRESHOLD_MIN_AREA_GLARE=int(os.environ["THRESHOLD_MIN_AREA_GLARE"])
GLARE_PADDING_SIZE=int(os.environ["GLARE_PADDING_SIZE"])

# Configure logger
# logger = get_logger()
# logger.setLevel(logging.ERROR)

def filter_result(result, width):
    """
    Memfilter hasil OCR berdasarkan posisi X agar tidak melewati ambang batas lebar dokumen.

    Args:
        result (list): Hasil OCR berupa list berisi koordinat dan teks.
        width (int | float): Lebar dokumen atau gambar.

    Returns:
        list: Baris hasil OCR yang semua titik X-nya berada di dalam ambang batas.
    """
    threshold = (width/5)*4
    filtered_line = []
    for line in result:
        cordinate = line[0]
        # print("Cordinate:", cordinate)
        if all(float(c[0]) <= threshold for c in cordinate):  # semua titik X harus <= 550
            filtered_line.append(line)
    return filtered_line

#check median confidence score dari hasil ocr
def check_confidencemedian(results: list) -> bool:
    """
    Mengecek apakah median confidence hasil OCR memenuhi threshold.

    Args:
        results (list): List hasil OCR, setiap elemen tuple (text, confidence).

    Returns:
        bool: True jika median confidence >= THRESHOLD_MEDIAN, False jika tidak.
    """
    if not results:
        return 0
    results = [line[1] for line in results]
    scores = [score for _, score in results if isinstance(score, (int, float))]
    if not scores:
        return 0
    median_score = statistics.median(scores)
    print("median score", median_score)
    return median_score <= THRESHOLD_MEDIAN


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

def meet_tgllength(parts: list) -> tuple:
    """
    Mengekstrak tempat, hari, bulan, dan tahun dari list hasil split tanggal lahir.

    Args:
        parts (list): List hasil split string tanggal lahir.

    Returns:
        tuple: (tempat, day, month, year) jika berhasil, jika gagal semuanya None.
    """
    try:
        # Normalisasi panjang parts
        if len(parts) >= 4:
            tempat = ' '.join(parts[:-3])
            day = parts[-3]
            month_raw = parts[-2]
            year = parts[-1]
        else:
            # Fallback jika parts kurang dari 4 elemen
            tempat = parts[0] if len(parts) > 0 else None
            day = parts[1][:2] if len(parts) > 1 else None
            month_raw = parts[2].lower() if len(parts) > 2 else None
            year = parts[3] if len(parts) > 3 else None

        # Konversi bulan jika perlu
        month = bulan_dict.get(month_raw, month_raw.zfill(2) if month_raw and month_raw.isdigit() else None)

        # Jika tempat mengandung digit, pisahkan
        print(f"Ekstrak tempat: {tempat}, day: {day}, month_raw: {month_raw}, year: {year}")
        tempat_clear = False
        if tempat:
            tempat = re.sub(r"^[^A-Za-z]+", "", tempat)  # buang karakter non-huruf dari awal
            colon_match = re.search(r"[：:]\s*(\S+)", tempat)
            if colon_match:
                tempat = colon_match.group(1)
            tempat = correct_digits_to_alphabets(tempat)
            match = re.match(r"([A-Za-z\s]+)", tempat)
            if match:
                tempat_clear = True
                tempat = match.group(1)
                print(f"Tempat terpisah: {tempat}{tempat_clear}")
                # day = match.group(2)
                # month_raw dan year tetap dari sebelumnya
        
        if day and len(day) > 2:
            match = re.match(r"([A-Za-z]+)(\d+)", day)
            if match:
                if tempat_clear:
                    # Jika tempat sudah dipisahkan, ambil day dan month dari match
                    tempat = f"{tempat} {match.group(1)}"
                    print(f"Tempat sudah dipisahkan: {tempat}")
                else:
                    tempat = match.group(1)
                day = match.group(2)

        # Validasi hasil
        if all([tempat, day, month, year]) and day.isdigit() and year.isdigit():
            return tempat, day, month, year
        else:
            return None, None, None, None
    except Exception as e:
        print(f"Gagal ekstrak tgl lahir di meet_tgllength: {e}")
        return None, None, None, None

def notmeet_tgllength_jadi_satu(data: str) -> tuple:
    """
    Ekstrak tempat, hari, bulan, dan tahun dari string TTL yang menyatu 
    (misal: 'Surabaya12051990' atau 'Medan-12/05/1990').

    Args:
        data (str): String hasil OCR tempat/tanggal lahir tanpa pemisah konsisten.

    Returns:
        tuple: (tempat, day, month_raw, year) atau (None, None, None, None) jika gagal.
    """
    try:
        colon_match = re.search(r"[：:]\s*(\S+)", data)
        if colon_match:
            data = colon_match.group(1)
        match = re.match(r"([A-Za-z]+)([\d\-\/\.]+)", data)
        if match:
            tempat = match.group(1)
            angka = re.findall(r"\d+", match.group(2))
            if len(angka) == 3:
                day, month_raw, year = angka
            elif len(angka) == 1 and len(angka[0]) == 8:
                # Format tanpa pemisah, misal 12051990
                day = angka[0][:2]
                month_raw = angka[0][2:4]
                year = angka[0][4:]
            return tempat, day, month_raw, year
        else:
            # Fallback: ekstrak semua huruf dan angka
            data = correct_alphabets_to_digits(data)
            tempat = ''.join(re.findall(r"[A-Za-z]+", data))
            angka = ''.join(re.findall(r"\d+", data))
            if len(angka) == 8:
                day = angka[:2]
                month_raw = angka[2:4]
                year = angka[4:]
            return tempat, day, month_raw, year
    except Exception as e:
        print(f"Gagal ekstrak data ttl jadi satu : {e}")
        return None, None, None, None
    
def notmeet_tgllength_terpisah(data: str) -> tuple:
    """
    Ekstrak tempat, hari, bulan, dan tahun dari string TTL dengan pemisah 
    (misal: 'Jakarta12-05-1990' atau 'Medan 12/05/1990').

    Args:
        data (str): String hasil OCR tempat/tanggal lahir dengan pemisah tanggal.

    Returns:
        tuple: (tempat, day, month_raw, year) atau (None, None, None, None) jika gagal.
    """
    try:
        match = re.match(r"([A-Za-z]+)(\d{2})[\/\-.](\d{2})[\/\-.](\d{4})", data)
        if not match:
            match = re.match(r"([A-Za-z]+)[\s\.\-]?(\d{2})(\d{2})[\/\-.](\d{4})", data)
        if match:
            tempat = match.group(1)
            day = match.group(2)
            month_raw = match.group(3)
            year = match.group(4)
        else:
            # Fallback: ekstrak semua huruf dan angka
            data = correct_digits_to_alphabets(data)
            tempat = ''.join(re.findall(r"[A-Za-z]+", data))
            angka = ''.join(re.findall(r"\d+", data))
            if len(angka) == 8:
                day = angka[:2]
                month_raw = angka[2:4]
                year = angka[4:]
        return tempat, day, month_raw, year
    except Exception as e:
        print(f"Gagal ekstrak data ttl jadi satu : {e}")
        return None, None, None, None

def notmeet_tgllength(data: str, next_line: bool) -> tuple:
    """
    Mengekstrak tempat, hari, bulan, dan tahun dari string tanggal lahir yang formatnya tidak standar.

    Args:
        data (str): String hasil OCR yang mengandung tempat/tanggal lahir.
        next_line (bool): True jika data berasal dari baris berikutnya.

    Returns:
        tuple: (tempat, day, month, year) jika berhasil, jika gagal semuanya None.
    """
    if not data or len(data) == 0:
        return None, None, None, None

    try:
        tempat, day, month, year = None, None, None, None
        match = None

        if not next_line:
            # Contoh: "Tempat:KOTA12-05-1990"
            tempat, day, month_raw, year = notmeet_tgllength_jadi_satu(data)
        else:
            # Contoh: "KOTA12/05/1990" atau "KOTA 12051990"
            tempat, day, month_raw, year = notmeet_tgllength_terpisah(data)
            
        # Mapping bulan jika perlu
        if month_raw:
            month = bulan_dict.get(month_raw.lower(), month_raw.zfill(2) if month_raw.isdigit() else None)
        else:
            month = None

        if all([tempat, day, month, year]) and day.isdigit() and year.isdigit():
            return tempat, day, month, year
        else:
            return None, None, None, None

    except Exception as e:
        print(f"Gagal ekstrak tgl lahir di notmeet_tgllength: {e}")
        return None, None, None, None

def correct_alphabets_to_digits(input_string):
    # def correct_string(input_string, correction_mapping):
    """
    Corrects characters in the input string using the provided correction mapping.
    A character is corrected only if it is between digits.

    Parameters:
        input_string (str): The string to process.
        correction_mapping (dict): A dictionary mapping characters to their replacements.

    Returns:
        str: The corrected string with replacements applied.
    """

    corrected = []
    n = len(input_string)

    for i, char in enumerate(input_string):
        # Check if the current character is in correction_mapping
        if char in symbol_mapping:
            corrected.append(str(symbol_mapping[char]))

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
    # def correct_string(input_string, correction_mapping):
    """
    Corrects characters in the input string using the provided correction mapping.
    A character is corrected only if it is between digits.

    Parameters:
        input_string (str): The string to process.
        correction_mapping (dict): A dictionary mapping characters to their replacements.

    Returns:
        str: The corrected string with replacements applied.
    """

    corrected = []
    n = len(input_string)

    for i, char in enumerate(input_string):
        # Check if the current character is in correction_mapping
        if char in digit_mapping:
            # Check if the character is between digits
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

def searchmapping(field: str, query: str) -> tuple:
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
    clean = re.sub(r"^[:\s]+", "", text)
    return clean

def split_tgllahir(text: str) -> list:
    """
    Memecah string tanggal lahir menjadi list bagian-bagian (tempat, hari, bulan, tahun).

    Args:
        text (str): String yang berisi tempat/tanggal lahir.

    Returns:
        list: List hasil split, elemen kosong dihapus.
    """
    if not text:
        return []
    try:
        text = re.sub(r"[^A-Za-z0-9\s]", "", text)
        pattern = r"\b([A-Z][A-Z]+.*)"
        match = re.search(pattern, text)
        text = match.group(1) if match else text
        tgl_split = [part for part in re.split(r"[ /,\-\.]+", text) if part]
        return tgl_split
    except Exception as e:
        print(f"Gagal split tgl lahir: {e}")
        return []

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
    cleaned = re.sub(r"\D", "", text)
    return cleaned

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
            if len(dat) == 16:
                return dat
        return ""
    except Exception as e:
        print(f"Gagal ekstrak NIK: {e}")
        return ""

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
            nama = re.sub(r'[^a-zA-Z0-9\s]', ' ', nama)

            # Hilangkan spasi berlebih
            nama = re.sub(r'\s+', ' ', nama).strip()
            return nama.upper()
        return ""
    except Exception as e:
        print(f"Gagal ekstrak nama: {e}")
        return ""
    
def format_ttl_new_sesuai(data):
    try:
        digit_data = len(re.findall(r"\d", str(data[1])))
        if len(data[0]) >= 3 and digit_data >= 5:
            number = re.sub(r'[^a-zA-Z0-9]', '', data[1])
            tempat = data[0]
            print("caps:", tempat)
            print("numbers:", number)
        else:
            # ekstrak tempat
            caps = re.findall(r"[A-Z][A-Z\.\- ]+", data)
            print("caps:", caps)
            # tempat = max(caps, key=len) if caps else ""
            tempat = tempat.replace(".", "").replace("-", " ").strip()

            # ekstrak angka
            numbers = re.findall(r"\d+", data)
            number = "".join(numbers)
            print("numbers:", numbers, "=>", number)
        return tempat, number
    except Exception as e:
        print(f"Gagal format data ttl new sesuai : {e}")
        return "", ""

def format_ttl_new_tidak_sesuai(data):
    try:
        caps = re.findall(r"[A-Z][A-Z\.\- ]+", data)
        print("caps:", caps)
        tempat = max(caps, key=len) if caps else ""
        tempat = tempat.replace(".", "").replace("-", " ").strip()

        # ekstrak angka
        numbers = re.findall(r"\d+", data)
        number = "".join(numbers)
        print("numbers:", numbers, "=>", number)
        return tempat, number
    except Exception as e:
        print(f"Gagal format data ttl new tidak sesuai : {e}")
        return "", ""


def matching_tempatlahir_new(data: list, next_data: list) -> tuple:
    dat = data[0] if data and len(data) > 0 else ""
    next_dat = next_data[0] if next_data and len(next_data) > 0 else ""

    try:
        digit_data = len(re.findall(r"\d", str(dat)))
        digit_nextdata = len(re.findall(r"\d", str(next_dat)))

        # pilih kandidat mana yg lebih cocok
        proses = dat if digit_data >= 6 and digit_nextdata < 6 else next_dat
        if len(proses) <= 12:  # kalau terlalu pendek, gabungkan
            proses = f"{dat} {next_dat}"
        print("proses:", proses)

        # cari teks kapital
        pattern = r"([A-Z]{2,}.*)"
        match = re.search(pattern, proses)
        if not match:
            print("Tidak ada match")
            return "", ""

        content = match.group(1)
        print("match:", content)

        content = re.split(r'[,.]', content)
        if len(content) == 2:
            tempat, number = format_ttl_new_sesuai(content)
        else:
            # ekstrak tempat
            tempat, number = format_ttl_new_tidak_sesuai(proses)

        if number and not number.isdigit():
            number = correct_alphabets_to_digits(number)
            print("corrected numbers:", number)

        day = month_raw = year = None
        if len(number) == 8:  # format DDMMYYYY
            day = number[:2]
            month_raw = number[2:4]
            year = number[4:]

        if all([day, month_raw, year, tempat]):
            month = bulan_dict.get(month_raw, month_raw)
            tgl = f"{day} {month} {year}"
            return tempat, tgl

        return "", ""

    except Exception as e:
        print(f"Gagal ekstrak tgl lahir di matching_tempatlahir_new: {e}")
        return "", ""
        
def validate_ttl_data(data, next_data):
    try:
        digit_data = len(re.findall(r"\d", data))
        digit_nextdata = len(re.findall(r"\d", next_data))

        # Cek jika next_data valid dan mengandung kata "jenis"
        if next_data and len(next_data) > 1 and digit_data >= 6 and digit_nextdata < 6:
            if data and len(data) > 1 and data[1] > THRESHOLD_CONFIDENCE:
                print("next_data:", next_data)
                match = re.search(r"[：:]\s*(.*)", data)
                if match:
                    dat = match.group(1)
                parts = split_tgllahir(dat)
        elif digit_nextdata >= 6:
            if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE:
                clean = clean_colon(next_data)
                parts = split_tgllahir(clean)
        return parts
    except Exception as e:
        print(f"Error Validating ttl_data : {e}")
        return []
    
def matching_tempatlahir(data: list, next_data: list) -> tuple:
    """
    Mengekstrak tempat dan tanggal lahir dari dua baris data.

    Args:
        data (list): Data OCR pada baris saat ini, format [text, confidence].
        next_data (list): Data OCR pada baris berikutnya, format [text, confidence].

    Returns:
        tuple: (tempat, tanggal_lahir) jika berhasil, jika gagal (None, None).
    """
    parts = []
    tempat, day, month, year = None, None, None, None

    dat = data[0] if data and len(data) > 0 else ""
    next_dat = next_data[0] if next_data and len(next_data) > 0 else ""

    parts = validate_ttl_data(dat, next_dat)

    print("parts:", parts)
    if len(parts) >= 4 and "jenis" not in next_dat.lower():
        tempat, day, month, year = meet_tgllength(parts)
    else:
        # Step 2: fallback — detect word+digit combo directly
        if next_data and len(next_data) > 1 and "jenis" in next_dat.lower():
            if data and len(data) > 1 and data[1] > THRESHOLD_CONFIDENCE:
                tempat, day, month, year = notmeet_tgllength(dat, False)
        else:
            if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE:
                tempat, day, month, year = notmeet_tgllength(next_dat, True)

    if tempat and day and month and year:
        tgl = f"{day.zfill(2)}-{month}-{year.zfill(4)}"
        tgl = re.sub(r"[,.-]", " ", tgl)
        tgl = f"{tgl[:2]} {tgl[2:4]} {tgl[4:]}" if len(tgl) == 8 else tgl
        if len(tempat) > 9:
            match = re.findall(r'[A-Z]{2,}', tempat)
            tempat = " ".join(match)
        return tempat, tgl

    return "", ""
    
##Matching Jenis Kelamin
def matching_jeniskelamin(key: str, data: list, next_data: list) -> str:
    """
    Melakukan mapping jenis kelamin menggunakan fuzzy matching.

    Args:
        key (str): Key field yang sedang diproses.
        data (list): Data OCR pada baris saat ini, format [text, confidence].
        next_data (list): Data OCR pada baris berikutnya, format [text, confidence].

    Returns:
        str: Hasil mapping jenis kelamin, atau string kosong jika tidak ditemukan.
    """
    clean = ""
    try:
        if key == "jenis kelamin":
            if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE:
                clean = clean_colon(next_data[0])
        else:
            if data and len(data) > 1 and data[1] > THRESHOLD_CONFIDENCE:
                clean = clean_colon(data[0])
        if clean:
            jeniskelamin, scorea = searchmapping("jenis_kelamin", clean)
            if scorea >= THRESHOLD_RATIO:
                return jeniskelamin
        return ""
    except Exception as e:
        print(f"Gagal ekstrak jenis kelamin: {e}")
        return ""

def matching_alamat(key: str, data: list, next_data: list) -> str:
    """
    Mengambil dan membersihkan alamat dari data berikutnya jika confidence mencukupi.

    Args:
        key (str): Key field yang sedang diproses.
        data (list): Data OCR pada baris saat ini, format [text, confidence].
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
        print(f"Gagal ekstrak alamat di matching_alamat: {e}")
        return ""

def rtrw_tidak_sesuai_format(rtrw: str) -> tuple:
    final = {}
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
        print("final:", final, "rt:", rt, "rw:", rw)
        if rt.isdigit() and rw.isdigit():
            rt = rt.zfill(3)
            rw = rw.zfill(3)
            return rt, rw
    except Exception as e:
        print(f"gagal ekstrak rtrw : {e}")
        return "", ""

def rtrw_sesuai_format(rtrw: str) -> tuple:
    try:
        if "/" in rtrw:
            rt, rw = rtrw.split("/")
        else:
            rt, rw = rtrw[:3], rtrw[3:]
        if rt.isdigit() and rw.isdigit():
            rt = rt.zfill(3)
            rw = rw.zfill(3)
            return rt, rw
    except Exception as e:
        print(f"gagal ekstrak rtrw : {e}")
        return "", ""

def matching_rtrw(next_data: list) -> tuple:
    """
    Mengekstrak RT dan RW dari data berikutnya jika confidence mencukupi.

    Args:
        key (str): Key field yang sedang diproses.
        data (list): Data OCR pada baris saat ini, format [text, confidence].
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

def matching_keldesa(key: str, data: list, next_data: list) -> str:
    """
    Mengambil dan membersihkan kelurahan/desa dari data berikutnya jika confidence mencukupi.

    Args:
        key (str): Key field yang sedang diproses.
        data (list): Data OCR pada baris saat ini, format [text, confidence].
        next_data (list): Data OCR pada baris berikutnya, format [text, confidence].

    Returns:
        str: Kelurahan/Desa dalam huruf kapital, atau string kosong jika tidak valid.
    """
    try:
        if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE:
            clean = clean_colon(next_data[0])
            clean = data_verification(clean, "normal")
            return clean.upper()
        return ""
    except Exception as e:
        print(f"Gagal ekstrak kelurahan/desa di matching_keldesa: {e}")
        return ""

def matching_kecamatan(key: str, data: list, next_data: list) -> str:
    """
    Mengambil dan membersihkan kecamatan dari data berikutnya jika confidence mencukupi.

    Args:
        key (str): Key field yang sedang diproses.
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
        match = re.findall(r'[A-Z]{2,}', clean)
        if match:
            kecamatan = " ".join(match)
            return kecamatan.upper()
        return ""
    except Exception as e:
        print(f"Gagal ekstrak kecamatan di matching_kecamatan: {e}")
        return ""

def matching_agama(key: str, data: list, next_data: list) -> str:
    """
    Melakukan mapping agama menggunakan fuzzy matching.

    Args:
        key (str): Key field yang sedang diproses.
        data (list): Data OCR pada baris saat ini, format [text, confidence].
        next_data (list): Data OCR pada baris berikutnya, format [text, confidence].

    Returns:
        str: Hasil mapping agama, atau string kosong jika tidak ditemukan.
    """
    clean = ""
    try:
        if key == "agama":
            if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE:
                clean = clean_colon(next_data[0])
        else:
            if data and len(data) > 1 and data[1] > THRESHOLD_CONFIDENCE:
                clean = clean_colon(data[0])
        if clean:
            a, scorea = searchmapping("agama", clean)
            if scorea >= THRESHOLD_RATIO:
                return a
        return ""
    except Exception as e:
        print(f"Gagal ekstrak agama di matching_agama: {e}")
        return ""

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
                prefix_sp = re.search(r'[A-Z]{2,}.*', next_data[0])
        else:
            if data and len(data) > 1 and data[1] > THRESHOLD_CONFIDENCE:
                prefix_sp = re.search(r'[A-Z]{2,}.*', data[0])
        if prefix_sp:
            match_sp = prefix_sp.group()
            a, scorea = searchmapping("status_perkawinan", match_sp)
            if scorea >= THRESHOLD_RATIO:
                return a
        return ""
    except Exception as e:
        print(f"Gagal ekstrak status perkawinan di matching_status: {e}")
        return ""

def data_verification(data, valuetype):
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