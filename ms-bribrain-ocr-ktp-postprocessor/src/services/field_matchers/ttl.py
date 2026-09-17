import logging
import re

from src.services.text_helpers import correct_digits_to_alphabets, correct_alphabets_to_digits, clean_colon
from src.services.constants import bulan_dict, THRESHOLD_CONFIDENCE, REGEX

logger = logging.getLogger(__name__)



def normalisasi_meet_tgllength(parts: list):
    tempat: str | None
    day: str | None
    month_raw: str | None
    year: str | None
    
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
    month = bulan_dict.get(month_raw or "", month_raw.zfill(2) if month_raw and month_raw.isdigit() else None)
    return tempat, day, month, year

def cleaning_tempat_meet_tgllength(tempat:str):
    tempat_clear = False  # BUG-12: bind before the conditional so the no-match path can't UnboundLocalError
    tempat = re.sub(REGEX['AMBIL_NON_HURUF_AWAL'], "", tempat)  # buang karakter non-huruf dari awal
    colon_match = re.search(REGEX['AMBIL_KONTEN_SETELAH_COLON'], tempat)
    if colon_match:
        tempat = colon_match.group(1)
    tempat = correct_digits_to_alphabets(tempat)
    match = re.match(REGEX['AMBIL_HURUF_SPASI'], tempat)
    if match:
        tempat = match.group(1)
        tempat_clear = True
    return tempat_clear, tempat

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
        tempat, day, month, year = normalisasi_meet_tgllength(parts=parts)

        # Jika tempat mengandung digit, pisahkan
        tempat_clear = False
        if tempat:
            tempat_clear, tempat = cleaning_tempat_meet_tgllength(tempat=tempat)
        
        if day and len(day) > 2:
            match = re.match(REGEX['AMBIL_HURUF_LALU_DIGIT'], day)
            if match:
                if tempat_clear:
                    # Jika tempat sudah dipisahkan, ambil day dan month dari match
                    tempat = f"{tempat} {match.group(1)}"
                else:
                    tempat = match.group(1)
                day = match.group(2)

        # Validasi hasil
        if all([tempat, day, month, year]) and day.isdigit() and year.isdigit():
            return tempat, day, month, year
        else:
            return None, None, None, None
    except Exception as e:
        logger.error(f"Gagal ekstrak tgl lahir di meet_tgllength: {e}")
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
        # Initialize all variables at function start
        tempat: str | None = None
        day: str | None = None
        month_raw: str | None = None
        year: str | None = None
        
        colon_match = re.search(REGEX['AMBIL_KONTEN_SETELAH_COLON'], data)
        if colon_match:
            data = colon_match.group(1)
        match = re.match(REGEX['AMBIL_HURUF_LALU_TANGGAL'], data)
        if match:
            tempat = match.group(1)
            angka: list[str] = re.findall(REGEX["AMBIL_DIGIT"], match.group(2))
            if len(angka) == 3:
                day = angka[0]
                month_raw = angka[1]
                year = angka[2]
            elif len(angka) == 1 and len(angka[0]) == 8:
                # Format tanpa pemisah, misal 12051990
                day = angka[0][:2]
                month_raw = angka[0][2:4]
                year = angka[0][4:]
            return tempat, day, month_raw, year
        else:
            # Fallback: ekstrak semua huruf dan angka
            data = correct_alphabets_to_digits(data)
            tempat = ''.join(re.findall(REGEX["AMBIL_HURUF"], data))
            angka_list: list[str] = re.findall(REGEX["AMBIL_DIGIT"], data)
            angka_str = ''. join(angka_list)
            if len(angka_str) == 8:
                day = angka_str[:2]
                month_raw = angka_str[2:4]
                year = angka_str[4:]
            return tempat, day, month_raw, year
    except Exception as e:
        logger.error(f"Gagal ekstrak data ttl jadi satu: {e}")
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
        match = re.match(REGEX["AMBIL_FORMAT_TANGGAL1"], data)
        if not match:
            match = re.match(REGEX['AMBIL_FORMAT_TANGGAL2'], data)
        if match:
            tempat = match.group(1)
            day = match.group(2)
            month_raw = match.group(3)
            year = match.group(4)
        else:
            # Fallback: ekstrak semua huruf dan angka
            data = correct_digits_to_alphabets(data)
            tempat = ''.join(re.findall(REGEX["AMBIL_HURUF"], data))
            angka = ''.join(re.findall(REGEX["AMBIL_DIGIT"], data))
            if len(angka) == 8:
                day = angka[:2]
                month_raw = angka[2:4]
                year = angka[4:]
        return tempat, day, month_raw, year
    except Exception as e:
        logger.error(f"Gagal ekstrak data ttl terpisah: {e}")
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
        logger.error(f"Gagal ekstrak tgl lahir di notmeet_tgllength: {e}")
        return None, None, None, None
    
def format_ttl_new_sesuai(data):
    # try:
    digit_data = len(re.findall(REGEX["AMBIL_SATU_DIGIT"], str(data[1])))
    if len(data[0]) >= 3 and digit_data >= 5:
        number = re.sub(REGEX['AMBIL_NON_ALFANUMERIK'], '', data[1])
        tempat = data[0]
    else:
        # ekstrak tempat — operate on a joined string; ``data`` is a list of two
        # strings and re.findall requires str/bytes, not a list (BUG-13).
        joined = ' '.join(data) if isinstance(data, (list, tuple)) else str(data)
        caps = re.findall(REGEX['AMBIL_KAPITAL_HURUF_TITIK'], joined)
        tempat = str(max(caps, key=len)) if caps else ""
        tempat = tempat.replace(".", "").replace("-", " ").strip()

        # ekstrak angka
        numbers = re.findall(REGEX['AMBIL_DUA_DIGIT_ALFANUMERIK_DASH'], joined)
        number = "".join(numbers)
    return tempat, number
    # except Exception as e:
    #     print(f"Gagal format data ttl new sesuai : {e}")
    #     return "", ""

def format_ttl_new_tidak_sesuai(data):
    try:
        caps = re.findall(REGEX['AMBIL_KAPITAL_QUOTES'], data)
        tempat = max(caps, key=len) if caps else ""
        # tempat = tempat.replace(".", "").replace("-", " ").strip()

        # ekstrak angka
        numbers = re.findall(REGEX['AMBIL_DUA_DIGIT_ALFANUMERIK'], data)
        number = "".join(numbers)
        return tempat, number
    except Exception as e:
        logger.error(f"Gagal format data ttl new tidak sesuai: {e}")
        return "", ""


def matching_tempatlahir_new(data: list, next_data: list) -> tuple:
    dat = data[0] if data and len(data) > 0 else ""
    next_dat = next_data[0] if next_data and len(next_data) > 0 else ""

    try:
        digit_data = len(re.findall(REGEX["AMBIL_SATU_DIGIT"], str(dat)))
        digit_nextdata = len(re.findall(REGEX["AMBIL_SATU_DIGIT"], str(next_dat)))

        # pilih kandidat mana yg lebih cocok
        proses = dat if digit_data >= 6 and digit_nextdata < 6 else next_dat
        if len(proses) <= 12:  # kalau terlalu pendek, gabungkan
            proses = f"{dat} {next_dat}"

        # cari teks kapital
        match = re.search(REGEX["AMBIL_KATA_KAPITAL"], proses)
        if not match:
            return "", ""

        content_awal = match.group(1)

        content_split: list[str] = re.split(REGEX["AMBIL_KOMA_TITIK"], content_awal)
        content: str | list[str]
        if not all(len(c.strip()) > 3 for c in content_split):
            content = content_awal
        else:
            content = content_split
        if len(content) == 2:
            tempat, number = format_ttl_new_sesuai(content)
        else:
            tempat, number = format_ttl_new_tidak_sesuai(proses)

        if number and not number.isdigit():
            number = correct_alphabets_to_digits(number)

        day = month_raw = year = None
        if len(number) == 8:  # format DDMMYYYY
            day = number[:2]
            month_raw = number[2:4]
            year = number[4:]

        if all([day, month_raw, year, tempat]):
            month = bulan_dict.get(str(month_raw), str(month_raw))
            tgl = f"{day} {month} {year}"
            return tempat, tgl

        return "", ""
    except Exception as e:
        logger.error(f"Gagal ekstrak tgl lahir di matching_tempatlahir_new: {e}")
        return "", ""
        
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
        text = re.sub(REGEX['AMBIL_NON_ALFANUMERIK_SPASI'], "", text)
        pattern = REGEX['AMBIL_KATA_KAPITAL_BATAS']
        match = re.search(pattern, text)
        text = match.group(1) if match else text
        tgl_split = [part for part in re.split(REGEX['AMBIL_PEMISAH'], text) if part]
        return tgl_split
    except Exception as e:
        logger.error(f"Gagal split tgl lahir: {e}")
        return []
    
def validate_ttl_data(data, next_data):
    """Validate and parse TTL (tempat/tanggal lahir) data."""
    parts = []  # Initialize to prevent UnboundLocalError
    dat = data[0] if data and len(data) > 0 else ""
    next_dat = next_data[0] if next_data and len(next_data) > 0 else ""

    digit_data = len(re.findall(REGEX["AMBIL_SATU_DIGIT"], dat))
    digit_nextdata = len(re.findall(REGEX["AMBIL_SATU_DIGIT"], next_dat))

    # Cek jika next_data valid dan mengandung kata "jenis"
    if next_data and len(next_data) > 1 and digit_data >= 6 and digit_nextdata < 6:
        if data and len(data) > 1 and data[1] > THRESHOLD_CONFIDENCE:
            logger.debug(f"Processing next_data: {next_data}")
            match = re.search(REGEX['AMBIL_COLON_APAPUN'], dat)
            if match:
                dat = match.group(1)
            parts = split_tgllahir(dat)
    elif digit_nextdata >= 6:
        if next_data and len(next_data) > 1 and next_data[1] > THRESHOLD_CONFIDENCE:
            clean = clean_colon(next_dat)
            parts = split_tgllahir(clean)
    return parts
    
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
    parts = validate_ttl_data(data, next_data)

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
        tgl = re.sub(REGEX['AMBIL_TANDA_BACA'], " ", tgl)
        tgl = f"{tgl[:2]} {tgl[2:4]} {tgl[4:]}" if len(tgl) == 8 else tgl
        if len(tempat) > 9:
            match = re.findall(REGEX['AMBIL_MINIMAL_DUA_KAPITAL'], tempat)
            tempat = " ".join(match)
        return tempat, tgl

    return "", ""