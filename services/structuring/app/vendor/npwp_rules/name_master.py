"""
PENGGANTI SEMENTARA. name_extraction.py meng-import modul ini, tetapi ML engineer
belum mengirimkannya (beserta daftar nama rujukannya). Kedua fungsi di bawah
adalah bentuk "tanpa sinyal" yang memang sudah ditangani pemanggilnya:

- is_recognized_name -> False: extract_name hanya mempersempit kandidat kalau
  ADA yang dikenali ("none recognized carries no signal"), jadi jarak ke nomor
  NPWP tetap menjadi penentu, persis perilaku tanpa master.
- correct_name_spacing -> nama apa adanya: tidak ada koreksi spasi.

Ganti file ini dengan name_master.py yang asli begitu tersedia; tidak ada kode
lain yang perlu diubah.
"""


def is_recognized_name(name: str) -> bool:
    return False


def correct_name_spacing(name: str) -> str:
    return name
