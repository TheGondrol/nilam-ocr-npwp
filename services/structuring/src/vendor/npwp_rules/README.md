# vendor/npwp_rules

Aturan ekstraksi NPWP (regex + posisi) dari ML engineer, diterima 20 September 2026 sebagai dua
file lepas tanpa konteks. Dipakai oleh backend structuring `npwp_rules`
(`src/models/structuring.py`).

| File | Asal | Perubahan dari kiriman asli |
|---|---|---|
| `npwp.py` | ML engineer | tidak ada |
| `name_extraction.py` | ML engineer | dua baris import dibuat relatif (`from .npwp import ...`, `from .name_master import ...`) |
| `name_master.py` | **pengganti sementara, bukan dari ML engineer** | lihat docstring-nya |

Folder ini dikecualikan dari ruff dan ty supaya isinya tetap sama dengan kiriman dan mudah
dibandingkan saat ada versi baru. Butuh `numpy` (`poly_center`).

Yang belum dikirim dan perlu diminta:
1. `name_master.py` asli + daftar nama rujukannya.
2. Modul pemanggilnya (komentar menyebut `GEN/ocr_npwp/app/`): cara memilih nomor saat ada
   beberapa kandidat, kapan guardrail halaman (`find_other_document_keyword`, `contains_captcha`,
   `contains_web_lookup_screenshot`, `MAX_EXPECTED_PAGES`) menolak dokumen, dan bentuk output-nya.
   Tanpa itu, perangkaian di `NpwpRulesStructurer` adalah tafsir kami atas docstring kedua file.
