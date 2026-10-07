# Data rujukan aturan NPWP

Taruh di sini (atau tunjuk lewat env var, lihat `../README.md`):

- `kode_wilayah_v2.json` — tabel wilayah Kemendagri (`{"_meta", "provinsi", "kabkota", "kecamatan": {"110101": {"name", "status", "replaced_by", ...}}}`), 8.063 kecamatan (7.285 aktif + 778 tidak aktif; dasar Kepmendagri 300.2.2-2138/2025). Hanya kunci `kecamatan` yang dibaca. Menggantikan `kode_wilayah.json` (versi lama, tabel 7.230 kecamatan, boleh dihapus). Di-commit.
- `kpp_codes_v2.json` — tabel kode KPP DJP (`{"kpp": [{"kode": "012", ...}], "kode_historis": [...]}`), dari ML engineer 30 September 2026; menggantikan `kpp_codes.json`. Di-commit.
- `name_lnmast.xlsx` — master nama (data internal). **Tidak di-commit**; pasang lewat volume + `NAME_MASTER_PATH`.

Per 23 September 2026 kedua JSON sudah ada di sini (dari kiriman ML engineer); `name_lnmast.xlsx` sengaja diabaikan dulu. Tanpa sebuah file, pemeriksaan yang bersangkutan tidak memberi sinyal (tidak pernah flag), service tetap berjalan.
