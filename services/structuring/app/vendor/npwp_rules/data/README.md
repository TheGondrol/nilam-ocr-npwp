# Data rujukan aturan NPWP

Taruh di sini (atau tunjuk lewat env var, lihat `../README.md`):

- `kode_wilayah.json` — tabel kecamatan Depdagri (`{"kecamatan": {"320101": ...}}`), dari `build_wilayah_codes.py` ML engineer. Di-commit.
- `kpp_codes.json` — tabel kode KPP DJP (`{"012": ...}`), dari `build_kpp_codes.py` ML engineer. Di-commit.
- `name_lnmast.xlsx` — master nama (data internal). **Tidak di-commit**; pasang lewat volume + `NAME_MASTER_PATH`.

Per 23 September 2026 kedua JSON sudah ada di sini (dari kiriman ML engineer); `name_lnmast.xlsx` sengaja diabaikan dulu. Tanpa sebuah file, pemeriksaan yang bersangkutan tidak memberi sinyal (tidak pernah flag), service tetap berjalan.
