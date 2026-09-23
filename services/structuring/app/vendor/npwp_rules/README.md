# vendor/npwp_rules

Aturan ekstraksi NPWP (regex + posisi) dari ML engineer. Versi ini dari repo `nilamnpwp` folder
`regex/` (commit `30c48c8`, 23 September 2026), menggantikan kiriman dua file lepas tanggal 20 September.
Dipakai oleh backend structuring `npwp_rules` (`app/ml/npwp_rules.py`), yang meniru cara `regex/main.py`
mereka merangkai modul-modul ini.

| File | Asal | Perubahan dari kiriman asli |
|---|---|---|
| `npwp.py` | ML engineer | dua baris import dibuat relatif (`from .kpp_codes import ...`, `from .wilayah_codes import ...`) |
| `name_extraction.py` | ML engineer | dua baris import dibuat relatif |
| `name_master.py` | ML engineer (asli, menggantikan pengganti sementara) | path data dibaca saat dipanggil (`default_master_path()`, `default_list_path()`), bukan saat import; default ke `data/` di folder ini |
| `kpp_codes.py` | ML engineer | sama: `default_data_path()`, default `data/kpp_codes.json` |
| `wilayah_codes.py` | ML engineer | sama: `default_data_path()`, default `data/kode_wilayah.json` |

Alasan perubahan path: kiriman asli menghitung path dari `os.environ` saat import dengan fallback ke
layout laptop ML engineer (`BRI/dataset/...`). Di sini nilai berasal dari `Settings` (bisa dari `.env`,
yang tidak diekspor pydantic ke environment), jadi `app/dependencies.py` mengekspor
`WILAYAH_CODES_PATH`, `KPP_CODES_PATH`, `NAME_MASTER_PATH`, `NPWP_NAME_LIST_PATH` sebelum aturan dipakai,
dan modul membacanya saat dipanggil. Nama env var tetap milik ML engineer supaya cara deploy mereka
(volume + env var) tetap berlaku.

Folder ini dikecualikan dari ruff dan ty supaya isinya tetap sama dengan kiriman dan mudah
dibandingkan saat ada versi baru (`diff` terhadap folder `regex/` mereka; selisihnya hanya baris di tabel
di atas). Butuh `numpy` (`poly_center`), `openpyxl` dan `rapidfuzz` (`name_master.py`).

## Data rujukan (`data/`)

| File | Dipakai oleh | Kalau tidak ada |
|---|---|---|
| `kode_wilayah.json` | `has_invalid_kecamatan_prefix` (NPWP 16 digit berbasis NIK) | pemeriksaan kecamatan tanpa sinyal (tidak pernah flag); kode provinsi 2 digit tetap diperiksa dari tabel di `npwp.py` |
| `kpp_codes.json` | `has_invalid_kpp_prefix` (NPWP 15 digit lama) | tanpa sinyal |
| `name_lnmast.xlsx` | `is_recognized_name`: tie-break antar kandidat nama | jarak ke nomor NPWP yang menentukan, seperti sebelumnya |
| `list_name_npwp.xlsx` | `correct_name_with_npwp_list` (koreksi nama per `file_id`) | **tidak dipakai di sini**: name matching fuzzy dilakukan di orkestrator, `extract_name` dipanggil tanpa `file_id` |

**Status 23 September 2026:** `kode_wilayah.json` (7.230 kecamatan) dan `kpp_codes.json` (173 KPP) dari ML
engineer ada di `data/` dan ikut ke image lewat `COPY app`. `name_lnmast.xlsx` (data internal, ~270 ribu nama)
sengaja diabaikan dulu (keputusan 23 September 2026); kalau nanti dipakai, pasang lewat volume + `NAME_MASTER_PATH` (di-gitignore).

## Yang berubah dari kiriman 20 September

- Semua pemeriksaan halaman (dokumen lain, CAPTCHA, screenshot "Cek NPWP", > 2 halaman) kini **flag lunak**
  (`flag`, `flag_reason`), bukan penolakan; batas 2 halaman yang keras dipindah ke guardrails.
- Koreksi homoglyph digit dimatikan (`normalize_npwp_raw`): huruf salah baca dibuang, bukan diganti angka,
  dan `flag_reason` menyebutkannya.
- Pemeriksaan validitas nomor: kode provinsi + kecamatan (Kode Wilayah) dan tanggal lahir untuk format 16 digit,
  kode KPP untuk format 15 digit; `_resolve_province_prefix` kini menangani "T" di posisi 1 maupun 2.
- Nama dilaporkan dua bentuk: `value` (setelah normalisasi spasi `/` dan `PT.`) dan `signals.name_base`
  (bacaan mentah) untuk trust model.
