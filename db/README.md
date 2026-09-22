# Database

Satu database PostgreSQL dipakai bersama oleh repo ini **dan** oleh service orkestrasi
(`bribrain_ocr_nilam` di Cloud SQL). Karena itu penting jelas: tabel mana milik siapa.

## Peta tabel

| Tabel | Pemilik schema | Ditulis | Dibaca | Isi |
|---|---|---|---|---|
| `ocr_jobs`, `ocr_results` | **repo ini** | ekstraksi | ekstraksi (`GET /v1/ekstraksi/jobs/{request_id}`), guardrails lewat API itu | status dan hasil tahap OCR |
| `structuring_jobs`, `structuring_results` | **repo ini** | structuring | structuring lewat API-nya | status dan field hasil structuring |
| `scoring_jobs`, `scoring_results` | **repo ini** | scoring | scoring lewat API-nya | status dan skor trust model |
| `ocr_npwp_requests` | **repo ini** | ekstraksi | ekstraksi | khusus kontrak lama sinkron (`generate-request-id` → `extract-ocr` → `get-ocr-result`) |
| `ocr_npwp_alembic_version` | **repo ini** | Alembic | Alembic | versi migrasi repo ini; namanya sengaja tidak `alembic_version` supaya tidak bentrok dengan migrasi tim lain |
| `orchestration_*`, `auth_*`, `datahub_lookup_log` | **orkestrasi** | orkestrasi | orkestrasi | di luar repo ini. Migrasi di sini tidak pernah membuat atau mengubahnya |
| `ocr.*`, `structuring.*`, `scoring.*` (schema terpisah) | — | tidak ada | tidak ada | sisa desain lama sebelum tabel pindah ke schema `public`. Kandidat dihapus lewat migrasi setelah dipastikan tidak dipakai siapa pun |

Guardrails tidak punya tabel: ia membaca status tahap lewat API, bukan lewat database.

Rencana yang belum dikerjakan: ketiga tahap juga menulis hasil akhir ke
`orchestration_extract_ocr` (kolom `downstream_status`, `downstream_stage`) di transaksi
yang sama dengan penyimpanan hasilnya. Tabel itu milik orkestrasi, jadi kolomnya mereka
yang menambahkan, bukan migrasi di sini.

## Sumber kebenaran

Kolom didefinisikan **sekali** di [`libs/ocr_common/ocr_common/tables.py`](../libs/ocr_common/ocr_common/tables.py).
Migrasi Alembic dan `create_all` di test memakai definisi yang sama, dan `make db-check`
gagal kalau keduanya menyimpang.

## Perintah

```bash
export DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/bribrain_ocr_nilam

make db-upgrade                  # jalankan migrasi sampai revisi terakhir
make db-check                    # gagal kalau definisi tabel di kode beda dengan database
make db-revision m="tambah kolom X"   # buat revisi baru dari selisihnya, lalu PERIKSA hasilnya
```

Database yang tabelnya sudah ada (mis. dev yang dulu dipasang manual) cukup dijalankan
`make db-upgrade`: revisi baseline memakai `CREATE TABLE IF NOT EXISTS`, jadi tabel dan
datanya dibiarkan, dan database itu tercatat berada di revisi baseline.

## Deploy

```bash
DB_HOST=<alamat postgres> deploy/helm/migrate-db.sh          # upgrade head
DB_HOST=<alamat postgres> deploy/helm/migrate-db.sh current  # lihat revisi sekarang
```

Script mengambil `DATABASE_URL` dari Secret release, menggantikan host-nya dengan
`DB_HOST`, lalu menjalankan Alembic di dalam image `db/Dockerfile`. **Jalankan sebelum**
men-deploy image yang membutuhkan perubahan tabelnya.

Untuk PostgreSQL lokal, `make up-db` menjalankan migrasi lebih dulu lewat service
`migrate` di `docker-compose.db.yml`; service lain baru start setelah migrasi selesai.
