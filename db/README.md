# Database

Satu database PostgreSQL dipakai bersama oleh repo ini **dan** oleh service orkestrasi
(`bribrain_ocr_nilam` di Cloud SQL). Karena itu penting jelas: tabel mana milik siapa.

**Semua tabel repo ini ada di schema `nilam_ocr_npwp` dan namanya berawalan `nilam_`** (konvensi nama dari
klien), termasuk tabel versi Alembic (`nilam_ocr_npwp_alembic_version`). Migrasi `0010_ocr_pipeline_schema`
memindahkannya dari `public` ke `ocr_pipeline`, `0011_ocr_pipeline_npwp_schema` ke `ocr_pipeline_npwp`, lalu
`0013_nilam_naming` ke `nilam_ocr_npwp` sambil memberi awalan `nilam_` (`ocr_jobs` → `nilam_ocr_jobs`,
`testing_ocr_jobs` → `nilam_testing_ocr_jobs`); `0011` dan `0013` membuang schema lama yang sudah kosong.
`0014_ocr_extraction_tables` mengganti nama tabel tahap extraction atas permintaan klien
(`nilam_ocr_jobs` → `nilam_ocr_extraction_jobs`, `nilam_ocr_results` → `nilam_ocr_extraction_result`,
kembarannya `nilam_testing_` juga) beserta index/PK/FK-nya, lalu memakai nama `nilam_ocr_results` untuk
tabel baru: jawaban akhir request.
Semuanya memakai `ALTER TABLE ... SET SCHEMA` (dan `RENAME` di `0013`): tabelnya sendiri yang pindah (baris,
index, constraint, sequence `id`), tanpa salin data, dalam satu transaksi; jumlah baris tiap tabel dicatat di
log job migrasi. `0013` juga mengganti nama index, primary key, foreign key, dan sequence `id` mengikuti nama
tabelnya (`idx_nilam_ocr_jobs_status`, `nilam_pipeline_outbox_id_seq`). Tabel versi dipindah dan diganti
namanya lebih dulu oleh [`env.py`](migrations/env.py) (dari `ocr_pipeline_npwp`, `ocr_pipeline`, atau `public`,
dengan nama lamanya `ocr_npwp_alembic_version`), karena Alembic membacanya sebelum revisi mana pun jalan.
Migrasi `0001`–`0009` tetap membuat tabelnya di `public` (search path bawaan) dan `0010`/`0011`/`0013` yang
memindahkan, jadi database kosong dan database lama berakhir sama. Query manual:
`nilam_ocr_npwp.nilam_ocr_extraction_jobs`, atau `SET search_path = nilam_ocr_npwp, public`. Di kode, nama schema ada di
`PIPELINE_SCHEMA` dan awalannya di `TABLE_PREFIX`
([`ocr_common/pipeline/database.py`](../libs/ocr_common/ocr_common/pipeline/database.py)); test SQLite
memakai tabel yang sama tanpa schema.

## Peta tabel

| Tabel | Pemilik schema | Ditulis | Dibaca | Isi |
|---|---|---|---|---|
| `nilam_ocr_extraction_jobs`, `nilam_ocr_extraction_result` | **repo ini** | extraction | extraction (`GET /v1/extraction/jobs/{request_id}`), orchestrator lewat API itu | status dan hasil tahap OCR |
| `nilam_structuring_jobs`, `nilam_structuring_results` | **repo ini** | structuring | structuring lewat API-nya (orchestrator) | status dan field hasil structuring |
| `nilam_scoring_jobs`, `nilam_scoring_results` | **repo ini** | scoring | scoring lewat API-nya (orchestrator) | status dan skor trust model |
| `nilam_ocr_results` | **repo ini** | tahap terakhir request (extraction / structuring / scoring), atau orchestrator kalau guardrails yang mengakhiri | tidak ada service; untuk Orkestrasi pusat, audit dan analisis | jawaban akhir request, **satu baris per `request_id`** dalam bentuk jawaban extract-ocr: `status_code`, `status_desc`, `message`, `data`, `errors`, `request_id`, `guardrails`, `created_at`, `update_at`. Ditulis oleh service yang mengakhiri request, dalam transaksi yang sama dengan hasilnya sendiri: selesai (200, `data` = hasil service terakhir apa adanya: laporan guardrails, hasil OCR, hasil structuring, atau field kontrak setelah scoring), ditolak aturan structuring atau model guardrails (400 `DOWNSTREAM_VALIDATION_ERROR`, `guardrails: 1`), atau gagal (422 `<TAHAP>_FAILED`). `guardrails` 0 lolos, 1 ditolak, **null kalau `pipeline_name_sequence` tidak memuat guardrails**. Status 202 tidak ditulis; `request_id` yang dijalankan ulang menimpa barisnya (`update_at`). Penolakan sebelum pipeline jalan (file, sequence, threshold) dan 5xx saat memanggil guardrails/extraction tidak dicatat (migrasi `0014`) |
| `nilam_guardrails_results` | **repo ini** | orchestrator | tidak ada service; untuk audit dan analisis | setiap putusan guardrails, termasuk dokumen yang ditolak, dengan threshold yang memutuskan dan asalnya (`threshold_source`: `request` dari Orkestrasi pusat, `service` milik guardrails), dan laporan lengkapnya. Append-only (migrasi `0008`; kolom `threshold_target` dibuang di `0012`, ambang guardrails kini satu sisi) |
| `ocr_npwp_requests` | — | tidak ada | tidak ada | tabel kontrak lama sinkron (`generate-request-id` → `extract-ocr` → `get-ocr-result`) yang sudah dihapus dari extraction; **dihapus oleh migrasi `0007_drop_ocr_npwp_requests`**. Jumlah barisnya dicatat di log job migrasi sebelum di-drop; `downgrade` membuat ulang tabel kosong, isinya tidak kembali |
| `nilam_pipeline_outbox` | **repo ini** | ketiga tahap (dalam transaksi job), relay | relay tiap service, `GET /v1/<tahap>/outbox` | callback dan handoff yang belum terkirim (`PIPELINE_OUTBOX`). Baris dihapus setelah terkirim; yang gagal permanen (4xx, atau 5xx lebih lama dari `PIPELINE_OUTBOX_MAX_AGE_SECONDS`) tetap ada sebagai dead letter dengan `failed_at` + `last_error`, tidak pernah diambil lagi oleh relay, dan dilepas manual dengan `failed_at = NULL, next_attempt_at = now()`. `ds` dipakai untuk membersihkan dead letter lama |
| `nilam_testing_ocr_extraction_jobs`/`_result`, `nilam_testing_ocr_results`, `nilam_testing_structuring_jobs`/`_results`, `nilam_testing_scoring_jobs`/`_results`, `nilam_testing_pipeline_outbox`, `nilam_testing_guardrails_results` | **repo ini** | ketiga tahap (dan orchestrator untuk putusan guardrails) lewat endpoint `-test` (`TESTING_ENDPOINTS`) | tahap itu sendiri, orchestrator lewat `GET /v1/<tahap>/jobs-test/{request_id}` | salinan persis tabel tahap dan outbox untuk load test tim ML (migrasi `0006`). Tidak pernah dibaca Orkestrasi; boleh di-`TRUNCATE` kapan saja setelah tes. Lihat README, "Endpoint Testing" |
| `nilam_ocr_npwp_alembic_version` | **repo ini** | Alembic | Alembic | versi migrasi repo ini (di `nilam_ocr_npwp`; sebelum `0013` bernama `ocr_npwp_alembic_version`: sampai `0009` di `public`, di `0010` di `ocr_pipeline`, sampai `0012` di `ocr_pipeline_npwp`); namanya sengaja tidak `alembic_version` supaya tidak bentrok dengan migrasi tim lain (`public.alembic_version`) |
| `ocr.orchestration_api_events` | **orkestrasi** | orkestrasi; ketiga tahap menambah baris keadaan akhir kalau `ORCHESTRATION_API_EVENTS_TABLE` diisi | orkestrasi | log API orkestrasi, append-only. Lihat bagian di bawah tabel ini |
| `orchestration_*` lainnya, `auth_*`, `datahub_lookup_log` (schema `ocr`) | **orkestrasi** | orkestrasi | orkestrasi | di luar repo ini. Migrasi di sini tidak pernah membuat atau mengubahnya |
| `ocr.*`, `structuring.*`, `scoring.*` (schema terpisah) | — | tidak ada | tidak ada | sisa desain lama sebelum tabel pindah ke schema `public` (dan sejak `0013` ke `nilam_ocr_npwp`); **dihapus oleh migrasi `0005_drop_legacy_schemas`**. Migrasi itu hanya membuang schema yang isinya persis `jobs` + `results`; kalau ada tabel atau view lain di dalamnya, migrasi berhenti dengan pesan supaya diperiksa dulu. Jumlah baris yang dibuang dicatat di log Alembic |

Orchestrator membaca status tahap lewat API, bukan lewat database; tabel yang ditulisnya hanya `nilam_guardrails_results` dan, kalau guardrails mengakhiri request, `nilam_ocr_results` (satu transaksi, best-effort). Guardrails tidak punya tabel.

Ketiga tahap juga bisa menulis status request ke `orchestration_extract_ocr` di transaksi
yang sama dengan penyimpanan hasilnya, kalau `ORCHESTRATION_OUTCOME_TABLE` diisi (default
mati). Yang ditulis: `processing` + tahapnya saat job diklaim, `completed` + `result_data`
oleh scoring, dan `failed` + `error_code` saat gagal. **Kontraknya adalah kolom `downstream_status`**:
Orkestrasi hanya membaca kolom itu (`processing` | `completed` | `failed`) untuk menjawab polling
client; kolom lain (`downstream_stage`, `status_code`, `error_code`, `error_message`, `result_data`)
adalah data pendamping. Request yang ditolak guardrails (400) tidak pernah menulis kolom ini,
karena tidak ada tahap yang berjalan; Orkestrasi menjawab client dari respons sinkron itu. Tabel itu **milik orkestrasi**, jadi
kolomnya mereka yang menambahkan; DDL yang dibutuhkan (termasuk `request_id` unik) ada di
[external/orchestration_extract_ocr.sql](external/orchestration_extract_ocr.sql) dan bisa
dipasang ke PostgreSQL lokal dengan `make db-external`.

Orkestrasi di dev membaca hasil dari log API-nya, `ocr.orchestration_api_events`. Kalau
`ORCHESTRATION_API_EVENTS_TABLE=ocr.orchestration_api_events` diisi, tiap tahap **menambah** satu
baris di transaksi yang sama dengan tabel job-nya sendiri (double write), hanya untuk keadaan akhir:

| Keadaan | `endpoint` | `status_code` | `downstream_status` | `downstream_stage` | `error_code` |
|---|---|---|---|---|---|
| selesai (scoring) | `GET_OCR_RESULT` | 200 | `COMPLETED` | `SCORING` | kosong |
| gagal di satu tahap | `GET_OCR_RESULT` | 422 | `FAILED` | `EXTRACTION`, `STRUCTURING`, atau `SCORING` | `OCR_FAILED`, `STRUCTURING_FAILED`, `SCORING_FAILED` |
| ditolak aturan structuring | `GET_OCR_RESULT` | 400 | `FAILED` | `STRUCTURING` | `DOWNSTREAM_VALIDATION_ERROR` |

`result_data` mengikuti bentuk baris polling orkestrasi sendiri: `{result, status, document_type,
error_code, error_message, created_at, updated_at}`. `result` berisi data kontrak `extract-ocr`
(`nomor_npwp`, `nama`) plus `document_type` dan `guardrails`, dan kosong kalau gagal atau ditolak.
Untuk penolakan, `error_message` adalah alasan dari aturan ML (bahasa Indonesia). Tabel ini tidak punya kunci unik per `request_id`, jadi request yang dijalankan ulang
mendapat baris baru; **baris terbaru per `request_id` adalah keadaannya**. Kalau tabel ini gagal
ditulis, penulisan job ikut dibatalkan. DDL tiruannya ada di
[external/orchestration_api_events.sql](external/orchestration_api_events.sql).

## Sumber kebenaran

Kolom didefinisikan **sekali** di [`libs/ocr_common/ocr_common/pipeline/tables.py`](../libs/ocr_common/ocr_common/pipeline/tables.py).
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

**`0010`, `0011`, dan `0013` (pindah schema, ganti nama tabel) tidak kompatibel ke belakang:** pod dengan image
lama mencari tabel di schema dan dengan nama lamanya dan gagal sejak migrasi itu jalan sampai pod-nya diganti. Jalankan `migrate-db.sh` lalu langsung
`deploy.sh all`, sebaiknya saat sepi. Job yang tertinggal `PROCESSING` diambil lagi oleh pengambil job basi
di pod baru; pesan outbox tetap di tabelnya dan dikirim relay pod baru. Query, dashboard, atau script lain
yang menyebut tabel ini dengan nama lama juga harus diubah ke `nilam_ocr_npwp.nilam_<tabel>`, dan hak akses
(GRANT) yang pernah diberikan pada schema lama tidak ikut pindah ke schema baru.

Database yang tidak terjangkau dari laptop (Cloud SQL, private IP) dimigrasi lewat Job di cluster:
`deploy/helm/db-job.sh alembic upgrade head`.

## Pindah database (Cloud SQL)

[`copy_database.py`](copy_database.py) menyalin ke-16 tabel `nilam_ocr_npwp` dari `SOURCE_DATABASE_URL` ke
`TARGET_DATABASE_URL` tanpa mengubah sumbernya (satu snapshot read-only): tabel tujuan dibuat dari
`tables.py`, di-stamp di revisi terakhir, dicek dengan `alembic check`, dan jumlah baris tiap tabel
dibandingkan. `--check` hanya mengecek, `--replace` membuat salinan persis. Di cluster lewat
`deploy/helm/db-job.sh copy`; langkah lengkap di [deploy/helm/README.md](../deploy/helm/README.md#pindah-ke-cloud-sql).
Satu beda yang disengaja: urutan fisik kolom di tujuan mengikuti `tables.py`, sedangkan di database lama
kolom yang ditambah migrasi belakangan (`ds`, `input`, ...) ada di akhir. Kode selalu memakai nama kolom,
jadi ini hanya terlihat di `SELECT *`.

Untuk PostgreSQL lokal, `make up-db` menjalankan migrasi lebih dulu lewat service
`migrate` di `docker-compose.db.yml`; service lain baru start setelah migrasi selesai.
