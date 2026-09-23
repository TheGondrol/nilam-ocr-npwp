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
| `pipeline_outbox` | **repo ini** | ketiga tahap (dalam transaksi job), relay | relay tiap service, `GET /v1/<tahap>/outbox` | callback dan handoff yang belum terkirim (`PIPELINE_OUTBOX`). Baris dihapus setelah terkirim; yang gagal permanen (4xx, atau 5xx lebih lama dari `PIPELINE_OUTBOX_MAX_AGE_SECONDS`) tetap ada sebagai dead letter dengan `failed_at` + `last_error`, tidak pernah diambil lagi oleh relay, dan dilepas manual dengan `failed_at = NULL, next_attempt_at = now()`. `ds` dipakai untuk membersihkan dead letter lama |
| `ocr_npwp_alembic_version` | **repo ini** | Alembic | Alembic | versi migrasi repo ini; namanya sengaja tidak `alembic_version` supaya tidak bentrok dengan migrasi tim lain |
| `ocr.orchestration_api_events` | **orkestrasi** | orkestrasi; ketiga tahap menambah baris keadaan akhir kalau `ORCHESTRATION_API_EVENTS_TABLE` diisi | orkestrasi | log API orkestrasi, append-only. Lihat bagian di bawah tabel ini |
| `orchestration_*` lainnya, `auth_*`, `datahub_lookup_log` (schema `ocr`) | **orkestrasi** | orkestrasi | orkestrasi | di luar repo ini. Migrasi di sini tidak pernah membuat atau mengubahnya |
| `ocr.*`, `structuring.*`, `scoring.*` (schema terpisah) | — | tidak ada | tidak ada | sisa desain lama sebelum tabel pindah ke schema `public`; **dihapus oleh migrasi `0005_drop_legacy_schemas`**. Migrasi itu hanya membuang schema yang isinya persis `jobs` + `results`; kalau ada tabel atau view lain di dalamnya, migrasi berhenti dengan pesan supaya diperiksa dulu. Jumlah baris yang dibuang dicatat di log Alembic |

Guardrails tidak punya tabel: ia membaca status tahap lewat API, bukan lewat database.

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

`result_data` mengikuti bentuk baris polling orkestrasi sendiri: `{result, status, document_type,
error_code, error_message, created_at, updated_at}`. `result` berisi data kontrak `extract-ocr`
(`nomor_npwp`, `nama`, `flag`, `flag_reason`) plus `document_type` dan `guardrails`, dan kosong
kalau gagal. Tabel ini tidak punya kunci unik per `request_id`, jadi request yang dijalankan ulang
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

Untuk PostgreSQL lokal, `make up-db` menjalankan migrasi lebih dulu lewat service
`migrate` di `docker-compose.db.yml`; service lain baru start setelah migrasi selesai.
