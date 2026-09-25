# tools/tracker — pelacak pipeline dan pola outbox (uji coba lokal, extra)

UI React + backend kecil yang memerankan Orkestrasi pusat: upload foto -> orchestrator
`/v1/extract-ocr` (menunggu maks. `PIPELINE_WAIT_SECONDS`) -> tiap tahap, tiap baris
`pipeline_outbox`, dan tiap callback tampil live di browser, dengan timeline `t+…`.
Event lewat Redis Streams (`ocr:events:<request_id>`), UI menerimanya lewat SSE.
Ikut repo sebagai alat uji coba, bukan bagian dari yang di-deploy; `.env`-nya dibuat dari `.env.example`.

## Apa yang diperlihatkan

Untuk satu request, dari kiri ke kanan di layar:

1. **Jawaban orchestrator**: HTTP 200 `completed` + data (pipeline selesai di dalam batas tunggu),
   202 `processing` (batas tunggu habis, hasil menyusul lewat callback), 422 (tahap gagal),
   200 `guardrails: 1` (ditolak guardrails atau aturan structuring), plus lamanya dibanding batas tunggu.
2. **Kartu tiap tahap** dengan dua fakta terpisah: *hasil di DB* (baris `<tahap>_jobs` DONE,
   dibaca langsung dari PostgreSQL) dan *orkestrasi tahu* (callback diterima tracker, attempt ke-n).
   Selisih keduanya adalah tempat outbox bekerja.
3. **Tabel `pipeline_outbox` request itu**: tiap baris (handoff / callback) dengan status
   QUEUED -> CLAIMED -> DELIVERED, atau RETRY (5xx) dan DEAD (4xx / lewat umur), attempt,
   `last_error`, dan riwayat waktunya. Tombol *Lepaskan dead letter* menjalankan
   `UPDATE pipeline_outbox SET failed_at = NULL, next_attempt_at = now()`.
4. **Timeline** semua event dengan `t+ms` sejak upload: commit transaksi, klaim relay,
   handoff, callback, retry, dead letter, END.

Di sidebar ada **backlog outbox tiap service** (`GET /v1/<tahap>/outbox`, di-refresh 2 detik)
dan dua simulasi:

| Simulasi | Cara kerja | Yang terlihat |
|---|---|---|
| **Pipeline lambat** (OCR ditunda N detik) | nama file diberi awalan `delay<N>s-`; ekstraksi menghormatinya hanya dengan `ENVIRONMENT=local` (`ocr_common/simulation.py`) | N > `PIPELINE_WAIT_SECONDS` (15): orchestrator menjawab **202**, tahap-tahap tetap selesai, hasil datang lewat callback SCORING. N kecil atau tanpa simulasi: **200** dengan data |
| **Callback orkestrasi mati (503)** | tracker menjawab 503 untuk setiap callback yang datang | baris callback jadi RETRY dengan backoff (0,5 dtk ×2 … maks 5 menit), attempt bertambah, **handoff tetap terkirim dan tahap berikutnya tetap jalan**; orchestrator tetap 200 kalau pipeline cepat. Kembalikan ke *normal*: retry berikutnya 200, baris dihapus |
| **Callback orkestrasi menolak (422)** | tracker menjawab 422 | baris jadi DEAD setelah satu attempt, tetap ada di tabel, backlog service menunjukkan `dead_letters`; tombol *Lepaskan* mengirimnya lagi |

Urutan yang enak untuk presentasi: (1) kirim tanpa simulasi -> 200 dan semua baris outbox
DELIVERED dalam ~1 detik; (2) lambat 20 dtk -> 202, lalu tunggu callback SCORING; (3) callback
mati + kirim dokumen -> tahap 2-4 selesai walau orkestrasi "mati", callback menumpuk sebagai
RETRY; nyalakan lagi -> semuanya terkirim; (4) callback menolak -> dead letter -> lepaskan.

Prasyarat pola outbox terlihat: service dijalankan dengan `DATABASE_URL` (stack
`docker-compose.db.yml`, semua migrasi dengan `make db-upgrade`) dan `PIPELINE_OUTBOX=true` (sudah di
`services/*/.env`). Tanpa itu service memakai mode langsung dan panel outbox tetap kosong.

## Memilih target: GKE atau lokal

Saklarnya ada di `tools/tracker/.env`. Blok GKE aktif = menembak cluster;
komentari **seluruh** blok itu = kembali ke stack lokal.

    # ── GKE: gc-ddb-dev-gke-cluster-01 / namespace nilam-ocr-npwp ──
    TRACKER_TARGET=gke
    ORCHESTRATOR_URL=http://127.0.0.1:9034
    GUARDRAILS_URL=http://127.0.0.1:9031
    ...

Yang berubah otomatis mengikuti saklar itu:

| | GKE | lokal |
|---|---|---|
| service | `kubectl port-forward` ke `svc/nilam-ocr-npwp-<service>` (satu per service) di 9030-9034; butuh chart 0.3.0+, karena chart lama belum punya orchestrator | container di 127.0.0.1:8030-8034 |
| `API_KEY` | `changeme` (service memaksa `X-API-Key`) | kosong, service jalan `AUTH_DISABLED=true` |
| hasil tiap tahap | polling `GET /v1/<tahap>/jobs/{id}` | callback ke `/v1/callbacks/stage` + baca DB |
| baris outbox | hanya bila `TRACKER_DATABASE_URL` diisi | otomatis: Postgres compose di `127.0.0.1:${POSTGRES_HOST_PORT:-5433}` |
| simulasi lambat | tidak (pod bukan `ENVIRONMENT=local`) | ya |
| Redis | lokal | lokal |

Kenapa polling di mode GKE: pod ada di dalam cluster dan tidak bisa memanggil
balik laptop, jadi `ORCHESTRATION_URL` pod tetap mengarah ke Orkestrasi di
cluster. Tracker menarik sendiri status tiap tahap dan memancarkan event yang
bentuknya sama persis dengan versi callback, sehingga UI tidak berubah.
Aturnya lewat `TRACKER_POLL_INTERVAL` (default 2 detik) dan
`TRACKER_POLL_TIMEOUT` (default 300 detik).

## Menjalankan

    # prasyarat kedua mode: redis di 127.0.0.1:6379
    # prasyarat GKE: kredensial cluster sudah ada
    #   gcloud container clusters get-credentials gc-ddb-dev-gke-cluster-01 \
    #     --project ddb-kubecluster-dev-01 --location asia-southeast2
    # prasyarat lokal: kelima service jalan; ketiga tahap pipeline (ekstraksi, structuring, scoring) dengan
    #   ORCHESTRATION_URL=http://host.docker.internal:8090 (container) / http://127.0.0.1:8090 (bare)
    #   dan, untuk melihat outbox, DATABASE_URL + PIPELINE_OUTBOX=true

    tools/tracker/run.sh            # backend :8090 + frontend :5173, Ctrl+C mematikan semuanya
    tools/tracker/run.sh --stack    # (lokal saja) nyalakan Redis, kelima container, dan Postgres dulu
    tools/tracker/stop.sh           # matikan yang jalan di latar, termasuk port-forward

Di mode GKE `run.sh` membuka lima port-forward sendiri (satu per Service) dan menutupnya saat
berhenti; log-nya di `tools/tracker/.port-forward.log`. Callback dari pod tetap tidak sampai ke
laptop, jadi status tahap diambil dengan polling dan panel outbox hanya terisi kalau
`TRACKER_DATABASE_URL` menunjuk database dev.

    # manual, kalau perlu:
    pip install -r backend/requirements.txt
    python backend/app.py                 # http://127.0.0.1:8090, ikut baca .env
    cd frontend && npm install && npm run dev   # http://127.0.0.1:5173

`GET /api/health` menyebut target yang sedang dipakai, apakah polling menyala, apakah
pemantau database hidup, simulasi yang aktif, dan backend tiap service.

## Endpoint backend

| | |
|---|---|
| `POST /api/requests` | form `file`, `document_type`, `slow_seconds` (0 = tanpa simulasi) |
| `POST /v1/callbacks/stage` | dipanggil relay tiap service; jawabannya mengikuti simulasi |
| `GET /api/requests/{id}/events` | SSE, setiap event punya `type`: client, http, stage, outbox, callback, pipeline |
| `POST /api/requests/{id}/outbox/release` | lepaskan dead letter request itu |
| `GET` / `PUT /api/simulation` | `{"callback": "ok" \| "down" \| "reject"}` |
| `GET /api/outbox` | backlog tiap service dari `GET /v1/<tahap>/outbox` |
| `GET /api/loadtest/config` | file uji yang tersedia (contoh `images/` dan unggahan `assets/`), image/network/target k6, batas laju & durasi |
| `POST /api/loadtest/images`, `DELETE /api/loadtest/images/{nama}` | unggah file uji ke `assets/` (di-.gitignore, dihapus otomatis saat run yang memakainya berakhir); hapus satu file |
| `DELETE /api/loadtest/assets` | hapus semua unggahan yang tertinggal di `assets/`; 409 selama ada run berjalan |
| `GET` / `POST /api/loadtest` | daftar run; mulai run `{"rate", "duration_seconds", "mode": "constant" \| "ramp", "images"}` |
| `POST /api/loadtest/{run}/samples` | dipanggil k6 tiap request: status, job_status, elapsed_ms |
| `GET /api/loadtest/{run}` | statistik run: campuran 200/ditolak/202/4xx/5xx/timeout, p50/p95, end-to-end, alasan penolakan, ringkasan k6 |
| `POST /api/loadtest/{run}/stop`, `DELETE /api/loadtest/{run}` | hentikan container k6; hapus run beserta baris `LT_<run>_%` di database |

## Load testing

Menu **Load testing** menjalankan [../load-tester](../load-tester) (k6 di Docker) dan menghitung
jawaban pintu masuk (200 selesai di dalam batas tunggu, 200 ditolak dengan `guardrails: 1`, 202 hasil
menyusul, 4xx, 5xx, timeout) serta waktu end-to-end dari submit sampai callback SCORING DONE. Dokumen yang
ditolak, baik lewat jawaban 200 maupun lewat callback FAILED `DOWNSTREAM_VALIDATION_ERROR` setelah 202,
dihitung terpisah dari tuntas dan gagal, dengan daftar alasannya. Request uji berprefiks `LT_` dan
tidak masuk daftar "Request terakhir". Env: K6_IMAGE (grafana/k6:latest), K6_NETWORK
(ocr_default), K6_TARGET (http://orchestrator:8034), K6_TRACKER (http://host.docker.internal:PORT),
K6_API_KEY, LOAD_TESTER_DIR.

Env backend: TRACKER_TARGET, ORCHESTRATOR_URL, GUARDRAILS_URL, EKSTRAKSI_URL, STRUCTURING_URL,
SCORING_URL (default 127.0.0.1:8030-8034), TRACKER_POLL, REDIS_URL, API_KEY, PORT,
TRACKER_DATABASE_URL, TRACKER_DB_INTERVAL (0.25), TRACKER_WAIT_SECONDS (15, label saja).
