# nilam-ocr-npwp — OCR NPWP sebagai lima service

Service OCR untuk dokumen NPWP (kartu identitas pajak) Indonesia, dipecah menjadi **lima service yang di-deploy terpisah** (lima image Docker), mengikuti sequence diagram NILAM OCR Orchestration:

| Service | Port | Image | Peran |
|---|---|---|---|
| **orchestrator** | 8034 | `nilam-ocr-orchestrator` | Orkestrasi khusus NPWP, **pintu masuk tunggal**: Orkestrasi pusat hanya memanggil `POST /v1/extract-ocr` (dan `GET /v1/extract-ocr/{request_id}`) di sini. Memeriksa file (tipe, ukuran), minta guardrails menilai dokumen, menyerahkan dokumen yang lolos ke extraction, lalu **menunggu pipeline** sampai `PIPELINE_WAIT_SECONDS`: 200 hasil akhir, 202 masih berjalan, 400 ditolak, 422 satu tahap gagal. Stateless: tanpa database, tanpa callback |
| **guardrails** | 8031 | `nilam-ocr-guardrails` | "ServiceGuardrails", internal: `POST /v1/guardrails/check` dipanggil orchestrator untuk tiap dokumen. Klasifikasi tiap halaman `accepted`/`reject` dengan model EfficientNet-B0 (lokal, CPU) + vonis dokumen; batas halaman (400) dan ukuran (413) dicek sebelum model |
| **extraction** | 8030 | `nilam-ocr-extraction` | "ServiceOCR": tahap pertama pipeline async (`/v1/extraction/jobs` → 202, OCR di background, hasil ke tabel Orkestrasi, handoff ke structuring). Juga OCR mentah sinkron (`/v1/extraction/extract`) |
| **structuring** | 8032 | `nilam-ocr-structuring` | "ServiceStructuring": `/v1/structuring/jobs` → 202, baris teks → `nomor_npwp`, `nama`, `nama_badan` dengan confidence per field, handoff ke scoring |
| **scoring** | 8033 | `nilam-ocr-scoring` | "ServiceScoring", tahap terakhir: `/v1/scoring/jobs` → 202, confidence per field (`npwp_confidence`, `name_confidence`) dari trust model ML engineer; menulis **hasil akhir** ke tabel Orkestrasi (`ORCHESTRATION_OUTCOME_TABLE`). Tanpa skor dokumen / keputusan: ambang milik Orkestrasi |

Orkestrasi pusat (repo `nilam-ocr-orchestration`) memegang `request_id`, status per tahap (`orkestrasi.requests`, `orkestrasi.stage_logs`), dan polling client; repo ini hanya kelima service di atas. Orkestrasi pusat hanya berbicara dengan orchestrator (request masuk); hasil untuk request yang dijawab 202 dikirim oleh tahap pipeline sendiri (callback dan tabel Orkestrasi), dan bisa dibaca kapan saja lewat `GET /v1/extract-ocr/{request_id}` di orchestrator.

Kontrak, envelope, dan auth `X-API-Key` sama dengan service `ocr-*` di `nilam-ocr-orchestration`. Kode yang harus identik di kelimanya tidak disalin-tempel seperti di `ocr-*`, melainkan satu package bersama `libs/ocr_common` yang di-`pip install` ke tiap image. Layout kode tiap service dijelaskan di [Struktur Repo](#struktur-repo).

## Daftar Isi

- [Struktur Repo](#struktur-repo)
- [Alur Request](#alur-request)
- [Menjalankan Secara Lokal](#menjalankan-secara-lokal)
- [Menjalankan dengan Docker](#menjalankan-dengan-docker)
- [Environment Variables](#environment-variables)
- [Endpoint API](#endpoint-api)
- [Skenario Testing via Nama File](#skenario-testing-via-nama-file)
- [Endpoint Testing (load test tim ML)](#endpoint-testing-load-test-tim-ml)
- [Observability](#observability)
- [Database](#database)
- [Mengganti Mock dengan Model Asli](#mengganti-mock-dengan-model-asli)
- [Menambah Service Baru](#menambah-service-baru)
- [Testing, Lint & openapi.yaml](#testing-lint--openapiyaml)
- [Keterbatasan & Langkah Berikutnya](#keterbatasan--langkah-berikutnya)

## Struktur Repo

Monorepo: satu lib bersama + satu folder per deployable. Tiap service memakai layout berlapis yang sama, sengaja lebih sederhana dari hexagonal:

| Folder di `app/` | Isi | Boleh import |
|---|---|---|
| `main.py` | `create_app(...)` + lifespan (buka/tutup koneksi) | `api`, `dependencies`, `config` |
| `config.py` | `Settings` (pydantic-settings) | `ocr_common.config` |
| `dependencies.py` | **composition root**: semua `get_*()` yang dipakai `Depends(...)`, registry `<X>_BACKENDS`, dan satu-satunya tempat `lru_cache` | semua lapisan di bawahnya |
| `api/` | endpoint FastAPI (satu file per router) + `schemas.py` request/response | `dependencies`, `services`, `api.schemas` |
| `services/` | logika bisnis, tanpa FastAPI | `ml`, `clients`, `config` |
| `ml/` | `base.py` (Protocol engine) + satu file per backend (`paddle.py`, `efficientnet.py`, `remote.py`, `mock.py`) + `utils.py` praproses | `ocr_common.clients` |
| `clients/` | klien HTTP ke service lain (hanya orchestrator) | `ocr_common.clients` |

Aturannya satu: panah import hanya ke bawah. `services/` tidak tahu FastAPI, `ml/` tidak tahu settings (yang membaca settings dan memilih backend adalah `dependencies.py`), dan test mengganti implementasi lewat `app.dependency_overrides[get_x]`, bukan `monkeypatch` path modul.

```
nilam-ocr-npwp/
├── libs/ocr_common/                  # package bersama (pip install), bukan salinan per service
│   ├── pyproject.toml                # ocr-common; extra [db] untuk SQLAlchemy
│   ├── ocr_common/
│   │   ├── web/                      # toolkit FastAPI; TIDAK berisi endpoint bisnis
│   │   │   ├── app.py                #   create_app(): request_id, metrik, logging, ServiceError -> envelope, /health /ready /metrics
│   │   │   ├── security.py           #   verify_api_key (X-API-Key; API_KEY + API_KEYS untuk rotasi)
│   │   │   ├── request_id.py         #   request_id di contextvar: middleware, job, pengiriman outbox, header X-Request-ID
│   │   │   ├── logging.py / metrics.py   # log JSON/teks ber-request_id; metrik HTTP + endpoint /metrics
│   │   │   ├── envelope.py / schemas.py   # envelope response, ErrorResponse + helper error()/success_examples()
│   │   │   ├── intake.py             #   terima `file` ATAU `file_url` (presigned MinIO)
│   │   │   └── openapi.py            #   python -m ocr_common.web.openapi -> openapi.yaml service
│   │   ├── pipeline/                 # mesin tahap pipeline async
│   │   │   ├── stage.py              #   StagePipeline: klaim job, kerja di background, simpan, callback, handoff
│   │   │   ├── repository.py / repository_sql.py   # <tahap>_jobs / <tahap>_results: memory / PostgreSQL
│   │   │   ├── callbacks.py          #   callback ke orkestrator + handoff ke tahap berikutnya, dengan retry
│   │   │   ├── outbox.py / outbox_sql.py / outbox_status.py   # transactional outbox + relay + isi GET /outbox
│   │   │   ├── outcomes.py           #   baris outcome milik orkestrator (ORCHESTRATION_OUTCOME_TABLE)
│   │   │   ├── reaper.py / results.py    # job basi dijalankan lagi; hasil tahap sebelumnya untuk handoff by reference
│   │   │   ├── metrics.py            #   metrik pipeline: job per outcome, durasi, job basi, pengiriman + backlog outbox
│   │   │   ├── factory.py            #   build_stage_pipeline / build_outbox_relay / build_next_stage_client
│   │   │   ├── schemas.py            #   payload antar tahap dan callback (Pydantic)
│   │   │   └── tables.py / database.py   # definisi kolom SEMUA tabel repo ini + engine SQLAlchemy async
│   │   ├── clients/                  # HTTP keluar
│   │   │   ├── remote.py             #   RemoteModelClient: klien ke model ML dan ke service lain
│   │   │   └── fetch_url.py          #   unduh dokumen dari URL dengan aman (SSRF)
│   │   ├── config.py                 # BaseServiceSettings (API_KEY/API_KEYS, LOG_*, batas upload) + PipelineSettings
│   │   ├── errors.py                 # ServiceError + subclass per status (BadRequest, NotFound, UpstreamUnavailable, ...)
│   │   ├── types.py                  # TypedDict data antar tahap: OcrBlock, OcrResult, StructuredField, FinalResult, ...
│   │   ├── registry.py               # build_backend(): pilih implementasi dari <APP>_BACKEND
│   │   ├── image_validation.py       # tipe / kosong / ukuran
│   │   ├── npwp.py                   # DOCUMENT_TYPE, NPWP_FIELDS, final_result(), contract_fields()
│   │   └── testing.py                # helper test yang sama untuk semua service
│   └── tests/
├── services/
│   ├── orchestrator/                 # port 8034, pintu masuk: app/{main, config, dependencies, api, services, clients}, tests (tanpa ml/)
│   │   ├── app/api/extract_ocr.py    # POST /v1/extract-ocr, GET /v1/extract-ocr/{request_id}; api/testing.py kembaran -test
│   │   ├── app/services/             # extract_service.py (cek file, guardrails, hand-off, tunggu); pipeline_waiter.py (wait, snapshot)
│   │   └── app/clients/              # guardrails.py (/v1/guardrails/check), extraction.py (hand-off), stages.py (status tahap)
│   ├── extraction/                    # port 8030 (slot ocr-npwp)
│   │   ├── Dockerfile · requirements.txt · requirements.lock · .env.example · openapi.yaml · pyproject.toml
│   │   ├── app/
│   │   │   ├── main.py               # create_app(...) + lifespan
│   │   │   ├── config.py             # Settings
│   │   │   ├── dependencies.py       # composition root: OCR_BACKENDS, get_ocr_engine, get_pipeline, get_*_service
│   │   │   ├── api/jobs.py           # pipeline async: POST/GET /v1/extraction/jobs, GET .../outbox, POST .../outbox/release
│   │   │   ├── api/extraction.py      # /v1/extraction/extract
│   │   │   ├── api/schemas.py        # request/response Pydantic
│   │   │   ├── services/job_service.py        # kerja tahap OCR + payload handoff ke structuring
│   │   │   ├── services/extraction_service.py
│   │   │   ├── ml/base.py            # Protocol OcrEngine
│   │   │   ├── ml/paddle.py · ml/remote.py · ml/mock.py   # satu backend per file
│   │   │   └── ml/utils.py           # normalisasi jawaban model -> blok teks
│   │   └── tests/
│   ├── guardrails/                   # port 8031, internal: app/{main, config, dependencies, api, services, ml}, tests
│   │   ├── weights/best_model.pt     # checkpoint EfficientNet-B0 (di-gitignore; di-COPY ke image saat build)
│   │   ├── app/ml/efficientnet.py    # EfficientNetPageClassifier (efficientnet); remote.py, mock.py; utils.py praproses
│   │   └── app/services/pages.py     # gambar -> 1 halaman, PDF -> halaman per halaman (PyMuPDF)
│   ├── structuring/                  # port 8032: sama; ml/npwp_rules.py + ml/rule_based.py; app/vendor/ aturan ML engineer
│   └── scoring/                      # port 8033: sama (tanpa handoff; tahap terakhir); ml/heuristic.py + ml/trust_model.py
├── deploy/helm/                      # chart GKE: Deployment + Service + NetworkPolicy per service, PDB, HPA; deploy.sh, migrate-db.sh
├── api/gateway.openapi.yaml          # satu spec untuk tim gateway/orkestrasi, dirakit dari openapi.yaml tiap service
├── docs/RINGKASAN.md                 # ringkasan satu halaman untuk reviewer
├── db/                               # migrasi Alembic + peta pemilik tabel (db/README.md)
├── docker-compose.yml                # 5 image, 5 container, satu network
├── docker-compose.db.yml             # overlay PostgreSQL lokal: satu database, semua tabel di schema public
├── scripts/smoke_e2e.py              # memerankan Orkestrasi pusat: orchestrator -> jobs -> GET status, lewat container
├── tools/                            # (di-gitignore, hanya di laptop) tracker UI pipeline + outbox, load-tester k6
├── Makefile · pyproject.toml (ruff) · requirements-dev.txt
```

Batas yang dijaga: **service tidak saling import**. Satu-satunya jalur antar service adalah HTTP (`ocr_common.pipeline.NextStageClient` untuk handoff, `services/orchestrator/app/clients/` untuk pintu masuk), sehingga tiap service bisa dipindah ke repo, VM, atau cluster lain tanpa mengubah kode. Yang boleh dibagi hanya `ocr_common`.

## Alur Request

### Pipeline async (sequence diagram)

Orkestrasi **tidak menyediakan endpoint callback**. Cara hasil sampai ke Orkestrasi adalah **tabel milik Orkestrasi** (`ORCHESTRATION_OUTCOME_TABLE`, mis. `orchestration_extract_ocr`, di database yang sama): tiap tahap meng-upsert baris `request_id` di transaksi job-nya sendiri. **Orkestrasi hanya membaca `downstream_status`** (`processing` → `completed` | `failed`) untuk menjawab polling client; `downstream_stage`, `result_data`, `error_code`, dan `error_message` ikut ditulis untuk data hasil dan diagnosa. Mode callback (`ORCHESTRATION_URL`) tetap ada di kode sebagai opsi, dijelaskan di akhir bagian ini, tapi tidak dipakai.

```
Client ─► Orkestrasi pusat: generate request_id · POST url file + document_type · POST ekstrak OCR
Orkestrasi pusat ─► orchestrator:8034 POST /v1/extract-ocr             SATU-SATUNYA PANGGILAN
                (request_id, document_type, params, file | file_url)
   orchestrator: file_url? unduh sekali dari MinIO (host di FILE_URL_ALLOWED_HOSTS) : pakai file
                > MAX_UPLOAD_BYTES (2,5 MB)        ◄─ 413 "Ukuran dokumen melebihi batas ..."   (sebelum guardrails)
                PDF > MAX_DOCUMENT_PAGES (2)       ◄─ 400 "Jumlah halaman melebihi batas ..."   (sebelum guardrails)
                ─► guardrails:8031 POST /v1/guardrails/check (file + request_id), SINKRON
                     hanya model guardrails per halaman ◄─ 200 {passed, reason, document, pages}
   passed=false ◄─ 400 {errors: DOWNSTREAM_VALIDATION_ERROR, job_status: failed, guardrails: 1, message, params}
                ─► Orkestrasi pusat jawab client 400 dari respons sinkron ini. Tidak ada tahap yang jalan,
                   jadi `downstream_status` untuk request yang ditolak guardrails TIDAK pernah ditulis pipeline.
   passed=true  ─► extraction:8030 POST /v1/extraction/jobs              202 segera
                     (request_id, document_type, guardrails, file | file_url yang sama)
                   orchestrator MENUNGGU maks. PIPELINE_WAIT_SECONDS (default 15 dtk, sejak request diterima):
                   GET /v1/{extraction,structuring,scoring}/jobs/{request_id} berurutan,
                   tiap PIPELINE_POLL_INTERVAL_SECONDS (default 0,5 dtk)
                ◄─ 200 {job_status: completed, data: {nomor_npwp, nama, flag, flag_reason}, guardrails: 0, params}
                ◄─ 422 {job_status: failed, errors: <TAHAP>_FAILED, guardrails: 0, message, params} gagal
                ◄─ 202 {job_status: processing, data: null, guardrails: null, params}            belum selesai
                ─► Orkestrasi pusat: 200/422 → jawab client; 202 → jawab client 202, hasil menyusul di tabelnya


   extraction:   ┌─ SATU TRANSAKSI (klaim) ──────────────────────────────────────────────────────┐
                │ INSERT ocr_jobs (PROCESSING, input: {document_type, guardrails, file_url})    │
                │   ON CONFLICT DO NOTHING                                                       │
                │ UPSERT orchestration_extract_ocr {downstream_status: processing, stage: OCR}  │
                └───────────────────────────────────────────────────────────────────────────────┘
                file_url? unduh dari MinIO : pakai file dari payload
                OCR (paddle / remote / mock)
                ┌─ SATU TRANSAKSI (hasil) ──────────────────────────────────────────────────────┐
                │ UPSERT ocr_results                                                            │
                │ UPDATE ocr_jobs DONE                                                          │
                │ INSERT pipeline_outbox (stage: OCR, kind: handoff, payload: body structuring) │
                └─ COMMIT ── lalu bangunkan relay proses ini ───────────────────────────────────┘
                job gagal (file rusak, model mati, …): satu transaksi juga:
                  UPDATE ocr_jobs FAILED + UPSERT orchestration_extract_ocr {failed, OCR_FAILED, error_message}

   relay OCR:   claim baris stage=OCR yang due, FOR UPDATE SKIP LOCKED, lease 30 dtk, attempts+1
                ─► structuring:8032 POST /v1/structuring/jobs   202 ─► DELETE baris
                     (request_id, document_type, guardrails [+ ocr, kecuali PIPELINE_HANDOFF_BY_REFERENCE])
                     5xx / timeout ─► next_attempt_at = now + backoff (0,5 dtk ×2 … maks 5 mnt),
                                      diulang sampai pesan berumur 24 jam
                     4xx, atau lewat 24 jam ─► DEAD LETTER: failed_at + last_error, baris TETAP ADA,
                                      + UPSERT orchestration_extract_ocr {failed, STRUCTURING_FAILED}

   structuring: [transaksi klaim: INSERT structuring_jobs + orkestrasi {processing, STRUCTURING}]
                ocr tidak di body? SELECT ocr_results
                aturan npwp_rules ML engineer: nomor + nama per posisi, flag lunak (tidak pernah menolak)
                [transaksi hasil: UPSERT structuring_results + DONE + outbox handoff scoring]
   relay STRUCTURING:
                ─► scoring:8033 POST /v1/scoring/jobs                  202 ─► DELETE baris
                     (request_id, document_type, guardrails [+ ocr, structuring])

   scoring:     [transaksi klaim: INSERT scoring_jobs + orkestrasi {processing, SCORING}]
                structuring tidak di body? SELECT structuring_results + ocr_results
                trust model
                ┌─ SATU TRANSAKSI (hasil) ──────────────────────────────────────────────────────┐
                │ UPSERT scoring_results + UPDATE scoring_jobs DONE                             │
                │ UPSERT orchestration_extract_ocr {status_code: 200, completed,                │
                │        result_data: {nomor_npwp, nama} bentuk kontrak extract-ocr}            │
                └───────────────────────────────────────────────────────────────────────────────┘
                (tahap terakhir: tidak ada handoff, tidak ada baris outbox)

Client ─► Orkestrasi: GET status by request_id  ─► Orkestrasi: SELECT downstream_status (+ result_data) FROM orchestration_extract_ocr
Orkestrasi pusat ─► orchestrator:8034 GET /v1/extract-ocr/{request_id}: kontrak yang sama (200/202/400/422, params null),
          dibaca sekali dari GET /v1/{extraction,structuring,scoring}/jobs/{request_id}, tanpa menunggu; 404 = tidak ada tahap
          yang punya job (ditolak model guardrails, atau belum dikirim)
Ops    ─► tiap tahap: GET /v1/<tahap>/outbox {pending, retrying, oldest_pending_seconds, dead_letters}
          GET /v1/<tahap>/jobs/{request_id} untuk debugging (internal, dari dalam namespace)
```

Cara membacanya:

- **Hasil dan statusnya satu transaksi.** Baris tabel Orkestrasi di-upsert di transaksi yang sama dengan `<tahap>_jobs` / `<tahap>_results`, jadi tidak ada keadaan "hasil tersimpan tapi Orkestrasi tidak tahu" atau sebaliknya. `processing` + tahapnya saat job diklaim, `completed` + `result_data` dari scoring, `failed` + `<TAHAP>_FAILED` + `error_message` saat sebuah tahap gagal atau handoff ke tahap berikutnya mati permanen. Definisi kolomnya di [db/external/](db/external), peta pemiliknya di [db/README.md](db/README.md).
- **Handoff lewat outbox** (`PIPELINE_OUTBOX=true`). Baris `pipeline_outbox` ditulis di transaksi hasil, dikirim oleh relay di tiap service (hanya baris `stage`-nya sendiri; `FOR UPDATE SKIP LOCKED` + lease membuatnya aman untuk banyak replika). Relay bangun saat proses yang sama commit, dan tetap polling tiap `PIPELINE_OUTBOX_INTERVAL_SECONDS` untuk baris milik replika lain. Kalau pod mati sesudah commit, relay lain menemukan barisnya. Tanpa outbox, task job sendiri yang memanggil tahap berikutnya dengan retry 3×, dan gagalnya dicatat ke tabel Orkestrasi sebagai `<TAHAP BERIKUTNYA>_FAILED`.
- **Gagal kirim tidak pernah menghapus pesan.** 5xx diulang dengan backoff; 4xx dan pesan yang lewat umur menjadi dead letter yang menetap di tabel, terlihat di `GET /v1/<tahap>/outbox` dan di log `WARNING` relay, dan pada saat itu tabel Orkestrasi ditandai `failed` atas nama tahap berikutnya supaya request tidak menggantung. Melepasnya manual: `failed_at = NULL, next_attempt_at = now()`.
- **Handoff bisa berupa referensi** (`PIPELINE_HANDOFF_BY_REFERENCE=true`, butuh `DATABASE_URL` yang sama di ketiga service). Payload ke tahap berikutnya hanya `request_id`, `document_type`, dan laporan guardrails; blok OCR dan field structuring dibaca tahap berikutnya dari `<tahap>_results`, sehingga payload handoff dan baris `pipeline_outbox` tidak membesar untuk PDF banyak halaman. Penerima menerima kedua bentuk: `ocr` / `structuring` yang ada di body dipakai apa adanya; yang tidak ada dibaca dari database, dan kalau tidak ada di sana job `FAILED` dengan pesan yang menyebutkannya. Tanpa `DATABASE_URL`, body tanpa `ocr` / `structuring` ditolak 422.
- **Backlog outbox bisa dilihat.** Tidak ada service lain yang mengamati `pipeline_outbox`, jadi tiap service melaporkannya sendiri: `GET /v1/<tahap>/outbox` (butuh `X-API-Key`) mengembalikan `pending`, `retrying`, `oldest_pending_seconds`, dan `dead_letters` untuk tahapnya, dan relay menulis log `WARNING` "outbox <TAHAP> backlog" tiap menit selama pesan tertua lebih tua dari `PIPELINE_OUTBOX_STALE_AFTER_SECONDS` atau ada dead letter. Pasang alert pada keduanya.

Ketiga tahap memakai mesin yang sama, `ocr_common/pipeline/stage.py`; tiap service hanya mengisi kerjanya (`services/job_service.py`). Perilaku yang sama di ketiganya:

- **202 segera, kerja di background.** Job jalan sebagai `asyncio` task (referensi kuat, di-drain saat shutdown). Job yang belum selesai saat batas drain habis dibatalkan, ditandai `FAILED`, dan dicatat ke tabel Orkestrasi, jadi orkestrator bisa mengirimnya ulang. Kerja sinkron yang CPU-bound (structuring, scoring) dijalankan di threadpool supaya event loop tetap menerima job lain.
- **Job basi diambil lagi** (`PIPELINE_STALE_JOBS`, butuh `DATABASE_URL`). Proses yang mati mendadak (OOM, SIGKILL, node hilang) meninggalkan baris `jobs` berstatus `PROCESSING` tanpa pemilik. Pengambil job basi di tiap proses, seperti relay outbox, tiap `PIPELINE_STALE_JOB_INTERVAL_SECONDS` mengklaim baris yang lewat lease dan menjalankannya lagi dari database: `input` yang disimpan saat klaim (`document_type`, laporan guardrails, `file_url`) dan hasil tahap sebelumnya di `*_results`. Structuring dan scoring selalu bisa diulang; OCR hanya kalau request ke orchestrator memakai `file_url`: orchestrator meneruskan URL itu (bukan byte-nya) ke `/v1/extraction/jobs`, extraction menyimpannya di `input` dan mengunduh lagi saat mengulang. Upload inline tidak disimpan, jadi job OCR-nya `FAILED` minta kirim ulang. Konsekuensinya: `FILE_URL_ALLOWED_HOSTS` harus diisi di extraction juga, dan presigned URL harus hidup lebih lama dari `PIPELINE_JOB_LEASE_SECONDS`. Orkestrasi tidak perlu lagi mengirim ulang, tapi kiriman ulang tetap aman (klaim ulang yang sama).
- **Idempoten per `request_id`.** `INSERT … ON CONFLICT DO NOTHING`: `request_id` yang sama dikirim lagi tetap 202 dengan `duplicate: true` dan kerja **tidak** diulang. Pengecualian: job `FAILED`, atau job `PROCESSING` yang melewati lease (`PIPELINE_JOB_LEASE_SECONDS`, default 300 detik; artinya proses yang menjalankannya mati tanpa mencatat apa pun, mis. OOM/SIGKILL), boleh diklaim ulang (`attempts` bertambah), supaya orkestrator bisa retry. Klaim ulang atomik: dari dua kiriman bersamaan hanya satu yang menang.
- **Gagal = `failed` di tabel Orkestrasi, bukan diam.** File rusak, model tidak terjangkau, tidak ada teks, dst. → `jobs.status=FAILED` + baris Orkestrasi `failed`, `<TAHAP>_FAILED`, `error_message`, dalam satu transaksi, dan rantai berhenti. Request-nya sendiri sudah dijawab 202 (atau 422 kalau masih di dalam batas tunggu orchestrator), jadi kegagalan sesudah itu hanya terlihat lewat tabel Orkestrasi dan `GET …/jobs/{request_id}`.
- **Handoff gagal dilaporkan atas nama tahap berikutnya.** Kalau extraction sudah DONE tapi structuring tidak terjangkau setelah retry (atau handoff-nya menjadi dead letter), tabel Orkestrasi ditandai `failed` + `STRUCTURING_FAILED`; tanpa itu request menggantung selamanya di `STRUCTURING`.
- **Handoff di-retry** (`PIPELINE_RETRY_ATTEMPTS`, backoff eksponensial) hanya untuk 5xx / tidak terjangkau; 4xx tidak. Pengiriman *at-least-once*: handoff yang terkirim dua kali tidak menjalankan ulang job tahap berikutnya, kecuali job itu sudah `FAILED` (diklaim ulang, sama seperti kiriman ulang dari Orkestrasi).

**Mode callback (opsional, tidak dipakai).** Kalau suatu saat Orkestrasi menyediakan endpoint callback, isi `ORCHESTRATION_URL` (+ `ORCHESTRATION_API_KEY`, `ORCHESTRATION_CALLBACK_PATH`, default `/v1/callbacks/stage`): tiap tahap lalu juga mengirim `POST {"request_id", "stage": "OCR | STRUCTURING | SCORING", "status": "DONE | FAILED", "result", "error_message"}` lewat outbox yang sama (handoff dikirim lebih dulu; urutan callback antar-tahap tidak dijamin, jadi harus idempoten per tahap), dengan `result` hasil akhir hanya di `SCORING`/`DONE` (`final_result` di `ocr_common/npwp.py`). Bentuk body ada di `OrchestrationCallback.notify` (`ocr_common/pipeline/callbacks.py`), satu tempat untuk ketiga service. Kedua mode boleh aktif bersamaan; di luar `local` salah satunya wajib. Setiap panggilan keluar (handoff, callback, service model) membawa header `X-Request-ID` berisi `request_id` request itu. Respons 200 orchestrator dan `result_data` di tabel Orkestrasi diturunkan dari hasil yang sama ke bentuk kontrak `extract-ocr` (`contract_fields` di `ocr_common/npwp.py`): `nomor_npwp` dan `nama` (nama badan untuk kartu perusahaan) dengan `confidence` 0/1 dari trust model dan `FIELD_CONFIDENCE_THRESHOLD`.

### Kontrak lama (sinkron): sudah dihapus

`generate-request-id` → `extract-ocr` → `get-ocr-result` di extraction (port 8030) dihapus pada 24 September 2026, tabelnya `ocr_npwp_requests` ikut dibuang migrasi `0007`. `POST /v1/extract-ocr` di guardrails (port 8031) juga sudah tidak ada: pintunya pindah ke orchestrator (port 8034). `/v1/structuring/structure` dan `/v1/scoring/score` (skor dokumen **heuristik lama**, bukan dari ML engineer) tersisa hanya untuk debugging.

Di dalam tiap service: `api/*.py` (controller; hanya memanggil service dan membungkus jawabannya, tidak menangkap error) → `services/*_service.py` (logika, tidak tahu HTTP; gagal = `raise BadRequest(...)`, `NotFound(...)`, `UpstreamUnavailable(...)` dari `ocr_common.errors`, yang diubah menjadi envelope oleh satu handler di `create_app`) → `ml/*.py` (pembungkus model, dipilih di `dependencies.py` lewat env) / `repositories/`. Data antar lapisan dan antar tahap memakai `TypedDict` dari `ocr_common/types.py` (`OcrBlock`, `OcrResult`, `StructuredField`, `StructuringResult`, `FieldConfidences`, `FinalResult`), jadi bentuknya terbaca di tanda tangan fungsi dan diperiksa ty.

## Menjalankan Secara Lokal

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt   # Windows; Linux/macOS: .venv/bin/pip
# memasang libs/ocr_common (editable) + requirements kelima service + tooling

for s in orchestrator guardrails extraction structuring scoring; do cp services/$s/.env.example services/$s/.env; done
# isi API_KEY di tiap .env (boleh sama semua; tiap service memakai API_KEY-nya sendiri ke service lain kalau *_API_KEY kosong)

make run-guardrails     # terminal 1, :8031
make run-structuring    # terminal 2, :8032
make run-scoring        # terminal 3, :8033
make run-extraction      # terminal 4, :8030  (default menunjuk structuring di 127.0.0.1:8032)
make run-orchestrator   # terminal 5, :8034  (default menunjuk 127.0.0.1:8030-8033)
make smoke              # rantai extract-ocr lewat kelimanya
```

Untuk melihat pipeline, baris outbox, dan callback berjalan di browser: `tools/tracker/run.sh --stack` (Redis + kelima container + PostgreSQL + UI di `http://127.0.0.1:5173`), lihat [tools/tracker/README.md](tools/tracker/README.md). Log di laptop berbentuk teks (`LOG_FORMAT` default `text` saat `ENVIRONMENT=local`); di cluster JSON.

Swagger UI tiap service di `http://127.0.0.1:<port>/docs`. Untuk mengerjakan satu service saja cukup jalankan service itu; endpoint per-app-nya tidak butuh service lain. Hanya orchestrator yang butuh service lain (keempatnya) untuk `extract-ocr`.

Editor: tiap service punya `pyproject.toml` yang menunjuk root import ke folder service dan ke `libs/ocr_common` (`[tool.ty.environment]`; ty adalah satu-satunya type checker repo ini, sama dengan `make typecheck` dan CI, jadi pakai ekstensi ty di editor); buka repo dari root dan reload window sekali setelah `pip install`.

## Menjalankan dengan Docker

Lima image, dibangun dari **root repo** karena tiap image butuh `libs/ocr_common`:

```bash
make build                      # docker compose build  (5 image: nilam-ocr-{orchestrator,guardrails,extraction,structuring,scoring})
make up                         # build + jalankan kelimanya, port hanya di 127.0.0.1
make ps                         # status container
make up-db                      # sama, plus PostgreSQL lokal (jobs/results ketiga tahap)
make smoke                      # memerankan Orkestrasi pusat: orchestrator -> jobs -> GET status
make logs-extraction
make down
```

Build satu image secara manual:

```bash
docker build -f services/scoring/Dockerfile -t nilam-ocr-scoring:1.0.0 .
```

**Build bisa diulang.** Tiap Dockerfile memasang dependensi dari `requirements.lock` dengan `pip --require-hashes` (semua versi transitif dan hash tiap wheel terkunci), memasang `ocr_common` tanpa resolusi ulang, dan memakai base image `python:3.11-slim@sha256:…` ber-digest. `requirements.txt` tetap daftar pin langsung yang dibaca manusia dan `requirements-dev.txt`; lock-nya dihasilkan dari situ plus dependensi `ocr_common`:

```bash
make lock                       # semua: services/*/requirements.lock + db/requirements.lock (uv pip compile)
make lock-scoring               # satu service, setelah mengubah requirements.txt atau pyproject ocr_common
make lock LOCK_FLAGS=--upgrade  # naikkan versi transitif; tanpa ini versi yang sudah terkunci dipertahankan
```

Image yang dipakai cluster dibangun dan didorong dari laptop oleh `deploy/helm/deploy.sh`, yang menolak working tree yang belum di-commit (kecuali `--allow-dirty`, untuk uji coba) dan memberi tag SHA pendek commit itu, jadi setiap image di Artifact Registry bisa dilacak ke kodenya dan dibangun ulang byte per byte dari lock dan digest yang sama. Sebelum deploy jalankan `make lock-check`: gagal kalau ada `requirements.lock` yang ketinggalan dari `requirements.txt` atau pyproject `ocr_common` (lock-nya ikut diperbarui, tinggal di-commit). Memperbarui digest base image: `docker buildx imagetools inspect python:3.11-slim`, salin `Digest` ke keenam Dockerfile.

Image guardrails berisi torch CPU (~1 GB) dan bobot model. Deploy terpisah: jalankan tiap image di mana saja, lalu sambungkan rantainya lewat env:

- orchestrator: `GUARDRAILS_SERVICE_URL`, `EXTRACTION_SERVICE_URL`, `STRUCTURING_SERVICE_URL`, `SCORING_SERVICE_URL`; extraction: `STRUCTURING_SERVICE_URL`; structuring: `SCORING_SERVICE_URL` (+ `*_API_KEY` kalau key-nya berbeda).
- extraction, structuring, scoring: `DATABASE_URL` yang sama (satu database, schema per service), plus `ORCHESTRATION_OUTCOME_TABLE` (hasil ke tabel orkestrasi) dan/atau `ORCHESTRATION_URL` (+ `ORCHESTRATION_API_KEY`) kalau orkestrasi punya endpoint callback.
- Orkestrasi pusat hanya perlu satu alamat: orchestrator (`:8034`).

Di compose, URL antar service sudah di-override ke nama service (`http://structuring:8032`, dst.).

Uji mode callback (opsional) tanpa orkestrator: `SMOKE_CALLBACK_PORT=8039 make smoke` membuat smoke script ikut menerima callback; arahkan ketiga service ke sana dengan `ORCHESTRATION_URL=http://host.docker.internal:8039` (container) atau `http://127.0.0.1:8039` (proses bare). Script memeriksa urutan `OCR → STRUCTURING → SCORING` dan menampilkan hasil akhir.

Deploy ke GKE memakai chart Helm di [deploy/helm/README.md](deploy/helm/README.md): satu Deployment, Service, PDB, dan NetworkPolicy per service, HPA opsional untuk guardrails, Service `nilam-ocr-npwp` (hanya port 8034, orchestrator) sebagai pintu masuk Orkestrasi pusat (`deploy/helm/deploy.sh <service|all>`, migrasi lewat `migrate-db.sh`).

## Environment Variables

Semua service (`ocr_common.config.BaseServiceSettings`):

| Variable | Wajib? | Default | Keterangan |
|---|---|---|---|
| `API_KEY` | **Ya** | – | Header `X-API-Key` yang harus dikirim pemanggil, sekaligus key yang dikirim service ini ke service lain. Tidak boleh kosong; service gagal start |
| `API_KEYS` | Tidak | – | Key tambahan yang juga diterima, dipisah koma, untuk rotasi tanpa downtime: (1) tambahkan key baru ke `API_KEYS` di semua service, (2) pindahkan pemanggil ke key baru, (3) jadikan key baru `API_KEY` dan kosongkan `API_KEYS`. Di Helm dibaca dari Secret key `API_KEYS` (opsional) |
| `LOG_FORMAT` | Tidak | `json`; `local`: `text` | `json` = satu objek JSON per baris (`severity`, `message`, `request_id`, `service`) yang dibaca Cloud Logging; `text` = satu baris teks untuk terminal. Setiap baris memuat `request_id` request atau job yang sedang dikerjakan |
| `LOG_LEVEL` | Tidak | `INFO` | Level logger root |
| `ENVIRONMENT` | Di laptop: `local` | `production` | `local` (laptop) atau `dev` / `staging` / `production` (ter-deploy). Di luar `local` service **menolak start** kalau: `AUTH_DISABLED=true`, backend `mock`, `DATABASE_URL` kosong, `ORCHESTRATION_URL` dan `ORCHESTRATION_OUTCOME_TABLE` dua-duanya kosong, atau alamat service menunjuk localhost. Default `production` supaya konfigurasi yang lupa mengisinya gagal keras, bukan berjalan dengan pengaman mati. `dev` bukan mode longgar: cluster dev GKE bernama "dev" |
| `AUTH_DISABLED` | Tidak | `false` | `true` = pemeriksaan `X-API-Key` dimatikan. Hanya diterima dengan `ENVIRONMENT=local`; service mencatat peringatan saat start. `API_KEY` tetap wajib karena dipakai sebagai key keluar |
| `SERVICE_BASE_URL` | Tidak | – | Nilai `servers` di OpenAPI (`/docs`) |
| `PORT` | Tidak | per service | orchestrator 8034, extraction 8030, guardrails 8031, structuring 8032, scoring 8033 |
| `MAX_UPLOAD_BYTES` | Tidak | `2621440` (2,5 MB) | Berlaku untuk `file` maupun `file_url`. Lebih besar → **413** `Ukuran dokumen melebihi batas 2,5 MB, pastikan hanya mengunggah dokumen NPWP`, sebelum model apa pun jalan. Angka dari ML engineer (NPWP umumnya 1–2 MB, ada yang 2,1 MB) |
| `FILE_URL_ALLOWED_HOSTS` | Produksi: ya, kalau `file_url` menunjuk storage internal; **di orchestrator dan extraction**, karena orchestrator mengunduhnya untuk cek guardrails lalu meneruskan `file_url` ke extraction | – | Host yang boleh diunduh lewat `file_url`, dipisah koma; entri berawalan titik (`.example.internal`) = semua subdomain. Host terdaftar boleh resolve ke IP privat (MinIO internal), tapi tidak ke loopback / link-local (metadata server). Kosong = hanya host yang resolve ke alamat publik. Di `ENVIRONMENT=local` pemeriksaan alamat dimatikan. Redirect tidak pernah diikuti |
| `ALLOWED_CONTENT_TYPES` | Tidak | `["image/jpeg","image/jpg","image/png","application/pdf"]` | JSON list |
| `FIELD_CONFIDENCE_THRESHOLD` | Tidak | `0.5` | Probabilitas trust model minimal agar `confidence` sebuah field bernilai `1` (di bawahnya `0`). Dipakai orchestrator untuk respons `/v1/extract-ocr` **dan** scoring untuk baris `ORCHESTRATION_OUTCOME_TABLE`, jadi **isinya harus sama di keduanya**. Keputusan bisnis: sesuaikan setelah divalidasi |

extraction, structuring, scoring (`ocr_common.config.PipelineSettings`, pipeline async):

| Variable | Wajib? | Default | Keterangan |
|---|---|---|---|
| `DATABASE_URL` | Produksi: ya | – | PostgreSQL untuk `<schema>.jobs` / `<schema>.results`. Kosong = in-memory: hanya untuk dev satu proses; hilang saat restart dan **tidak idempoten antar replika** |
| `ORCHESTRATION_URL` | Salah satu dengan `ORCHESTRATION_OUTCOME_TABLE` | – | Base URL Orkestrasi untuk callback tahap. Kosong = tidak ada callback yang dikirim maupun diantrekan; hasil lewat `ORCHESTRATION_OUTCOME_TABLE` dan `GET /v1/<tahap>/jobs/{request_id}` |
| `ORCHESTRATION_CALLBACK_PATH` | Tidak | `/v1/callbacks/stage` | |
| `ORCHESTRATION_API_KEY` | Tidak | = `API_KEY` | `X-API-Key` yang dikirim ke Orkestrasi |
| `ORCHESTRATION_TIMEOUT_SECONDS` | Tidak | `10.0` | |
| `PIPELINE_RETRY_ATTEMPTS` / `PIPELINE_RETRY_DELAY_SECONDS` | Tidak | `3` / `0.5` | Retry callback dan handoff (5xx / tidak terjangkau), backoff ×2 |
| `PIPELINE_DRAIN_TIMEOUT_SECONDS` | Tidak | `30.0` | Saat shutdown, tunggu job yang masih jalan; sisanya dibatalkan dan dilaporkan `FAILED` (diberi 5 detik untuk mencatat dan mengirim callback). Dengan `PIPELINE_OUTBOX`, relay lalu diberi 5 detik lagi untuk mengirim pesan yang diantrekan job-job itu. Jaga `preStop` + nilai ini + 5 (+ 5 bila outbox) < `terminationGracePeriodSeconds` |
| `PIPELINE_JOB_LEASE_SECONDS` | Tidak | `300.0` | Job `PROCESSING` yang lebih tua dari ini boleh diklaim ulang saat `request_id`-nya dikirim lagi. Harus jauh di atas durasi job terlama (timeout model + unduhan) |
| `PIPELINE_STALE_JOBS` | Tidak | `true` | Tiap proses menjalankan pengambil job basi: job `PROCESSING` yang `updated_at`-nya lebih tua dari `PIPELINE_JOB_LEASE_SECONDS` (pemiliknya mati tanpa sempat mencatat) diklaim ulang (`attempts` bertambah, `FOR UPDATE SKIP LOCKED`) dan dijalankan lagi dari data di database: `input` yang disimpan saat klaim plus `*_results` tahap sebelumnya. Hanya aktif dengan `DATABASE_URL`; butuh migrasi `0004`. Job OCR hanya bisa diulang kalau request ke orchestrator memakai `file_url` (orchestrator meneruskan URL-nya, bukan byte-nya); upload inline hilang bersama prosesnya dan job itu langsung `FAILED` dengan pesan minta kirim ulang |
| `PIPELINE_STALE_JOB_INTERVAL_SECONDS` / `PIPELINE_STALE_JOB_BATCH` | Tidak | `30.0` / `10` | Jeda antar pencarian job basi, dan berapa job yang diambil per putaran |
| `PIPELINE_OUTBOX` | Tidak | `false` | `true` = callback dan handoff ditulis ke `pipeline_outbox` dalam transaksi job, lalu dikirim relay. Butuh `DATABASE_URL` dan migrasi `0003`. Menghilangkan kehilangan pesan saat pod mati dan kopling latensi ke callback, tapi urutan callback antar-tahap tidak lagi dijamin |
| `TESTING_ENDPOINTS` | Tidak | `false` | `true` = orchestrator dan ketiga tahap membuka kembaran `-test` dari endpoint pipeline (`/v1/extract-ocr-test` + `GET /v1/extract-ocr-test/{request_id}`, `/v1/<tahap>/jobs-test`): pipeline yang sama di tabel `testing_*`, tanpa callback dan tanpa menulis ke tabel Orkestrasi. Untuk load test tim ML di dev; butuh migrasi `0006`. `false` = route-nya tidak ada (404). Lihat [Endpoint Testing](#endpoint-testing-load-test-tim-ml) |
| `PIPELINE_OUTBOX_INTERVAL_SECONDS` | Tidak | `1.0` | Jeda relay saat outbox kosong. Pesan baru langsung membangunkan relay di proses yang sama, jadi nilai ini hanya berlaku untuk pesan sisa milik replika lain |
| `PIPELINE_OUTBOX_BATCH` | Tidak | `20` | Pesan per putaran relay |
| `PIPELINE_OUTBOX_LEASE_SECONDS` | Tidak | `30.0` | Lama sebuah pesan "dipegang" satu relay sebelum relay lain boleh mencobanya lagi |
| `PIPELINE_OUTBOX_MAX_BACKOFF_SECONDS` | Tidak | `300.0` | Batas atas jeda retry sebuah pesan (backoff ×2 mulai dari `PIPELINE_RETRY_DELAY_SECONDS`) |
| `PIPELINE_OUTBOX_MAX_AGE_SECONDS` | Tidak | `86400.0` | Pesan yang masih gagal 5xx setelah berumur segini menjadi dead letter: berhenti dicoba, tetap di tabel dengan `failed_at` dan `last_error`. Pesan yang dijawab 4xx langsung menjadi dead letter |
| `PIPELINE_HANDOFF_BY_REFERENCE` | Tidak | `false` | `true` = handoff ke tahap berikutnya hanya membawa `request_id`, `document_type`, dan laporan guardrails; blok OCR dan field structuring **tidak** ikut, tahap berikutnya membacanya dari `ocr_results` / `structuring_results` di database yang sama. Butuh `DATABASE_URL`, dan database itu harus sama di ketiga service. Menyusutkan payload handoff dan baris `pipeline_outbox` untuk PDF banyak halaman. Penerima selalu menerima kedua bentuk: kalau `ocr` / `structuring` ada di body, itu yang dipakai |
| `PIPELINE_OUTBOX_STALE_AFTER_SECONDS` | Tidak | `300.0` | Relay menulis log `WARNING` tiap menit selama pesan tertua yang belum terkirim lebih tua dari ini (atau ada dead letter) |
| `ORCHESTRATION_OUTCOME_TABLE` | Salah satu dengan `ORCHESTRATION_URL` | – | Nama tabel milik orkestrasi yang ikut ditulis di transaksi job (mis. `orchestration_extract_ocr`). Kosong = tidak menulis. Tabelnya wajib punya `request_id` unik plus kolom `downstream_status`, `downstream_stage`, `status_code`, `error_code`, `error_message`, `result_data`, `ds`; DDL yang dibutuhkan ada di [db/external/](db/external) |

extraction:

| Variable | Wajib? | Default | Keterangan |
|---|---|---|---|
| `EXTRACTION_BACKEND` | Tidak | `mock` | `remote` (service model ML engineer, `/v1/predict/json`), `paddle` (API lama PaddleOCR, `/ocr`), atau `mock` |
| `EXTRACTION_OCR_URL` | Jika `remote` / `paddle` | – | `remote`: mis. `http://localhost:8082`; `paddle`: mis. `http://10.213.128.67:8070` |
| `EXTRACTION_OCR_API_KEY` | Tidak | – | Hanya `remote`: `X-API-Key` yang dikirim **ke service model** (bukan `API_KEY` service ini) |
| `EXTRACTION_OCR_TIMEOUT_SECONDS` | Tidak | `30.0` | |
| `EXTRACTION_OCR_PARAMS` | Tidak | `{}` | Hanya `remote`: JSON object yang dikirim sebagai form field tambahan di tiap panggilan `/v1/predict/json` (mis. `{"document_type": "npwp"}`). Di produksi model OCR ML engineer melayani banyak jenis dokumen dan menerima parameternya per panggilan |
| `STRUCTURING_SERVICE_URL` | Tidak | `http://127.0.0.1:8032` | Tujuan handoff tahap OCR |
| `STRUCTURING_API_KEY` | Tidak | = `API_KEY` | Key structuring, kalau berbeda |
| `STRUCTURING_TIMEOUT_SECONDS` | Tidak | `10.0` | |

orchestrator (pintu masuk, tanpa database):

| Variable | Wajib? | Default | Keterangan |
|---|---|---|---|
| `GUARDRAILS_SERVICE_URL` | Produksi: ya | `http://127.0.0.1:8031` | Service guardrails; `POST /v1/guardrails/check` untuk tiap dokumen. `GUARDRAILS_API_KEY` = `API_KEY` kalau kosong |
| `MAX_DOCUMENT_PAGES` | Tidak | `2` | PDF dengan halaman lebih dari ini ditolak **400** `Jumlah halaman melebihi batas, pastikan hanya mengunggah dokumen NPWP` di sini, sebelum guardrails dipanggil (permintaan ML engineer: NPWP asli maksimal 2 halaman). PDF yang tidak terbaca juga 400 di sini. Halaman dihitung dengan PyMuPDF, parser yang sama dengan render guardrails |
| `GUARDRAILS_TIMEOUT_SECONDS` | Tidak | `20.0` | Cek guardrails tidak diulang (sudah di dalam anggaran waktu pemanggil): 503/504 diteruskan ke Orkestrasi pusat, yang boleh mengirim ulang |
| `GUARDRAILS_SKIP_ALLOWED` | Tidak | `false` | Boleh tidaknya request melewati model guardrails dengan field `skip_guardrails=true`. `false`: request seperti itu ditolak **403** `GUARDRAILS_SKIP_NOT_ALLOWED`, jadi API key saja tidak cukup untuk melewatinya. Pengecekan file (tipe, `MAX_UPLOAD_BYTES`, `MAX_DOCUMENT_PAGES`) dan aturan structuring tetap jalan; trust model bekerja tanpa probabilitas guardrails. Nyala = peringatan di log saat start dan tiap request yang melewati guardrails (metrik `guardrails_skipped_total`) |
| `EXTRACTION_SERVICE_URL` | Produksi: ya | `http://127.0.0.1:8030` | Tujuan hand-off (`POST /v1/extraction/jobs`) dan status tahap OCR. `EXTRACTION_API_KEY`, `EXTRACTION_TIMEOUT_SECONDS` (`10.0`) |
| `STRUCTURING_SERVICE_URL` / `SCORING_SERVICE_URL` | Produksi: ya | `http://127.0.0.1:8032` / `:8033` | Untuk membaca status tahap (`GET …/jobs/{request_id}`) saat menunggu dan untuk `GET /v1/extract-ocr/{request_id}`. `STRUCTURING_API_KEY` / `SCORING_API_KEY` = `API_KEY` kalau kosong. Di luar `local` keempat URL ditolak kalau menunjuk localhost |
| `PIPELINE_RETRY_ATTEMPTS` / `PIPELINE_RETRY_DELAY_SECONDS` | Tidak | `3` / `0.5` | Retry hand-off ke extraction (5xx / tidak terjangkau), backoff ×2 |
| `PIPELINE_WAIT_SECONDS` | Tidak | `15` | Berapa lama `POST /v1/extract-ocr` menunggu pipeline, dihitung sejak request diterima. Selesai dalam waktu ini → **200** `job_status: completed` dengan `data` (atau **422** `<TAHAP>_FAILED` kalau gagal); belum → **202** `processing`. `0` = tidak menunggu (selalu 202 setelah hand-off). HTTP timeout pemanggil harus di atas nilai ini |
| `PIPELINE_POLL_INTERVAL_SECONDS` | Tidak | `0.5` | Jeda antar-cek status tahap selama menunggu. Menambah latensi paling banyak sebesar ini tiap kali sebuah tahap masih berjalan; lebih kecil = lebih cepat tapi lebih banyak `GET` |

`MAX_UPLOAD_BYTES` dan `FIELD_CONFIDENCE_THRESHOLD` (tabel pertama) juga dipakai di sini. Semua penjagaan file (tipe, kosong, ukuran 413, jumlah halaman) ada **di orchestrator**; guardrails hanya menjalankan model. Samakan `FIELD_CONFIDENCE_THRESHOLD` dengan scoring.

guardrails (internal, model):

| Variable | Wajib? | Default | Keterangan |
|---|---|---|---|
| `GUARDRAILS_BACKEND` | Tidak | `mock` | `efficientnet` (model di proses ini), `remote` (service model ML engineer), atau `mock`; `.env.example` mengaktifkan `efficientnet` |
| `GUARDRAILS_MODEL_URL` | Jika `remote` | – | Service model guardrails, mis. `http://localhost:8081` (`POST {url}/v1/predict/json`) |
| `GUARDRAILS_MODEL_API_KEY` | Tidak | – | `X-API-Key` yang dikirim **ke service model** (bukan `API_KEY` service ini) |
| `GUARDRAILS_MODEL_TIMEOUT_SECONDS` | Tidak | `30.0` | |
| `GUARDRAILS_MODEL_PATH` | Jika `efficientnet` | `weights/best_model.pt` | Checkpoint (`model_state_dict`, `class_names`, `image_size`, `reject_threshold`) |
| `GUARDRAILS_DEVICE` | Tidak | `cpu` | Inference CPU saja (image memakai wheel torch CPU); nilai lain jatuh ke cpu dengan peringatan |
| `GUARDRAILS_TORCH_THREADS` | Tidak | bawaan torch | Batas thread torch; samakan dengan limit CPU container |
| `GUARDRAILS_REJECT_THRESHOLD` | Tidak | dari checkpoint (`0.5`) | Backend lokal saja. Halaman reject kalau `proba_reject >= ambang`. Ini **default**-nya: ambang dari Orkestrasi pusat (`GUARDRAILS_THRESHOLD_URL`) didahulukan |
| `GUARDRAILS_THRESHOLD_URL` | Tidak | – | Base URL Orkestrasi pusat, pemilik ambang reject (bisa diubah tanpa deploy di sini). Guardrails `GET {URL}{GUARDRAILS_THRESHOLD_PATH}` → `{"reject_threshold": 0.5}`. Kosong, tidak terjangkau, atau jawabannya bukan angka di antara 0 dan 1 → nilai terakhir yang pernah diberikan, kalau belum pernah → default. Ambang yang dipakai tercatat di `document.reject_threshold`. Endpoint aslinya belum ada; dummy-nya di tracker (`http://127.0.0.1:8090`) |
| `GUARDRAILS_THRESHOLD_PATH` | Tidak | `/v1/thresholds/guardrails` | Path endpoint ambang di Orkestrasi pusat; sesuaikan begitu endpoint aslinya ada |
| `GUARDRAILS_THRESHOLD_API_KEY` | Tidak | – | Dikirim sebagai `X-API-Key` ke endpoint ambang |
| `GUARDRAILS_THRESHOLD_TIMEOUT_SECONDS` | Tidak | `2` | Batas tunggu endpoint ambang; lewat → default, dokumen tetap dinilai |
| `GUARDRAILS_THRESHOLD_CACHE_SECONDS` | Tidak | `60` | Ambang disimpan selama ini per pod; perubahan di Orkestrasi berlaku paling lambat setelah ini |
| `GUARDRAILS_DOCUMENT_POLICY` | Tidak | `all` | Backend lokal saja. `all`: accepted hanya kalau semua halaman accepted; `majority`: accepted > reject |
| `GUARDRAILS_PDF_DPI` / `GUARDRAILS_MAX_PAGES` | Tidak | `150` / `20` | Backend lokal saja. Render PDF per halaman |

structuring: `STRUCTURING_BACKEND` (`npwp_rules`, default: aturan regex + posisi dari ML engineer; atau `rule_based`: regex berbasis label, hanya untuk teks berlabel). Data rujukan aturan, semuanya opsional (file yang tidak ada = pemeriksaan itu tanpa sinyal, service tetap jalan; default nama file yang sama di `app/vendor/npwp_rules/data/`): `WILAYAH_CODES_PATH` (`kode_wilayah.json`, kode kecamatan Depdagri untuk NPWP 16 digit), `KPP_CODES_PATH` (`kpp_codes.json`, kode KPP untuk NPWP 15 digit), `NAME_MASTER_PATH` (`name_lnmast.xlsx`, master nama untuk tie-break kandidat nama; data internal, pasang lewat volume). Tahap berikutnya `SCORING_SERVICE_URL` (`http://127.0.0.1:8033`), `SCORING_API_KEY` (= `API_KEY`), `SCORING_TIMEOUT_SECONDS` (`10.0`).

scoring: `SCORING_MODEL_PATH` (`weights/trust_model.joblib`: trust model ML engineer, dipakai pipeline dan `/v1/scoring/confidence`). Hanya untuk endpoint lama `/v1/scoring/score`: `SCORING_BACKEND` (`heuristic`), `SCORING_APPROVE_THRESHOLD` (`0.8`), `SCORING_REVIEW_THRESHOLD` (`0.5`).

## Endpoint API

Semua response memakai envelope `ocr-*`: `{status_code, status_desc, message, data, errors, request_id}`; `extract-ocr` menambah `guardrails` (skor dokumen) sejajar `data`. `request_id` diambil dari form/path, atau dari header `X-Request-ID` untuk endpoint per-app (kalau tidak ada, dibuat `REQ_<uuid>`).

| Service | Method | Path | Body |
|---|---|---|---|
| semua | GET | `/health` | – (tanpa API key; `backends` menunjukkan implementasi aktif dan `storage`: `postgres` / `memory`). **Liveness**: hanya "proses hidup", tidak menyentuh dependensi |
| semua | GET | `/ready` | – (tanpa API key). **Readiness**: 200 `{status: ready, checks}` kalau dependensi wajib menjawab (database untuk extraction / structuring / scoring), 503 `not_ready` kalau tidak. Tahap berikutnya dan service model sengaja tidak diperiksa: gangguannya dilaporkan per job |
| semua | GET | `/metrics` | – (tanpa API key). Metrik Prometheus proses ini; lihat [Observability](#observability) |
| orchestrator | POST | `/v1/extract-ocr` | **Pintu masuk pipeline, kontrak `extract-ocr` Orkestrasi pusat.** form: `request_id`, `document_type` (default `npwp`), `params` (opsional, JSON; dikembalikan apa adanya di setiap jawaban), `skip_guardrails` (opsional, boolean; lewati model guardrails kalau `GUARDRAILS_SKIP_ALLOWED`, selain itu 403) + tepat satu dari `file` / `file_url`. Dicek tipe/ukuran, lalu dinilai guardrails. Ditolak → 400 `DOWNSTREAM_VALIDATION_ERROR`, `guardrails: 1`. Lolos → diteruskan ke `extraction/jobs`, lalu menunggu sampai `PIPELINE_WAIT_SECONDS`: 200 `job_status: completed` + `data` {`nomor_npwp`, `nama`} (`confidence` 0/1), 422 `<TAHAP>_FAILED`, atau 202 `processing` |
| orchestrator | GET | `/v1/extract-ocr/{request_id}` | Kontrak yang sama, **tanpa menunggu**: status tiap tahap dibaca sekali, berurutan. 200 / 202 / 400 (ditolak aturan structuring) / 422, `params: null`; 404 kalau tidak ada tahap yang punya job (ditolak model guardrails, atau belum dikirim); 503/504 kalau sebuah tahap tidak terjangkau. Hand-off yang mati permanen (dead letter) tetap terbaca 202: keadaan finalnya ada di callback / tabel Orkestrasi |
| guardrails | POST | `/v1/guardrails/check` | **Internal, dipanggil orchestrator untuk tiap dokumen; hanya menilai, selalu 200.** `file` / `file_url` → `passed`, `reason`, `document` {verdict, confidence, n_pages, n_approve, n_reject}, `pages[]`. Tidak memulai apa pun; untuk debugging |
| extraction | POST | `/v1/extraction/jobs` | **202.** Dipanggil orchestrator. form: `request_id`, `document_type` (default `npwp`), `guardrails` (JSON object, laporan guardrails), + tepat satu dari `file` / `file_url` |
| structuring | POST | `/v1/structuring/jobs` | **202.** JSON `{"request_id", "document_type", "guardrails", "ocr": {"blocks": [{"text", "confidence", ...}], ...}}`; `ocr` boleh dihilangkan kalau pengirim handoff by reference (dibaca dari `ocr_results`) |
| scoring | POST | `/v1/scoring/jobs` | **202.** JSON `{"request_id", "document_type", "guardrails", "ocr", "structuring": {"fields": {...}}}`; `ocr` dan `structuring` boleh dihilangkan kalau pengirim handoff by reference (dibaca dari `*_results`) |
| ketiganya | GET | `/v1/<tahap>/jobs/{request_id}` | – → `{request_id, stage, status: PROCESSING\|DONE\|FAILED, result, error_message, created_at, updated_at}`; untuk debug/rekonsiliasi, sumber status resmi tetap Orkestrasi |
| ketiganya | GET | `/v1/<tahap>/outbox` | – → `{enabled, stage, pending, retrying, oldest_pending_seconds, dead_letters}`: backlog outbox tahap ini (`PIPELINE_OUTBOX`) |
| ketiganya | POST | `/v1/<tahap>/outbox/release` | query `request_id` opsional → `{stage, request_id, released}`: dead letter tahap ini diantrekan ulang dan relay dibangunkan; 409 kalau outbox mati |
| extraction | POST | `/v1/extraction/extract` | form: `file` / `file_url` → blok teks, `confidence`, `bbox`, `page`, `model` |
| structuring | POST | `/v1/structuring/structure` | JSON `{"lines": [{"text", "confidence", "bbox"?, "page"?}]}` → `fields`, `flag`, `flag_reason`. `bbox` (dari `blocks[]` extraction) dipakai `npwp_rules` untuk mencari nama dari posisinya; tanpa itu dipakai urutan baris. Tidak ada penolakan berdasarkan isi: dokumen lain yang tergabung, CAPTCHA, screenshot "Cek NPWP", nomor/nama tidak ketemu, nama satu kata, huruf di nomor, kode wilayah/tanggal lahir/KPP tidak valid, > 2 halaman → `flag: true` + `flag_reason` (bahasa Indonesia, untuk reviewer) |
| scoring | POST | `/v1/scoring/confidence` | **Kontrak ML engineer (model 21 Sep 2026).** JSON 9 kunci (`npwp`, `npwp_score`, `npwp_candidate_count`, `name_base`, `name_score`, `avg_doc_score`, `min_doc_score`, `flag`, `guardrail_probability`; semua boleh null) → `{"npwp_confidence", "name_confidence"}`. `name_base` = nama sebelum normalisasi (`signals.name_base` structuring). Field yang nilainya null mendapat confidence null |
| scoring | POST | `/v1/scoring/score` | lama, heuristik. JSON `{"document_type", "fields": {name: {"value", "confidence"}}}` → `score`, `decision` |

**Response `/jobs`** (202): `data: {"request_id", "stage", "status", "duplicate"}`. `status` adalah `PROCESSING` untuk job baru, atau status terkini kalau `duplicate: true`. Validasi isi (file rusak, tidak ada teks, `document_type` tidak didukung) terjadi di background, jadi muncul sebagai `failed` + `<TAHAP>_FAILED` di tabel Orkestrasi (dan callback `FAILED` bila mode callback aktif), bukan 4xx; yang langsung ditolak hanya bentuk request yang salah (400 intake / JSON `guardrails`, 401, 422).

**Response guardrails** (`message: "OK"`). `passed` / `reason` adalah "true/false + alasan" di sequence diagram; orchestrator meneruskan `data` ini apa adanya sebagai form `guardrails` ke `/v1/extraction/jobs`:

```json
{
  "status_code": 200, "status_desc": "OK", "message": "OK", "errors": null, "request_id": "OCR_...",
  "data": {
    "passed": true,
    "reason": null,
    "document": {"verdict": "accepted", "confidence": 0.9821, "n_pages": 2, "n_approve": 2, "n_reject": 0},
    "pages": [
      {"page_index": 0, "proba_approve": 0.9821, "proba_reject": 0.0179, "verdict": "accepted"},
      {"page_index": 1, "proba_approve": 0.9950, "proba_reject": 0.0050, "verdict": "accepted"}
    ]
  }
}
```

`confidence` dokumen: kalau `accepted`, `proba_approve` halaman terlemah; kalau `reject`, `proba_reject` tertinggi di antara halaman yang ditolak. Saat ditolak: `"passed": false, "reason": "Document rejected by guardrails: 1/1 page(s) rejected (confidence 0.88)"`. Modelnya biner, jadi alasan hanya bisa menyebut jumlah halaman dan keyakinannya, bukan "blur" atau "bukan NPWP".

**Data `extract-ocr`** (sheet "OCR Nilam - Document Type"): `nomor_npwp` dan `nama` (nama wajib pajak, atau nama badan pada kartu perusahaan), tiap field `{"value", "confidence"}` dengan `confidence` 1 atau 0; yang tidak ditemukan `{"value": null, "confidence": 0}`. `nama_badan` terpisah hanya ada di callback hasil ke Orkestrasi. Flag dari aturan structuring **tidak** ada di data ini: 9 dari 11 flag menolak dokumen (400 `DOWNSTREAM_VALIDATION_ERROR`, `guardrails: 1`, `message` = alasannya), 2 sisanya (nama satu kata, huruf di nomor NPWP) hanya menjadi input trust model.

## Skenario Testing via Nama File

Berlaku selama backend masih mock (guardrails: `GUARDRAILS_BACKEND=mock`; extraction: `EXTRACTION_BACKEND=mock`), kecuali `delay<N>s` yang berlaku untuk semua backend selama `ENVIRONMENT=local`:

| Nama file mengandung | Pipeline async | Sumber |
|---|---|---|
| `blur` / `invalid` / `notnpwp` | guardrails `passed: false` + `reason`; orchestrator menjawab 400 `DOWNSTREAM_VALIDATION_ERROR`, tidak ada tahap yang jalan | guardrails (mock) |
| `servererror` | job OCR `FAILED` + tabel Orkestrasi `failed`, `OCR_FAILED` | extraction (mock engine) |
| `delay<N>s` (mis. `delay20s-npwp.jpg`) | tahap OCR menunggu N detik (maks. 120) sebelum bekerja, **apa pun backend-nya**, hanya dengan `ENVIRONMENT=local`. Dipakai [tools/tracker](tools/tracker) untuk memperlihatkan orchestrator menjawab 202 ketika pipeline melewati `PIPELINE_WAIT_SECONDS` | extraction (`ocr_common/simulation.py`) |
| lainnya | ketiga tahap `DONE`, data dummy deterministik dari isi file | |

`request_id` sama dua kali → 202 `duplicate: true`.

## Endpoint Testing (load test tim ML)

Supaya tim ML bisa menguji performa dengan burst request tanpa mengotori data Orkestrasi, setiap endpoint
pipeline punya kembaran `-test`. Kembaran ini **menjalankan handler dan kode yang sama persis** (model
guardrails, OCR, aturan structuring, trust model, waktu tunggu `PIPELINE_WAIT_SECONDS`, idempotensi, outbox),
hanya tempat datanya yang berbeda. Aktif hanya dengan `TESTING_ENDPOINTS=true` (di Helm: hanya
`values-ddb-dev.yaml`). Kalau setting ini mati, route-nya tidak ada dan menjawab 404.

| Live | Testing | Dipanggil oleh |
|---|---|---|
| `POST /v1/extract-ocr` (orchestrator) | `POST /v1/extract-ocr-test` | tim ML (k6 / curl) |
| `GET /v1/extract-ocr/{request_id}` (orchestrator) | `GET /v1/extract-ocr-test/{request_id}` | tim ML |
| `POST`/`GET /v1/extraction/jobs` | `POST`/`GET /v1/extraction/jobs-test` | orchestrator |
| `POST`/`GET /v1/structuring/jobs` | `POST`/`GET /v1/structuring/jobs-test` | extraction, orchestrator |
| `POST`/`GET /v1/scoring/jobs` | `POST`/`GET /v1/scoring/jobs-test` | structuring, orchestrator |

Guardrails tidak punya kembaran: `/v1/guardrails/check` tidak menyimpan apa pun, jadi kembaran di orchestrator memanggil endpoint yang sama.

Yang berbeda dari jalur live:

- **Tabel**: `testing_ocr_jobs`/`_results`, `testing_structuring_jobs`/`_results`,
  `testing_scoring_jobs`/`_results`, `testing_pipeline_outbox` (migrasi `0006`). Tabel live tidak disentuh.
- **Tidak ada efek ke Orkestrasi**: tanpa callback, tanpa `orchestration_extract_ocr`, tanpa
  `ocr.orchestration_api_events`.
- **Metrik** memakai label `stage="TESTING_OCR"` / `TESTING_STRUCTURING` / `TESTING_SCORING`, jadi angka load
  test tidak tercampur dengan traffic Orkestrasi di `/metrics`.

Kodenya: [ocr_common/testing_endpoints.py](libs/ocr_common/ocr_common/testing_endpoints.py) (nama tabel dan
path), [ocr_common/web/testing_routes.py](libs/ocr_common/ocr_common/web/testing_routes.py) (mendaftarkan
kembaran route), `get_testing_*` di `app/dependencies.py` tiap service, dan `app/api/testing.py`.

### Cara pakai

API key-nya sama dengan yang dipakai Orkestrasi (`API_KEY` di Secret `nilam-ocr-npwp-secrets`), di header
`X-API-Key`. Akses lewat port-forward ke orchestrator:

```bash
kubectl -n nilam-ocr-npwp port-forward svc/nilam-ocr-npwp-orchestrator 8034:8034

curl -X POST http://127.0.0.1:8034/v1/extract-ocr-test \
  -H "X-API-Key: $API_KEY" -F run_id=run1 -F file=@npwp.jpg
```

Jawabannya sama dengan `/v1/extract-ocr` (200 / 202 / 400 / 422). Bedanya, **`request_id` dibuat oleh
orchestrator**, tidak dikirim pemanggil: `TEST_<uuid>`, atau `TEST_<run_id>_<uuid>` kalau field opsional
`run_id` diisi (huruf, angka, `-`, `_`, maks. 40 karakter). Field `request_id` di form diabaikan. Id baru per
request berarti burst tidak pernah bentrok dengan idempotensi run sebelumnya. Cari request lewat `request_id` di
jawaban, atau semua request satu run lewat prefiks `TEST_<run_id>_`. Untuk burst dengan k6, pakai
[tools/load-tester](tools/load-tester) dengan `-e ENDPOINT=/v1/extract-ocr-test
-e TARGET=http://host.docker.internal:8034 -e API_KEY=...`.

Durasi per tahap langsung dari tabelnya (`created_at` = job diterima tahap itu, `updated_at` = selesai):

```sql
SELECT o.request_id,
       o.updated_at - o.created_at AS ocr,
       s.updated_at - s.created_at AS structuring,
       c.updated_at - c.created_at AS scoring,
       c.updated_at - o.created_at AS total,
       o.status AS ocr_status, s.status AS structuring_status, c.status AS scoring_status
FROM testing_ocr_jobs o
LEFT JOIN testing_structuring_jobs s USING (request_id)
LEFT JOIN testing_scoring_jobs c USING (request_id)
WHERE o.request_id LIKE 'TEST_run1_%'
ORDER BY o.created_at;
```

Setelah selesai, kosongkan tabelnya (isinya tidak dipakai apa pun):

```sql
TRUNCATE testing_ocr_results, testing_ocr_jobs, testing_structuring_results, testing_structuring_jobs,
         testing_scoring_results, testing_scoring_jobs, testing_pipeline_outbox;
```

### Perhatikan

- **Pod-nya sama dengan traffic Orkestrasi di dev.** Burst ikut memperlambat request mereka selama tes
  berjalan, dan server model OCR Paddle juga dipakai bersama. Kabari tim Orkestrasi sebelum tes besar.
- **Port-forward bukan jalur produksi.** Semua request lewat satu tunnel kubectl (lewat API server GKE), yang
  menambah latensi dan bisa jadi batas throughput sendiri pada laju tinggi. Kalau angkanya janggal,
  bandingkan dengan `pipeline_job_duration_seconds{stage="TESTING_..."}` di `/metrics`, yang diukur di dalam pod.
- Karena ada di pod yang sama, `/metrics` dan log bercampur dengan live; bedakan lewat label `TESTING_*` dan
  `request_id`.

## Observability

Setiap service menghasilkan tiga hal yang bisa dipantau tanpa service tambahan:

- **Log** satu baris per record, JSON di cluster (`LOG_FORMAT=json`: `time`, `severity`, `logger`, `message`, `request_id`, `service`, `exception`) dan teks di laptop. `request_id` diikat ke contextvar oleh middleware untuk setiap request, oleh pipeline untuk setiap job background dan pengiriman outbox, dan diteruskan ke service berikutnya sebagai header `X-Request-ID`, jadi satu id bisa diikuti dari orchestrator sampai scoring dengan satu filter di Cloud Logging. Orchestrator memakai `request_id` dari Orkestrasi pusat (bukan header-nya) sebagai `X-Request-ID` untuk semua panggilannya ke guardrails dan tahap-tahap.
- **Metrik Prometheus** di `GET /metrics` (tanpa API key, seperti `/health`): `http_requests_total` dan `http_request_duration_seconds` per template rute; di tahap pipeline `pipeline_jobs_total{stage,outcome=done|failed|crashed|interrupted}`, `pipeline_job_duration_seconds`, `pipeline_stale_jobs_reclaimed_total`, `pipeline_outbox_deliveries_total{kind,outcome=delivered|retry|dead}`, dan gauge backlog `pipeline_outbox_pending`, `_retrying`, `_dead_letters`, `_oldest_pending_seconds` (diperbarui relay tiap menit saat idle). Alert yang disarankan: `outcome!="done"` naik, `oldest_pending_seconds` melewati `PIPELINE_OUTBOX_STALE_AFTER_SECONDS`, `dead_letters` > 0.
- **Endpoint operasional** `GET /v1/<tahap>/outbox` (backlog) dan `POST /v1/<tahap>/outbox/release` (kirim ulang dead letter), keduanya dengan API key.

## Database

Satu database PostgreSQL; **semua tabel repo ini di schema `public`**. Database yang sama juga dipakai service orkestrasi untuk tabelnya sendiri (`orchestration_*`, `auth_*`), jadi peta lengkap siapa memiliki tabel apa ada di [db/README.md](db/README.md).

| Tabel | Pemilik | Isi |
|---|---|---|
| `ocr_jobs` / `ocr_results` | extraction | status tahap OCR per `request_id` dan blok teks mentah |
| `structuring_jobs` / `structuring_results` | structuring | status tahap structuring dan field bernama |
| `scoring_jobs` / `scoring_results` | scoring | status tahap scoring dan skor trust model |

`*_jobs`: `request_id` (PK), `status` (`PROCESSING` → `DONE` \| `FAILED`), `error_message`, `attempts`, `input` (JSONB: `document_type`, laporan guardrails, `file_url`; dipakai pengambil job basi untuk mengulang job), `created_at`, `updated_at`, `ds`. `*_results`: `request_id` (PK, FK ke `jobs`), `result` JSONB, timestamp, `ds`. Orchestrator dan guardrails tidak punya tabel: orchestrator membaca status tahap lewat API, bukan lewat database.

Kolom didefinisikan **sekali** di [`ocr_common/pipeline/tables.py`](libs/ocr_common/ocr_common/pipeline/tables.py). Migrasi Alembic di [db/](db/) dan `create_all` di test memakai definisi yang sama, dan `make db-check` gagal kalau keduanya menyimpang.

```bash
export DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/bribrain_ocr_nilam
make db-upgrade                       # pasang atau perbarui tabel
make db-check                         # definisi di kode vs database
make db-revision m="tambah kolom X"   # revisi baru dari selisihnya, lalu periksa hasilnya
```

Database yang tabelnya sudah dipasang manual cukup `make db-upgrade`: revisi baseline memakai `CREATE TABLE IF NOT EXISTS`, jadi tabel dan datanya dibiarkan dan database itu tercatat berada di revisi baseline. Di cluster, jalankan [deploy/helm/migrate-db.sh](deploy/helm/migrate-db.sh) **sebelum** men-deploy image yang membutuhkan perubahan tabelnya. PostgreSQL lokal: `make up-db` menjalankan migrasi dulu lewat service `migrate`, service lain start setelah itu selesai.

Akses lewat SQLAlchemy 2.0 async + asyncpg (pola sama dengan `ocr-orchestration`). Tanpa `DATABASE_URL` semuanya in-memory: cukup untuk dev dan test, tapi klaim job hanya idempoten di dalam satu proses.

## Mengganti Mock dengan Model Asli

Model ML di-deploy ML engineer sebagai service HTTP; repo ini membungkusnya di `services/<nama>/app/ml/`. Semua klien memakai `ocr_common.clients.remote.RemoteModelClient`: koneksi persisten, dan kegagalan dipetakan seragam (tidak bisa connect → 503, tidak menjawab dalam timeout → 504, jawaban ≥ 400 → 500 dengan detail, body bukan JSON → 500).

**Guardrails, backend `efficientnet` (sudah ada)**: checkpoint torchvision EfficientNet-B0 dari ML engineer, dimuat di proses service saat startup (CPU, ~3 detik) dan dijalankan per halaman (~70 ms/halaman di CPU). Checkpoint (kiriman 23 September 2026: epoch 32, `val_macro_f1 0.895`, menggantikan epoch 9 / 0,879) membawa metadata `class_names ['reject','accepted']`, `image_size 224`, `reject_threshold 0.5`. **Urutan kelas dibaca dari checkpoint**, bukan diasumsikan: checkpoint baru menaruh `reject` di indeks 0, kode serving ML engineer (`guardrail/model.py`) masih memakai urutan `['accepted','reject']` dan pada checkpoint ini akan menerima halaman kosong dan menolak kartu asli (dibuktikan dengan foto NPWP asli, halaman putih, dan noise; `tests/test_efficientnet.py::test_a_blank_page_is_rejected` menjaganya). Ambang reject masih menunggu keputusan final ML engineer (`GUARDRAILS_REJECT_THRESHOLD`). Praproses: halaman → RGB → resize `224×224` → normalisasi ImageNet; PDF dirender per halaman oleh PyMuPDF. Taruh bobot di `services/guardrails/weights/best_model.pt` (file `best_model.pt.zip` dari ML engineer adalah arsip `torch.save`; cukup ganti namanya) dan set `GUARDRAILS_BACKEND=efficientnet`. Saat `docker build`, bobot itu **ikut ke dalam image** (`COPY services/guardrails/weights`; build gagal dengan pesan jelas kalau filenya tidak ada di build context), sehingga container tidak butuh volume. Di CI, unduh bobot dulu dengan `make weights` ([scripts/fetch_weights.py](scripts/fetch_weights.py)); sumbernya dari env supaya pilihan penyimpanan tinggal mengganti nilai:

```bash
# GCS (gsutil), MinIO (mc dengan alias), atau presigned URL dari keduanya
GUARDRAILS_MODEL_URI=gs://<bucket>/npwp-guardrails/v1/best_model.pt
# GUARDRAILS_MODEL_URI=s3://<alias>/<bucket>/npwp-guardrails/v1/best_model.pt
# GUARDRAILS_MODEL_URI=https://<host>/<bucket>/npwp-guardrails/v1/best_model.pt?X-Amz-...
GUARDRAILS_MODEL_SHA256=<sha256 file>   # opsional, file yang tidak cocok ditolak
make weights && make build
```

Simpan bobot di path berversi yang tidak pernah ditimpa (`.../v1/`, `.../v2/`), supaya rollback image berarti rollback bobot juga. Penyimpanannya (GCS atau MinIO) belum ditetapkan; dev lokal cukup menyalin file secara manual seperti sekarang. Image memasang wheel torch **CPU** dari index PyTorch dan menyetel `GUARDRAILS_DEVICE=cpu`. **Catatan validasi:** pada dataset lokal (halaman mutasi rekening, Paket 1/2) model menolak hampir semua halaman dan tidak berkorelasi dengan label manusia di sana; itu konsisten dengan model yang dilatih untuk foto NPWP, tapi belum diverifikasi dengan foto NPWP asli. Uji dengan sampel NPWP sungguhan sebelum dipakai untuk memutuskan, dan sesuaikan `GUARDRAILS_REJECT_THRESHOLD` bila perlu.

**Guardrails, backend `remote` (sudah ada)**: klien ke service model guardrails milik ML engineer. Kontrak model:

```bash
curl -X POST http://localhost:8081/v1/predict/json -H "X-API-Key: dummy-key" -F "file=@sample_npwp.jpg"
# {"status_code": 200, "message": "OK",
#  "data": {"document": {"verdict", "confidence", "n_pages", "n_approve", "n_reject"},
#           "pages": [{"page_index", "proba_approve", "proba_reject", "verdict"}]}}
```

Aktifkan dengan `GUARDRAILS_BACKEND=remote` + `GUARDRAILS_MODEL_URL` (+ `GUARDRAILS_MODEL_API_KEY`). Kontrak **kita** (`POST /v1/guardrails/check`, dipanggil orchestrator) tidak berubah; service ini tetap menambah yang tidak dimiliki service model: envelope dan `passed`/`reason` (tipe, ukuran, dan jumlah halaman sudah dicek orchestrator sebelum berkas sampai di sini). Model yang mati atau lambat menjadi 503/504 di sini, dan orchestrator meneruskannya apa adanya. Bedanya dengan `efficientnet`: berkas dikirim **utuh** (PDF dirender di service model) dan `data.document` / `data.pages` diteruskan **apa adanya**, jadi ambang reject, kebijakan dokumen, dan render PDF adalah keputusan service model; `GUARDRAILS_REJECT_THRESHOLD`, `_DOCUMENT_POLICY`, `_PDF_DPI`, `_MAX_PAGES` tidak berlaku. Hanya field kontrak yang diambil (field tambahan dibuang); bentuk lain, termasuk `verdict` di luar `accepted`/`reject`, menjadi 500 `guardrails model returned an unexpected response` alih-alih vonis tebakan. Status ≥ 400 dari service model (termasuk 401 karena key salah) menjadi 500 dengan detailnya, bukan diteruskan: 401 itu salah konfigurasi kita, bukan salah pemanggil kita. Image dengan backend ini tidak butuh torch maupun bobot, tapi Dockerfile sekarang masih memasang keduanya. Test live: `cd services/guardrails && GUARDRAILS_MODEL_URL=http://localhost:8081 GUARDRAILS_MODEL_API_KEY=dummy-key python -m pytest tests/test_guardrails_live.py`.

**Extraction, backend `remote` (sudah ada)**: klien ke service model extraction milik ML engineer. Kontrak model (`nilamnpwp/ocr`, 23 September 2026):

```bash
curl -X POST http://localhost:8082/v1/predict/json -H "X-API-Key: dummy-key" -F "file=@sample_npwp.jpg"
# {"models": {"detection": "PP-OCRv6_medium_det", "recognition": "PP-OCRv6_medium_rec"},
#  "num_pages": 1,
#  "pages": [{"page_index": 0,
#             "rec_texts":  ["npwp", "KPP PRATAMA WATAMPONE", "95.844.800.1-805.000", "RAHMAT HIDAYAT", ...],
#             "rec_scores": [0.9847, 0.9804, 0.99996, 0.9931, ...],
#             "rec_polys":  [[[260,155],[610,178],[599,352],[249,329]], ...]}],
#  "n_boxes": 10, "avg_doc_score": 0.9813, "min_doc_score": 0.9438}
```

Yang dipakai hanya `pages` dan `models` (`avg/min_doc_score` dihitung ulang scoring dari blok); bentuk lama (list halaman telanjang, atau `{"data": [...]}`) tetap diterima. `EXTRACTION_OCR_PARAMS` dikirim sebagai form field tambahan. Satu elemen `pages` per halaman, dan ketiga array-nya **paralel** (indeks `i` = satu baris teks): itu hasil mentah PaddleOCR. Pemetaan: `rec_texts[i]` → `text`, `rec_scores[i]` → `confidence` (dibulatkan 4 desimal), `rec_polys[i]` (4 titik, bisa miring) → `bbox` kotak tegak yang melingkupinya, `page_index` → `page` (default: posisi elemen). Teks kosong dilewati; `[]` → `blocks: []` (di pipeline menjadi `No text lines to structure` dari structuring). `rec_polys` boleh tidak ada (`bbox: null`), tapi kalau ada harus sejajar. Envelope `{"data": [...]}` juga diterima. Array yang **tidak sama panjang** menjadi 500 `extraction OCR model returned an unexpected response`, bukan dipotong diam-diam: confidence yang menempel ke baris yang salah akan merusak skor dokumen. `data.model` diisi dari `models.detection+recognition` (`null` pada bentuk lama). Status ≥ 400 dari service model (termasuk 401 karena key salah) menjadi 500 dengan detailnya. Aktifkan dengan `EXTRACTION_BACKEND=remote` + `EXTRACTION_OCR_URL` (+ `EXTRACTION_OCR_API_KEY`); kontrak **kita** (`/v1/extraction/jobs`, `/v1/extraction/extract`, `extract-ocr`) tidak berubah. Response asli di atas disimpan sebagai fixture test (`services/extraction/tests/fixtures/remote_npwp_response.json`). Test live: `cd services/extraction && EXTRACTION_REMOTE_URL=http://localhost:8082 EXTRACTION_REMOTE_API_KEY=dummy-key python -m pytest tests/test_extraction_remote_live.py`.

**Extraction, backend `paddle` (API lama)**: klien ke PaddleOCR PP-OCRv6 (`POST {EXTRACTION_OCR_URL}/ocr`, multipart `file`). Pemetaan response `pages[].texts[]{text, score, poly}` → `blocks[]{text, confidence, bbox, page}`, `models.detection+recognition` → `model`. File yang tidak terbaca (`num_pages: 0`) → `blocks: []`, di pipeline menjadi 400 `No text lines to structure` dari structuring. Aktifkan dengan `EXTRACTION_BACKEND=paddle` + `EXTRACTION_OCR_URL`.

**Scoring, trust model (sudah ada)**: `services/scoring/weights/trust_model.joblib` dari ML engineer (3 KB, ikut di repo dan di-COPY ke image; versi retrain **21 September 2026**, diterima 23 September). Isinya, dibaca dari file-nya: dict `pipeline` (`SimpleImputer(median)` → `StandardScaler` → `LogisticRegression`, scikit-learn **1.9.0**), `feature_cols` (**9 fitur**: `field_is_npwp`, `field_score`, `shape_confidence`, `field_multiple_candidates`, `name_max_char_len`, `avg_doc_score`, `min_doc_score`, `flag`, `guardrail_probability`; `field_corrected` dan `n_boxes_per_page` dibuang), `train_report` (n=1860 = 930 dokumen × 2 field, 1738 benar / 122 salah, AUC 0,93). Modelnya **per field**: satu baris fitur untuk nomor NPWP dan satu untuk nama; keluarannya P(field itu benar). Versi scikit-learn harus sama dengan saat training: beda versi = service menolak start, bukan sekadar peringatan.

Pemetaan payload → fitur adalah port dari `scoring/trust_scoring.py` ML engineer (`app/ml/trust_model.py`; diverifikasi memberi angka yang sama persis dengan kode mereka): `shape_confidence` = 2 untuk nomor 16 digit / 1 untuk 15 digit / 0 lainnya, dan 2 untuk nama ≥ 2 kata / 1 satu kata / 0 kosong; `name_max_char_len` = token terpanjang nama; keduanya diukur pada **`name_base`** (bacaan sebelum normalisasi, `signals.name_base` dari structuring), bukan nama akhir. Asal tiap kunci payload di pipeline ada di `app/services/confidence_service.py` (`npwp*`/`name_base`/`flag` dari structuring, `avg`/`min_doc_score` dari blok OCR, `guardrail_probability` dari `document.confidence` guardrails yang `accepted`). Perilaku model yang perlu diketahui: nomor 16 digit dipercaya jauh lebih tinggi dari 15 digit (contoh README ML engineer: 0,90 vs 0,33), nomor yang kehilangan digit (huruf salah baca dibuang) ≈ 0,02, nama satu kata atau kata tergabung (`MUHAMADRINOSUKIRMA`) ≈ 0, `flag` hanya sedikit menggeser angka (fitur, bukan veto). Contoh keluaran di README mereka (`0.93` / `0.88`) **bukan hasil hitungan**: kode mereka sendiri memberi 0,3283 / 0,0406 untuk payload contoh itu (dijaga `tests/test_trust_model.py`).

**Menambah backend lain** (structuring dan scoring menunggu dokumentasi modelnya):

1. Tambahkan file `services/<nama>/app/ml/<nama_backend>.py` berisi satu kelas yang memenuhi Protocol di `app/ml/base.py` (method yang sama dengan mock-nya); untuk model HTTP terima `RemoteModelClient` lewat constructor.
2. Daftarkan di `<X>_BACKENDS` di `app/dependencies.py` (`"nama": lambda settings: ...`) dan tambahkan field URL/timeout di `app/config.py`.
3. Set `<APP>_BACKEND=nama` di `.env`. Tidak ada controller, service, atau skema yang berubah.

Kontrak method per app: guardrails lokal `classify(filename, pages: list[PIL.Image]) -> [(proba_approve, proba_reject), ...]` (+ atribut `reject_threshold`), atau guardrails remote `async check_document(filename, content, content_type) -> {"document", "pages"}` (service memilih dari ada-tidaknya `check_document`); extraction `async extract(filename, content, content_type) -> {"blocks", "model"}`; structuring `structure(lines) -> {field: {"value", "confidence", "source"}}` untuk semua `NPWP_FIELDS`; scoring `score(fields) -> {"score", "field_scores", "reasons"}` + atribut `supported_document_types`.

## Menambah Service Baru

Salin salah satu folder `services/<nama>` (yang paling kecil: `scoring`; untuk service tanpa model: `orchestrator`), ganti nama dan port. Nama service dipakai chart Helm sebagai nama port Kubernetes, jadi maksimal 15 karakter, huruf kecil/angka, tanpa `-`. Lalu daftarkan di semua tempat yang menyebut daftar service:

- `Makefile`: `SERVICES`, `PORT_<nama>`, dan satu rule `lock-<nama>` (lock-nya dengan `--extra db` hanya kalau service memakai database), lalu `make lock-<nama>` dan `make openapi`.
- `requirements-dev.txt`, root `pyproject.toml` (`[tool.ruff] src`), `docker-compose.yml` (plus `docker-compose.db.yml` kalau butuh database).
- `.github/workflows/ci.yml`: matrix job `image` (service, port, env mock).
- `deploy/helm/deploy.sh` (`ALL_SERVICES`) dan `deploy/helm/nilam-ocr-npwp/values.yaml` (`services.<nama>`: image, port, `entrypoint`, `pipeline`, `upstreams`, `startupFailureThreshold`, resources); service yang memanggilnya menambahkannya ke `upstreams`.
- Kalau ada operasinya yang dipanggil Orkestrasi pusat: `scripts/build_gateway_openapi.py` dan `api/index.html`; `scripts/smoke_e2e.py` dan `tools/` kalau ikut rantai request.

Kalau menjadi tahap baru pipeline async: tambah konstanta `STAGE_*` di `ocr_common/pipeline/stage.py`, prefix tabelnya di `PIPELINE_TABLE_PREFIXES` (`ocr_common/pipeline/tables.py`) plus revisi Alembic baru (`make db-revision`), `dependencies.py` + `services/job_service.py` + `api/jobs.py` (salin dari structuring), arahkan `get_next_stage()` tahap sebelumnya ke `/v1/<nama>/jobs`, dan tambahkan klien statusnya di `services/orchestrator/app/clients/stages.py`.

## Testing, Lint & openapi.yaml

```bash
make test          # lib + kelima service (pytest dijalankan di folder masing-masing)
make test-scoring  # satu service
make lint          # ruff seluruh repo
make typecheck     # ty per package
make openapi       # tulis ulang openapi.yaml tiap service dari kodenya, lalu api/gateway.openapi.yaml
make api-docs      # Swagger UI keenam spec tanpa menjalankan service: http://127.0.0.1:8088/api/
```

Test unit tiap service memakai stub untuk model, jadi tidak butuh jaringan: pipeline async (`tests/test_jobs.py`) mengganti Orkestrasi dan tahap berikutnya dengan perekam dari `ocr_common.testing` (`RecordingCallback`, `RecordingNextStage`) dan menunggu job background dengan `wait_for_job`. Implementasi diganti lewat `app.dependency_overrides[get_x]` pada fungsi di `app/dependencies.py`; `conftest.py` tiap service menyediakan fixture-nya (`use_classifier` di guardrails, `use_engine` di extraction; di orchestrator guardrails, extraction, dan penunggu pipeline diganti stub secara otomatis), jadi test tidak perlu `monkeypatch` path modul. Mesin pipelinenya sendiri (`libs/ocr_common/tests/test_jobs.py`) diuji terhadap repository in-memory **dan** SQL (SQLite sungguhan: `ON CONFLICT`, upsert, join). Rantai HTTP sungguhan antar container diuji `make smoke` (`scripts/smoke_e2e.py`). `SMOKE_LATENCY_RUNS=20 make smoke` menambah pengukuran latensi: p50 / p95 / max end-to-end dari sisi klien, jumlah jawaban 200 vs 202, dan durasi tiap tahap di server. Angkanya baru bermakna terhadap backend asli, jadi jalankan di cluster dev dengan `ORCHESTRATOR_URL`, `GUARDRAILS_URL`, `EXTRACTION_URL`, `STRUCTURING_URL`, `SCORING_URL` menunjuk ke service di sana. Test live ke PaddleOCR: `cd services/extraction && EXTRACTION_OCR_URL=http://10.213.128.67:8070 python -m pytest tests/test_extraction_live.py`.

Di `libs/ocr_common/tests`: `test_app.py` (envelope, API key termasuk rotasi `API_KEYS`, readiness), `test_outbox.py` (transaksi, relay, retry, dead letter, release, stop), `test_outcomes.py`, `test_reaper.py`, `test_config.py` (penjaga `ENVIRONMENT`), `test_fetch_url.py` (SSRF), `test_gateway_spec.py`.

`openapi.yaml` tiap service adalah turunan kode, dijaga `tests/test_openapi.py`, dan format dump-nya sama dengan `scripts/check_openapi.py` di monorepo orkestrasi.

Pemeriksaan yang dijalankan sebelum commit dan sebelum deploy, semuanya dari laptop: `make lint`, `ruff format --check .`, `make typecheck`, `make test`, `make lock-check`, dan untuk perubahan tabel `make db-check` terhadap PostgreSQL compose. Deploy tidak lewat CI: `deploy/helm/deploy.sh` yang membangun, mendorong, dan meng-upgrade release. Workflow GitHub Actions di `.github/workflows/ci.yml` menjalankan pemeriksaan yang sama plus migrasi terhadap PostgreSQL sungguhan dan build kelima image (container harus sehat dan berhenti dengan exit code 0 saat `docker stop`) kalau repo suatu saat dipasang di GitHub; image yang dibangun di sana tidak pernah di-push.

### Spec untuk tim gateway / orkestrasi

`api/gateway.openapi.yaml` adalah **satu** spec OpenAPI 3.1 berisi hanya yang dibutuhkan pihak yang mengintegrasikan pipeline, dirakit `scripts/build_gateway_openapi.py` dari `openapi.yaml` orchestrator dan ketiga tahap (jadi tidak pernah ditulis tangan, dan dijaga `libs/ocr_common/tests/test_gateway_spec.py`):

| Bagian | Isi |
|---|---|
| 1. Start the pipeline | `POST /v1/extract-ocr` di orchestrator: 200 hasil akhir, 202 masih berjalan, 400 ditolak, 422 tahap gagal |
| 2. Callbacks | webhook `stageCallback`: request yang **dikirim** tiap tahap ke `{ORCHESTRATION_URL}{ORCHESTRATION_CALLBACK_PATH}`; body `StageCallback` (OCR, STRUCTURING) atau `ScoringStageCallback` (hasil akhir), aturan retry, idempotensi, urutan |
| 3. Status | `GET /v1/extract-ocr/{request_id}` di orchestrator: kontrak yang sama, tanpa menunggu |

Tiap operasi membawa `servers` orchestrator: entry Service `nilam-ocr-npwp` dari namespace lain (variabel `namespace`, default `nilam-ocr-npwp`), Service komponen dari namespace yang sama, Compose, dan localhost; test menjaga bahwa server pertamanya entry Service di port 8034. Guardrails, `GET /v1/<tahap>/jobs/{request_id}`, panggilan internal antar tahap (`structuring/jobs`, `scoring/jobs`), dan helper sinkron sengaja tidak ikut: semuanya internal, ada di spec per service dan di `/docs` masing-masing.

Aturan penulisan yang dijaga test: setiap operasi punya `operationId` (dipakai generator klien: `extractOcr`, `getExtractOcr`), setiap respons 4xx/5xx punya contohnya sendiri, dan teks `Field(description=...)` berbahasa Inggris karena dibaca tim lain. Contoh di spec memakai data fiktif.

## Keterbatasan & Langkah Berikutnya

Keadaan per branch `refactor/arch` (September 2026). Butir yang butuh keputusan orang lain diberi tanda **[keputusan]**.

### Yang belum selesai di kode ini

- **Structuring memakai aturan ML engineer (`npwp_rules`) versi 23 September 2026**, di-vendor dari folder `regex/` repo `nilamnpwp` mereka (`services/structuring/app/vendor/npwp_rules/README.md` mencatat selisihnya), dan `app/ml/npwp_rules.py` meniru `regex/main.py` mereka: nomor per halaman lewat `npwp_priority` (16 digit mengalahkan 15), nama dari posisinya terhadap nomor, halaman pertama yang punya nilai yang dipakai. Semua pemeriksaan isi adalah **flag lunak** (`flag`, `flag_reason`), tidak ada yang menolak; koreksi homoglyph digit **dimatikan** oleh ML engineer (huruf salah baca dibuang, nomor jadi lebih pendek dan confidence-nya jatuh; ditandai `has_homoglyph`). Name matching fuzzy per `file_id` (`correct_name_with_npwp_list`) **tidak** dipakai di sini: ML engineer memutuskan itu dilakukan di orkestrator. `kode_wilayah.json` (7.230 kecamatan) dan `kpp_codes.json` (173 KPP) dari ML engineer ada di `app/vendor/npwp_rules/data/`; `name_lnmast.xlsx` (tie-break nama dikenal) **sengaja tidak dipakai dulu** (keputusan 23 September 2026); tanpa itu jarak ke nomor yang menentukan, dan `NAME_MASTER_PATH` tinggal diisi kalau nanti dipakai. **[keputusan]** nomor 15 digit tetap `XX.XXX.XXX.X-XXX.XXX` hanya bila OCR membacanya dalam bentuk bertitik lengkap; 16 digit dan nomor yang kehilangan huruf dikembalikan sebagai digit polos.
- **Model kiriman 23 September 2026 terpasang** (checkpoint guardrails epoch 32 dan trust model 21 Sep), tapi belum divalidasi dengan sampel NPWP produksi: pada dua foto NPWP dari internet model guardrails baru menerima satu (0,99) dan menolak satu (foto kartu dipegang tangan, ada wajah dan watermark, 0,99 reject). Ambang guardrails dan ambang `FIELD_CONFIDENCE_THRESHOLD` menunggu keputusan ML engineer. Trust model baru memberi confidence rendah untuk nomor 15 digit (contoh mereka: 0,33), jadi ambang 0,5 akan menandai banyak kartu lama sebagai `confidence: 0`.
- **Job hidup di memori proses; pemulihannya menunggu lease.** Job jalan sebagai `asyncio` task. Shutdown normal menunggu job selesai (`PIPELINE_DRAIN_TIMEOUT_SECONDS`), lalu sisanya dilaporkan `FAILED`. Kalau proses mati mendadak, baris `jobs` tertinggal `PROCESSING` sampai pengambil job basi (`PIPELINE_STALE_JOBS`) mengklaimnya setelah `PIPELINE_JOB_LEASE_SECONDS` (default 5 menit), jadi latensi kasus itu paling cepat sebesar lease. Dua batasan: job OCR yang dokumennya diunggah inline ke orchestrator tidak bisa diulang (langsung `FAILED` minta kirim ulang; kirim `file_url` ke orchestrator supaya bisa, URL-nya diteruskan ke extraction dan diunduh lagi saat mengulang), dan job yang berjalan lebih lama dari lease akan dianggap basi dan dijalankan dua kali, jadi lease harus jauh di atas durasi job terlama. Kalau volume naik, ganti `BackgroundRunner` di `ocr_common/pipeline/runner.py` dengan antrean yang tahan restart (Cloud Tasks / Pub/Sub) tanpa mengubah service.
- **Dead letter menunggu keputusan manusia.** Dengan outbox aktif, handoff (dan callback, bila mode itu aktif) yang gagal disimpan dan dicoba ulang sampai berumur `PIPELINE_OUTBOX_MAX_AGE_SECONDS`, lalu menetap sebagai dead letter; melepasnya (mis. setelah kontrak callback diperbaiki) lewat `POST /v1/<tahap>/outbox/release` (opsional `?request_id=`), yang mengantrekan ulang dead letter tahap itu dan membangunkan relay. Tanpa outbox, callback yang gagal setelah retry hanya dicatat di log. Apa pun modenya, orkestrasi bisa merekonsiliasi lewat `GET /v1/<tahap>/jobs/{request_id}`.
- **`PIPELINE_OUTBOX` dan `ORCHESTRATION_OUTCOME_TABLE` default mati.** Keduanya sudah diuji lokal (SQLite di test, PostgreSQL di compose) tapi belum dinyalakan di cluster dev; `values-ddb-dev.yaml` perlu diisi sebelum perilaku yang dijelaskan di [Alur Request](#alur-request) berlaku di sana.
- Kontrak lama sinkron di extraction (`generate-request-id` → `extract-ocr` → `get-ocr-result`) sudah dihapus (24 September 2026), tabelnya `ocr_npwp_requests` ikut dibuang oleh migrasi `0007`. `POST /v1/guardrails/check` sekarang dipanggil orchestrator untuk tiap dokumen; `/v1/structuring/structure` dan `/v1/scoring/score` dulu dipanggil kontrak itu, sekarang hanya untuk debugging dan belum dihapus.
- **`GET /v1/extract-ocr/{request_id}` membaca status tahap, bukan keadaan final request.** Hand-off yang gagal permanen (retry langsung habis, atau dead letter outbox) hanya tercatat di callback `FAILED` dan tabel Orkestrasi; tahap sebelumnya tetap `DONE` dan tahap berikutnya belum punya job, jadi GET menjawab 202 untuk request itu (dan bisa berubah menjadi 200 setelah `POST /v1/<tahap>/outbox/release`). Orchestrator sengaja tanpa database; menutup celah ini butuh status hand-off per request yang bisa dibaca dari tahap pengirim.

### Utang teknis yang sudah diketahui

Belum dikerjakan, diurutkan dari dampak terbesar; tiap butir perlu tiket.

1. **PostgreSQL terbuka ke publik.** Database dev bisa dihubungi langsung dari laptop lewat IP LoadBalancer, dengan user `postgres`. Default IP-nya sudah dihapus dari script deploy (`DB_HOST` wajib diisi), tapi aksesnya sendiri masih terbuka dan IP lamanya ada di riwayat git. Usulan: minta tim infra menutup akses publik (Private IP atau Cloud SQL Auth Proxy, plus authorized networks), rotasi password, dan pakai user per service dengan hak minimal.
2. **Release dev masih memakai image dari working tree kotor.** Di sisi repo, build sudah bisa diulang: dependensi terkunci per image di `requirements.lock` (versi transitif + hash, dipasang dengan `pip --require-hashes`; `ocr_common` boleh tetap `>=` di `pyproject.toml`-nya karena versi persisnya diambil dari lock), base image dipin ke digest, dan `deploy/helm/deploy.sh` hanya membangun dari commit bersih dengan tag = SHA commit. Tidak ada CI yang membangun image; deploy dari laptop. Database dev sudah di revisi `0005` (head), jadi tidak ada migrasi yang tertunda. Yang tersisa: release dev (revisi Helm 8) masih menjalankan tag `b2300c3-dirty-20260922161537` di keempat service, dibangun sebelum semua ini ada, sehingga kode yang jalan di sana tidak bisa dilacak ke satu commit. Menutupnya: commit, `make lock-check`, `deploy/helm/deploy.sh all`, lalu tulis SHA yang terpasang ke `image.tag` di `values-ddb-dev.yaml`. Mode tanpa callback (`ORCHESTRATION_OUTCOME_TABLE`) belum bisa dinyalakan di dev karena tabel `orchestration_extract_ocr` belum dibuat tim orkestrasi di database itu; sampai ada, `orchestration.url` di values tetap menunjuk ke service ini sendiri dan `PIPELINE_OUTBOX` tetap mati supaya callback yang dijawab 404 tidak menumpuk sebagai dead letter.
3. **Egress belum dibatasi.** NetworkPolicy hanya mengatur `Ingress`. `file_url` sudah diperiksa di kode (`FILE_URL_ALLOWED_HOSTS`, alamat internal ditolak, redirect tidak diikuti), tapi egress policy (DNS, PostgreSQL, PaddleOCR, Orkestrasi, host MinIO) tetap perlu sebagai lapisan kedua. Menunggu daftar tujuan final dari tim infra.
4. **Observability: sisi cluster.** Kode sudah menghasilkan log JSON ber-`request_id` (ikut ke service berikutnya lewat `X-Request-ID`) dan metrik Prometheus di `/metrics` (lihat [Observability](#observability)). Yang belum: scraping `/metrics` (PodMonitor / Managed Prometheus), dashboard, dan aturan alert untuk `pipeline_jobs_total{outcome!="done"}`, `pipeline_outbox_oldest_pending_seconds`, dan `pipeline_outbox_dead_letters`; tracing OpenTelemetry belum dipasang, `request_id` di log adalah penggantinya.
5. **Struktur di dalam service.** Layout `app/` dengan composition root, `Protocol` per engine, `TypedDict` untuk data antar tahap (`ocr_common/types.py`), dan exception domain (`ocr_common/errors.py`) yang menjadi envelope di satu handler `create_app` sudah dikerjakan; tidak perlu hexagonal penuh. Yang tersisa: payload guardrails masih `dict[str, Any]` (bentuknya milik model), dan `RemoteModelClient` masih meneruskan 4xx service lain sebagai `ServiceError` mentah karena statusnya memang data.
6. **Docstring di lapisan service.** Seluruh API publik `ocr_common` sudah berdocstring; lapisan `services/` dan `api/` tiap service masih sebagian, dan semantik yang tidak obvious (idempotensi, lease, urutan callback) tetap dijelaskan di README ini sebagai sumber utama.
