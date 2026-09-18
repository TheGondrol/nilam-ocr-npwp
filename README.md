# nilam-ocr-npwp — OCR NPWP sebagai empat service

Service OCR untuk dokumen NPWP (kartu identitas pajak) Indonesia, dipecah menjadi **empat service yang di-deploy terpisah** (empat image Docker), mengikuti sequence diagram NILAM OCR Orchestration:

| Service | Port | Image | Peran |
|---|---|---|---|
| **guardrails** | 8031 | `nilam-ocr-guardrails` | "ServiceGuardrails": dipanggil Orkestrasi **sinkron**. Klasifikasi tiap halaman `accepted`/`reject` dengan model EfficientNet-B0 (lokal, CPU) + vonis dokumen; selalu 200 dengan `data.passed` (true/false) + `data.reason` |
| **ekstraksi** | 8030 | `nilam-ocr-ekstraksi` | "ServiceOCR": tahap pertama pipeline async (`/v1/ekstraksi/jobs` → 202, OCR di background, callback, handoff ke structuring). Juga OCR mentah sinkron (`/v1/ekstraksi/extract`) dan kontrak lama `generate-request-id` → `extract-ocr` → `get-ocr-result` (slot `ocr-npwp`, `OCR_NPWP_SERVICE_URL`) |
| **structuring** | 8032 | `nilam-ocr-structuring` | "ServiceStructuring": `/v1/structuring/jobs` → 202, baris teks → `nomor_npwp`, `nama`, `nama_badan` dengan confidence per field, callback, handoff ke scoring |
| **scoring** | 8033 | `nilam-ocr-scoring` | "ServiceScoring", tahap terakhir: `/v1/scoring/jobs` → 202, skor dokumen 0..1 + `approve`/`review`/`reject`; callback-nya membawa **hasil akhir** ke Orkestrasi |

Orkestrasi (repo `nilam-ocr-orchestration`) memegang `request_id`, status per tahap (`orkestrasi.requests`, `orkestrasi.stage_logs`), dan polling client; repo ini hanya keempat service di atas.

Kontrak, envelope, auth `X-API-Key`, dan layout kode tiap service sama dengan service `ocr-*` di `nilam-ocr-orchestration`. Kode yang harus identik di keempatnya tidak disalin-tempel seperti di `ocr-*`, melainkan satu package bersama `libs/ocr_common` yang di-`pip install` ke tiap image.

## Daftar Isi

- [Struktur Repo](#struktur-repo)
- [Alur Request](#alur-request)
- [Menjalankan Secara Lokal](#menjalankan-secara-lokal)
- [Menjalankan dengan Docker](#menjalankan-dengan-docker)
- [Environment Variables](#environment-variables)
- [Endpoint API](#endpoint-api)
- [Skenario Testing via Nama File](#skenario-testing-via-nama-file)
- [Database](#database)
- [Mengganti Mock dengan Model Asli](#mengganti-mock-dengan-model-asli)
- [Menambah Service Baru](#menambah-service-baru)
- [Testing, Lint & openapi.yaml](#testing-lint--openapiyaml)
- [Keterbatasan & Langkah Berikutnya](#keterbatasan--langkah-berikutnya)

## Struktur Repo

Monorepo: satu lib bersama + satu folder per deployable. Tiap service di dalamnya memakai pola **Layered by Type** yang sama dengan `ocr-*` (`api/v1` → `services` → `models`/`repositories`, plus `core` dan `schemas`), jadi orang yang sudah kenal `ocr-ktp` langsung paham.

```
nilam-ocr-npwp/
├── libs/ocr_common/                  # package bersama (pip install), bukan salinan per service
│   ├── pyproject.toml                # ocr-common; extra [db] untuk SQLAlchemy
│   ├── ocr_common/
│   │   ├── app.py                    # create_app(): middleware request_id, exception handler envelope, /health
│   │   ├── config.py                 # BaseServiceSettings (API_KEY, port, batas upload) + PipelineSettings
│   │   ├── jobs.py                   # mesin tahap pipeline async: klaim job, runner background, callback, handoff, retry
│   │   ├── jobs_sql.py               # <schema>.jobs / <schema>.results di PostgreSQL (INSERT ... ON CONFLICT DO NOTHING)
│   │   ├── security.py               # verify_api_key (X-API-Key), API_KEY_ERROR
│   │   ├── envelope.py / errors.py / schemas.py   # envelope response, ServiceError, ErrorResponse + helper error()
│   │   ├── intake.py / fetch_url.py  # terima `file` ATAU `file_url` (presigned MinIO)
│   │   ├── image_validation.py       # tipe / kosong / ukuran
│   │   ├── remote.py                 # RemoteModelClient: klien HTTP ke model ML dan ke service lain
│   │   ├── registry.py               # pilih implementasi model dari <APP>_BACKEND
│   │   ├── database.py               # engine SQLAlchemy async (opsional)
│   │   ├── npwp.py                   # DOCUMENT_TYPE, NPWP_FIELDS
│   │   ├── openapi.py                # python -m ocr_common.openapi -> openapi.yaml service
│   │   └── testing.py                # helper test yang sama untuk semua service
│   └── tests/
├── services/
│   ├── ekstraksi/                    # port 8030 (slot ocr-npwp)
│   │   ├── Dockerfile · requirements.txt · .env.example · openapi.yaml · pyproject.toml
│   │   ├── db/schema.sql             # schema ocr (jobs, results) + tabel ocr_npwp_requests (kontrak lama)
│   │   ├── src/
│   │   │   ├── main.py               # create_app(...) + lifespan
│   │   │   ├── api/v1/jobs.py        # pipeline async: POST/GET /v1/ekstraksi/jobs
│   │   │   ├── services/job_service.py        # kerja tahap OCR + payload handoff ke structuring
│   │   │   ├── core/pipeline.py      # perakitan StagePipeline (schema `ocr`) + klien tahap berikutnya
│   │   │   ├── api/v1/ocr.py         # kontrak lama orkestrator (sinkron)
│   │   │   ├── api/v1/ekstraksi.py   # /v1/ekstraksi/extract
│   │   │   ├── services/ocr_service.py        # kontrak lama: guardrails -> OCR -> structuring -> scoring
│   │   │   ├── services/ekstraksi_service.py
│   │   │   ├── clients/stages.py     # klien HTTP ke guardrails/structuring/scoring (seperti clients/ di orchestration)
│   │   │   ├── models/ekstraksi.py   # PaddleOcrEngine (paddle) + MockOcrEngine (mock)
│   │   │   ├── repositories/request_repository.py   # status request_id: memory / PostgreSQL
│   │   │   ├── core/config.py · schemas/ocr.py · schemas/ekstraksi.py
│   │   └── tests/
│   ├── guardrails/                   # port 8031: src/{main, api/v1, services, models, schemas, core}, tests
│   │   ├── weights/best_model.pt     # checkpoint EfficientNet-B0 (di-gitignore; di-COPY ke image saat build)
│   │   ├── src/models/guardrails.py  # EfficientNetPageClassifier (efficientnet) + MockPageClassifier (mock)
│   │   └── src/services/pages.py     # gambar -> 1 halaman, PDF -> halaman per halaman (PyMuPDF)
│   ├── structuring/                  # port 8032: sama, plus api/v1/jobs.py · services/job_service.py · core/pipeline.py · db/schema.sql
│   └── scoring/                      # port 8033: sama (tanpa handoff; tahap terakhir)
├── docker-compose.yml                # 4 image, 4 container, satu network
├── docker-compose.db.yml             # overlay PostgreSQL lokal: satu database, schema ocr / structuring / scoring
├── scripts/smoke_e2e.py              # memerankan Orkestrasi: guardrails -> jobs -> callback, lewat container
├── Makefile · pyproject.toml (ruff) · requirements-dev.txt
```

Batas yang dijaga: **service tidak saling import**. Satu-satunya jalur antar service adalah HTTP (`ocr_common.jobs.NextStageClient` untuk pipeline async, `services/ekstraksi/src/clients/stages.py` untuk kontrak lama), sehingga tiap service bisa dipindah ke repo, VM, atau cluster lain tanpa mengubah kode. Yang boleh dibagi hanya `ocr_common`.

## Alur Request

### Pipeline async (sequence diagram)

```
Client ─► Orkestrasi: generate request_id · POST url file + document_type · POST ekstrak OCR
Orkestrasi ─► guardrails:8031 POST /v1/guardrails/check (binary)      SINKRON
           ◄─ 200 {passed, reason, document, pages}
   passed=false ─► Orkestrasi: requests.status=REJECTED, 422 + reason ke client. Selesai.
   passed=true  ─► Orkestrasi: requests.status=OCR_PROCESSING, 202 ke client, lalu:

Orkestrasi ─► ekstraksi:8030 POST /v1/ekstraksi/jobs                   202 segera
                (request_id, document_type, guardrails, file | file_url)
   ekstraksi:   INSERT ocr.jobs (PROCESSING) ON CONFLICT DO NOTHING
                file_url? unduh dari MinIO : pakai file dari payload
                OCR (paddle / mock)
                UPSERT ocr.results + UPDATE ocr.jobs DONE
                ─► Orkestrasi  POST callback {request_id, stage: OCR, status: DONE}
                ─► structuring:8032 POST /v1/structuring/jobs          202 segera
                     (request_id, document_type, guardrails, ocr)
   structuring: INSERT structuring.jobs … regex/LLM … UPSERT structuring.results + DONE
                ─► Orkestrasi  POST callback {stage: STRUCTURING, status: DONE}
                ─► scoring:8033 POST /v1/scoring/jobs                  202 segera
                     (request_id, document_type, guardrails, ocr, structuring)
   scoring:     INSERT scoring.jobs … skor … UPSERT scoring.results + DONE
                ─► Orkestrasi  POST callback {stage: SCORING, status: DONE, result: <hasil akhir>}

Client ─► Orkestrasi: GET status by request_id (polling orkestrasi.requests + stage_logs)
```

Ketiga tahap memakai mesin yang sama, `ocr_common/jobs.py`; tiap service hanya mengisi kerjanya (`services/job_service.py`). Perilaku yang sama di ketiganya:

- **202 segera, kerja di background.** Job jalan sebagai `asyncio` task (referensi kuat, di-drain saat shutdown). Kerja sinkron yang CPU-bound (structuring, scoring) dijalankan di threadpool supaya event loop tetap menerima job lain.
- **Idempoten per `request_id`.** `INSERT … ON CONFLICT DO NOTHING`: `request_id` yang sama dikirim lagi tetap 202 dengan `duplicate: true` dan kerja **tidak** diulang. Pengecualian: job `FAILED` boleh diklaim ulang (`attempts` bertambah), supaya orkestrator bisa retry.
- **Gagal = callback `FAILED`, bukan diam.** File rusak, model tidak terjangkau, tidak ada teks, dst. → `jobs.status=FAILED` + callback `{stage, status: FAILED, error_message}`, dan rantai berhenti. Request-nya sendiri sudah dijawab 202, jadi kegagalan hanya terlihat lewat callback / `GET …/jobs/{request_id}`.
- **Handoff gagal dilaporkan atas nama tahap berikutnya.** Kalau ekstraksi sudah DONE tapi structuring tidak terjangkau setelah retry, ekstraksi mengirim callback `{stage: STRUCTURING, status: FAILED}`; tanpa itu request menggantung selamanya di `STRUCTURING`.
- **Callback dan handoff di-retry** (`PIPELINE_RETRY_ATTEMPTS`, backoff eksponensial) hanya untuk 5xx / tidak terjangkau; 4xx tidak. Callback yang tetap gagal dicatat di log dan tidak menggagalkan job.

**Kontrak callback** (yang harus disediakan Orkestrasi): `POST {ORCHESTRATION_URL}{ORCHESTRATION_CALLBACK_PATH}` (default `/v1/callbacks/stage`), header `X-API-Key`, body:

```json
{"request_id": "REQ_...", "stage": "OCR | STRUCTURING | SCORING", "status": "DONE | FAILED",
 "result": null, "error_message": null}
```

`result` hanya terisi di callback `SCORING`/`DONE` (hasil akhir untuk `requests.final_result`):

```json
{"document_type": "npwp",
 "fields": {"nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0.96}, "nama": {...}, "nama_badan": {...}},
 "scoring": {"score": 0.91, "decision": "approve", "field_scores": [...], "reasons": []},
 "guardrails": {"passed": true, "reason": null, "document": {...}, "pages": [...]}}
```

Path dan bentuk ini adalah usulan dari sisi repo ini; kalau tim orkestrasi memilih path lain cukup ubah `ORCHESTRATION_CALLBACK_PATH`, kalau bentuk body lain ubah `OrchestrationCallback.notify` di `ocr_common/jobs.py` (satu tempat untuk ketiga service).

### Kontrak lama (sinkron)

Dipertahankan sampai orkestrator pindah ke alur async. `POST /v1/extract-ocr` ke service ekstraksi:

```
orkestrator ──► ekstraksi:8030 /v1/extract-ocr (request_id + file/file_url)
                 │ 1. POST guardrails:8031 /v1/guardrails/check   ── verdict=reject ──► 400
                 │ 2. OCR engine (paddle / mock)
                 │ 3. POST structuring:8032 /v1/structuring/structure
                 │ 4. POST scoring:8033 /v1/scoring/score
                 └─► 200 {data: {nomor_npwp, nama, nama_badan: {value, confidence}}, guardrails: <skor>}
```

Endpoint sinkron tiap tahap (`/v1/structuring/structure`, `/v1/scoring/score`) melakukan kerja yang sama dengan `/jobs`-nya, tanpa job/callback. Error dari tahap lain diteruskan: 4xx dari service lain dipertahankan status dan pesannya (mis. 400 `No text lines to structure`), 5xx/tidak terjangkau menjadi 500/503/504 dengan nama service-nya (`structuring service is unavailable`).

Di dalam tiap service: `api/v1/*.py` (controller, `ServiceError` → `HTTPException`) → `services/*_service.py` (logika, tidak tahu HTTP) → `models/*.py` (pembungkus model, dipilih lewat env) / `repositories/`. Handler envelope, `X-Request-ID`, 401, dan 422 (`errors: "VALIDATION_ERROR"`) datang dari `ocr_common.app.create_app`.

## Menjalankan Secara Lokal

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt   # Windows; Linux/macOS: .venv/bin/pip
# memasang libs/ocr_common (editable) + requirements keempat service + tooling

for s in guardrails ekstraksi structuring scoring; do cp services/$s/.env.example services/$s/.env; done
# isi API_KEY di tiap .env (boleh sama semua; ekstraksi memakai API_KEY-nya sendiri ke service lain kalau *_API_KEY kosong)

make run-guardrails     # terminal 1, :8031
make run-structuring    # terminal 2, :8032
make run-scoring        # terminal 3, :8033
make run-ekstraksi      # terminal 4, :8030  (default .env menunjuk 127.0.0.1:8031/8032/8033)
make smoke              # rantai extract-ocr lewat keempatnya
```

Swagger UI tiap service di `http://127.0.0.1:<port>/docs`. Untuk mengerjakan satu service saja cukup jalankan service itu; endpoint per-app-nya tidak butuh service lain. Hanya `extract-ocr` di ekstraksi yang butuh ketiganya.

Editor: tiap service punya `pyproject.toml` yang menunjuk root import ke folder service dan ke `libs/ocr_common` (`[tool.ty.environment]`, `[tool.pyright]`); buka repo dari root dan reload window sekali setelah `pip install`.

## Menjalankan dengan Docker

Empat image, dibangun dari **root repo** karena tiap image butuh `libs/ocr_common`:

```bash
make build                      # docker compose build  (4 image: nilam-ocr-{guardrails,ekstraksi,structuring,scoring})
make up                         # build + jalankan keempatnya, port hanya di 127.0.0.1
make ps                         # status container
make up-db                      # sama, plus PostgreSQL lokal (jobs/results ketiga tahap)
make smoke                      # memerankan Orkestrasi: guardrails -> jobs -> polling (+ kontrak lama)
make logs-ekstraksi
make down
```

Build satu image secara manual (mis. di CI, satu pipeline per service):

```bash
docker build -f services/scoring/Dockerfile -t nilam-ocr-scoring:1.0.0 .
```

Image guardrails berisi torch CPU (~1 GB) dan bobot model. Deploy terpisah: jalankan tiap image di mana saja, lalu sambungkan rantainya lewat env:

- ekstraksi: `STRUCTURING_SERVICE_URL`; structuring: `SCORING_SERVICE_URL` (+ `*_API_KEY` kalau key-nya berbeda).
- ekstraksi, structuring, scoring: `ORCHESTRATION_URL` (+ `ORCHESTRATION_API_KEY`) untuk callback, dan `DATABASE_URL` yang sama (satu database, schema per service).
- Orkestrator perlu tahu dua alamat: guardrails (`:8031`, sinkron) dan ekstraksi (`:8030`, `/v1/ekstraksi/jobs`).
- Kontrak lama saja: `GUARDRAILS_SERVICE_URL` dan `SCORING_SERVICE_URL` di ekstraksi, dan di orkestrator `OCR_NPWP_SERVICE_URL=http://<host>:8030`, `OCR_NPWP_API_KEY=<API_KEY ekstraksi>`.

Di compose, URL antar service sudah di-override ke nama service (`http://structuring:8032`, dst.).

Uji callback tanpa orkestrator: `SMOKE_CALLBACK_PORT=8039 make smoke` membuat smoke script ikut menerima callback; arahkan ketiga service ke sana dengan `ORCHESTRATION_URL=http://host.docker.internal:8039` (container) atau `http://127.0.0.1:8039` (proses bare). Script memeriksa urutan `OCR → STRUCTURING → SCORING` dan menampilkan hasil akhir.

## Environment Variables

Semua service (`ocr_common.config.BaseServiceSettings`):

| Variable | Wajib? | Default | Keterangan |
|---|---|---|---|
| `API_KEY` | **Ya** | – | Header `X-API-Key` yang harus dikirim pemanggil. `MOCK_API_KEY` juga diterima. Tidak boleh kosong; service gagal start |
| `SERVICE_BASE_URL` | Tidak | – | Nilai `servers` di OpenAPI (`/docs`) |
| `PORT` | Tidak | per service | 8030 / 8031 / 8032 / 8033 |
| `MAX_UPLOAD_BYTES` | Tidak | `5242880` | Berlaku untuk `file` maupun `file_url` |
| `ALLOWED_CONTENT_TYPES` | Tidak | `["image/jpeg","image/jpg","image/png","application/pdf"]` | JSON list |

ekstraksi, structuring, scoring (`ocr_common.config.PipelineSettings`, pipeline async):

| Variable | Wajib? | Default | Keterangan |
|---|---|---|---|
| `DATABASE_URL` | Produksi: ya | – | PostgreSQL untuk `<schema>.jobs` / `<schema>.results` (ekstraksi: juga `ocr_npwp_requests`). Kosong = in-memory: hanya untuk dev satu proses; hilang saat restart dan **tidak idempoten antar replika** |
| `ORCHESTRATION_URL` | Produksi: ya | – | Base URL Orkestrasi untuk callback tahap. Kosong = callback dilewati (dicatat di log) |
| `ORCHESTRATION_CALLBACK_PATH` | Tidak | `/v1/callbacks/stage` | |
| `ORCHESTRATION_API_KEY` | Tidak | = `API_KEY` | `X-API-Key` yang dikirim ke Orkestrasi |
| `ORCHESTRATION_TIMEOUT_SECONDS` | Tidak | `10.0` | |
| `PIPELINE_RETRY_ATTEMPTS` / `PIPELINE_RETRY_DELAY_SECONDS` | Tidak | `3` / `0.5` | Retry callback dan handoff (5xx / tidak terjangkau), backoff ×2 |
| `PIPELINE_DRAIN_TIMEOUT_SECONDS` | Tidak | `30.0` | Saat shutdown, tunggu job yang masih jalan |

ekstraksi:

| Variable | Wajib? | Default | Keterangan |
|---|---|---|---|
| `EKSTRAKSI_BACKEND` | Tidak | `mock` | `mock` atau `paddle` |
| `EKSTRAKSI_OCR_URL` | Jika `paddle` | – | Service PaddleOCR, mis. `http://10.213.128.67:8070` |
| `EKSTRAKSI_OCR_TIMEOUT_SECONDS` | Tidak | `30.0` | |
| `GUARDRAILS_SERVICE_URL` / `STRUCTURING_SERVICE_URL` / `SCORING_SERVICE_URL` | Tidak | `http://127.0.0.1:803x` | Alamat service lain. Pipeline async hanya memakai `STRUCTURING_*`; kontrak lama memakai ketiganya |
| `GUARDRAILS_API_KEY` / `STRUCTURING_API_KEY` / `SCORING_API_KEY` | Tidak | = `API_KEY` | Key service lain, kalau berbeda |
| `*_TIMEOUT_SECONDS` | Tidak | `10.0` | Per service lain |

guardrails:

| Variable | Wajib? | Default | Keterangan |
|---|---|---|---|
| `GUARDRAILS_BACKEND` | Tidak | `mock` | `efficientnet` (model di proses ini), `remote` (service model ML engineer), atau `mock`; `.env.example` mengaktifkan `efficientnet` |
| `GUARDRAILS_MODEL_URL` | Jika `remote` | – | Service model guardrails, mis. `http://localhost:8081` (`POST {url}/v1/predict/json`) |
| `GUARDRAILS_MODEL_API_KEY` | Tidak | – | `X-API-Key` yang dikirim **ke service model** (bukan `API_KEY` service ini) |
| `GUARDRAILS_MODEL_TIMEOUT_SECONDS` | Tidak | `30.0` | |
| `GUARDRAILS_MODEL_PATH` | Jika `efficientnet` | `weights/best_model.pt` | Checkpoint (`model_state_dict`, `class_names`, `image_size`, `reject_threshold`) |
| `GUARDRAILS_DEVICE` | Tidak | `cpu` | Inference CPU saja (image memakai wheel torch CPU); nilai lain jatuh ke cpu dengan peringatan |
| `GUARDRAILS_TORCH_THREADS` | Tidak | bawaan torch | Batas thread torch; samakan dengan limit CPU container |
| `GUARDRAILS_REJECT_THRESHOLD` | Tidak | dari checkpoint (`0.5`) | Backend lokal saja. Halaman reject kalau `proba_reject >= ambang` |
| `GUARDRAILS_DOCUMENT_POLICY` | Tidak | `all` | Backend lokal saja. `all`: accepted hanya kalau semua halaman accepted; `majority`: accepted > reject |
| `GUARDRAILS_PDF_DPI` / `GUARDRAILS_MAX_PAGES` | Tidak | `150` / `20` | Backend lokal saja. Render PDF per halaman |

structuring: `STRUCTURING_BACKEND` (`rule_based`); tahap berikutnya `SCORING_SERVICE_URL` (`http://127.0.0.1:8033`), `SCORING_API_KEY` (= `API_KEY`), `SCORING_TIMEOUT_SECONDS` (`10.0`).

scoring: `SCORING_BACKEND` (`heuristic`), `SCORING_APPROVE_THRESHOLD` (`0.8`), `SCORING_REVIEW_THRESHOLD` (`0.5`).

## Endpoint API

Semua response memakai envelope `ocr-*`: `{status_code, status_desc, message, data, errors, request_id}`; `extract-ocr`/`get-ocr-result` menambah `guardrails` (skor dokumen) sejajar `data`. `request_id` diambil dari form/path, atau dari header `X-Request-ID` untuk endpoint per-app (kalau tidak ada, dibuat `REQ_<uuid>`).

| Service | Method | Path | Body |
|---|---|---|---|
| semua | GET | `/health` | – (tanpa API key; `backends` menunjukkan implementasi aktif dan `storage`: `postgres` / `memory`) |
| guardrails | POST | `/v1/guardrails/check` | form: `request_id` + `file` / `file_url` → `passed`, `reason`, `document` {verdict, confidence, n_pages, n_approve, n_reject}, `pages[]` {page_index, proba_approve, proba_reject, verdict} |
| ekstraksi | POST | `/v1/ekstraksi/jobs` | **202.** form: `request_id`, `document_type` (default `npwp`), `guardrails` (JSON object, `data` dari guardrails/check), + tepat satu dari `file` / `file_url` |
| structuring | POST | `/v1/structuring/jobs` | **202.** JSON `{"request_id", "document_type", "guardrails", "ocr": {"blocks": [{"text", "confidence", ...}], ...}}` |
| scoring | POST | `/v1/scoring/jobs` | **202.** JSON `{"request_id", "document_type", "guardrails", "ocr", "structuring": {"fields": {...}}}` |
| ketiganya | GET | `/v1/<tahap>/jobs/{request_id}` | – → `{request_id, stage, status: PROCESSING\|DONE\|FAILED, result, error_message, created_at, updated_at}`; untuk debug/rekonsiliasi, sumber status resmi tetap Orkestrasi |
| ekstraksi | POST | `/v1/generate-request-id` | kontrak lama. – |
| ekstraksi | POST | `/v1/extract-ocr` | kontrak lama. form: `request_id` + tepat satu dari `file` / `file_url` |
| ekstraksi | GET | `/v1/get-ocr-result/{request_id}` | kontrak lama. – |
| ekstraksi | POST | `/v1/ekstraksi/extract` | form: `file` / `file_url` → blok teks, `confidence`, `bbox`, `page`, `model` |
| structuring | POST | `/v1/structuring/structure` | JSON `{"lines": [{"text", "confidence"}]}` → `fields` |
| scoring | POST | `/v1/scoring/score` | JSON `{"document_type", "fields": {name: {"value", "confidence"}}}` → `score`, `decision` |

**Response `/jobs`** (202): `data: {"request_id", "stage", "status", "duplicate"}`. `status` adalah `PROCESSING` untuk job baru, atau status terkini kalau `duplicate: true`. Validasi isi (file rusak, tidak ada teks, `document_type` tidak didukung) terjadi di background, jadi muncul sebagai callback `FAILED`, bukan 4xx; yang langsung ditolak hanya bentuk request yang salah (400 intake / JSON `guardrails`, 401, 422).

**Response guardrails** (`message: "OK"`). `passed` / `reason` adalah "true/false + alasan" di sequence diagram; orkestrator meneruskan `data` ini apa adanya sebagai form `guardrails` ke `/v1/ekstraksi/jobs`:

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

**Data `extract-ocr`** (sheet "OCR Nilam - Document Type"): `nomor_npwp`, `nama`, `nama_badan`, tiap field `{"value", "confidence"}`, yang tidak ditemukan `{"value": null, "confidence": 0}`.

## Skenario Testing via Nama File

Berlaku selama backend masih mock (guardrails: `GUARDRAILS_BACKEND=mock`; ekstraksi: `EKSTRAKSI_BACKEND=mock`):

| Nama file mengandung | Pipeline async | `extract-ocr` (kontrak lama) | Sumber |
|---|---|---|---|
| `blur` / `invalid` / `notnpwp` | guardrails `passed: false` + `reason`; orkestrator berhenti (422) | 400 `Document rejected by guardrails: 1/1 page(s) rejected (confidence 0.88)` | guardrails (mock) |
| `servererror` | job OCR `FAILED` + callback `{stage: OCR, status: FAILED}` | 500 `Internal server error while processing OCR` | ekstraksi (mock engine) |
| lainnya | ketiga tahap `DONE`, data dummy deterministik dari isi file | 200, data yang sama | |

Kontrak lama: `request_id` sama dua kali → 409; tidak dikenal → 400; tidak ada di `get-ocr-result` → 404. Pipeline async: `request_id` sama dua kali → 202 `duplicate: true`.

## Database

Satu database PostgreSQL, **satu schema per service**, masing-masing dengan dua tabel yang sama bentuknya:

| Schema | Pemilik | `jobs` | `results` |
|---|---|---|---|
| `ocr` | ekstraksi | status tahap OCR per `request_id` | blok teks mentah |
| `structuring` | structuring | status tahap structuring | field bernama |
| `scoring` | scoring | status tahap scoring | laporan skor |

`jobs`: `request_id` (PK), `status` (`PROCESSING` → `DONE` \| `FAILED`), `error_message`, `attempts`, `created_at`, `updated_at`, `ds`. `results`: `request_id` (PK, FK ke `jobs`), `result` JSONB, timestamp, `ds`. Guardrails tidak punya tabel (hasilnya dicatat Orkestrasi di `stage_logs`), dan hasil akhir disimpan Orkestrasi di `requests.final_result` dari callback scoring, bukan di sini. Ekstraksi juga masih punya `ocr_npwp_requests` untuk kontrak lama.

Skema dipasang manual (pgAdmin / psql) dari [services/ekstraksi/db/schema.sql](services/ekstraksi/db/schema.sql), [services/structuring/db/schema.sql](services/structuring/db/schema.sql), dan [services/scoring/db/schema.sql](services/scoring/db/schema.sql); aman dijalankan berkali-kali, aplikasi tidak memigrasi sendiri. Definisi kolomnya harus tetap sama dengan `build_tables()` di `ocr_common/jobs_sql.py`. Akses lewat SQLAlchemy 2.0 async + asyncpg (pola sama dengan `ocr-orchestration`). PostgreSQL lokal: `make up-db` (volume lama dari sebelum pipeline async tidak punya schema baru; jalankan ketiga file itu manual atau `down -v`).

Tanpa `DATABASE_URL` semuanya in-memory: cukup untuk dev dan test, tapi klaim job hanya idempoten di dalam satu proses.

## Mengganti Mock dengan Model Asli

Model ML di-deploy ML engineer sebagai service HTTP; repo ini membungkusnya di `services/<nama>/src/models/`. Semua klien memakai `ocr_common.remote.RemoteModelClient`: koneksi persisten, dan kegagalan dipetakan seragam (tidak bisa connect → 503, tidak menjawab dalam timeout → 504, jawaban ≥ 400 → 500 dengan detail, body bukan JSON → 500).

**Guardrails, backend `efficientnet` (sudah ada)**: checkpoint torchvision EfficientNet-B0 dari ML engineer, dimuat di proses service saat startup (CPU, ~3 detik) dan dijalankan per halaman (~70 ms/halaman di CPU). Checkpoint membawa metadata `class_names ['accepted','reject']`, `image_size 224`, `reject_threshold 0.5`, `val_macro_f1 0.879`. Praproses: halaman → RGB → resize `224×224` → normalisasi ImageNet; PDF dirender per halaman oleh PyMuPDF. Taruh bobot di `services/guardrails/weights/best_model.pt` (file `best_model.pt.zip` dari ML engineer adalah arsip `torch.save`; cukup ganti namanya) dan set `GUARDRAILS_BACKEND=efficientnet`. Saat `docker build`, bobot itu **ikut ke dalam image** (`COPY services/guardrails/weights`; build gagal dengan pesan jelas kalau filenya tidak ada di build context), sehingga container tidak butuh volume. Di CI, unduh bobot dulu dengan `make weights` ([scripts/fetch_weights.py](scripts/fetch_weights.py)); sumbernya dari env supaya pilihan penyimpanan tinggal mengganti nilai:

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

Aktifkan dengan `GUARDRAILS_BACKEND=remote` + `GUARDRAILS_MODEL_URL` (+ `GUARDRAILS_MODEL_API_KEY`). Kontrak **kita** ke orkestrator (`POST /v1/guardrails/check`) tidak berubah; service ini tetap menambah yang tidak dimiliki service model: `request_id`, `file_url` (unduh dari MinIO), validasi tipe/ukuran sebelum berkas dikirim, envelope, dan `passed`/`reason`. Bedanya dengan `efficientnet`: berkas dikirim **utuh** (PDF dirender di service model) dan `data.document` / `data.pages` diteruskan **apa adanya**, jadi ambang reject, kebijakan dokumen, dan render PDF adalah keputusan service model; `GUARDRAILS_REJECT_THRESHOLD`, `_DOCUMENT_POLICY`, `_PDF_DPI`, `_MAX_PAGES` tidak berlaku. Hanya field kontrak yang diambil (field tambahan dibuang); bentuk lain, termasuk `verdict` di luar `accepted`/`reject`, menjadi 500 `guardrails model returned an unexpected response` alih-alih vonis tebakan. Status ≥ 400 dari service model (termasuk 401 karena key salah) menjadi 500 dengan detailnya, bukan diteruskan: 401 itu salah konfigurasi kita, bukan salah pemanggil kita. Image dengan backend ini tidak butuh torch maupun bobot, tapi Dockerfile sekarang masih memasang keduanya. Test live: `cd services/guardrails && GUARDRAILS_MODEL_URL=http://localhost:8081 GUARDRAILS_MODEL_API_KEY=dummy-key python -m pytest tests/test_guardrails_live.py`.

**Ekstraksi, backend `paddle` (sudah ada)**: klien ke PaddleOCR PP-OCRv6 (`POST {EKSTRAKSI_OCR_URL}/ocr`, multipart `file`). Pemetaan response `pages[].texts[]{text, score, poly}` → `blocks[]{text, confidence, bbox, page}`, `models.detection+recognition` → `model`. File yang tidak terbaca (`num_pages: 0`) → `blocks: []`, di pipeline menjadi 400 `No text lines to structure` dari structuring. Aktifkan dengan `EKSTRAKSI_BACKEND=paddle` + `EKSTRAKSI_OCR_URL`.

**Menambah backend lain** (structuring dan scoring menunggu dokumentasi modelnya):

1. Tambahkan kelas di `services/<nama>/src/models/<app>.py` dengan method yang sama dengan mock-nya; untuk model HTTP terima `RemoteModelClient` lewat constructor.
2. Daftarkan di registry modul yang sama (`<X>_BACKENDS["nama"] = lambda settings: ...`) dan tambahkan field URL/timeout di `src/core/config.py`.
3. Set `<APP>_BACKEND=nama` di `.env`. Tidak ada controller, service, atau skema yang berubah.

Kontrak method per app: guardrails lokal `classify(filename, pages: list[PIL.Image]) -> [(proba_approve, proba_reject), ...]` (+ atribut `reject_threshold`), atau guardrails remote `async check_document(filename, content, content_type) -> {"document", "pages"}` (service memilih dari ada-tidaknya `check_document`); ekstraksi `async extract(filename, content, content_type) -> {"blocks", "model"}`; structuring `structure(lines) -> {field: {"value", "confidence", "source"}}` untuk semua `NPWP_FIELDS`; scoring `score(fields) -> {"score", "field_scores", "reasons"}` + atribut `supported_document_types`.

## Menambah Service Baru

Salin salah satu folder `services/<nama>` (yang paling kecil: `scoring`), ganti nama dan port, lalu tambahkan ke `docker-compose.yml`, `requirements-dev.txt`, dan `SERVICES` di `Makefile`. Kalau menjadi tahap baru pipeline async: tambah konstanta `STAGE_*` di `ocr_common/jobs.py`, schema `db/schema.sql` sendiri, `core/pipeline.py` + `services/job_service.py` + `api/v1/jobs.py` (salin dari structuring), lalu arahkan `get_next_stage()` tahap sebelumnya ke `/v1/<nama>/jobs`. Kalau ikut kontrak lama `extract-ocr`, tambahkan klien di `services/ekstraksi/src/clients/stages.py` dan pemanggilannya di `services/ocr_service.py`.

## Testing, Lint & openapi.yaml

```bash
make test          # lib + keempat service (pytest dijalankan di folder masing-masing)
make test-scoring  # satu service
make lint          # ruff seluruh repo
make typecheck     # ty per package
make openapi       # tulis ulang openapi.yaml tiap service dari kodenya
```

Test unit tiap service memakai stub untuk model, jadi tidak butuh jaringan: pipeline async (`tests/test_jobs.py`) mengganti Orkestrasi dan tahap berikutnya dengan perekam dari `ocr_common.testing` (`RecordingCallback`, `RecordingNextStage`) dan menunggu job background dengan `wait_for_job`; kontrak lama di ekstraksi memakai `tests/fakes.py`. Mesin pipelinenya sendiri (`libs/ocr_common/tests/test_jobs.py`) diuji terhadap repository in-memory **dan** SQL (SQLite sungguhan: `ON CONFLICT`, upsert, join). Rantai HTTP sungguhan antar container diuji `make smoke` (`scripts/smoke_e2e.py`). Test live ke PaddleOCR: `cd services/ekstraksi && EKSTRAKSI_OCR_URL=http://10.213.128.67:8070 python -m pytest tests/test_ekstraksi_live.py`.

`openapi.yaml` tiap service adalah turunan kode, dijaga `tests/test_openapi.py`, dan format dump-nya sama dengan `scripts/check_openapi.py` di monorepo orkestrasi.

## Keterbatasan & Langkah Berikutnya

- Ekstraksi (`paddle`, service PaddleOCR) dan guardrails (`efficientnet`, checkpoint lokal, atau `remote`, service model ML engineer) sudah memakai model sungguhan. Backend `remote` baru diuji terhadap kontrak tertulisnya dan stand-in yang mengikutinya, belum terhadap service model yang asli (`tests/test_guardrails_live.py`); structuring dan scoring masih rule-based / heuristik sampai modelnya tersedia. Model guardrails belum divalidasi dengan foto NPWP asli (lihat catatan di atas).
- **Kontrak callback belum disepakati dengan tim orkestrasi.** Path (`/v1/callbacks/stage`), bentuk body, dan bentuk hasil akhir di [Alur Request](#alur-request) adalah usulan dari repo ini; sisi orkestrasi (endpoint callback, `orkestrasi.requests` / `stage_logs`, 422/202, polling) dikerjakan di repo `nilam-ocr-orchestration`. Sampai itu ada, uji callback dengan `SMOKE_CALLBACK_PORT`.
- **Job hidup di memori proses.** Job jalan sebagai `asyncio` task; kalau proses mati di tengah job (deploy, OOM), baris `jobs` tertinggal `PROCESSING` dan tidak ada callback. Shutdown normal menunggu job selesai (`PIPELINE_DRAIN_TIMEOUT_SECONDS`), tapi crash tidak. Orkestrasi perlu timeout per tahap; kalau volume naik, ganti `BackgroundRunner` di `ocr_common/jobs.py` dengan antrean yang tahan restart (mis. worker yang mengambil `jobs.status=PROCESSING` yang basi) tanpa mengubah service.
- **Callback yang gagal setelah retry hanya dicatat di log**; tidak ada outbox. Orkestrasi bisa merekonsiliasi lewat `GET /v1/<tahap>/jobs/{request_id}`.
- **Payload membesar di tiap handoff** (guardrails + OCR + structuring ikut dibawa, sesuai diagram). Untuk PDF banyak halaman, pertimbangkan tahap berikutnya membaca `results` tahap sebelumnya dari DB alih-alih menerimanya di payload.
- Kontrak lama `extract-ocr` (sinkron) dan tabel `ocr_npwp_requests` dipertahankan sampai orkestrator pindah ke alur async; setelah itu `api/v1/ocr.py`, `services/ocr_service.py`, `clients/stages.py`, dan `repositories/` di ekstraksi bisa dihapus.
