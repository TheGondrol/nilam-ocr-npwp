# nilam-ocr-npwp — OCR API untuk NPWP

Service OCR untuk dokumen NPWP (kartu identitas pajak) Indonesia. Pengganti mock `ocr-npwp` di `nilam-ocr-orchestration`: kontrak, envelope, auth, dan struktur kodenya sama, tapi di balik `extract-ocr` ada pipeline sungguhan yang terdiri dari **empat app**:

| App | Endpoint langsung | Peran di pipeline |
|---|---|---|
| **Guardrails** | `POST /v1/guardrails/check` | kualitas gambar + jenis dokumen; gagal → `extract-ocr` menjawab 400 |
| **Ekstraksi** | `POST /v1/ekstraksi/extract` | OCR mentah: blok teks + confidence + bounding box |
| **Structuring** | `POST /v1/structuring/structure` | baris teks → `nomor_npwp`, `nama`, `nama_badan` dengan confidence per field |
| **Scoring** | `POST /v1/scoring/score` | skor dokumen 0..1 → dikirim sebagai `guardrails` di envelope; plus keputusan `approve`/`review`/`reject` |

Endpoint per-app dipakai untuk debugging dan evaluasi model per tahap. Orchestrator hanya memanggil kontrak OCR (`generate-request-id` → `extract-ocr` → `get-ocr-result`).

## Daftar Isi

- [Struktur Folder](#struktur-folder)
- [Environment Variables](#environment-variables)
- [Menjalankan Secara Lokal](#menjalankan-secara-lokal)
- [Menjalankan dengan Docker](#menjalankan-dengan-docker)
- [Database (opsional)](#database-opsional)
- [Endpoint API](#endpoint-api)
- [Skenario Testing via Nama File](#skenario-testing-via-nama-file)
- [Contoh Pemakaian Lengkap (curl)](#contoh-pemakaian-lengkap-curl)
- [Mengganti Mock dengan Model Asli](#mengganti-mock-dengan-model-asli)
- [Menambah App Baru](#menambah-app-baru)
- [Testing, Lint & openapi.yaml](#testing-lint--openapiyaml)
- [Keterbatasan](#keterbatasan)

## Struktur Folder

Mengikuti pola **Layered by Type (Technical Layering)** seperti `ocr-ktp`/`ocr-npwp`: Controller → Service → Repository. Tambahannya satu folder, `models/`, untuk pembungkus model ML (perannya sama dengan `clients/` di `ocr-orchestration`: dependensi luar yang dipanggil service).

```
nilam-ocr-npwp/
├── src/
│   ├── main.py                    # FastAPI app, exception handler (envelope), include router
│   ├── api/v1/                    # Controller: parsing HTTP, panggil service, ServiceError -> HTTPException
│   │   ├── health.py
│   │   ├── ocr.py                 # kontrak orchestrator: generate-request-id, extract-ocr, get-ocr-result
│   │   ├── intake.py              # terima `file` ATAU `file_url` (dipakai ocr, guardrails, ekstraksi)
│   │   ├── guardrails.py / ekstraksi.py / structuring.py / scoring.py
│   ├── services/                  # Business logic: tidak tahu HTTP, tidak tahu model apa yang dipakai
│   │   ├── ocr_service.py         # pipeline: guardrails -> ekstraksi -> structuring -> scoring
│   │   ├── guardrails_service.py / ekstraksi_service.py / structuring_service.py / scoring_service.py
│   │   └── image_validation.py    # validasi upload bersama (tipe, kosong, ukuran)
│   ├── repositories/
│   │   └── request_repository.py  # status request_id: in-memory, atau PostgreSQL via SQLAlchemy kalau DATABASE_URL diisi
│   ├── models/                    # Pembungkus model ML (mock / rule-based / heuristik, nanti ONNX/LLM)
│   │   ├── guardrails.py          # MockImageQualityAssessor, MockDocumentClassifier
│   │   ├── ekstraksi.py           # MockOcrEngine
│   │   ├── structuring.py         # RuleBasedNpwpStructurer
│   │   ├── scoring.py             # HeuristicNpwpScorer
│   │   └── registry.py            # pilih implementasi dari <APP>_BACKEND
│   ├── core/                      # config, security, envelope, errors, fetch_url, request_id, database (engine SQLAlchemy)
│   └── schemas/                   # pydantic: ocr (kontrak), per app, errors (+ helper contoh error), health
├── scripts/export_openapi.py      # tulis ulang openapi.yaml dari kode (make openapi)
├── db/schema.sql                  # skema PostgreSQL, dipasang manual (make db-schema)
├── tests/
├── weights/                       # bobot model (di-gitignore, di-mount ke container)
├── openapi.yaml                   # turunan kode, dijaga tests/test_openapi.py
├── requirements.txt / requirements-dev.txt
├── pyproject.toml                 # ruff + pytest + ty
├── Dockerfile / docker-compose.yml / docker-compose.db.yml (overlay postgres lokal) / .dockerignore
├── .env / .env.example / .gitignore / Makefile
```

Alur `POST /v1/extract-ocr`:

1. `api/v1/ocr.py` membaca form (`request_id` + `file`/`file_url` lewat `intake.py`), membuat `OcrService` dari `get_request_repository()` (in-memory atau SQL, tergantung `DATABASE_URL`) dan keempat service.
2. `services/ocr_service.py` mengecek `request_id` (400 tidak dikenal, 409 sudah diproses), lalu menjalankan pipeline: guardrails gagal → 400 dengan pesan yang sama seperti mock (`Image quality too low...` / `...not recognized as an NPWP`); sukses → field `{value, confidence}` + skor dokumen.
3. Controller membungkusnya: `data` berisi `NpwpFields`, `guardrails` (skor dokumen) sejajar `data` di envelope. `get-ocr-result` mengembalikan bentuk yang sama.

Error di service dilempar sebagai `ServiceError(status_code, message)` (`core/errors.py`), controller mengubahnya jadi `HTTPException`, handler di `main.py` membungkusnya dengan envelope. Validasi form/body yang gagal (422) memakai `errors: "VALIDATION_ERROR"` dan pesan `lokasi: masalah`, sama seperti `ocr-*`.

## Environment Variables

| Variable | Wajib? | Default | Keterangan |
|---|---|---|---|
| `API_KEY` | **Ya** | – | Key yang harus dikirim client (orchestrator) lewat header `X-API-Key`. `MOCK_API_KEY` juga diterima supaya `.env`/script dari mock `ocr-npwp` bisa dipakai apa adanya. Tidak boleh kosong; service gagal start |
| `SERVICE_BASE_URL` | Tidak | – | Nilai `servers` di OpenAPI spec (`/docs`) |
| `DATABASE_URL` | Tidak | – | Diisi → status `request_id` disimpan di PostgreSQL (lihat [Database](#database-opsional)). Kosong → in-memory seperti mock. Format `postgresql+asyncpg://user:pass@host:5432/db` |
| `PORT` | Tidak | `8030` | Port container (slot `ocr-npwp` di `DEPLOYMENT.md`, dipanggil `OCR_NPWP_SERVICE_URL`) |
| `GUARDRAILS_BACKEND` | Tidak | `mock` | Implementasi di `src/models/guardrails.py` |
| `EKSTRAKSI_BACKEND` | Tidak | `mock` | Implementasi di `src/models/ekstraksi.py` |
| `STRUCTURING_BACKEND` | Tidak | `rule_based` | Implementasi di `src/models/structuring.py` |
| `SCORING_BACKEND` | Tidak | `heuristic` | Implementasi di `src/models/scoring.py` |
| `WEIGHTS_DIR` | Tidak | `weights` | Folder bobot model, dipakai implementasi asli |
| `GUARDRAILS_MIN_QUALITY_SCORE` | Tidak | `0.5` | Ambang lolos check `image_quality` |
| `GUARDRAILS_EXPECTED_DOCUMENT_TYPE` | Tidak | `npwp` | Label yang harus dikenali classifier |
| `GUARDRAILS_MIN_CLASSIFICATION_CONFIDENCE` | Tidak | `0.7` | Ambang lolos check `document_type` |
| `SCORING_APPROVE_THRESHOLD` | Tidak | `0.8` | skor ≥ ini → `approve` (hanya dipakai `/v1/scoring/score`; orchestrator punya ambang sendiri per role) |
| `SCORING_REVIEW_THRESHOLD` | Tidak | `0.5` | skor ≥ ini → `review`, di bawahnya `reject` |
| `MAX_UPLOAD_BYTES` | Tidak | `5242880` | 5 MB, berlaku untuk `file` maupun `file_url` |
| `ALLOWED_CONTENT_TYPES` | Tidak | `["image/jpeg","image/jpg","image/png"]` | Format JSON list |

## Menjalankan Secara Lokal

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt     # Windows; Linux/macOS: .venv/bin/pip
cp .env.example .env                                  # lalu isi API_KEY
.venv/Scripts/uvicorn src.main:app --port 8030
```

Swagger UI: **http://127.0.0.1:8030/docs**. Dengan `make`: `make dev`, `make run`, `make test`, `make lint`, `make typecheck`, `make openapi`.

Modul di-import sebagai `src.<...>` (sama seperti `ocr-*`). Supaya editor tidak menebak `src/` sebagai import root, `pyproject.toml` sudah menyetel `[tool.ty.environment]` dan `[tool.pyright]`; setelah clone, reload window VS Code sekali agar konfigurasi terbaca.

## Menjalankan dengan Docker

```bash
cp .env.example .env    # isi API_KEY
docker compose up -d --build
curl http://127.0.0.1:8030/health
```

Port hanya dibuka ke `127.0.0.1` (sama seperti `ocr-*`); orchestrator di VM yang sama memanggil lewat `host.docker.internal:8030`.

## Database (opsional)

Tanpa `DATABASE_URL`, status `request_id` (pending → completed/failed) disimpan in-memory, persis seperti mock `ocr-*`: cukup untuk satu container, hilang saat restart. Untuk lebih dari satu replika atau status yang tahan restart, isi `DATABASE_URL` dan status disimpan di PostgreSQL lewat **SQLAlchemy 2.0 async + asyncpg**, versi dan pola yang sama dengan `ocr-orchestration` (`core/database.py` untuk engine/session, model `Mapped`/`mapped_column` di dalam `repositories/request_repository.py`, cek `SELECT 1` saat startup dan gagal keras kalau database tidak terjangkau).

Skema **tidak** dibuat aplikasi; pasang manual dari [db/schema.sql](db/schema.sql), sama seperti `ocr-nilam-db` (aman diulang):

```bash
psql "postgresql://postgres:changeme@localhost:5432/bribrain_ocr_nilam" -f db/schema.sql
# atau: make db-schema   (DATABASE_URL_PSQL=... untuk menunjuk database lain)
```

Satu tabel, `ocr_npwp_requests`: satu baris per `request_id`, diperbarui di tempat mengikuti siklusnya. Kolom `result` (JSONB) menyimpan field hasil OCR, `guardrails` skor dokumen, `ds` (YYYYMMDD UTC) partisi harian untuk penarikan batch ke Big Data, konvensi yang sama dengan tabel `orchestration_*`.

PostgreSQL lokal untuk mencoba (overlay compose, skema dipasang otomatis saat volume pertama dibuat):

```bash
make docker-up-db      # docker compose -f docker-compose.yml -f docker-compose.db.yml up -d --build
curl http://127.0.0.1:8030/health   # backends.storage: "postgres"
```

Memilih implementasi: `get_request_repository()` di `repositories/request_repository.py` mengembalikan `SqlRequestRepository` kalau `DATABASE_URL` diisi, kalau tidak `InMemoryRequestRepository`. Keduanya punya method yang sama (`create`/`get`/`update`) dan diuji dengan test yang sama; yang SQL dijalankan terhadap SQLite (aiosqlite) sungguhan di `tests/test_request_repository.py`.

## Endpoint API

Kontrak yang dipanggil `ocr-orchestration` (identik dengan mock `ocr-npwp`):

| Method | Path | Auth | Keterangan |
|---|---|---|---|
| GET | `/health` | Tidak | Cek service hidup + backend aktif per app + `storage` (`memory`/`postgres`) |
| POST | `/v1/generate-request-id` | Ya | Buat `request_id` baru |
| POST | `/v1/extract-ocr` | Ya | form-data: `request_id`, plus tepat satu dari `file` atau `file_url`. Dengan `file_url`, service ini yang mengunduh objeknya (mis. presigned MinIO GET); host di URL itu harus terjangkau dari container ini |
| GET | `/v1/get-ocr-result/{request_id}` | Ya | Cek status & hasil |

**Data yang dihasilkan** (mengikuti sheet "OCR Nilam - Document Type"): `nomor_npwp`, `nama` (wajib pajak orang pribadi), `nama_badan` (badan usaha). Tiap field berbentuk `{"value": ..., "confidence": 0..1}`; field yang tidak ditemukan bernilai `{"value": null, "confidence": 0}`. `guardrails` (skor dokumen 0..1 dari app scoring) ada di level envelope, sejajar `data`.

```json
{
  "status_code": 200, "status_desc": "OK", "message": "Success",
  "data": {
    "nomor_npwp": {"value": "12.345.678.9-012.345", "confidence": 0.96},
    "nama": {"value": "BUDI SANTOSO", "confidence": 0.95},
    "nama_badan": {"value": "PT CIPTA KARYA MANDIRI", "confidence": 0.93}
  },
  "errors": null, "request_id": "OCR_...", "guardrails": 0.95
}
```

Endpoint per-app (untuk debugging/evaluasi; `request_id` diambil dari header `X-Request-ID` kalau dikirim, kalau tidak dibuat `REQ_<uuid>`):

| Method | Path | Body |
|---|---|---|
| POST | `/v1/guardrails/check` | form-data: `file` atau `file_url`. Selalu 200; lihat `data.passed` |
| POST | `/v1/ekstraksi/extract` | form-data: `file` atau `file_url` |
| POST | `/v1/structuring/structure` | JSON `{"lines": [{"text": "...", "confidence": 0.9}]}` |
| POST | `/v1/scoring/score` | JSON `{"document_type": "npwp", "fields": {"nama": {"value": "...", "confidence": 0.9}}}` |

## Skenario Testing via Nama File

Berlaku selama backend masih `mock` (sama dengan mock `ocr-npwp`, jadi Postman collection orchestrator tetap jalan):

| Nama file mengandung | `extract-ocr` | Sumber |
|---|---|---|
| `blur` atau `invalid` | `400` — `Image quality too low, NPWP could not be read` | mock quality assessor (guardrails) |
| `notnpwp` | `400` — `Uploaded image is not recognized as an NPWP` | mock classifier (guardrails) |
| `servererror` | `500` — `Internal server error while processing OCR` | mock OCR engine (ekstraksi) |
| (nama lain) | `200` — data NPWP dummy, deterministik dari isi file | |

`request_id` yang sama disubmit dua kali → `409`. `request_id` tidak dikenal → `400`. `request_id` tidak ditemukan di `get-ocr-result` → `404`.

## Contoh Pemakaian Lengkap (curl)

```bash
API_KEY="isi-dengan-API_KEY-dari-.env"
BASE="http://127.0.0.1:8030"

REQUEST_ID=$(curl -s -X POST "$BASE/v1/generate-request-id" \
  -H "X-API-Key: $API_KEY" | python -c "import json,sys; print(json.load(sys.stdin)['data']['request_id'])")

curl -s -X POST "$BASE/v1/extract-ocr" \
  -H "X-API-Key: $API_KEY" \
  -F "request_id=$REQUEST_ID" \
  -F "file=@/path/ke/foto-npwp.jpg;type=image/jpeg"

# atau lewat URL (service ini yang mengunduh):
# -F "request_id=$REQUEST_ID" -F "file_url=https://minio.internal/bucket/npwp.jpg?X-Amz-..."

curl -s "$BASE/v1/get-ocr-result/$REQUEST_ID" -H "X-API-Key: $API_KEY"
```

Menjalankan pipeline per tahap:

```bash
curl -s -X POST "$BASE/v1/guardrails/check" -H "X-API-Key: $API_KEY" -F "file=@npwp.jpg;type=image/jpeg"
curl -s -X POST "$BASE/v1/ekstraksi/extract" -H "X-API-Key: $API_KEY" -F "file=@npwp.jpg;type=image/jpeg" > ocr.json
python -c "import json; d=json.load(open('ocr.json'))['data']; print(json.dumps({'lines':[{'text':b['text'],'confidence':b['confidence']} for b in d['blocks']]}))" > lines.json
curl -s -X POST "$BASE/v1/structuring/structure" -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d @lines.json > structured.json
python -c "import json; d=json.load(open('structured.json'))['data']; print(json.dumps({'document_type':d['document_type'],'fields':{k:{'value':v['value'],'confidence':v['confidence']} for k,v in d['fields'].items()}}))" > fields.json
curl -s -X POST "$BASE/v1/scoring/score" -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d @fields.json
```

## Mengganti Mock dengan Model Asli

Contoh: OCR engine ONNX menggantikan `MockOcrEngine`. **Tidak ada** service, controller, atau skema yang perlu diubah.

1. Tambahkan dependency ML ke `requirements.txt` (mis. `onnxruntime`, `numpy`, `Pillow`), pin versinya.
2. Tambahkan kelas baru di `src/models/ekstraksi.py` dengan method yang sama dengan mock:

   ```python
   class OnnxOcrEngine:
       name = "onnx"

       def __init__(self, weights_dir: str):
           self._session = onnxruntime.InferenceSession(f"{weights_dir}/ocr.onnx")  # load sekali (lru_cache)

       def extract(self, filename: str, content: bytes) -> list[dict]:
           ...  # decode content -> tensor -> session.run -> [{"text", "confidence", "bbox"}]
   ```

3. Daftarkan di registry modul yang sama:

   ```python
   OCR_BACKENDS = {
       "mock": lambda settings: MockOcrEngine(),
       "onnx": lambda settings: OnnxOcrEngine(settings.weights_dir),
   }
   ```

4. Set `EKSTRAKSI_BACKEND=onnx` di `.env`, taruh bobot di `weights/`. Selesai.

Kontrak method per app:

| Modul | Method | Return |
|---|---|---|
| `models/guardrails.py` | `assess(filename, content)` | `{"score": 0..1, "notes": [str]}` |
| | `classify(filename, content)` | `{"label": str, "confidence": 0..1}` |
| `models/ekstraksi.py` | `extract(filename, content)` | `[{"text", "confidence", "bbox": {x1,y1,x2,y2} \| None}]` |
| `models/structuring.py` | `structure(lines)` | `{field: {"value", "confidence", "source"}}` untuk semua field di `NPWP_FIELDS` |
| `models/scoring.py` | `score(fields)` + atribut `supported_document_types` | `{"score", "field_scores": [{"name","score","issues"}], "reasons": [str]}` |

Kalau implementasi gagal saat inference, lempar `ServiceError(500, "...")` dari `src/core/errors.py` supaya client mendapat envelope 500 yang rapi.

## Menambah App Baru

1. `src/models/<app>.py`: implementasi + registry + `get_<app>_model()`.
2. `src/services/<app>_service.py`: business logic, menerima model lewat constructor.
3. `src/schemas/<app>.py`: request/response.
4. `src/api/v1/<app>.py`: router + `get_<app>_service()`.
5. `src/main.py`: `include_router` + tag OpenAPI. `src/core/config.py`: field `<app>_backend`. Kalau ikut pipeline, panggil dari `services/ocr_service.py`.
6. `make openapi`.

## Testing, Lint & openapi.yaml

```bash
make test       # pytest: unit (service + stub model), HTTP per app, kontrak OCR, rantai pipeline, spec up-to-date
make lint       # ruff
make typecheck  # ty
make openapi    # tulis ulang openapi.yaml dari kode; wajib setelah mengubah route/schema
```

`openapi.yaml` adalah turunan kode, bukan sumber (sama seperti `ocr-orchestration`): `tests/test_openapi.py` gagal kalau ketinggalan. Format dump-nya sama dengan `scripts/check_openapi.py` di `nilam-ocr-orchestration`, jadi kalau repo ini ditaruh di monorepo itu sebagai `ocr-npwp`, `make check-openapi` di sana membaca spec ini. Catatan: skrip itu juga membaca penanda `if "notnpwp" in name:` dari `services/ocr_service.py` milik mock; di sini skenario itu ada di `models/guardrails.py`, jadi satu pemeriksaan tersebut akan mengeluh. Perilakunya sendiri sama.

Test HTTP memakai `src.main:app` langsung dengan `API_KEY` di-set lewat env di `tests/conftest.py`.

## Keterbatasan

- Semua implementasi di `src/models/` masih **mock / rule-based / heuristik**, belum ada model ML sungguhan. Lihat [Mengganti Mock dengan Model Asli](#mengganti-mock-dengan-model-asli).
- Tanpa `DATABASE_URL`, status `request_id` in-memory (sama dengan `ocr-*`), cocok untuk satu container. Untuk lebih dari satu replika, isi `DATABASE_URL` (lihat [Database](#database-opsional)).
- `extract-ocr` sinkron: model dimuat dan dijalankan di proses HTTP. Orchestrator sudah punya jalur async (202 + worker) di sisinya; kalau inference sangat lama, naikkan `OCR_NPWP_TIMEOUT_SECONDS` di `.env` orchestrator.
