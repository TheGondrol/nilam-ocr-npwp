# OCR Postprocess API

FastAPI service for post-processing OCR text extraction from Indonesian KTP (ID Card). Extracts structured fields (NIK, nama, alamat, etc.) from raw OCR output using fuzzy matching and rule-based text processing.

## Features

- **KTP Field Extraction**: Extracts 12 fields (NIK, nama, tempat lahir, tanggal lahir, jenis kelamin, alamat, RT, RW, kel/desa, kecamatan, agama, status perkawinan)
- **Fuzzy Matching**: RapidFuzz-based robust text matching with configurable thresholds
- **OCR Error Correction**: Character-level corrections for common OCR misreads
- **Database Logging**: Async PostgreSQL logging for every request
- **Health Checks**: Liveness, readiness, and full health endpoints (K8s-ready)
- **API Key Authentication**: Prediction endpoint requires `X-API-Key` header
- **Request ID Tracking**: Middleware injects/propagates `x-request-id` headers
- **E2E Testing**: End-to-end tests run during Docker build against the real service
- **Type-Safe Configuration**: YAML-based configuration management

## Project Structure

```
ms-bribrain-ocr-ktp-postprocessor/
├── src/
│   ├── core/                            # Core functionality
│   │   ├── config.py                    # Configuration management (singleton)
│   │   ├── logging.py                   # Logging setup with request_id context
│   │   └── validation.py               # Startup configuration validation
│   ├── api/                             # API layer
│   │   ├── dependencies.py             # API key verification
│   │   └── routes.py                   # API endpoints
│   ├── middleware/
│   │   └── add_requestid.py            # Request ID injection middleware
│   ├── schemas/
│   │   ├── api_schema.py               # Pydantic request/response models
│   │   └── database_schema.py          # SQLAlchemy ORM models
│   ├── services/                        # Business logic
│   │   ├── ocr_processor.py            # Main OCR processing pipeline
│   │   ├── database_service.py         # Async database connection & logging
│   │   └── field_matchers/             # Field-specific matchers
│   │       ├── nik.py
│   │       ├── nama.py
│   │       ├── agama.py
│   │       ├── jenis_kelamin.py
│   │       ├── status_perkawinan.py
│   │       ├── alamat.py
│   │       ├── rtrw.py
│   │       ├── kecamatan.py
│   │       ├── keldesa.py
│   │       └── ttl.py
│   └── main.py                          # FastAPI application entry point
├── tests/
│   ├── unit/                            # Unit tests
│   ├── integration/                     # Integration tests
│   ├── e2e/                             # End-to-end tests
│   │   ├── conftest.py                  # E2E fixtures (in-process or live server)
│   │   ├── test_e2e_postprocess.py     # E2E health & pipeline tests
│   │   ├── test_ktp_extraction_e2e.py  # E2E KTP extraction tests
│   │   ├── test_field_validation_e2e.py # E2E field validation tests
│   │   ├── test_performance_e2e.py     # E2E performance tests
│   │   └── test_real_world_scenarios_e2e.py # Real-world scenario tests
│   └── data/                            # Test data
├── scripts/
│   └── run_e2e_tests.sh                 # E2E test runner (used in Docker build)
├── config.yaml                          # Configuration file
├── Dockerfile                           # Multi-stage build (base → e2e-test → production)
├── pyproject.toml                       # Python project config & dependencies
├── uv.lock                              # Dependency lock file
└── README.md
```

## Configuration

All configuration is managed through `config.yaml`:

```yaml
# Threshold values for fuzzy matching
thresholds:
  partial: 75
  ratio: 80
  confidence: 0.8

# CORS
cors:
  allow_origins:
    - "https://api-bribrain.dev.example.com"
    - "https://api-brispot.dev.example.com"
    - "https://api-brispot-testing.dev.example.com"
    - "https://api-brispot-staging.dev.example.com"
    - "https://api-staging.example.com"

# Logging
logging:
  insert_to_database: true
  level: INFO
  file:
    enabled: true
    path: "logs/ocr_postprocess.log"
    max_bytes: 10485760          # 10MB
    backup_count: 5

# API
api:
  title: "OCR Postprocess API"
  version: "1.0.0"
  host: "0.0.0.0"
  port: 8000

# Database (async PostgreSQL)
database:
  pool_size: 5
  max_overflow: 10
```

## Environment Variables

Loaded from a `.env` file (copy `.env.template` → `.env`). `${VAR}` placeholders in `config.yaml` are resolved at startup; a missing variable resolves to an empty string.

Startup runs a configuration validator (`src/core/validation.py`) that **aborts the process** (`sys.exit(1)`) if any of the variables marked *Required* below is unset or empty.

### Core

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `DATABASE_URL` | Yes | asyncpg PostgreSQL connection string used for request logging into `bribrain_ocr_ktp_postprocessor`. | `postgresql+asyncpg://user:pass@host:5432/bribrain_ocr_ktp` |
| `API_KEY` | Yes | Secret compared (constant-time) against the `X-API-Key` header on `POST /v1/ocr_postprocess`. Requests with a missing/wrong key get `401`. | `change-me-please` |
| `LOG_ENCRYPTION_KEY` | Yes | AES-256-GCM key for encrypting the `payload` and `result` log columns. Base64 of 32 random bytes; the same key must be shared across all services. | `<base64-of-32-random-bytes>` |

### MinIO (validated but unused)

This service does **not** download ML models and contains no MinIO client. The variables below are nonetheless listed as required by the startup validator, so they must be present (any non-empty value) for the process to boot. They are otherwise unused by the application.

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `MINIO_ENDPOINT` | Yes | Required by the startup validator only; not consumed by application code. | `minio.internal:9000` |
| `MINIO_ACCESS_KEY` | Yes | Required by the startup validator only; not consumed by application code. | `placeholder` |
| `MINIO_SECRET_KEY` | Yes | Required by the startup validator only; not consumed by application code. | `placeholder` |
| `MINIO_SECURE` | No | Present in `.env.template`; not validated and not consumed by application code. | `false` |

### Elastic APM (optional)

APM is enabled in `config.yaml` (`elastic_apm.enabled: true`) but only attaches at runtime when `ELASTIC_APM_SERVER_URL` is set. Leave `ELASTIC_APM_SERVER_URL` empty to disable tracing.

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `ELASTIC_APM_SERVER_URL` | No | APM server endpoint. APM activates only when this is non-empty. | `https://apm.internal:8200` |
| `ELASTIC_APM_SERVICE_NAME` | No | Service name reported to APM. Defaults to `ocr-ktp-postprocessor`. | `ocr-ktp-postprocessor` |
| `ELASTIC_APM_ENVIRONMENT` | No | Environment tag reported to APM. Defaults to `dev`. | `production` |
| `ELASTIC_APM_SECRET_TOKEN` | No | Secret token for authenticating to the APM server. | `your-apm-token` |

> Use placeholder values in committed files — never commit real secrets.

## Log encryption (DPIA)

The PII columns of this service's log table `bribrain_ocr_ktp_postprocessor` are encrypted at rest with **AES-256-GCM** (see `src/core/crypto.py`):

- **Encrypted (`BYTEA`):** `payload` and `result` — they hold the parsed/structured KTP fields (NIK, nama, alamat, …). Stored as ciphertext, not queryable as JSON.
- **Plaintext:** `error_message` (`TEXT`) — kept plaintext but **scrubbed of OCR field values** (the error paths store a generic message and log the full detail separately). Also plaintext: `request_id`, `response_code`, `processing_time`, `created_at`.
- **Key:** `LOG_ENCRYPTION_KEY` — base64 of 32 random bytes, the **same key across all encrypting services** (`dgc_ext`, `dgc_irl`, `dgc_oct`, `dgc_pps`). Startup fails if it is missing while DB logging is enabled. Generate with `python -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())"`.
- **Stored format (`BYTEA`):** `version(1) || key_id(1) || nonce(12) || ciphertext+tag` — a fresh random 12-byte nonce per write and a 16-byte authentication tag (tamper-evident; wrong key or altered bytes fail to decrypt).
- **Migration:** run `schema_encryption.sql` (repo root) once per environment to convert these columns to `BYTEA` (it truncates existing rows).

This service only **writes** logs; it does not read them back at runtime.

## Installation

### Using uv (recommended)

```bash
# Install dependencies
uv sync

# Run the service
uv run uvicorn src.main:app --host 0.0.0.0 --port 8090
```

### Using Docker

```bash
# Build the production image (skips e2e tests)
docker build -t ms-bribrain-ocr-ktp-postprocessor .

# Build with e2e tests (requires DB to be reachable)
docker build --target e2e-test --network host --secret id=env,src=.env -t postprocessor-test .

# Run the container
docker run -d \
  --name ms-bribrain-ocr-ktp-postprocessor \
  --env-file .env \
  -p 8090:8090 \
  --restart always \
  ms-bribrain-ocr-ktp-postprocessor

# View logs
docker logs -f ms-bribrain-ocr-ktp-postprocessor
```

## API Endpoints

The postprocess endpoint requires the `X-API-Key` header. Health and root endpoints do not.

### POST /v1/ocr_postprocess

Post-process raw OCR output and extract structured KTP fields.

**Request:**

- Content-Type: `application/json` (**required** — any other content type is rejected with `415 Unsupported Media Type`; a `; charset=utf-8` suffix is tolerated)
- Body: `{"ocr_text": "<JSON string of OCR data>"}`
- Header: `X-API-Key: <your-api-key>` (a missing or invalid key returns `401`)

**Response (200):**

```json
{
  "status_code": 200,
  "status_desc": "OK",
  "message": "Success",
  "data": {
    "ocr_result": {
      "nik": "3174012801900001",
      "nama": "AHMAD FAUZI",
      "tempat_lahir": "JAKARTA",
      "tanggal_lahir": "28 Jan 1990",
      "jenis_kelamin": "LAKI-LAKI",
      "alamat": "JL. GATOT SUBROTO KAV. 53",
      "rt": "005",
      "rw": "008",
      "kel_desa": "KUNINGAN TIMUR",
      "kecamatan": "SETIABUDI",
      "agama": "ISLAM",
      "status_perkawinan": "KAWIN"
    },
    "nik_image_box": [[190, 140], [450, 170]]
  },
  "error_code": null,
  "errors": null,
  "request_id": "abc123"
}
```

### Health Endpoints

#### GET /health/live

Liveness probe. Returns 200 if the process is alive. Use for K8s `livenessProbe`.

```json
{ "status": "alive", "version": "1.0.0" }
```

#### GET /health/ready

Readiness probe. Returns 200 if database is up. Use for K8s `readinessProbe`. Returns 503 if not ready.

```json
{
  "status": "ready",
  "checks": {
    "database": { "status": "up" }
  }
}
```

#### GET /health

Full health check with detailed component info.

- `healthy`: Database up
- `unhealthy`: Database down

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "checks": {
    "database": { "status": "up" }
  }
}
```

### GET /

Service information and endpoint listing.

```json
{
  "message": "OCR Postprocess API",
  "description": "API for post-processing OCR text extraction from KTP",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "liveness": "/health/live",
    "readiness": "/health/ready",
    "predict": "/v1/ocr_postprocess",
    "docs": "/docs"
  }
}
```

## Error Codes

All errors — including early rejections raised before the route body runs (auth,
validation, wrong content type) — are returned in the same response envelope as
success responses, with `error_code` set and `data` empty.

| `error_code` | HTTP Status | Description |
|--------------|-------------|-------------|
| `INVALID_INPUT` | 400 | Invalid JSON or malformed OCR text (`ocr_text` is not a valid JSON list) |
| `UNAUTHORIZED` | 401 | Missing or invalid `X-API-Key` header |
| `NOT_FOUND` | 404 | Unknown path |
| `METHOD_NOT_ALLOWED` | 405 | HTTP method not allowed on the path |
| `UNSUPPORTED_MEDIA_TYPE` | 415 | `Content-Type` of the request is not `application/json` |
| `VALIDATION_ERROR` | 422 | Request body failed schema validation (e.g. missing `ocr_text`) |
| `INTERNAL_ERROR` | 500 | Internal processing error |

## Testing

### Unit Tests

```bash
# Run all unit tests
uv run pytest tests/unit/ -v --tb=short
```

### Integration Tests

```bash
# Run integration tests
uv run pytest tests/integration/ -v --tb=short
```

### End-to-End Tests

E2E tests validate the full processing pipeline. Supports two modes:

```bash
# In-process mode (mocked DB, full API flow)
uv run pytest tests/e2e/ -m e2e -v

# Against a live running service
E2E_BASE_URL=http://127.0.0.1:8090 uv run pytest tests/e2e/ -m e2e -v
```

E2E tests also run automatically during Docker build when using `--target e2e-test`.

## Docker Build

The Dockerfile uses a multi-stage build:

1. **base**: Installs system deps + Python deps + copies source code
2. **e2e-test**: Starts the service, polls `/health/ready`, runs e2e tests. Build fails if tests fail.
3. **production**: Lean final image with `/health/live` for Docker HEALTHCHECK

```bash
docker build .                         # Builds production image (skips tests)
docker build --target e2e-test ...     # Runs e2e tests during build
```

## Interactive API Documentation

Once the server is running:

- Swagger UI: `http://localhost:8090/docs`
- ReDoc: `http://localhost:8090/redoc`

## License

Internal use only.
