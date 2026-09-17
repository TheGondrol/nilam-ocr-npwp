# OCR KTP Orchestrator

FastAPI-based orchestration service for Indonesian ID card (KTP) OCR extraction with integrated quality checks, spoof detection, and tamper analysis.

## Features

- **OCR Extraction**: Extracts text fields from KTP images via external OCR + postprocessing services
- **Quality Checks**:
  - Rule-based quality validation (blur, glare, rotation detection)
  - Deep learning-based quality assessment (optional)
- **Spoof Detection**:
  - Lamination detection
  - Recapture detection
  - Graycopy detection
- **Tamper Detection**: Detects tampered/edited KTP images
- **KTP Classification**: Validates that the uploaded image is actually a KTP
- **Request Tracking**: Generate request IDs, track processing status, and retrieve results via database
- **Image Logging**: Saves processed images to MinIO or Google Cloud Storage
- **Rate Limiting**: Configurable sliding window rate limiter middleware
- **Health Checks**: Liveness, readiness, and full health endpoints (K8s-ready)
- **Elastic APM**: Optional application performance monitoring
- **Connection Pooling**: Shared aiohttp session for efficient downstream service calls
- **API Key Authentication**: All endpoints require `X-API-Key` header
- **E2E Testing**: End-to-end tests run during Docker build against real downstream services

## Project Structure

```
ms-bribrain-ocr-ktp-orchestrator/
├── src/
│   ├── core/                        # Core functionality
│   │   ├── config.py                # YAML configuration loader with env var substitution
│   │   ├── device.py                # GPU/CPU detection
│   │   ├── http_client.py           # Shared aiohttp session (connection pooling)
│   │   ├── logging.py               # Logging setup with rotation
│   │   └── validation.py            # Startup configuration validation
│   ├── api/                         # API layer
│   │   ├── dependencies.py          # API key auth, request ID validation
│   │   ├── models.py                # Pydantic request/response models
│   │   └── routes.py                # API endpoints
│   ├── middleware/
│   │   └── rate_limiter.py          # Sliding window rate limiter
│   ├── schemas/
│   │   └── database_schema.py       # SQLAlchemy ORM models
│   ├── services/                    # Business logic
│   │   ├── manage_service.py        # Main OCR pipeline orchestration
│   │   ├── ocr_service.py           # OCR extraction service client
│   │   ├── postprocess_service.py   # Postprocessing service client
│   │   ├── quality_service.py       # Quality check service client
│   │   ├── spoof_service.py         # Spoof detection service client (laminate, recapture, graycopy)
│   │   ├── classifier_service.py    # KTP classifier service client
│   │   ├── temper_service.py        # Tamper detection service client
│   │   ├── crop_helper.py           # Image cropping utilities
│   │   ├── database_service.py      # Async database operations
│   │   ├── minio_service.py         # MinIO image storage
│   │   └── gcs_service.py           # Google Cloud Storage image storage
│   └── main.py                      # FastAPI application entry point
├── tests/
│   ├── unit/                        # Unit tests (mocked dependencies)
│   ├── integration/                 # Integration tests
│   ├── e2e/                         # End-to-end tests (real downstream services)
│   └── data/                        # Test images
├── scripts/
│   └── run_e2e_tests.sh             # E2E test runner (used in Docker build)
├── config.yaml                      # Configuration file
├── Dockerfile                       # Multi-stage build (base → e2e-test → production)
├── start.sh                         # Local Docker deployment script
├── pyproject.toml                   # Python project config & dependencies
├── uv.lock                          # Dependency lock file
└── pytest.ini                       # Pytest configuration
```

## Configuration

All configuration is managed through `config.yaml` with environment variable substitution (`${VAR_NAME}`):

```yaml
# Downstream service URLs
services:
  ocr:
    url: "${OCR_SERVICE_URL}"
    api_key: "${OCR_SERVICE_API}"
    timeout: 30
  classifier:
    url: "${CLASSIFIER_SERVICE_URL}"
    api_key: "${CLASSIFIER_SERVICE_API}"
    timeout: 30
  quality:
    url: "${QUALITY_SERVICE_URL}"
    api_key: "${QUALITY_SERVICE_API}"
    timeout: 30
  lamination:
    url: "${LAMINATION_SERVICE_URL}"
    api_key: "${LAMINATION_SERVICE_API}"
    timeout: 30
  recapture:
    url: "${RECAPTURE_SERVICE_URL}"
    api_key: "${RECAPTURE_SERVICE_API}"
    timeout: 30
  graycopy:
    url: "${GRAYCOPY_SERVICE_URL}"
    api_key: "${GRAYCOPY_SERVICE_API}"
    timeout: 30
  temper:
    url: "${TEMPER_SERVICE_URL}"
    api_key: "${TEMPER_SERVICE_API}"
    timeout: 30
  postprocess:
    url: "${POSTPROCESS_SERVICE_URL}"
    api_key: "${POSTPROCESS_SERVICE_API}"
    timeout: 30
  orchestrator:
    url: "${ORCHESTRATOR_SERVICE_URL}"
    api_key: "${ORCHESTRATOR_SERVICE_API}"
    timeout: 30
    overall_timeout: 60

# Toggle individual pipeline stages
run_services:
  ocr: true
  classifier: true
  quality: true
  qualitydl: false
  lamination: true
  recapture: true
  graycopy: true
  temper: true
  postprocess: true

# Application settings
app:
  name: "OCR KTP Orchestrator"
  version: "1.0.0"
  host: "0.0.0.0"
  port: 8000

# Database (async PostgreSQL)
database:
  table_name: "bribrain_ocr_ktp_orchestrator"
  result_table_name: "bribrain_ocr_ktp_result"
  pool_size: 5
  max_overflow: 10

# Rate limiting
rate_limit:
  enabled: false
  requests_per_minute: 60
  requests_per_second: 10
  burst_size: 20

# Elastic APM (optional)
elastic_apm:
  enabled: false
  server_url: "${ELASTIC_APM_SERVER_URL}"
  service_name: "${ELASTIC_APM_SERVICE_NAME}"
```

Environment variables are loaded from a `.env` file at the project root.

## Environment Variables

Loaded from a `.env` file (copy `.env.template` → `.env`). `${VAR}` placeholders in `config.yaml` are resolved at startup; a missing variable resolves to an empty string.

On startup the application validates that all variables marked **Required** below are set (see `src/core/validation.py`); if any are missing the process logs the missing names and exits with code 1.

### Core

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `DATABASE_URL` | Yes | asyncpg PostgreSQL connection string. Backs the `bribrain_ocr_ktp_orchestrator` and `bribrain_ocr_ktp_result` tables. | `postgresql+asyncpg://user:pass@host:5432/bribrain_ocr_ktp` |
| `ORCHESTRATOR_SERVICE_API` | Yes | The expected value of the `X-API-Key` header on every endpoint. Incoming requests are authenticated by a constant-time compare against this value. (This is the orchestrator's *own* inbound API key — see Downstream services below for the matching `ORCHESTRATOR_SERVICE_URL`.) | `a-long-random-secret` |
| `API_KEY` | No | Present in `.env.template` but **not read by the application** — inbound auth uses `ORCHESTRATOR_SERVICE_API`. Kept for backward compatibility; safe to leave blank or omit. | _(unused)_ |
| `LOG_ENCRYPTION_KEY` | Yes | AES-256-GCM key for encrypting the `payload` and `result` log columns (in both `bribrain_ocr_ktp_orchestrator` and `bribrain_ocr_ktp_result`). Base64 of 32 random bytes; the **same key must be shared across all services** — the orchestrator also uses it to decrypt `bribrain_ocr_ktp_result` on `GET /v1/get-ocr-result/{id}`. | `<base64-of-32-random-bytes>` |
| `APP_ENVIRO` | No | Selects image-log storage backend: `onprem` → MinIO, `gcp` → Google Cloud Storage. Defaults to `gcp` when unset. Note: an **empty** value is NOT treated as `onprem`; any value other than the literal `onprem` (including empty) routes image storage to GCS. | `onprem` |

### Downstream services

The orchestrator calls each microservice over HTTP. Each `*_SERVICE_URL` is the **full URL the orchestrator POSTs to, including the path** (e.g. `http://host:8001/v1/ocr`), and each `*_SERVICE_API` is the API key sent in that call's `x-api-key` header. The `*_SERVICE_API` keys are **required** at startup; the `*_SERVICE_URL` values are required only for the pipeline stages enabled via `run_services.*` in `config.yaml`.

| URL variable | API-key variable | Required | `run_services` toggle | Description |
|--------------|------------------|----------|------------------------|-------------|
| `OCR_SERVICE_URL` | `OCR_SERVICE_API` | URL when `run_services.ocr`; API key always | `ocr` | OCR text extraction from the KTP image. |
| `CLASSIFIER_SERVICE_URL` | `CLASSIFIER_SERVICE_API` | URL when `run_services.classifier`; API key always | `classifier` | Validates the image is a KTP. |
| `QUALITY_SERVICE_URL` | `QUALITY_SERVICE_API` | URL when `run_services.quality`; API key always | `quality` | Rule-based quality check (blur, glare, rotation). |
| `QUALITYDL_SERVICE_URL` | `QUALITYDL_SERVICE_API` | URL when `run_services.qualitydl`; API key always | `qualitydl` | Deep-learning (IQA-DL) quality check. |
| `LAMINATION_SERVICE_URL` | `LAMINATION_SERVICE_API` | URL when `run_services.lamination`; API key always | `lamination` | Lamination / spoof check. |
| `RECAPTURE_SERVICE_URL` | `RECAPTURE_SERVICE_API` | URL when `run_services.recapture`; API key always | `recapture` | Recapture (photo-of-photo) detection. |
| `GRAYCOPY_SERVICE_URL` | `GRAYCOPY_SERVICE_API` | URL when `run_services.graycopy`; API key always | `graycopy` | Grayscale-photocopy detection. |
| `TEMPER_SERVICE_URL` | `TEMPER_SERVICE_API` | URL when `run_services.temper`; API key always | `temper` | Tamper / digital-alteration detection. |
| `POSTPROCESS_SERVICE_URL` | `POSTPROCESS_SERVICE_API` | URL when `run_services.postprocess`; API key always | `postprocess` | Structures raw OCR output into KTP fields. |
| `ORCHESTRATOR_SERVICE_URL` | `ORCHESTRATOR_SERVICE_API` | API key always | `orchestrator` | The orchestrator's own URL / inbound API key (used for self-reference and overall pipeline timeout config). |

> **⚠️ `QUALITYDL_SERVICE_URL` must point at the full path ending in `/v1/ocr_qualitydl`.** The IQA-DL endpoint was renamed from `/filter` to `/v1/ocr_qualitydl`. A stale value (e.g. one still ending in `/filter`) causes the DL quality check to fail with a 404, which surfaces as a `SERVICE_ERROR` (HTTP 500) from the orchestrator. Example: `http://iqa-dl-host:8005/v1/ocr_qualitydl`.

### Image-log storage — on-prem (`APP_ENVIRO=onprem`)

Used when `APP_ENVIRO=onprem` and `logging.log_images` is enabled in `config.yaml`. Processed images are written to MinIO.

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `MINIO_ENDPOINT` | Yes | MinIO host:port (no scheme). | `minio.internal:9000` |
| `MINIO_ACCESS_KEY` | Yes | MinIO access key. | `minio-access-key` |
| `MINIO_SECRET_KEY` | Yes | MinIO secret key. | `minio-secret-key` |
| `MINIO_SECURE` | No | Use TLS for the MinIO connection. Any value other than `true` (case-insensitive) is treated as `false`. Defaults to `false`. | `false` |

> MinIO variables are validated as **required at startup regardless of `APP_ENVIRO`** (they appear in the required-env-var list). For a pure GCS deployment, set them to placeholder values to satisfy validation, or run with `APP_ENVIRO=onprem` if you intend to use MinIO.

### Image-log storage — cloud (`APP_ENVIRO=gcp`)

Used when `APP_ENVIRO=gcp` (the default) and `logging.log_images` is enabled. Processed images are written to Google Cloud Storage. The service account is assembled from the `SA_*` variables below (read directly from the environment by `src/services/gcs_service.py`).

> **Note:** these `SA_*` variables are **not present in `.env.template`**, but the GCS code reads them with `os.environ[...]` and will raise `KeyError` if they are missing. When `APP_ENVIRO=gcp` and image logging is enabled, operators must add the full `SA_*` block to `.env`.

| Variable | Required (GCS) | Description | Example |
|----------|----------------|-------------|---------|
| `SA_TYPE` | Yes | Service-account credential type. | `service_account` |
| `SA_PROJECT_ID` | Yes | GCP project ID. | `my-gcp-project` |
| `SA_PRIVATE_KEY` | Yes | Private key with literal `\n` escapes (the code converts `\n` → newline). Do **not** wrap in extra quotes. | `-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n` |
| `SA_CLIENT_MAIL` | Yes | Service-account client email. | `svc@my-gcp-project.iam.gserviceaccount.com` |
| `SA_CLIENT_ID` | Yes | Service-account client ID. | `1234567890` |
| `SA_AUTH_URI` | Yes | OAuth2 auth URI. | `https://accounts.google.com/o/oauth2/auth` |
| `SA_TOKEN_URI` | Yes | OAuth2 token URI. | `https://oauth2.googleapis.com/token` |
| `SA_AUTH_PROVIDER` | Yes | Auth-provider x509 cert URL. | `https://www.googleapis.com/oauth2/v1/certs` |
| `SA_CERT_URL` | Yes | Client x509 cert URL. | `https://www.googleapis.com/robot/v1/metadata/x509/svc%40my-gcp-project.iam.gserviceaccount.com` |

### Elastic APM (optional)

Resolved into the `elastic_apm` block of `config.yaml`. APM is only attached when it is enabled in config **and** `ELASTIC_APM_SERVER_URL` is non-empty (an empty server URL disables it).

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `ELASTIC_APM_SERVER_URL` | No | APM server URL. Empty disables APM. | `http://apm-server:8200` |
| `ELASTIC_APM_SERVICE_NAME` | No | APM service name. | `ms-bribrain-ocr-ktp-orchestrator` |
| `ELASTIC_APM_SERVICE_VERSION` | No | Reported service version. | `1.0.0` |

## Log encryption (DPIA)

The orchestrator persists KTP personal data, so its PII columns are encrypted at rest with **AES-256-GCM** (see `src/core/crypto.py`):

- **Encrypted (`BYTEA`):**
  - `payload` and `result` in the log table `bribrain_ocr_ktp_orchestrator`.
  - `result` in the async-result table `bribrain_ocr_ktp_result` (the final extracted KTP fields).
- **Plaintext:** `error_message` (`TEXT`), `status`, `request_id`, `response_code`, `processing_time`, `created_at`, `updated_at`.
- **Decrypt-on-read:** this is the only service that decrypts at runtime — `GET /v1/get-ocr-result/{request_id}` decrypts `result` from `bribrain_ocr_ktp_result` before returning it to the client. All log tables are otherwise write-only.
- **Key:** `LOG_ENCRYPTION_KEY` — base64 of 32 random bytes, the **same key across all encrypting services** (`dgc_ext`, `dgc_irl`, `dgc_oct`, `dgc_pps`). Startup fails if it is missing while DB logging is enabled. Generate with `python -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())"`.
- **Stored format (`BYTEA`):** `version(1) || key_id(1) || nonce(12) || ciphertext+tag` — a fresh random 12-byte nonce per write and a 16-byte authentication tag (tamper-evident; wrong key or altered bytes fail to decrypt).
- **Migration:** run `schema_encryption.sql` (repo root) once per environment to convert these columns to `BYTEA` (it truncates existing rows).

## Rate limiting

A per-client **sliding-window** rate limiter (`src/middleware/rate_limiter.py`) is available as middleware. It is **opt-in** — the middleware is only attached when `rate_limit.enabled: true` (the `config.yaml` default is `false`, and startup logs `Rate limiting: DISABLED`). State is kept **in-memory, per process**.

**Configuration (`config.yaml` → `rate_limit`):**

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `false` | Attach the middleware at startup |
| `requests_per_second` | `10` | Per-client cap over the trailing 1 s (burst guard) |
| `requests_per_minute` | `60` | Per-client cap over the trailing 60 s |
| `burst_size` | `20` | Accepted but **not currently enforced** |
| `cleanup_interval` | `60` | Seconds between pruning of expired timestamps |
| `exclude_paths` | `/health`, `/`, `/docs`, `/openapi.json`, `/redoc` | Paths never rate-limited |

**How it works:**

- **Client key:** the first IP in `X-Forwarded-For` (for proxies/load balancers), else the direct client IP, else `unknown`.
- On each request it counts that client's timestamps in the trailing **1 second** and **60 seconds**, measured from *now* — so the window slides continuously rather than resetting on fixed boundaries. It rejects if either the per-second or per-minute count has reached its limit; otherwise it records the timestamp and allows the request.
- **Rejection → HTTP 429** with the standard envelope (`error_code: RATE_LIMIT_EXCEEDED`, `errors.retry_after`) plus a `Retry-After` header. For the per-minute limit, `retry_after` is the time until the oldest request in the window expires.
- **Allowed responses** carry `X-RateLimit-Limit` and `X-RateLimit-Remaining` headers. A periodic cleanup (every `cleanup_interval` seconds) prunes old timestamps so memory stays bounded.

**Caveats:**

- Counters are **per process** — with multiple workers/replicas the effective limit is `N ×` the configured value. A truly shared limit needs an external store (e.g. Redis).
- `burst_size` is configured but not enforced; `requests_per_second` is the effective burst control.
- The limiter trusts the first `X-Forwarded-For` value to identify clients, so set that header authoritatively at your ingress/LB to prevent header-spoofing bypass.

## Installation

### Using uv (recommended)

```bash
# Install dependencies
uv sync

# Run the application
uv run uvicorn src.main:app --port 8000 --host 0.0.0.0
```

### Using Docker

```bash
# Build the production image (skips e2e tests)
docker build -t ms-bribrain-ocr-orchestrator .

# Build with e2e tests (requires downstream services to be reachable)
docker build --target e2e-test --network host --secret id=env,src=.env -t ms-bribrain-ocr-orchestrator-test .

# Run the container
docker run -d \
  --name ms-bribrain-ocr-orchestrator \
  --env-file .env \
  -p 8000:8000 \
  --restart always \
  ms-bribrain-ocr-orchestrator

# View logs
docker logs -f ms-bribrain-ocr-orchestrator
```

### Using deploy script

```bash
# Build + e2e test + deploy
./start.sh

# Build + deploy (skip e2e tests)
./start.sh --skip-test
```

## API Endpoints

All endpoints require the `X-API-Key` header for authentication.

### POST /v1/generate-request-id

Generate a unique request ID for tracking OCR processing.

**Response (200):**

```json
{
  "status_code": 200,
  "status_desc": "OK",
  "message": "Success",
  "data": {
    "request_id": "OCR_550e8400-e29b-41d4-a716-446655440000"
  },
  "error_code": null,
  "errors": null,
  "request_id": "OCR_550e8400-e29b-41d4-a716-446655440000"
}
```

### POST /v1/extract-ocr

Extract KTP data from an image. Requires a pre-generated `request_id`.

**Request:**

- Content-Type: `multipart/form-data`
- Fields:
  - `request_id` (string, required): Pre-generated request ID
  - `file` (file, required): KTP image (JPEG/JPG/PNG, max 1MB)

**Response (200 - Success):**

```json
{
  "status_code": 200,
  "status_desc": "OK",
  "message": "Success",
  "data": {
    "nik": "1234567890123456",
    "nama": "JOHN DOE",
    "tempat_lahir": "JAKARTA"
  },
  "error_code": null,
  "errors": null,
  "request_id": "OCR_..."
}
```

**Response (400 - Rejection):**

```json
{
  "status_code": 400,
  "status_desc": "Bad Request",
  "message": "Image rejected: spoof detected (unlaminated)",
  "data": "",
  "error_code": "SPOOF_UNLAMINATED",
  "errors": null,
  "request_id": "OCR_..."
}
```

### GET /v1/get-ocr-result/{request_id}

Retrieve OCR processing result and status by request ID.

**Response (200):**

```json
{
  "status_code": 200,
  "status_desc": "OK",
  "message": "Success",
  "data": {
    "request_id": "OCR_...",
    "status": "completed",
    "result": { "nik": "...", "nama": "..." },
    "error_message": null,
    "created_at": "2025-01-01T00:00:00+00:00",
    "updated_at": "2025-01-01T00:00:05+00:00"
  },
  "error_code": null,
  "errors": null,
  "request_id": "OCR_..."
}
```

### Health Endpoints

#### GET /health/live

Liveness probe. Returns 200 if the process is alive. Use for K8s `livenessProbe`.

```json
{
  "status": "alive",
  "version": "1.0.0"
}
```

#### GET /health/ready

Readiness probe. Returns 200 if database and HTTP client are up. Use for K8s `readinessProbe`. Returns 503 if not ready.

```json
{
  "status": "ready",
  "checks": {
    "database": { "status": "up" },
    "http_client": { "status": "up" }
  }
}
```

#### GET /health

Full health check with downstream service probes.

- `healthy` (200): All checks pass
- `degraded` (200): DB + HTTP client OK, but some downstream services are down
- `unhealthy` (503): DB or HTTP client is down

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "cpu",
  "checks": {
    "database": { "status": "up" },
    "http_client": { "status": "up" },
    "services": {
      "ocr": { "status": "up", "url": "http://...:8001/health" },
      "classifier": { "status": "up", "url": "http://...:8002/health" },
      "quality": { "status": "up", "url": "http://...:8003/health" }
    }
  }
}
```

### GET /

Service information and links.

```json
{
  "service": "OCR KTP Orchestrator",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health",
  "liveness": "/health/live",
  "readiness": "/health/ready"
}
```

## Error Codes

| Code | Description |
|------|-------------|
| `FILE_TOO_LARGE` | Uploaded file exceeds size limit |
| `INVALID_FILE_TYPE` | File is not JPEG/PNG |
| `INVALID_REQUEST_ID` | Request ID format is invalid |
| `REQUEST_NOT_FOUND` | No record found for request ID |
| `REQUEST_ALREADY_PROCESSED` | Request ID already used |
| `QUALITY_BLUR` | Image rejected due to blur |
| `QUALITY_GLARE` | Image rejected due to glare |
| `QUALITY_ROTATION` | Image rejected due to rotation |
| `QUALITY_MULTIPLE` | Multiple quality issues detected |
| `QUALITY_DL_POOR` | Deep learning quality check failed |
| `SPOOF_UNLAMINATED` | KTP is not laminated |
| `SPOOF_RECAPTURE` | Image is a photo of a photo |
| `SPOOF_GRAYCOPY` | Image is a grayscale photocopy |
| `NOT_KTP` | Image is not a KTP |
| `TAMPERED` | KTP image has been tampered with |
| `PIPELINE_TIMEOUT` | Processing exceeded time limit |
| `INTERNAL_ERROR` | Unexpected server error |
| `SERVICE_ERROR` | Downstream service error |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `UNAUTHORIZED` | Invalid or missing API key |

## Testing

### Unit & Integration Tests

```bash
# Run all unit and integration tests with coverage
uv run pytest tests/unit/ tests/integration/ -v --tb=short

# Run specific test file
uv run pytest tests/unit/test_routes.py -v
```

### End-to-End Tests

E2E tests call real downstream microservices. All services must be running and reachable.

```bash
# In-process mode (mocks DB, real downstream calls)
uv run pytest tests/e2e/ -m e2e -v --no-cov

# Against a live running orchestrator
E2E_BASE_URL=http://127.0.0.1:8000 uv run pytest tests/e2e/ -m e2e -v --no-cov
```

E2E tests also run automatically during Docker build when using `--target e2e-test`.

## Docker Build

The Dockerfile uses a multi-stage build:

1. **base**: Installs system deps + Python deps + copies source code
2. **e2e-test**: Starts the orchestrator, polls `/health/ready`, runs e2e tests. Build fails if tests fail.
3. **production**: Lean final image with only runtime dependencies

```
docker build .                         # Builds production image (skips tests)
docker build --target e2e-test ...     # Runs e2e tests during build
```

## Downstream Services

The orchestrator calls these microservices in its pipeline:

| Service | Purpose | Config key |
|---------|---------|------------|
| OCR Extractor | Text extraction from KTP images | `services.ocr` |
| KTP Classifier | Validates image is a KTP | `services.classifier` |
| Quality (Rule-based) | Blur, glare, rotation detection | `services.quality` |
| Quality (DL) | Deep learning quality assessment | `services.qualitydl` |
| Lamination | Checks KTP is laminated | `services.lamination` |
| Recapture | Detects photo-of-photo | `services.recapture` |
| Graycopy | Detects grayscale photocopies | `services.graycopy` |
| Tamper | Detects image tampering | `services.temper` |
| Postprocessor | Structures raw OCR output into fields | `services.postprocess` |

Each service can be toggled on/off via `run_services` in `config.yaml`.

## Interactive API Documentation

Once the server is running:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License

Internal use only.
