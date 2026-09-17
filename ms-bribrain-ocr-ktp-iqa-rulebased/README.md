# IQA Rule-Based Service

FastAPI service for checking KTP image quality using rule-based detection methods: blur, glare, rotation, and OCR confidence analysis.

## Features

- **Blur Detection**: Laplacian variance method to detect blurry images
- **Glare Detection**: Adaptive threshold-based glare detection with OCR text region overlap analysis
- **Rotation Detection**: InsightFace (RetinaFace) face-based rotation detection
- **Confidence Analysis**: Median OCR confidence score checking
- **Database Logging**: Async PostgreSQL logging for every request
- **Health Checks**: Liveness, readiness, and full health endpoints (K8s-ready)
- **API Key Authentication**: Quality endpoint requires `X-API-Key` header
- **Request ID Tracking**: Middleware injects/propagates `x-request-id` headers
- **Thread Pool Processing**: Non-blocking image processing via `asyncio.to_thread`
- **E2E Testing**: End-to-end tests run during Docker build
- **Type-Safe Configuration**: YAML-based configuration with Pydantic validation

## Project Structure

```
ms-bribrain-ocr-ktp-iqa-rulebased/
├── src/
│   ├── core/                            # Core functionality
│   │   ├── config.py                    # Pydantic config management (singleton)
│   │   ├── device.py                    # GPU/CPU detection
│   │   ├── logging.py                   # Logging setup with request_id context
│   │   └── validation.py               # Startup configuration validation
│   ├── api/                             # API layer
│   │   ├── dependencies.py             # API key verification
│   │   └── routes.py                   # OCR quality endpoint
│   ├── middleware/
│   │   └── add_requestid.py            # Request ID injection middleware
│   ├── schemas/
│   │   ├── api_schema.py               # Pydantic response models
│   │   └── database_schema.py          # SQLAlchemy ORM models
│   ├── services/                        # Business logic
│   │   ├── image_quality.py            # Main orchestration service
│   │   ├── blur_detection.py           # Laplacian variance blur detection
│   │   ├── glare_detection.py          # Adaptive threshold glare detection
│   │   ├── rotation_detection.py       # InsightFace (RetinaFace) face-based rotation
│   │   └── database_service.py         # Async database connection & logging
│   └── main.py                          # FastAPI application entry point
├── tests/
│   ├── unit/                            # Unit tests
│   ├── integration/                     # Integration tests
│   ├── e2e/                             # End-to-end tests
│   │   ├── conftest.py                  # E2E fixtures (in-process or live server)
│   │   └── test_e2e_iqa_rulebased.py   # E2E quality pipeline tests
│   └── data/                            # Test images
├── scripts/
│   └── run_e2e_tests.sh                 # E2E test runner (used in Docker build)
├── config.yaml                          # Configuration file
├── Dockerfile                           # Multi-stage build (base → e2e-test → production)
├── pyproject.toml                       # Python project config & dependencies
└── README.md
```

## Configuration

All configuration is managed through `config.yaml` with Pydantic validation:

```yaml
# Application
app:
  name: "OCR Quality Service"
  version: "0.1.0"
  debug: false

# Server
server:
  host: "0.0.0.0"
  port: 8000          # NOTE: not used in containers — see note below
  workers: 1
  max_file_size_mb: 10

# Image quality thresholds
quality:
  blur:
    threshold: 100.0
  confidence:
    threshold_median: 0.8
  glare:
    min_area: 100
    padding_size: 20
    kernel_size: 5
    min_text_confidence: 0.6
    affected_percentage_threshold: 5
  rotation:
    min_face_proportion: 0.12
    face_detection_confidence: 0.5

# Database (async PostgreSQL)
database:
  table_name: "bribrain_ocr_ktp_iqa_rulebased"
  schema: "public"
  pool_size: 5
  max_overflow: 10

# Logging
logging:
  log_to_database: true
  level: "INFO"
  file:
    enabled: true
    path: "logs/ocr_quality.log"
    max_bytes: 10485760          # 10MB
    backup_count: 5
```

> **Port:** `config.yaml` sets `server.port: 8000`, but that value is only used by the `python -m src.main` entrypoint. The container ignores it: the Dockerfile `EXPOSE`s **8070** and runs uvicorn with `--port 8070`, and `docker run` publishes `-p 8070:8070`. The service listens on **8070** in all documented deployments.

The InsightFace RetinaFace model pack (`buffalo_sc`) is baked into the Docker image at build time (pre-cached in the Dockerfile), so no model download happens at runtime.

## Environment Variables

Loaded from a `.env` file (copy `.env.template` → `.env`). `${VAR}` placeholders in `config.yaml` are resolved at startup; a missing variable resolves to an empty string.

Startup validation (`src/core/validation.py`) requires all variables in the table below and exits if any are missing. Note: `MINIO_*` are validated as **required at startup even though this service ships the model inside the Docker image** — the validation block lists them unconditionally, so they must be present (non-empty) for the service to start even though no model is fetched from MinIO at runtime.

### Core

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `DATABASE_URL` | Yes | asyncpg PostgreSQL connection string used for request logging. | `postgresql+asyncpg://user:pass@host:5432/bribrain_ocr_ktp` |
| `API_KEY` | Yes | Expected value of the `X-API-Key` header on the quality endpoint. | `your-api-key` |
| `LOG_ENCRYPTION_KEY` | Yes | AES-256-GCM key for encrypting the `payload` and `result` log columns. Base64 of 32 random bytes; the same key must be shared across all services. | `<base64-of-32-random-bytes>` |

### MinIO

Required at startup by `validate_environment_variables()` even though the InsightFace model is baked into the image at build time and not downloaded at runtime.

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `MINIO_ENDPOINT` | Yes | MinIO host:port. Validated as required at startup. | `minio.internal:9000` |
| `MINIO_ACCESS_KEY` | Yes | MinIO access key. Validated as required at startup. | `minioadmin` |
| `MINIO_SECRET_KEY` | Yes | MinIO secret key. Validated as required at startup. | `minio-secret` |

### Elastic APM (optional)

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `ELASTIC_APM_SERVER_URL` | No | APM server URL. APM is disabled when empty/unset. | `http://apm-server:8200` |
| `ELASTIC_APM_SERVICE_NAME` | No | Service name reported to APM. Defaults to `ocr-ktp-iqa-rulebased` if unset. | `ms-bribrain-ocr-ktp-iqa-rulebased` |
| `ELASTIC_APM_SECRET_TOKEN` | No | APM secret token, if the APM server requires one. | `your-apm-token` |

## Log encryption (DPIA)

The PII columns of this service's log table `bribrain_ocr_ktp_iqa_rulebased` are encrypted at rest with **AES-256-GCM** (see `src/core/crypto.py`):

- **Encrypted (`BYTEA`):** `payload` and `result`. Note `payload` stores the **full OCR result text** this service receives for the quality check (it contains KTP field values), so it is treated as PII.
- **Plaintext:** `error_message` (`TEXT`), `request_id`, `response_code`, `processing_time`, `created_at`.
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
uv run uvicorn src.main:app --host 0.0.0.0 --port 8070
```

### Using Docker

```bash
# Build the production image (skips e2e tests)
docker build -t ms-bribrain-ocr-ktp-iqa-rulebased .

# Build with e2e tests (requires DB to be reachable)
docker build --target e2e-test --network host --secret id=env,src=.env -t iqa-rulebased-test .

# Run the container
docker run -d \
  --name ms-bribrain-ocr-ktp-iqa-rulebased \
  --env-file .env \
  -p 8070:8070 \
  --restart always \
  ms-bribrain-ocr-ktp-iqa-rulebased

# View logs
docker logs -f ms-bribrain-ocr-ktp-iqa-rulebased
```

## API Endpoints

The quality endpoint requires the `X-API-Key` header. Health and root endpoints do not.

### POST /v1/ocr_quality

Check image quality based on blur, glare, rotation, and OCR confidence.

**Request:**

- Content-Type: `multipart/form-data`
- Fields:
  - `file` (image file, JPEG/PNG)
  - `ocr_result` (JSON string of OCR results: `[[[coordinates], (text, confidence)], ...]`)
- Header: `X-API-Key: <your-api-key>`

**Response (200):**

```json
{
  "status_code": 200,
  "status_desc": "OK",
  "message": "Success",
  "data": {
    "low_confidence": false,
    "is_blurry": false,
    "is_glare": false,
    "is_rotated": false
  },
  "error_code": null,
  "errors": null,
  "request_id": "abc123"
}
```

### Quality Check Output

| Field | Type | Description |
|-------|------|-------------|
| `low_confidence` | bool | OCR median confidence below threshold |
| `is_blurry` | bool | Image detected as blurry (Laplacian variance) |
| `is_glare` | bool | Glare detected overlapping text regions |
| `is_rotated` | bool | Image detected as rotated (face position analysis) |

### Health Endpoints

#### GET /health/live

Liveness probe. Returns 200 if the process is alive. Use for K8s `livenessProbe`.

```json
{ "status": "alive", "version": "0.1.0" }
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

- `healthy`: DB up
- `unhealthy`: DB down

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "checks": {
    "database": { "status": "up" }
  }
}
```

### GET /

Service information and endpoint listing.

```json
{
  "service": "OCR Quality Service",
  "version": "0.1.0",
  "status": "running",
  "endpoints": {
    "health": "/health",
    "liveness": "/health/live",
    "readiness": "/health/ready",
    "ocr_quality": "/v1/ocr_quality (POST)"
  }
}
```

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_INPUT` | 400 | Invalid OCR result JSON format, or undecodable image |
| `FILE_TOO_LARGE` | 413 | Uploaded image exceeds the 1 MB size limit (`server.max_file_size_mb`) |
| `INTERNAL_ERROR` | 500 | Unexpected server error during processing |

## Detection Methods

### Blur Detection
- Uses Laplacian variance method on grayscale image
- Threshold: 100.0 (configurable)
- Lower variance = blurrier image

### Glare Detection
- Adaptive brightness threshold based on median image brightness
- Detects bright regions using morphological operations
- Checks overlap with OCR text regions (min confidence 0.6)
- Flags if affected text area exceeds threshold (5%)

### Rotation Detection
- Uses InsightFace RetinaFace detector (ONNX, CPU) — model pack `buffalo_sc`
- Checks face position relative to expected verification box
- Compares face proportion against minimum threshold (0.12)

### Confidence Analysis
- Calculates median OCR confidence score
- Flags if median falls below threshold (0.8)

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

E2E tests validate the full quality check pipeline. Supports two modes:

```bash
# In-process mode (mocked quality service + DB, full API flow)
uv run pytest tests/e2e/ -m e2e -v

# Against a live running service
E2E_BASE_URL=http://127.0.0.1:8070 uv run pytest tests/e2e/ -m e2e -v
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

- Swagger UI: `http://localhost:8070/docs`
- ReDoc: `http://localhost:8070/redoc`

## License

Internal use only.
