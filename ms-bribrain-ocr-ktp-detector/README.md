# KTP Detection Service

FastAPI-based service for KTP (Indonesian ID card) and non-KTP object detection using a YOLO model with OpenVINO acceleration.

## Features

- **YOLO Object Detection**: Detects KTP and non-KTP objects in images
- **OpenVINO Acceleration**: 2-3x faster CPU inference via OpenVINO model format
- **GPU/CPU Support**: Automatic CUDA GPU detection with CPU fallback
- **Model Download**: Auto-downloads model from MinIO (on-prem) or GCS (cloud) at startup
- **Database Logging**: Async PostgreSQL logging for every request
- **Health Checks**: Liveness, readiness, and full health endpoints (K8s-ready)
- **API Key Authentication**: All detection endpoints require `X-API-Key` header
- **Request ID Tracking**: Middleware injects/propagates `x-request-id` headers
- **Thread Pool Inference**: Non-blocking inference via ThreadPoolExecutor
- **E2E Testing**: End-to-end tests run during Docker build against the real model
- **Configurable**: YAML-based configuration for model, server, device, and logging

## Project Structure

```
ms-bribrain-ocr-ktp-detector/
├── src/
│   ├── core/                        # Core functionality
│   │   ├── config.py                # Singleton config manager from YAML
│   │   ├── device.py                # GPU/CPU detection, cgroup-aware threading
│   │   ├── logging.py               # Logging setup with request_id context
│   │   └── validation.py            # Startup configuration validation
│   ├── api/                         # API layer
│   │   ├── dependencies.py          # API key verification
│   │   ├── routes.py                # API endpoints
│   │   └── schemas.py               # Pydantic request/response models
│   ├── middleware/
│   │   └── add_requestid.py         # Request ID injection middleware
│   ├── models/                      # YOLO model files (downloaded at startup)
│   │   └── best_openvino_model/     # OpenVINO model directory
│   ├── schemas/
│   │   └── database.py              # SQLAlchemy ORM models
│   ├── services/                    # Business logic
│   │   ├── predictor.py             # YOLO model loading & prediction
│   │   ├── database_service.py      # Async database connection & logging
│   │   ├── minio_service.py         # MinIO model download (on-prem)
│   │   └── gcs_service.py           # GCS model download (cloud)
│   └── main.py                      # FastAPI application entry point
├── tests/
│   ├── unit/                        # Unit tests (via test_*.py in tests/)
│   ├── e2e/                         # End-to-end tests (real model or mocked)
│   │   ├── conftest.py              # E2E fixtures (in-process or live server)
│   │   └── test_e2e_detection.py    # E2E detection pipeline tests
│   └── data/                        # Test images
├── scripts/
│   └── run_e2e_tests.sh             # E2E test runner (used in Docker build)
├── config.yaml                      # Configuration file
├── Dockerfile                       # Multi-stage build (base → e2e-test → production)
├── pyproject.toml                   # Python project config & dependencies
├── uv.lock                          # Dependency lock file
└── pytest.ini                       # Pytest configuration & markers
```

## Configuration

All configuration is managed through `config.yaml`:

```yaml
# Model settings
model:
  path: ./src/models/ktp_detection_model.pt
  openvino_path: ./src/models/best_openvino_model
  export_format: openvino     # 'pt' or 'openvino'
  confidence: 0.5
  iou_threshold: 0.45

# Detection classes
classes:
  names:
    0: "ktp"
    1: "non-ktp"

# Performance tuning
performance:
  thread_pool_multiplier: 1
  openvino_mode: latency      # or 'throughput'

# Server
server:
  host: 0.0.0.0
  port: 8000

# Device
device:
  prefer_gpu: false
  force_cpu: true

# Database (async PostgreSQL)
database:
  pool_size: 5
  max_overflow: 10

# Logging
logging:
  log_to_database: true
  level: INFO

# Model storage (on-prem)
minio:
  bucket: bribrain-webviewasset-dev
  object: ocr_ktp_models/ktp_detection_model.pt

# Model storage (cloud)
gcs:
  bucket_name: "gc-bribrain-dev-gcs-iguru-01"
  prefix: "ocr_models/detection_models/"
```

> **Note on the listening port:** `config.yaml` declares `server.port: 8000`, but
> that value is only used when launching via `python -m src.main`. The Docker image
> (`EXPOSE`/`CMD`) and the recommended `uvicorn` command run the service on **port
> 8060**, which is the authoritative port for all deployments.

Environment variables are loaded from a `.env` file. Set `APP_ENVIRO=onprem` to download models from MinIO, otherwise GCS is used.

## Environment Variables

Loaded from a `.env` file (copy `.env.template` → `.env`). `${VAR}` placeholders in `config.yaml` are resolved at startup; a missing variable resolves to an empty string.

### Core

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `DATABASE_URL` | Yes | PostgreSQL (asyncpg) connection string for request logging. | `postgresql+asyncpg://user:pass@host:5432/bribrain_ocr_ktp` |
| `API_KEY` | Yes | Secret checked against the `X-API-Key` header on the detection endpoint. | `super-secret-key` |
| `APP_ENVIRO` | Yes | Selects the model storage backend: `onprem` → MinIO, anything else (e.g. `gcp`) → GCS. An **empty value is NOT treated as `onprem`** — it falls through to the GCS path — so set it explicitly. | `onprem` |

### On-prem model storage (MinIO) — used when `APP_ENVIRO=onprem`

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `MINIO_ENDPOINT` | Yes (validated at startup) | MinIO host\:port the model is downloaded from. | `minio.internal:9000` |
| `MINIO_ACCESS_KEY` | Yes (validated at startup) | MinIO access key. | `minio-access-key` |
| `MINIO_SECRET_KEY` | Yes (validated at startup) | MinIO secret key. | `minio-secret-key` |
| `MINIO_SECURE` | No | Use TLS for MinIO when `true`; any other value means plain HTTP. Defaults to `false`. | `false` |

> Startup validation (`validate_environment_variables`) always requires `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, and `MINIO_SECRET_KEY`, so populate them even for the GCS path (or the service exits at boot).

### Cloud model storage (GCS) — used when `APP_ENVIRO` is not `onprem`

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `SA_TYPE` | Yes (GCS) | Service-account credential type. | `service_account` |
| `SA_PROJECT_ID` | Yes (GCS) | GCP project ID. | `gc-bribrain-dev` |
| `SA_PRIVATE_KEY` | Yes (GCS) | Service-account private key. Contains escaped `\n` sequences which are converted to real newlines at runtime — do NOT add extra surrounding quotes. | `-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n` |
| `SA_CLIENT_MAIL` | Yes (GCS) | Service-account client email. | `svc@project.iam.gserviceaccount.com` |
| `SA_CLIENT_ID` | Yes (GCS) | Service-account client ID. | `123456789012345678901` |
| `SA_AUTH_URI` | Yes (GCS) | OAuth2 auth URI. | `https://accounts.google.com/o/oauth2/auth` |
| `SA_TOKEN_URI` | Yes (GCS) | OAuth2 token URI. | `https://oauth2.googleapis.com/token` |
| `SA_AUTH_PROVIDER` | Yes (GCS) | Auth provider x509 cert URL. | `https://www.googleapis.com/oauth2/v1/certs` |
| `SA_CERT_URL` | Yes (GCS) | Client x509 cert URL. | `https://www.googleapis.com/robot/v1/metadata/x509/svc%40project.iam.gserviceaccount.com` |

### Elastic APM (optional)

APM is wired up but only activates when `ELASTIC_APM_SERVER_URL` is non-empty; leaving it blank disables APM entirely.

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `ELASTIC_APM_SERVER_URL` | No | Elastic APM server URL. Empty disables APM. | `http://apm-server:8200` |
| `ELASTIC_APM_SERVICE_NAME` | No | Service name reported to APM. Defaults to `ocr-ktp-detector`. | `ocr-ktp-detector` |
| `ELASTIC_APM_SECRET_TOKEN` | No | Secret token for the APM server (if configured). | `apm-secret-token` |

## Installation

### Using uv (recommended)

```bash
# Install dependencies
uv sync

# Run the service
uv run uvicorn src.main:app --host 0.0.0.0 --port 8060
```

### Using Docker

```bash
# Build the production image (skips e2e tests)
docker build -t ms-bribrain-ocr-ktp-detector .

# Build with e2e tests (requires DB and model storage to be reachable)
docker build --target e2e-test --network host --secret id=env,src=.env -t detector-test .

# Run the container
docker run -d \
  --name ms-bribrain-ocr-ktp-detector \
  --env-file .env \
  -p 8060:8060 \
  --restart always \
  ms-bribrain-ocr-ktp-detector

# View logs
docker logs -f ms-bribrain-ocr-ktp-detector
```

## API Endpoints

All detection endpoints require the `X-API-Key` header. Health and root endpoints do not.

### POST /v1/ocr_ktp_detection

Detect KTP and non-KTP objects in an image.

**Request:**

- Content-Type: `multipart/form-data`
- Fields: `file` (image file, JPEG/PNG)
- Header: `X-API-Key: <your-api-key>`
- Max upload size: **1 MB** (configurable via `image.max_size_mb`). Oversized uploads are rejected with `413` / `FILE_TOO_LARGE`.

**Response (200):**

```json
{
  "status_code": 200,
  "status_desc": "OK",
  "message": "Success",
  "data": {
    "filename": "image.jpg",
    "detected": true,
    "num_detected": 1,
    "detections": [
      {
        "class_id": 0,
        "class_name": "ktp",
        "confidence": 0.95,
        "bbox": { "x1": 100.0, "y1": 50.0, "x2": 400.0, "y2": 300.0 }
      }
    ],
    "status": "OK",
    "reason": null,
    "timestamp": "2025-01-01T10:00:00"
  },
  "error_code": null,
  "errors": null,
  "request_id": "abc123"
}
```

### Detection Status Values

| Status | Condition | `detected` |
|--------|-----------|------------|
| `OK` | Exactly 1 KTP, no non-KTP | `true` |
| `no KTP detected` | No objects found | `false` |
| `only non-KTP detected` | No KTP, only non-KTP objects | `false` |
| `multiple KTPs detected` | More than 1 KTP, no non-KTP | `true` |
| `KTP and non-KTP detected` | Both KTP and non-KTP found | `true` |
| `ambiguous` | Any other case | `false` |

### Health Endpoints

#### GET /health/live

Liveness probe. Returns 200 if the process is alive. Use for K8s `livenessProbe`.

```json
{ "status": "alive", "version": "1.0.0" }
```

#### GET /health/ready

Readiness probe. Returns 200 if model is loaded and database is up. Use for K8s `readinessProbe`. Returns 503 if not ready.

```json
{
  "status": "ready",
  "checks": {
    "model": { "status": "up", "device": "cpu" },
    "database": { "status": "up" }
  }
}
```

#### GET /health

Full health check with detailed component info.

- `healthy`: Model loaded and DB up
- `degraded`: Model loaded but DB down
- `unhealthy`: Model not loaded

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "device_info": { "type": "CPU", "count": 4 },
  "timestamp": "2025-01-01T10:00:00",
  "checks": {
    "model": { "status": "up", "device": "cpu" },
    "database": { "status": "up" }
  }
}
```

### GET /

Service information and endpoint listing.

```json
{
  "message": "KTP Detection API",
  "description": "Detects ktp and non-ktp objects using YOLO model",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "liveness": "/health/live",
    "readiness": "/health/ready",
    "predict": "/v1/ocr_ktp_detection",
    "docs": "/docs"
  }
}
```

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_FILE_TYPE` | 400 | Uploaded file is not an image |
| `IMAGE_PROCESSING_FAILED` | 400 | Failed to decode/process image |
| `FILE_TOO_LARGE` | 413 | Upload exceeds the 1 MB size limit |
| `MODEL_NOT_LOADED` | 503 | YOLO model not loaded yet |
| `PREDICTION_FAILED` | 500 | Inference error during prediction |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

## Testing

### Unit & Integration Tests

```bash
# Run all unit and integration tests
uv run pytest tests/ -k "not e2e" -v --tb=short

# Run specific test file
uv run pytest tests/test_routes.py -v
```

### End-to-End Tests

E2E tests validate the full detection pipeline. Supports two modes:

```bash
# In-process mode (mocked predictor + DB, full API flow)
uv run pytest tests/e2e/ -m e2e -v

# Against a live running detector
E2E_BASE_URL=http://127.0.0.1:8060 uv run pytest tests/e2e/ -m e2e -v
```

E2E tests also run automatically during Docker build when using `--target e2e-test`.

## Docker Build

The Dockerfile uses a multi-stage build:

1. **base**: Installs system deps + Python deps + copies source code
2. **e2e-test**: Starts the detector, polls `/health/ready`, runs e2e tests. Build fails if tests fail.
3. **production**: Lean final image with `/health/live` for Docker HEALTHCHECK

```bash
docker build .                         # Builds production image (skips tests)
docker build --target e2e-test ...     # Runs e2e tests during build
```

## Interactive API Documentation

Once the server is running:

- Swagger UI: `http://localhost:8060/docs`
- ReDoc: `http://localhost:8060/redoc`

## License

Internal use only.
