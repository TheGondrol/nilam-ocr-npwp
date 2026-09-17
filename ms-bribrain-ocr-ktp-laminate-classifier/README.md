# Lamination Detection API

FastAPI service for detecting unlaminated (suspicious) vs laminated (normal) ID documents using a CNN deep learning model.

## Features

- **Binary Classification**: Detects LAMINATED (normal) vs UNLAMINATED (suspicious) documents
- **Custom CNN Architecture**: 7-layer CNN with 76x76 grayscale input
- **GPU/CPU Support**: Automatic CUDA GPU detection with CPU fallback
- **Model Optimization**: torch.compile() for CUDA, JIT tracing for CPU
- **Model Download**: Auto-downloads model from MinIO (on-prem) or GCS (cloud) at startup
- **Database Logging**: Async PostgreSQL logging for every request
- **Health Checks**: Liveness, readiness, and full health endpoints (K8s-ready)
- **API Key Authentication**: Prediction endpoint requires `X-API-Key` header
- **Request ID Tracking**: Middleware injects/propagates `x-request-id` headers
- **Thread Pool Inference**: Non-blocking inference via ThreadPoolExecutor
- **E2E Testing**: End-to-end tests run during Docker build against the real model
- **Type-Safe Configuration**: YAML-based configuration management

## Project Structure

```
ms-bribrain-ocr-ktp-laminate-classifier/
├── src/
│   ├── core/                            # Core functionality
│   │   ├── config.py                    # Configuration management (singleton)
│   │   ├── device.py                    # GPU/CPU detection
│   │   ├── logging.py                   # Logging setup with request_id context
│   │   └── validation.py               # Startup configuration validation
│   ├── api/                             # API layer
│   │   ├── dependencies.py             # API key verification
│   │   ├── routes.py                   # API endpoints
│   │   └── schemas.py                  # Pydantic response models
│   ├── middleware/
│   │   └── add_requestid.py            # Request ID injection middleware
│   ├── models/
│   │   └── architecture.py            # CNN model architecture
│   ├── schemas/
│   │   └── database_schema.py          # SQLAlchemy ORM models
│   ├── services/                        # Business logic
│   │   ├── predictor.py               # Model loading & prediction
│   │   ├── database_service.py         # Async database connection & logging
│   │   ├── minio_service.py            # MinIO model download (on-prem)
│   │   └── gcs_service.py             # GCS model download (cloud)
│   └── main.py                          # FastAPI application entry point
├── tests/
│   ├── unit/                            # Unit tests
│   ├── integration/                     # Integration tests
│   ├── e2e/                             # End-to-end tests
│   │   ├── conftest.py                  # E2E fixtures (in-process or live server)
│   │   └── test_e2e_laminate.py        # E2E prediction pipeline tests
│   └── data/                            # Test images
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
# Model settings
model:
  path: ./src/models/laminate_model.pth
  image_size: 76
  torch_compile: true

# Prediction threshold
prediction:
  threshold: 0.5

# Server
server:
  host: 0.0.0.0
  port: 8020
  thread_pool_workers: 4
  max_file_size_mb: 10

# Device
device:
  prefer_gpu: true
  force_cpu: false

# Database (async PostgreSQL)
database:
  pool_size: 5
  max_overflow: 10

# Logging
logging:
  log_to_database: true
  level: INFO
  file:
    enabled: true
    directory: ./logs
    filename: "ocr_laminate.log"
    max_bytes: 10485760          # 10MB
    backup_count: 5

# Model storage (on-prem)
minio:
  bucket: bribrain-webviewasset-dev
  object: ocr_ktp_models/laminate_model.pth

# Model storage (cloud)
gcs:
  bucket_name: "gc-bribrain-dev-gcs-iguru-01"
  prefix: "ocr_models/laminate_models/"
  model_prefix: "laminate_model"
```

Set `APP_ENVIRO=onprem` to download models from MinIO, otherwise GCS is used.

## Environment Variables

Loaded from a `.env` file (copy `.env.template` → `.env`). `${VAR}` placeholders in `config.yaml` are resolved at startup; a missing variable resolves to an empty string.

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `DATABASE_URL` | Yes | asyncpg PostgreSQL connection string for request logging | `postgresql+asyncpg://user:pass@host:5432/bribrain_ocr_ktp` |
| `API_KEY` | Yes | Secret expected in the `X-API-Key` header on the prediction endpoint | `change-me-strong-secret` |
| `APP_ENVIRO` | Yes | Model source selector: `onprem` uses MinIO, any other value (e.g. `gcp`) uses GCS. An empty value is **not** treated as `onprem` — set it explicitly. | `onprem` |
| `MINIO_ENDPOINT` | On-prem only | MinIO host:port the model is downloaded from | `minio.internal:9000` |
| `MINIO_ACCESS_KEY` | On-prem only | MinIO access key | `minio-access-key` |
| `MINIO_SECRET_KEY` | On-prem only | MinIO secret key | `minio-secret-key` |
| `MINIO_SECURE` | No | Use HTTPS to reach MinIO (`true`/`false`); defaults to `false` | `false` |
| `SA_TYPE` | Cloud only | GCS service-account type | `service_account` |
| `SA_PROJECT_ID` | Cloud only | GCP project ID | `gc-bribrain-dev` |
| `SA_PRIVATE_KEY` | Cloud only | Service-account private key. Contains escaped `\n` sequences — paste as-is, do **not** add extra surrounding quotes. | `-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n` |
| `SA_CLIENT_MAIL` | Cloud only | Service-account client email | `svc@gc-bribrain-dev.iam.gserviceaccount.com` |
| `SA_CLIENT_ID` | Cloud only | Service-account client ID | `1234567890` |
| `SA_AUTH_URI` | Cloud only | OAuth2 auth URI | `https://accounts.google.com/o/oauth2/auth` |
| `SA_TOKEN_URI` | Cloud only | OAuth2 token URI | `https://oauth2.googleapis.com/token` |
| `SA_AUTH_PROVIDER` | Cloud only | Auth provider x509 cert URL | `https://www.googleapis.com/oauth2/v1/certs` |
| `SA_CERT_URL` | Cloud only | Client x509 cert URL | `https://www.googleapis.com/robot/v1/metadata/x509/svc%40gc-bribrain-dev.iam.gserviceaccount.com` |
| `ELASTIC_APM_SERVER_URL` | No | Elastic APM server URL. Empty disables APM. | `https://apm.internal:8200` |
| `ELASTIC_APM_SERVICE_NAME` | No | APM service name | `ms-bribrain-ocr-ktp-laminate-classifier` |
| `ELASTIC_APM_SERVICE_VERSION` | No | APM service version | `1.0.0` |

On-prem (MinIO) variables are required only when `APP_ENVIRO=onprem`; cloud (GCS service account) variables are required otherwise. Startup validation (`src/core/validation.py`) fails fast if a required variable for the active environment is missing.

## Installation

### Using uv (recommended)

```bash
# Install dependencies
uv sync

# Run the service
uv run uvicorn src.main:app --host 0.0.0.0 --port 8020
```

### Using Docker

```bash
# Build the production image (skips e2e tests)
docker build -t ms-bribrain-ocr-ktp-laminate-classifier .

# Build with e2e tests (requires DB and model storage to be reachable)
docker build --target e2e-test --network host --secret id=env,src=.env -t laminate-test .

# Run the container
docker run -d \
  --name ms-bribrain-ocr-ktp-laminate-classifier \
  --env-file .env \
  -p 8020:8020 \
  --restart always \
  ms-bribrain-ocr-ktp-laminate-classifier

# View logs
docker logs -f ms-bribrain-ocr-ktp-laminate-classifier
```

## API Endpoints

The prediction endpoint requires the `X-API-Key` header. Health and root endpoints do not.

### POST /v1/ocr_laminate

Classify whether an ID document image is laminated or unlaminated.

**Request:**

- Content-Type: `multipart/form-data`
- Fields: `file` (image file, JPEG/PNG, max 1MB)
- Header: `X-API-Key: <your-api-key>`

**Response (200):**

```json
{
  "status_code": 200,
  "status_desc": "OK",
  "message": "Success",
  "data": {
    "filename": "image.jpg",
    "prediction": "LAMINATED",
    "prob": 0.1234,
    "threshold": 0.5,
    "timestamp": "2025-01-01T10:00:00"
  },
  "error_code": null,
  "errors": null,
  "request_id": "abc123"
}
```

### Classification Output

| Prediction | Class | Description |
|------------|-------|-------------|
| `LAMINATED` | 0 | Normal/legitimate laminated document |
| `UNLAMINATED` | 1 | Suspicious unlaminated document |

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
  "version": "1.0.0",
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
  "message": "Document Lamination Detection API",
  "description": "Detects unlaminated (suspicious) vs laminated (normal) ID documents",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "liveness": "/health/live",
    "readiness": "/health/ready",
    "predict": "/v1/ocr_laminate",
    "docs": "/docs"
  }
}
```

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `MODEL_NOT_LOADED` | 503 | Model not loaded yet |
| `INVALID_FILE_TYPE` | 400 | File is not an image |
| `FILE_TOO_LARGE` | 413 | File exceeds max size (1MB) |
| `IMAGE_PROCESSING_FAILED` | 400 | Failed to decode/process image |
| `PREDICTION_FAILED` | 500 | Inference error during prediction |

## Model Information

- **Architecture**: Custom CNN (`UnlaminatedCopyDetector`) — 7 conv layers, 2 max pooling, dropout(0.4), FC(1)
- **Input**: 76x76 grayscale images
- **Output**: Binary classification (LAMINATED vs UNLAMINATED)
- **Preprocessing**: Convert to grayscale → Resize(76x76) → ToTensor

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

E2E tests validate the full prediction pipeline. Supports two modes:

```bash
# In-process mode (mocked model + DB, full API flow)
uv run pytest tests/e2e/ -m e2e -v

# Against a live running service
E2E_BASE_URL=http://127.0.0.1:8020 uv run pytest tests/e2e/ -m e2e -v
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

- Swagger UI: `http://localhost:8020/docs`
- ReDoc: `http://localhost:8020/redoc`

## License

Internal use only.
