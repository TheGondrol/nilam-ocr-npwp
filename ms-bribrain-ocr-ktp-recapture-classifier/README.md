# Screen Recapture Detection API

FastAPI service for detecting screen recaptured photos vs original captures using a ResNet-50 deep learning model.

## Features

- **Binary Classification**: Detects ORIGINAL (legitimate) vs RECAPTURED (suspicious) images
- **ResNet-50 Architecture**: Pre-trained ResNet-50 with 224x224 padded input (aspect-preserving, configurable via `model.resize_mode`)
- **GPU/CPU Support**: Automatic CUDA GPU detection with CPU fallback
- **Model Optimization**: torch.compile() for CUDA, configurable compile modes
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
ms-bribrain-ocr-ktp-recapture-classifier/
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
│   │   └── ml_model.py                 # ResNet-50 model loading & transform
│   ├── schemas/
│   │   ├── api_schema.py               # API response models
│   │   └── database_schema.py          # SQLAlchemy ORM models
│   ├── services/                        # Business logic
│   │   ├── recapture_service.py        # Model globals & prediction logic
│   │   ├── database_services.py        # Async database connection & logging
│   │   ├── minio_service.py            # MinIO model download (on-prem)
│   │   └── gcs_service.py             # GCS model download (cloud)
│   └── main.py                          # FastAPI application entry point
├── tests/
│   ├── unit/                            # Unit tests
│   ├── e2e/                             # End-to-end tests
│   │   ├── conftest.py                  # E2E fixtures (in-process or live server)
│   │   └── test_e2e_recapture.py       # E2E prediction pipeline tests
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
  path: ./src/models/recapture_model.pth
  num_classes: 2
  normalize_size: 512
  crop_size: 224
  use_compile: true

# Prediction threshold
prediction:
  threshold: 0.5

# Server
server:
  host: 0.0.0.0
  port: 8000
  thread_pool_workers: 4

# Device
device:
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
    path: "logs/ocr_recapture.log"
    max_bytes: 10485760          # 10MB
    backup_count: 5

# Model storage (on-prem)
minio:
  bucket: bribrain-webviewasset-dev
  object: ocr_ktp_models/recapture_model.pth

# Model storage (cloud)
gcs:
  bucket_name: "gc-bribrain-dev-gcs-iguru-01"
  prefix: "ocr_models/recapture_models/"
  model_prefix: "recapture_model"
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
| `ELASTIC_APM_SERVICE_NAME` | No | APM service name | `ms-bribrain-ocr-ktp-recapture-classifier` |
| `ELASTIC_APM_SERVICE_VERSION` | No | APM service version | `1.0.0` |

On-prem (MinIO) variables are required only when `APP_ENVIRO=onprem`; cloud (GCS service account) variables are required otherwise. Startup validation (`src/core/validation.py`) fails fast if a required variable for the active environment is missing.

## Installation

### Using uv (recommended)

```bash
# Install dependencies
uv sync

# Run the service
uv run uvicorn src.main:app --host 0.0.0.0 --port 8030
```

### Using Docker

```bash
# Build the production image (skips e2e tests)
docker build -t ms-bribrain-ocr-ktp-recapture-classifier .

# Build with e2e tests (requires DB and model storage to be reachable)
docker build --target e2e-test --network host --secret id=env,src=.env -t recapture-test .

# Run the container
docker run -d \
  --name ms-bribrain-ocr-ktp-recapture-classifier \
  --env-file .env \
  -p 8030:8030 \
  --restart always \
  ms-bribrain-ocr-ktp-recapture-classifier

# View logs
docker logs -f ms-bribrain-ocr-ktp-recapture-classifier
```

## API Endpoints

The prediction endpoint requires the `X-API-Key` header. Health and root endpoints do not.

### POST /v1/ocr_recapture

Classify whether an image is a screen recapture or original capture.

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
    "prediction": "ORIGINAL",
    "confidence": 0.9234,
    "probability_recaptured": 0.0766,
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
| `ORIGINAL` | 0 | Normal/legitimate original capture (training folder: `good`) |
| `RECAPTURED` | 1 | Suspicious screen recaptured photo (training folder: `recaptured`) |

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
  "message": "Screen Recapture Detection API",
  "description": "Detects recaptured screen photos vs original captures",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "liveness": "/health/live",
    "readiness": "/health/ready",
    "predict": "/v1/ocr_recapture",
    "docs": "/docs"
  }
}
```

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `MODEL_NOT_LOADED` | 503 | Model not loaded yet |
| `INVALID_FILE_TYPE` | 400 | File is not an image |
| `FILE_TOO_LARGE` | 413 | Uploaded image exceeds the max size (1MB) |
| `IMAGE_PROCESSING_FAILED` | 400 | Failed to decode/process image |
| `INTERNAL_ERROR` | 500 | Inference error during prediction |

## Image Preprocessing

The API automatically applies the same preprocessing used during training. The
resize step is governed by `model.resize_mode` in `config.yaml` (default `pad`):

1. **Resize**: `pad` mode — aspect-preserving resize so the longest side equals
   `model.crop_size` (224), then white-padding (`model.pad_fill`, default 255)
   the shorter side back up to a square. The full image is preserved with no
   cropping and no aspect-ratio distortion. Alternative modes `crop` (Resize
   short=256 + CenterCrop 224) and `squash` (direct Resize to 224x224) are
   available for backward compatibility — see `src/models/ml_model.py:get_transform`.
2. **Normalization**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Testing

### Unit Tests

```bash
# Run all unit tests
uv run pytest tests/unit/ -v --tb=short
```

### End-to-End Tests

E2E tests validate the full prediction pipeline. Supports two modes:

```bash
# In-process mode (mocked model + DB, full API flow)
uv run pytest tests/e2e/ -m e2e -v

# Against a live running service
E2E_BASE_URL=http://127.0.0.1:8030 uv run pytest tests/e2e/ -m e2e -v
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

- Swagger UI: `http://localhost:8030/docs`
- ReDoc: `http://localhost:8030/redoc`

## License

Internal use only.
