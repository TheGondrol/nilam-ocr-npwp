# KTP Image Quality Assessment (IQA DL) API

FastAPI service for classifying KTP image quality by analyzing OCR text crop quality using a MobileNet deep learning model.

## Features

- **Crop-Level Classification**: Classifies each OCR text crop as good/bad quality
- **Overall Quality Label**: Determines overall image quality based on bad crop threshold
- **Smart Crop Filtering**: Filters crops to focus on text regions (width > height)
- **GPU/CPU Support**: Automatic CUDA GPU detection with CPU fallback
- **Model Optimization**: torch.compile() for CUDA acceleration
- **Model Download**: Auto-downloads model from MinIO (on-prem) or GCS (cloud) at startup
- **Database Logging**: Async PostgreSQL logging for every request
- **Health Checks**: Liveness, readiness, and full health endpoints (K8s-ready)
- **API Key Authentication**: Classification endpoint requires `X-API-Key` header
- **Request ID Tracking**: Middleware injects/propagates `x-request-id` headers
- **Thread Pool Inference**: Non-blocking inference via ThreadPoolExecutor
- **E2E Testing**: End-to-end tests run during Docker build against the real model
- **Type-Safe Configuration**: YAML-based configuration with Pydantic validation

## Project Structure

```
ms-bribrain-ocr-ktp-iqa-dl/
├── src/
│   ├── core/                            # Core functionality
│   │   ├── config.py                    # Pydantic config management (singleton)
│   │   ├── device.py                    # GPU/CPU detection
│   │   ├── logging.py                   # Logging setup with request_id context
│   │   └── validation.py               # Startup configuration validation
│   ├── api/                             # API layer
│   │   ├── dependencies.py             # API key verification
│   │   └── routes.py                   # API endpoints
│   ├── middleware/
│   │   └── add_requestid.py            # Request ID injection middleware
│   ├── models/
│   │   └── ml_model.py                 # MobileNet model loading & transforms
│   ├── schemas/
│   │   ├── api_schema.py               # Pydantic response models
│   │   └── database_schema.py          # SQLAlchemy ORM models
│   ├── services/                        # Business logic
│   │   ├── quality_service.py          # Quality classification logic
│   │   ├── database_services.py        # Async database connection & logging
│   │   ├── minio_service.py            # MinIO model download (on-prem)
│   │   └── gcs_service.py             # GCS model download (cloud)
│   └── main.py                          # FastAPI application entry point
├── tests/
│   ├── unit/                            # Unit tests
│   ├── e2e/                             # End-to-end tests
│   │   ├── conftest.py                  # E2E fixtures (in-process or live server)
│   │   └── test_e2e_iqa_dl.py          # E2E classification pipeline tests
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
# Model settings
model:
  path: ./src/models/qualitydl_model.pth
  num_classes: 2
  img_height: 64
  img_width: 320
  normalize_mean: [0.485, 0.456, 0.406]   # ImageNet normalization
  normalize_std: [0.229, 0.224, 0.225]

# Prediction thresholds
prediction:
  confidence_threshold: 0.67
  bad_crop_threshold: 4
  batch_max_files: 10

# Crop filtering
crop_filter:
  min_width_ratio: 1.0
  min_width: 0

# Server
server:
  host: 0.0.0.0
  port: 8100
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
  log_to_database: false
  level: INFO
  file:
    path: logs/ocr_quality.log
    max_bytes: 10485760          # 10MB
    backup_count: 5

# Model storage (on-prem)
minio:
  bucket: bribrain-webviewasset-dev
  object: ocr_ktp_models/quality_model.pth

# Model storage (cloud)
gcs:
  bucket_name: "gc-bribrain-dev-gcs-iguru-01"
  prefix: "ocr_models/qualitydl_models/"
  model_prefix: "qualitydl_model"
```

Set `APP_ENVIRO=onprem` to download models from MinIO, otherwise GCS is used.

## Environment Variables

Loaded from a `.env` file (copy `.env.template` → `.env`). `${VAR}` placeholders in `config.yaml` are resolved at startup; a missing variable resolves to an empty string.

Startup validation (`src/core/validation.py`) checks the required variables below and exits if any are missing. `DATABASE_URL` and `API_KEY` are always required. Storage credentials are required conditionally based on `APP_ENVIRO`: when `APP_ENVIRO=onprem` the `MINIO_*` variables are required (model `qualitydl_model.pth` is pulled from MinIO); otherwise the `SA_*` GCS service-account variables are required (model is pulled from GCS).

### Core

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `DATABASE_URL` | Yes | asyncpg PostgreSQL connection string used for request logging. | `postgresql+asyncpg://user:pass@host:5432/bribrain_ocr_ktp` |
| `API_KEY` | Yes | Expected value of the `X-API-Key` header on the classification endpoint. | `your-api-key` |
| `APP_ENVIRO` | No | Selects the model storage backend: `onprem` (MinIO) or `gcp` (GCS). Defaults to `onprem` when unset. Note: an empty value is NOT treated as `onprem` — anything other than `onprem` selects GCS. | `onprem` |

### On-prem (MinIO) — required when `APP_ENVIRO=onprem`

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `MINIO_ENDPOINT` | Yes (onprem) | MinIO host:port the model is downloaded from. | `minio.internal:9000` |
| `MINIO_ACCESS_KEY` | Yes (onprem) | MinIO access key. | `minioadmin` |
| `MINIO_SECRET_KEY` | Yes (onprem) | MinIO secret key. | `minio-secret` |
| `MINIO_SECURE` | No | Use HTTPS when connecting to MinIO. Defaults to `true` when unset. | `false` |

### Cloud (GCS service account) — required when `APP_ENVIRO` is not `onprem`

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `SA_TYPE` | Yes (gcp) | Service-account credential type. | `service_account` |
| `SA_PROJECT_ID` | Yes (gcp) | GCP project ID. | `gc-bribrain-dev` |
| `SA_PRIVATE_KEY` | Yes (gcp) | Service-account private key with `\n` escaped as literal `\n` (no surrounding quotes). | `-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n` |
| `SA_CLIENT_MAIL` | Yes (gcp) | Service-account client email. | `sa-name@project.iam.gserviceaccount.com` |
| `SA_CLIENT_ID` | Yes (gcp) | Service-account client ID. | `123456789012345678901` |
| `SA_AUTH_URI` | Yes (gcp) | OAuth2 auth URI. | `https://accounts.google.com/o/oauth2/auth` |
| `SA_TOKEN_URI` | Yes (gcp) | OAuth2 token URI. | `https://oauth2.googleapis.com/token` |
| `SA_AUTH_PROVIDER` | Yes (gcp) | Auth provider x509 cert URL. | `https://www.googleapis.com/oauth2/v1/certs` |
| `SA_CERT_URL` | Yes (gcp) | Client x509 cert URL. | `https://www.googleapis.com/robot/v1/metadata/x509/sa-name%40project.iam.gserviceaccount.com` |

### Elastic APM (optional)

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `ELASTIC_APM_SERVER_URL` | No | APM server URL. APM is disabled when empty/unset. | `http://apm-server:8200` |
| `ELASTIC_APM_SERVICE_NAME` | No | Service name reported to APM. | `ms-bribrain-ocr-ktp-iqa-dl` |
| `ELASTIC_APM_SERVICE_VERSION` | No | Service version reported to APM. | `1.0.0` |

## Installation

### Using uv (recommended)

```bash
# Install dependencies
uv pip install --system -r pyproject.toml

# Run the service
uv run uvicorn src.main:app --host 0.0.0.0 --port 8100
```

### Using Docker

```bash
# Build the production image (skips e2e tests)
docker build -t ms-bribrain-ocr-ktp-iqa-dl .

# Build with e2e tests (requires DB and model storage to be reachable)
docker build --target e2e-test --network host --secret id=env,src=.env -t iqa-dl-test .

# Run the container
docker run -d \
  --name ms-bribrain-ocr-ktp-iqa-dl \
  --env-file .env \
  -p 8100:8100 \
  --gpus all \
  --restart always \
  ms-bribrain-ocr-ktp-iqa-dl

# View logs
docker logs -f ms-bribrain-ocr-ktp-iqa-dl
```

## API Endpoints

The classification endpoint requires the `X-API-Key` header. Health and root endpoints do not.

### POST /v1/ocr_qualitydl

Classify image quality based on OCR text crop quality.

**Request:**

- Content-Type: `multipart/form-data`
- Fields:
  - `file` (image file, JPEG/PNG)
  - `crops` (JSON string of crop objects with bbox, text, confidence)
- Header: `X-API-Key: <your-api-key>`

**Response (200):**

```json
{
  "status_code": 200,
  "status_desc": "OK",
  "message": "Success",
  "data": {
    "label": "good",
    "num_bad": 0,
    "num_filtered": 5,
    "total_crops": 8,
    "num_failed_crops": 0,
    "bad_crop_threshold": 4,
    "predictions": [
      {
        "bbox": [[10, 20], [200, 20], [200, 50], [10, 50]],
        "prediction": 1,
        "label": "good",
        "score": 0.95
      }
    ]
  },
  "error_code": null,
  "errors": null,
  "request_id": "abc123"
}
```

### Classification Output

| Label | Description |
|-------|-------------|
| `good` | Image quality is acceptable (fewer than threshold bad crops) |
| `bad` | Image quality is poor (>= threshold bad crops) |

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
  "name": "KTP Image Quality Classifier API",
  "description": "Classifies KTP image quality by analyzing OCR text crop quality",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "liveness": "/health/live",
    "readiness": "/health/ready",
    "filter": "/v1/ocr_qualitydl (POST)"
  }
}
```

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `MODEL_NOT_LOADED` | 503 | Model not loaded yet |
| `INVALID_FILE_TYPE` | 400 | File is not an image |
| `INVALID_INPUT` | 400 | Invalid JSON, crop data, or undecodable image |
| `FILE_TOO_LARGE` | 413 | Uploaded image exceeds the 1 MB size limit (`image.max_size_mb`) |
| `PREDICTION_FAILED` | 500 | Inference error during classification |

## Model Information

- **Architecture**: MobileNet with custom classification head
- **Input**: RGB crop images, 320x64 pixels
- **Output**: Binary classification (good vs bad quality) per crop
- **Overall Logic**: Image is "bad" if >= `bad_crop_threshold` crops are classified as bad
- **Preprocessing**: Resize to model input size → normalize with ImageNet mean/std

## Testing

### Unit Tests

```bash
# Run all unit tests
uv run pytest tests/unit/ -v --tb=short
```

### End-to-End Tests

E2E tests validate the full classification pipeline. Supports two modes:

```bash
# In-process mode (mocked model + DB, full API flow)
uv run pytest tests/e2e/ -m e2e -v

# Against a live running service
E2E_BASE_URL=http://127.0.0.1:8100 uv run pytest tests/e2e/ -m e2e -v
```

E2E tests also run automatically during Docker build when using `--target e2e-test`.

## Docker Build

The Dockerfile uses a multi-stage build:

1. **base**: Installs system deps + Python deps + copies source code (NVIDIA CUDA runtime)
2. **e2e-test**: Starts the service, polls `/health/ready`, runs e2e tests. Build fails if tests fail.
3. **production**: Lean final image with `/health/live` for Docker HEALTHCHECK

```bash
docker build .                         # Builds production image (skips tests)
docker build --target e2e-test ...     # Runs e2e tests during build
```

## Interactive API Documentation

Once the server is running:

- Swagger UI: `http://localhost:8100/docs`
- ReDoc: `http://localhost:8100/redoc`

## License

Internal use only.
