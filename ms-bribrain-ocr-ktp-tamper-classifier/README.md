# Document Tamper Detection API

FastAPI service for detecting tampered documents (text modifications) using a fine-tuned ViT (Vision Transformer) model.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/` | No | API information and endpoint listing |
| `GET` | `/health/live` | No | Liveness probe (always 200 if process alive) |
| `GET` | `/health/ready` | No | Readiness probe (model + DB must be up) |
| `GET` | `/health` | No | Full health check (healthy / degraded / unhealthy) |
| `POST` | `/v1/ocr_tamper` | Yes | Predict tampered vs authentic for a single image |
| `GET` | `/docs` | No | Interactive Swagger UI |

### Health status logic

| Model | Database | Status | HTTP |
|-------|----------|--------|------|
| up | up | `healthy` | 200 |
| up | down | `degraded` | 200 |
| down | any | `unhealthy` | 503 |

### Predict

```bash
curl -X POST http://localhost:8050/v1/ocr_tamper \
  -H "X-API-Key: <key>" \
  -F "file=@document.jpg"
```

The endpoint enforces a 1MB upload limit; larger files are rejected with `413 FILE_TOO_LARGE`.

Response (envelope):
```json
{
  "status_code": 200,
  "status_desc": "OK",
  "message": "Success",
  "data": {
    "filename": "document.jpg",
    "prediction": "authentic",
    "confidence": 0.9876,
    "probabilities": { "authentic": 0.9876, "tampered": 0.0124 },
    "threshold": 0.5,
    "timestamp": "2026-01-01T09:00:00"
  },
  "error_code": null,
  "errors": null,
  "request_id": "abc-123"
}
```

## Configuration

All settings live in `config.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `model.path` | | `./src/models/tamper_model/` | Path to model directory |
| `model.image_size` | | `320` | Input image size |
| `device.force_cpu` | | `true` | Force CPU even if GPU available |
| `prediction.threshold` | | `0.5` | Classification threshold |
| `server.host` | | `0.0.0.0` | Bind address |
| `server.port` | | `8000` | Server port for local `uvicorn` runs. The Docker image and `docker run -p 8050:8050` expose the service on **8050** — that is the deployed port. |
| `image.max_size_mb` | | `10` | Maximum uploaded image size in MB (enforced by the predict endpoint) |

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
| `ELASTIC_APM_SERVICE_NAME` | No | APM service name | `ms-bribrain-ocr-ktp-tamper-classifier` |
| `ELASTIC_APM_SERVICE_VERSION` | No | APM service version | `1.0.0` |

On-prem (MinIO) variables are required only when `APP_ENVIRO=onprem`; cloud (GCS service account) variables are required otherwise. Startup validation (`src/core/validation.py`) fails fast if a required variable for the active environment is missing.

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `FILE_TOO_LARGE` | 413 | Uploaded image exceeds the max size (1MB) |
| `INVALID_FILE` | 400 | File could not be read/decoded as an image |
| `INTERNAL_ERROR` | 500 | Unexpected inference or server error |

Early rejections (missing/invalid `X-API-Key` → 401/403, request validation → 422, unknown path → 404) are returned in the same response envelope with `error_code` values such as `UNAUTHORIZED`, `FORBIDDEN`, `VALIDATION_ERROR`, and `NOT_FOUND`.

## Testing

```bash
# Unit tests
uv run pytest test/unit/ -m unit -v

# E2E tests (in-process, mocked model + DB)
uv run pytest test/e2e/ -m e2e -v --no-cov

# E2E tests against live server
E2E_BASE_URL=http://127.0.0.1:8050 uv run pytest test/e2e/ -m e2e -v --no-cov

# All tests
uv run pytest
```

## Docker

```bash
# Build (with e2e tests during build)
docker build \
  --secret id=env,src=.env \
  --target production \
  -t ocr-tamper-detection .

# Run
docker run -p 8050:8050 --env-file .env ocr-tamper-detection
```

The Dockerfile uses a three-stage build:
1. **base** - installs OS deps and Python packages
2. **e2e-test** - runs e2e tests (fails build on error)
3. **production** - lean runtime with health check on `/health/live`

## Model

- **Base:** ariadnak/font-identifier (ViT-based)
- **Classes:** `authentic` (original document), `tampered` (text modifications)
- **Input:** 320x320 with padding to maintain aspect ratio
