# OCR Extract API

FastAPI service for extracting text from images using PaddleOCR.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/` | No | API information and endpoint listing |
| `GET` | `/health/live` | No | Liveness probe (always 200 if process alive) |
| `GET` | `/health/ready` | No | Readiness probe (OCR + DB must be up) |
| `GET` | `/health` | No | Full health check (healthy / degraded / unhealthy) |
| `POST` | `/v1/ocr_extract` | Yes | Extract text from an uploaded image |
| `GET` | `/docs` | No | Interactive Swagger UI |

### Health status logic

| OCR Engine | Database | Status | HTTP |
|------------|----------|--------|------|
| up | up | `healthy` | 200 |
| up | down | `degraded` | 200 |
| down | any | `unhealthy` | 503 |

### OCR Extract

Accepts a `multipart/form-data` upload with a single `file` field (JPEG or PNG).
Maximum upload size is **1 MB** (configurable via `ocr.image.max_size_mb`);
oversized uploads are rejected with `413` / `FILE_TOO_LARGE`.

```bash
curl -X POST http://localhost:8010/v1/ocr_extract \
  -H "X-API-Key: <key>" \
  -F "file=@document.jpg"
```

Response (envelope):
```json
{
  "status_code": 200,
  "status_desc": "OK",
  "message": "Success",
  "data": {
    "ocr_result": [
      [[[10, 10], [100, 10], [100, 30], [10, 30]], ["Sample Text", 0.95]]
    ],
    "processing_time": 1.23,
    "text_regions_count": 1
  },
  "error_code": null,
  "errors": null,
  "request_id": "abc-123"
}
```

### Error Codes

On failure the same envelope is returned with a non-null `error_code`:

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_FILE_TYPE` | 400 | Uploaded file is not an allowed image type (JPEG/PNG) |
| `IMAGE_PROCESSING_FAILED` | 400 | Failed to read/decode the uploaded image |
| `FILE_TOO_LARGE` | 413 | Upload exceeds the 1 MB size limit |
| `OCR_PROCESSING_FAILED` | 500 | OCR engine failed during extraction |
| `INTERNAL_ERROR` | 500 | Unexpected server error |
| `UNAUTHORIZED` | 401 | Missing or invalid `X-API-Key` header |
| `VALIDATION_ERROR` | 422 | Request validation failed (e.g. missing `file` field) |

## Configuration

All settings live in `config.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `server.host` | | `0.0.0.0` | Bind address |
| `server.port` | | `8010` | Server port (used by `python -m src.main`; Docker also runs on 8010) |
| `ocr.server_config_path` | | `PaddleOCR_hybrid.yaml` | GPU/server OCR config |
| `ocr.mobile_config_path` | | `PaddleOCR_mobile.yaml` | CPU/mobile OCR config |
| `ocr.image.max_size_mb` | | `10` | Max upload size (MB) |
| `ocr.image.allowed_types` | | `image/jpeg, image/png` | Accepted MIME types |
| `ocr.filter.width_threshold_ratio` | | `0.8` | Width filter ratio |
| `device.preferred` | | `auto` | Device mode (auto/gpu/cpu) |
| `device.force_cpu` | | `false` | Force CPU even if GPU available |

## Environment Variables

Loaded from a `.env` file (copy `.env.template` → `.env`). `${VAR}` placeholders in `config.yaml` are resolved at startup; a missing variable resolves to an empty string.

### Core

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `DATABASE_URL` | No (request logging is skipped if unset) | PostgreSQL (asyncpg) connection string for request logging. | `postgresql+asyncpg://user:pass@host:5432/bribrain_ocr_ktp` |
| `API_KEY` | Yes | Secret checked against the `X-API-Key` header on `/v1/ocr_extract`. Validated at startup. | `super-secret-key` |
| `LOG_ENCRYPTION_KEY` | Yes | AES-256-GCM key for encrypting the `payload` and `result` log columns. Base64 of 32 random bytes; the same key must be shared across all services. | `<base64-of-32-random-bytes>` |
| `APP_ENVIRO` | Yes | Controls model loading: `onprem` skips the GCS download (models are expected to be present in the image); any other value (e.g. `gcp`) downloads models from GCS at startup. An **empty value is NOT treated as `onprem`** — it falls through to the GCS path — so set it explicitly. | `gcp` |

### Cloud model storage (GCS) — required when `APP_ENVIRO` is not `onprem`

These are validated at startup whenever `APP_ENVIRO != onprem`.

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

### Elastic APM

APM is **enabled and live for this service**. The middleware activates whenever `ELASTIC_APM_SERVER_URL` is non-empty; leaving it blank disables APM.

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `ELASTIC_APM_SERVER_URL` | No | Elastic APM server URL. Empty disables APM. | `http://apm-server:8200` |
| `ELASTIC_APM_SERVICE_NAME` | No | Service name reported to APM. Defaults to `ocr-ktp-extractor`. | `ocr-ktp-extractor` |
| `ELASTIC_APM_SERVICE_VERSION` | No | Service version label reported to APM. | `1.0.0` |

## Log encryption (DPIA)

The PII columns of this service's log table `bribrain_ocr_ktp_extract` are encrypted at rest with **AES-256-GCM** (see `src/core/crypto.py`):

- **Encrypted (`BYTEA`):** `payload` and `result` — they hold the raw OCR text extracted from the KTP. Stored as ciphertext, not queryable as JSON.
- **Plaintext:** `error_message` (`TEXT`), `request_id`, `response_code`, `processing_time`, `created_at`.
- **Key:** `LOG_ENCRYPTION_KEY` — base64 of 32 random bytes, the **same key across all encrypting services** (`dgc_ext`, `dgc_irl`, `dgc_oct`, `dgc_pps`). Startup fails if it is missing while DB logging is enabled. Generate with `python -c "import os,base64;print(base64.b64encode(os.urandom(32)).decode())"`.
- **Stored format (`BYTEA`):** `version(1) || key_id(1) || nonce(12) || ciphertext+tag` — a fresh random 12-byte nonce per write and a 16-byte authentication tag (tamper-evident; wrong key or altered bytes fail to decrypt).
- **Migration:** run `schema_encryption.sql` (repo root) once per environment to convert these columns to `BYTEA` (it truncates existing rows).

This service only **writes** logs — decryption of the OCR result happens in the orchestrator on `GET /v1/get-ocr-result/{id}`.

## Testing

```bash
# Unit tests
uv run pytest tests/ -v --ignore=tests/e2e

# E2E tests (in-process, mocked OCR + DB)
uv run pytest tests/e2e/ -m e2e -v --no-cov

# E2E tests against live server
E2E_BASE_URL=http://127.0.0.1:8010 uv run pytest tests/e2e/ -m e2e -v --no-cov

# All tests
uv run pytest
```

## Docker

```bash
# Build (with e2e tests during build)
docker build \
  --secret id=env,src=.env \
  --target production \
  -t ocr-extract .

# Run
docker run -p 8010:8010 --env-file .env ocr-extract
```

The Dockerfile uses a three-stage build:
1. **base** - installs OS deps, Python packages, and application code
2. **e2e-test** - runs e2e tests (fails build on error)
3. **production** - lean runtime with health check on `/health/live`

## Model

- **Engine:** PaddleOCR2Pytorch (PP-OCRv5 detection + recognition running on PyTorch). A server/GPU flow and a mobile/CPU flow are selected based on the detected device.
- **Model source:** When `APP_ENVIRO` is not `onprem`, the detection and recognition weights are downloaded from GCS at startup. When `APP_ENVIRO=onprem`, the GCS download is skipped and models are expected to be baked into the image.
- **Warm-up:** Inference kernels are pre-compiled at startup (`warmup_ocr`) to avoid cold-start latency on the first request.
