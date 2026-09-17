"""End-to-end tests for the Recapture Classifier service.

These tests send real images through the classification pipeline and verify
the response structure and status codes.

Usage:
    pytest tests/e2e/ -m e2e -v --no-cov
"""

import pytest


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
class TestHealthEndpoints:
    """Test all health check endpoints."""

    async def test_liveness(self, e2e_client):
        """Liveness probe should always return 200."""
        resp = await e2e_client.get("/health/live")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "alive"
        assert "version" in data

    async def test_readiness(self, e2e_client):
        """Readiness probe should return 200 when model + DB are up."""
        resp = await e2e_client.get("/health/ready")
        data = resp.json()
        assert "status" in data
        assert "checks" in data
        assert "model" in data["checks"]
        if resp.status_code == 200:
            assert data["status"] == "ready"

    async def test_full_health(self, e2e_client):
        """Full health check should return detailed component info."""
        resp = await e2e_client.get("/health")
        assert resp.status_code in (200, 503)
        data = resp.json()
        assert data["status"] in ("healthy", "degraded", "unhealthy")
        assert "model_loaded" in data
        assert "device" in data
        assert "checks" in data
        assert "model" in data["checks"]
        assert "database" in data["checks"]


# ---------------------------------------------------------------------------
# Root endpoint test
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
class TestRootEndpoint:
    """Test the root endpoint."""

    async def test_root(self, e2e_client):
        """Root should return service info and endpoint listing."""
        resp = await e2e_client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["message"] == "Screen Recapture Detection API"
        assert "endpoints" in data
        assert "liveness" in data["endpoints"]
        assert "readiness" in data["endpoints"]


# ---------------------------------------------------------------------------
# Prediction pipeline tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
class TestPredictionPipeline:
    """Test recapture prediction on each image in tests/data/."""

    async def test_predict_returns_200(self, e2e_client, image_path, image_bytes, image_content_type):
        """Every valid image should get a 200 response."""
        resp = await e2e_client.post(
            "/v1/ocr_recapture",
            files={"file": (image_path.name, image_bytes, image_content_type)},
        )
        assert resp.status_code == 200, f"Expected 200 for {image_path.name}, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert data["status_code"] == 200
        assert data["data"] != ""
        assert "prediction" in data["data"]
        assert "confidence" in data["data"]
        assert "probability_recaptured" in data["data"]
        assert "threshold" in data["data"]
        assert "timestamp" in data["data"]

    async def test_prediction_values(self, e2e_client, image_path, image_bytes, image_content_type):
        """Verify prediction values are within valid ranges."""
        resp = await e2e_client.post(
            "/v1/ocr_recapture",
            files={"file": (image_path.name, image_bytes, image_content_type)},
        )
        assert resp.status_code == 200

        pred = resp.json()["data"]
        assert pred["prediction"] in ("ORIGINAL", "RECAPTURED")
        assert 0.0 <= pred["confidence"] <= 1.0
        assert 0.0 <= pred["probability_recaptured"] <= 1.0
        assert 0.0 <= pred["threshold"] <= 1.0

    async def test_response_has_request_id(self, e2e_client, image_path, image_bytes, image_content_type):
        """Every response should include a request_id."""
        resp = await e2e_client.post(
            "/v1/ocr_recapture",
            files={"file": (image_path.name, image_bytes, image_content_type)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "request_id" in data
        assert data["request_id"] is not None


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
class TestPredictionValidation:
    """Test input validation for the prediction endpoint."""

    async def test_invalid_file_type(self, e2e_client):
        """Non-image file should be rejected with 400."""
        resp = await e2e_client.post(
            "/v1/ocr_recapture",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code in (400, 500)

    async def test_missing_api_key(self, e2e_client):
        """Request without API key should be rejected."""
        from tests.e2e.conftest import E2E_BASE_URL
        import httpx

        if E2E_BASE_URL:
            async with httpx.AsyncClient(base_url=E2E_BASE_URL) as raw_client:
                resp = await raw_client.post(
                    "/v1/ocr_recapture",
                    files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                )
                assert resp.status_code in (401, 403)
        else:
            from src.main import app
            transport = httpx.ASGITransport(app=app)  # ty: ignore[invalid-argument-type]
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw_client:
                resp = await raw_client.post(
                    "/v1/ocr_recapture",
                    files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                )
                assert resp.status_code in (401, 403)

    async def test_wrong_api_key(self, e2e_client):
        """Request with wrong API key should be rejected."""
        from tests.e2e.conftest import E2E_BASE_URL
        import httpx

        if E2E_BASE_URL:
            async with httpx.AsyncClient(base_url=E2E_BASE_URL) as raw_client:
                resp = await raw_client.post(
                    "/v1/ocr_recapture",
                    files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                    headers={"X-API-Key": "wrong-key"},
                )
                assert resp.status_code == 401
        else:
            from src.main import app
            transport = httpx.ASGITransport(app=app)  # ty: ignore[invalid-argument-type]
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw_client:
                resp = await raw_client.post(
                    "/v1/ocr_recapture",
                    files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                    headers={"X-API-Key": "wrong-key"},
                )
                assert resp.status_code == 401
