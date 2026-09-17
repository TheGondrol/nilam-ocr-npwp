"""
End-to-end tests for the IQA DL (Image Quality Assessment) service.

These tests send real images through the classification pipeline and verify
the response structure and status codes.

When running against a live server (E2E_BASE_URL set), the tests use the
actual MobileNet model. In in-process mode, the model is mocked but the
full API flow is exercised.

Usage:
    pytest tests/e2e/ -m e2e -v --no-cov
"""

import json

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
        assert resp.status_code == 200
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
        assert "name" in data
        assert "endpoints" in data
        assert "liveness" in data["endpoints"]
        assert "readiness" in data["endpoints"]


# ---------------------------------------------------------------------------
# Filter (classification) pipeline tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
class TestFilterPipeline:
    """Test quality classification on each image in tests/data/."""

    async def test_filter_returns_200(self, e2e_client, image_path, image_bytes, image_content_type, sample_crops):
        """Every valid image with valid crops should get a 200 response."""
        resp = await e2e_client.post(
            "/filter",
            files={"file": (image_path.name, image_bytes, image_content_type)},
            data={"crops": sample_crops},
        )
        assert resp.status_code == 200, f"Expected 200 for {image_path.name}, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert data["status_code"] == 200
        assert data["data"] != ""
        assert "label" in data["data"]
        assert "num_bad" in data["data"]
        assert "num_filtered" in data["data"]
        assert "total_crops" in data["data"]
        assert "predictions" in data["data"]

    async def test_classification_values(self, e2e_client, image_path, image_bytes, image_content_type, sample_crops):
        """Verify classification values are within valid ranges."""
        resp = await e2e_client.post(
            "/filter",
            files={"file": (image_path.name, image_bytes, image_content_type)},
            data={"crops": sample_crops},
        )
        assert resp.status_code == 200

        result = resp.json()["data"]
        assert result["label"] in ("good", "bad")
        assert result["num_bad"] >= 0
        assert result["num_filtered"] >= 0
        assert result["total_crops"] >= 0
        assert result["num_failed_crops"] >= 0
        assert result["bad_crop_threshold"] >= 0

        for pred in result["predictions"]:
            assert pred["prediction"] in (0, 1)
            assert pred["label"] in ("good", "bad")

    async def test_response_has_request_id(self, e2e_client, image_path, image_bytes, image_content_type, sample_crops):
        """Every response should include a request_id."""
        resp = await e2e_client.post(
            "/filter",
            files={"file": (image_path.name, image_bytes, image_content_type)},
            data={"crops": sample_crops},
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
class TestFilterValidation:
    """Test input validation for the filter endpoint."""

    async def test_invalid_file_type(self, e2e_client, sample_crops):
        """Non-image file should be rejected with 400."""
        resp = await e2e_client.post(
            "/filter",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            data={"crops": sample_crops},
        )
        assert resp.status_code in (400, 500)

    async def test_invalid_crops_json(self, e2e_client):
        """Invalid JSON in crops should be rejected with 400."""
        resp = await e2e_client.post(
            "/filter",
            files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
            data={"crops": "not json"},
        )
        data = resp.json()
        assert data["status_code"] == 400
        assert data["error_code"] == "INVALID_INPUT"

    async def test_crops_not_a_list(self, e2e_client):
        """Crops that aren't a list should be rejected with 400."""
        resp = await e2e_client.post(
            "/filter",
            files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
            data={"crops": json.dumps({"not": "a list"})},
        )
        data = resp.json()
        assert data["status_code"] == 400
        assert data["error_code"] == "INVALID_INPUT"

    async def test_missing_api_key(self, e2e_client):
        """Request without API key should be rejected."""
        from tests.e2e.conftest import E2E_BASE_URL
        import httpx

        if E2E_BASE_URL:
            async with httpx.AsyncClient(base_url=E2E_BASE_URL) as raw_client:
                resp = await raw_client.post(
                    "/filter",
                    files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                    data={"crops": "[]"},
                )
                assert resp.status_code in (401, 403)
        else:
            from src.main import app
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw_client:
                resp = await raw_client.post(
                    "/filter",
                    files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                    data={"crops": "[]"},
                )
                assert resp.status_code == 403

    async def test_wrong_api_key(self, e2e_client):
        """Request with wrong API key should be rejected."""
        from tests.e2e.conftest import E2E_BASE_URL
        import httpx

        if E2E_BASE_URL:
            async with httpx.AsyncClient(base_url=E2E_BASE_URL) as raw_client:
                resp = await raw_client.post(
                    "/filter",
                    files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                    data={"crops": "[]"},
                    headers={"X-API-Key": "wrong-key"},
                )
                assert resp.status_code == 401
        else:
            from src.main import app
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw_client:
                resp = await raw_client.post(
                    "/filter",
                    files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                    data={"crops": "[]"},
                    headers={"X-API-Key": "wrong-key"},
                )
                assert resp.status_code == 401
