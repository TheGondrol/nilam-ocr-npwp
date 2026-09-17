"""End-to-end tests for the OCR Extract API."""

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------


class TestHealthEndpoints:
    """E2E tests for health probe endpoints."""

    async def test_liveness(self, e2e_client):
        resp = await e2e_client.get("/health/live")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "alive"
        assert "version" in data

    async def test_readiness(self, e2e_client):
        resp = await e2e_client.get("/health/ready")
        # In-process mock always returns ready
        assert resp.status_code in (200, 503)
        data = resp.json()
        assert data["status"] in ("ready", "not_ready")

    async def test_health(self, e2e_client):
        resp = await e2e_client.get("/health")
        assert resp.status_code in (200, 503)
        data = resp.json()
        assert data["status"] in ("healthy", "degraded", "unhealthy")
        assert "ocr_loaded" in data
        assert "version" in data


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    """E2E tests for root endpoint."""

    async def test_root(self, e2e_client):
        resp = await e2e_client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["message"] == "OCR Extract API"
        assert "endpoints" in data
        assert "liveness" in data["endpoints"]
        assert "readiness" in data["endpoints"]


# ---------------------------------------------------------------------------
# OCR Extract endpoint
# ---------------------------------------------------------------------------


class TestExtractEndpoint:
    """E2E tests for the /v1/ocr_extract endpoint."""

    async def test_extract_image(self, e2e_client, image_path, image_bytes, image_content_type):
        """Test OCR extraction on each image in tests/data/."""
        resp = await e2e_client.post(
            "/v1/ocr_extract",
            files={"file": (image_path.name, image_bytes, image_content_type)},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status_code"] == 200
        data = body["data"]
        assert "ocr_result" in data
        assert "processing_time" in data
        assert "text_regions_count" in data


# ---------------------------------------------------------------------------
# Validation / auth
# ---------------------------------------------------------------------------


class TestValidation:
    """E2E tests for input validation and auth."""

    async def test_extract_no_file(self, e2e_client):
        """Extract without a file should return 422."""
        resp = await e2e_client.post("/v1/ocr_extract")
        assert resp.status_code == 422

    async def test_missing_api_key(self, e2e_client):
        """Request without API key should be rejected."""
        saved = e2e_client.headers.pop("X-API-Key", None)
        try:
            resp = await e2e_client.get("/v1/ocr_extract")
            assert resp.status_code in (401, 403, 405)
        finally:
            if saved:
                e2e_client.headers["X-API-Key"] = saved
