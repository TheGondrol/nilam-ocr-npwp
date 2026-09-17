"""
End-to-end tests for the IQA Rule-Based service.

These tests send real images through the quality check pipeline and verify
the response structure and status codes.

When running against a live server (E2E_BASE_URL set), the tests use the
actual rule-based detectors (blur, glare, rotation). In in-process mode,
the quality service is mocked but the full API flow is exercised.

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
        """Readiness probe should return 200 when DB is up."""
        resp = await e2e_client.get("/health/ready")
        data = resp.json()
        assert "status" in data
        assert "checks" in data
        assert "database" in data["checks"]
        if resp.status_code == 200:
            assert data["status"] == "ready"

    async def test_full_health(self, e2e_client):
        """Full health check should return detailed component info."""
        resp = await e2e_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "unhealthy")
        assert "version" in data
        assert "checks" in data
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
        assert "service" in data
        assert "endpoints" in data
        assert "liveness" in data["endpoints"]
        assert "readiness" in data["endpoints"]


# ---------------------------------------------------------------------------
# OCR quality pipeline tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
class TestOcrQualityPipeline:
    """Test OCR quality check on each image in tests/data/."""

    async def test_quality_returns_200(self, e2e_client, image_path, image_bytes, image_content_type, sample_ocr_result):
        """Every valid image with valid OCR result should get a 200 response."""
        resp = await e2e_client.post(
            "/v1/ocr_quality",
            files={"file": (image_path.name, image_bytes, image_content_type)},
            data={"ocr_result": sample_ocr_result},
        )
        assert resp.status_code == 200, f"Expected 200 for {image_path.name}, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert data["status_code"] == 200
        assert data["data"] != ""
        assert "low_confidence" in data["data"]
        assert "is_blurry" in data["data"]
        assert "is_glare" in data["data"]
        assert "is_rotated" in data["data"]

    async def test_quality_values_are_booleans(self, e2e_client, image_path, image_bytes, image_content_type, sample_ocr_result):
        """Verify quality check values are booleans."""
        resp = await e2e_client.post(
            "/v1/ocr_quality",
            files={"file": (image_path.name, image_bytes, image_content_type)},
            data={"ocr_result": sample_ocr_result},
        )
        assert resp.status_code == 200

        result = resp.json()["data"]
        assert isinstance(result["low_confidence"], bool)
        assert isinstance(result["is_blurry"], bool)
        assert isinstance(result["is_glare"], bool)
        assert isinstance(result["is_rotated"], bool)

    async def test_response_has_request_id(self, e2e_client, image_path, image_bytes, image_content_type, sample_ocr_result):
        """Every response should include a request_id."""
        resp = await e2e_client.post(
            "/v1/ocr_quality",
            files={"file": (image_path.name, image_bytes, image_content_type)},
            data={"ocr_result": sample_ocr_result},
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
class TestOcrQualityValidation:
    """Test input validation for the OCR quality endpoint."""

    async def test_invalid_ocr_json(self, e2e_client):
        """Invalid JSON in ocr_result should be rejected with 400."""
        resp = await e2e_client.post(
            "/v1/ocr_quality",
            files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
            data={"ocr_result": "not valid json{{{"},
        )
        data = resp.json()
        assert data["status_code"] == 400
        assert data["error_code"] == "INVALID_INPUT"

    async def test_empty_ocr_result(self, e2e_client, image_path, image_bytes, image_content_type):
        """Empty OCR result list should still return 200."""
        resp = await e2e_client.post(
            "/v1/ocr_quality",
            files={"file": (image_path.name, image_bytes, image_content_type)},
            data={"ocr_result": json.dumps([])},
        )
        assert resp.status_code == 200

    async def test_missing_api_key(self, e2e_client):
        """Request without API key should be rejected."""
        from tests.e2e.conftest import E2E_BASE_URL
        import httpx

        if E2E_BASE_URL:
            async with httpx.AsyncClient(base_url=E2E_BASE_URL) as raw_client:
                resp = await raw_client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                    data={"ocr_result": "[]"},
                )
                assert resp.status_code in (401, 403)
        else:
            from src.main import app
            transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw_client:
                resp = await raw_client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                    data={"ocr_result": "[]"},
                )
                assert resp.status_code == 403

    async def test_wrong_api_key(self, e2e_client):
        """Request with wrong API key should be rejected."""
        from tests.e2e.conftest import E2E_BASE_URL
        import httpx

        if E2E_BASE_URL:
            async with httpx.AsyncClient(base_url=E2E_BASE_URL) as raw_client:
                resp = await raw_client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                    data={"ocr_result": "[]"},
                    headers={"X-API-Key": "wrong-key"},
                )
                assert resp.status_code == 401
        else:
            from src.main import app
            transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw_client:
                resp = await raw_client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                    data={"ocr_result": "[]"},
                    headers={"X-API-Key": "wrong-key"},
                )
                assert resp.status_code == 401
