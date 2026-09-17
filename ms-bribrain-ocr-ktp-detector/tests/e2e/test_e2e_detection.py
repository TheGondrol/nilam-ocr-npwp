"""
End-to-end tests for the KTP Detector service.

These tests send real images through the detection pipeline and verify
the response structure and status codes.

When running against a live server (E2E_BASE_URL set), the tests use the
actual YOLO model. In in-process mode, the predictor is mocked but the
full API flow is exercised.

Usage:
    pytest tests/e2e/ -m e2e -v --no-cov
"""

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Expected outcomes per image
# ---------------------------------------------------------------------------

# Images where we expect a KTP to be detected (good_data, unlaminated, etc. are all real KTPs)
EXPECTED_DETECTED_IMAGES = {
    "good_data.png",
    "unlaminated.jpeg",
    "recapture.jpeg",
    "graycopy.jpeg",
    "tamper.jpeg",
}

# Images where a non-KTP object is expected
EXPECTED_NOT_DETECTED_IMAGES = {
    "classifier.jpeg",
}


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
        # In live mode model should be up; in-process mode it's mocked as up
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
        assert "device_info" in data
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
        assert data["message"] == "KTP Detection API"
        assert "endpoints" in data
        assert "liveness" in data["endpoints"]
        assert "readiness" in data["endpoints"]


# ---------------------------------------------------------------------------
# Detection pipeline tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
class TestDetectionPipeline:
    """Test KTP detection on each image in tests/data/."""

    async def test_detect_returns_200(self, e2e_client, image_path, image_bytes, image_content_type):
        """Every valid image should get a 200 response (detection succeeds regardless of result)."""
        resp = await e2e_client.post(
            "/v1/ocr_ktp_detection",
            files={"file": (image_path.name, image_bytes, image_content_type)},
        )
        assert resp.status_code == 200, f"Expected 200 for {image_path.name}, got {resp.status_code}: {resp.text}"

        data = resp.json()
        assert data["status_code"] == 200
        assert data["data"] != ""
        assert "detected" in data["data"]
        assert "num_detected" in data["data"]
        assert "detections" in data["data"]
        assert "status" in data["data"]
        assert "timestamp" in data["data"]

    async def test_detection_result_structure(self, e2e_client, image_path, image_bytes, image_content_type):
        """Verify detection result has proper structure for each detection."""
        resp = await e2e_client.post(
            "/v1/ocr_ktp_detection",
            files={"file": (image_path.name, image_bytes, image_content_type)},
        )
        assert resp.status_code == 200

        detections = resp.json()["data"]["detections"]
        for det in detections:
            assert "class_id" in det
            assert "class_name" in det
            assert det["class_name"] in ("ktp", "non-ktp")
            assert "confidence" in det
            assert 0.0 <= det["confidence"] <= 1.0
            assert "bbox" in det
            bbox = det["bbox"]
            assert all(k in bbox for k in ("x1", "y1", "x2", "y2"))

    async def test_response_has_request_id(self, e2e_client, image_path, image_bytes, image_content_type):
        """Every response should include a request_id."""
        resp = await e2e_client.post(
            "/v1/ocr_ktp_detection",
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
class TestDetectionValidation:
    """Test input validation for the detection endpoint."""

    async def test_invalid_file_type(self, e2e_client):
        """Non-image file should be rejected with 400."""
        resp = await e2e_client.post(
            "/v1/ocr_ktp_detection",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["error_code"] == "INVALID_FILE_TYPE"

    async def test_missing_api_key(self, e2e_client):
        """Request without API key should be rejected."""
        from tests.e2e.conftest import E2E_BASE_URL
        import httpx

        if E2E_BASE_URL:
            # Live server mode — send request without API key
            async with httpx.AsyncClient(base_url=E2E_BASE_URL) as raw_client:
                resp = await raw_client.post(
                    "/v1/ocr_ktp_detection",
                    files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                )
                assert resp.status_code in (401, 403)
        else:
            # In-process: httpx ASGITransport doesn't send Security headers the same way.
            # Use a fresh client without the X-API-Key header.
            from src.main import app
            transport = httpx.ASGITransport(app=app)  # ty: ignore[invalid-argument-type]
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw_client:
                resp = await raw_client.post(
                    "/v1/ocr_ktp_detection",
                    files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                )
                assert resp.status_code == 403

    async def test_wrong_api_key(self, e2e_client):
        """Request with wrong API key should be rejected."""
        from tests.e2e.conftest import E2E_BASE_URL
        import httpx

        if E2E_BASE_URL:
            async with httpx.AsyncClient(base_url=E2E_BASE_URL) as raw_client:
                resp = await raw_client.post(
                    "/v1/ocr_ktp_detection",
                    files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                    headers={"X-API-Key": "wrong-key"},
                )
                assert resp.status_code == 401
        else:
            from src.main import app
            transport = httpx.ASGITransport(app=app)  # ty: ignore[invalid-argument-type]
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw_client:
                resp = await raw_client.post(
                    "/v1/ocr_ktp_detection",
                    files={"file": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
                    headers={"X-API-Key": "wrong-key"},
                )
                assert resp.status_code == 401
