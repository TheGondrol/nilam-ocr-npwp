"""End-to-end tests for the Postprocessor service.

These tests exercise the full API flow including health checks and
the OCR postprocessing pipeline.

Usage:
    pytest tests/e2e/test_e2e_postprocess.py -m e2e -v --no-cov
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
        assert resp.status_code in (200, 503)
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
        assert data["message"] == "OCR Postprocess API"
        assert "endpoints" in data
        assert "liveness" in data["endpoints"]
        assert "readiness" in data["endpoints"]


# ---------------------------------------------------------------------------
# Prediction pipeline tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
class TestPostprocessPipeline:
    """Test OCR postprocessing pipeline."""

    async def test_postprocess_returns_200(self, e2e_client, complete_jakarta_ktp):
        """Valid OCR data should get a 200 response."""
        resp = await e2e_client.post(
            "/v1/ocr_postprocess",
            json={"ocr_text": json.dumps(complete_jakarta_ktp)},
        )
        assert resp.status_code == 200

        data = resp.json()
        assert data["status_code"] == 200
        assert data["data"] != ""
        assert "ocr_result" in data["data"]
        assert "nik_image_box" in data["data"]

    async def test_postprocess_extracts_fields(self, e2e_client, complete_jakarta_ktp):
        """Verify extracted fields contain expected data."""
        resp = await e2e_client.post(
            "/v1/ocr_postprocess",
            json={"ocr_text": json.dumps(complete_jakarta_ktp)},
        )
        assert resp.status_code == 200

        result = resp.json()["data"]["ocr_result"]
        assert isinstance(result, dict)
        assert result.get("nik") == "3174012801900001"
        assert result.get("nama") == "AHMAD FAUZI"

    async def test_response_has_request_id(self, e2e_client, complete_jakarta_ktp):
        """Every response should include a request_id."""
        resp = await e2e_client.post(
            "/v1/ocr_postprocess",
            json={"ocr_text": json.dumps(complete_jakarta_ktp)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "request_id" in data
        assert data["request_id"] is not None

    async def test_multiple_ktp_samples(self, e2e_client, complete_surabaya_ktp, complete_bandung_ktp):
        """Multiple KTP samples should all process successfully."""
        for ktp in [complete_surabaya_ktp, complete_bandung_ktp]:
            resp = await e2e_client.post(
                "/v1/ocr_postprocess",
                json={"ocr_text": json.dumps(ktp)},
            )
            assert resp.status_code == 200
            assert "ocr_result" in resp.json()["data"]


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
class TestPostprocessValidation:
    """Test input validation for the postprocess endpoint."""

    async def test_invalid_json_format(self, e2e_client):
        """Invalid JSON in ocr_text should be rejected with 400."""
        resp = await e2e_client.post(
            "/v1/ocr_postprocess",
            json={"ocr_text": "{invalid json data"},
        )
        assert resp.status_code == 400
        assert resp.json()["error_code"] is not None

    async def test_missing_ocr_text_field(self, e2e_client):
        """Request without ocr_text should be rejected with 422."""
        resp = await e2e_client.post(
            "/v1/ocr_postprocess",
            json={},
        )
        assert resp.status_code == 422

    async def test_missing_api_key(self, e2e_client):
        """Request without API key should be rejected."""
        from tests.e2e.conftest import E2E_BASE_URL
        import httpx

        if E2E_BASE_URL:
            async with httpx.AsyncClient(base_url=E2E_BASE_URL) as raw_client:
                resp = await raw_client.post(
                    "/v1/ocr_postprocess",
                    json={"ocr_text": "[]"},
                )
                assert resp.status_code in (401, 403)
        else:
            from src.main import app

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw_client:
                resp = await raw_client.post(
                    "/v1/ocr_postprocess",
                    json={"ocr_text": "[]"},
                )
                assert resp.status_code in (401, 403)

    async def test_wrong_api_key(self, e2e_client):
        """Request with wrong API key should be rejected."""
        from tests.e2e.conftest import E2E_BASE_URL
        import httpx

        if E2E_BASE_URL:
            async with httpx.AsyncClient(base_url=E2E_BASE_URL) as raw_client:
                resp = await raw_client.post(
                    "/v1/ocr_postprocess",
                    json={"ocr_text": "[]"},
                    headers={"X-API-Key": "wrong-key"},
                )
                assert resp.status_code == 401
        else:
            from src.main import app

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as raw_client:
                resp = await raw_client.post(
                    "/v1/ocr_postprocess",
                    json={"ocr_text": "[]"},
                    headers={"X-API-Key": "wrong-key"},
                )
                assert resp.status_code == 401
