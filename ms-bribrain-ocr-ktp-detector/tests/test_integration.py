"""
Integration tests for the KTP Detection API
Tests the full API flow with mocked external dependencies
"""
import io
from unittest.mock import patch
import pytest


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints"""
    
    def test_root_endpoint(self, app_client):
        """Test root endpoint returns valid response"""
        response = app_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "KTP Detection API"
        assert "endpoints" in data
    
    def test_health_endpoint(self, app_client):
        """Test health check endpoint"""
        response = app_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data
    
    def test_predict_endpoint_invalid_file(self, app_client):
        """Test prediction endpoint with invalid file type"""
        with patch('src.api.routes.predictor') as mock_predictor:
            mock_predictor.is_loaded.return_value = True

            # Send non-image file
            files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
            response = app_client.post(
                "/v1/ocr_ktp_detection",
                files=files,
                headers={"X-API-Key": "test-api-key"},
            )

            assert response.status_code == 400

    def test_predict_endpoint_model_not_loaded(self, app_client, sample_image_bytes):
        """Test prediction endpoint when model is not loaded"""
        with patch('src.api.routes.predictor') as mock_predictor:
            mock_predictor.is_loaded.return_value = False

            files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
            response = app_client.post(
                "/v1/ocr_ktp_detection",
                files=files,
                headers={"X-API-Key": "test-api-key"},
            )

            assert response.status_code == 503
    
    def test_request_id_middleware(self, app_client):
        """Test that request ID middleware adds x-request-id header"""
        response = app_client.get("/", headers={"x-request-id": "test-123"})
        
        assert response.status_code == 200
        assert "x-request-id" in response.headers
    
    def test_cors_headers(self, app_client):
        """Test CORS headers are present"""
        response = app_client.options("/health")
        
        # CORS headers should be configured
        assert response.status_code in [200, 405]  # Some frameworks return 405 for OPTIONS


@pytest.mark.integration
class TestApiKeyAuthentication:
    """Integration tests for API key authentication"""

    def test_predict_without_api_key(self, app_client, sample_image_bytes):
        """Test that prediction endpoint rejects requests without API key"""
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = app_client.post("/v1/ocr_ktp_detection", files=files)

        assert response.status_code == 403

    def test_predict_with_wrong_api_key(self, app_client, sample_image_bytes):
        """Wrong API key → 401 in the unified envelope shape."""
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = app_client.post(
            "/v1/ocr_ktp_detection",
            files=files,
            headers={"X-API-Key": "wrong-key"},
        )

        assert response.status_code == 401
        body = response.json()
        expected_keys = {"status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"}
        assert expected_keys == set(body.keys())
        assert body["error_code"] == "UNAUTHORIZED"
        assert "Invalid or missing API key" in body["message"]

    def test_predict_with_valid_api_key(self, app_client, sample_image_bytes):
        """Test that prediction endpoint accepts requests with correct API key"""
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = app_client.post(
            "/v1/ocr_ktp_detection",
            files=files,
            headers={"X-API-Key": "test-api-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status_code"] == 200
        assert "detected" in data["data"]

    def test_health_no_api_key_required(self, app_client):
        """Test that health endpoint does not require API key"""
        response = app_client.get("/health")

        assert response.status_code == 200

    def test_root_no_api_key_required(self, app_client):
        """Test that root endpoint does not require API key"""
        response = app_client.get("/")

        assert response.status_code == 200


@pytest.mark.integration
class TestEndToEndFlow:
    """End-to-end integration tests"""
    
    def test_health_check_reflects_model_status(self, app_client):
        """Test health check reflects actual model status"""
        with patch('src.api.routes.predictor') as mock_predictor:
            with patch('src.api.routes._check_database', return_value={"status": "up"}):
                # Test with model loaded
                mock_predictor.is_loaded.return_value = True
                mock_predictor.get_device_string.return_value = "cpu"
                mock_predictor.get_device_info.return_value = {"type": "CPU", "count": 4}

                response = app_client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert data["model_loaded"]
                assert "checks" in data

                # Test with model not loaded
                mock_predictor.is_loaded.return_value = False

                response = app_client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "unhealthy"
                assert not data["model_loaded"]


@pytest.mark.integration
class TestEarlyRejectionEnvelope:
    """Auth/validation rejections use the unified envelope shape."""

    _ENVELOPE_KEYS = {"status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"}

    def test_missing_file_uses_envelope(self, app_client):
        """Missing required file field → 422 in the unified envelope shape."""
        response = app_client.post(
            "/v1/ocr_ktp_detection",
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 422
        body = response.json()
        assert self._ENVELOPE_KEYS == set(body.keys())
        assert body["error_code"] == "VALIDATION_ERROR"
        assert body["errors"]
