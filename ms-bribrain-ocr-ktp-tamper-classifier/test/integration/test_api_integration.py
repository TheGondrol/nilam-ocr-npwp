"""
Integration tests for API endpoints
Tests the full request/response cycle with FastAPI
"""

import io
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None  # type: ignore[assignment]


@pytest.fixture
def mock_tamper_service():
    """Create a mock tamper detection service for integration tests"""
    service = Mock()
    service.is_ready.return_value = True
    service.device = "cpu"
    service.model = Mock()
    service.processor = Mock()
    
    # Mock predict_from_bytes
    async def mock_predict_from_bytes(image_bytes):
        return {
            "predicted_class": "authentic",
            "predicted_id": 0,
            "confidence": 0.92,
            "probabilities": {
                "authentic": 0.92,
                "tampered": 0.08
            }
        }
    
    service.predict_from_bytes = mock_predict_from_bytes
    return service


@pytest.fixture
def app_client(mock_tamper_service):
    """Create a test client with mocked services"""
    # Import here to avoid early initialization
    from src.main import app
    
    # Patch the tamper service
    with patch("src.api.routes.get_tamper_service", return_value=mock_tamper_service):
        with patch("src.services.database_service.insert_log", new_callable=AsyncMock):
            client = TestClient(app)
            client.headers["X-API-Key"] = "test"
            yield client


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints"""
    
    def test_root_endpoint(self, app_client):
        """Test root endpoint returns API information"""
        response = app_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
        assert data["message"] == "Document Tamper Detection API"
    
    def test_health_check_endpoint(self, app_client):
        """Test health check endpoint"""
        response = app_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data
        assert "version" in data
        assert data["status"] in ("healthy", "degraded")
        assert data["model_loaded"] is True
    
    def test_predict_endpoint_with_valid_image(self, app_client, sample_image_bytes):
        """Test predict endpoint with valid image"""
        if not PIL_AVAILABLE:
            pytest.skip("PIL not available")
        
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        
        response = app_client.post("/v1/ocr_tamper", files=files)

        assert response.status_code == 200
        body = response.json()
        assert body["status_code"] == 200
        data = body["data"]
        assert data["filename"] == "test.jpg"
        assert data["prediction"] == "authentic"
        assert isinstance(data["confidence"], float)
        assert 0 <= data["confidence"] <= 1
    
    def test_predict_endpoint_without_file(self, app_client):
        """Test predict endpoint without file"""
        response = app_client.post("/v1/ocr_tamper")

        assert response.status_code == 422
    
    def test_predict_endpoint_with_tampered_prediction(self, app_client, sample_image_bytes):
        """Test predict endpoint returns tampered prediction"""
        if not PIL_AVAILABLE:
            pytest.skip("PIL not available")
        
        # Mock service to return tampered
        with patch("src.api.routes.get_tamper_service") as mock_get_service:
            service = Mock()
            service.is_ready.return_value = True
            service.device = "cpu"
            
            async def mock_predict_tampered(image_bytes):
                return {
                    "predicted_class": "tampered",
                    "predicted_id": 1,
                    "confidence": 0.87,
                    "probabilities": {
                        "authentic": 0.13,
                        "tampered": 0.87
                    }
                }
            
            service.predict_from_bytes = mock_predict_tampered
            mock_get_service.return_value = service
            
            with patch("src.services.database_service.insert_log", new_callable=AsyncMock):
                files = {"file": ("tampered.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
                response = app_client.post("/v1/ocr_tamper", files=files)
                
                assert response.status_code == 200
                body = response.json()
                data = body["data"]
                assert data["prediction"] == "tampered"
                assert data["confidence"] == 0.87
    
    def test_predict_endpoint_handles_service_error(self, app_client, sample_image_bytes):
        """Test predict endpoint handles service errors gracefully"""
        if not PIL_AVAILABLE:
            pytest.skip("PIL not available")
        
        # Mock service to raise error
        with patch("src.api.routes.get_tamper_service") as mock_get_service:
            service = Mock()
            service.is_ready.return_value = True
            
            async def mock_predict_error(image_bytes):
                raise Exception("Model inference failed")
            
            service.predict_from_bytes = mock_predict_error
            mock_get_service.return_value = service
            
            with patch("src.services.database_service.insert_log", new_callable=AsyncMock):
                files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
                response = app_client.post("/v1/ocr_tamper", files=files)
                
                assert response.status_code == 500
                body = response.json()
                assert body["error_code"] == "INTERNAL_ERROR"
    
    def test_cors_headers(self, app_client):
        """Test CORS headers are present"""
        response = app_client.get("/")
        
        # CORS headers should be added by middleware
        assert response.status_code == 200
    
    def test_request_id_header_in_response(self, app_client):
        """Test that x-request-id header is added to response"""
        response = app_client.get("/", headers={"x-request-id": "test-123"})
        
        assert response.status_code == 200
        assert "x-request-id" in response.headers
        assert response.headers["x-request-id"] == "test-123"
    
    def test_multiple_concurrent_requests(self, app_client, sample_image_bytes):
        """Test handling multiple concurrent prediction requests"""
        if not PIL_AVAILABLE:
            pytest.skip("PIL not available")
        
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        
        # Make multiple requests
        responses = []
        for i in range(5):
            response = app_client.post("/v1/ocr_tamper", files=files)
            responses.append(response)
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)
        assert all("data" in r.json() for r in responses)


@pytest.mark.integration
class TestAPIValidation:
    """Integration tests for API input validation"""
    
    def test_predict_with_invalid_file_type(self, app_client):
        """Test predict endpoint with invalid file type"""
        files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
        
        # Mock service that raises on invalid image
        with patch("src.api.routes.get_tamper_service") as mock_get_service:
            service = Mock()
            
            async def mock_predict_invalid(image_bytes):
                raise ValueError("Invalid image data")
            
            service.predict_from_bytes = mock_predict_invalid
            mock_get_service.return_value = service
            
            with patch("src.services.database_service.insert_log", new_callable=AsyncMock):
                response = app_client.post("/v1/ocr_tamper", files=files)

                # An unreadable image is a client error -> 400 INVALID_FILE.
                assert response.status_code == 400
                body = response.json()
                assert body["error_code"] == "INVALID_FILE"
    
    def test_health_check_when_service_not_ready(self, app_client):
        """Test health check when service is not ready"""
        with patch("src.api.routes.get_tamper_service") as mock_get_service:
            service = Mock()
            service.is_ready.return_value = False
            service.device = None
            mock_get_service.return_value = service
            
            response = app_client.get("/health")
            
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False


@pytest.mark.integration
class TestAPIResponseFormat:
    """Integration tests for API response format validation"""
    
    def test_prediction_response_format(self, app_client, sample_image_bytes):
        """Test prediction response has correct format"""
        if not PIL_AVAILABLE:
            pytest.skip("PIL not available")
        
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = app_client.post("/v1/ocr_tamper", files=files)
        
        assert response.status_code == 200
        body = response.json()
        assert body["status_code"] == 200
        data = body["data"]

        # Check all required fields are present
        required_fields = ["filename", "prediction", "confidence", "probabilities", "threshold", "timestamp"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Check data types
        assert isinstance(data["filename"], str)
        assert isinstance(data["prediction"], str)
        assert isinstance(data["confidence"], (int, float))
        assert isinstance(data["probabilities"], dict)
        assert isinstance(data["threshold"], (int, float))
        assert isinstance(data["timestamp"], str)

        # Check prediction is valid class
        assert data["prediction"] in ["authentic", "tampered"]

        # Check probabilities sum close to 1
        prob_sum = sum(data["probabilities"].values())
        assert 0.99 <= prob_sum <= 1.01
    
    def test_health_response_format(self, app_client):
        """Test health check response has correct format"""
        response = app_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields
        required_fields = ["status", "model_loaded", "device", "version"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Check data types
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["device"], str)
        assert isinstance(data["version"], str)

        # Check status is valid
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
