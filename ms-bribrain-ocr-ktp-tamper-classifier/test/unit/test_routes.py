"""
Unit tests for API routes
"""

import io
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

from src.api.routes import router, PredictionResponse, HealthResponse, LivenessResponse


@pytest.fixture
def test_client():
    """Create a test client for the API"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    client.headers["X-API-Key"] = "test"
    return client


@pytest.fixture
def mock_tamper_service():
    """Create a mock tamper detection service"""
    service = Mock()
    service.is_ready.return_value = True
    service.device = "cpu"
    service.predict_from_bytes = AsyncMock(return_value={
        "predicted_class": "authentic",
        "confidence": 0.92,
        "probabilities": {"authentic": 0.92, "tampered": 0.08}
    })
    return service


@pytest.mark.unit
class TestRootEndpoint:
    """Tests for root endpoint"""

    def test_root_returns_api_info(self, test_client):
        """Test that root endpoint returns API information"""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
        assert data["message"] == "Document Tamper Detection API"
        assert "liveness" in data["endpoints"]
        assert "readiness" in data["endpoints"]


@pytest.mark.unit
class TestLivenessEndpoint:
    """Tests for liveness probe"""

    def test_liveness_returns_200(self, test_client):
        response = test_client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert "version" in data


@pytest.mark.unit
class TestReadinessEndpoint:
    """Tests for readiness probe"""

    @patch("src.api.routes._check_database", new_callable=AsyncMock, return_value={"status": "up"})
    @patch("src.api.routes.get_tamper_service")
    def test_readiness_ready(self, mock_get_service, mock_db, test_client, mock_tamper_service):
        mock_get_service.return_value = mock_tamper_service
        response = test_client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    @patch("src.api.routes._check_database", new_callable=AsyncMock, return_value={"status": "down", "reason": "not initialised"})
    @patch("src.api.routes.get_tamper_service")
    def test_readiness_not_ready(self, mock_get_service, mock_db, test_client):
        service = Mock()
        service.is_ready.return_value = False
        mock_get_service.return_value = service
        response = test_client.get("/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"


@pytest.mark.unit
class TestHealthCheckEndpoint:
    """Tests for health check endpoint"""

    @patch("src.api.routes._check_database", new_callable=AsyncMock, return_value={"status": "up"})
    @patch("src.api.routes.get_tamper_service")
    def test_health_check_when_healthy(self, mock_get_service, mock_db, test_client, mock_tamper_service):
        """Test health check when service is healthy"""
        mock_get_service.return_value = mock_tamper_service

        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "checks" in data

    @patch("src.api.routes._check_database", new_callable=AsyncMock, return_value={"status": "down", "reason": "connection refused"})
    @patch("src.api.routes.get_tamper_service")
    def test_health_check_when_degraded(self, mock_get_service, mock_db, test_client, mock_tamper_service):
        """Test health check when model loaded but DB down"""
        mock_get_service.return_value = mock_tamper_service

        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is True

    @patch("src.api.routes._check_database", new_callable=AsyncMock, return_value={"status": "up"})
    @patch("src.api.routes.get_tamper_service")
    def test_health_check_when_unhealthy(self, mock_get_service, mock_db, test_client):
        """Test health check when service is not ready"""
        service = Mock()
        service.is_ready.return_value = False
        service.device = None
        mock_get_service.return_value = service

        response = test_client.get("/health")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False


@pytest.mark.unit
class TestPredictEndpoint:
    """Tests for predict endpoint"""
    
    @patch("src.api.routes.get_tamper_service")
    @patch("src.api.routes.insert_log")
    @patch("src.api.routes.request_id_ctx")
    def test_predict_success(self, mock_request_id, mock_insert_log, mock_get_service, 
                            test_client, mock_tamper_service, sample_image_bytes):
        """Test successful prediction"""
        mock_get_service.return_value = mock_tamper_service
        mock_request_id.get.return_value = "test-request-id"
        mock_insert_log.return_value = AsyncMock()
        
        # Create file upload
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        
        response = test_client.post("/v1/ocr_tamper", files=files)

        assert response.status_code == 200
        body = response.json()
        assert body["status_code"] == 200
        data = body["data"]
        assert data["filename"] == "test.jpg"
        assert data["prediction"] == "authentic"
        assert data["confidence"] == 0.92
        assert "probabilities" in data
        assert "timestamp" in data
    
    @patch("src.api.routes.get_tamper_service")
    @patch("src.api.routes.insert_log")
    @patch("src.api.routes.request_id_ctx")
    def test_predict_tampered_document(self, mock_request_id, mock_insert_log, 
                                       mock_get_service, test_client, sample_image_bytes):
        """Test prediction for tampered document"""
        service = Mock()
        service.predict_from_bytes = AsyncMock(return_value={
            "predicted_class": "tampered",
            "confidence": 0.87,
            "probabilities": {"authentic": 0.13, "tampered": 0.87}
        })
        mock_get_service.return_value = service
        mock_request_id.get.return_value = "test-request-id"
        mock_insert_log.return_value = AsyncMock()
        
        files = {"file": ("tampered.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        
        response = test_client.post("/v1/ocr_tamper", files=files)

        assert response.status_code == 200
        body = response.json()
        data = body["data"]
        assert data["prediction"] == "tampered"
        assert data["confidence"] == 0.87
    
    @patch("src.api.routes.get_tamper_service")
    @patch("src.api.routes.insert_log")
    @patch("src.api.routes.request_id_ctx")
    def test_predict_handles_error(self, mock_request_id, mock_insert_log, 
                                   mock_get_service, test_client, sample_image_bytes):
        """Test that predict handles errors gracefully"""
        service = Mock()
        service.predict_from_bytes = AsyncMock(side_effect=Exception("Processing error"))
        mock_get_service.return_value = service
        mock_request_id.get.return_value = "test-request-id"
        mock_insert_log.return_value = AsyncMock()
        
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        
        response = test_client.post("/v1/ocr_tamper", files=files)
        
        assert response.status_code == 500
        body = response.json()
        assert body["error_code"] == "INTERNAL_ERROR"
    
    @patch("src.api.routes.get_tamper_service")
    @patch("src.api.routes.insert_log")
    @patch("src.api.routes.request_id_ctx")
    def test_predict_logs_request(self, mock_request_id, mock_insert_log, 
                                  mock_get_service, test_client, mock_tamper_service, 
                                  sample_image_bytes):
        """Test that prediction logs are inserted"""
        mock_get_service.return_value = mock_tamper_service
        mock_request_id.get.return_value = "test-request-id"
        mock_insert_log.return_value = AsyncMock()
        
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        
        response = test_client.post("/v1/ocr_tamper", files=files)

        assert response.status_code == 200
        body = response.json()
        assert body["status_code"] == 200
        # Verify insert_log was called
        assert mock_insert_log.called


@pytest.mark.unit
@pytest.mark.skip(reason="Batch predict endpoint not implemented yet")
class TestBatchPredictEndpoint:
    """Tests for batch predict endpoint"""
    
    @patch("src.api.routes.get_tamper_service")
    @patch("src.api.routes.insert_log")
    @patch("src.api.routes.request_id_ctx")
    def test_batch_predict_success(self, mock_request_id, mock_insert_log, 
                                   mock_get_service, test_client, sample_image_bytes):
        """Test successful batch prediction"""
        service = Mock()
        service.predict_batch_from_bytes = AsyncMock(return_value=[
            {
                "predicted_class": "authentic",
                "confidence": 0.92,
                "probabilities": {"authentic": 0.92, "tampered": 0.08}
            },
            {
                "predicted_class": "tampered",
                "confidence": 0.85,
                "probabilities": {"authentic": 0.15, "tampered": 0.85}
            }
        ])
        mock_get_service.return_value = service
        mock_request_id.get.return_value = "test-request-id"
        mock_insert_log.return_value = AsyncMock()
        
        files = [
            ("files", ("test1.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")),
            ("files", ("test2.jpg", io.BytesIO(sample_image_bytes), "image/jpeg"))
        ]
        
        response = test_client.post("/predict/batch", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["total_processed"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["filename"] == "test1.jpg"
        assert data["results"][1]["filename"] == "test2.jpg"
    
    @patch("src.api.routes.get_tamper_service")
    @patch("src.api.routes.insert_log")
    @patch("src.api.routes.request_id_ctx")
    def test_batch_predict_empty_files(self, mock_request_id, mock_insert_log, 
                                       mock_get_service, test_client):
        """Test batch prediction with no files"""
        mock_get_service.return_value = Mock()
        mock_request_id.get.return_value = "test-request-id"
        mock_insert_log.return_value = AsyncMock()
        
        response = test_client.post("/predict/batch", files=[])
        
        assert response.status_code == 400
        assert "No files provided" in response.json()["detail"]
    
    @patch("src.api.routes.get_tamper_service")
    @patch("src.api.routes.insert_log")
    @patch("src.api.routes.request_id_ctx")
    def test_batch_predict_handles_error(self, mock_request_id, mock_insert_log, 
                                        mock_get_service, test_client, sample_image_bytes):
        """Test that batch predict handles errors gracefully"""
        service = Mock()
        service.predict_batch_from_bytes = AsyncMock(side_effect=Exception("Batch processing error"))
        mock_get_service.return_value = service
        mock_request_id.get.return_value = "test-request-id"
        mock_insert_log.return_value = AsyncMock()
        
        files = [
            ("files", ("test1.jpg", io.BytesIO(sample_image_bytes), "image/jpeg"))
        ]
        
        response = test_client.post("/predict/batch", files=files)
        
        assert response.status_code == 500
        assert "Batch prediction failed" in response.json()["detail"]


@pytest.mark.unit
class TestResponseModels:
    """Tests for response model validation"""
    
    def test_prediction_response_model(self):
        """Test PredictionResponse model"""
        response = PredictionResponse(
            filename="test.jpg",
            prediction="authentic",
            confidence=0.92,
            probabilities={"authentic": 0.92, "tampered": 0.08},
            threshold=0.5,
            timestamp=datetime.now().isoformat()
        )
        
        assert response.filename == "test.jpg"
        assert response.prediction == "authentic"
        assert response.confidence == 0.92
    
    def test_health_response_model(self):
        """Test HealthResponse model"""
        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            device="cpu",
            version="1.0.0",
            checks={"model": {"status": "up"}, "database": {"status": "up"}},
        )

        assert response.status == "healthy"
        assert response.model_loaded is True
        assert response.device == "cpu"


@pytest.mark.unit
class TestEarlyRejectionEnvelope:
    """401/403/422 use the unified envelope shape and are logged to OcrKtpLog."""

    _ENVELOPE_KEYS = {"status_code", "status_desc", "message", "data", "error_code", "errors", "request_id"}

    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from src.api.routes import router, register_exception_handlers
        from src.middleware.add_requestid import RequestIdMiddleware
        app = FastAPI()
        app.add_middleware(RequestIdMiddleware)
        app.include_router(router)
        register_exception_handlers(app)
        return TestClient(app)

    def _jpeg(self):
        from PIL import Image
        img = Image.new("RGB", (10, 10))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return buf

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    def test_wrong_api_key_uses_envelope_and_logs(self, mock_log, client):
        response = client.post(
            "/v1/ocr_tamper",
            files={"file": ("test.jpg", self._jpeg(), "image/jpeg")},
            headers={"X-API-Key": "wrong"},
        )
        assert response.status_code == 401
        body = response.json()
        assert self._ENVELOPE_KEYS == set(body.keys())
        assert body["error_code"] == "UNAUTHORIZED"
        mock_log.assert_awaited_once()
        assert mock_log.call_args.kwargs["response_code"] == 401

    @patch("src.api.routes.insert_log", new_callable=AsyncMock)
    def test_missing_file_uses_envelope_and_logs(self, mock_log, client):
        response = client.post(
            "/v1/ocr_tamper",
            headers={"X-API-Key": "test"},  # file field omitted
        )
        assert response.status_code == 422
        body = response.json()
        assert self._ENVELOPE_KEYS == set(body.keys())
        assert body["error_code"] == "VALIDATION_ERROR"
        assert body["errors"]
        mock_log.assert_awaited_once()
        assert mock_log.call_args.kwargs["response_code"] == 422
