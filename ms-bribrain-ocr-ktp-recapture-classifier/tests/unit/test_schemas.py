"""Tests for API schema models."""

import pytest
from datetime import datetime
from pydantic import ValidationError


class TestPredictionResponse:
    """Test cases for PredictionResponse schema."""

    def test_valid_prediction_response(self):
        """Test creating valid PredictionResponse."""
        from src.schemas.api_schema import PredictionResponse
        
        response = PredictionResponse(
            filename="test.jpg",
            prediction="ORIGINAL",
            confidence=0.85,
            probability_recaptured=0.15,
            threshold=0.5,
            timestamp=datetime.now().isoformat()
        )
        
        assert response.filename == "test.jpg"
        assert response.prediction == "ORIGINAL"
        assert response.confidence == 0.85
        assert response.probability_recaptured == 0.15

    def test_prediction_response_confidence_validation(self):
        """Test confidence field validation."""
        from src.schemas.api_schema import PredictionResponse
        
        # Valid confidence (0-1) should work
        response = PredictionResponse(
            filename="test.jpg",
            prediction="RECAPTURED",
            confidence=0.95,
            probability_recaptured=0.95,
            threshold=0.5,
            timestamp=datetime.now().isoformat()
        )
        assert response.confidence == 0.95

        # Confidence > 1 should fail
        with pytest.raises(ValidationError):
            PredictionResponse(
                filename="test.jpg",
                prediction="ORIGINAL",
                confidence=1.5,
                probability_recaptured=0.5,
                threshold=0.5,
                timestamp=datetime.now().isoformat()
            )

        # Confidence < 0 should fail
        with pytest.raises(ValidationError):
            PredictionResponse(
                filename="test.jpg",
                prediction="ORIGINAL",
                confidence=-0.1,
                probability_recaptured=0.5,
                threshold=0.5,
                timestamp=datetime.now().isoformat()
            )

    def test_prediction_response_missing_required_field(self):
        """Test that missing required fields raise validation error."""
        from src.schemas.api_schema import PredictionResponse
        
        with pytest.raises(ValidationError):
            PredictionResponse(
                filename="test.jpg",
                prediction="ORIGINAL",
                # Missing confidence, probability_recaptured, threshold, timestamp
            )


class TestHealthResponse:
    """Test cases for HealthResponse schema."""

    def test_valid_health_response(self):
        """Test creating valid HealthResponse."""
        from src.schemas.api_schema import HealthResponse
        
        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            device="cuda:0",
            version="1.0.0",
            checks={"model": {"status": "up"}, "database": {"status": "up"}},
        )

        assert response.status == "healthy"
        assert response.model_loaded is True
        assert response.device == "cuda:0"

    def test_health_response_unhealthy(self):
        """Test unhealthy status."""
        from src.schemas.api_schema import HealthResponse

        response = HealthResponse(
            status="unhealthy",
            model_loaded=False,
            device="cpu",
            version="1.0.0",
        )

        assert response.status == "unhealthy"
        assert response.model_loaded is False

    def test_health_response_model_field_allowed(self):
        """Test that model_ prefix is allowed in field names."""
        from src.schemas.api_schema import HealthResponse

        # Should not raise error for model_loaded field
        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            device="cpu",
            version="1.0.0",
        )
        assert hasattr(response, 'model_loaded')


class TestBatchPredictionItem:
    """Test cases for BatchPredictionItem schema."""

    def test_successful_batch_item(self):
        """Test batch item with successful result."""
        from src.schemas.api_schema import BatchPredictionItem, PredictionResponse
        
        result = PredictionResponse(
            filename="test.jpg",
            prediction="ORIGINAL",
            confidence=0.90,
            probability_recaptured=0.10,
            threshold=0.5,
            timestamp=datetime.now().isoformat()
        )
        
        item = BatchPredictionItem(
            filename="test.jpg",
            result=result,
            error=None
        )
        
        assert item.filename == "test.jpg"
        assert item.result is not None
        assert item.error is None

    def test_failed_batch_item(self):
        """Test batch item with error."""
        from src.schemas.api_schema import BatchPredictionItem
        
        item = BatchPredictionItem(
            filename="invalid.jpg",
            result=None,
            error="Failed to process image"
        )
        
        assert item.filename == "invalid.jpg"
        assert item.result is None
        assert item.error == "Failed to process image"


class TestBatchPredictionResponse:
    """Test cases for BatchPredictionResponse schema."""

    def test_batch_prediction_response(self):
        """Test creating batch prediction response."""
        from src.schemas.api_schema import BatchPredictionResponse, BatchPredictionItem
        
        response = BatchPredictionResponse(
            results=[
                BatchPredictionItem(filename="img1.jpg", result=None, error=None),
                BatchPredictionItem(filename="img2.jpg", result=None, error="Error")
            ],
            total=2,
            successful=1,
            failed=1
        )
        
        assert response.total == 2
        assert response.successful == 1
        assert response.failed == 1
        assert len(response.results) == 2
