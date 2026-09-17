"""
Unit tests for api.schemas module
"""
import pytest
from pydantic import ValidationError

from src.api.schemas import (
    BoundingBox,
    Detection,
    DetectionBox,
    KTPDetectionResponse,
    HealthResponse,
    RootResponse
)


@pytest.mark.unit
class TestBoundingBox:
    """Test BoundingBox schema"""
    
    def test_bounding_box_valid(self):
        """Test valid bounding box"""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        
        assert bbox.x1 == 10.0
        assert bbox.y1 == 20.0
        assert bbox.x2 == 100.0
        assert bbox.y2 == 200.0
    
    def test_bounding_box_type_coercion(self):
        """Test bounding box with integer values"""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        
        assert isinstance(bbox.x1, float)
        assert isinstance(bbox.y1, float)
        assert isinstance(bbox.x2, float)
        assert isinstance(bbox.y2, float)
    
    def test_bounding_box_missing_fields(self):
        """Test bounding box with missing fields"""
        with pytest.raises(ValidationError):
            BoundingBox(x1=10.0, y1=20.0)  # ty: ignore[missing-argument]


@pytest.mark.unit
class TestDetection:
    """Test Detection schema"""
    
    def test_detection_valid(self):
        """Test valid detection"""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        detection = Detection(
            class_id=0,
            class_name="ktp",
            confidence=0.85,
            bbox=bbox
        )
        
        assert detection.class_id == 0
        assert detection.class_name == "ktp"
        assert detection.confidence == 0.85
        assert detection.bbox == bbox
    
    def test_detection_confidence_range(self):
        """Test detection with confidence values"""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        
        # Valid confidence
        detection = Detection(
            class_id=0,
            class_name="ktp",
            confidence=0.5,
            bbox=bbox
        )
        assert detection.confidence == 0.5
        
        # Should allow confidence > 1.0 (no validation)
        detection = Detection(
            class_id=0,
            class_name="ktp",
            confidence=1.5,
            bbox=bbox
        )
        assert detection.confidence == 1.5


@pytest.mark.unit
class TestDetectionBox:
    """Test DetectionBox schema"""
    
    def test_detection_box_valid(self):
        """Test valid detection box"""
        box = DetectionBox(
            label="ktp",
            confidence=0.85,
            bbox=[10.0, 20.0, 100.0, 200.0]
        )
        
        assert box.label == "ktp"
        assert box.confidence == 0.85
        assert len(box.bbox) == 4


@pytest.mark.unit
class TestKTPDetectionResponse:
    """Test KTPDetectionResponse schema"""
    
    def test_ktp_detection_response_with_detections(self):
        """Test KTP detection response with detections"""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        detection = Detection(
            class_id=0,
            class_name="ktp",
            confidence=0.85,
            bbox=bbox
        )
        
        response = KTPDetectionResponse(
            filename="test.jpg",
            detected=True,
            num_detected=1,
            detections=[detection],
            status="success",
            timestamp="2024-01-01T00:00:00"
        )
        
        assert response.filename == "test.jpg"
        assert response.detected
        assert response.num_detected == 1
        assert len(response.detections) == 1
        assert response.status == "success"
        assert response.reason is None
    
    def test_ktp_detection_response_no_detections(self):
        """Test KTP detection response with no detections"""
        response = KTPDetectionResponse(
            filename="test.jpg",
            detected=False,
            num_detected=0,
            detections=[],
            status="success",
            timestamp="2024-01-01T00:00:00"
        )
        
        assert not response.detected
        assert response.num_detected == 0
        assert len(response.detections) == 0
    
    def test_ktp_detection_response_with_reason(self):
        """Test KTP detection response with error reason"""
        response = KTPDetectionResponse(
            filename="test.jpg",
            detected=False,
            num_detected=0,
            detections=[],
            status="error",
            reason="Invalid image format",
            timestamp="2024-01-01T00:00:00"
        )
        
        assert response.reason == "Invalid image format"
        assert response.status == "error"


@pytest.mark.unit
class TestHealthResponse:
    """Test HealthResponse schema"""
    
    def test_health_response_healthy(self):
        """Test healthy response"""
        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            device="cuda:0",
            device_info={"type": "GPU", "name": "Tesla"},
            timestamp="2024-01-01T00:00:00"
        )
        
        assert response.status == "healthy"
        assert response.model_loaded
        assert response.device == "cuda:0"
        assert response.device_info["type"] == "GPU"
    
    def test_health_response_unhealthy(self):
        """Test unhealthy response"""
        response = HealthResponse(
            status="unhealthy",
            model_loaded=False,
            device="cpu",
            device_info={"type": "CPU"},
            timestamp="2024-01-01T00:00:00"
        )
        
        assert response.status == "unhealthy"
        assert not response.model_loaded


@pytest.mark.unit
class TestRootResponse:
    """Test RootResponse schema"""
    
    def test_root_response_valid(self):
        """Test valid root response"""
        response = RootResponse(
            message="KTP Detection API",
            description="API description",
            version="1.0.0",
            endpoints={
                "health": "/health",
                "predict": "/predict"
            }
        )
        
        assert response.message == "KTP Detection API"
        assert response.version == "1.0.0"
        assert "health" in response.endpoints
        assert "predict" in response.endpoints
    
    def test_root_response_serialization(self):
        """Test root response JSON serialization"""
        response = RootResponse(
            message="Test API",
            description="Test",
            version="1.0.0",
            endpoints={"test": "/test"}
        )
        
        json_data = response.model_dump()
        assert json_data["message"] == "Test API"
        assert json_data["version"] == "1.0.0"
