"""
Unit tests for tamper_detection service
"""

import pytest
from unittest.mock import Mock, patch

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None  # type: ignore

from src.services.tamper_detection import (
    TamperDetectionService,
    get_inference_executor
)


@pytest.mark.unit
class TestGetInferenceExecutor:
    """Tests for get_inference_executor function"""
    
    @patch("src.services.tamper_detection._inference_executor", None)
    @patch("os.cpu_count")
    def test_creates_executor_with_correct_workers(self, mock_cpu_count):
        """Test that executor is created with correct number of workers"""
        mock_cpu_count.return_value = 8
        
        executor = get_inference_executor()
        
        assert executor is not None
        assert executor._max_workers == 4  # Half of 8
    
    @patch("src.services.tamper_detection._inference_executor", None)
    @patch("os.cpu_count")
    def test_minimum_workers_is_two(self, mock_cpu_count):
        """Test that minimum workers is 2"""
        mock_cpu_count.return_value = 2
        
        executor = get_inference_executor()
        
        assert executor._max_workers == 2  # max(2, 2//2) = 2
    
    @patch("src.services.tamper_detection._inference_executor")
    def test_returns_existing_executor(self, mock_executor):
        """Test that existing executor is returned"""
        mock_executor_instance = Mock()

        
        # This should return the mocked executor without creating new one
        from src.services import tamper_detection
        original_executor = tamper_detection._inference_executor
        tamper_detection._inference_executor = mock_executor_instance
        
        executor = get_inference_executor()
        
        assert executor == mock_executor_instance
        tamper_detection._inference_executor = original_executor


@pytest.mark.unit
class TestTamperDetectionService:
    """Tests for TamperDetectionService class"""
    
    def test_initialization(self):
        """Test service initialization"""
        service = TamperDetectionService(
            model_path="/fake/path",
            image_size=320,
            force_cpu=True
        )
        
        assert service.model_path == "/fake/path"
        assert service.image_size == 320
        assert service.force_cpu is True
        assert service.model is None
        assert service.processor is None
        assert service.device is None
    
    def test_is_ready_when_not_initialized(self):
        """Test is_ready returns False when not initialized"""
        service = TamperDetectionService(
            model_path="/fake/path",
            image_size=320
        )
        
        assert service.is_ready() is False
    
    @patch("src.services.tamper_detection.load_model")
    @patch("src.services.tamper_detection.get_inference_executor")
    def test_initialize_loads_model(self, mock_executor, mock_load_model, mock_model, mock_processor):
        """Test that initialize loads the model"""
        mock_load_model.return_value = (mock_model, mock_processor, "cpu")
        mock_executor.return_value = Mock()
        
        service = TamperDetectionService(
            model_path="/fake/path",
            image_size=320,
            force_cpu=True
        )
        service.initialize()
        
        assert service.model == mock_model
        assert service.processor == mock_processor
        assert service.device == "cpu"
        assert service.is_ready() is True
    
    @patch("src.services.tamper_detection.load_model")
    def test_initialize_raises_on_error(self, mock_load_model):
        """Test that initialize raises exception on error"""
        mock_load_model.side_effect = Exception("Failed to load model")
        
        service = TamperDetectionService(model_path="/fake/path")
        
        with pytest.raises(Exception, match="Failed to load model"):
            service.initialize()
    
    @pytest.mark.asyncio
    @patch("src.services.tamper_detection.preprocess_image")
    async def test_predict(self, mock_preprocess, mock_model, mock_processor, sample_image):
        """Test prediction"""
        service = TamperDetectionService(model_path="/fake/path", image_size=320)
        service.model = mock_model
        service.processor = mock_processor
        service.device = "cpu"
        
        # Mock preprocessing
        mock_tensor = Mock()
        mock_tensor.to = Mock(return_value=mock_tensor)
        mock_preprocess.return_value = mock_tensor
        
        if PIL_AVAILABLE:
            result = await service.predict(sample_image)
            
            assert "predicted_class" in result
            assert "confidence" in result
            assert "probabilities" in result
            mock_preprocess.assert_called_once()
        else:
            pytest.skip("PIL not available")
    
    @pytest.mark.asyncio
    @patch("src.services.tamper_detection.load_image_from_bytes")
    async def test_predict_from_bytes(self, mock_load_image, mock_model, mock_processor, sample_image_bytes):
        """Test prediction from bytes"""
        service = TamperDetectionService(model_path="/fake/path", image_size=320)
        service.model = mock_model
        service.processor = mock_processor
        service.device = "cpu"
        
        if PIL_AVAILABLE and sample_image_bytes:
            from PIL import Image as PILImage
            mock_load_image.return_value = PILImage.new("RGB", (100, 100))
            
            with patch.object(service, "predict") as mock_predict:
                mock_predict.return_value = {
                    "predicted_class": "tampered",
                    "confidence": 0.85,
                    "probabilities": {"authentic": 0.15, "tampered": 0.85}
                }
                
                result = await service.predict_from_bytes(sample_image_bytes)
                
                assert result["predicted_class"] == "tampered"
                assert result["confidence"] == 0.85
                mock_load_image.assert_called_once_with(sample_image_bytes)
        else:
            pytest.skip("PIL not available")
    
    @pytest.mark.asyncio
    async def test_predict_not_ready(self, sample_image_bytes):
        """Test predict raises when service not ready"""
        service = TamperDetectionService(model_path="/fake/path")
        
        with pytest.raises(RuntimeError, match="not initialized"):
            await service.predict_from_bytes(sample_image_bytes)
    
    @pytest.mark.asyncio
    @patch("src.services.tamper_detection.load_image_from_bytes")
    async def test_predict_batch(self, mock_load_image, mock_model, mock_processor):
        """Test batch prediction"""
        service = TamperDetectionService(model_path="/fake/path", image_size=320)
        service.model = mock_model
        service.processor = mock_processor
        service.device = "cpu"
        
        if PIL_AVAILABLE:
            from PIL import Image as PILImage
            images = [PILImage.new("RGB", (100, 100)) for _ in range(3)]
            
            with patch.object(service, "predict") as mock_predict:
                mock_predict.return_value = {
                    "predicted_class": "authentic",
                    "confidence": 0.92,
                    "probabilities": {"authentic": 0.92, "tampered": 0.08}
                }
                
                results = await service.predict_batch(images)
                
                assert len(results) == 3
                assert all(r["predicted_class"] == "authentic" for r in results)
        else:
            pytest.skip("PIL not available")


@pytest.mark.unit
class TestInitializeTamperService:
    """Tests for initialize_tamper_service function"""
    
    @patch("src.services.tamper_detection.tamper_service", None)
    @patch("src.services.tamper_detection.TamperDetectionService")
    def test_initialize_creates_service(self, mock_service_class):
        """Test that initialize creates and initializes service"""
        from src.services.tamper_detection import initialize_tamper_service
        
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        initialize_tamper_service(
            model_path="/fake/path",
            image_size=320,
            force_cpu=True
        )
        
        # Check that constructor was called with positional args
        mock_service_class.assert_called_once()
        call_args = mock_service_class.call_args[0]
        assert call_args[0] == "/fake/path"
        assert call_args[1] == 320
        assert call_args[2] is True
        mock_service.initialize.assert_called_once()


@pytest.mark.unit
class TestGetTamperService:
    """Tests for get_tamper_service function"""
    
    @patch("src.services.tamper_detection.tamper_service", None)
    def test_get_service_raises_when_not_initialized(self):
        """Test that get_tamper_service raises when service not initialized"""
        from src.services.tamper_detection import get_tamper_service
        
        with pytest.raises(RuntimeError, match="not initialized"):
            get_tamper_service()
