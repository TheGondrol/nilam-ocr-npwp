"""Unit tests for src.services.ocr_service module"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from src.services.ocr_service import (
    get_ocr,
    cleanup_ocr,
    is_ocr_ready,
    transform_ocr_result,
    filter_result,
    perform_ocr,
    perform_ocr_async,
    _process_image_sync,
    _run_ocr_sync
)
from src.core.exceptions import (
    OCRInitializationError,
    OCRProcessingError,
    ImageValidationError
)


class TestGetOcr:
    """Test cases for get_ocr function"""

    def setup_method(self):
        """Reset global OCR instance before each test"""
        import src.services.ocr_service
        src.services.ocr_service._ocr_instance = None
        src.services.ocr_service._ocr_initialized = False

    def test_get_ocr_creates_instance_gpu(self, mock_settings):
        """Test OCR instance creation with GPU"""
        mock_settings.ocr_server_config_path = "server_config.yaml"
        
        mock_paddle_ocr = MagicMock()
        
        with patch('src.services.ocr_service.settings', mock_settings):
            with patch('src.services.ocr_service.get_use_gpu', return_value=True):
                with patch('paddleocr.PaddleOCR', return_value=mock_paddle_ocr):
                    result = get_ocr()
                    
                    assert result.name == "paddle"
                    assert result.engine == mock_paddle_ocr
                    assert is_ocr_ready() is True

    def test_get_ocr_creates_instance_cpu(self, mock_settings):
        """Test OCR instance creation with CPU"""
        mock_settings.ocr_mobile_config_path = "mobile_config.yaml"
        
        mock_paddle_ocr = MagicMock()
        
        with patch('src.services.ocr_service.settings', mock_settings):
            with patch('src.services.ocr_service.get_use_gpu', return_value=False):
                with patch('paddleocr.PaddleOCR', return_value=mock_paddle_ocr):
                    result = get_ocr()
                    
                    assert result.name == "paddle"
                    assert result.engine == mock_paddle_ocr
                    assert is_ocr_ready() is True

    def test_get_ocr_returns_cached_instance(self, mock_settings):
        """Test that get_ocr returns cached instance"""
        mock_paddle_ocr = MagicMock()
        
        with patch('src.services.ocr_service.settings', mock_settings):
            with patch('src.services.ocr_service.get_use_gpu', return_value=False):
                with patch('paddleocr.PaddleOCR', return_value=mock_paddle_ocr) as mock_constructor:
                    # First call
                    result1 = get_ocr()
                    # Second call
                    result2 = get_ocr()
                    
                    # Should return same instance
                    assert result1 == result2
                    # Constructor should only be called once
                    assert mock_constructor.call_count == 1

    def test_get_ocr_initialization_error(self, mock_settings):
        """Test OCR initialization failure"""
        with patch('src.services.ocr_service.settings', mock_settings):
            with patch('src.services.ocr_service.get_use_gpu', return_value=False):
                with patch('paddleocr.PaddleOCR', side_effect=Exception("Init failed")):
                    with pytest.raises(OCRInitializationError) as exc_info:
                        get_ocr()
                    
                    assert "OCR initialization failed" in str(exc_info.value)


class TestCleanupOcr:
    """Test cases for cleanup_ocr function"""

    def test_cleanup_ocr_with_instance(self):
        """Test cleanup when OCR instance exists"""
        import src.services.ocr_service
        src.services.ocr_service._ocr_instance = MagicMock()
        src.services.ocr_service._ocr_initialized = True
        src.services.ocr_service._executor = MagicMock()
        
        cleanup_ocr()
        
        assert src.services.ocr_service._ocr_instance is None
        assert src.services.ocr_service._ocr_initialized is False
        assert src.services.ocr_service._executor is None

    def test_cleanup_ocr_without_instance(self):
        """Test cleanup when no OCR instance exists"""
        import src.services.ocr_service
        src.services.ocr_service._ocr_instance = None
        src.services.ocr_service._ocr_initialized = False
        
        # Should not raise error
        cleanup_ocr()


class TestIsOcrReady:
    """Test cases for is_ocr_ready function"""

    def test_is_ocr_ready_true(self):
        """Test is_ocr_ready returns True when initialized"""
        import src.services.ocr_service
        src.services.ocr_service._ocr_instance = MagicMock()
        src.services.ocr_service._ocr_initialized = True
        
        assert is_ocr_ready() is True

    def test_is_ocr_ready_false_no_instance(self):
        """Test is_ocr_ready returns False when no instance"""
        import src.services.ocr_service
        src.services.ocr_service._ocr_instance = None
        src.services.ocr_service._ocr_initialized = True
        
        assert is_ocr_ready() is False

    def test_is_ocr_ready_false_not_initialized(self):
        """Test is_ocr_ready returns False when not initialized"""
        import src.services.ocr_service
        src.services.ocr_service._ocr_instance = MagicMock()
        src.services.ocr_service._ocr_initialized = False
        
        assert is_ocr_ready() is False


class TestTransformOcrResult:
    """Test cases for transform_ocr_result function"""

    def test_transform_ocr_result_success(self, mock_ocr_result):
        """Test successful OCR result transformation"""
        width = 200
        
        result = transform_ocr_result(mock_ocr_result, width)
        
        assert isinstance(result, list)
        assert len(result) <= len(mock_ocr_result["rec_texts"])
        
        # Check structure
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], list)  # coordinates
            assert isinstance(item[1], tuple)  # (text, score)

    def test_transform_ocr_result_empty(self):
        """Test transformation with empty results"""
        empty_result = {
            "rec_texts": [],
            "rec_scores": [],
            "rec_polys": []
        }
        
        result = transform_ocr_result(empty_result, 100)
        
        assert result == []

    def test_transform_ocr_result_filters_by_width(self, mock_settings):
        """Test that results are filtered by width threshold"""
        mock_settings.ocr_width_threshold_ratio = 0.5
        
        ocr_result = {
            "rec_texts": ["Text 1", "Text 2"],
            "rec_scores": [0.95, 0.88],
            "rec_polys": [
                np.array([[10, 10], [40, 10], [40, 30], [10, 30]]),  # Within threshold
                np.array([[10, 10], [200, 10], [200, 30], [10, 30]])  # Beyond threshold
            ]
        }
        
        with patch('src.services.ocr_service.settings', mock_settings):
            result = transform_ocr_result(ocr_result, 100)
            
            # Only first result should pass (width 40 < 50)
            assert len(result) == 1


class TestFilterResult:
    """Test cases for filter_result function"""

    def test_filter_result_within_threshold(self, mock_settings):
        """Test filtering with results within threshold"""
        mock_settings.ocr_width_threshold_ratio = 0.8
        
        result = [
            ([[10, 10], [70, 10], [70, 30], [10, 30]], ("Text", 0.95)),
        ]
        
        with patch('src.services.ocr_service.settings', mock_settings):
            filtered = filter_result(result, 100)
            
            assert len(filtered) == 1

    def test_filter_result_beyond_threshold(self, mock_settings):
        """Test filtering with results beyond threshold"""
        mock_settings.ocr_width_threshold_ratio = 0.5
        
        result = [
            ([[10, 10], [90, 10], [90, 30], [10, 30]], ("Text", 0.95)),
        ]
        
        with patch('src.services.ocr_service.settings', mock_settings):
            filtered = filter_result(result, 100)
            
            assert len(filtered) == 0

    def test_filter_result_mixed(self, mock_settings):
        """Test filtering with mixed results"""
        mock_settings.ocr_width_threshold_ratio = 0.5
        
        result = [
            ([[10, 10], [40, 10], [40, 30], [10, 30]], ("Text 1", 0.95)),  # Within
            ([[10, 10], [90, 10], [90, 30], [10, 30]], ("Text 2", 0.88)),  # Beyond
            ([[10, 10], [45, 10], [45, 30], [10, 30]], ("Text 3", 0.92)),  # Within
        ]
        
        with patch('src.services.ocr_service.settings', mock_settings):
            filtered = filter_result(result, 100)
            
            assert len(filtered) == 2


class TestProcessImageSync:
    """Test cases for _process_image_sync function"""

    def test_process_image_sync_success(self, sample_image_bytes):
        """Test successful image processing"""
        image_np, height, width = _process_image_sync(sample_image_bytes)
        
        assert isinstance(image_np, np.ndarray)
        assert len(image_np.shape) == 3
        assert image_np.shape[2] == 3  # RGB
        assert height > 0
        assert width > 0

    def test_process_image_sync_invalid_data(self):
        """Test processing invalid image data"""
        invalid_bytes = b"not an image"
        
        with pytest.raises(Exception):
            _process_image_sync(invalid_bytes)


class TestRunOcrSync:
    """Test cases for _run_ocr_sync function"""

    def test_run_ocr_sync_success(self):
        """Test successful OCR prediction"""
        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = [{"rec_texts": ["Test"]}]
        
        image_np = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = _run_ocr_sync(mock_ocr, image_np)
        
        assert result is not None
        mock_ocr.predict.assert_called_once_with(image_np)


class TestPerformOcr:
    """Test cases for perform_ocr function"""

    def test_perform_ocr_success(self, sample_image_bytes, mock_paddle_ocr, mock_settings):
        """Test successful OCR operation"""
        import numpy as np
        
        # Create mock result with coordinates within threshold (small x values)
        mock_ocr_result = {
            "rec_texts": ["Text 1", "Text 2"],
            "rec_scores": [0.95, 0.88],
            "rec_polys": [
                np.array([[5, 10], [20, 10], [20, 30], [5, 30]]),   # Within 80% of 100 width
                np.array([[5, 40], [25, 40], [25, 60], [5, 60]])    # Within 80% of 100 width
            ]
        }
        mock_paddle_ocr.predict.return_value = [mock_ocr_result]
        mock_settings.ocr_width_threshold_ratio = 0.8
        
        with patch('src.services.ocr_service.get_ocr', return_value=mock_paddle_ocr):
            with patch('src.services.ocr_service.settings', mock_settings):
                result = perform_ocr(sample_image_bytes)
                
                assert isinstance(result, list)
                assert len(result) > 0

    def test_perform_ocr_invalid_image(self):
        """Test OCR with invalid image"""
        invalid_bytes = b"not an image"
        
        with patch('src.services.ocr_service.get_ocr'):
            with pytest.raises(ImageValidationError):
                perform_ocr(invalid_bytes)

    def test_perform_ocr_empty_results(self, sample_image_bytes, mock_paddle_ocr):
        """Test OCR with empty results"""
        mock_paddle_ocr.predict.return_value = []
        
        with patch('src.services.ocr_service.get_ocr', return_value=mock_paddle_ocr):
            result = perform_ocr(sample_image_bytes)
            
            assert result == []

    def test_perform_ocr_prediction_error(self, sample_image_bytes, mock_paddle_ocr):
        """Test OCR with prediction error"""
        mock_paddle_ocr.predict.side_effect = Exception("Prediction failed")
        
        with patch('src.services.ocr_service.get_ocr', return_value=mock_paddle_ocr):
            with pytest.raises(OCRProcessingError):
                perform_ocr(sample_image_bytes)


@pytest.mark.asyncio
class TestPerformOcrAsync:
    """Test cases for perform_ocr_async function"""

    async def test_perform_ocr_async_success(self, sample_image_bytes, mock_paddle_ocr, mock_settings):
        """Test successful async OCR operation"""
        import numpy as np
        
        # Create mock result with coordinates within threshold (small x values)
        mock_ocr_result = {
            "rec_texts": ["Text 1", "Text 2"],
            "rec_scores": [0.95, 0.88],
            "rec_polys": [
                np.array([[5, 10], [20, 10], [20, 30], [5, 30]]),   # Within 80% of 100 width
                np.array([[5, 40], [25, 40], [25, 60], [5, 60]])    # Within 80% of 100 width
            ]
        }
        mock_paddle_ocr.predict.return_value = [mock_ocr_result]
        mock_settings.ocr_width_threshold_ratio = 0.8
        
        with patch('src.services.ocr_service.get_ocr', return_value=mock_paddle_ocr):
            with patch('src.services.ocr_service.settings', mock_settings):
                result = await perform_ocr_async(sample_image_bytes)
                
                assert isinstance(result, list)
                assert len(result) > 0

    async def test_perform_ocr_async_invalid_image(self):
        """Test async OCR with invalid image"""
        invalid_bytes = b"not an image"
        
        with patch('src.services.ocr_service.get_ocr'):
            with pytest.raises(ImageValidationError):
                await perform_ocr_async(invalid_bytes)

    async def test_perform_ocr_async_empty_results(self, sample_image_bytes, mock_paddle_ocr):
        """Test async OCR with empty results"""
        mock_paddle_ocr.predict.return_value = []
        
        with patch('src.services.ocr_service.get_ocr', return_value=mock_paddle_ocr):
            result = await perform_ocr_async(sample_image_bytes)
            
            assert result == []

    async def test_perform_ocr_async_prediction_error(self, sample_image_bytes, mock_paddle_ocr):
        """Test async OCR with prediction error"""
        mock_paddle_ocr.predict.side_effect = Exception("Prediction failed")
        
        with patch('src.services.ocr_service.get_ocr', return_value=mock_paddle_ocr):
            with pytest.raises(OCRProcessingError):
                await perform_ocr_async(sample_image_bytes)
