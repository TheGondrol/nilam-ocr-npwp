"""Unit tests for src.core.device module"""

import pytest
from unittest.mock import patch, MagicMock
from src.core.device import detect_device, get_use_gpu


class TestDetectDevice:
    """Test cases for detect_device function"""

    def setup_method(self):
        """Reset cached device detection before each test"""
        import src.core.device
        src.core.device._detected_device = None

    def test_force_cpu_mode(self, mock_settings):
        """Test that CPU mode is forced when configured"""
        mock_settings.device_force_cpu = True
        
        with patch('src.core.device.settings', mock_settings):
            result = detect_device()
            assert result == "cpu"

    def test_cpu_preferred(self, mock_settings):
        """Test CPU mode when preferred in settings"""
        mock_settings.device_force_cpu = False
        mock_settings.device_preferred = "cpu"
        
        with patch('src.core.device.settings', mock_settings):
            result = detect_device()
            assert result == "cpu"

    def test_gpu_detected_and_available(self, mock_settings):
        """Test GPU mode when GPU is detected and available"""
        mock_settings.device_force_cpu = False
        mock_settings.device_preferred = "gpu"
        
        mock_paddle = MagicMock()
        mock_paddle.device.is_compiled_with_cuda.return_value = True
        mock_paddle.device.cuda.device_count.return_value = 1
        
        with patch('src.core.device.settings', mock_settings):
            with patch.dict('sys.modules', {'paddle': mock_paddle}):
                with patch('src.core.device.setup_logging'):
                    result = detect_device()
                    assert result == "gpu"

    def test_gpu_preferred_but_not_available(self, mock_settings):
        """Test fallback to CPU when GPU is preferred but not available"""
        mock_settings.device_force_cpu = False
        mock_settings.device_preferred = "gpu"
        
        mock_paddle = MagicMock()
        mock_paddle.device.is_compiled_with_cuda.return_value = False
        
        with patch('src.core.device.settings', mock_settings):
            with patch.dict('sys.modules', {'paddle': mock_paddle}):
                with patch('src.core.device.setup_logging'):
                    result = detect_device()
                    assert result == "cpu"

    def test_gpu_zero_devices(self, mock_settings):
        """Test fallback to CPU when GPU count is 0"""
        mock_settings.device_force_cpu = False
        mock_settings.device_preferred = "auto"
        
        mock_paddle = MagicMock()
        mock_paddle.device.is_compiled_with_cuda.return_value = True
        mock_paddle.device.cuda.device_count.return_value = 0
        
        with patch('src.core.device.settings', mock_settings):
            with patch.dict('sys.modules', {'paddle': mock_paddle}):
                with patch('src.core.device.setup_logging'):
                    result = detect_device()
                    assert result == "cpu"

    def test_auto_mode_with_gpu(self, mock_settings):
        """Test auto mode selects GPU when available"""
        mock_settings.device_force_cpu = False
        mock_settings.device_preferred = "auto"
        
        mock_paddle = MagicMock()
        mock_paddle.device.is_compiled_with_cuda.return_value = True
        mock_paddle.device.cuda.device_count.return_value = 1
        
        with patch('src.core.device.settings', mock_settings):
            with patch.dict('sys.modules', {'paddle': mock_paddle}):
                with patch('src.core.device.setup_logging'):
                    result = detect_device()
                    assert result == "gpu"

    def test_auto_mode_without_gpu(self, mock_settings):
        """Test auto mode falls back to CPU when GPU not available"""
        mock_settings.device_force_cpu = False
        mock_settings.device_preferred = "auto"
        
        mock_paddle = MagicMock()
        mock_paddle.device.is_compiled_with_cuda.return_value = False
        
        with patch('src.core.device.settings', mock_settings):
            with patch.dict('sys.modules', {'paddle': mock_paddle}):
                with patch('src.core.device.setup_logging'):
                    result = detect_device()
                    assert result == "cpu"

    def test_caching_behavior(self, mock_settings):
        """Test that device detection is cached"""
        mock_settings.device_force_cpu = False
        mock_settings.device_preferred = "cpu"
        
        with patch('src.core.device.settings', mock_settings):
            # First call
            result1 = detect_device()
            # Second call should return cached result
            result2 = detect_device()
            
            assert result1 == result2
            assert result1 == "cpu"

    def test_exception_during_gpu_detection(self, mock_settings):
        """Test fallback to CPU when GPU detection throws exception"""
        mock_settings.device_force_cpu = False
        mock_settings.device_preferred = "auto"
        
        mock_paddle = MagicMock()
        mock_paddle.device.is_compiled_with_cuda.side_effect = Exception("CUDA error")
        
        with patch('src.core.device.settings', mock_settings):
            with patch.dict('sys.modules', {'paddle': mock_paddle}):
                with patch('src.core.device.setup_logging'):
                    result = detect_device()
                    assert result == "cpu"


class TestGetUseGpu:
    """Test cases for get_use_gpu function"""

    def setup_method(self):
        """Reset cached device detection before each test"""
        import src.core.device
        src.core.device._detected_device = None

    def test_returns_true_for_gpu(self, mock_settings):
        """Test get_use_gpu returns True when device is GPU"""
        mock_settings.device_force_cpu = False
        mock_settings.device_preferred = "gpu"
        
        mock_paddle = MagicMock()
        mock_paddle.device.is_compiled_with_cuda.return_value = True
        mock_paddle.device.cuda.device_count.return_value = 1
        
        with patch('src.core.device.settings', mock_settings):
            with patch.dict('sys.modules', {'paddle': mock_paddle}):
                with patch('src.core.device.setup_logging'):
                    result = get_use_gpu()
                    assert result is True

    def test_returns_false_for_cpu(self, mock_settings):
        """Test get_use_gpu returns False when device is CPU"""
        mock_settings.device_force_cpu = True
        
        with patch('src.core.device.settings', mock_settings):
            result = get_use_gpu()
            assert result is False
