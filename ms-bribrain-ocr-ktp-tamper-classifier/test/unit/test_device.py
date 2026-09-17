"""
Unit tests for device management module
"""

import pytest
from unittest.mock import Mock, patch

from src.core.device import DeviceManager, get_device, log_device_info


@pytest.mark.unit
class TestDeviceManager:
    """Tests for DeviceManager class"""
    
    @patch("src.core.device.get_config")
    def test_initialization(self, mock_get_config):
        """Test DeviceManager initialization"""
        mock_config = Mock()
        mock_config.device = Mock()
        mock_get_config.return_value = mock_config
        
        dm = DeviceManager()
        
        assert dm.config == mock_config
        assert dm._device is None
    
    @patch("src.core.device.get_config")
    @patch("torch.cuda.is_available")
    def test_get_optimal_device_auto_cuda(self, mock_cuda, mock_get_config):
        """Test auto device selection with CUDA available"""
        mock_config = Mock()
        mock_config.device.preferred = "auto"
        mock_config.device.cuda_device_id = 0
        mock_get_config.return_value = mock_config
        
        mock_cuda.return_value = True
        
        dm = DeviceManager()
        device = dm.get_optimal_device()
        
        assert device == "cuda:0"
    
    @patch("src.core.device.get_config")
    @patch("torch.cuda.is_available")
    def test_get_optimal_device_cpu_fallback(self, mock_cuda, mock_get_config):
        """Test CPU fallback when CUDA not available"""
        mock_config = Mock()
        mock_config.device.preferred = "cuda"
        mock_config.device.allow_cpu_fallback = True
        mock_get_config.return_value = mock_config
        
        mock_cuda.return_value = False
        
        dm = DeviceManager()
        device = dm.get_optimal_device()
        
        assert device == "cpu"
    
    @patch("src.core.device.get_config")
    @patch("torch.cuda.is_available")
    def test_get_optimal_device_raises_without_fallback(self, mock_cuda, mock_get_config):
        """Test that RuntimeError is raised when fallback disabled"""
        mock_config = Mock()
        mock_config.device.preferred = "cuda"
        mock_config.device.allow_cpu_fallback = False
        mock_get_config.return_value = mock_config
        
        mock_cuda.return_value = False
        
        dm = DeviceManager()
        
        with pytest.raises(RuntimeError, match="CUDA requested but not available"):
            dm.get_optimal_device()
    
    @patch("src.core.device.get_config")
    def test_get_optimal_device_cpu_explicit(self, mock_get_config):
        """Test explicit CPU device selection"""
        mock_config = Mock()
        mock_config.device.preferred = "cpu"
        mock_get_config.return_value = mock_config
        
        dm = DeviceManager()
        device = dm.get_optimal_device()
        
        assert device == "cpu"
    
    @patch("src.core.device.get_config")
    def test_get_optimal_device_cached(self, mock_get_config):
        """Test that device is cached after first call"""
        mock_config = Mock()
        mock_config.device.preferred = "cpu"
        mock_get_config.return_value = mock_config
        
        dm = DeviceManager()
        device1 = dm.get_optimal_device()
        device2 = dm.get_optimal_device()
        
        assert device1 == device2
        assert device1 == "cpu"
    
    @patch("src.core.device.get_config")
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_properties")
    def test_get_device_info_cuda(self, mock_props, mock_cuda, mock_get_config):
        """Test getting device info for CUDA"""
        mock_config = Mock()
        mock_config.device.preferred = "cuda"
        mock_config.device.cuda_device_id = 0
        mock_get_config.return_value = mock_config
        
        mock_cuda.return_value = True
        mock_device_props = Mock()
        mock_device_props.name = "Tesla V100"
        mock_device_props.total_memory = 16 * 1024**3  # 16GB
        mock_props.return_value = mock_device_props
        
        dm = DeviceManager()
        dm._device = "cuda:0"
        
        info = dm.get_device_info()
        
        assert info["device"] == "cuda:0"
        assert info["type"] == "cuda"
        assert "name" in info


@pytest.mark.unit
class TestGetDevice:
    """Tests for get_device helper function"""
    
    @patch("src.core.device.DeviceManager")
    def test_get_device(self, mock_dm_class):
        """Test get_device helper function"""
        mock_dm = Mock()
        mock_dm.get_optimal_device.return_value = "cpu"
        mock_dm_class.return_value = mock_dm
        
        device = get_device()
        
        assert device == "cpu"


@pytest.mark.unit
class TestLogDeviceInfo:
    """Tests for log_device_info function"""
    
    @patch("src.core.device.get_device")
    @patch("torch.cuda.is_available")
    def test_log_device_info_cpu(self, mock_cuda, mock_get_device):
        """Test logging device info for CPU"""
        mock_cuda.return_value = False
        mock_get_device.return_value = "cpu"
        
        # Should not raise any errors
        log_device_info("cpu")
    
    @patch("src.core.device.get_device")
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_properties")
    def test_log_device_info_cuda(self, mock_props, mock_cuda, mock_get_device):
        """Test logging device info for CUDA"""
        mock_cuda.return_value = True
        mock_get_device.return_value = "cuda:0"

        mock_device_props = Mock()
        mock_device_props.name = "Tesla V100"
        mock_device_props.total_memory = 16 * 1024**3
        mock_props.return_value = mock_device_props

        # Should not raise any errors
        log_device_info("cuda:0")


@pytest.mark.unit
class TestDeviceManagerMPS:
    """Tests for MPS device paths in DeviceManager."""

    @patch("src.core.device.get_config")
    @patch("platform.system", return_value="Darwin")
    @patch("torch.backends.mps.is_available", return_value=True, create=True)
    @patch("torch.cuda.is_available", return_value=False)
    def test_auto_detect_mps(self, mock_cuda, mock_mps, mock_sys, mock_get_config):
        """Test auto-detect selects MPS on macOS."""
        mock_config = Mock()
        mock_config.device.preferred = "auto"
        mock_config.device.cuda_device_id = 0
        mock_get_config.return_value = mock_config

        dm = DeviceManager()
        device = dm.get_optimal_device()
        assert device == "mps"

    @patch("src.core.device.get_config")
    @patch("platform.system", return_value="Linux")
    @patch("torch.cuda.is_available", return_value=False)
    def test_auto_detect_cpu_fallback(self, mock_cuda, mock_sys, mock_get_config):
        """Test auto-detect falls back to CPU."""
        mock_config = Mock()
        mock_config.device.preferred = "auto"
        mock_config.device.cuda_device_id = 0
        mock_get_config.return_value = mock_config

        dm = DeviceManager()
        device = dm.get_optimal_device()
        assert device == "cpu"

    @patch("src.core.device.get_config")
    @patch("platform.system", return_value="Darwin")
    @patch("torch.backends.mps.is_available", return_value=True, create=True)
    def test_mps_preferred(self, mock_mps, mock_sys, mock_get_config):
        """Test MPS preferred when available."""
        mock_config = Mock()
        mock_config.device.preferred = "mps"
        mock_get_config.return_value = mock_config

        dm = DeviceManager()
        device = dm.get_optimal_device()
        assert device == "mps"

    @patch("src.core.device.get_config")
    @patch("platform.system", return_value="Linux")
    def test_mps_fallback_to_cpu(self, mock_sys, mock_get_config):
        """Test MPS falls back to CPU when not on macOS."""
        mock_config = Mock()
        mock_config.device.preferred = "mps"
        mock_config.device.allow_cpu_fallback = True
        mock_get_config.return_value = mock_config

        dm = DeviceManager()
        device = dm.get_optimal_device()
        assert device == "cpu"

    @patch("src.core.device.get_config")
    @patch("platform.system", return_value="Linux")
    def test_mps_no_fallback_raises(self, mock_sys, mock_get_config):
        """Test MPS raises when no fallback allowed."""
        mock_config = Mock()
        mock_config.device.preferred = "mps"
        mock_config.device.allow_cpu_fallback = False
        mock_get_config.return_value = mock_config

        dm = DeviceManager()
        with pytest.raises(RuntimeError, match="MPS requested but not available"):
            dm.get_optimal_device()

    @patch("src.core.device.get_config")
    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_preferred_selects_cuda(self, mock_cuda, mock_get_config):
        """Test CUDA preferred path selects cuda device."""
        mock_config = Mock()
        mock_config.device.preferred = "cuda"
        mock_config.device.cuda_device_id = 0
        mock_get_config.return_value = mock_config

        dm = DeviceManager()
        device = dm.get_optimal_device()
        assert device == "cuda:0"


@pytest.mark.unit
class TestDeviceManagerInfo:
    """Tests for device info and memory management."""

    @patch("src.core.device.get_config")
    def test_get_device_info_cpu(self, mock_get_config):
        """Test get_device_info for CPU device."""
        mock_config = Mock()
        mock_config.device.preferred = "cpu"
        mock_get_config.return_value = mock_config

        dm = DeviceManager()
        info = dm.get_device_info()
        assert info["device"] == "cpu"
        assert info["type"] == "cpu"
        assert "cores" in info
        assert "platform" in info

    @patch("src.core.device.get_config")
    def test_get_device_info_cached(self, mock_get_config):
        """Test get_device_info returns cached result."""
        mock_config = Mock()
        mock_config.device.preferred = "cpu"
        mock_get_config.return_value = mock_config

        dm = DeviceManager()
        info1 = dm.get_device_info()
        info2 = dm.get_device_info()
        assert info1 is info2

    @patch("src.core.device.get_config")
    @patch("platform.system", return_value="Darwin")
    @patch("torch.backends.mps.is_available", return_value=True, create=True)
    @patch("torch.cuda.is_available", return_value=False)
    def test_get_device_info_mps(self, mock_cuda, mock_mps, mock_sys, mock_get_config):
        """Test get_device_info for MPS device."""
        mock_config = Mock()
        mock_config.device.preferred = "mps"
        mock_get_config.return_value = mock_config

        dm = DeviceManager()
        info = dm.get_device_info()
        assert info["device"] == "mps"
        assert info["name"] == "Apple MPS (Metal Performance Shaders)"

    @patch("src.core.device.get_config")
    def test_cleanup_memory_cpu_noop(self, mock_get_config):
        """Test cleanup_memory is a no-op for CPU."""
        mock_config = Mock()
        mock_config.device.preferred = "cpu"
        mock_get_config.return_value = mock_config

        dm = DeviceManager()
        dm.cleanup_memory()  # should not raise

    @patch("src.core.device.get_config")
    def test_set_memory_fraction_non_cuda(self, mock_get_config):
        """Test set_memory_fraction is a no-op for non-CUDA."""
        mock_config = Mock()
        mock_config.device.preferred = "cpu"
        mock_get_config.return_value = mock_config

        dm = DeviceManager()
        dm.set_memory_fraction(0.8)  # should not raise

    @patch("src.core.device.get_config")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    def test_cleanup_memory_cuda(self, mock_empty, mock_cuda, mock_get_config):
        """Test cleanup_memory calls empty_cache for CUDA."""
        mock_config = Mock()
        mock_config.device.preferred = "cuda"
        mock_config.device.cuda_device_id = 0
        mock_get_config.return_value = mock_config

        dm = DeviceManager()
        dm.cleanup_memory()
        mock_empty.assert_called_once()

    @patch("src.core.device.get_config")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_per_process_memory_fraction")
    def test_set_memory_fraction_cuda(self, mock_set_frac, mock_cuda, mock_get_config):
        """Test set_memory_fraction on CUDA device."""
        mock_config = Mock()
        mock_config.device.preferred = "cuda"
        mock_config.device.cuda_device_id = 0
        mock_config.device.memory_fraction = 0.9
        mock_get_config.return_value = mock_config

        dm = DeviceManager()
        dm.set_memory_fraction()
        mock_set_frac.assert_called_once_with(0.9, 0)

    @patch("src.core.device.get_config")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_per_process_memory_fraction")
    def test_set_memory_fraction_cuda_explicit(self, mock_set_frac, mock_cuda, mock_get_config):
        """Test set_memory_fraction with explicit value."""
        mock_config = Mock()
        mock_config.device.preferred = "cuda"
        mock_config.device.cuda_device_id = 0
        mock_get_config.return_value = mock_config

        dm = DeviceManager()
        dm.set_memory_fraction(0.5)
        mock_set_frac.assert_called_once_with(0.5, 0)


@pytest.mark.unit
class TestCleanupDeviceMemory:
    """Tests for cleanup_device_memory convenience function."""

    @patch("src.core.device._device_manager", None)
    @patch("src.core.device.get_config")
    def test_cleanup_device_memory(self, mock_get_config):
        """Test cleanup_device_memory convenience function."""
        from src.core.device import cleanup_device_memory
        mock_config = Mock()
        mock_config.device.preferred = "cpu"
        mock_get_config.return_value = mock_config

        cleanup_device_memory()  # should not raise
