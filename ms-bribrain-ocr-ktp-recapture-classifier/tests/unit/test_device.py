"""Tests for device detection utilities."""

import torch
from unittest.mock import patch, MagicMock


class TestGetDevice:
    """Tests for get_device function."""

    def test_force_cpu(self):
        """Test forcing CPU device."""
        from src.core.device import get_device

        device = get_device(force_cpu=True)
        assert str(device) == "cpu"

    def test_cpu_fallback_when_no_cuda(self):
        """Test CPU fallback when CUDA is not available."""
        from src.core.device import get_device

        with patch('torch.cuda.is_available', return_value=False):
            device = get_device(force_cpu=False)
            assert str(device) == "cpu"

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_name', return_value="NVIDIA RTX 3090")
    @patch('torch.cuda.get_device_properties')
    def test_cuda_device_selected(self, mock_props, mock_name, mock_avail):
        """Test CUDA device selection when available."""
        from src.core.device import get_device

        mock_props.return_value = MagicMock(total_memory=24 * 1024**3)

        device = get_device(force_cpu=False)
        assert str(device) == "cuda"


class TestGetDeviceInfo:
    """Tests for get_device_info function."""

    def test_cpu_info(self):
        """Test device info when only CPU is available."""
        from src.core.device import get_device_info

        with patch('torch.cuda.is_available', return_value=False):
            info = get_device_info()

        assert info["device_type"] == "cpu"
        assert info["cuda_available"] is False

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_name', return_value="NVIDIA RTX 3090")
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.device_count', return_value=1)
    def test_cuda_info(self, mock_count, mock_props, mock_name, mock_avail):
        """Test device info when CUDA is available."""
        from src.core.device import get_device_info

        mock_props.return_value = MagicMock(total_memory=24 * 1024**3)

        info = get_device_info()

        assert info["device_type"] == "cuda"
        assert info["cuda_available"] is True
        assert info["gpu_name"] == "NVIDIA RTX 3090"
        assert info["gpu_memory_gb"] == 24.0
        assert info["device_count"] == 1
