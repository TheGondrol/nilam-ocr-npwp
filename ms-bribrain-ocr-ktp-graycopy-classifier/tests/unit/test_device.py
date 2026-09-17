"""Unit tests for src.core.device module"""

from unittest.mock import patch, MagicMock

import torch

from src.core.device import get_device, is_cuda_available, get_device_info


class TestGetDevice:
    def test_force_cpu(self):
        device = get_device(prefer_gpu=True, force_cpu=True)
        assert device.type == "cpu"

    def test_cpu_when_no_gpu(self):
        with patch("src.core.device.torch.cuda.is_available", return_value=False):
            device = get_device(prefer_gpu=True, force_cpu=False)
            assert device.type == "cpu"

    def test_cpu_preferred(self):
        device = get_device(prefer_gpu=False, force_cpu=False)
        assert device.type == "cpu"

    def test_gpu_when_available(self):
        with patch("src.core.device.torch.cuda.is_available", return_value=True):
            with patch("src.core.device.torch.cuda.get_device_name", return_value="RTX 3090"):
                mock_props = MagicMock()
                mock_props.total_memory = 8 * 1024**3
                with patch("src.core.device.torch.cuda.get_device_properties", return_value=mock_props):
                    with patch("src.core.device.torch.version") as mock_ver:
                        mock_ver.cuda = "12.0"
                        device = get_device(prefer_gpu=True, force_cpu=False)
                        assert device.type == "cuda"


class TestIsCudaAvailable:
    def test_returns_bool(self):
        result = is_cuda_available()
        assert isinstance(result, bool)


class TestGetDeviceInfo:
    def test_cpu_device(self):
        device = torch.device("cpu")
        info = get_device_info(device)
        assert info["type"] == "cpu"
        assert info["is_cuda"] is False
        assert "name" not in info

    def test_cuda_device(self):
        device = torch.device("cuda")
        with patch("src.core.device.torch.cuda.get_device_name", return_value="RTX 3090"):
            mock_props = MagicMock()
            mock_props.total_memory = 8 * 1024**3
            with patch("src.core.device.torch.cuda.get_device_properties", return_value=mock_props):
                with patch("src.core.device.torch.version") as mock_ver:
                    mock_ver.cuda = "12.0"
                    info = get_device_info(device)
                    assert info["type"] == "cuda"
                    assert info["is_cuda"] is True
                    assert info["name"] == "RTX 3090"
                    assert "memory_gb" in info
