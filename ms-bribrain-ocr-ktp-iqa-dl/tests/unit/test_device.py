"""Unit tests for src.core.device module."""

from unittest.mock import patch, MagicMock

import torch

from src.core.device import get_device


class TestGetDevice:
    def test_force_cpu_returns_cpu(self):
        device = get_device(force_cpu=True)
        assert device.type == "cpu"

    @patch("src.core.device.torch.cuda.is_available", return_value=False)
    def test_no_cuda_returns_cpu(self, _mock):
        device = get_device(force_cpu=False)
        assert device.type == "cpu"

    @patch("src.core.device.torch.cuda.is_available", return_value=True)
    @patch("src.core.device.torch.cuda.get_device_name", return_value="Test GPU")
    @patch("src.core.device.torch.cuda.get_device_properties")
    @patch("src.core.device.torch.version", create=True)
    def test_cuda_available_returns_cuda(self, mock_version, mock_props, _name, _avail):
        mock_version.cuda = "12.0"
        mock_props.return_value.total_memory = 8 * (1024**3)
        device = get_device(force_cpu=False)
        assert device.type == "cuda"

    def test_default_not_force_cpu(self):
        # default should not force CPU
        device = get_device()
        assert device.type in ("cpu", "cuda")
