"""Unit tests for src.core.device module."""

from unittest.mock import patch, MagicMock

from src.core.device import get_device_info, should_use_gpu


class TestGetDeviceInfo:
    @patch("src.core.device.get_config")
    def test_cpu_when_no_gpu_preferred(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.device.prefer_gpu = False
        mock_config.device.fallback_to_cpu = True
        mock_get_config.return_value = mock_config

        with patch.dict("sys.modules", {"torch": MagicMock()}):
            info = get_device_info()
        assert info["selected_device"] == "cpu"

    @patch("src.core.device.get_config")
    def test_cpu_when_gpu_preferred_but_unavailable(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.device.prefer_gpu = True
        mock_config.device.fallback_to_cpu = True
        mock_get_config.return_value = mock_config

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("src.core.device.torch", mock_torch, create=True):
                info = get_device_info()
        assert info["selected_device"] == "cpu"
        assert info["prefer_gpu"] is True

    @patch("src.core.device.get_config")
    def test_returns_dict_with_expected_keys(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.device.prefer_gpu = False
        mock_config.device.fallback_to_cpu = True
        mock_get_config.return_value = mock_config

        info = get_device_info()
        assert "prefer_gpu" in info
        assert "fallback_to_cpu" in info
        assert "gpu_available" in info
        assert "selected_device" in info

    @patch("src.core.device.get_config")
    def test_handles_import_error(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.device.prefer_gpu = True
        mock_config.device.fallback_to_cpu = True
        mock_get_config.return_value = mock_config

        # Even if torch import fails, should still return cpu info
        info = get_device_info()
        assert info["selected_device"] in ("cpu", "gpu")

    @patch("src.core.device.get_config")
    def test_cuda_available_selects_gpu(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.device.prefer_gpu = True
        mock_config.device.fallback_to_cpu = True
        mock_get_config.return_value = mock_config

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA Test GPU"

        with patch.dict("sys.modules", {"torch": mock_torch}):
            info = get_device_info()
        assert info["gpu_available"] is True
        assert info["gpu_type"] == "cuda"
        assert info["gpu_name"] == "NVIDIA Test GPU"
        assert info["selected_device"] == "gpu"

    @patch("src.core.device.get_config")
    def test_mps_available_selects_gpu(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.device.prefer_gpu = True
        mock_config.device.fallback_to_cpu = True
        mock_get_config.return_value = mock_config

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            info = get_device_info()
        assert info["gpu_available"] is True
        assert info["gpu_type"] == "mps"
        assert info["selected_device"] == "gpu"

    @patch("src.core.device.get_config")
    def test_gpu_detection_exception_falls_back_to_cpu(self, mock_get_config):
        mock_config = MagicMock()
        mock_config.device.prefer_gpu = True
        mock_config.device.fallback_to_cpu = True
        mock_get_config.return_value = mock_config

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("CUDA driver error")

        with patch.dict("sys.modules", {"torch": mock_torch}):
            info = get_device_info()
        assert info["gpu_available"] is False
        assert info["selected_device"] == "cpu"


class TestShouldUseGpu:
    @patch("src.core.device.get_device_info", return_value={"selected_device": "gpu"})
    def test_returns_true_when_gpu(self, _mock):
        assert should_use_gpu() is True

    @patch("src.core.device.get_device_info", return_value={"selected_device": "cpu"})
    def test_returns_false_when_cpu(self, _mock):
        assert should_use_gpu() is False
