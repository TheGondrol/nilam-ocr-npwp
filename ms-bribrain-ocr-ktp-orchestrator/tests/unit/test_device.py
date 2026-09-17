"""
Unit tests for src/core/device module — GPU detection and device selection.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestDetectGpu:

    def test_detect_cuda_available(self):
        """detect_gpu returns cuda when CUDA is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from src.core.device import detect_gpu
            available, device = detect_gpu()
            assert available is True
            assert device == "cuda"

    def test_detect_mps_available(self):
        """detect_gpu returns mps when MPS is available and CUDA is not."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from src.core.device import detect_gpu
            available, device = detect_gpu()
            assert available is True
            assert device == "mps"

    def test_detect_no_gpu(self):
        """detect_gpu returns cpu when no GPU available."""
        with patch.dict("sys.modules", {"torch": None}):
            from src.core.device import detect_gpu
            available, device = detect_gpu()
            assert available is False
            assert device == "cpu"

    def test_detect_gpu_cuda_exception(self):
        """detect_gpu handles non-ImportError from CUDA check."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.side_effect = RuntimeError("CUDA error")
        mock_torch.backends.mps.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            from src.core.device import detect_gpu
            available, device = detect_gpu()
            assert device in ("mps", "cpu")


class TestGetDevice:

    def test_prefer_gpu_with_cuda(self):
        """get_device returns cuda when prefer_gpu and CUDA available."""
        mock_settings = MagicMock()
        mock_settings.device.prefer_gpu = True
        mock_settings.device.fallback_to_cpu = True

        with patch("src.core.device.get_settings", return_value=mock_settings), \
             patch("src.core.device.detect_gpu", return_value=(True, "cuda")):
            from src.core.device import get_device
            assert get_device() == "cuda"

    def test_prefer_gpu_no_gpu_fallback(self):
        """get_device falls back to cpu when GPU preferred but not available."""
        mock_settings = MagicMock()
        mock_settings.device.prefer_gpu = True
        mock_settings.device.fallback_to_cpu = True

        with patch("src.core.device.get_settings", return_value=mock_settings), \
             patch("src.core.device.detect_gpu", return_value=(False, "cpu")):
            from src.core.device import get_device
            assert get_device() == "cpu"

    def test_prefer_gpu_no_gpu_no_fallback_raises(self):
        """get_device raises RuntimeError when GPU preferred, not available, no fallback."""
        mock_settings = MagicMock()
        mock_settings.device.prefer_gpu = True
        mock_settings.device.fallback_to_cpu = False

        with patch("src.core.device.get_settings", return_value=mock_settings), \
             patch("src.core.device.detect_gpu", return_value=(False, "cpu")):
            from src.core.device import get_device
            with pytest.raises(RuntimeError, match="GPU not available"):
                get_device()

    def test_no_prefer_gpu(self):
        """get_device returns cpu when GPU not preferred."""
        mock_settings = MagicMock()
        mock_settings.device.prefer_gpu = False

        with patch("src.core.device.get_settings", return_value=mock_settings), \
             patch("src.core.device.detect_gpu", return_value=(True, "cuda")):
            from src.core.device import get_device
            assert get_device() == "cpu"


class TestLogDeviceInfo:

    def test_log_device_info_error(self):
        """log_device_info handles exceptions gracefully."""
        with patch("src.core.device.get_device", side_effect=RuntimeError("test")), \
             patch("src.core.device.logger"):
            from src.core.device import log_device_info
            log_device_info()

    def test_log_device_info_with_gpu(self):
        """log_device_info logs GPU availability."""
        with patch("src.core.device.get_device", return_value="cuda"), \
             patch("src.core.device.detect_gpu", return_value=(True, "cuda")), \
             patch("src.core.device.logger") as mock_logger:
            from src.core.device import log_device_info
            log_device_info()
            assert mock_logger.info.call_count >= 2

    def test_log_device_info_no_gpu(self):
        """log_device_info logs no GPU message."""
        with patch("src.core.device.get_device", return_value="cpu"), \
             patch("src.core.device.detect_gpu", return_value=(False, "cpu")), \
             patch("src.core.device.logger") as mock_logger:
            from src.core.device import log_device_info
            log_device_info()
            log_messages = [str(c) for c in mock_logger.info.call_args_list]
            assert any("No GPU" in m for m in log_messages)
