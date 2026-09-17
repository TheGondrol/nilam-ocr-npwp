"""Unit tests for src.core.device module."""

from unittest.mock import patch, MagicMock

from src.core.device import get_device, log_device_info


class TestGetDevice:
    def test_force_cpu_config(self):
        with patch("src.core.device.config") as mock_cfg:
            mock_cfg.get.side_effect = lambda k, d=None: True if k == "device.force_cpu" else d
            device, info = get_device()
        assert device == "cpu"
        assert info["type"] == "cpu"

    def test_cuda_visible_devices_empty(self):
        with patch("src.core.device.config") as mock_cfg, \
             patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}):
            mock_cfg.get.side_effect = lambda k, d=None: False if k == "device.force_cpu" else d
            device, info = get_device()
        assert device == "cpu"

    def test_prefer_gpu_false(self):
        with patch("src.core.device.config") as mock_cfg, \
             patch.dict("os.environ", {}, clear=False):
            def side_effect(k, d=None):
                if k == "device.force_cpu":
                    return False
                if k == "device.prefer_gpu":
                    return False
                return d
            mock_cfg.get.side_effect = side_effect
            # Remove CUDA_VISIBLE_DEVICES if set
            import os
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            device, info = get_device()
        assert device == "cpu"

    def test_no_cuda_available(self):
        with patch("src.core.device.config") as mock_cfg, \
             patch("src.core.device.torch") as mock_torch:
            def side_effect(k, d=None):
                if k == "device.force_cpu":
                    return False
                if k == "device.prefer_gpu":
                    return True
                return d
            mock_cfg.get.side_effect = side_effect
            mock_torch.cuda.is_available.return_value = False
            import os
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            device, info = get_device()
        assert device == "cpu"


class TestLogDeviceInfo:
    def test_log_cpu(self):
        info = {"type": "cpu", "name": "CPU", "cuda_available": False}
        log_device_info("cpu", info)  # Should not raise

    def test_log_cuda(self):
        info = {
            "type": "cuda",
            "name": "Test GPU",
            "cuda_available": True,
            "memory_gb": 8.0,
            "cuda_version": "12.0",
        }
        log_device_info("cuda", info)  # Should not raise
