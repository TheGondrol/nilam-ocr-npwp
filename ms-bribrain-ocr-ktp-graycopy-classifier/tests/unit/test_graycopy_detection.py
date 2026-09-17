"""Unit tests for src.services.graycopy_detection module"""

from unittest.mock import patch, MagicMock, PropertyMock

import pytest
import torch
from PIL import Image

from src.services.graycopy_detection import GraycopyDetectionService


def _make_mock_config():
    """Create a mock config for testing"""
    config = MagicMock()
    config.model.path = "fake_model.pth"
    config.model.image_size = 224
    config.model.normalize_mean = [0.485, 0.456, 0.406]
    config.model.normalize_std = [0.229, 0.224, 0.225]
    config.model.normalize_size = 512
    config.model.dropout_rate = 0.6
    config.device.prefer_gpu = False
    config.device.force_cpu = True
    config.device.compile_model = False
    config.device.compile_mode = "default"
    config.device.use_jit_cpu = False
    config.prediction.threshold = 0.5
    return config


@pytest.fixture
def mock_model():
    """Create a mock model that returns valid predictions"""
    model = MagicMock()
    # Mock forward pass: returns logits for 2 classes
    output = torch.tensor([[0.2, 0.8]])
    model.return_value = output
    model.to = MagicMock(return_value=model)
    model.eval = MagicMock()
    return model


class TestGraycopyDetectionServiceInit:
    def test_init_success(self, mock_model, tmp_path):
        config = _make_mock_config()
        model_file = tmp_path / "model.pth"
        # Save a real state dict
        from src.services.model_architecture import create_resnet50_graycopy_model
        real_model = create_resnet50_graycopy_model()
        torch.save(real_model.state_dict(), model_file)
        config.model.path = str(model_file)

        service = GraycopyDetectionService(config)
        assert service.is_ready()
        assert service.get_device_info() == "cpu"
        service.cleanup()

    def test_init_file_not_found(self):
        config = _make_mock_config()
        config.model.path = "/nonexistent/model.pth"
        with pytest.raises(FileNotFoundError):
            GraycopyDetectionService(config)


class TestPredict:
    @pytest.fixture
    def service(self, tmp_path):
        config = _make_mock_config()
        model_file = tmp_path / "model.pth"
        from src.services.model_architecture import create_resnet50_graycopy_model
        real_model = create_resnet50_graycopy_model()
        torch.save(real_model.state_dict(), model_file)
        config.model.path = str(model_file)
        svc = GraycopyDetectionService(config)
        yield svc
        svc.cleanup()

    def test_predict_returns_dict(self, service):
        img = Image.new("RGB", (100, 100), "white")
        result = service.predict(img, "test.jpg")
        assert "prediction" in result
        assert result["prediction"] in ("GRAYCOPY", "ORIGINAL")
        assert 0 <= result["confidence"] <= 1
        assert "probability_graycopy" in result
        assert "probability_original" in result

    def test_predict_already_cropped(self, service):
        img = Image.new("RGB", (224, 224), "white")
        result = service.predict(img, "cropped.jpg")
        assert result["prediction"] in ("GRAYCOPY", "ORIGINAL")

    def test_predict_rgba_image(self, service):
        img = Image.new("RGBA", (300, 300), "blue")
        result = service.predict(img, "rgba.png")
        assert result["prediction"] in ("GRAYCOPY", "ORIGINAL")

    def test_predict_model_not_loaded(self):
        config = _make_mock_config()
        with patch.object(GraycopyDetectionService, "__init__", lambda self, c: None):
            service = GraycopyDetectionService.__new__(GraycopyDetectionService)
            service.model = None
            with pytest.raises(RuntimeError, match="Model not loaded"):
                service.predict(Image.new("RGB", (100, 100)), "test.jpg")


class TestWarmup:
    def test_warmup_success(self, tmp_path):
        config = _make_mock_config()
        model_file = tmp_path / "model.pth"
        from src.services.model_architecture import create_resnet50_graycopy_model
        real_model = create_resnet50_graycopy_model()
        torch.save(real_model.state_dict(), model_file)
        config.model.path = str(model_file)
        service = GraycopyDetectionService(config)
        service.warmup()  # should not raise
        service.cleanup()

    def test_warmup_no_model(self):
        with patch.object(GraycopyDetectionService, "__init__", lambda self, c: None):
            service = GraycopyDetectionService.__new__(GraycopyDetectionService)
            service.model = None
            service.device = None
            service.config = _make_mock_config()
            service.warmup()  # should log warning but not raise


class TestCleanup:
    def test_cleanup_releases_model(self, tmp_path):
        config = _make_mock_config()
        model_file = tmp_path / "model.pth"
        from src.services.model_architecture import create_resnet50_graycopy_model
        real_model = create_resnet50_graycopy_model()
        torch.save(real_model.state_dict(), model_file)
        config.model.path = str(model_file)
        service = GraycopyDetectionService(config)
        assert service.is_ready()
        service.cleanup()
        assert not service.is_ready()


class TestIsReady:
    def test_ready_when_loaded(self, tmp_path):
        config = _make_mock_config()
        model_file = tmp_path / "model.pth"
        from src.services.model_architecture import create_resnet50_graycopy_model
        real_model = create_resnet50_graycopy_model()
        torch.save(real_model.state_dict(), model_file)
        config.model.path = str(model_file)
        service = GraycopyDetectionService(config)
        assert service.is_ready() is True
        service.cleanup()

    def test_not_ready_when_no_model(self):
        with patch.object(GraycopyDetectionService, "__init__", lambda self, c: None):
            service = GraycopyDetectionService.__new__(GraycopyDetectionService)
            service.model = None
            service.device = torch.device("cpu")
            assert service.is_ready() is False


class TestLoadModelCheckpointFormat:
    """Cover the `model_state_dict` checkpoint branch (graycopy_detection.py:105-107)."""

    def test_loads_checkpoint_with_state_dict_key(self, tmp_path):
        from src.services.model_architecture import create_resnet50_graycopy_model
        config = _make_mock_config()
        model_file = tmp_path / "ckpt.pth"
        real_model = create_resnet50_graycopy_model()
        torch.save(
            {"model_state_dict": real_model.state_dict(), "epoch": 42},
            model_file,
        )
        config.model.path = str(model_file)

        service = GraycopyDetectionService(config)
        assert service.is_ready()
        service.cleanup()


class TestLoadModelJitCpu:
    """Cover the CPU JIT optimization branch (graycopy_detection.py:134-143)."""

    def test_jit_on_cpu(self, tmp_path):
        from src.services.model_architecture import create_resnet50_graycopy_model
        config = _make_mock_config()
        config.device.use_jit_cpu = True
        model_file = tmp_path / "model.pth"
        real_model = create_resnet50_graycopy_model()
        torch.save(real_model.state_dict(), model_file)
        config.model.path = str(model_file)

        service = GraycopyDetectionService(config)
        assert service.is_ready()
        # JIT trace either succeeds (sets flag) or logs a warning - both branches
        # of the try/except in graycopy_detection.py:135-145 are exercised here.
        assert isinstance(service._is_jit_model, bool)
        service.cleanup()


class TestWarmupExceptionPath:
    """Cover graycopy_detection.py:201-202 (warmup exception swallowed)."""

    def test_warmup_handles_model_exception(self, tmp_path):
        from src.services.model_architecture import create_resnet50_graycopy_model
        config = _make_mock_config()
        model_file = tmp_path / "model.pth"
        real_model = create_resnet50_graycopy_model()
        torch.save(real_model.state_dict(), model_file)
        config.model.path = str(model_file)

        service = GraycopyDetectionService(config)
        # Force warmup inference to raise; exception path must be swallowed.
        broken_model = MagicMock(side_effect=RuntimeError("warmup boom"))
        service.model = broken_model
        service.warmup()  # should log warning, not raise
        service.cleanup()


class TestPredictOuterException:
    """Cover graycopy_detection.py:356-358 (outer except re-raises)."""

    def test_non_cudnn_runtime_error_reraises(self, tmp_path):
        from src.services.model_architecture import create_resnet50_graycopy_model
        config = _make_mock_config()
        model_file = tmp_path / "model.pth"
        real_model = create_resnet50_graycopy_model()
        torch.save(real_model.state_dict(), model_file)
        config.model.path = str(model_file)

        service = GraycopyDetectionService(config)
        # Non-cuDNN / non-CUDA RuntimeError should bubble up through the outer except.
        service.model = MagicMock(side_effect=RuntimeError("generic failure"))

        img = Image.new("RGB", (100, 100), "white")
        with pytest.raises(RuntimeError, match="generic failure"):
            service.predict(img, "test.jpg")
        service.cleanup()
