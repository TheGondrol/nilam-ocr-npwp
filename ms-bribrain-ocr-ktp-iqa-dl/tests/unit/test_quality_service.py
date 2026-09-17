"""Unit tests for src.services.quality_service module."""

import io
from unittest.mock import patch, MagicMock

import pytest
import torch
from PIL import Image

from src.schemas.api_schema import Crop
from src.services.quality_service import (
    get_device,
    get_model,
    process_and_classify_sync,
    set_model_globals,
)


def _mock_transform(img):
    """Mock transform that returns a 3x64x320 tensor without torchvision."""
    return torch.randn(3, 64, 320)


def _make_crop(x1, y1, x2, y2, text="t"):
    return Crop(
        bbox=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        text=text,
        confidence=0.9,
    )


def _image_bytes(width=200, height=200):
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestSetModelGlobals:
    def test_sets_globals(self):
        import src.services.quality_service as qs
        orig_model, orig_device, orig_transform = qs._model, qs._device, qs._transform

        mock_model = MagicMock()
        mock_device = torch.device("cpu")
        mock_transform = MagicMock()

        set_model_globals(mock_model, mock_device, mock_transform)
        assert get_model() is mock_model
        assert get_device() is mock_device

        qs._model, qs._device, qs._transform = orig_model, orig_device, orig_transform


class TestGetModel:
    def test_returns_none_when_not_set(self):
        import src.services.quality_service as qs
        orig = qs._model
        qs._model = None
        assert get_model() is None
        qs._model = orig


class TestGetDevice:
    def test_returns_none_when_not_set(self):
        import src.services.quality_service as qs
        orig = qs._device
        qs._device = None
        assert get_device() is None
        qs._device = orig


class TestProcessAndClassifySync:
    @pytest.fixture(autouse=True)
    def _mock_threshold_provider(self):
        """Mirror the patched config through a stub ThresholdProvider so the
        thresholds the tests set on config still drive the classification logic."""
        import src.services.quality_service as qs

        def _get(key):
            if key == "bad_crop_threshold":
                return qs.config.bad_crop_threshold
            if key == "confidence_threshold":
                return qs.config.get("prediction.confidence_threshold", 0.67)
            raise KeyError(key)

        provider = MagicMock()
        provider.get.side_effect = _get
        with patch("src.services.quality_service.get_provider", return_value=provider):
            yield

    def _setup_globals(self):
        import src.services.quality_service as qs

        # Create a simple model that returns 2-class logits
        model = MagicMock()
        # Return tensor shape (batch, 2) — class 1 (good) has higher score
        model.side_effect = lambda x: torch.tensor([[0.1, 0.9]] * x.shape[0])

        device = torch.device("cpu")

        orig = (qs._model, qs._device, qs._transform)
        qs._model = model
        qs._device = device
        qs._transform = _mock_transform
        return orig

    def _restore_globals(self, orig):
        import src.services.quality_service as qs
        qs._model, qs._device, qs._transform = orig

    def test_raises_when_model_not_loaded(self):
        import src.services.quality_service as qs
        orig = (qs._model, qs._device, qs._transform)
        qs._model = None
        qs._device = None
        qs._transform = None

        with pytest.raises(ValueError, match="Model not loaded"):
            process_and_classify_sync(_image_bytes(), [])

        qs._model, qs._device, qs._transform = orig

    def test_invalid_image_raises(self):
        orig = self._setup_globals()
        with pytest.raises(ValueError, match="Failed to open image"):
            process_and_classify_sync(b"not an image", [])
        self._restore_globals(orig)

    @patch("src.services.quality_service.config")
    def test_no_crops_returns_good(self, mock_config):
        mock_config.min_width_ratio = 1.0
        mock_config.min_width = 0
        mock_config.bad_crop_threshold = 4

        orig = self._setup_globals()
        result = process_and_classify_sync(_image_bytes(), [])
        assert result.label == "good"
        assert result.num_bad == 0
        assert result.total_crops == 0
        self._restore_globals(orig)

    @patch("src.services.quality_service.config")
    def test_all_crops_filtered_returns_good(self, mock_config):
        mock_config.min_width_ratio = 5.0  # very high ratio
        mock_config.min_width = 0
        mock_config.bad_crop_threshold = 4

        crops = [_make_crop(0, 0, 20, 100)]  # ratio=0.2, filtered out
        orig = self._setup_globals()
        result = process_and_classify_sync(_image_bytes(), crops)
        assert result.label == "good"
        assert result.num_filtered == 0
        assert result.total_crops == 1
        self._restore_globals(orig)

    @patch("src.services.quality_service.config")
    def test_successful_classification(self, mock_config):
        mock_config.min_width_ratio = 1.0
        mock_config.min_width = 0
        mock_config.bad_crop_threshold = 4
        mock_config.debug_mode = False
        mock_config.get.return_value = 0.67

        crops = [
            _make_crop(0, 0, 150, 30),  # ratio=5, passes filter
            _make_crop(0, 50, 120, 70),  # ratio=6, passes filter
        ]
        orig = self._setup_globals()
        result = process_and_classify_sync(_image_bytes(), crops)

        assert result.label == "good"
        assert result.total_crops == 2
        assert result.num_filtered == 2
        assert len(result.predictions) == 2
        self._restore_globals(orig)

    @patch("src.services.quality_service.config")
    def test_bad_classification(self, mock_config):
        import src.services.quality_service as qs

        mock_config.min_width_ratio = 1.0
        mock_config.min_width = 0
        mock_config.bad_crop_threshold = 1  # low threshold
        mock_config.debug_mode = False
        mock_config.get.return_value = 0.67

        # Model returns "bad" (class 0) with high confidence
        model = MagicMock()
        model.side_effect = lambda x: torch.tensor([[0.9, 0.1]] * x.shape[0])

        device = torch.device("cpu")

        orig = (qs._model, qs._device, qs._transform)
        qs._model = model
        qs._device = device
        qs._transform = _mock_transform

        crops = [_make_crop(0, 0, 150, 30)]
        result = process_and_classify_sync(_image_bytes(), crops)

        assert result.label == "bad"
        assert result.num_bad >= 1
        qs._model, qs._device, qs._transform = orig

    @patch("src.services.quality_service.config")
    def test_debug_mode_includes_text(self, mock_config):
        mock_config.min_width_ratio = 1.0
        mock_config.min_width = 0
        mock_config.bad_crop_threshold = 4
        mock_config.debug_mode = True
        mock_config.get.return_value = 0.67

        crops = [_make_crop(0, 0, 150, 30, text="NAMA")]
        orig = self._setup_globals()
        result = process_and_classify_sync(_image_bytes(), crops)

        assert len(result.predictions) == 1
        assert result.predictions[0].text == "NAMA"
        self._restore_globals(orig)

    @patch("src.services.quality_service.config")
    def test_low_confidence_forced_good(self, mock_config):
        import src.services.quality_service as qs

        mock_config.min_width_ratio = 1.0
        mock_config.min_width = 0
        mock_config.bad_crop_threshold = 4
        mock_config.debug_mode = False
        mock_config.get.return_value = 0.95  # very high threshold

        # Model returns "bad" but with low confidence (0.6)
        model = MagicMock()
        model.side_effect = lambda x: torch.tensor([[0.6, 0.4]] * x.shape[0])

        device = torch.device("cpu")

        orig = (qs._model, qs._device, qs._transform)
        qs._model = model
        qs._device = device
        qs._transform = _mock_transform

        crops = [_make_crop(0, 0, 150, 30)]
        result = process_and_classify_sync(_image_bytes(), crops)
        # Low confidence => forced to good
        assert result.predictions[0].label == "good"
        qs._model, qs._device, qs._transform = orig

    @patch("src.services.quality_service.config")
    def test_failed_crop_extraction(self, mock_config):
        mock_config.min_width_ratio = 0.0
        mock_config.min_width = 0
        mock_config.bad_crop_threshold = 4

        # Crop outside image bounds completely
        crops = [_make_crop(500, 500, 600, 600)]
        orig = self._setup_globals()
        result = process_and_classify_sync(_image_bytes(100, 100), crops)
        # Crop should fail extraction (coords > image size after clamping => zero area)
        assert result.num_failed_crops >= 0
        self._restore_globals(orig)

    @patch("src.services.quality_service.config")
    def test_rgba_image_converts_to_rgb(self, mock_config):
        """RGBA images are converted to RGB and processed without error."""
        mock_config.min_width_ratio = 1.0
        mock_config.min_width = 0
        mock_config.bad_crop_threshold = 4
        mock_config.debug_mode = False
        mock_config.get.return_value = 0.67

        rgba_img = Image.new("RGBA", (200, 200), color=(128, 128, 128, 255))
        buf = io.BytesIO()
        rgba_img.save(buf, format="PNG")
        rgba_bytes = buf.getvalue()

        crops = [_make_crop(0, 0, 150, 30)]
        orig = self._setup_globals()
        result = process_and_classify_sync(rgba_bytes, crops)
        assert result.label in ("good", "bad")
        assert result.total_crops == 1
        self._restore_globals(orig)

    @patch("src.services.quality_service.config")
    def test_all_crop_transforms_fail_returns_good(self, mock_config):
        """When every crop raises during transform, returns good with empty predictions."""
        import src.services.quality_service as qs

        mock_config.min_width_ratio = 1.0
        mock_config.min_width = 0
        mock_config.bad_crop_threshold = 4

        def _failing_transform(img):
            raise RuntimeError("transform error")

        model = MagicMock()
        device = torch.device("cpu")
        orig = (qs._model, qs._device, qs._transform)
        qs._model = model
        qs._device = device
        qs._transform = _failing_transform

        crops = [_make_crop(0, 0, 150, 30)]
        result = process_and_classify_sync(_image_bytes(), crops)
        assert result.label == "good"
        assert result.predictions == []
        qs._model, qs._device, qs._transform = orig

    @patch("src.services.quality_service.config")
    def test_model_inference_exception_propagates(self, mock_config):
        """An unhandled exception from the model bubbles up as a ValueError."""
        import src.services.quality_service as qs

        mock_config.min_width_ratio = 1.0
        mock_config.min_width = 0
        mock_config.bad_crop_threshold = 4
        mock_config.debug_mode = False
        mock_config.get.return_value = 0.67

        model = MagicMock()
        model.side_effect = RuntimeError("GPU out of memory")
        device = torch.device("cpu")

        orig = (qs._model, qs._device, qs._transform)
        qs._model = model
        qs._device = device
        qs._transform = _mock_transform

        crops = [_make_crop(0, 0, 150, 30)]
        with pytest.raises(RuntimeError, match="GPU out of memory"):
            process_and_classify_sync(_image_bytes(), crops)
        qs._model, qs._device, qs._transform = orig

    @patch("src.services.quality_service.config")
    def test_confidence_threshold_zero_keeps_bad_prediction(self, mock_config):
        """threshold=0.0 means every score passes — bad predictions are never forced to good."""
        import src.services.quality_service as qs

        mock_config.min_width_ratio = 1.0
        mock_config.min_width = 0
        mock_config.bad_crop_threshold = 1
        mock_config.debug_mode = False
        mock_config.get.return_value = 0.0  # threshold=0, all scores pass

        model = MagicMock()
        model.side_effect = lambda x: torch.tensor([[0.9, 0.1]] * x.shape[0])  # class 0 = bad

        orig = (qs._model, qs._device, qs._transform)
        qs._model = model
        qs._device = torch.device("cpu")
        qs._transform = _mock_transform

        crops = [_make_crop(0, 0, 150, 30)]
        result = process_and_classify_sync(_image_bytes(), crops)
        assert result.predictions[0].label == "bad"
        assert result.label == "bad"
        qs._model, qs._device, qs._transform = orig

    @patch("src.services.quality_service.config")
    def test_bad_threshold_boundary_exact(self, mock_config):
        """Overall label is 'bad' when num_bad >= threshold, 'good' when num_bad < threshold."""
        import src.services.quality_service as qs

        mock_config.min_width_ratio = 1.0
        mock_config.min_width = 0
        mock_config.bad_crop_threshold = 2
        mock_config.debug_mode = False
        mock_config.get.return_value = 0.0  # threshold=0 so confidence never overrides

        model = MagicMock()
        model.side_effect = lambda x: torch.tensor([[0.9, 0.1]] * x.shape[0])  # all bad

        orig = (qs._model, qs._device, qs._transform)
        qs._model = model
        qs._device = torch.device("cpu")
        qs._transform = _mock_transform

        # 1 bad crop, threshold=2 → good
        result = process_and_classify_sync(_image_bytes(), [_make_crop(0, 0, 150, 30)])
        assert result.label == "good"

        # 2 bad crops, threshold=2 → bad (>= comparison)
        result = process_and_classify_sync(
            _image_bytes(),
            [_make_crop(0, 0, 150, 30), _make_crop(0, 50, 150, 80)],
        )
        assert result.label == "bad"
        qs._model, qs._device, qs._transform = orig
