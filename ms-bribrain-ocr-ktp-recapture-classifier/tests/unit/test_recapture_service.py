"""Tests for recapture detection service."""

import pytest
from unittest.mock import MagicMock
from PIL import Image
import io
import torch


class TestRecaptureService:
    """Test cases for recapture service functions."""

    def test_set_model_globals(self, mock_model, mock_device, mock_transform):
        """Test setting global model, device, and transform."""
        from src.services.recapture_service import set_model_globals, get_model, get_device, get_transform
        
        set_model_globals(mock_model, mock_device, mock_transform)
        
        assert get_model() == mock_model
        assert get_device() == mock_device
        assert get_transform() == mock_transform

    def test_get_model_before_initialization(self):
        """Test getting model before initialization returns None."""
        from src.services import recapture_service
        # Reset globals
        recapture_service._model = None
        assert recapture_service.get_model() is None

    def test_get_device_before_initialization(self):
        """Test getting device before initialization returns None."""
        from src.services import recapture_service
        # Reset globals
        recapture_service._device = None
        assert recapture_service.get_device() is None

    def test_process_and_predict_sync_rgb_image(self, sample_image_bytes):
        """Test process_and_predict_sync with RGB image."""
        from src.services.recapture_service import set_model_globals, process_and_predict_sync
        
        # Setup mocks
        mock_model = MagicMock()
        mock_device = torch.device("cpu")
        
        # Mock transform
        def mock_transform(img):
            return torch.zeros((3, 224, 224))
        
        # Mock model output
        mock_model.return_value = torch.tensor([[0.3, 0.7]])
        
        set_model_globals(mock_model, mock_device, mock_transform)
        
        # Run prediction
        predicted_class, prob_recaptured, confidence, original_size = process_and_predict_sync(sample_image_bytes)
        
        # Verify results
        assert predicted_class in [0, 1]
        assert 0.0 <= prob_recaptured <= 1.0
        assert 0.0 <= confidence <= 1.0
        assert isinstance(original_size, tuple)
        assert len(original_size) == 2

    def test_process_and_predict_sync_grayscale_conversion(self):
        """Test that grayscale images are converted to RGB."""
        from src.services.recapture_service import set_model_globals, process_and_predict_sync
        
        # Create grayscale image
        img = Image.new('L', (100, 100), color=128)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        # Setup mocks
        mock_model = MagicMock()
        mock_device = torch.device("cpu")
        
        def mock_transform(img):
            # Verify image is RGB
            assert img.mode == 'RGB'
            return torch.zeros((3, 224, 224))
        
        mock_model.return_value = torch.tensor([[0.4, 0.6]])
        
        set_model_globals(mock_model, mock_device, mock_transform)
        
        # Should not raise error
        predicted_class, prob_recaptured, confidence, original_size = process_and_predict_sync(img_bytes)
        assert predicted_class in [0, 1]

    def test_process_and_predict_sync_aspect_ratio_preserved(self):
        """Test that aspect ratio is preserved during resize."""
        from src.services.recapture_service import set_model_globals, process_and_predict_sync
        
        # Create wide image (2:1 aspect ratio)
        img = Image.new('RGB', (800, 400), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        # Setup mocks
        mock_model = MagicMock()
        mock_device = torch.device("cpu")
        
        def mock_transform(img_pil):
            # Image should be resized maintaining aspect ratio
            # For normalize_size=512, width should be larger than height
            return torch.zeros((3, 224, 224))
        
        mock_model.return_value = torch.tensor([[0.2, 0.8]])
        
        set_model_globals(mock_model, mock_device, mock_transform)
        
        predicted_class, prob_recaptured, confidence, original_size = process_and_predict_sync(img_bytes)
        assert original_size == (800, 400)

    def test_process_and_predict_sync_class_0_original(self):
        """Test prediction for class 0 (ORIGINAL)."""
        from src.services.recapture_service import set_model_globals, process_and_predict_sync

        mock_model = MagicMock()
        mock_device = torch.device("cpu")

        def mock_transform(img):
            return torch.zeros((3, 224, 224))

        # Mock model output - class 0 (ORIGINAL) has higher probability
        mock_model.return_value = torch.tensor([[0.8, 0.2]])

        set_model_globals(mock_model, mock_device, mock_transform)

        # Create sample image
        img = Image.new('RGB', (100, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')

        predicted_class, prob_recaptured, confidence, _ = process_and_predict_sync(img_bytes.getvalue())

        assert predicted_class == 0  # ORIGINAL
        assert prob_recaptured < 0.5  # probs[1] is the recaptured class
        assert confidence == pytest.approx(1.0 - prob_recaptured)  # For class 0, confidence == probs[0]

    def test_process_and_predict_sync_class_1_recaptured(self):
        """Test prediction for class 1 (RECAPTURED)."""
        from src.services.recapture_service import set_model_globals, process_and_predict_sync

        mock_model = MagicMock()
        mock_device = torch.device("cpu")

        def mock_transform(img):
            return torch.zeros((3, 224, 224))

        # Mock model output - class 1 (RECAPTURED) has higher probability
        mock_model.return_value = torch.tensor([[0.25, 0.75]])

        set_model_globals(mock_model, mock_device, mock_transform)

        # Create sample image
        img = Image.new('RGB', (100, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')

        predicted_class, prob_recaptured, confidence, _ = process_and_predict_sync(img_bytes.getvalue())

        assert predicted_class == 1  # RECAPTURED
        assert prob_recaptured > 0.5
        assert confidence == pytest.approx(prob_recaptured)  # For class 1, confidence == probs[1] == prob_recaptured

    def test_process_image_sync(self):
        """Test _process_image_sync function."""
        from src.services.recapture_service import _process_image_sync
        
        # Create image bytes
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        
        result = _process_image_sync(img_bytes.getvalue())
        
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'
        assert result.size == (100, 100)

    def test_resize_image_sync_landscape(self):
        """Test _resize_image_sync with landscape image."""
        from src.services.recapture_service import _resize_image_sync
        
        # Create landscape image (800x400)
        img = Image.new('RGB', (800, 400))
        
        resized = _resize_image_sync(img)
        
        assert isinstance(resized, Image.Image)
        # Height should be normalized to 512
        assert resized.size[1] == 512
        # Width should maintain aspect ratio
        assert resized.size[0] == 1024

    def test_resize_image_sync_portrait(self):
        """Test _resize_image_sync with portrait image."""
        from src.services.recapture_service import _resize_image_sync
        
        # Create portrait image (400x800)
        img = Image.new('RGB', (400, 800))
        
        resized = _resize_image_sync(img)
        
        assert isinstance(resized, Image.Image)
        # Width should be normalized to 512
        assert resized.size[0] == 512
        # Height should maintain aspect ratio
        assert resized.size[1] == 1024

    def test_predict_sync(self, mock_model, mock_device, mock_transform):
        """Test _predict_sync function."""
        from src.services.recapture_service import set_model_globals, _predict_sync
        
        mock_model.return_value = torch.tensor([[0.35, 0.65]])
        set_model_globals(mock_model, mock_device, mock_transform)
        
        img = Image.new('RGB', (224, 224))
        
        predicted_class, prob_recaptured, confidence = _predict_sync(img)
        
        assert predicted_class in [0, 1]
        assert 0.0 <= prob_recaptured <= 1.0
        assert 0.0 <= confidence <= 1.0

    def test_process_and_predict_sync_already_rgb_skips_convert(self):
        """RGB image should skip the convert('RGB') branch (covers line 56 false branch)."""
        from src.services import recapture_service
        from src.services.recapture_service import set_model_globals, process_and_predict_sync

        captured = {}
        real_open = Image.open

        def tracking_open(*args, **kwargs):
            img = real_open(*args, **kwargs)
            original_convert = img.convert

            def wrapped_convert(mode, *a, **kw):
                captured["converted"] = True
                return original_convert(mode, *a, **kw)

            img.convert = wrapped_convert  # type: ignore[method-assign]
            return img

        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.3, 0.7]])

        def mock_transform(img_pil):
            assert img_pil.mode == "RGB"
            return torch.zeros((3, 224, 224))

        set_model_globals(mock_model, torch.device("cpu"), mock_transform)

        img = Image.new("RGB", (120, 120))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")

        from unittest.mock import patch as _p
        with _p("src.services.recapture_service.Image.open", side_effect=tracking_open):
            process_and_predict_sync(buf.getvalue())

        assert captured.get("converted", False) is False

    def test_process_and_predict_sync_rgba_converted_to_rgb(self):
        """RGBA image with alpha must be converted to RGB before transform."""
        from src.services.recapture_service import set_model_globals, process_and_predict_sync

        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        seen_mode = {}

        def mock_transform(img_pil):
            seen_mode["mode"] = img_pil.mode
            return torch.zeros((3, 224, 224))

        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.1, 0.9]])

        set_model_globals(mock_model, torch.device("cpu"), mock_transform)
        process_and_predict_sync(buf.getvalue())

        assert seen_mode["mode"] == "RGB"

    def test_process_and_predict_sync_passes_raw_image_to_transform(self):
        """process_and_predict_sync hands the raw image to the transform (which
        owns the resize strategy now), so the transform sees the original size."""
        from src.services.recapture_service import set_model_globals, process_and_predict_sync

        img = Image.new("RGB", (600, 600))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")

        seen_sizes = []

        def mock_transform(img_pil):
            seen_sizes.append(img_pil.size)
            return torch.zeros((3, 224, 224))

        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.4, 0.6]])

        set_model_globals(mock_model, torch.device("cpu"), mock_transform)
        process_and_predict_sync(buf.getvalue())

        # No in-function resize anymore: the transform receives the original image.
        assert seen_sizes[0] == (600, 600)

    def test_process_and_predict_sync_probability_tie_picks_class_zero(self):
        """When logits produce exactly equal probs, argmax returns index 0 (ORIGINAL)."""
        from src.services.recapture_service import set_model_globals, process_and_predict_sync

        mock_model = MagicMock()
        # Equal logits → softmax = [0.5, 0.5]; argmax picks 0.
        mock_model.return_value = torch.tensor([[1.0, 1.0]])

        def mock_transform(img):
            return torch.zeros((3, 224, 224))

        set_model_globals(mock_model, torch.device("cpu"), mock_transform)

        img = Image.new("RGB", (50, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")

        predicted_class, prob_recaptured, confidence, _ = process_and_predict_sync(buf.getvalue())
        assert predicted_class == 0
        assert abs(prob_recaptured - 0.5) < 1e-6
        assert abs(confidence - 0.5) < 1e-6

    def test_process_and_predict_sync_asserts_when_transform_missing(self):
        """Calling predict without initializing transform must assert."""
        from src.services import recapture_service
        from src.services.recapture_service import process_and_predict_sync

        recapture_service._transform = None
        recapture_service._model = MagicMock()
        recapture_service._device = torch.device("cpu")

        img = Image.new("RGB", (30, 30))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")

        with pytest.raises(AssertionError, match="Transform not initialized"):
            process_and_predict_sync(buf.getvalue())

    def test_process_and_predict_sync_asserts_when_model_missing(self):
        from src.services import recapture_service
        from src.services.recapture_service import process_and_predict_sync

        recapture_service._transform = lambda img: torch.zeros((3, 224, 224))
        recapture_service._model = None
        recapture_service._device = torch.device("cpu")

        img = Image.new("RGB", (30, 30))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")

        with pytest.raises(AssertionError, match="Model not initialized"):
            process_and_predict_sync(buf.getvalue())

    def test_resize_image_sync_square(self):
        """Square input — exercises the `else` branch of _resize_image_sync."""
        from src.services.recapture_service import _resize_image_sync, _normalize_size

        img = Image.new("RGB", (300, 300))
        resized = _resize_image_sync(img)
        assert resized.size == (_normalize_size, _normalize_size)

    def test_process_image_sync_converts_rgba_to_rgb(self):
        """_process_image_sync should coerce RGBA to RGB."""
        from src.services.recapture_service import _process_image_sync

        img = Image.new("RGBA", (64, 64), (10, 20, 30, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = _process_image_sync(buf.getvalue())
        assert result.mode == "RGB"

    def test_process_and_predict_sync_invalid_image_raises_error(self):
        """Test that invalid image data raises error."""
        from src.services.recapture_service import set_model_globals, process_and_predict_sync
        
        mock_model = MagicMock()
        mock_device = torch.device("cpu")
        
        def mock_transform(img):
            return torch.zeros((3, 224, 224))
        
        set_model_globals(mock_model, mock_device, mock_transform)
        
        # Invalid image bytes
        invalid_bytes = b"not an image"
        
        with pytest.raises(Exception):
            process_and_predict_sync(invalid_bytes)
