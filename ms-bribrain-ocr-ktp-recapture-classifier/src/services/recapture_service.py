from typing import Any
from PIL import Image
import io
import torch
from ..core.config import config
from ..core.logging import logger

# Global variables for model and device (will be set by main app)
_model: Any = None
_device: Any = None
_transform: Any = None

# Cached config values (avoid repeated lookups per request)
_normalize_size = config.get('model.normalize_size', 512)
_crop_size = config.get('model.crop_size', 224)


def set_model_globals(loaded_model, loaded_device, loaded_transform):
    """Set global model, device, and transform"""
    global _model, _device, _transform
    _model = loaded_model
    _device = loaded_device
    _transform = loaded_transform


def get_model():
    """Get the loaded model (use this instead of importing model directly)"""
    return _model


def get_device():
    """Get the device (use this instead of importing device directly)"""
    return _device


def get_transform():
    """Get the transform (use this instead of importing transform directly)"""
    return _transform


def process_and_predict_sync(image_bytes: bytes) -> tuple[int, float, float, tuple[int, int]]:
    """
    Combined image processing and prediction - runs in single executor call.
    
    This combines _process_image_sync, _resize_image_sync, and _predict_sync
    into one function to reduce executor call overhead (~30-50ms savings).

    Args:
        image_bytes: Raw image data in bytes

    Returns:
        tuple of (predicted_class, prob_recaptured, confidence, original_size)
    """
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    original_size = image.size

    # Hand the raw PIL image to _transform — the transform built by
    # ml_model.get_transform() owns the resize strategy (configured via
    # `model.resize_mode` in config.yaml; default `pad` matches the
    # latest pad-trained checkpoints).
    assert _transform is not None, "Transform not initialized"
    assert _model is not None, "Model not initialized"
    tensor = _transform(image).unsqueeze(0).to(_device)
    with torch.inference_mode():
        outputs = _model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        # Class mapping from the trained checkpoint: 0=ORIGINAL, 1=RECAPTURED.
        # Verified empirically against real images — earlier comments/README
        # documented the opposite ordering, which produced inverted labels.
        prob_recaptured = float(probs[1].item())
        predicted_class = int(torch.argmax(outputs, dim=1).item())
        confidence = float(probs[predicted_class].item())

    return predicted_class, prob_recaptured, confidence, original_size


# Keep original functions for backwards compatibility
def _process_image_sync(image_bytes: bytes) -> Image.Image:
    """
    Synchronous image processing - runs in thread pool.

    Args:
        image_bytes: Image data in bytes

    Returns:
        Image
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


def _resize_image_sync(image: Image.Image) -> Image.Image:
    """
    Synchronous image resizing - runs in thread pool.

    Args:
        image: PIL Image

    Returns:
        Resized image
    """
    width, height = image.size
    if height < width:
        new_height = _normalize_size
        new_width = int(width * (_normalize_size / height))
    else:
        new_width = _normalize_size
        new_height = int(height * (_normalize_size / width))
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    logger.info(f"Normalized to: {image.size}, ready for {_crop_size}x{_crop_size} center crop")
    return image


def _predict_sync(image: Image.Image) -> tuple[int, float, float]:
    """
    Synchronous prediction - runs in thread pool.

    Args:
        image: PIL Image

    Returns:
        Tuple of (predicted_class, prob_recaptured, confidence)
    """
    assert _transform is not None, "Transform not initialized"
    assert _model is not None, "Model not initialized"
    tensor = _transform(image).unsqueeze(0).to(_device)
    with torch.inference_mode():
        outputs = _model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        # Class mapping from the trained checkpoint: 0=ORIGINAL, 1=RECAPTURED.
        prob_recaptured = float(probs[1].item())
        predicted_class = int(torch.argmax(outputs, dim=1).item())
        confidence = float(probs[predicted_class].item())
    return predicted_class, prob_recaptured, confidence