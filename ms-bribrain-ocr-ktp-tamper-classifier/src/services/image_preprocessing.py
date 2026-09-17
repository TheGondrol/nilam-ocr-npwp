"""
Image preprocessing service
Handles image resizing, padding, and preprocessing for model input
"""

import asyncio
import io
import logging
from typing import Any, cast

import torch
import torchvision.transforms.functional as TF
from PIL import Image

logger = logging.getLogger(__name__)


def resize_with_padding(img: Image.Image, size: int = 320) -> Image.Image:
    """
    Resize image with padding to maintain aspect ratio
    Same preprocessing as training
    
    Args:
        img: PIL Image
        size: Target size (default: 320)
    
    Returns:
        PIL Image resized with padding
    """
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    # torchvision functional accepts PIL images at runtime, but its stubs
    # only declare Tensor inputs — route through Any to satisfy the checker.
    resized: Any = TF.resize(cast(Any, img), [new_h, new_w])

    pad_w = size - new_w
    pad_h = size - new_h
    padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
    padded: Any = TF.pad(resized, padding, fill=255)
    return cast(Image.Image, padded)


def preprocess_image(image: Image.Image, processor: Any, image_size: int = 320) -> torch.Tensor:
    """
    Preprocess image for inference
    
    Args:
        image: PIL Image
        processor: AutoImageProcessor
        image_size: Target image size (default: 320)
    
    Returns:
        Preprocessed tensor
    """
    # Resize with padding
    img = resize_with_padding(image, image_size)
    
    # Process with AutoImageProcessor
    inputs = processor(
        img,
        do_resize=False,
        do_center_crop=False,
        return_tensors="pt"
    )
    
    return inputs["pixel_values"]


def _decode_image_sync(image_bytes: bytes, max_megapixels: float = 50.0) -> Image.Image:
    """Synchronous PIL decode (runs off the event loop via run_in_executor).

    Forces the pixel decode inside the try so a truncated/malformed image fails
    here as a ValueError (-> HTTP 400) regardless of color mode, instead of
    surfacing later as an OSError (-> 500) for already-RGB inputs (BUG-19).
    """
    try:
        image: Image.Image = Image.open(io.BytesIO(image_bytes))
        # Force the (otherwise lazy) pixel decode so corruption is caught here.
        image.load()
        # Only convert if not already RGB to avoid unnecessary memory allocation
        if image.mode != "RGB":
            image = image.convert("RGB")
        if image.width * image.height > max_megapixels * 1_000_000:
            raise ValueError(
                f"Image too large: {image.width}x{image.height} exceeds {max_megapixels} MP"
            )
        return image
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to load image from bytes: {str(e)}")
        raise ValueError(f"Invalid image data: {str(e)}")


async def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Load PIL Image from bytes asynchronously.

    The CPU-bound PIL decode is offloaded to a thread-pool worker so it does not
    block the asyncio event loop under concurrency (BUG-18).

    Args:
        image_bytes: Image bytes

    Returns:
        PIL Image in RGB mode
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _decode_image_sync, image_bytes)
