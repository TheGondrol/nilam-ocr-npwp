"""Quality classification service with batch inference."""

from __future__ import annotations

import io
import logging
from typing import Callable, Optional

import torch
from PIL import Image
from torchvision import transforms

from src.core.config import config
from src.schemas.api_schema import ClassificationResponse, Crop, CropPrediction
from src.services.image_service import extract_crop, filter_crops
from src.services.threshold_provider import get_provider

logger = logging.getLogger(__name__)

ImageTransform = Callable[[Image.Image], torch.Tensor]


class ImageDecodeError(ValueError):
    """Raised when the uploaded bytes cannot be opened as an image.

    Treated as a client error (HTTP 400) by the API layer — the request is
    well-formed but the payload is not a usable image.
    """

# Global state for model, device, and transform
_model: Optional[torch.nn.Module] = None
_device: Optional[torch.device] = None
_transform: Optional[ImageTransform] = None


def set_model_globals(
    loaded_model: torch.nn.Module,
    loaded_device: torch.device,
    loaded_transform: ImageTransform,
) -> None:
    """
    Set global model, device, and transform.

    Args:
        loaded_model: Loaded PyTorch model
        loaded_device: Device model is on
        loaded_transform: Preprocessing transforms
    """
    global _model, _device, _transform
    _model = loaded_model
    _device = loaded_device
    _transform = loaded_transform
    logger.info("Model globals set successfully")


def get_model() -> Optional[torch.nn.Module]:
    """Get the global model instance."""
    return _model


def get_device() -> Optional[torch.device]:
    """Get the global device."""
    return _device


def process_and_classify_sync(
    image_bytes: bytes,
    crops: list[Crop],
) -> ClassificationResponse:
    """
    Process image with crops and classify quality.

    This function:
    1. Opens the image from bytes
    2. Filters crops based on width/height ratio
    3. Extracts crop images from the main image
    4. Runs batch inference on all crops
    5. Counts bad predictions
    6. Returns classification response

    Args:
        image_bytes: Image file bytes
        crops: List of Crop objects with bbox, text, confidence

    Returns:
        ClassificationResponse with predictions and overall label

    Raises:
        ValueError: If model not loaded or image processing fails
    """
    if _model is None or _device is None or _transform is None:
        raise ValueError("Model not loaded. Please initialize model first.")

    # Load image
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        # Undecodable upload — a client error, surfaced as 400 by the route.
        raise ImageDecodeError(f"Failed to open image: {e}") from e

    total_crops = len(crops)
    logger.info(f"Processing {total_crops} crops")

    # Filter crops (width > height for text)
    min_width_ratio = config.min_width_ratio
    min_width = config.min_width
    filtered_crops = filter_crops(crops, min_width_ratio, min_width)
    num_filtered = len(filtered_crops)

    logger.info(f"Filtered to {num_filtered} crops (min_width_ratio={min_width_ratio}, min_width={min_width})")

    if num_filtered == 0:
        logger.warning("No crops passed filtering")
        return ClassificationResponse(
            label="good",
            num_bad=0,
            num_filtered=0,
            total_crops=total_crops,
            num_failed_crops=0,
            bad_crop_threshold=int(get_provider().get("bad_crop_threshold")),
            predictions=[],
        )

    # Extract crop images
    crop_images: list[tuple[Crop, Optional[Image.Image]]] = []
    num_failed = 0

    for crop in filtered_crops:
        crop_img = extract_crop(image, crop.bbox)
        if crop_img is None:
            num_failed += 1
        crop_images.append((crop, crop_img))

    # Filter out failed extractions
    valid_crop_data = [(crop, img) for crop, img in crop_images if img is not None]

    if not valid_crop_data:
        logger.warning(f"All {num_filtered} crops failed extraction")
        return ClassificationResponse(
            label="good",
            num_bad=0,
            num_filtered=num_filtered,
            total_crops=total_crops,
            num_failed_crops=num_failed,
            bad_crop_threshold=int(get_provider().get("bad_crop_threshold")),
            predictions=[],
        )

    logger.info(f"Successfully extracted {len(valid_crop_data)} crops ({num_failed} failed)")

    # Preprocess all crops into batch tensor
    crop_tensors = []
    for _, crop_img in valid_crop_data:
        try:
            tensor = _transform(crop_img)
            crop_tensors.append(tensor)
        except Exception as e:
            logger.warning(f"Failed to transform crop: {e}")
            continue

    if not crop_tensors:
        logger.warning("No crops successfully transformed")
        return ClassificationResponse(
            label="good",
            num_bad=0,
            num_filtered=num_filtered,
            total_crops=total_crops,
            num_failed_crops=num_failed,
            bad_crop_threshold=int(get_provider().get("bad_crop_threshold")),
            predictions=[],
        )

    # Stack into batch
    batch_tensor = torch.stack(crop_tensors).to(_device)
    logger.info(f"Running batch inference on {batch_tensor.shape[0]} crops")

    # Run batch inference
    with torch.inference_mode():
        outputs = _model(batch_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predictions_tensor = torch.argmax(probabilities, dim=1)

    # Convert to CPU and numpy
    predictions_np = predictions_tensor.cpu().numpy()
    probabilities_np = probabilities.cpu().numpy()

    # Build predictions list
    predictions: list[CropPrediction] = []
    num_bad = 0
    debug_mode = config.debug_mode
    confidence_threshold = get_provider().get("confidence_threshold")

    for idx, (crop, _) in enumerate(valid_crop_data):
        if idx >= len(predictions_np):
            break

        pred_class = int(predictions_np[idx])
        pred_score = float(probabilities_np[idx][pred_class])
        
        # If confidence is below threshold, consider it as "good"
        if pred_score < confidence_threshold:
            pred_class = 1  # Force to good
            pred_label = "good"
            # Keep the reported score consistent with the forced label: report
            # the good-class probability, not the original argmax (bad) one (BUG-24).
            pred_score = float(probabilities_np[idx][1])
        else:
            pred_label = "good" if pred_class == 1 else "bad"

        if pred_class == 0:  # bad crop
            num_bad += 1

        # Build prediction object
        crop_pred = CropPrediction(
            bbox=crop.bbox,
            prediction=pred_class,
            label=pred_label,
            score=pred_score,
        )

        # Add text and confidence only in debug mode
        if debug_mode:
            crop_pred.text = crop.text
            crop_pred.confidence = crop.confidence

        predictions.append(crop_pred)

    # Determine overall label
    bad_threshold = int(get_provider().get("bad_crop_threshold"))
    overall_label = "bad" if num_bad >= bad_threshold else "good"

    logger.info(
        f"Classification complete: {num_bad}/{len(predictions)} bad crops "
        f"(threshold={bad_threshold}) -> {overall_label}"
    )

    return ClassificationResponse(
        label=overall_label,
        num_bad=num_bad,
        num_filtered=num_filtered,
        total_crops=total_crops,
        num_failed_crops=num_failed,
        bad_crop_threshold=bad_threshold,
        predictions=predictions,
    )
