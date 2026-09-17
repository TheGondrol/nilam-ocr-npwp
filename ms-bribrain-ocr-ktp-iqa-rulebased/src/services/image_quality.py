"""
Image quality service.
Main orchestration service for image quality checks.
"""

import time
import numpy as np
import cv2
from typing import Dict, Any, List

from src.core.logging import get_logger
from src.core.config import get_config
from src.services.blur_detection import blur_detection, check_confidence_median
from src.services.glare_detection import detect_glare_with_text_analysis
from src.services.rotation_detection import detect_image_rotation

logger = get_logger(__name__)


class ImageDecodeError(ValueError):
    """Raised when the uploaded bytes cannot be decoded into an image.

    Treated as a client error (HTTP 400) by the API layer — the request is
    well-formed but the payload is not a usable image.
    """


def image_quality(image_bytes: bytes, ocr_result: List) -> Dict[str, Any]:
    """
    Check image quality based on blur, glare, and rotation.

    Args:
        image_bytes: Image data as bytes
        ocr_result: OCR results containing text boxes and confidence scores

    Returns:
        Dict containing quality check results:
            - low_confidence: bool
            - is_blurry: bool
            - is_glare: bool
            - is_rotated: bool
    """
    total_start = time.perf_counter()

    try:
        config = get_config()
        confidence_enabled = config.quality.confidence.enabled
        blur_enabled = config.quality.blur.enabled
        glare_enabled = config.quality.glare.enabled
        rotation_enabled = config.quality.rotation.enabled

        # Convert bytes to numpy array using OpenCV (memory efficient, no PIL)
        decode_start = time.perf_counter()
        np_array = np.frombuffer(image_bytes, np.uint8)
        # cv2.imdecode returns None for malformed data and raises cv2.error for an
        # empty buffer — both are "client sent something that isn't an image".
        try:
            np_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        except cv2.error as e:
            raise ImageDecodeError(f"Failed to decode image from bytes: {e}") from e

        if np_image is None:
            raise ImageDecodeError("Failed to decode image from bytes")

        # Only convert color spaces when needed.
        # Rotation detection consumes BGR directly (InsightFace), so no
        # conversion is needed for it — np_image is already BGR from cv2.imdecode.
        image_gray = (
            cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY) if blur_enabled else np_image
        )
        decode_time = (time.perf_counter() - decode_start) * 1000
        logger.debug(f"Image decode/convert: {decode_time:.2f}ms")

        # Process quality checks with timing
        # - confidence check: uses OCR result only
        if confidence_enabled:
            confidence_start = time.perf_counter()
            is_low_confidence = check_confidence_median(ocr_result)
            confidence_time = (time.perf_counter() - confidence_start) * 1000
            logger.debug(f"Confidence check: {confidence_time:.2f}ms")
        else:
            is_low_confidence = False
            logger.debug("Confidence check: SKIPPED (disabled)")

        # - blur detection: uses grayscale image
        if blur_enabled:
            blur_start = time.perf_counter()
            is_blurry, blur_variance = blur_detection(image_gray)
            blur_time = (time.perf_counter() - blur_start) * 1000
            logger.debug(f"Blur detection: {blur_time:.2f}ms (variance={blur_variance:.2f})")
        else:
            is_blurry = False
            logger.debug("Blur detection: SKIPPED (disabled)")

        # - glare detection: uses BGR image (OpenCV format)
        if glare_enabled:
            glare_start = time.perf_counter()
            is_glare = detect_glare_with_text_analysis(np_image, ocr_result)  # BGR
            glare_time = (time.perf_counter() - glare_start) * 1000
            logger.debug(f"Glare detection: {glare_time:.2f}ms")
        else:
            is_glare = False
            logger.debug("Glare detection: SKIPPED (disabled)")

        # - rotation detection: uses BGR image (InsightFace format)
        if rotation_enabled:
            rotation_start = time.perf_counter()
            is_rotated = detect_image_rotation(np_image)  # BGR
            rotation_time = (time.perf_counter() - rotation_start) * 1000
            logger.debug(f"Rotation detection: {rotation_time:.2f}ms")
        else:
            is_rotated = False
            logger.debug("Rotation detection: SKIPPED (disabled)")

        result = {
            "low_confidence": bool(is_low_confidence),
            "is_blurry": bool(is_blurry),
            "is_glare": bool(is_glare),
            "is_rotated": bool(is_rotated),
        }

        # CRITICAL: Explicitly delete numpy arrays to free memory immediately
        del np_array, np_image, image_gray

        total_time = (time.perf_counter() - total_start) * 1000
        logger.info(f"Image quality analysis complete: {total_time:.2f}ms")

        return result

    except ImageDecodeError:
        # Undecodable upload — a client error. Propagate so the API returns 400
        # rather than reporting a non-image as "perfect quality".
        raise
    except Exception as e:
        # Unexpected orchestration failure — propagate as a server error (500).
        # Individual detectors self-guard and return their "no issue" default,
        # so reaching here means something outside the detectors broke.
        logger.error(f"Error in image quality analysis: {e}", exc_info=True)
        raise

