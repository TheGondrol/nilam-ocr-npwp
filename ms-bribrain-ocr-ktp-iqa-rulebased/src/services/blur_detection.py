"""
Blur detection service.
Detects blur in images using Laplacian variance method.
"""

import statistics
import cv2
import numpy as np
from typing import Tuple, List

from src.core.config import get_config
from src.core.logging import get_logger
from src.services.threshold_provider import get_provider

logger = get_logger(__name__)

_config = get_config()


def check_confidence_median(results: List) -> bool:
    """
    Check if median confidence score from OCR meets threshold.

    Args:
        results: List of OCR results, each element tuple (text, confidence)

    Returns:
        bool: True if median confidence is below threshold (low confidence), False otherwise
    """
    if not results:
        logger.warning("Empty OCR results provided for confidence check")
        return False

    try:
        # Extract text-confidence tuples from OCR results
        # OCR format: [[[coords], (text, confidence)], ...]
        text_confidence_pairs = [line[1] for line in results]
        scores = [score for _, score in text_confidence_pairs if isinstance(score, (int, float))]

        if not scores:
            logger.warning("No valid confidence scores found in OCR results")
            return False

        median_score = statistics.median(scores)
        threshold_median = get_provider().get("confidence_threshold_median")
        is_low_confidence = median_score <= threshold_median

        logger.info(
            f"Median confidence score: {median_score:.3f}, threshold: {threshold_median}, low_confidence: {is_low_confidence}"
        )

        return is_low_confidence

    except Exception as e:
        logger.error(f"Error checking confidence median: {e}", exc_info=True)
        return False


def blur_detection(gray_image: np.ndarray) -> Tuple[bool, float]:
    """
    Detect blur in an image using the Laplacian variance method.

    Args:
        gray_image: Grayscale image as numpy ndarray.
                   Expected shape: (H, W) - 2D grayscale image.
                   Expected dtype: uint8 or compatible numeric type.

    Returns:
        Tuple of (is_blurry: bool, variance: float)
            - is_blurry: True if image variance is below threshold (blurry)
            - variance: Computed Laplacian variance value
    """
    try:
        # Compute the Laplacian
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

        # Calculate variance
        variance = laplacian.var()

        # CRITICAL: Cleanup to prevent memory leaks
        del laplacian

        # Determine if the image is blurry
        threshold_blur = get_provider().get("blur_threshold")
        is_blurry = variance < threshold_blur

        logger.info(
            f"Blur detection - variance: {variance:.2f}, threshold: {threshold_blur}, is_blurry: {is_blurry}"
        )

        return is_blurry, variance

    except Exception as e:
        logger.error(f"Error in blur detection: {e}", exc_info=True)
        return False, 0.0
