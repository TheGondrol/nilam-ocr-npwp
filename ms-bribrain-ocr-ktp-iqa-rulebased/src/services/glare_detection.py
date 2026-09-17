"""
Glare detection service.
Detects glare in images with text analysis and adaptive thresholding.
"""

import numpy as np
import cv2
from typing import List, Optional

from src.core.config import get_config
from src.core.logging import get_logger
from src.services.threshold_provider import get_provider

logger = get_logger(__name__)

_config = get_config()
_threshold_min_area_glare = _config.quality.glare.min_area
_glare_padding_size = _config.quality.glare.padding_size
_glare_kernel_size = _config.quality.glare.kernel_size


def calculate_median_brightness(img: np.ndarray) -> float:
    """
    Calculate the median brightness of an image using HSV.

    Args:
        img: Image as numpy ndarray in BGR color format.
            Expected shape: (H, W, 3) - 3-channel BGR image.
            Expected dtype: uint8.

    Returns:
        float: Median brightness value (0-255)
    """
    try:
        # Convert the image from BGR to HSV color space
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Extract the V channel (brightness)
        v_channel = hsv_img[:, :, 2]

        # Calculate the median brightness
        median_brightness = np.median(v_channel)

        return float(median_brightness)

    except Exception as e:
        logger.error(f"Error calculating median brightness: {e}", exc_info=True)
        return 128.0  # Default middle value


def determine_glare_threshold(median_brightness: float) -> int:
    """
    Determine appropriate glare threshold based on median brightness.

    Args:
        median_brightness: Median brightness value of the image (0-255)

    Returns:
        int: Threshold value for glare detection
    """
    if median_brightness > 240:
        # For very bright images, use higher threshold
        # but cap at 254 or median + 5, whichever is smaller
        return min(254, int(median_brightness) + 5)
    elif 220 <= median_brightness <= 240:
        return 250
    elif 200 <= median_brightness < 220:
        return 240
    elif 120 <= median_brightness < 200:
        return 220
    else:  # median_brightness < 120
        return 150


def detect_glare_with_text_analysis(np_image: np.ndarray, ocr_result: Optional[List]) -> bool:
    """
    Optimized glare detection with text analysis and adaptive thresholding.

    Args:
        np_image: Image as numpy ndarray in BGR color format (OpenCV format).
                 Expected shape: (H, W, 3) - 3-channel BGR image.
                 Expected dtype: uint8.
        ocr_result: OCR results containing text boxes and confidence scores.
                   Expected format: List of [[[coordinates], (text, confidence)], ...]

    Returns:
        bool: True if glare affects text regions, False otherwise
    """
    try:
        # Image is already in BGR format (OpenCV format)
        img = np_image

        # Calculate median brightness and determine threshold
        median_brightness = calculate_median_brightness(img)
        threshold_value = determine_glare_threshold(median_brightness)

        logger.debug(
            f"Image median brightness: {median_brightness:.2f}, using adaptive glare threshold: {threshold_value}"
        )

        # Convert directly to V channel without creating full HSV
        bgr_channels = cv2.split(img)
        v_channel = cv2.max(cv2.max(bgr_channels[0], bgr_channels[1]), bgr_channels[2])
        del bgr_channels  # Cleanup intermediate arrays

        # Threshold the V channel (brightness)
        _, glare_mask = cv2.threshold(
            v_channel, threshold_value, 255, cv2.THRESH_BINARY
        )
        del v_channel  # Cleanup after threshold

        # Clean up noise with optimized morphological operations
        kernel = np.ones((_glare_kernel_size, _glare_kernel_size), np.uint8)
        glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_OPEN, kernel)

        # Find contours of glare regions with simplified chain approximation
        contours, _ = cv2.findContours(
            glare_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area more efficiently
        significant_contours = []
        glare_areas = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > _threshold_min_area_glare:
                significant_contours.append(cnt)
                glare_areas.append(area)

        logger.debug(f"Found {len(significant_contours)} significant glare regions")

        # Sort only if needed
        if len(significant_contours) > 3:
            # Sort only the top 3 by area for padding
            indices = np.argsort(glare_areas)[-3:][::-1]
            top_contours = [significant_contours[i] for i in indices]
        else:
            top_contours = significant_contours

        # Create padded glare mask more efficiently
        padded_glare_mask = np.zeros_like(glare_mask)

        # First, fill all original glare contours
        cv2.drawContours(padded_glare_mask, significant_contours, -1, (255,), -1)

        # Add padding to only the largest contours
        if _glare_padding_size > 0:
            dilation_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * _glare_padding_size + 1, 2 * _glare_padding_size + 1),
            )

            for cnt in top_contours:
                # Create mask for this contour
                mask_for_dilation = np.zeros_like(glare_mask)
                cv2.drawContours(mask_for_dilation, [cnt], 0, (255,), -1)

                # Dilate to create padding
                dilated_mask = cv2.dilate(
                    mask_for_dilation, dilation_kernel, iterations=1
                )

                # Add to padding mask
                padded_glare_mask = cv2.bitwise_or(padded_glare_mask, dilated_mask)

                # Cleanup loop variables to prevent memory buildup
                del mask_for_dilation, dilated_mask

        # Process OCR results
        text_regions = []
        if ocr_result is not None:
            for line in ocr_result:
                box = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text = line[1][0]
                confidence = line[1][1]

                # Skip low confidence text to reduce noise
                if confidence < get_provider().get("glare_min_text_confidence"):
                    continue

                # Convert box to contour format
                box_np = np.array(box, dtype=np.int32)
                text_regions.append(
                    {"contour": box_np, "text": text, "confidence": confidence}
                )

        logger.debug(f"Processing {len(text_regions)} text regions for glare analysis")

        # Analyze intersection between glare and text regions
        affected_regions = []

        for text_region in text_regions:
            # Create binary mask for text region
            text_mask = np.zeros(padded_glare_mask.shape, dtype=np.uint8)
            cv2.fillPoly(text_mask, [text_region["contour"]], (255,))

            # Calculate intersection quickly
            intersection = cv2.bitwise_and(text_mask, padded_glare_mask)

            # Calculate areas
            text_area = cv2.countNonZero(text_mask)
            intersection_area = cv2.countNonZero(intersection)

            # Calculate percentage of text affected by glare
            if text_area > 0:
                affected_percentage = (intersection_area / text_area) * 100
            else:
                affected_percentage = 0

            # Consider text affected if percentage exceeds threshold
            if affected_percentage > get_provider().get("glare_affected_percentage_threshold"):
                affected_regions.append(
                    {
                        "text": text_region["text"],
                        "affected_percentage": affected_percentage,
                    }
                )

        # Calculate overall impact metrics
        affected_text_count = len(affected_regions)
        text_regions_count = len(text_regions)

        if text_regions_count > 0:
            affected_text_percentage = (affected_text_count / text_regions_count) * 100
        else:
            affected_text_percentage = 0

        # Determine overall impact level
        if affected_text_percentage > 30:
            overall_impact = "High"
        elif affected_text_percentage > 10:
            overall_impact = "Medium"
        else:
            overall_impact = "Low"

        # Make final decision: reject if any text is affected, accept otherwise
        decision = affected_text_count > 0

        logger.info(
            f"Glare detection result - affected_text_count: {affected_text_count}, "
            f"affected_percentage: {affected_text_percentage:.1f}%, "
            f"impact: {overall_impact}, decision: {decision}"
        )

        # CRITICAL: Cleanup locally created arrays to prevent memory leaks
        # Note: img is a reference to np_image, don't delete it
        del glare_mask, padded_glare_mask, kernel

        return decision

    except Exception as e:
        logger.error(f"Error in glare detection: {e}", exc_info=True)
        return False
