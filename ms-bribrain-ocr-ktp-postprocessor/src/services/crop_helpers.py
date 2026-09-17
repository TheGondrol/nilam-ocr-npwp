"""
Crop helper utilities for perspective transformation of images.

Provides functions for cropping images using 4-point bounding boxes
with automatic point ordering and perspective correction.
"""

import cv2  # type: ignore[unresolved-import]
import numpy as np  # type: ignore[unresolved-import]
from typing import List
import base64


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points in clockwise order starting from top-left.

    Args:
        pts: Array of 4 points with shape (4, 2).

    Returns:
        Ordered points: [top-left, top-right, bottom-right, bottom-left]
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum of coordinates: smallest = top-left, largest = bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    # Difference of coordinates: smallest = top-right, largest = bottom-left
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def crop_image(
    image: np.ndarray,
    bbox: List[List[int]],
    min_size: int = 1
) -> str:
    """
    Crop an image using a 4-point bounding box with perspective correction.

    Args:
        image: OpenCV image (H, W, C) in BGR format.
        bbox: List of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
        min_size: Minimum width/height for the output image (default: 1).

    Returns:
        Cropped and perspective-corrected image as numpy array.

    Raises:
        ValueError: If bbox doesn't contain exactly 4 points or image is invalid.
    """
    # Input validation
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")
    
    if len(bbox) != 4:
        raise ValueError(f"Bounding box must have exactly 4 points, got {len(bbox)}")
    
    for i, point in enumerate(bbox):
        if len(point) != 2:
            raise ValueError(f"Point {i} must have 2 coordinates, got {len(point)}")

    pts = np.array(bbox, dtype=np.float32)

    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = order_points(pts)
    tl, tr, br, bl = rect

    # Compute output width (max of top and bottom edges)
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(max(width_top, width_bottom)), min_size)

    # Compute output height (max of left and right edges)
    height_left = np.linalg.norm(tl - bl)
    height_right = np.linalg.norm(tr - br)
    max_height = max(int(max(height_left, height_right)), min_size)

    # Destination points for perspective transform
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    # Apply perspective transform
    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))
    _, buffer = cv2.imencode(".jpg", warped)
    return base64.b64encode(buffer.tobytes()).decode()