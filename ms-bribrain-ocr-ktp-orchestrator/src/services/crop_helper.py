"""
Crop helper utilities for perspective transformation of images.

Provides functions for cropping images using 4-point bounding boxes
with automatic point ordering and perspective correction.
"""

from typing import List

import cv2
import numpy as np


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
    image_bytes: bytes,
    bbox: List[List[int]],
    min_size: int = 1
) -> bytes:
    """
    Crop an image using a 4-point bounding box with perspective correction.

    Args:
        image_bytes: Image as bytes (JPEG/PNG encoded).
        bbox: List of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]. A flat
            axis-aligned box [x1, y1, x2, y2] is also accepted (normalized to 4
            corner points).
        min_size: Minimum width/height for the output image (default: 1).

    Returns:
        Cropped image as JPEG bytes, ready for API calls.

    Raises:
        ValueError: If bbox doesn't contain exactly 4 points or image is invalid.
    """
    # Decode bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Input validation
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")
    
    # Tolerate a flat axis-aligned box [x1, y1, x2, y2] by normalizing it to 4
    # corner points. A degenerate/heuristic NIK box from the postprocess service
    # can arrive in this shape; normalizing avoids a TypeError on len(point)
    # that would otherwise discard an already-successful OCR result (BUG-03).
    if len(bbox) == 4 and all(isinstance(c, (int, float)) for c in bbox):
        x1, y1, x2, y2 = bbox
        bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

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
    
    # Encode as JPEG bytes
    _, buffer = cv2.imencode(".jpg", warped)
    return buffer.tobytes()