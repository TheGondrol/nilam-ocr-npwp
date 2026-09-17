"""Image processing service for crop extraction and filtering."""

from __future__ import annotations

from typing import Optional

from PIL import Image

from src.schemas.api_schema import Crop


def bbox_to_rect(bbox: list[list[int]]) -> tuple[int, int, int, int]:
    """
    Convert bounding box coordinates to rectangle format.

    Args:
        bbox: List of 4 points [[x,y], [x,y], [x,y], [x,y]]

    Returns:
        Tuple of (x1, y1, x2, y2) representing rectangle corners
    """
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def extract_crop(image: Image.Image, bbox: list[list[int]]) -> Optional[Image.Image]:
    """
    Extract a crop from image using bounding box.

    Args:
        image: PIL Image to crop from
        bbox: Bounding box coordinates

    Returns:
        Cropped PIL Image or None if invalid crop
    """
    try:
        x1, y1, x2, y2 = bbox_to_rect(bbox)

        # Clamp coordinates to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.width, x2)
        y2 = min(image.height, y2)

        # Check if valid crop
        if x2 <= x1 or y2 <= y1:
            return None

        return image.crop((x1, y1, x2, y2))
    except Exception:
        return None


def filter_crops(
    crops: list[Crop],
    min_width_ratio: float,
    min_width: int,
) -> list[Crop]:
    """
    Filter crops based on width/height ratio and minimum width.

    This filters out non-text crops like signs, watermarks, etc.
    Text crops typically have width > height.

    Args:
        crops: List of Crop objects
        min_width_ratio: Minimum width/height ratio (e.g., 1.0 for width >= height)
        min_width: Minimum width in pixels

    Returns:
        Filtered list of Crop objects
    """
    filtered: list[Crop] = []

    for crop in crops:
        try:
            x1, y1, x2, y2 = bbox_to_rect(crop.bbox)
            width = x2 - x1
            height = y2 - y1

            # Skip invalid dimensions
            if height <= 0 or width <= 0:
                continue

            ratio = width / height

            # Keep if passes both filters
            if ratio >= min_width_ratio and width >= min_width:
                filtered.append(crop)
        except Exception:
            # Skip crops that fail processing
            continue

    return filtered
