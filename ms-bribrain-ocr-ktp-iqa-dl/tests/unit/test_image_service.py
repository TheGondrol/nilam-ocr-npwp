"""Unit tests for src.services.image_service module."""

import pytest
from PIL import Image

from src.schemas.api_schema import Crop
from src.services.image_service import bbox_to_rect, extract_crop, filter_crops


class TestBboxToRect:
    def test_simple_rectangle(self):
        bbox = [[10, 20], [100, 20], [100, 50], [10, 50]]
        assert bbox_to_rect(bbox) == (10, 20, 100, 50)

    def test_unordered_points(self):
        bbox = [[100, 50], [10, 20], [100, 20], [10, 50]]
        assert bbox_to_rect(bbox) == (10, 20, 100, 50)

    def test_zero_origin(self):
        bbox = [[0, 0], [50, 0], [50, 30], [0, 30]]
        assert bbox_to_rect(bbox) == (0, 0, 50, 30)

    def test_single_point(self):
        bbox = [[5, 5], [5, 5], [5, 5], [5, 5]]
        assert bbox_to_rect(bbox) == (5, 5, 5, 5)

    def test_float_truncation(self):
        bbox = [[10, 20], [100, 20], [100, 50], [10, 50]]
        x1, y1, x2, y2 = bbox_to_rect(bbox)
        assert isinstance(x1, int)
        assert isinstance(y1, int)

    def test_negative_coordinates(self):
        """Negative coords are valid inputs — min/max still produce correct bounds."""
        bbox = [[-10, -20], [50, -20], [50, 30], [-10, 30]]
        assert bbox_to_rect(bbox) == (-10, -20, 50, 30)


class TestExtractCrop:
    def test_valid_crop(self):
        img = Image.new("RGB", (200, 200), color=(255, 0, 0))
        bbox = [[10, 10], [50, 10], [50, 50], [10, 50]]
        result = extract_crop(img, bbox)
        assert result is not None
        assert result.size == (40, 40)

    def test_full_image_crop(self):
        img = Image.new("RGB", (100, 100))
        bbox = [[0, 0], [100, 0], [100, 100], [0, 100]]
        result = extract_crop(img, bbox)
        assert result is not None
        assert result.size == (100, 100)

    def test_crop_clamped_to_bounds(self):
        img = Image.new("RGB", (100, 100))
        bbox = [[-10, -10], [110, -10], [110, 110], [-10, 110]]
        result = extract_crop(img, bbox)
        assert result is not None
        assert result.size == (100, 100)

    def test_zero_area_returns_none(self):
        img = Image.new("RGB", (100, 100))
        bbox = [[50, 50], [50, 50], [50, 50], [50, 50]]
        result = extract_crop(img, bbox)
        assert result is None

    def test_inverted_coords_returns_none(self):
        img = Image.new("RGB", (100, 100))
        # After clamping: x1=100, x2=100 (min of image width) -> zero width
        bbox = [[200, 200], [300, 200], [300, 300], [200, 300]]
        result = extract_crop(img, bbox)
        assert result is None

    def test_invalid_bbox_returns_none(self):
        img = Image.new("RGB", (100, 100))
        result = extract_crop(img, [])
        assert result is None

    def test_partially_negative_bbox_clamped_to_valid_crop(self):
        """Coordinates partly outside image (negative side) are clamped to 0."""
        img = Image.new("RGB", (100, 100))
        bbox = [[-20, -10], [60, -10], [60, 40], [-20, 40]]
        result = extract_crop(img, bbox)
        assert result is not None
        assert result.size == (60, 40)  # clamped: x1=0, y1=0, x2=60, y2=40


class TestFilterCrops:
    def _make_crop(self, x1, y1, x2, y2, text="t"):
        return Crop(
            bbox=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            text=text,
            confidence=0.9,
        )

    def test_keeps_wide_crops(self):
        crops = [self._make_crop(0, 0, 200, 50)]  # ratio=4.0
        result = filter_crops(crops, min_width_ratio=1.0, min_width=0)
        assert len(result) == 1

    def test_removes_tall_crops(self):
        crops = [self._make_crop(0, 0, 20, 100)]  # ratio=0.2
        result = filter_crops(crops, min_width_ratio=1.0, min_width=0)
        assert len(result) == 0

    def test_min_width_filter(self):
        crops = [self._make_crop(0, 0, 30, 10)]  # width=30, ratio=3
        result = filter_crops(crops, min_width_ratio=1.0, min_width=50)
        assert len(result) == 0

    def test_mixed_crops(self):
        crops = [
            self._make_crop(0, 0, 200, 50),   # ratio=4, width=200 -> keep
            self._make_crop(0, 0, 20, 100),    # ratio=0.2 -> reject
            self._make_crop(0, 0, 100, 50),    # ratio=2, width=100 -> keep
        ]
        result = filter_crops(crops, min_width_ratio=1.0, min_width=50)
        assert len(result) == 2

    def test_empty_list(self):
        result = filter_crops([], min_width_ratio=1.0, min_width=0)
        assert result == []

    def test_zero_height_skipped(self):
        crops = [self._make_crop(0, 50, 100, 50)]  # height=0
        result = filter_crops(crops, min_width_ratio=0.0, min_width=0)
        assert len(result) == 0

    def test_zero_width_skipped(self):
        crops = [self._make_crop(50, 0, 50, 100)]  # width=0
        result = filter_crops(crops, min_width_ratio=0.0, min_width=0)
        assert len(result) == 0

    def test_min_width_ratio_zero_accepts_all_valid_dimension_crops(self):
        """ratio=0.0 means any non-zero width/height crop passes the ratio check."""
        crops = [
            self._make_crop(0, 0, 10, 100),  # tall, ratio=0.1 — still ≥ 0.0
            self._make_crop(0, 0, 100, 10),  # wide, ratio=10
        ]
        result = filter_crops(crops, min_width_ratio=0.0, min_width=0)
        assert len(result) == 2

    def test_min_width_exact_boundary(self):
        """Crop with width exactly equal to min_width passes."""
        crops = [self._make_crop(0, 0, 50, 10)]  # width=50
        assert len(filter_crops(crops, min_width_ratio=1.0, min_width=50)) == 1
        assert len(filter_crops(crops, min_width_ratio=1.0, min_width=51)) == 0
