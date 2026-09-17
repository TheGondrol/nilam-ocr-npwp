"""
Unit tests for crop helper utilities.
"""

import pytest
import numpy as np
import cv2

from src.services.crop_helper import order_points, crop_image


class TestCropHelper:
    """Tests for crop helper functions."""

    def test_order_points(self):
        """Test point ordering function."""
        # Create 4 points in random order
        pts = np.array([
            [100, 50],   # top-right
            [10, 10],    # top-left
            [90, 150],   # bottom-right
            [5, 140]     # bottom-left
        ], dtype=np.float32)
        
        ordered = order_points(pts)
        
        # Verify order: top-left, top-right, bottom-right, bottom-left
        assert ordered[0][0] < 20  # top-left x
        assert ordered[0][1] < 20  # top-left y
        assert ordered[1][0] > 80  # top-right x
        assert ordered[1][1] < 60  # top-right y
        assert ordered[2][0] > 80  # bottom-right x
        assert ordered[2][1] > 140  # bottom-right y
        assert ordered[3][0] < 20  # bottom-left x
        assert ordered[3][1] > 130  # bottom-left y

    def test_crop_image_success(self):
        """Test successful image cropping."""
        # Create a simple test image (100x100 white rectangle)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        _, img_bytes = cv2.imencode('.jpg', img)
        img_bytes = img_bytes.tobytes()
        
        # Define bounding box (slightly smaller than image)
        bbox = [
            [10, 10],
            [90, 10],
            [90, 90],
            [10, 90]
        ]
        
        result = crop_image(img_bytes, bbox)
        
        # Verify result is bytes
        assert isinstance(result, bytes)
        assert len(result) > 0
        
        # Verify it can be decoded
        nparr = np.frombuffer(result, np.uint8)
        decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape[0] > 0
        assert decoded.shape[1] > 0

    def test_crop_image_with_perspective(self):
        """Test image cropping with perspective transformation."""
        # Create a test image
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), -1)
        _, img_bytes = cv2.imencode('.jpg', img)
        img_bytes = img_bytes.tobytes()
        
        # Define a skewed bounding box
        bbox = [
            [40, 40],
            [160, 50],
            [150, 160],
            [45, 155]
        ]
        
        result = crop_image(img_bytes, bbox)
        
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_crop_image_invalid_bbox_length(self):
        """Test crop_image with invalid bbox length."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        _, img_bytes = cv2.imencode('.jpg', img)
        img_bytes = img_bytes.tobytes()
        
        # Invalid bbox with only 3 points
        bbox = [[10, 10], [90, 10], [90, 90]]
        
        with pytest.raises(ValueError) as exc_info:
            crop_image(img_bytes, bbox)
        
        assert "exactly 4 points" in str(exc_info.value)

    def test_crop_image_invalid_point_coordinates(self):
        """Test crop_image with invalid point coordinates."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        _, img_bytes = cv2.imencode('.jpg', img)
        img_bytes = img_bytes.tobytes()
        
        # Invalid point with 3 coordinates instead of 2
        bbox = [[10, 10], [90, 10, 5], [90, 90], [10, 90]]
        
        with pytest.raises(ValueError) as exc_info:
            crop_image(img_bytes, bbox)
        
        assert "2 coordinates" in str(exc_info.value)

    def test_crop_image_empty_image(self):
        """Test crop_image with empty image bytes."""
        bbox = [[10, 10], [90, 10], [90, 90], [10, 90]]
        
        with pytest.raises((ValueError, cv2.error)):
            crop_image(b'', bbox)

    def test_crop_image_invalid_image_data(self):
        """Test crop_image with invalid image data."""
        bbox = [[10, 10], [90, 10], [90, 90], [10, 90]]
        
        with pytest.raises(ValueError) as exc_info:
            crop_image(b'not_an_image', bbox)
        
        assert "empty" in str(exc_info.value).lower()

    def test_crop_image_with_min_size(self):
        """Test crop_image respects minimum size."""
        # Create a small test image
        img = np.ones((50, 50, 3), dtype=np.uint8) * 255
        _, img_bytes = cv2.imencode('.jpg', img)
        img_bytes = img_bytes.tobytes()
        
        # Very small crop area
        bbox = [[20, 20], [22, 20], [22, 22], [20, 22]]
        
        result = crop_image(img_bytes, bbox, min_size=10)
        
        # Verify it doesn't crash and returns valid result
        assert isinstance(result, bytes)
        assert len(result) > 0
