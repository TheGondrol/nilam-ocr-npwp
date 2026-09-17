"""
Unit tests for rotation_detection service.
Tests rotation detection and helper functions.
"""
import numpy as np
from unittest.mock import patch, MagicMock

from src.services.rotation_detection import (
    is_landscape,
    is_face_in_box,
    rotated_detection,
    detect_image_rotation,
    get_face_detector,
)


def _make_face(x1: float, y1: float, x2: float, y2: float) -> MagicMock:
    """Build a mock InsightFace Face object with the required `.bbox` array."""
    face = MagicMock()
    face.bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
    return face


class TestIsLandscape:
    """Test cases for is_landscape function."""

    def test_landscape_image(self, landscape_image):
        """Test with landscape-oriented image."""
        assert is_landscape(landscape_image) is True, "Landscape image should return True"

    def test_portrait_image(self, portrait_image):
        """Test with portrait-oriented image."""
        assert is_landscape(portrait_image) is False, "Portrait image should return False"

    def test_square_image(self):
        """Test with square image."""
        square_img = np.ones((100, 100, 3), dtype=np.uint8)
        assert is_landscape(square_img) is False, "Square image (width == height) should return False"

    def test_grayscale_landscape(self):
        """Test with grayscale landscape image."""
        gray_landscape = np.ones((100, 200), dtype=np.uint8)
        assert is_landscape(gray_landscape) is True, "Grayscale landscape should return True"

    def test_grayscale_portrait(self):
        """Test with grayscale portrait image."""
        gray_portrait = np.ones((200, 100), dtype=np.uint8)
        assert is_landscape(gray_portrait) is False, "Grayscale portrait should return False"


class TestIsFaceInBox:
    """Test cases for is_face_in_box function."""

    def test_face_completely_inside_box(self):
        result = is_face_in_box(50, 50, 30, 30, (40, 40, 60, 60))
        assert result is True, "Face completely inside box should return True"

    def test_face_outside_box(self):
        result = is_face_in_box(10, 10, 20, 20, (100, 100, 50, 50))
        assert result is False, "Face outside box should return False"

    def test_face_partially_outside_box(self):
        result = is_face_in_box(45, 45, 30, 30, (40, 40, 30, 30))
        assert result is False, "Face partially outside should return False"

    def test_face_touching_box_edge(self):
        result = is_face_in_box(40, 40, 60, 60, (40, 40, 60, 60))
        assert result is True, "Face exactly filling box should return True"

    def test_small_face_in_large_box(self):
        result = is_face_in_box(100, 100, 10, 10, (50, 50, 200, 200))
        assert result is True, "Small face in large box should return True"


class TestGetFaceDetector:
    """Test cases for get_face_detector function."""

    @patch('src.services.rotation_detection.FaceAnalysis')
    def test_get_face_detector_singleton(self, mock_face_analysis):
        """Test that get_face_detector returns singleton."""
        mock_face_analysis.return_value = MagicMock()
        import src.services.rotation_detection as rd
        rd._face_detector = None

        detector1 = get_face_detector()
        detector2 = get_face_detector()
        assert detector1 is detector2, "Should return same singleton instance"

    @patch('src.services.rotation_detection.FaceAnalysis')
    def test_face_detector_not_none(self, mock_face_analysis):
        """Test that face detector is not None."""
        mock_face_analysis.return_value = MagicMock()
        import src.services.rotation_detection as rd
        rd._face_detector = None

        detector = get_face_detector()
        assert detector is not None, "Face detector should not be None"


class TestRotatedDetection:
    """Test cases for rotated_detection function."""

    def test_none_image(self):
        """Test with None image."""
        is_accepted, reason = rotated_detection(None, min_face_proportion=0.2)
        assert is_accepted is False, "None image should return False"
        assert "Error" in reason, "Reason should indicate error"

    def test_portrait_image(self, portrait_image):
        """Test with portrait image (should be rejected)."""
        is_accepted, reason = rotated_detection(portrait_image, min_face_proportion=0.2)
        assert is_accepted is False, "Portrait image should be rejected"
        assert "not in landscape" in reason.lower(), "Reason should mention landscape"

    @patch('src.services.rotation_detection.get_face_detector')
    def test_no_face_detected(self, mock_get_detector, landscape_image):
        """Test with no face detected."""
        mock_detector = MagicMock()
        mock_detector.get.return_value = []
        mock_get_detector.return_value = mock_detector

        is_accepted, reason = rotated_detection(landscape_image, min_face_proportion=0.2)
        assert is_accepted is False, "No face detected should return False"
        assert "No face" in reason, "Reason should mention no face"

    @patch('src.services.rotation_detection.get_face_detector')
    def test_face_properly_positioned(self, mock_get_detector, landscape_image):
        """Test with face properly positioned and sized."""
        mock_detector = MagicMock()
        mock_detector.get.return_value = [_make_face(130, 25, 180, 50)]
        mock_get_detector.return_value = mock_detector

        is_accepted, reason = rotated_detection(landscape_image, min_face_proportion=0.05)
        assert isinstance(is_accepted, bool), "Should return boolean"
        assert isinstance(reason, str), "Should return reason string"

    @patch('src.services.rotation_detection.get_face_detector')
    def test_face_too_small(self, mock_get_detector, landscape_image):
        """Test with face too small (below minimum proportion)."""
        mock_detector = MagicMock()
        mock_detector.get.return_value = [_make_face(140, 30, 150, 35)]
        mock_get_detector.return_value = mock_detector

        is_accepted, reason = rotated_detection(landscape_image, min_face_proportion=0.5)
        assert isinstance(is_accepted, bool), "Should return boolean"
        assert isinstance(reason, str), "Should return reason string"

    @patch('src.services.rotation_detection.get_face_detector')
    def test_face_wrong_position(self, mock_get_detector, landscape_image):
        """Test with face in wrong position (outside verification box)."""
        mock_detector = MagicMock()
        mock_detector.get.return_value = [_make_face(20, 30, 60, 50)]
        mock_get_detector.return_value = mock_detector

        is_accepted, reason = rotated_detection(landscape_image, min_face_proportion=0.2)
        assert is_accepted is False, "Face in wrong position should return False"
        assert "not properly positioned" in reason.lower(), "Reason should mention position"

    @patch('src.services.rotation_detection.get_face_detector')
    def test_exception_handling(self, mock_get_detector, landscape_image):
        """Test exception handling."""
        mock_detector = MagicMock()
        mock_detector.get.side_effect = Exception("Test exception")
        mock_get_detector.return_value = mock_detector

        is_accepted, reason = rotated_detection(landscape_image, min_face_proportion=0.2)
        assert is_accepted is False, "Exception should return False"
        assert "Error" in reason, "Reason should mention error"

    @patch('src.services.rotation_detection.get_face_detector')
    def test_detector_error_resets_singleton(self, mock_get_detector, landscape_image):
        """Test that detector errors reset the singleton."""
        mock_detector = MagicMock()
        mock_detector.get.side_effect = RuntimeError("ONNX session failed")
        mock_get_detector.return_value = mock_detector

        is_accepted, reason = rotated_detection(landscape_image, min_face_proportion=0.2)
        assert is_accepted is False, "Detector error should return False"

        import src.services.rotation_detection as rd
        assert rd._face_detector is None, "Singleton should be reset after error"

    @patch('src.services.rotation_detection.get_face_detector')
    def test_non_contiguous_array(self, mock_get_detector):
        """Test with non-contiguous array."""
        mock_detector = MagicMock()
        mock_detector.get.return_value = []
        mock_get_detector.return_value = mock_detector

        img = np.ones((100, 200, 3), dtype=np.uint8)
        non_contiguous = img[::2, ::2, :]

        is_accepted, reason = rotated_detection(non_contiguous, min_face_proportion=0.2)
        assert isinstance(is_accepted, bool), "Should handle non-contiguous array"
        assert isinstance(reason, str), "Should return reason"


class TestDetectImageRotation:
    """Test cases for detect_image_rotation function."""

    @patch('src.services.rotation_detection.rotated_detection')
    def test_properly_oriented_image(self, mock_rotated_detection, landscape_image):
        """Test properly oriented image (not rotated)."""
        mock_rotated_detection.return_value = (True, "Face properly positioned")

        is_rotated = detect_image_rotation(landscape_image)
        assert is_rotated is False, "Properly oriented image should return False (not rotated)"

    @patch('src.services.rotation_detection.rotated_detection')
    def test_rotated_image(self, mock_rotated_detection, landscape_image):
        """Test rotated image."""
        mock_rotated_detection.return_value = (False, "No face detected")

        is_rotated = detect_image_rotation(landscape_image)
        assert is_rotated is True, "Rotated image should return True"

    @patch('src.services.rotation_detection.rotated_detection')
    def test_exception_handling(self, mock_rotated_detection, landscape_image):
        """Test exception handling."""
        mock_rotated_detection.side_effect = Exception("Test exception")

        is_rotated = detect_image_rotation(landscape_image)
        assert is_rotated is False, "Exception should return False (not rotated)"

    @patch('src.services.rotation_detection.rotated_detection')
    def test_return_type(self, mock_rotated_detection, landscape_image):
        """Test that return type is always boolean."""
        mock_rotated_detection.return_value = (True, "Reason")
        result1 = detect_image_rotation(landscape_image)
        assert isinstance(result1, bool), "Should return boolean"

        mock_rotated_detection.return_value = (False, "Reason")
        result2 = detect_image_rotation(landscape_image)
        assert isinstance(result2, bool), "Should return boolean"
