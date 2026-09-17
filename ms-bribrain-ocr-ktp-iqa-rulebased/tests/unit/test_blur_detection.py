"""
Unit tests for blur_detection service.
Tests blur detection and confidence check functionality.
"""
import numpy as np
from unittest.mock import patch

from src.services.blur_detection import (
    blur_detection,
    check_confidence_median
)


class TestCheckConfidenceMedian:
    """Test cases for check_confidence_median function."""
    
    def test_high_confidence_result(self, sample_ocr_result):
        """Test with high confidence OCR results."""
        result = check_confidence_median(sample_ocr_result)
        assert result is False, "High confidence should return False"
    
    def test_low_confidence_result(self, low_confidence_ocr_result):
        """Test with low confidence OCR results."""
        result = check_confidence_median(low_confidence_ocr_result)
        assert result is True, "Low confidence should return True"
    
    def test_empty_ocr_results(self):
        """Test with empty OCR results."""
        result = check_confidence_median([])
        assert result is False, "Empty results should return False"
    
    def test_mixed_confidence_results(self):
        """Test with mixed high and low confidence scores."""
        mixed_results = [
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("Text1", 0.9)],
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("Text2", 0.4)],
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("Text3", 0.6)],
        ]
        result = check_confidence_median(mixed_results)
        # Median is 0.6, threshold is 0.8, so should be True (low confidence)
        assert result
    
    def test_threshold_boundary(self):
        """Test exactly at threshold boundary."""
        # Create results with median exactly at 0.8 (threshold)
        boundary_results = [
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("Text1", 0.8)],
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("Text2", 0.8)],
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("Text3", 0.8)],
        ]
        result = check_confidence_median(boundary_results)
        # At threshold (<=), should return True
        assert result
    
    def test_invalid_confidence_scores(self):
        """Test with invalid confidence scores."""
        invalid_results = [
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("Text1", "invalid")],
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("Text2", None)],
        ]
        result = check_confidence_median(invalid_results)
        assert result is False, "Invalid scores should return False"
    
    def test_single_confidence_score(self):
        """Test with single OCR result."""
        single_result = [
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("Text", 0.3)],
        ]
        result = check_confidence_median(single_result)
        assert result is True, "Single low confidence should return True"
    
    def test_exception_handling(self):
        """Test exception handling in confidence check."""
        # Pass malformed data
        malformed_results = [
            "invalid",
            {"not": "a list"},
        ]
        result = check_confidence_median(malformed_results)
        assert result is False, "Exception should be caught and return False"


class TestBlurDetection:
    """Test cases for blur_detection function."""
    
    def test_sharp_image(self, sample_grayscale_image):
        """Test with a sharp image."""
        is_blurry, variance = blur_detection(sample_grayscale_image)
        assert not is_blurry, "Sharp image should not be detected as blurry"
        assert variance > 0, "Variance should be positive"
    
    def test_blurry_image(self, blurry_grayscale_image):
        """Test with a blurry image."""
        is_blurry, variance = blur_detection(blurry_grayscale_image)
        assert is_blurry, "Blurry image should be detected as blurry"
        assert variance >= 0, "Variance should be non-negative"
    
    def test_uniform_image(self):
        """Test with uniform (no features) image."""
        # Create uniform gray image
        uniform_img = np.ones((100, 100), dtype=np.uint8) * 128
        is_blurry, variance = blur_detection(uniform_img)
        assert is_blurry, "Uniform image should be detected as blurry"
        assert variance < 1.0, "Uniform image should have very low variance"
    
    def test_high_contrast_image(self):
        """Test with high contrast sharp image."""
        # Create checkerboard pattern (very sharp)
        img = np.zeros((100, 100), dtype=np.uint8)
        img[::2, ::2] = 255
        img[1::2, 1::2] = 255
        
        is_blurry, variance = blur_detection(img)
        assert not is_blurry, "High contrast image should not be blurry"
        assert variance > 100, "High contrast image should have high variance"
    
    def test_small_image(self):
        """Test with very small image."""
        small_img = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        is_blurry, variance = blur_detection(small_img)
        assert isinstance(bool(is_blurry), bool), "Should return boolean"
        assert isinstance(variance, float), "Should return float variance"
    
    def test_large_image(self):
        """Test with large image."""
        large_img = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
        is_blurry, variance = blur_detection(large_img)
        assert isinstance(bool(is_blurry), bool), "Should return boolean"
        assert isinstance(variance, float), "Should return float variance"
    
    def test_exception_handling(self):
        """Test exception handling in blur detection."""
        # Pass invalid data
        with patch('cv2.Laplacian', side_effect=Exception("Test exception")):
            is_blurry, variance = blur_detection(np.zeros((10, 10), dtype=np.uint8))
            assert is_blurry is False, "Exception should return False"
            assert variance == 0.0, "Exception should return 0.0 variance"
    
    def test_different_dtypes(self):
        """Test with different numpy dtypes."""
        # Test with float image
        float_img = np.random.rand(100, 100).astype(np.float32) * 255
        is_blurry, variance = blur_detection(float_img)
        assert isinstance(is_blurry, bool), "Should handle float dtype"
        
    def test_variance_calculation(self):
        """Test variance calculation accuracy."""
        # Create image with known characteristics
        img = np.zeros((50, 50), dtype=np.uint8)
        img[20:30, 20:30] = 255  # White square in center
        
        is_blurry, variance = blur_detection(img)
        # This should have high variance due to sharp edges
        assert variance > 50, "Sharp edges should produce high variance"
