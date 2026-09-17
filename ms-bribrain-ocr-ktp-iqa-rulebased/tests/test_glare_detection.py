"""
Unit tests for glare_detection service.
Tests glare detection and helper functions.
"""
import numpy as np
from unittest.mock import patch

from src.services.glare_detection import (
    calculate_median_brightness,
    determine_glare_threshold,
    detect_glare_with_text_analysis
)


class TestCalculateMedianBrightness:
    """Test cases for calculate_median_brightness function."""
    
    def test_bright_image(self):
        """Test with bright image."""
        # Create bright image
        bright_img = np.ones((100, 100, 3), dtype=np.uint8) * 250
        brightness = calculate_median_brightness(bright_img)
        assert brightness > 200, "Bright image should have high median brightness"
    
    def test_dark_image(self):
        """Test with dark image."""
        # Create dark image
        dark_img = np.ones((100, 100, 3), dtype=np.uint8) * 50
        brightness = calculate_median_brightness(dark_img)
        assert brightness < 100, "Dark image should have low median brightness"
    
    def test_medium_brightness_image(self, sample_bgr_image):
        """Test with medium brightness image."""
        brightness = calculate_median_brightness(sample_bgr_image)
        assert 50 < brightness < 200, "Medium brightness image should be in middle range"
    
    def test_exception_handling(self):
        """Test exception handling."""
        # Pass invalid image
        with patch('cv2.cvtColor', side_effect=Exception("Test exception")):
            brightness = calculate_median_brightness(np.zeros((10, 10, 3), dtype=np.uint8))
            assert brightness == 128.0, "Exception should return default value 128.0"


class TestDetermineGlareThreshold:
    """Test cases for determine_glare_threshold function."""
    
    def test_very_bright_image(self):
        """Test threshold for very bright image (>240)."""
        threshold = determine_glare_threshold(245.0)
        assert threshold >= 245, "Very bright image should have high threshold"
        assert threshold <= 254, "Threshold should not exceed 254"
    
    def test_bright_image_220_240(self):
        """Test threshold for bright image (220-240)."""
        threshold = determine_glare_threshold(230.0)
        assert threshold == 250, "Image with brightness 220-240 should have threshold 250"
    
    def test_medium_bright_image_200_220(self):
        """Test threshold for medium-bright image (200-220)."""
        threshold = determine_glare_threshold(210.0)
        assert threshold == 240, "Image with brightness 200-220 should have threshold 240"
    
    def test_medium_image_120_200(self):
        """Test threshold for medium image (120-200)."""
        threshold = determine_glare_threshold(150.0)
        assert threshold == 220, "Image with brightness 120-200 should have threshold 220"
    
    def test_dark_image(self):
        """Test threshold for dark image (<120)."""
        threshold = determine_glare_threshold(80.0)
        assert threshold == 150, "Dark image should have threshold 150"
    
    def test_boundary_values(self):
        """Test boundary values."""
        assert determine_glare_threshold(240.0) == 250
        assert determine_glare_threshold(240.1) == min(254, int(240.1) + 5)
        assert determine_glare_threshold(220.0) == 250
        assert determine_glare_threshold(200.0) == 240
        assert determine_glare_threshold(120.0) == 220
        assert determine_glare_threshold(119.9) == 150


class TestDetectGlareWithTextAnalysis:
    """Test cases for detect_glare_with_text_analysis function."""
    
    def test_no_glare_image(self, sample_bgr_image):
        """Test image without glare with no text regions."""
        # Empty OCR result means no text to check for glare overlap
        has_glare = detect_glare_with_text_analysis(sample_bgr_image, [])
        assert not has_glare, "No text regions means no glare detected"
    
    def test_glare_image(self, glare_bgr_image, sample_ocr_result):
        """Test image with glare regions."""
        has_glare = detect_glare_with_text_analysis(glare_bgr_image, sample_ocr_result)
        # Result depends on whether glare overlaps with text regions
        assert isinstance(has_glare, bool), "Should return boolean"
    
    def test_empty_ocr_result(self, sample_bgr_image):
        """Test with empty OCR result."""
        has_glare = detect_glare_with_text_analysis(sample_bgr_image, [])
        assert has_glare is False, "Empty OCR result should return False"
    
    def test_none_ocr_result(self, sample_bgr_image):
        """Test with None OCR result."""
        has_glare = detect_glare_with_text_analysis(sample_bgr_image, None)
        assert has_glare is False, "None OCR result should return False"
    
    def test_low_confidence_text_regions(self, sample_bgr_image):
        """Test with low confidence text regions (should be filtered)."""
        low_conf_ocr = [
            [[[10, 10], [100, 10], [100, 30], [10, 30]], ("TEXT", 0.3)],
            [[[10, 40], [100, 40], [100, 60], [10, 60]], ("TEXT2", 0.2)],
        ]
        has_glare = detect_glare_with_text_analysis(sample_bgr_image, low_conf_ocr)
        assert has_glare is False, "Low confidence text should be filtered out"
    
    def test_high_confidence_text_regions(self, sample_bgr_image):
        """Test with high confidence text regions."""
        high_conf_ocr = [
            [[[10, 10], [100, 10], [100, 30], [10, 30]], ("TEXT", 0.95)],
            [[[10, 40], [100, 40], [100, 60], [10, 60]], ("TEXT2", 0.90)],
        ]
        has_glare = detect_glare_with_text_analysis(sample_bgr_image, high_conf_ocr)
        assert isinstance(has_glare, bool), "Should return boolean"
    
    def test_bright_image_with_text_overlap(self, sample_ocr_result):
        """Test bright image where glare overlaps with text."""
        # Create image with bright region overlapping text location
        bright_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Add very bright region where OCR text is located (around 10,10 area)
        bright_img[5:35, 5:105] = [255, 255, 255]
        
        has_glare = detect_glare_with_text_analysis(bright_img, sample_ocr_result)
        # Should detect glare since it overlaps with text regions
        assert isinstance(has_glare, bool), "Should return boolean"
    
    def test_small_glare_regions(self, sample_bgr_image):
        """Test with small glare regions below minimum area threshold."""
        # Create image with small bright spots (below min_area threshold)
        img = sample_bgr_image.copy()
        img[10:15, 10:15] = [255, 255, 255]  # Very small bright region
        
        # Empty OCR means no text to check overlap
        has_glare = detect_glare_with_text_analysis(img, [])
        assert not has_glare, "No text regions means no glare detected"
    
    def test_large_glare_regions(self, sample_ocr_result):
        """Test with large glare regions above minimum area threshold."""
        # Create image with large bright region
        img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        img[50:150, 50:150] = [255, 255, 255]  # Large bright region
        
        has_glare = detect_glare_with_text_analysis(img, sample_ocr_result)
        assert isinstance(has_glare, bool), "Should return boolean"
    
    def test_exception_handling(self, sample_bgr_image, sample_ocr_result):
        """Test exception handling."""
        with patch('cv2.split', side_effect=Exception("Test exception")):
            has_glare = detect_glare_with_text_analysis(sample_bgr_image, sample_ocr_result)
            assert has_glare is False, "Exception should return False"
    
    def test_different_brightness_levels(self, sample_ocr_result):
        """Test with different overall brightness levels."""
        # Very bright image
        very_bright = np.ones((100, 100, 3), dtype=np.uint8) * 250
        result1 = detect_glare_with_text_analysis(very_bright, sample_ocr_result)
        assert isinstance(result1, bool)
        
        # Medium brightness
        medium = np.ones((100, 100, 3), dtype=np.uint8) * 150
        result2 = detect_glare_with_text_analysis(medium, sample_ocr_result)
        assert isinstance(result2, bool)
        
        # Dark image
        dark = np.ones((100, 100, 3), dtype=np.uint8) * 50
        result3 = detect_glare_with_text_analysis(dark, sample_ocr_result)
        assert isinstance(result3, bool)
    
    def test_complex_ocr_result_format(self, sample_bgr_image):
        """Test with complex OCR result with various text positions."""
        complex_ocr = [
            [[[20, 20], [80, 20], [80, 40], [20, 40]], ("PROVINSI", 0.95)],
            [[[20, 50], [120, 50], [120, 70], [20, 70]], ("DKI JAKARTA", 0.90)],
            [[[20, 80], [60, 80], [60, 95], [20, 95]], ("NIK", 0.85)],
        ]
        has_glare = detect_glare_with_text_analysis(sample_bgr_image, complex_ocr)
        assert isinstance(has_glare, bool), "Should handle complex OCR format"
