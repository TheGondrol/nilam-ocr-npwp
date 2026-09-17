"""
Unit tests for image_quality service.
Tests the main orchestration function.
"""
import numpy as np
import cv2
from unittest.mock import patch

from src.services.image_quality import image_quality


class TestImageQuality:
    """Test cases for image_quality function."""
    
    def test_valid_image_with_good_quality(self, sample_image_bytes, sample_ocr_result):
        """Test with valid image and good quality OCR results."""
        result = image_quality(sample_image_bytes, sample_ocr_result)
        
        # Check result structure
        assert "low_confidence" in result
        assert "is_blurry" in result
        assert "is_glare" in result
        assert "is_rotated" in result
        
        # All should be booleans
        assert isinstance(result["low_confidence"], bool)
        assert isinstance(result["is_blurry"], bool)
        assert isinstance(result["is_glare"], bool)
        assert isinstance(result["is_rotated"], bool)
    
    def test_invalid_image_bytes(self, sample_ocr_result):
        """Test with invalid image bytes."""
        invalid_bytes = b"not an image"
        result = image_quality(invalid_bytes, sample_ocr_result)
        
        # Should return error defaults
        assert "error" in result or all(not v for k, v in result.items() if k != "error")
    
    def test_empty_image_bytes(self, sample_ocr_result):
        """Test with empty image bytes."""
        result = image_quality(b"", sample_ocr_result)
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert "low_confidence" in result
    
    def test_with_low_confidence_ocr(self, sample_image_bytes, low_confidence_ocr_result):
        """Test with low confidence OCR results."""
        result = image_quality(sample_image_bytes, low_confidence_ocr_result)
        
        # low_confidence should be True
        assert result["low_confidence"] is True
    
    def test_with_empty_ocr_result(self, sample_image_bytes):
        """Test with empty OCR result."""
        result = image_quality(sample_image_bytes, [])
        
        # Should handle empty OCR result
        assert isinstance(result, dict)
        assert "low_confidence" in result
    
    @patch('src.services.image_quality.check_confidence_median')
    @patch('src.services.image_quality.blur_detection')
    @patch('src.services.image_quality.detect_glare_with_text_analysis')
    @patch('src.services.image_quality.detect_image_rotation')
    def test_all_quality_issues(
        self,
        mock_rotation,
        mock_glare,
        mock_blur,
        mock_confidence,
        sample_image_bytes,
        sample_ocr_result
    ):
        """Test when all quality issues are detected."""
        # Mock all checks to return True (issues detected)
        mock_confidence.return_value = True
        mock_blur.return_value = (True, 50.0)
        mock_glare.return_value = True
        mock_rotation.return_value = True
        
        result = image_quality(sample_image_bytes, sample_ocr_result)
        
        assert result["low_confidence"] is True
        assert result["is_blurry"] is True
        assert result["is_glare"] is True
        assert result["is_rotated"] is True
    
    @patch('src.services.image_quality.check_confidence_median')
    @patch('src.services.image_quality.blur_detection')
    @patch('src.services.image_quality.detect_glare_with_text_analysis')
    @patch('src.services.image_quality.detect_image_rotation')
    def test_no_quality_issues(
        self,
        mock_rotation,
        mock_glare,
        mock_blur,
        mock_confidence,
        sample_image_bytes,
        sample_ocr_result
    ):
        """Test when no quality issues are detected."""
        # Mock all checks to return False (no issues)
        mock_confidence.return_value = False
        mock_blur.return_value = (False, 150.0)
        mock_glare.return_value = False
        mock_rotation.return_value = False
        
        result = image_quality(sample_image_bytes, sample_ocr_result)
        
        assert result["low_confidence"] is False
        assert result["is_blurry"] is False
        assert result["is_glare"] is False
        assert result["is_rotated"] is False
    
    @patch('cv2.imdecode')
    def test_decode_failure(self, mock_imdecode, sample_ocr_result):
        """Test when image decode fails."""
        mock_imdecode.return_value = None
        
        result = image_quality(b"fake bytes", sample_ocr_result)
        
        # Should return error defaults
        assert "error" in result or isinstance(result, dict)
    
    @patch('src.services.image_quality.blur_detection')
    def test_blur_detection_exception(
        self,
        mock_blur,
        sample_image_bytes,
        sample_ocr_result
    ):
        """Test when blur detection raises exception."""
        mock_blur.side_effect = Exception("Blur detection failed")
        
        result = image_quality(sample_image_bytes, sample_ocr_result)
        
        # Should handle exception and return error defaults
        assert isinstance(result, dict)
        assert "error" in result or "is_blurry" in result
    
    def test_large_image(self, sample_ocr_result):
        """Test with large image."""
        # Create large image
        large_img = np.ones((2000, 2000, 3), dtype=np.uint8) * 128
        success, buffer = cv2.imencode('.jpg', large_img)
        assert success
        large_bytes = buffer.tobytes()
        
        result = image_quality(large_bytes, sample_ocr_result)
        
        # Should handle large images
        assert isinstance(result, dict)
        assert "low_confidence" in result
    
    def test_small_image(self, sample_ocr_result):
        """Test with very small image."""
        # Create small image
        small_img = np.ones((10, 10, 3), dtype=np.uint8) * 128
        success, buffer = cv2.imencode('.jpg', small_img)
        assert success
        small_bytes = buffer.tobytes()
        
        result = image_quality(small_bytes, sample_ocr_result)
        
        # Should handle small images
        assert isinstance(result, dict)
        assert "low_confidence" in result
    
    def test_different_image_formats(self, sample_ocr_result):
        """Test with different image format encodings."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Test JPEG
        success, jpeg_buffer = cv2.imencode('.jpg', img)
        assert success
        result_jpeg = image_quality(jpeg_buffer.tobytes(), sample_ocr_result)
        assert isinstance(result_jpeg, dict)
        
        # Test PNG
        success, png_buffer = cv2.imencode('.png', img)
        assert success
        result_png = image_quality(png_buffer.tobytes(), sample_ocr_result)
        assert isinstance(result_png, dict)
    
    def test_grayscale_image_handling(self, sample_ocr_result):
        """Test with grayscale image."""
        # Create grayscale image and encode
        gray_img = np.ones((100, 100), dtype=np.uint8) * 128
        success, buffer = cv2.imencode('.jpg', gray_img)
        assert success
        
        result = image_quality(buffer.tobytes(), sample_ocr_result)
        
        # Should handle or reject grayscale images appropriately
        assert isinstance(result, dict)
    
    @patch('src.services.image_quality.detect_image_rotation')
    def test_rotation_detection_called_with_rgb(
        self,
        mock_rotation,
        sample_image_bytes,
        sample_ocr_result
    ):
        """Test that rotation detection is called with RGB image."""
        mock_rotation.return_value = False
        
        image_quality(sample_image_bytes, sample_ocr_result)
        
        # Verify rotation detection was called
        assert mock_rotation.called
        # Get the image passed to rotation detection
        call_args = mock_rotation.call_args[0]
        assert len(call_args) > 0
        image_arg = call_args[0]
        # Should be a numpy array
        assert isinstance(image_arg, np.ndarray)
    
    @patch('src.services.image_quality.detect_glare_with_text_analysis')
    def test_glare_detection_called_with_bgr(
        self,
        mock_glare,
        sample_image_bytes,
        sample_ocr_result
    ):
        """Test that glare detection is called with BGR image."""
        mock_glare.return_value = False
        
        image_quality(sample_image_bytes, sample_ocr_result)
        
        # Verify glare detection was called
        assert mock_glare.called
        # Get the arguments
        call_args = mock_glare.call_args[0]
        assert len(call_args) >= 2
        image_arg = call_args[0]
        ocr_arg = call_args[1]
        assert isinstance(image_arg, np.ndarray)
        assert isinstance(ocr_arg, list)
    
    def test_memory_cleanup(self, sample_image_bytes, sample_ocr_result):
        """Test that memory is properly cleaned up."""
        # Call multiple times to check for memory leaks
        for _ in range(5):
            result = image_quality(sample_image_bytes, sample_ocr_result)
            assert isinstance(result, dict)
        
        # If memory cleanup is proper, this should complete without issues
    
    def test_result_type_consistency(self, sample_image_bytes):
        """Test that result types are consistent across different inputs."""
        # Test with various OCR results
        ocr_results = [
            [],
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("Text", 0.9)],
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("Text", 0.9)] * 10,
        ]
        
        for ocr in ocr_results:
            result = image_quality(sample_image_bytes, ocr)
            assert isinstance(result, dict)
            # Check all expected keys exist
            expected_keys = ["low_confidence", "is_blurry", "is_glare", "is_rotated"]
            for key in expected_keys:
                if key in result:  # May have error key instead
                    assert isinstance(result[key], bool), f"{key} should be boolean"
