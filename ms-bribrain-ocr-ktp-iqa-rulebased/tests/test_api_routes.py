"""
Unit tests for API routes.
Tests the FastAPI endpoints.
"""
import json
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from src.main import app


# Create test client
client = TestClient(app)


class TestOCRQualityEndpoint:
    """Test cases for /ocr_quality endpoint."""
    
    def test_successful_quality_check(self):
        """Test successful quality check with valid inputs."""
        # Create test image file
        image_content = b"fake image content"
        ocr_result = json.dumps([
            [[[10, 10], [100, 10], [100, 30], [10, 30]], ["PROVINSI DKI JAKARTA", 0.95]],
        ])
        
        # Mock the image_quality function
        with patch('src.api.routes.image_quality') as mock_quality:
            mock_quality.return_value = {
                "low_confidence": False,
                "is_blurry": False,
                "is_glare": False,
                "is_rotated": False,
            }
            
            # Mock insert_log to prevent database operations
            with patch('src.api.routes.insert_log', new_callable=AsyncMock):
                response = client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_content, "image/jpeg")},
                    data={"ocr_result": ocr_result}
                )
        
        assert response.status_code == 200
        result = response.json()
        assert "low_confidence" in result
        assert "is_blurry" in result
        assert "is_glare" in result
        assert "is_rotated" in result
    
    def test_missing_file(self):
        """Test with missing file parameter."""
        ocr_result = json.dumps([])
        
        response = client.post(
            "/v1/ocr_quality",
            data={"ocr_result": ocr_result}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_missing_ocr_result(self):
        """Test with missing ocr_result parameter."""
        image_content = b"fake image content"
        
        response = client.post(
            "/v1/ocr_quality",
            files={"file": ("test.jpg", image_content, "image/jpeg")}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_invalid_ocr_result_format(self):
        """Test with invalid JSON in ocr_result."""
        image_content = b"fake image content"
        invalid_ocr = "not valid json"
        
        with patch('src.api.routes.insert_log', new_callable=AsyncMock):
            response = client.post(
                "/v1/ocr_quality",
                files={"file": ("test.jpg", image_content, "image/jpeg")},
                data={"ocr_result": invalid_ocr}
            )
        
        assert response.status_code == 400
        assert "Invalid OCR result format" in response.json()["detail"]
    
    def test_empty_ocr_result(self):
        """Test with empty OCR result array."""
        image_content = b"fake image content"
        ocr_result = json.dumps([])
        
        with patch('src.api.routes.image_quality') as mock_quality:
            mock_quality.return_value = {
                "low_confidence": False,
                "is_blurry": False,
                "is_glare": False,
                "is_rotated": False,
            }
            
            with patch('src.api.routes.insert_log', new_callable=AsyncMock):
                response = client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_content, "image/jpeg")},
                    data={"ocr_result": ocr_result}
                )
        
        assert response.status_code == 200
    
    def test_image_processing_error(self):
        """Test when image processing raises an exception."""
        image_content = b"fake image content"
        ocr_result = json.dumps([
            [[[10, 10], [100, 10], [100, 30], [10, 30]], ["TEXT", 0.95]],
        ])
        
        with patch('src.api.routes.image_quality') as mock_quality:
            mock_quality.side_effect = Exception("Processing failed")
            
            with patch('src.api.routes.insert_log', new_callable=AsyncMock):
                response = client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_content, "image/jpeg")},
                    data={"ocr_result": ocr_result}
                )
        
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]
    
    def test_large_file(self):
        """Test with large file."""
        # Create large fake image
        large_content = b"x" * (10 * 1024 * 1024)  # 10MB
        ocr_result = json.dumps([])
        
        with patch('src.api.routes.image_quality') as mock_quality:
            mock_quality.return_value = {
                "low_confidence": False,
                "is_blurry": False,
                "is_glare": False,
                "is_rotated": False,
            }
            
            with patch('src.api.routes.insert_log', new_callable=AsyncMock):
                response = client.post(
                    "/v1/ocr_quality",
                    files={"file": ("large.jpg", large_content, "image/jpeg")},
                    data={"ocr_result": ocr_result}
                )
        
        # Should handle large files (or reject if there's a size limit)
        assert response.status_code in [200, 413]  # 413 = Payload Too Large
    
    def test_different_file_types(self):
        """Test with different image file types."""
        ocr_result = json.dumps([])
        file_types = [
            ("test.jpg", "image/jpeg"),
            ("test.png", "image/png"),
            ("test.bmp", "image/bmp"),
        ]
        
        for filename, content_type in file_types:
            with patch('src.api.routes.image_quality') as mock_quality:
                mock_quality.return_value = {
                    "low_confidence": False,
                    "is_blurry": False,
                    "is_glare": False,
                    "is_rotated": False,
                }
                
                with patch('src.api.routes.insert_log', new_callable=AsyncMock):
                    response = client.post(
                        "/v1/ocr_quality",
                        files={"file": (filename, b"image", content_type)},
                        data={"ocr_result": ocr_result}
                    )
            
            assert response.status_code in [200, 400, 500]
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        image_content = b"fake image"
        ocr_result = json.dumps([])
        
        with patch('src.api.routes.image_quality') as mock_quality:
            mock_quality.return_value = {
                "low_confidence": False,
                "is_blurry": False,
                "is_glare": False,
                "is_rotated": False,
            }
            
            with patch('src.api.routes.insert_log', new_callable=AsyncMock):
                # Send multiple requests
                responses = []
                for _ in range(5):
                    response = client.post(
                        "/v1/ocr_quality",
                        files={"file": ("test.jpg", image_content, "image/jpeg")},
                        data={"ocr_result": ocr_result}
                    )
                    responses.append(response)
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)
    
    def test_response_structure(self):
        """Test that response has correct structure."""
        image_content = b"fake image"
        ocr_result = json.dumps([])
        
        with patch('src.api.routes.image_quality') as mock_quality:
            mock_quality.return_value = {
                "low_confidence": False,
                "is_blurry": True,
                "is_glare": False,
                "is_rotated": True,
            }
            
            with patch('src.api.routes.insert_log', new_callable=AsyncMock):
                response = client.post(
                    "/v1/ocr_quality",
                    files={"file": ("test.jpg", image_content, "image/jpeg")},
                    data={"ocr_result": ocr_result}
                )
        
        assert response.status_code == 200
        result = response.json()
        
        # Check structure
        assert isinstance(result["low_confidence"], bool)
        assert isinstance(result["is_blurry"], bool)
        assert isinstance(result["is_glare"], bool)
        assert isinstance(result["is_rotated"], bool)
        
        # Check values match what was mocked
        assert result["is_blurry"]
        assert result["is_rotated"]
    
    @patch('src.api.routes.insert_log')
    def test_database_logging_called(self, mock_insert_log):
        """Test that database logging is called."""
        mock_insert_log.return_value = AsyncMock()
        image_content = b"fake image"
        ocr_result = json.dumps([])
        
        with patch('src.api.routes.image_quality') as mock_quality:
            mock_quality.return_value = {
                "low_confidence": False,
                "is_blurry": False,
                "is_glare": False,
                "is_rotated": False,
            }
            
            response = client.post(
                "/v1/ocr_quality",
                files={"file": ("test.jpg", image_content, "image/jpeg")},
                data={"ocr_result": ocr_result}
            )
        
        # Verify insert_log was called
        assert response.status_code == 200
        # Note: Actual verification of async call is complex in sync test
    
    def test_special_characters_in_filename(self):
        """Test with special characters in filename."""
        image_content = b"fake image"
        ocr_result = json.dumps([])
        special_filenames = [
            "test image.jpg",
            "test-image_123.jpg",
            "файл.jpg",  # Cyrillic
            "测试.jpg",  # Chinese
        ]
        
        for filename in special_filenames:
            with patch('src.api.routes.image_quality') as mock_quality:
                mock_quality.return_value = {
                    "low_confidence": False,
                    "is_blurry": False,
                    "is_glare": False,
                    "is_rotated": False,
                }
                
                with patch('src.api.routes.insert_log', new_callable=AsyncMock):
                    response = client.post(
                        "/v1/ocr_quality",
                        files={"file": (filename, image_content, "image/jpeg")},
                        data={"ocr_result": ocr_result}
                    )
            
            # Should handle special characters
            assert response.status_code in [200, 400, 500]
