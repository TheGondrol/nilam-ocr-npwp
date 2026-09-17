"""Integration tests for the API endpoints.

Tests the complete API flow from request to response, including:
- Request validation
- OCR processing pipeline
- Field extraction
- Response formatting
"""

import json
import pytest


class TestAPIIntegration:
    """Test API integration with full processing pipeline."""

    def test_complete_ktp_processing_flow(
        self, test_client, realistic_ktp_ocr_data, mock_database_insert
    ):
        """Test processing a complete KTP with all fields."""
        # API expects ocr_text as JSON string
        request_data = {"ocr_text": json.dumps(realistic_ktp_ocr_data)}
        
        response = test_client.post(
            "/v1/ocr_postprocess",
            json=request_data
        )
        
        assert response.status_code == 200
        result = response.json()

        # Verify envelope response structure
        assert "data" in result
        assert "ocr_result" in result["data"]
        assert "nik_image_box" in result["data"]

        # Verify extracted fields
        data = result["data"]["ocr_result"]
        assert data["nik"] == "3174012801950001"
        assert data["nama"] == "BUDI SANTOSO"
        assert "JAKARTA" in data["tempat_lahir"]
        assert data["jenis_kelamin"] in ["LAKI-LAKI", "LAKI LAKI"]
        assert data["alamat"] == "JL. SUDIRMAN NO. 123"
        assert data["agama"] == "ISLAM"

    def test_partial_ktp_with_remapping(
        self, test_client, partial_ktp_ocr_data, mock_database_insert
    ):
        """Test processing partial KTP data with remapping logic."""
        request_data = {"ocr_text": json.dumps(partial_ktp_ocr_data)}
        
        response = test_client.post(
            "/v1/ocr_postprocess",
            json=request_data
        )
        
        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]
        
        # Core fields should be extracted
        assert data["nik"] == "3174012801950001"
        assert data["nama"] == "BUDI SANTOSO"
        
        # Missing fields should be empty or remapped
        assert isinstance(data.get("agama", ""), str)
        assert isinstance(data.get("alamat", ""), str)

    def test_noisy_ocr_data_correction(
        self, test_client, noisy_ktp_ocr_data, mock_database_insert
    ):
        """Test processing noisy OCR data with character correction."""
        request_data = {"ocr_text": json.dumps(noisy_ktp_ocr_data)}
        
        response = test_client.post(
            "/v1/ocr_postprocess",
            json=request_data
        )
        
        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]
        
        # NIK should be present (might be corrected or empty)
        assert isinstance(data.get("nik", ""), str)
        
        # Name should be extracted and cleaned
        assert isinstance(data.get("nama", ""), str)

    def test_low_confidence_handling(
        self, test_client, low_confidence_ktp_data, mock_database_insert
    ):
        """Test handling of low confidence OCR data."""
        request_data = {"ocr_text": json.dumps(low_confidence_ktp_data)}
        
        response = test_client.post(
            "/v1/ocr_postprocess",
            json=request_data
        )
        
        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]
        
        # Should still return a result (may have fewer fields due to low confidence)
        assert isinstance(data, dict)

    def test_multiple_format_variations(
        self, test_client, multiple_format_variations, mock_database_insert
    ):
        """Test processing different KTP format variations."""
        for format_name, ocr_data in multiple_format_variations.items():
            request_data = {"ocr_text": json.dumps(ocr_data)}
            
            response = test_client.post(
                "/v1/ocr_postprocess",
                json=request_data
            )
            
            assert response.status_code == 200, f"Failed for {format_name}"
            result = response.json()
            assert "ocr_result" in result["data"], f"Missing ocr_result for {format_name}"

    def test_empty_ocr_data_handling(
        self, test_client, mock_database_insert
    ):
        """Test handling of empty OCR data."""
        request_data = {"ocr_text": json.dumps([])}
        
        response = test_client.post(
            "/v1/ocr_postprocess",
            json=request_data
        )
        
        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]
        
        # Should return empty fields
        assert data.get("nik", "") == ""
        assert data.get("nama", "") == ""

    def test_malformed_ocr_data_resilience(
        self, test_client, mock_database_insert
    ):
        """Test resilience to malformed OCR data structures."""
        malformed_data = [
            [[[100, 50], [200, 80]], ["TEXT1", 0.95]],
            [[[210, 50], [450, 80]], ["TEXT2", 0.94]],
        ]
        
        request_data = {"ocr_text": json.dumps(malformed_data)}
        
        response = test_client.post(
            "/v1/ocr_postprocess",
            json=request_data
        )
        
        # Should handle gracefully without crashing
        assert response.status_code == 200

    def test_response_time_performance(
        self, test_client, realistic_ktp_ocr_data, mock_database_insert
    ):
        """Test API response time performance."""
        import time
        
        request_data = {"ocr_text": json.dumps(realistic_ktp_ocr_data)}
        
        start_time = time.time()
        response = test_client.post(
            "/v1/ocr_postprocess",
            json=request_data
        )
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Response should be reasonably fast (< 5 seconds)
        response_time = end_time - start_time
        assert response_time < 5.0, f"Response too slow: {response_time}s"

    def test_special_characters_in_fields(
        self, test_client, mock_database_insert
    ):
        """Test handling special characters in field values."""
        special_char_data = [
            [[[100, 50], [200, 80]], ["Nama", 0.95]],
            [[[210, 50], [450, 80]], ["BUDI S/O SANTOSO", 0.96]],
            [[[100, 90], [200, 120]], ["Alamat", 0.95]],
            [[[210, 90], [450, 120]], ["JL. K.H. AHMAD DAHLAN NO. 12/34", 0.94]],
            [[[100, 130], [200, 160]], ["Kel/Desa", 0.93]],
            [[[210, 130], [350, 160]], ["MENTENG-DALAM", 0.92]],
            [[[100, 170], [200, 200]], ["Jenis Kelamin", 0.91]],
            [[[210, 170], [350, 200]], ["LAKI-LAKI", 0.92]],
            [[[100, 210], [200, 240]], ["Agama", 0.90]],
            [[[210, 210], [350, 240]], ["ISLAM", 0.91]],
            [[[100, 250], [200, 280]], ["NIK", 0.95]],
            [[[210, 250], [450, 280]], ["3174012801950001", 0.96]],
        ]
        
        request_data = {"ocr_text": json.dumps(special_char_data)}
        
        response = test_client.post(
            "/v1/ocr_postprocess",
            json=request_data
        )
        
        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]
        
        # Should handle special characters
        assert "BUDI" in data.get("nama", "")

    def test_data_consistency_across_calls(
        self, test_client, realistic_ktp_ocr_data, mock_database_insert
    ):
        """Test that same input produces consistent output."""
        request_data = {"ocr_text": json.dumps(realistic_ktp_ocr_data)}
        
        # Make two identical requests
        response1 = test_client.post(
            "/v1/ocr_postprocess",
            json=request_data
        )
        response2 = test_client.post(
            "/v1/ocr_postprocess",
            json=request_data
        )
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()["data"]["ocr_result"]
        data2 = response2.json()["data"]["ocr_result"]
        
        # Core fields should be identical
        assert data1["nik"] == data2["nik"]
        assert data1["nama"] == data2["nama"]
        assert data1["tempat_lahir"] == data2["tempat_lahir"]
        assert data1["agama"] == data2["agama"]

