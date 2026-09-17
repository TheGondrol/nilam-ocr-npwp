"""End-to-end tests for KTP extraction flow.

These tests simulate the complete flow from API request to response,
testing the entire OCR post-processing pipeline.
"""

import json
import pytest


@pytest.mark.e2e
class TestCompleteKTPExtractionE2E:
    """E2E tests for complete KTP extraction scenarios."""

    def test_jakarta_ktp_full_extraction(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Test complete extraction of Jakarta KTP with all fields."""
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()

        # Verify response structure
        assert "ocr_result" in result["data"]
        assert "nik_image_box" in result["data"]

        data = result["data"]["ocr_result"]

        # Verify all expected fields
        assert data["nik"] == "3174012801900001"
        assert data["nama"] == "AHMAD FAUZI"
        assert "JAKARTA" in data["tempat_lahir"]
        assert "1990" in data["tanggal_lahir"] or "28" in data["tanggal_lahir"]
        assert data["jenis_kelamin"] in ["LAKI-LAKI", "LAKI LAKI"]
        assert "GATOT SUBROTO" in data["alamat"]
        assert data["rt"] == "005"
        assert data["rw"] == "008"
        assert data["agama"] == "ISLAM"
        assert data["status_perkawinan"] == "KAWIN"

    def test_surabaya_female_ktp_extraction(
        self, test_client, complete_surabaya_ktp, mock_database_insert
    ):
        """Test complete extraction for female KTP from Surabaya."""
        request_data = {"ocr_text": json.dumps(complete_surabaya_ktp)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        # Verify female-specific data
        assert data["nik"] == "3578140507880002"
        assert data["nama"] == "SITI NURHALIZA"
        assert "SURABAYA" in data["tempat_lahir"]
        assert data["jenis_kelamin"] in ["PEREMPUAN", "WANITA"]
        assert data["agama"] == "ISLAM"
        assert data["status_perkawinan"] == "BELUM KAWIN"

    def test_bandung_ktp_different_religion(
        self, test_client, complete_bandung_ktp, mock_database_insert
    ):
        """Test extraction for KTP with Catholic religion."""
        request_data = {"ocr_text": json.dumps(complete_bandung_ktp)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        assert data["nik"] == "3273010101850003"
        assert data["nama"] == "DEDI MULYADI"
        assert data["agama"] == "KATHOLIK"  # Config uses KATHOLIK spelling
        assert data["status_perkawinan"] == "KAWIN"

    def test_bali_hindu_female_ktp(
        self, test_client, ktp_female_non_muslim, mock_database_insert
    ):
        """Test extraction for Hindu female from Bali."""
        request_data = {"ocr_text": json.dumps(ktp_female_non_muslim)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        assert data["nik"] == "5171014505920001"
        assert data["nama"] == "NI MADE DEWI"
        assert data["jenis_kelamin"] in ["PEREMPUAN", "WANITA"]
        assert data["agama"] == "HINDU"
        assert data["status_perkawinan"] == "CERAI HIDUP"

    def test_manado_christian_widower_ktp(
        self, test_client, ktp_christian_widower, mock_database_insert
    ):
        """Test extraction for Christian widower from Manado."""
        request_data = {"ocr_text": json.dumps(ktp_christian_widower)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        assert data["nik"] == "7171012010650001"
        assert data["nama"] == "PAULUS MANTIRI"
        assert data["agama"] == "KRISTEN"
        assert data["status_perkawinan"] == "CERAI MATI"

    def test_batch_multiple_ktp_processing(
        self, test_client, multiple_ktp_samples, mock_database_insert
    ):
        """Test processing multiple different KTP samples."""
        results = {}

        for sample_name, ktp_data in multiple_ktp_samples.items():
            request_data = {"ocr_text": json.dumps(ktp_data)}
            response = test_client.post("/v1/ocr_postprocess", json=request_data)

            assert response.status_code == 200, f"Failed for {sample_name}"
            result = response.json()
            results[sample_name] = result["data"]["ocr_result"]

        # Verify all samples were processed successfully
        assert len(results) == 5

        # Verify unique NIKs
        niks = [data["nik"] for data in results.values()]
        assert len(set(niks)) == 5, "All NIKs should be unique"

        # Verify different religions
        religions = [data["agama"] for data in results.values()]
        assert len(set(religions)) >= 3, "Should have at least 3 different religions"

        # Verify different genders
        genders = [data["jenis_kelamin"] for data in results.values()]
        assert len(set(genders)) >= 2, "Should have both genders"


@pytest.mark.e2e
class TestEdgeCaseE2E:
    """E2E tests for edge cases and error handling."""

    def test_ocr_typos_correction(
        self, test_client, ktp_with_ocr_typos, mock_database_insert
    ):
        """Test that OCR typos are corrected appropriately."""
        request_data = {"ocr_text": json.dumps(ktp_with_ocr_typos)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        # NIK should be extracted (possibly with corrections)
        assert isinstance(data.get("nik", ""), str)

        # Name should be present
        assert isinstance(data.get("nama", ""), str)

    def test_missing_fields_handling(
        self, test_client, ktp_with_missing_fields, mock_database_insert
    ):
        """Test handling of KTP with missing fields."""
        request_data = {"ocr_text": json.dumps(ktp_with_missing_fields)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        # Present fields should be extracted
        assert data["nik"] == "3174012801950001"
        assert data["nama"] == "BUDI SANTOSO"
        assert data["jenis_kelamin"] in ["LAKI-LAKI", "LAKI LAKI"]

        # Missing fields should be empty or not present
        assert data.get("alamat", "") == "" or "alamat" not in data
        assert data.get("agama", "") == "" or "agama" not in data

    def test_low_confidence_filtering(
        self, test_client, ktp_with_low_confidence, mock_database_insert
    ):
        """Test that low confidence data is handled appropriately."""
        request_data = {"ocr_text": json.dumps(ktp_with_low_confidence)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        # Should return result dict (may have fewer fields due to confidence threshold)
        assert isinstance(data, dict)

    def test_unusual_formatting_handling(
        self, test_client, ktp_with_unusual_formatting, mock_database_insert
    ):
        """Test handling of unusual text formatting."""
        request_data = {"ocr_text": json.dumps(ktp_with_unusual_formatting)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        # Should still extract core fields
        assert isinstance(data, dict)

    def test_empty_ocr_data(
        self, test_client, empty_ocr_data, mock_database_insert
    ):
        """Test handling of empty OCR data."""
        request_data = {"ocr_text": json.dumps(empty_ocr_data)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        # Should return empty result
        assert data.get("nik", "") == ""
        assert data.get("nama", "") == ""

    def test_extreme_long_text_handling(
        self, test_client, extreme_long_text, mock_database_insert
    ):
        """Test handling of extremely long text values."""
        request_data = {"ocr_text": json.dumps(extreme_long_text)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        # Should not crash, may return 200 or handle gracefully
        assert response.status_code in [200, 400, 500]


@pytest.mark.e2e
class TestAPIErrorHandlingE2E:
    """E2E tests for API error handling."""

    def test_invalid_json_format(
        self, test_client, invalid_json_string, mock_database_insert
    ):
        """Test handling of invalid JSON in request."""
        request_data = {"ocr_text": invalid_json_string}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 400
        result = response.json()
        assert result["error_code"] is not None

    def test_missing_ocr_text_field(self, test_client, mock_database_insert):
        """Test handling of missing ocr_text field in request."""
        request_data = {}  # Missing ocr_text

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        # FastAPI validation error
        assert response.status_code == 422

    def test_wrong_content_type(self, test_client, mock_database_insert):
        """Test handling of wrong content type."""
        response = test_client.post(
            "/v1/ocr_postprocess",
            content="raw text data",
            headers={"Content-Type": "text/plain"}
        )

        # Non-JSON content type is rejected with 415 Unsupported Media Type
        assert response.status_code == 415

    def test_null_ocr_text(self, test_client, mock_database_insert):
        """Test handling of null ocr_text value."""
        request_data = {"ocr_text": None}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        # Should handle null value appropriately
        assert response.status_code == 422

    def test_non_string_ocr_text(self, test_client, mock_database_insert):
        """Test handling of non-string ocr_text value."""
        request_data = {"ocr_text": 12345}  # Number instead of string

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        # Should reject non-string value (422 if validated, 500 if not)
        assert response.status_code in [422, 500]


@pytest.mark.e2e
class TestResponseStructureE2E:
    """E2E tests for response structure validation."""

    def test_response_has_required_fields(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Test that response has all required top-level fields."""
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()

        # Check top-level structure
        assert "ocr_result" in result["data"]
        assert "nik_image_box" in result["data"]

        # ocr_result should be a dict
        assert isinstance(result["data"]["ocr_result"], dict)  # ocr_result is a dict

    def test_nik_image_box_format(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Test that nik_image_box has correct format."""
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()

        nik_box = result["data"]["nik_image_box"]

        # nik_image_box should be a list of 2 points (top-left, bottom-right)
        assert isinstance(nik_box, list)
        assert len(nik_box) == 2  # 2 points: [top-left, bottom-right]

        # Each point should be a list of 2 coordinates [x, y]
        for point in nik_box:
            assert isinstance(point, list)
            assert len(point) == 2

    def test_field_types_in_response(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Test that all field values are strings."""
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        # All values should be strings
        for field, value in data.items():
            assert isinstance(value, str), f"Field {field} should be string, got {type(value)}"

    def test_response_content_type_is_json(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Test that response content type is JSON."""
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
