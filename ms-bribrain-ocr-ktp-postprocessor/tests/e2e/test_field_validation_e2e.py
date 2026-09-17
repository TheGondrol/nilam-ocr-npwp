"""End-to-end tests for field validation and data quality.

These tests verify that the extracted fields meet expected format
and validation requirements.
"""

import json
import re
import pytest


@pytest.mark.e2e
class TestNIKValidationE2E:
    """E2E tests for NIK field validation."""

    def test_nik_has_16_digits(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Test that extracted NIK has exactly 16 digits."""
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        data = response.json()["data"]["ocr_result"]

        nik = data["nik"]
        assert len(nik) == 16, f"NIK should be 16 digits, got {len(nik)}: {nik}"
        assert nik.isdigit(), f"NIK should contain only digits: {nik}"

    def test_nik_province_code_valid(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Test that NIK starts with valid province code."""
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        data = response.json()["data"]["ocr_result"]

        nik = data["nik"]
        province_code = int(nik[:2])

        # Valid Indonesian province codes are between 11-94
        assert 11 <= province_code <= 94, f"Invalid province code: {province_code}"

    def test_multiple_nik_formats(
        self, test_client, multiple_ktp_samples, mock_database_insert
    ):
        """Test NIK extraction across different KTP samples."""
        for sample_name, ktp_data in multiple_ktp_samples.items():
            request_data = {"ocr_text": json.dumps(ktp_data)}
            response = test_client.post("/v1/ocr_postprocess", json=request_data)

            assert response.status_code == 200, f"Failed for {sample_name}"
            data = response.json()["data"]["ocr_result"]

            nik = data.get("nik", "")
            if nik:  # If NIK is extracted
                assert len(nik) == 16, f"NIK invalid for {sample_name}: {nik}"
                assert nik.isdigit(), f"NIK contains non-digits for {sample_name}: {nik}"


@pytest.mark.e2e
class TestNameValidationE2E:
    """E2E tests for name field validation."""

    def test_name_is_uppercase(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Test that extracted name is in uppercase."""
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        data = response.json()["data"]["ocr_result"]

        nama = data["nama"]
        # Name should be uppercase (allowing spaces and some special chars)
        assert nama == nama.upper() or any(c.isupper() for c in nama)

    def test_name_contains_valid_characters(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Test that name contains only valid characters."""
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        data = response.json()["data"]["ocr_result"]

        nama = data["nama"]
        # Name should contain letters, spaces, periods, commas, and hyphens
        valid_pattern = r'^[A-Z\s\.\,\-\']+$'
        assert re.match(valid_pattern, nama) or nama.replace(" ", "").isalpha()


@pytest.mark.e2e
class TestDateValidationE2E:
    """E2E tests for date field validation."""

    def test_tanggal_lahir_format(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Test that tanggal lahir has valid date format."""
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        data = response.json()["data"]["ocr_result"]

        tanggal = data.get("tanggal_lahir", "")
        if tanggal:
            # Should contain day, month, and year
            # Common formats: DD-MM-YYYY, DD/MM/YYYY, DDMMYYYY
            assert re.search(r'\d+', tanggal), f"Date should contain numbers: {tanggal}"

    def test_tempat_lahir_is_text(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Test that tempat lahir is valid text."""
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        data = response.json()["data"]["ocr_result"]

        tempat = data.get("tempat_lahir", "")
        if tempat:
            # Should be text (city name)
            assert any(c.isalpha() for c in tempat), f"Tempat lahir should contain letters: {tempat}"


@pytest.mark.e2e
class TestGenderValidationE2E:
    """E2E tests for gender field validation."""

    def test_valid_gender_values(
        self, test_client, complete_jakarta_ktp, complete_surabaya_ktp, mock_database_insert
    ):
        """Test that gender has valid values."""
        valid_genders = ["LAKI-LAKI", "LAKI LAKI", "PEREMPUAN", "WANITA"]

        # Test male KTP
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}
        response = test_client.post("/v1/ocr_postprocess", json=request_data)
        assert response.status_code == 200
        data = response.json()["data"]["ocr_result"]
        assert data["jenis_kelamin"] in valid_genders, f"Invalid gender: {data['jenis_kelamin']}"

        # Test female KTP
        request_data = {"ocr_text": json.dumps(complete_surabaya_ktp)}
        response = test_client.post("/v1/ocr_postprocess", json=request_data)
        assert response.status_code == 200
        data = response.json()["data"]["ocr_result"]
        assert data["jenis_kelamin"] in valid_genders, f"Invalid gender: {data['jenis_kelamin']}"


@pytest.mark.e2e
class TestReligionValidationE2E:
    """E2E tests for religion field validation."""

    def test_valid_religion_values(
        self, test_client, multiple_ktp_samples, mock_database_insert
    ):
        """Test that religion has valid values."""
        valid_religions = [
            "ISLAM", "KRISTEN", "KATOLIK", "KATHOLIK", "HINDU", "BUDDHA", "KONGHUCU",
            "KRISTEN PROTESTAN", "KRISTEN KATOLIK"
        ]

        for sample_name, ktp_data in multiple_ktp_samples.items():
            request_data = {"ocr_text": json.dumps(ktp_data)}
            response = test_client.post("/v1/ocr_postprocess", json=request_data)

            assert response.status_code == 200, f"Failed for {sample_name}"
            data = response.json()["data"]["ocr_result"]

            agama = data.get("agama", "")
            if agama:
                assert agama in valid_religions, f"Invalid religion for {sample_name}: {agama}"


@pytest.mark.e2e
class TestMaritalStatusValidationE2E:
    """E2E tests for marital status field validation."""

    def test_valid_marital_status_values(
        self, test_client, multiple_ktp_samples, mock_database_insert
    ):
        """Test that marital status has valid values."""
        valid_statuses = [
            "BELUM KAWIN", "KAWIN", "CERAI HIDUP", "CERAI MATI",
            "JANDA", "DUDA"
        ]

        for sample_name, ktp_data in multiple_ktp_samples.items():
            request_data = {"ocr_text": json.dumps(ktp_data)}
            response = test_client.post("/v1/ocr_postprocess", json=request_data)

            assert response.status_code == 200, f"Failed for {sample_name}"
            data = response.json()["data"]["ocr_result"]

            status = data.get("status_perkawinan", "")
            if status:
                assert status in valid_statuses, f"Invalid marital status for {sample_name}: {status}"


@pytest.mark.e2e
class TestRTRWValidationE2E:
    """E2E tests for RT/RW field validation."""

    def test_rt_rw_format(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Test that RT/RW has valid format."""
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        data = response.json()["data"]["ocr_result"]

        rt = data.get("rt", "")
        rw = data.get("rw", "")

        if rt:
            # RT should be numeric (possibly with leading zeros)
            assert rt.isdigit() or rt.lstrip("0").isdigit() or rt == "0"

        if rw:
            # RW should be numeric (possibly with leading zeros)
            assert rw.isdigit() or rw.lstrip("0").isdigit() or rw == "0"

    def test_rt_rw_values_range(
        self, test_client, multiple_ktp_samples, mock_database_insert
    ):
        """Test that RT/RW values are in reasonable range."""
        for sample_name, ktp_data in multiple_ktp_samples.items():
            request_data = {"ocr_text": json.dumps(ktp_data)}
            response = test_client.post("/v1/ocr_postprocess", json=request_data)

            assert response.status_code == 200, f"Failed for {sample_name}"
            data = response.json()["data"]["ocr_result"]

            rt = data.get("rt", "")
            rw = data.get("rw", "")

            if rt and rt.isdigit():
                rt_num = int(rt)
                assert 0 <= rt_num <= 999, f"RT out of range for {sample_name}: {rt}"

            if rw and rw.isdigit():
                rw_num = int(rw)
                assert 0 <= rw_num <= 999, f"RW out of range for {sample_name}: {rw}"


@pytest.mark.e2e
class TestAddressValidationE2E:
    """E2E tests for address field validation."""

    def test_alamat_not_empty_for_complete_ktp(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Test that alamat is extracted for complete KTP."""
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        data = response.json()["data"]["ocr_result"]

        alamat = data.get("alamat", "")
        assert len(alamat) > 0, "Alamat should not be empty for complete KTP"

    def test_address_components(
        self, test_client, complete_jakarta_ktp, mock_database_insert
    ):
        """Test that address components are present."""
        request_data = {"ocr_text": json.dumps(complete_jakarta_ktp)}

        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        data = response.json()["data"]["ocr_result"]

        # At least some address-related fields should be present
        has_alamat = bool(data.get("alamat", ""))
        has_rt = bool(data.get("rt", ""))
        has_rw = bool(data.get("rw", ""))

        assert has_alamat or has_rt or has_rw, "At least some address info should be present"
