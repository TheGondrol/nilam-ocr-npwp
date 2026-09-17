"""End-to-end tests for real-world scenarios.

These tests simulate various real-world use cases and edge scenarios
that might be encountered in production.
"""

import json
import pytest


@pytest.mark.e2e
class TestRealWorldScenariosE2E:
    """E2E tests simulating real-world scenarios."""

    def test_old_worn_ktp_scan(
        self, test_client, mock_database_insert
    ):
        """Test processing of old/worn KTP with degraded quality."""
        # Simulate old KTP with lower confidence and some noise
        old_ktp_data = [
            [[[100, 50], [300, 80]], ["PROVINSI DKI JAKARTA", 0.75]],
            [[[100, 140], [180, 170]], ["NIK", 0.82]],
            [[[190, 140], [450, 170]], ["3174015503750001", 0.78]],
            [[[100, 180], [180, 210]], ["Nama", 0.80]],
            [[[190, 180], [450, 210]], ["SUDIRMAN", 0.76]],
            [[[100, 220], [200, 250]], ["Tempat/Tgl Lahir", 0.72]],
            [[[210, 220], [450, 250]], ["JAKARTA, 15-03-1975", 0.70]],
            [[[100, 260], [200, 290]], ["Jenis Kelamin", 0.78]],
            [[[210, 260], [350, 290]], ["LAKI-LAKI", 0.75]],
            [[[100, 300], [180, 330]], ["Alamat", 0.77]],
            [[[190, 300], [500, 330]], ["JL. MERDEKA NO. 17", 0.73]],
            [[[100, 460], [180, 490]], ["Agama", 0.79]],
            [[[190, 460], [280, 490]], ["ISLAM", 0.81]],
        ]

        request_data = {"ocr_text": json.dumps(old_ktp_data)}
        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        # Should still extract core fields despite lower quality
        assert isinstance(data, dict)

    def test_mobile_camera_capture(
        self, test_client, mock_database_insert
    ):
        """Test processing of KTP captured with mobile camera (slight rotation/blur)."""
        # Simulate mobile capture with slight positioning issues
        mobile_capture_data = [
            [[[95, 48], [305, 82]], ["PROVINSI JAWA BARAT", 0.88]],
            [[[98, 138], [182, 172]], ["NIK", 0.92]],
            [[[195, 140], [452, 172]], ["3273051201900001", 0.89]],
            [[[102, 182], [185, 215]], ["Nama", 0.90]],
            [[[198, 180], [448, 212]], ["RUDI HARTONO", 0.87]],
            [[[96, 218], [202, 252]], ["Tempat/Tgl Lahir", 0.85]],
            [[[208, 222], [455, 255]], ["BANDUNG, 12-01-1990", 0.84]],
            [[[103, 262], [205, 295]], ["Jenis Kelamin", 0.88]],
            [[[215, 265], [355, 295]], ["LAKI-LAKI", 0.86]],
            [[[105, 302], [188, 335]], ["Alamat", 0.87]],
            [[[200, 305], [510, 340]], ["JL. BRAGA NO. 45", 0.83]],
            [[[100, 340], [180, 370]], ["RT/RW", 0.85]],
            [[[190, 340], [300, 370]], ["003/007", 0.84]],
            [[[99, 458], [182, 492]], ["Agama", 0.89]],
            [[[195, 462], [285, 495]], ["ISLAM", 0.88]],
            [[[100, 500], [220, 530]], ["Status Perkawinan", 0.86]],
            [[[230, 500], [350, 530]], ["BELUM KAWIN", 0.84]],
        ]

        request_data = {"ocr_text": json.dumps(mobile_capture_data)}
        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        # Verify key fields are extracted
        assert data["nik"] == "3273051201900001"
        assert data["nama"] == "RUDI HARTONO"

    def test_photocopy_ktp(
        self, test_client, mock_database_insert
    ):
        """Test processing of photocopied KTP (lower contrast)."""
        photocopy_data = [
            [[[100, 140], [180, 170]], ["NIK", 0.85]],
            [[[190, 140], [450, 170]], ["3174010101800001", 0.82]],
            [[[100, 180], [180, 210]], ["Nama", 0.83]],
            [[[190, 180], [450, 210]], ["AHMAD BASUKI", 0.80]],
            [[[100, 220], [200, 250]], ["Tempat/Tgl Lahir", 0.78]],
            [[[210, 220], [450, 250]], ["JAKARTA 01-01-1980", 0.76]],
            [[[100, 260], [200, 290]], ["Jenis Kelamin", 0.80]],
            [[[210, 260], [350, 290]], ["LAKI-LAKI", 0.78]],
            [[[100, 300], [180, 330]], ["Alamat", 0.79]],
            [[[190, 300], [500, 330]], ["JL THAMRIN", 0.75]],
            [[[100, 460], [180, 490]], ["Agama", 0.82]],
            [[[190, 460], [280, 490]], ["ISLAM", 0.80]],
        ]

        request_data = {"ocr_text": json.dumps(photocopy_data)}
        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()

        # Should process photocopy
        assert "ocr_result" in result["data"]

    def test_laminated_ktp_with_reflection(
        self, test_client, mock_database_insert
    ):
        """Test processing of laminated KTP with reflection artifacts."""
        laminated_data = [
            [[[100, 50], [300, 80]], ["PROVINSI DKI JAKARTA", 0.92]],
            [[[100, 140], [180, 170]], ["NIK", 0.94]],
            [[[190, 140], [450, 170]], ["3174020203850001", 0.91]],
            [[[100, 180], [180, 210]], ["Nama", 0.93]],
            [[[190, 180], [450, 210]], ["MARIA SUSANTI", 0.90]],
            [[[100, 220], [200, 250]], ["Tempat/Tgl Lahir", 0.88]],
            [[[210, 220], [450, 250]], ["JAKARTA, 02-03-1985", 0.86]],
            [[[100, 260], [200, 290]], ["Jenis Kelamin", 0.91]],
            [[[210, 260], [350, 290]], ["PEREMPUAN", 0.89]],
            [[[100, 300], [180, 330]], ["Alamat", 0.90]],
            [[[190, 300], [500, 330]], ["JL. KEBON JERUK NO. 10", 0.87]],
            [[[100, 340], [180, 370]], ["RT/RW", 0.88]],
            [[[190, 340], [300, 370]], ["004/006", 0.86]],
            [[[100, 380], [180, 410]], ["Kel/Desa", 0.87]],
            [[[190, 380], [350, 410]], ["KEBON JERUK", 0.85]],
            [[[100, 420], [180, 450]], ["Kecamatan", 0.88]],
            [[[190, 420], [350, 450]], ["KEBON JERUK", 0.86]],
            [[[100, 460], [180, 490]], ["Agama", 0.91]],
            [[[190, 460], [280, 490]], ["KATOLIK", 0.89]],
            [[[100, 500], [220, 530]], ["Status Perkawinan", 0.87]],
            [[[230, 500], [350, 530]], ["KAWIN", 0.85]],
        ]

        request_data = {"ocr_text": json.dumps(laminated_data)}
        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        assert data["nik"] == "3174020203850001"
        assert data["nama"] == "MARIA SUSANTI"
        assert data["jenis_kelamin"] in ["PEREMPUAN", "WANITA"]
        assert data["agama"] in ["KATOLIK", "KATHOLIK"]  # Both spellings are valid

    def test_new_ektp_format(
        self, test_client, mock_database_insert
    ):
        """Test processing of new e-KTP format."""
        ektp_data = [
            [[[50, 30], [250, 60]], ["REPUBLIK INDONESIA", 0.96]],
            [[[100, 70], [300, 100]], ["PROVINSI JAWA TENGAH", 0.95]],
            [[[100, 110], [300, 140]], ["KOTA SEMARANG", 0.94]],
            [[[100, 160], [180, 190]], ["NIK", 0.97]],
            [[[190, 160], [450, 190]], ["3374012506950001", 0.96]],
            [[[100, 200], [180, 230]], ["Nama", 0.96]],
            [[[190, 200], [450, 230]], ["TEGUH PRASETYO", 0.95]],
            [[[100, 240], [200, 270]], ["Tempat/Tgl Lahir", 0.94]],
            [[[210, 240], [450, 270]], ["SEMARANG, 25-06-1995", 0.93]],
            [[[100, 280], [200, 310]], ["Jenis Kelamin", 0.95]],
            [[[210, 280], [350, 310]], ["LAKI-LAKI", 0.94]],
            [[[360, 280], [450, 310]], ["Gol. Darah", 0.92]],
            [[[460, 280], [500, 310]], ["AB", 0.91]],
            [[[100, 320], [180, 350]], ["Alamat", 0.95]],
            [[[190, 320], [500, 350]], ["JL. PEMUDA NO. 150", 0.93]],
            [[[100, 360], [180, 390]], ["RT/RW", 0.94]],
            [[[190, 360], [300, 390]], ["002/004", 0.93]],
            [[[100, 400], [180, 430]], ["Kel/Desa", 0.93]],
            [[[190, 400], [350, 430]], ["SEKAYU", 0.92]],
            [[[100, 440], [180, 470]], ["Kecamatan", 0.94]],
            [[[190, 440], [350, 470]], ["SEMARANG TENGAH", 0.93]],
            [[[100, 480], [180, 510]], ["Agama", 0.95]],
            [[[190, 480], [280, 510]], ["ISLAM", 0.94]],
            [[[100, 520], [220, 550]], ["Status Perkawinan", 0.93]],
            [[[230, 520], [380, 550]], ["BELUM KAWIN", 0.92]],
            [[[100, 560], [180, 590]], ["Pekerjaan", 0.92]],
            [[[190, 560], [380, 590]], ["MAHASISWA", 0.91]],
            [[[100, 600], [220, 630]], ["Kewarganegaraan", 0.93]],
            [[[230, 600], [300, 630]], ["WNI", 0.94]],
            [[[100, 640], [200, 670]], ["Berlaku Hingga", 0.92]],
            [[[210, 640], [350, 670]], ["SEUMUR HIDUP", 0.93]],
        ]

        request_data = {"ocr_text": json.dumps(ektp_data)}
        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        # Verify complete extraction
        assert data["nik"] == "3374012506950001"
        assert data["nama"] == "TEGUH PRASETYO"
        assert "SEMARANG" in data["tempat_lahir"]
        assert data["jenis_kelamin"] in ["LAKI-LAKI", "LAKI LAKI"]
        assert data["agama"] == "ISLAM"
        assert data["status_perkawinan"] == "BELUM KAWIN"


@pytest.mark.e2e
class TestSpecialCasesE2E:
    """E2E tests for special edge cases."""

    def test_name_with_title(
        self, test_client, mock_database_insert
    ):
        """Test processing name with academic/religious titles."""
        data_with_title = [
            [[[100, 140], [180, 170]], ["NIK", 0.96]],
            [[[190, 140], [450, 170]], ["3174015001800001", 0.95]],
            [[[100, 180], [180, 210]], ["Nama", 0.95]],
            [[[190, 180], [500, 210]], ["DR. H. ABDUL RAHMAN, M.SC", 0.93]],
            [[[100, 220], [200, 250]], ["Tempat/Tgl Lahir", 0.93]],
            [[[210, 220], [450, 250]], ["JAKARTA, 01-05-1980", 0.92]],
            [[[100, 260], [200, 290]], ["Jenis Kelamin", 0.94]],
            [[[210, 260], [350, 290]], ["LAKI-LAKI", 0.93]],
            [[[100, 460], [180, 490]], ["Agama", 0.95]],
            [[[190, 460], [280, 490]], ["ISLAM", 0.94]],
        ]

        request_data = {"ocr_text": json.dumps(data_with_title)}
        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        # Name should include title
        assert "ABDUL RAHMAN" in data["nama"] or "DR" in data["nama"]

    def test_name_with_alias(
        self, test_client, mock_database_insert
    ):
        """Test processing name with alias notation."""
        data_with_alias = [
            [[[100, 140], [180, 170]], ["NIK", 0.96]],
            [[[190, 140], [450, 170]], ["3174016506850001", 0.95]],
            [[[100, 180], [180, 210]], ["Nama", 0.95]],
            [[[190, 180], [500, 210]], ["SITI AMINAH ALIAS AMINAH", 0.92]],
            [[[100, 220], [200, 250]], ["Tempat/Tgl Lahir", 0.93]],
            [[[210, 220], [450, 250]], ["JAKARTA, 25-06-1985", 0.91]],
            [[[100, 260], [200, 290]], ["Jenis Kelamin", 0.94]],
            [[[210, 260], [350, 290]], ["PEREMPUAN", 0.93]],
            [[[100, 460], [180, 490]], ["Agama", 0.95]],
            [[[190, 460], [280, 490]], ["ISLAM", 0.94]],
        ]

        request_data = {"ocr_text": json.dumps(data_with_alias)}
        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        # Should extract the name (with or without alias)
        assert "AMINAH" in data["nama"]

    def test_address_with_complex_format(
        self, test_client, mock_database_insert
    ):
        """Test processing complex address format."""
        complex_address_data = [
            [[[100, 140], [180, 170]], ["NIK", 0.96]],
            [[[190, 140], [450, 170]], ["3174010101900001", 0.95]],
            [[[100, 180], [180, 210]], ["Nama", 0.95]],
            [[[190, 180], [450, 210]], ["BUDI HARTONO", 0.94]],
            [[[100, 220], [200, 250]], ["Tempat/Tgl Lahir", 0.93]],
            [[[210, 220], [450, 250]], ["JAKARTA, 01-01-1990", 0.92]],
            [[[100, 260], [200, 290]], ["Jenis Kelamin", 0.94]],
            [[[210, 260], [350, 290]], ["LAKI-LAKI", 0.93]],
            [[[100, 300], [180, 330]], ["Alamat", 0.95]],
            [[[190, 300], [550, 330]], ["JL. K.H. AHMAD DAHLAN GG. MAWAR NO. 5/7-A", 0.90]],
            [[[100, 340], [180, 370]], ["RT/RW", 0.93]],
            [[[190, 340], [300, 370]], ["010/015", 0.92]],
            [[[100, 380], [180, 410]], ["Kel/Desa", 0.92]],
            [[[190, 380], [400, 410]], ["TANAH ABANG UTARA", 0.91]],
            [[[100, 420], [180, 450]], ["Kecamatan", 0.93]],
            [[[190, 420], [350, 450]], ["TANAH ABANG", 0.92]],
            [[[100, 460], [180, 490]], ["Agama", 0.95]],
            [[[190, 460], [280, 490]], ["ISLAM", 0.94]],
        ]

        request_data = {"ocr_text": json.dumps(complex_address_data)}
        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        # Should extract address
        assert "AHMAD DAHLAN" in data.get("alamat", "") or len(data.get("alamat", "")) > 0
        assert data["rt"] == "010"
        assert data["rw"] == "015"

    def test_very_young_person(
        self, test_client, mock_database_insert
    ):
        """Test processing KTP of young person (just turned 17)."""
        young_person_data = [
            [[[100, 50], [300, 80]], ["PROVINSI BANTEN", 0.95]],
            [[[100, 140], [180, 170]], ["NIK", 0.96]],
            [[[190, 140], [450, 170]], ["3671010101080001", 0.95]],  # Born 2008
            [[[100, 180], [180, 210]], ["Nama", 0.95]],
            [[[190, 180], [450, 210]], ["ANDI PRATAMA", 0.94]],
            [[[100, 220], [200, 250]], ["Tempat/Tgl Lahir", 0.93]],
            [[[210, 220], [450, 250]], ["TANGERANG, 01-01-2008", 0.92]],
            [[[100, 260], [200, 290]], ["Jenis Kelamin", 0.94]],
            [[[210, 260], [350, 290]], ["LAKI-LAKI", 0.93]],
            [[[100, 300], [180, 330]], ["Alamat", 0.94]],
            [[[190, 300], [500, 330]], ["JL. RAYA SERPONG NO. 100", 0.92]],
            [[[100, 460], [180, 490]], ["Agama", 0.95]],
            [[[190, 460], [280, 490]], ["ISLAM", 0.94]],
            [[[100, 500], [220, 530]], ["Status Perkawinan", 0.93]],
            [[[230, 500], [380, 530]], ["BELUM KAWIN", 0.92]],
            [[[100, 540], [180, 570]], ["Pekerjaan", 0.92]],
            [[[190, 540], [380, 570]], ["PELAJAR", 0.91]],
        ]

        request_data = {"ocr_text": json.dumps(young_person_data)}
        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        assert data["nik"] == "3671010101080001"
        assert "2008" in data["tanggal_lahir"] or "01" in data["tanggal_lahir"]
        assert data["status_perkawinan"] == "BELUM KAWIN"

    def test_elderly_person(
        self, test_client, mock_database_insert
    ):
        """Test processing KTP of elderly person."""
        elderly_data = [
            [[[100, 50], [300, 80]], ["PROVINSI JAWA TIMUR", 0.94]],
            [[[100, 140], [180, 170]], ["NIK", 0.95]],
            [[[190, 140], [450, 170]], ["3578011507450001", 0.93]],  # Born 1945
            [[[100, 180], [180, 210]], ["Nama", 0.94]],
            [[[190, 180], [450, 210]], ["SLAMET WIDODO", 0.92]],
            [[[100, 220], [200, 250]], ["Tempat/Tgl Lahir", 0.92]],
            [[[210, 220], [450, 250]], ["SURABAYA, 15-07-1945", 0.90]],
            [[[100, 260], [200, 290]], ["Jenis Kelamin", 0.93]],
            [[[210, 260], [350, 290]], ["LAKI-LAKI", 0.92]],
            [[[100, 300], [180, 330]], ["Alamat", 0.93]],
            [[[190, 300], [500, 330]], ["JL. VETERAN NO. 88", 0.91]],
            [[[100, 460], [180, 490]], ["Agama", 0.94]],
            [[[190, 460], [280, 490]], ["ISLAM", 0.93]],
            [[[100, 500], [220, 530]], ["Status Perkawinan", 0.92]],
            [[[230, 500], [350, 530]], ["CERAI MATI", 0.90]],
            [[[100, 540], [180, 570]], ["Pekerjaan", 0.91]],
            [[[190, 540], [380, 570]], ["PENSIUNAN", 0.89]],
        ]

        request_data = {"ocr_text": json.dumps(elderly_data)}
        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        assert data["nik"] == "3578011507450001"
        assert "1945" in data["tanggal_lahir"] or "15" in data["tanggal_lahir"]
        assert data["status_perkawinan"] == "CERAI MATI"


@pytest.mark.e2e
class TestRegionalVariationsE2E:
    """E2E tests for different regional KTP variations."""

    def test_papua_ktp(
        self, test_client, mock_database_insert
    ):
        """Test processing KTP from Papua region."""
        papua_data = [
            [[[100, 50], [300, 80]], ["PROVINSI PAPUA", 0.94]],
            [[[100, 90], [300, 120]], ["KOTA JAYAPURA", 0.93]],
            [[[100, 140], [180, 170]], ["NIK", 0.96]],
            [[[190, 140], [450, 170]], ["9171010101850001", 0.94]],
            [[[100, 180], [180, 210]], ["Nama", 0.95]],
            [[[190, 180], [450, 210]], ["YOHANES WAMBRAUW", 0.93]],
            [[[100, 220], [200, 250]], ["Tempat/Tgl Lahir", 0.93]],
            [[[210, 220], [450, 250]], ["JAYAPURA, 01-01-1985", 0.91]],
            [[[100, 260], [200, 290]], ["Jenis Kelamin", 0.94]],
            [[[210, 260], [350, 290]], ["LAKI-LAKI", 0.93]],
            [[[100, 300], [180, 330]], ["Alamat", 0.94]],
            [[[190, 300], [500, 330]], ["JL. PERCETAKAN NEGARA NO. 5", 0.91]],
            [[[100, 460], [180, 490]], ["Agama", 0.95]],
            [[[190, 460], [300, 490]], ["KRISTEN", 0.94]],
            [[[100, 500], [220, 530]], ["Status Perkawinan", 0.92]],
            [[[230, 500], [350, 530]], ["KAWIN", 0.91]],
        ]

        request_data = {"ocr_text": json.dumps(papua_data)}
        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        assert data["nik"].startswith("91")  # Papua province code
        assert data["nama"] == "YOHANES WAMBRAUW"
        assert data["agama"] == "KRISTEN"

    def test_aceh_ktp(
        self, test_client, mock_database_insert
    ):
        """Test processing KTP from Aceh region."""
        aceh_data = [
            [[[100, 50], [300, 80]], ["PROVINSI ACEH", 0.95]],
            [[[100, 90], [300, 120]], ["KOTA BANDA ACEH", 0.94]],
            [[[100, 140], [180, 170]], ["NIK", 0.97]],
            [[[190, 140], [450, 170]], ["1171011508900001", 0.95]],
            [[[100, 180], [180, 210]], ["Nama", 0.96]],
            [[[190, 180], [450, 210]], ["TEUKU UMAR", 0.94]],
            [[[100, 220], [200, 250]], ["Tempat/Tgl Lahir", 0.94]],
            [[[210, 220], [450, 250]], ["BANDA ACEH, 15-08-1990", 0.92]],
            [[[100, 260], [200, 290]], ["Jenis Kelamin", 0.95]],
            [[[210, 260], [350, 290]], ["LAKI-LAKI", 0.94]],
            [[[100, 300], [180, 330]], ["Alamat", 0.95]],
            [[[190, 300], [500, 330]], ["JL. CUT NYAK DHIEN NO. 1", 0.92]],
            [[[100, 460], [180, 490]], ["Agama", 0.96]],
            [[[190, 460], [280, 490]], ["ISLAM", 0.95]],
            [[[100, 500], [220, 530]], ["Status Perkawinan", 0.93]],
            [[[230, 500], [350, 530]], ["KAWIN", 0.92]],
        ]

        request_data = {"ocr_text": json.dumps(aceh_data)}
        response = test_client.post("/v1/ocr_postprocess", json=request_data)

        assert response.status_code == 200
        result = response.json()
        data = result["data"]["ocr_result"]

        assert data["nik"].startswith("11")  # Aceh province code
        assert data["nama"] == "TEUKU UMAR"
        assert data["agama"] == "ISLAM"
