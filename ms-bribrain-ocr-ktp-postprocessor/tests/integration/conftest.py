"""Fixtures for integration tests."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.main import app


@pytest.fixture
def test_client():
    """Synchronous test client for API testing with API key."""
    client = TestClient(app)
    client.headers["X-API-Key"] = "test"
    return client


@pytest.fixture
def realistic_ktp_ocr_data():
    """Realistic KTP OCR data with all fields - sufficient data points."""
    return [
        [[[100, 50], [300, 80]], ["PROVINSI DKI JAKARTA", 0.95]],
        [[[100, 90], [300, 120]], ["KOTA JAKARTA SELATAN", 0.94]],
        [[[100, 130], [200, 160]], ["NIK", 0.96]],
        [[[210, 130], [450, 160]], ["3174012801950001", 0.97]],
        [[[100, 170], [200, 200]], ["Nama", 0.95]],
        [[[210, 170], [450, 200]], ["BUDI SANTOSO", 0.96]],
        [[[100, 210], [200, 240]], ["Tempat/Tgl Lahir", 0.94]],
        [[[210, 210], [450, 240]], ["JAKARTA, 28-01-1995", 0.95]],
        [[[100, 250], [200, 280]], ["Jenis Kelamin", 0.93]],
        [[[210, 250], [350, 280]], ["LAKI-LAKI", 0.94]],
        [[[350, 250], [450, 280]], ["Gol. Darah", 0.92]],
        [[[460, 250], [500, 280]], ["O", 0.91]],
        [[[100, 290], [200, 320]], ["Alamat", 0.95]],
        [[[210, 290], [450, 320]], ["JL. SUDIRMAN NO. 123", 0.96]],
        [[[100, 330], [200, 360]], ["RT/RW", 0.94]],
        [[[210, 330], [350, 360]], ["001/002", 0.95]],
        [[[100, 370], [200, 400]], ["Kel/Desa", 0.93]],
        [[[210, 370], [350, 400]], ["KEBAYORAN BARU", 0.94]],
        [[[100, 410], [200, 440]], ["Kecamatan", 0.94]],
        [[[210, 410], [350, 440]], ["KEBAYORAN BARU", 0.95]],
        [[[100, 450], [200, 480]], ["Agama", 0.95]],
        [[[210, 450], [350, 480]], ["ISLAM", 0.96]],
        [[[100, 490], [200, 520]], ["Status Perkawinan", 0.94]],
        [[[210, 490], [350, 520]], ["BELUM KAWIN", 0.93]],
        [[[100, 530], [200, 560]], ["Pekerjaan", 0.93]],
        [[[210, 530], [350, 560]], ["KARYAWAN SWASTA", 0.92]],
        [[[100, 570], [200, 600]], ["Kewarganegaraan", 0.94]],
        [[[210, 570], [350, 600]], ["WNI", 0.95]],
        [[[100, 610], [200, 640]], ["Berlaku Hingga", 0.93]],
        [[[210, 610], [350, 640]], ["SEUMUR HIDUP", 0.94]],
    ]


@pytest.fixture
def partial_ktp_ocr_data():
    """KTP OCR data with some fields - needs enough data for pipeline."""
    return [
        [[[100, 50], [200, 80]], ["NIK", 0.96]],
        [[[210, 50], [450, 80]], ["3174012801950001", 0.97]],
        [[[100, 90], [200, 120]], ["Nama", 0.95]],
        [[[210, 90], [450, 120]], ["BUDI SANTOSO", 0.96]],
        [[[100, 130], [200, 160]], ["Tempat/Tgl Lahir", 0.94]],
        [[[210, 130], [450, 160]], ["JAKARTA 28011995", 0.85]],
        [[[100, 170], [200, 200]], ["Jenis Kelamin", 0.93]],
        [[[210, 170], [350, 200]], ["LAKI-LAKI", 0.94]],
        [[[100, 210], [200, 240]], ["Alamat", 0.92]],
        [[[210, 210], [450, 240]], ["JL. SUDIRMAN NO. 100", 0.91]],
        [[[100, 250], [200, 280]], ["Agama", 0.90]],
        [[[210, 250], [350, 280]], ["ISLAM", 0.91]],
    ]


@pytest.fixture
def noisy_ktp_ocr_data():
    """KTP OCR data with noise and OCR errors - sufficient data."""
    return [
        [[[100, 50], [200, 80]], ["NlK", 0.89]],
        [[[210, 50], [450, 80]], ["3l74Ol28Ol95OOOl", 0.87]],
        [[[100, 90], [200, 120]], ["Nama:", 0.92]],
        [[[210, 90], [450, 120]], ["BUDl SANT0S0", 0.88]],
        [[[100, 130], [200, 160]], ["Tempat/Tgl Lahir", 0.85]],
        [[[210, 130], [450, 160]], ["JAKARTA,28-01-1995", 0.82]],
        [[[100, 170], [200, 200]], ["Jenis Kelamin", 0.90]],
        [[[210, 170], [350, 200]], ["LAKI LAKl", 0.85]],
        [[[100, 210], [200, 240]], ["Alamat", 0.88]],
        [[[210, 210], [450, 240]], ["JL SUDIRMAN", 0.86]],
        [[[100, 250], [200, 280]], ["Agama", 0.91]],
        [[[210, 250], [350, 280]], ["ISLAM", 0.86]],
    ]


@pytest.fixture
def low_confidence_ktp_data():
    """KTP OCR data with low confidence scores - sufficient data."""
    return [
        [[[100, 50], [200, 80]], ["NIK", 0.65]],
        [[[210, 50], [450, 80]], ["3174012801950001", 0.68]],
        [[[100, 90], [200, 120]], ["Nama", 0.60]],
        [[[210, 90], [450, 120]], ["BUDI SANTOSO", 0.62]],
        [[[100, 130], [200, 160]], ["Jenis Kelamin", 0.58]],
        [[[210, 130], [350, 160]], ["LAKI-LAKI", 0.59]],
        [[[100, 170], [200, 200]], ["Alamat", 0.55]],
        [[[210, 170], [450, 200]], ["JL SUDIRMAN", 0.56]],
        [[[100, 210], [200, 240]], ["Agama", 0.64]],
        [[[210, 210], [350, 240]], ["ISLAM", 0.66]],
        [[[100, 250], [200, 280]], ["Pekerjaan", 0.50]],
        [[[210, 250], [350, 280]], ["WIRASWASTA", 0.52]],
    ]


@pytest.fixture
def multiple_format_variations():
    """Different format variations with sufficient data."""
    return {
        "format1": [
            [[[100, 50], [200, 80]], ["NIK", 0.95]],
            [[[210, 50], [450, 80]], ["3174012801950001", 0.96]],
            [[[100, 90], [200, 120]], ["Nama", 0.94]],
            [[[210, 90], [450, 120]], ["BUDI SANTOSO", 0.95]],
            [[[100, 130], [200, 160]], ["Tempat/Tgl Lahir", 0.94]],
            [[[210, 130], [450, 160]], ["JAKARTA, 28-01-1995", 0.95]],
            [[[100, 170], [200, 200]], ["Jenis Kelamin", 0.93]],
            [[[210, 170], [350, 200]], ["LAKI-LAKI", 0.94]],
            [[[100, 210], [200, 240]], ["Alamat", 0.92]],
            [[[210, 210], [450, 240]], ["JL SUDIRMAN", 0.93]],
            [[[100, 250], [200, 280]], ["Agama", 0.91]],
            [[[210, 250], [350, 280]], ["ISLAM", 0.92]],
        ],
        "format2": [
            [[[100, 50], [450, 80]], ["NIK: 3174012801950001", 0.96]],
            [[[100, 90], [450, 120]], ["Nama: BUDI SANTOSO", 0.95]],
            [[[100, 130], [450, 160]], ["Tempat/Tgl Lahir: JAKARTA 28011995", 0.94]],
            [[[100, 170], [450, 200]], ["Jenis Kelamin: LAKI-LAKI", 0.93]],
            [[[100, 210], [450, 240]], ["Alamat: JL SUDIRMAN", 0.92]],
            [[[100, 250], [450, 280]], ["Agama: ISLAM", 0.91]],
            [[[100, 290], [450, 320]], ["Pekerjaan: KARYAWAN", 0.90]],
            [[[100, 330], [450, 360]], ["Status Perkawinan: KAWIN", 0.89]],
            [[[100, 370], [450, 400]], ["Kewarganegaraan: WNI", 0.88]],
            [[[100, 410], [450, 440]], ["RT/RW: 001/002", 0.87]],
            [[[100, 450], [450, 480]], ["Kel/Desa: MENTENG", 0.86]],
            [[[100, 490], [450, 520]], ["Kecamatan: MENTENG", 0.85]],
        ],
        "format3": [
            [[[100, 50], [200, 80]], ["NIK", 0.95]],
            [[[100, 90], [450, 120]], ["3174012801950001", 0.96]],
            [[[100, 130], [200, 160]], ["Nama", 0.94]],
            [[[100, 170], [450, 200]], ["BUDI SANTOSO", 0.95]],
            [[[100, 210], [200, 240]], ["Tempat/Tgl Lahir", 0.93]],
            [[[100, 250], [450, 280]], ["JAKARTA 28 01 1995", 0.94]],
            [[[100, 290], [200, 320]], ["Jenis Kelamin", 0.92]],
            [[[100, 330], [450, 360]], ["LAKI-LAKI", 0.93]],
            [[[100, 370], [200, 400]], ["Alamat", 0.91]],
            [[[100, 410], [450, 440]], ["JL SUDIRMAN NO 1", 0.92]],
            [[[100, 450], [200, 480]], ["Agama", 0.90]],
            [[[100, 490], [450, 520]], ["ISLAM", 0.91]],
        ],
    }


@pytest.fixture
def mock_database_insert():
    """Mock database insert for integration tests."""
    with patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_insert:
        mock_insert.return_value = None
        yield mock_insert
