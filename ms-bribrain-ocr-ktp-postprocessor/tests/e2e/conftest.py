"""Pytest configuration for e2e tests.

Supports two modes:
  1. In-process (default): uses httpx ASGITransport with mocked DB.
  2. Live server: set E2E_BASE_URL env var to hit a running service
     (e.g. during Docker build where the server is already up).

Usage:
    # In-process:
    pytest tests/e2e/ -m e2e -v --no-cov

    # Against live server:
    E2E_BASE_URL=http://127.0.0.1:8090 pytest tests/e2e/ -m e2e -v --no-cov
"""

import json
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load .env so API_KEY is available to tests (same as server's load_dotenv)
from dotenv import load_dotenv
load_dotenv(project_root / ".env", override=False)

import pytest
import httpx


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

E2E_BASE_URL = os.getenv("E2E_BASE_URL", "")


# ==================== Complete KTP Data Fixtures ====================

@pytest.fixture
def complete_jakarta_ktp():
    """Complete KTP from Jakarta with all fields populated."""
    return [
        [[[100, 50], [300, 80]], ["PROVINSI DKI JAKARTA", 0.98]],
        [[[100, 90], [300, 120]], ["KOTA JAKARTA SELATAN", 0.97]],
        [[[100, 140], [180, 170]], ["NIK", 0.99]],
        [[[190, 140], [450, 170]], ["3174012801900001", 0.98]],
        [[[100, 180], [180, 210]], ["Nama", 0.99]],
        [[[190, 180], [450, 210]], ["AHMAD FAUZI", 0.97]],
        [[[100, 220], [200, 250]], ["Tempat/Tgl Lahir", 0.96]],
        [[[210, 220], [450, 250]], ["JAKARTA, 28-01-1990", 0.95]],
        [[[100, 260], [200, 290]], ["Jenis Kelamin", 0.97]],
        [[[210, 260], [350, 290]], ["LAKI-LAKI", 0.96]],
        [[[360, 260], [450, 290]], ["Gol. Darah", 0.94]],
        [[[460, 260], [500, 290]], ["A", 0.93]],
        [[[100, 300], [180, 330]], ["Alamat", 0.98]],
        [[[190, 300], [500, 330]], ["JL. GATOT SUBROTO KAV. 53", 0.95]],
        [[[100, 340], [180, 370]], ["RT/RW", 0.97]],
        [[[190, 340], [300, 370]], ["005/008", 0.96]],
        [[[100, 380], [180, 410]], ["Kel/Desa", 0.96]],
        [[[190, 380], [350, 410]], ["KUNINGAN TIMUR", 0.95]],
        [[[100, 420], [180, 450]], ["Kecamatan", 0.97]],
        [[[190, 420], [350, 450]], ["SETIABUDI", 0.96]],
        [[[100, 460], [180, 490]], ["Agama", 0.98]],
        [[[190, 460], [280, 490]], ["ISLAM", 0.97]],
        [[[100, 500], [220, 530]], ["Status Perkawinan", 0.96]],
        [[[230, 500], [350, 530]], ["KAWIN", 0.95]],
        [[[100, 540], [180, 570]], ["Pekerjaan", 0.95]],
        [[[190, 540], [380, 570]], ["KARYAWAN SWASTA", 0.94]],
        [[[100, 580], [220, 610]], ["Kewarganegaraan", 0.96]],
        [[[230, 580], [300, 610]], ["WNI", 0.97]],
        [[[100, 620], [200, 650]], ["Berlaku Hingga", 0.95]],
        [[[210, 620], [350, 650]], ["SEUMUR HIDUP", 0.96]],
    ]


@pytest.fixture
def complete_surabaya_ktp():
    """Complete KTP from Surabaya with all fields."""
    return [
        [[[100, 50], [300, 80]], ["PROVINSI JAWA TIMUR", 0.97]],
        [[[100, 90], [300, 120]], ["KOTA SURABAYA", 0.96]],
        [[[100, 140], [180, 170]], ["NIK", 0.98]],
        [[[190, 140], [450, 170]], ["3578140507880002", 0.97]],
        [[[100, 180], [180, 210]], ["Nama", 0.98]],
        [[[190, 180], [450, 210]], ["SITI NURHALIZA", 0.96]],
        [[[100, 220], [200, 250]], ["Tempat/Tgl Lahir", 0.95]],
        [[[210, 220], [450, 250]], ["SURABAYA, 05-07-1988", 0.94]],
        [[[100, 260], [200, 290]], ["Jenis Kelamin", 0.96]],
        [[[210, 260], [350, 290]], ["PEREMPUAN", 0.95]],
        [[[360, 260], [450, 290]], ["Gol. Darah", 0.93]],
        [[[460, 260], [500, 290]], ["B", 0.92]],
        [[[100, 300], [180, 330]], ["Alamat", 0.97]],
        [[[190, 300], [500, 330]], ["JL. RAYA DARMO NO. 87", 0.94]],
        [[[100, 340], [180, 370]], ["RT/RW", 0.96]],
        [[[190, 340], [300, 370]], ["002/010", 0.95]],
        [[[100, 380], [180, 410]], ["Kel/Desa", 0.95]],
        [[[190, 380], [350, 410]], ["DARMO", 0.94]],
        [[[100, 420], [180, 450]], ["Kecamatan", 0.96]],
        [[[190, 420], [350, 450]], ["WONOKROMO", 0.95]],
        [[[100, 460], [180, 490]], ["Agama", 0.97]],
        [[[190, 460], [280, 490]], ["ISLAM", 0.96]],
        [[[100, 500], [220, 530]], ["Status Perkawinan", 0.95]],
        [[[230, 500], [350, 530]], ["BELUM KAWIN", 0.94]],
        [[[100, 540], [180, 570]], ["Pekerjaan", 0.94]],
        [[[190, 540], [380, 570]], ["WIRASWASTA", 0.93]],
        [[[100, 580], [220, 610]], ["Kewarganegaraan", 0.95]],
        [[[230, 580], [300, 610]], ["WNI", 0.96]],
        [[[100, 620], [200, 650]], ["Berlaku Hingga", 0.94]],
        [[[210, 620], [350, 650]], ["SEUMUR HIDUP", 0.95]],
    ]


@pytest.fixture
def complete_bandung_ktp():
    """Complete KTP from Bandung with typical West Java data."""
    return [
        [[[100, 50], [300, 80]], ["PROVINSI JAWA BARAT", 0.96]],
        [[[100, 90], [300, 120]], ["KOTA BANDUNG", 0.95]],
        [[[100, 140], [180, 170]], ["NIK", 0.98]],
        [[[190, 140], [450, 170]], ["3273010101850003", 0.97]],
        [[[100, 180], [180, 210]], ["Nama", 0.97]],
        [[[190, 180], [450, 210]], ["DEDI MULYADI", 0.96]],
        [[[100, 220], [200, 250]], ["Tempat/Tgl Lahir", 0.95]],
        [[[210, 220], [450, 250]], ["BANDUNG, 01-01-1985", 0.94]],
        [[[100, 260], [200, 290]], ["Jenis Kelamin", 0.96]],
        [[[210, 260], [350, 290]], ["LAKI-LAKI", 0.95]],
        [[[100, 300], [180, 330]], ["Alamat", 0.96]],
        [[[190, 300], [500, 330]], ["JL. ASIA AFRIKA NO. 65", 0.94]],
        [[[100, 340], [180, 370]], ["RT/RW", 0.95]],
        [[[190, 340], [300, 370]], ["001/003", 0.94]],
        [[[100, 380], [180, 410]], ["Kel/Desa", 0.94]],
        [[[190, 380], [350, 410]], ["BRAGA", 0.93]],
        [[[100, 420], [180, 450]], ["Kecamatan", 0.95]],
        [[[190, 420], [350, 450]], ["SUMUR BANDUNG", 0.94]],
        [[[100, 460], [180, 490]], ["Agama", 0.96]],
        [[[190, 460], [280, 490]], ["KATOLIK", 0.95]],
        [[[100, 500], [220, 530]], ["Status Perkawinan", 0.94]],
        [[[230, 500], [350, 530]], ["KAWIN", 0.93]],
        [[[100, 540], [180, 570]], ["Pekerjaan", 0.93]],
        [[[190, 540], [380, 570]], ["PNS", 0.92]],
        [[[100, 580], [220, 610]], ["Kewarganegaraan", 0.94]],
        [[[230, 580], [300, 610]], ["WNI", 0.95]],
    ]


# ==================== Edge Case Fixtures ====================

@pytest.fixture
def ktp_with_ocr_typos():
    """KTP with common OCR typos/errors that need correction."""
    return [
        [[[100, 140], [180, 170]], ["NlK", 0.89]],
        [[[190, 140], [450, 170]], ["3l740128O195OO01", 0.85]],
        [[[100, 180], [180, 210]], ["Nama:", 0.92]],
        [[[190, 180], [450, 210]], ["BUDI SANT0S0", 0.88]],
        [[[100, 220], [200, 250]], ["Tempat/Tgl Lahir", 0.87]],
        [[[210, 220], [450, 250]], ["JAKARTA,28-Ol-l995", 0.82]],
        [[[100, 260], [200, 290]], ["Jenis Ke1amin", 0.85]],
        [[[210, 260], [350, 290]], ["LAKl-LAKl", 0.84]],
        [[[100, 300], [180, 330]], ["A1amat", 0.86]],
        [[[190, 300], [500, 330]], ["JL. SUDlRMAN NO. l0", 0.83]],
        [[[100, 340], [180, 370]], ["RT/RW", 0.88]],
        [[[190, 340], [300, 370]], ["O01/OO2", 0.85]],
        [[[100, 460], [180, 490]], ["Agama", 0.90]],
        [[[190, 460], [280, 490]], ["lSLAM", 0.87]],
        [[[100, 500], [220, 530]], ["Status Perkawinan", 0.86]],
        [[[230, 500], [350, 530]], ["KAWlN", 0.84]],
    ]


@pytest.fixture
def ktp_with_missing_fields():
    """KTP with several missing fields."""
    return [
        [[[100, 140], [180, 170]], ["NIK", 0.95]],
        [[[190, 140], [450, 170]], ["3174012801950001", 0.94]],
        [[[100, 180], [180, 210]], ["Nama", 0.96]],
        [[[190, 180], [450, 210]], ["BUDI SANTOSO", 0.95]],
        [[[100, 220], [200, 250]], ["Tempat/Tgl Lahir", 0.94]],
        [[[210, 220], [450, 250]], ["JAKARTA, 28-01-1995", 0.93]],
        [[[100, 260], [200, 290]], ["Jenis Kelamin", 0.94]],
        [[[210, 260], [350, 290]], ["LAKI-LAKI", 0.93]],
    ]


@pytest.fixture
def ktp_with_low_confidence():
    """KTP with low confidence scores across all fields."""
    return [
        [[[100, 140], [180, 170]], ["NIK", 0.55]],
        [[[190, 140], [450, 170]], ["3174012801950001", 0.52]],
        [[[100, 180], [180, 210]], ["Nama", 0.48]],
        [[[190, 180], [450, 210]], ["BUDI SANTOSO", 0.50]],
        [[[100, 220], [200, 250]], ["Tempat/Tgl Lahir", 0.45]],
        [[[210, 220], [450, 250]], ["JAKARTA, 28-01-1995", 0.47]],
        [[[100, 260], [200, 290]], ["Jenis Kelamin", 0.42]],
        [[[210, 260], [350, 290]], ["LAKI-LAKI", 0.44]],
        [[[100, 300], [180, 330]], ["Alamat", 0.40]],
        [[[190, 300], [500, 330]], ["JL. SUDIRMAN NO. 10", 0.43]],
        [[[100, 460], [180, 490]], ["Agama", 0.53]],
        [[[190, 460], [280, 490]], ["ISLAM", 0.51]],
    ]


@pytest.fixture
def ktp_with_unusual_formatting():
    """KTP with unusual text formatting."""
    return [
        [[[100, 140], [450, 170]], ["NIK : 3174012801950001", 0.94]],
        [[[100, 180], [450, 210]], ["Nama : BUDI SANTOSO", 0.93]],
        [[[100, 220], [450, 250]], ["Tempat/Tgl Lahir : JAKARTA 28011995", 0.91]],
        [[[100, 260], [450, 290]], ["Jenis Kelamin : LAKI-LAKI", 0.92]],
        [[[100, 300], [450, 330]], ["Alamat : JL SUDIRMAN NO 10", 0.90]],
        [[[100, 340], [450, 370]], ["RT/RW : 001/002", 0.91]],
        [[[100, 380], [450, 410]], ["Kel/Desa : MENTENG", 0.89]],
        [[[100, 420], [450, 450]], ["Kecamatan : MENTENG", 0.88]],
        [[[100, 460], [450, 490]], ["Agama : ISLAM", 0.92]],
        [[[100, 500], [450, 530]], ["Status Perkawinan : KAWIN", 0.90]],
    ]


@pytest.fixture
def ktp_female_non_muslim():
    """KTP for a female with non-Muslim religion."""
    return [
        [[[100, 50], [300, 80]], ["PROVINSI BALI", 0.96]],
        [[[100, 90], [300, 120]], ["KOTA DENPASAR", 0.95]],
        [[[100, 140], [180, 170]], ["NIK", 0.97]],
        [[[190, 140], [450, 170]], ["5171014505920001", 0.96]],
        [[[100, 180], [180, 210]], ["Nama", 0.96]],
        [[[190, 180], [450, 210]], ["NI MADE DEWI", 0.95]],
        [[[100, 220], [200, 250]], ["Tempat/Tgl Lahir", 0.94]],
        [[[210, 220], [450, 250]], ["DENPASAR, 05-05-1992", 0.93]],
        [[[100, 260], [200, 290]], ["Jenis Kelamin", 0.95]],
        [[[210, 260], [350, 290]], ["PEREMPUAN", 0.94]],
        [[[100, 300], [180, 330]], ["Alamat", 0.95]],
        [[[190, 300], [500, 330]], ["JL. PANTAI KUTA NO. 15", 0.93]],
        [[[100, 340], [180, 370]], ["RT/RW", 0.94]],
        [[[190, 340], [300, 370]], ["003/005", 0.93]],
        [[[100, 380], [180, 410]], ["Kel/Desa", 0.93]],
        [[[190, 380], [350, 410]], ["KUTA", 0.92]],
        [[[100, 420], [180, 450]], ["Kecamatan", 0.94]],
        [[[190, 420], [350, 450]], ["KUTA", 0.93]],
        [[[100, 460], [180, 490]], ["Agama", 0.95]],
        [[[190, 460], [280, 490]], ["HINDU", 0.94]],
        [[[100, 500], [220, 530]], ["Status Perkawinan", 0.93]],
        [[[230, 500], [380, 530]], ["CERAI HIDUP", 0.92]],
        [[[100, 540], [180, 570]], ["Pekerjaan", 0.92]],
        [[[190, 540], [380, 570]], ["PENGUSAHA", 0.91]],
    ]


@pytest.fixture
def ktp_christian_widower():
    """KTP for Christian widower."""
    return [
        [[[100, 50], [300, 80]], ["PROVINSI SULAWESI UTARA", 0.95]],
        [[[100, 90], [300, 120]], ["KOTA MANADO", 0.94]],
        [[[100, 140], [180, 170]], ["NIK", 0.97]],
        [[[190, 140], [450, 170]], ["7171012010650001", 0.96]],
        [[[100, 180], [180, 210]], ["Nama", 0.96]],
        [[[190, 180], [450, 210]], ["PAULUS MANTIRI", 0.95]],
        [[[100, 220], [200, 250]], ["Tempat/Tgl Lahir", 0.94]],
        [[[210, 220], [450, 250]], ["MANADO, 20-10-1965", 0.93]],
        [[[100, 260], [200, 290]], ["Jenis Kelamin", 0.95]],
        [[[210, 260], [350, 290]], ["LAKI-LAKI", 0.94]],
        [[[100, 300], [180, 330]], ["Alamat", 0.95]],
        [[[190, 300], [500, 330]], ["JL. SAM RATULANGI NO. 88", 0.93]],
        [[[100, 340], [180, 370]], ["RT/RW", 0.94]],
        [[[190, 340], [300, 370]], ["007/012", 0.93]],
        [[[100, 380], [180, 410]], ["Kel/Desa", 0.93]],
        [[[190, 380], [350, 410]], ["WENANG UTARA", 0.92]],
        [[[100, 420], [180, 450]], ["Kecamatan", 0.94]],
        [[[190, 420], [350, 450]], ["WENANG", 0.93]],
        [[[100, 460], [180, 490]], ["Agama", 0.95]],
        [[[190, 460], [300, 490]], ["KRISTEN", 0.94]],
        [[[100, 500], [220, 530]], ["Status Perkawinan", 0.93]],
        [[[230, 500], [380, 530]], ["CERAI MATI", 0.92]],
    ]


# ==================== Batch Fixtures ====================

@pytest.fixture
def multiple_ktp_samples(
    complete_jakarta_ktp,
    complete_surabaya_ktp,
    complete_bandung_ktp,
    ktp_female_non_muslim,
    ktp_christian_widower,
):
    """Collection of multiple KTP samples for batch testing."""
    return {
        "jakarta_male_muslim": complete_jakarta_ktp,
        "surabaya_female_muslim": complete_surabaya_ktp,
        "bandung_male_catholic": complete_bandung_ktp,
        "bali_female_hindu": ktp_female_non_muslim,
        "manado_male_christian": ktp_christian_widower,
    }


@pytest.fixture
def edge_case_samples(
    ktp_with_ocr_typos,
    ktp_with_missing_fields,
    ktp_with_low_confidence,
    ktp_with_unusual_formatting,
):
    """Collection of edge case samples."""
    return {
        "ocr_typos": ktp_with_ocr_typos,
        "missing_fields": ktp_with_missing_fields,
        "low_confidence": ktp_with_low_confidence,
        "unusual_formatting": ktp_with_unusual_formatting,
    }


# ==================== Invalid/Error Case Fixtures ====================

@pytest.fixture
def invalid_json_string():
    """Invalid JSON string for error testing."""
    return "{invalid json data"


@pytest.fixture
def empty_ocr_data():
    """Empty OCR data list."""
    return []


@pytest.fixture
def malformed_ocr_structure():
    """Malformed OCR data structure."""
    return ["just a string", [1, 2, 3], {"key": "value"}]


@pytest.fixture
def extreme_long_text():
    """Extremely long text in fields."""
    return [
        [[[100, 140], [180, 170]], ["NIK", 0.95]],
        [[[190, 140], [450, 170]], ["3174012801950001" + "0" * 100, 0.94]],
        [[[100, 180], [180, 210]], ["Nama", 0.96]],
        [[[190, 180], [450, 210]], ["NAMA SANGAT PANJANG " * 20, 0.95]],
        [[[100, 300], [180, 330]], ["Alamat", 0.95]],
        [[[190, 300], [500, 330]], ["JL. ALAMAT SANGAT PANJANG " * 30, 0.93]],
        [[[100, 260], [200, 290]], ["Jenis Kelamin", 0.94]],
        [[[210, 260], [350, 290]], ["LAKI-LAKI", 0.93]],
        [[[100, 460], [180, 490]], ["Agama", 0.95]],
        [[[190, 460], [280, 490]], ["ISLAM", 0.94]],
    ]


# ---------------------------------------------------------------------------
# Client fixture (dual-mode: in-process or live server)
# ---------------------------------------------------------------------------


@pytest.fixture
async def e2e_client():
    """
    Create an async httpx client for e2e tests.

    If E2E_BASE_URL is set, connects to a live running service.
    Otherwise, runs in-process with mocked DB.
    """
    api_key = os.getenv("API_KEY", "test-api-key")

    if E2E_BASE_URL:
        # ---- Live server mode ----
        async with httpx.AsyncClient(base_url=E2E_BASE_URL) as client:
            client.headers["X-API-Key"] = api_key
            yield client
    else:
        # ---- In-process mode ----
        from unittest.mock import patch, AsyncMock

        with patch.dict(os.environ, {"API_KEY": api_key}):
            with patch("src.services.database_service.init_engine", new_callable=AsyncMock):
                with patch("src.services.database_service.dispose_engine", new_callable=AsyncMock):
                    with patch("src.api.routes.insert_log", new_callable=AsyncMock):
                        from src.main import app

                        transport = httpx.ASGITransport(app=app)
                        async with httpx.AsyncClient(
                            transport=transport, base_url="http://test"
                        ) as client:
                            client.headers["X-API-Key"] = api_key
                            yield client


# Keep the sync test_client for backward compatibility with existing e2e tests
@pytest.fixture
def test_client():
    """Synchronous test client for e2e API testing with API key."""
    from fastapi.testclient import TestClient
    from src.main import app

    client = TestClient(app)
    client.headers["X-API-Key"] = "test"
    return client


@pytest.fixture
def mock_database_insert():
    """Mock database insert for e2e tests to avoid database dependency."""
    from unittest.mock import patch, AsyncMock

    with patch("src.api.routes.insert_log", new_callable=AsyncMock) as mock_insert:
        mock_insert.return_value = None
        yield mock_insert
