"""
Pytest configuration and shared fixtures for OCR KTP Orchestrator tests.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
# ruff: noqa: E402

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from aiohttp import ClientSession

from src.api.models import ServiceResult, OCRData


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings():
    """Mock application settings."""
    class MockAppSettings:
        name = "OCR KTP Orchestrator"
        version = "1.0.0"
        host = "0.0.0.0"
        port = 8060
        debug = False

    class MockLoggingSettings:
        level = "INFO"
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        file = "logs/app.log"
        log_to_database = False
        log_images = False
        max_bytes = 10485760
        backup_count = 5
        console = True

    class MockServiceConfig:
        url = "http://localhost:8000"
        timeout = 30

    class MockServicesSettings:
        ocr = MockServiceConfig()
        classifier = MockServiceConfig()
        quality = MockServiceConfig()
        quality_dl = MockServiceConfig()
        lamination = MockServiceConfig()
        recapture = MockServiceConfig()
        graycopy = MockServiceConfig()
        postprocess = MockServiceConfig()
        temper = MockServiceConfig()

    class MockRunServicesSettings:
        ocr = True
        quality = True
        quality_dl = False
        qualitydl = False
        postprocess = True
        classifier = True
        lamination = True
        recapture = True
        graycopy = True
        temper = False

    class MockMinioSettings:
        endpoint = "localhost:9000"
        access_key = "minioadmin"
        secret_key = "minioadmin"
        secure = False
        bucket = "test-bucket"
        region = "us-east-1"

    class MockSettings:
        app = MockAppSettings()
        logging = MockLoggingSettings()
        services = MockServicesSettings()
        run_services = MockRunServicesSettings()
        minio = MockMinioSettings()

    return MockSettings()


@pytest.fixture
def mock_logger():
    """Mock logger for tests."""
    return MagicMock()


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp ClientSession for service calls."""
    session = MagicMock(spec=ClientSession)
    return session


@pytest.fixture
def sample_image_bytes():
    """Sample image bytes for testing."""
    # Create a minimal PNG image (1x1 pixel)
    return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'


@pytest.fixture
def sample_jpeg_bytes():
    """Sample JPEG image bytes for testing."""
    # Minimal JPEG header
    return b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9'


@pytest.fixture
def sample_ocr_response():
    """Sample OCR service response."""
    return {
        "ocr_result": [
            {"text": "PROVINSI JAWA BARAT", "bbox": [100, 50, 300, 80]},
            {"text": "NAMA: JOHN DOE", "bbox": [100, 100, 300, 130]},
            {"text": "NIK: 1234567890123456", "bbox": [100, 150, 300, 180]},
        ]
    }


@pytest.fixture
def sample_postprocess_response():
    """Sample postprocess service response."""
    return {
        "nama": "JOHN DOE",
        "nik": "1234567890123456",
        "provinsi": "JAWA BARAT",
        "kota_kabupaten": "KOTA BANDUNG",
        "tempat_lahir": "BANDUNG",
        "tanggal_lahir": "01-01-1990",
        "jenis_kelamin": "LAKI-LAKI",
        "alamat": "JL. TEST NO. 123",
        "rt_rw": "001/002",
        "kelurahan": "TEST",
        "kecamatan": "TEST",
        "agama": "ISLAM",
        "status_perkawinan": "BELUM KAWIN",
        "pekerjaan": "KARYAWAN SWASTA",
        "kewarganegaraan": "WNI",
        "berlaku_hingga": "SEUMUR HIDUP"
    }


@pytest.fixture
def sample_quality_response_pass():
    """Sample quality check response - passing."""
    return {
        "is_blurry": False,
        "is_glare": False,
        "is_rotated": False
    }


@pytest.fixture
def sample_quality_response_fail():
    """Sample quality check response - failing."""
    return {
        "is_blurry": True,
        "is_glare": True,
        "is_rotated": False
    }


@pytest.fixture
def sample_classifier_response_valid():
    """Sample classifier response - valid KTP."""
    return {
        "prediction": "valid",
        "confidence": 0.95
    }


@pytest.fixture
def sample_classifier_response_invalid():
    """Sample classifier response - invalid KTP."""
    return {
        "prediction": "invalid",
        "confidence": 0.85
    }


@pytest.fixture
def sample_spoof_response_pass():
    """Sample spoof detection response - no spoof detected."""
    return {
        "prediction": 0,
        "confidence": 0.90
    }


@pytest.fixture
def sample_spoof_response_fail():
    """Sample spoof detection response - spoof detected."""
    return {
        "prediction": 1,
        "confidence": 0.85
    }


@pytest.fixture
def mock_database_session():
    """Mock database session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def mock_service_result_success():
    """Mock successful service result."""
    return ServiceResult.success(OCRData(text=[{"text": "test data", "bbox": [0, 0, 100, 100]}]))


@pytest.fixture
def mock_service_result_error():
    """Mock error service result."""
    return ServiceResult.fail(
        error_type="timeout",
        message="Service timeout",
        details="Connection timed out"
    )


@pytest.fixture
def mock_service_result_rejection():
    """Mock rejection service result."""
    return ServiceResult.reject(
        rejection_type="quality",
        message="Quality check failed",
        details={"is_blurry": True}
    )


@pytest.fixture
def request_id():
    """Sample request ID for testing."""
    return "OCR_test-request-id-12345"


@pytest.fixture
def sample_filename():
    """Sample filename for testing."""
    return "test_ktp_image.jpg"


@pytest.fixture
def sample_content_type():
    """Sample content type for testing."""
    return "image/jpeg"
