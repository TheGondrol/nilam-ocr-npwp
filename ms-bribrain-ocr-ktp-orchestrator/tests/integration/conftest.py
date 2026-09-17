"""
Pytest configuration for integration tests.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
# ruff: noqa: E402

import pytest
import asyncio
from unittest.mock import MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
import aiohttp
from sqlalchemy import create_engine, Column, String, Text, Integer, TIMESTAMP, Float, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func

from src.api.routes import router


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def test_app():
    """Create FastAPI test application."""
    app = FastAPI(title="OCR KTP Orchestrator Test")
    app.include_router(router)
    return app


@pytest.fixture(scope="module")
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture(scope="function")
async def aiohttp_session():
    """Create aiohttp session for integration tests."""
    async with aiohttp.ClientSession() as session:
        yield session


@pytest.fixture(scope="function")
def mock_settings():
    """Mock settings for integration tests."""
    settings = MagicMock()
    
    # App settings
    settings.app.name = "OCR KTP Orchestrator"
    settings.app.version = "1.0.0"
    settings.app.debug = True
    
    # Database settings
    settings.database.url = "sqlite:///:memory:"
    settings.database.enabled = False
    
    # Service URLs (mock endpoints)
    settings.services.ocr.url = "http://localhost:8001/ocr"
    settings.services.ocr.timeout = 30
    
    settings.services.classifier.url = "http://localhost:8002/classify"
    settings.services.classifier.timeout = 30
    
    settings.services.quality.url = "http://localhost:8003/quality"
    settings.services.quality.timeout = 30
    
    settings.services.quality_dl.url = "http://localhost:8004/quality-dl"
    settings.services.quality_dl.timeout = 30
    
    settings.services.postprocess.url = "http://localhost:8005/postprocess"
    settings.services.postprocess.timeout = 30
    
    settings.services.lamination.url = "http://localhost:8006/lamination"
    settings.services.lamination.timeout = 30
    
    settings.services.recapture.url = "http://localhost:8007/recapture"
    settings.services.recapture.timeout = 30
    
    settings.services.graycopy.url = "http://localhost:8008/graycopy"
    settings.services.graycopy.timeout = 30
    
    settings.services.temper.url = "http://localhost:8009/temper"
    settings.services.temper.timeout = 30
    
    settings.services.minio.endpoint = "localhost:9000"
    settings.services.minio.access_key = "minioadmin"
    settings.services.minio.secret_key = "minioadmin"
    settings.services.minio.bucket_name = "test-bucket"
    settings.services.minio.secure = False
    
    # Run services settings
    settings.run_services.classifier = True
    settings.run_services.quality = True
    settings.run_services.qualitydl = False
    settings.run_services.lamination = True
    settings.run_services.recapture = True
    settings.run_services.graycopy = True
    settings.run_services.temper = False
    settings.run_services.ocr = True
    settings.run_services.postprocess = True
    
    return settings


@pytest.fixture(scope="function")
def in_memory_db():
    """Create in-memory SQLite database for testing."""
    
    # Create a modified version of OcrKtpLog for SQLite
    Base = declarative_base()
    
    class OcrKtpLogTest(Base):
        __tablename__ = "ocr_ktp_log"
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        request_id = Column(String(100), nullable=False)
        response_code = Column(Integer)
        payload = Column(JSON)  # Use JSON instead of JSONB for SQLite
        error_message = Column(Text)
        result = Column(JSON)  # Use JSON instead of JSONB for SQLite
        processing_time = Column(Float)
        created_at = Column(
            TIMESTAMP(),
            server_default=func.now(),
            nullable=False
        )
    
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session, OcrKtpLogTest
    
    session.close()
    Base.metadata.drop_all(engine)


@pytest.fixture
def sample_ktp_image_bytes():
    """Sample KTP image as bytes for integration tests."""
    import cv2
    import numpy as np
    
    # Create a more realistic test image with text-like patterns
    img = np.ones((600, 900, 3), dtype=np.uint8) * 255
    
    # Add blue header
    cv2.rectangle(img, (0, 0), (900, 100), (255, 100, 100), -1)
    
    # Add text-like rectangles
    cv2.rectangle(img, (50, 150), (400, 180), (0, 0, 0), 2)
    cv2.rectangle(img, (50, 200), (600, 230), (0, 0, 0), 2)
    cv2.rectangle(img, (50, 250), (500, 280), (0, 0, 0), 2)
    
    # Add photo placeholder
    cv2.rectangle(img, (700, 150), (850, 350), (200, 200, 200), -1)
    
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()


@pytest.fixture
def sample_ocr_response():
    """Sample OCR service response."""
    return [
        [[[100, 50], [300, 50], [300, 80], [100, 80]], ["PROVINSI DKI JAKARTA", 0.95]],
        [[[100, 100], [250, 100], [250, 130], [100, 130]], ["NIK", 0.92]],
        [[[260, 100], [600, 100], [600, 130], [260, 130]], ["1234567890123456", 0.98]],
        [[[100, 150], [200, 150], [200, 180], [100, 180]], ["Nama", 0.90]],
        [[[260, 150], [500, 150], [500, 180], [260, 180]], ["JOHN DOE", 0.93]],
    ]


@pytest.fixture
def sample_postprocess_response():
    """Sample postprocess service response."""
    return {
        "ocr_result": {
            "nik": "1234567890123456",
            "nama": "JOHN DOE",
            "tempat_lahir": "JAKARTA",
            "tanggal_lahir": "01-01-1990",
            "jenis_kelamin": "LAKI-LAKI",
            "golongan_darah": "A",
            "alamat": "JL. CONTOH NO. 123",
            "rt_rw": "001/002",
            "kelurahan": "KEBAYORAN BARU",
            "kecamatan": "KEBAYORAN BARU",
            "agama": "ISLAM",
            "status_perkawinan": "BELUM KAWIN",
            "pekerjaan": "KARYAWAN SWASTA",
            "kewarganegaraan": "WNI",
            "berlaku_hingga": "SEUMUR HIDUP"
        },
        "nik_image_box": None
    }
