"""Pytest configuration and shared fixtures."""

import os
import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(autouse=True)
def _set_api_key():
    """Set API_KEY env var for all tests."""
    with patch.dict("os.environ", {"API_KEY": "test"}):
        yield


@pytest.fixture
def test_client():
    """Create a FastAPI test client with API key header."""
    from src.main import app
    client = TestClient(app)
    client.headers["X-API-Key"] = "test"
    return client


@pytest.fixture
def sample_ocr_data():
    """Sample OCR data for testing."""
    return [
        [[[100, 50], [200, 50], [200, 100], [100, 100]], ("PROVINSI DKI JAKARTA", 0.95)],
        [[[100, 120], [200, 120], [200, 170], [100, 170]], ("NIK", 0.98)],
        [[[220, 120], [400, 120], [400, 170], [220, 170]], ("3174012345678901", 0.97)],
        [[[100, 190], [200, 190], [200, 240], [100, 240]], ("Nama", 0.96)],
        [[[220, 190], [400, 190], [400, 240], [220, 240]], ("BUDI SANTOSO", 0.95)],
        [[[100, 260], [250, 260], [250, 310], [100, 310]], ("Tempat/Tgl Lahir", 0.94)],
        [[[260, 260], [400, 260], [400, 310], [260, 310]], ("JAKARTA, 15-08-1990", 0.93)],
        [[[100, 330], [250, 330], [250, 380], [100, 380]], ("Jenis Kelamin", 0.95)],
        [[[260, 330], [350, 330], [350, 380], [260, 380]], ("LAKI-LAKI", 0.94)],
        [[[100, 400], [200, 400], [200, 450], [100, 450]], ("Alamat", 0.96)],
        [[[220, 400], [500, 400], [500, 450], [220, 450]], ("JL SUDIRMAN NO 10", 0.92)],
        [[[100, 470], [200, 470], [200, 520], [100, 520]], ("RT/RW", 0.95)],
        [[[220, 470], [300, 470], [300, 520], [220, 520]], ("001/002", 0.94)],
        [[[100, 540], [200, 540], [200, 590], [100, 590]], ("Kel/Desa", 0.93)],
        [[[220, 540], [350, 540], [350, 590], [220, 590]], ("TANAH ABANG", 0.92)],
        [[[100, 610], [200, 610], [200, 660], [100, 660]], ("Kecamatan", 0.94)],
        [[[220, 610], [350, 610], [350, 660], [220, 660]], ("TANAH ABANG", 0.93)],
        [[[100, 680], [200, 680], [200, 730], [100, 730]], ("Agama", 0.95)],
        [[[220, 680], [300, 680], [300, 730], [220, 730]], ("ISLAM", 0.96)],
        [[[100, 750], [250, 750], [250, 800], [100, 800]], ("Status Perkawinan", 0.94)],
        [[[260, 750], [350, 750], [350, 800], [260, 800]], ("KAWIN", 0.93)],
    ]


@pytest.fixture
def sample_cleaned_data():
    """Sample cleaned OCR data for testing."""
    return [
        ["PROVINSI DKI JAKARTA", 0.95, [[100, 50], [200, 50], [200, 100], [100, 100]]],
        ["NIK", 0.98, [[100, 120], [200, 120], [200, 170], [100, 170]]],
        ["3174012345678901", 0.97, [[220, 120], [400, 120], [400, 170], [220, 170]]],
        ["Nama", 0.96, [[100, 190], [200, 190], [200, 240], [100, 240]]],
        ["BUDI SANTOSO", 0.95, [[220, 190], [400, 190], [400, 240], [220, 240]]],
        ["Tempat/Tgl Lahir", 0.94, [[100, 260], [250, 260], [250, 310], [100, 310]]],
        ["JAKARTA, 15-08-1990", 0.93, [[260, 260], [400, 260], [400, 310], [260, 310]]],
        ["Jenis Kelamin", 0.95, [[100, 330], [250, 330], [250, 380], [100, 380]]],
        ["LAKI-LAKI", 0.94, [[260, 330], [350, 330], [350, 380], [260, 380]]],
        ["Alamat", 0.96, [[100, 400], [200, 400], [200, 450], [100, 450]]],
        ["JL SUDIRMAN NO 10", 0.92, [[220, 400], [500, 400], [500, 450], [220, 450]]],
        ["RT/RW", 0.95, [[100, 470], [200, 470], [200, 520], [100, 520]]],
        ["001/002", 0.94, [[220, 470], [300, 470], [300, 520], [220, 520]]],
        ["Kel/Desa", 0.93, [[100, 540], [200, 540], [200, 590], [100, 590]]],
        ["TANAH ABANG", 0.92, [[220, 540], [350, 540], [350, 590], [220, 590]]],
        ["Kecamatan", 0.94, [[100, 610], [200, 610], [200, 660], [100, 660]]],
        ["TANAH ABANG", 0.93, [[220, 610], [350, 610], [350, 660], [220, 660]]],
        ["Agama", 0.95, [[100, 680], [200, 680], [200, 730], [100, 730]]],
        ["ISLAM", 0.96, [[220, 680], [300, 680], [300, 730], [220, 730]]],
        ["Status Perkawinan", 0.94, [[100, 750], [250, 750], [250, 800], [100, 800]]],
        ["KAWIN", 0.93, [[260, 750], [350, 750], [350, 800], [260, 800]]],
    ]


@pytest.fixture
def mock_config(monkeypatch):
    """Mock configuration for testing."""
    config_data = {
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "version": "1.0.0"
        },
        "thresholds": {
            "confidence": 0.6,
            "partial": 80,
            "ratio": 85
        },
        "logging": {
            "level": "INFO",
            "format": "json",
            "insert_to_database": False
        },
        "database": {
            "pool_size": 5,
            "max_overflow": 10,
            "pool_pre_ping": True
        }
    }
    
    def mock_get_config():
        from src.core.config import Config
        config = Config()
        config._config = config_data
        return config
    
    monkeypatch.setattr("src.core.config.get_config", mock_get_config)
    return config_data


@pytest.fixture
def sample_nik_data():
    """Sample NIK data for testing."""
    return [
        ["3174012345678901", 0.97, [[220, 120], [400, 120], [400, 170], [220, 170]]],
        ["3174O12345678901", 0.85, [[220, 120], [400, 120], [400, 170], [220, 170]]],  # With O instead of 0
        ["31740123456789", 0.90, [[220, 120], [400, 120], [400, 170], [220, 170]]],  # Too short
        ["31740123456789012", 0.88, [[220, 120], [400, 120], [400, 170], [220, 170]]],  # Too long
    ]


@pytest.fixture
def sample_nama_data():
    """Sample nama data for testing."""
    return [
        ["BUDI SANTOSO", 0.95, [[220, 190], [400, 190], [400, 240], [220, 240]]],
        ["SITI NURHALIZA", 0.94, [[220, 190], [400, 190], [400, 240], [220, 240]]],
        ["AHMAD@FAUZI", 0.92, [[220, 190], [400, 190], [400, 240], [220, 240]]],  # With special char
    ]


@pytest.fixture
def sample_ttl_data():
    """Sample tempat/tanggal lahir data for testing."""
    return [
        ["JAKARTA, 15-08-1990", 0.93, [[260, 260], [400, 260], [400, 310], [260, 310]]],
        ["SURABAYA 20-05-1985", 0.92, [[260, 260], [400, 260], [400, 310], [260, 310]]],
        ["BANDUNG, 01-01-2000", 0.91, [[260, 260], [400, 260], [400, 310], [260, 310]]],
    ]


@pytest.fixture
def sample_rtrw_data():
    """Sample RT/RW data for testing."""
    return [
        ["001/002", 0.94, [[220, 470], [300, 470], [300, 520], [220, 520]]],
        ["010/020", 0.93, [[220, 470], [300, 470], [300, 520], [220, 520]]],
        ["5/3", 0.92, [[220, 470], [300, 470], [300, 520], [220, 520]]],
    ]
