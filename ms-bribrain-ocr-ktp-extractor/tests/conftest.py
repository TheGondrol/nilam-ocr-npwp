"""Pytest configuration and shared fixtures"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from io import BytesIO
from PIL import Image
import numpy as np

# Mock settings for testing
@pytest.fixture
def mock_settings():
    """Mock settings object with test configuration"""
    settings = MagicMock()
    settings.server_host = "0.0.0.0"
    settings.server_port = 8001
    settings.server_reload = False
    settings.ocr_server_config_path = "test_server_config.yaml"
    settings.ocr_mobile_config_path = "test_mobile_config.yaml"
    settings.ocr_max_size_mb = 10
    settings.ocr_allowed_types = ["image/jpeg", "image/png"]
    settings.ocr_width_threshold_ratio = 0.8
    settings.device_preferred = "auto"
    settings.device_force_cpu = False
    settings.log_level = "INFO"
    settings.log_console_enabled = True
    settings.log_console_level = "INFO"
    settings.log_file_enabled = False
    settings.log_file_level = "DEBUG"
    settings.log_file_path = "logs/test.log"
    settings.log_file_max_bytes = 10485760
    settings.log_file_backup_count = 5
    settings.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    settings.log_date_format = "%Y-%m-%d %H:%M:%S"
    settings.log_to_database = False
    return settings


@pytest.fixture
def temp_config_file():
    """Create a temporary config.yaml file"""
    config_content = """
server:
  host: "0.0.0.0"
  port: 8001
  reload: false

ocr:
  server_config_path: "test_server_config.yaml"
  mobile_config_path: "test_mobile_config.yaml"
  image:
    max_size_mb: 10
    allowed_types:
      - "image/jpeg"
      - "image/png"
  filter:
    width_threshold_ratio: 0.8

device:
  preferred: "auto"
  force_cpu: false

logging:
  level: "INFO"
  console:
    enabled: true
    level: "INFO"
  file:
    enabled: false
    level: "DEBUG"
    path: "logs/test.log"
    max_bytes: 10485760
    backup_count: 5
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"
  insert_to_database: false
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing"""
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='white')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.read()


@pytest.fixture
def large_image_bytes():
    """Create large image bytes (>10MB) for testing"""
    # Create a large test image
    img = Image.new('RGB', (5000, 5000), color='white')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG', quality=95)
    img_bytes.seek(0)
    return img_bytes.read()


@pytest.fixture
def mock_ocr_result():
    """Mock OCR result from PaddleOCR"""
    return {
        "rec_texts": ["Text 1", "Text 2", "Text 3"],
        "rec_scores": [0.95, 0.88, 0.92],
        "rec_polys": [
            np.array([[10, 10], [100, 10], [100, 30], [10, 30]]),
            np.array([[10, 40], [100, 40], [100, 60], [10, 60]]),
            np.array([[10, 70], [100, 70], [100, 90], [10, 90]])
        ]
    }


@pytest.fixture
def mock_paddle_ocr():
    """Mock PaddleOCR instance"""
    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = {
        "rec_texts": ["Test"],
        "rec_scores": [0.95],
        "rec_polys": [np.array([[10, 10], [100, 10], [100, 30], [10, 30]])]
    }
    return mock_ocr


@pytest.fixture
def mock_database_url(monkeypatch):
    """Mock DATABASE_URL environment variable"""
    test_db_url = "postgresql://test_user:test_pass@localhost:5432/test_db"
    monkeypatch.setenv("DATABASE_URL", test_db_url)
    return test_db_url


@pytest.fixture
def mock_sqlalchemy_engine():
    """Mock SQLAlchemy engine"""
    mock_engine = MagicMock()
    mock_session = MagicMock()
    mock_engine.begin.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_engine.begin.return_value.__exit__ = MagicMock(return_value=False)
    return mock_engine


@pytest.fixture(autouse=True)
def reset_module_globals():
    """Reset global module variables between tests"""
    yield
    # This runs after each test
    # Reset any global state if needed


@pytest.fixture
def mock_request_id():
    """Mock request ID for testing"""
    return "test-request-123"
