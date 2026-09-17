"""
Pytest configuration and shared fixtures for the test suite.
"""
import sys
from pathlib import Path

# Add project root to Python path for src module imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest  # noqa: E402
from PIL import Image  # noqa: E402
from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402
from typing import Dict, Any  # noqa: E402
import io  # noqa: E402


@pytest.fixture(autouse=True)
def _set_api_key():
    """Set API_KEY env var for all tests."""
    with patch.dict("os.environ", {"API_KEY": "test"}):
        yield


@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Sample configuration dictionary for testing."""
    return {
        "model": {
            "path": "./models/best_cnn_detector_corrected.pth",
            "image_size": 224,
            "crop_size": 224,
            "normalize_size": 512,
            "num_classes": 2
        },
        "prediction": {
            "threshold": 0.5
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8002,
            "thread_pool_workers": 4,
            "cors_allowed_origins": ["*"],
            "cors_allowed_credentials": True,
            "cors_allowed_methods": ["*"],
            "cors_allowed_headers": ["*"]
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "log_to_database": True,
            "file": {
                "enabled": True,
                "directory": "./logs",
                "filename": "recapture_{date}.log",
                "max_bytes": 10485760,
                "backup_count": 5
            }
        },
        "device": {
            "prefer_gpu": True,
            "force_cpu": False
        },
        "api": {
            "title": "Screen Recapture Detection API",
            "description": "API for detecting recaptured screen photos",
            "version": "1.0.0"
        },
        "performance": {
            "num_threads": 4
        }
    }


@pytest.fixture
def sample_image_stream():
    """Create a sample image as BytesIO for testing."""
    img = Image.new('RGB', (800, 600), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing."""
    img = Image.new('RGB', (800, 600), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.read()


@pytest.fixture
def small_image_bytes():
    """Create a small image for testing."""
    img = Image.new('RGB', (100, 100), color='green')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.read()


@pytest.fixture
def mock_model():
    """Mock PyTorch model for testing."""
    model = MagicMock()
    # Mock forward pass return value
    import torch
    model.return_value = torch.tensor([[0.3, 0.7]])  # Class 1 (RECAPTURED) with higher prob
    return model


@pytest.fixture
def mock_device():
    """Mock PyTorch device."""
    import torch
    return torch.device("cpu")


@pytest.fixture
def mock_transform():
    """Mock transform function."""
    def transform(img):
        import torch
        # Return a fake tensor
        return torch.zeros((3, 224, 224))
    return transform


@pytest.fixture
def mock_database_session():
    """Mock async database session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def mock_request_id():
    """Mock request ID for testing."""
    return "test-request-12345"
