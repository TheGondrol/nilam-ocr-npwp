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
from unittest.mock import AsyncMock, MagicMock  # noqa: E402
from typing import Dict, Any  # noqa: E402
import io  # noqa: E402


@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Sample configuration dictionary for testing."""
    return {
        "model": {
            "path": "./models/model.pth",
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
            "port": 8003,
            "thread_pool_workers": 4,
        },
        "logging": {
            "level": "INFO",
            "log_to_database": True,
        },
        "device": {
            "prefer_gpu": True,
            "force_cpu": False
        },
        "api": {
            "title": "Graycopy Classifier API",
            "description": "API for detecting graycopy images",
            "version": "1.0.0"
        }
    }


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing."""
    img = Image.new('RGB', (800, 600), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.read()


@pytest.fixture
def mock_model():
    """Mock PyTorch model for testing."""
    model = MagicMock()
    import torch
    model.return_value = torch.tensor([[0.3, 0.7]])
    return model


@pytest.fixture
def mock_device():
    """Mock PyTorch device."""
    import torch
    return torch.device("cpu")


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
