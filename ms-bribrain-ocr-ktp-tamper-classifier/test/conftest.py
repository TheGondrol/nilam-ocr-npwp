"""
Pytest configuration and shared fixtures for unit tests
"""

import io
import os
from unittest.mock import Mock

import pytest


@pytest.fixture(autouse=True)
def _init_threshold_provider():
    """The predict route reads get_provider().get("threshold"); the FastAPI
    lifespan (which calls init_provider) is not run under TestClient/
    ASGITransport, so initialize a provider with a default threshold for every
    test. threshold=0.5 keeps the existing prediction assertions valid."""
    import src.services.threshold_provider as _tp
    _tp._provider = None
    _tp.init_provider("tamper", {"threshold": 0.5})
    yield
    _tp._provider = None


try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None  # type: ignore

# Set environment variables for testing
os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost/testdb"
os.environ["MINIO_ENDPOINT"] = "localhost:9000"
os.environ["MINIO_ACCESS_KEY"] = "test_access_key"
os.environ["MINIO_SECRET_KEY"] = "test_secret_key"
os.environ["MINIO_SECURE"] = "false"
os.environ.setdefault("API_KEY", "test")


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    if not PIL_AVAILABLE:
        pytest.skip("PIL not available")
    return Image.new("RGB", (100, 100), color="white")


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Create sample image bytes for testing."""
    if not PIL_AVAILABLE:
        pytest.skip("PIL not available")
    img = Image.new("RGB", (100, 100), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    return img_bytes.getvalue()


@pytest.fixture
def mock_processor() -> Mock:
    """Create a mock image processor."""
    processor = Mock()
    if TORCH_AVAILABLE:
        processor.return_value = {"pixel_values": torch.randn(1, 3, 320, 320)}
    else:
        # Mock tensor-like object
        mock_tensor = Mock()
        mock_tensor.shape = (1, 3, 320, 320)
        processor.return_value = {"pixel_values": mock_tensor}
    return processor


@pytest.fixture
def mock_model() -> Mock:
    """Create a mock model with proper structure."""
    model = Mock()
    model.config = Mock()
    model.config.id2label = {0: "authentic", 1: "tampered"}
    
    # Mock outputs
    if TORCH_AVAILABLE:
        logits = torch.tensor([[0.3, 0.7]])
    else:
        # Mock tensor-like object
        logits = Mock()
        logits.shape = (1, 2)
    
    outputs = Mock()
    outputs.logits = logits
    model.return_value = outputs
    
    return model


@pytest.fixture
def mock_torch_device() -> str:
    """Return a mock torch device."""
    return "cpu"


@pytest.fixture
def sample_config_dict() -> dict:
    """Sample configuration dictionary."""
    return {
        "app": {
            "name": "Test App",
            "version": "1.0.0",
            "environment": "test"
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8000
        },
        "logging": {
            "level": "INFO",
            "log_to_database": False
        },
        "device": {
            "force_cpu": True
        },
        "model": {
            "path": "/tmp/model",
            "image_size": 320,
            "threshold": 0.5,
            "file_path": "/tmp/model.zip"
        },
        "cors": {
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"]
        },
        "minio": {
            "bucket": "test-bucket",
            "object": "test-object.zip"
        }
    }


@pytest.fixture
def temp_model_path(tmp_path):
    """Create a temporary model path."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    
    # Create dummy model files
    (model_dir / "config.json").write_text('{"id2label": {"0": "authentic", "1": "tampered"}}')
    (model_dir / "model.safetensors").write_bytes(b"dummy model data")
    (model_dir / "preprocessor_config.json").write_text('{}')
    
    return str(model_dir)
