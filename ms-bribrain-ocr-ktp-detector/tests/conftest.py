"""
Pytest configuration and fixtures
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient


# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Mock configuration dictionary"""
    return {
        'model': {
            'path': './src/models/test_model.pt',
            'confidence': 0.5,
            'iou_threshold': 0.45,
            'export_format': 'pt',
        },
        'classes': {
            'names': {
                0: 'ktp',
                1: 'non-ktp'
            }
        },
        'server': {
            'host': '0.0.0.0',
            'port': 8003
        },
        'logging': {
            'level': 'INFO',
            'log_to_database': False
        },
        'device': {
            'prefer_gpu': True,
            'force_cpu': False
        },
        'performance': {
            'thread_pool_multiplier': 1
        },
        'cors': {
            'allow_origins': ["*"],
            'allow_credentials': True,
            'allow_methods': ["*"],
            'allow_headers': ["*"]
        }
    }


@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model"""
    model = MagicMock()
    
    # Mock prediction results
    mock_result = MagicMock()
    mock_result.boxes = MagicMock()
    mock_result.boxes.xyxy = np.array([[100, 100, 200, 200]])
    mock_result.boxes.conf = np.array([0.85])
    mock_result.boxes.cls = np.array([0])
    
    model.return_value = [mock_result]
    model.names = {0: 'ktp', 1: 'non-ktp'}
    model.to = MagicMock(return_value=model)
    
    return model


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample test image"""
    # Create a simple RGB image
    img_array = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_image_bytes(sample_image) -> bytes:
    """Create sample image bytes"""
    import io
    buf = io.BytesIO()
    sample_image.save(buf, format='JPEG')
    return buf.getvalue()


@pytest.fixture
def temp_config_file(mock_config, tmp_path):
    """Create a temporary config file"""
    import yaml
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(mock_config, f)
    return config_file


@pytest.fixture
def mock_database_session():
    """Mock async database session"""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def mock_minio_client():
    """Mock MinIO client"""
    client = MagicMock()
    client.get_object = MagicMock()
    return client


@pytest.fixture
def clean_env():
    """Clean environment variables before and after test"""
    original_env = os.environ.copy()
    
    # Clean up any test-specific env vars
    test_vars = ['DATABASE_URL', 'MINIO_ENDPOINT', 'MINIO_ACCESS_KEY',
                 'MINIO_SECRET_KEY', 'CONFIG_PATH', 'CPU_LIMIT', 'API_KEY']
    for var in test_vars:
        os.environ.pop(var, None)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def app_client():
    """Create test client for FastAPI app"""
    # Mock everything BEFORE importing app
    with patch.dict(os.environ, {"API_KEY": "test-api-key"}):
      with patch('src.services.database_service.init_engine', new_callable=AsyncMock):
        with patch('src.services.database_service.dispose_engine', new_callable=AsyncMock):
            with patch('src.services.minio_service.download_model_minio'):
                with patch('src.services.predictor.predictor') as mock_predictor:
                    # Set up predictor mock - is_loaded should be a callable
                    mock_predictor.is_loaded = MagicMock(return_value=True)
                    mock_predictor.get_device_string.return_value = "cpu"
                    mock_predictor.get_device_info.return_value = {"device": "cpu"}
                    mock_predictor.load_model = MagicMock()

                    # Mock predict method with a function that returns proper format
                    # This ensures it works correctly with run_in_executor
                    def mock_predict(image, filename):
                        return [
                            {
                                "class_id": 0,
                                "class_name": "ktp",
                                "confidence": 0.95,
                                "bbox": {"x1": 10.0, "y1": 20.0, "x2": 100.0, "y2": 200.0}
                            }
                        ]

                    mock_predictor.predict = mock_predict

                    # NOW import app after all mocks are in place
                    from src.main import app

                    # Also patch the routes-level predictor reference
                    # in case the module was already imported by earlier tests
                    with patch('src.api.routes.predictor', mock_predictor):
                        # Create TestClient
                        client = TestClient(app)
                        yield client


@pytest.fixture
def mock_torch_cuda():
    """Mock torch.cuda for GPU testing"""
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.cuda.get_device_name', return_value='NVIDIA GeForce GTX 1080'):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value = MagicMock(
                    total_memory=8589934592,  # 8GB
                    multi_processor_count=20
                )
                yield


@pytest.fixture
def mock_torch_cpu():
    """Mock torch.cuda for CPU-only testing"""
    with patch('torch.cuda.is_available', return_value=False):
        yield
