"""
Pytest configuration and shared fixtures for the test suite.
"""
import pytest
import numpy as np
import cv2
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any


@pytest.fixture
def sample_ocr_result():
    """
    Sample OCR result for testing.
    Format: [[[coordinates], (text, confidence)], ...]
    """
    return [
        [[[10, 10], [100, 10], [100, 30], [10, 30]], ("PROVINSI DKI JAKARTA", 0.95)],
        [[[10, 40], [100, 40], [100, 60], [10, 60]], ("KOTA JAKARTA PUSAT", 0.90)],
        [[[10, 70], [80, 70], [80, 90], [10, 90]], ("NIK", 0.85)],
        [[[90, 70], [200, 70], [200, 90], [90, 90]], ("3175012345678901", 0.92)],
        [[[10, 100], [80, 100], [80, 120], [10, 120]], ("Nama", 0.88)],
    ]


@pytest.fixture
def low_confidence_ocr_result():
    """OCR result with low confidence scores."""
    return [
        [[[10, 10], [100, 10], [100, 30], [10, 30]], ("PROVINSI", 0.45)],
        [[[10, 40], [100, 40], [100, 60], [10, 60]], ("KOTA", 0.40)],
        [[[10, 70], [80, 70], [80, 90], [10, 90]], ("NIK", 0.35)],
    ]


@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale image for testing (sharp)."""
    # Create a sharp image with clear edges
    img = np.ones((100, 100), dtype=np.uint8) * 128
    # Add some clear edges
    img[40:60, 40:60] = 255
    img[45:55, 45:55] = 0
    return img


@pytest.fixture
def blurry_grayscale_image():
    """Create a blurry grayscale image for testing."""
    # Create a uniformly blurry image
    img = np.ones((100, 100), dtype=np.uint8) * 128
    # Apply Gaussian blur to make it blurry
    img = cv2.GaussianBlur(img, (51, 51), 0)
    return img


@pytest.fixture
def sample_bgr_image():
    """Create a sample BGR color image for testing."""
    # Create a simple BGR image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    # Add some colored rectangles
    img[20:40, 20:40] = [255, 0, 0]  # Blue
    img[60:80, 60:80] = [0, 255, 0]  # Green
    return img


@pytest.fixture
def glare_bgr_image():
    """Create a BGR image with bright glare regions."""
    # Create image with high brightness areas
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    # Add very bright regions (simulating glare)
    img[30:70, 30:70] = [255, 255, 255]
    return img


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB color image for testing."""
    # Create a simple RGB image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    # Add some colored rectangles
    img[20:40, 20:40] = [255, 0, 0]  # Red
    img[60:80, 60:80] = [0, 255, 0]  # Green
    return img


@pytest.fixture
def landscape_image():
    """Create a landscape-oriented BGR image (OpenCV-native)."""
    return np.ones((100, 200, 3), dtype=np.uint8) * 128


@pytest.fixture
def portrait_image():
    """Create a portrait-oriented BGR image (OpenCV-native)."""
    return np.ones((200, 100, 3), dtype=np.uint8) * 128


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing."""
    # Create a simple image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    # Encode to JPEG
    success, buffer = cv2.imencode('.jpg', img)
    if success:
        return buffer.tobytes()
    return b""


@pytest.fixture
def mock_database_session():
    """Mock async database session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def mock_face_detector():
    """Mock InsightFace FaceAnalysis detector."""
    detector = MagicMock()
    # Mock get() method that returns a list of Face objects
    detector.get = MagicMock(return_value=[])
    return detector


@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Sample configuration dictionary for testing."""
    return {
        "app": {
            "name": "OCR Quality Service",
            "version": "1.0.0"
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8000
        },
        "quality": {
            "blur": {
                "threshold": 100.0
            },
            "confidence": {
                "threshold_median": 0.5
            },
            "glare": {
                "min_area": 500,
                "padding_size": 10,
                "kernel_size": 5,
                "min_text_confidence": 0.5,
                "affected_percentage_threshold": 0.3
            },
            "rotation": {
                "min_face_proportion": 0.05,
                "face_detection_confidence": 0.5,
                "verification_box_width_ratio": 0.65,
                "verification_box_height_ratio": 0.25,
                "verification_box_margin_ratio": 0.05,
                "verification_box_vertical_center": 0.33
            }
        },
        "database": {
            "pool_size": 5,
            "max_overflow": 10
        },
        "logging": {
            "log_to_database": True,
            "level": "INFO"
        }
    }


@pytest.fixture(autouse=True)
def _init_threshold_provider():
    """Initialise the ThresholdProvider singleton for unit tests.

    The blur/glare/rotation detectors read their thresholds live via
    ``get_provider().get(...)``. In production the provider is initialised in the
    FastAPI lifespan (``src/main.py``); unit tests don't run that, so initialise
    it here with the same defaults the service seeds from ``config.yaml``. Only
    the in-memory defaults are served — ``init_provider`` (not ``initialize``)
    opens no database connection and starts no refresh task.
    """
    import src.services.threshold_provider as threshold_provider

    threshold_provider._provider = None
    threshold_provider.init_provider(
        "dgc_irl",
        {
            "blur_threshold": 100.0,
            "confidence_threshold_median": 0.8,
            "glare_min_text_confidence": 0.6,
            "glare_affected_percentage_threshold": 5.0,
            "rotation_min_face_proportion": 0.12,
            "rotation_face_detection_confidence": 0.5,
        },
    )
    yield
    threshold_provider._provider = None
