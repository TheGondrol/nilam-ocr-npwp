"""Shared fixtures for integration tests."""

import json
from unittest.mock import patch

import cv2
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _set_api_key():
    with patch.dict("os.environ", {"API_KEY": "test"}):
        yield


@pytest.fixture
def image_bytes():
    """Create valid JPEG image bytes."""
    img = np.ones((200, 300, 3), dtype=np.uint8) * 128
    img[40:60, 40:60] = 255
    img[45:55, 45:55] = 0
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture
def ocr_result_str():
    """OCR result as JSON string."""
    data = [
        [[[10, 10], [150, 10], [150, 30], [10, 30]], ("PROVINSI DKI JAKARTA", 0.95)],
        [[[10, 40], [150, 40], [150, 60], [10, 60]], ("KOTA JAKARTA PUSAT", 0.90)],
        [[[10, 70], [80, 70], [80, 90], [10, 90]], ("NIK", 0.85)],
    ]
    return json.dumps(data)


@pytest.fixture
def empty_ocr_result_str():
    return json.dumps([])
