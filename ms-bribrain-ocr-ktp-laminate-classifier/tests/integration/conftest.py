"""Shared fixtures for integration tests."""

import io
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def _set_api_key():
    with patch.dict("os.environ", {"API_KEY": "test"}):
        yield


@pytest.fixture
def image_bytes():
    """Create valid JPEG image bytes."""
    img = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 128)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def large_image_bytes():
    """Create image bytes larger than MAX_FILE_SIZE (>10MB)."""
    return b"\x00" * (11 * 1024 * 1024)
