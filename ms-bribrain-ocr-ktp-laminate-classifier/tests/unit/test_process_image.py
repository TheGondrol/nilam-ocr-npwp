"""Unit tests for _process_image_sync in src.api.routes."""

import io
import numpy as np
from PIL import Image
import pytest

from src.api.routes import _process_image_sync


class TestProcessImageSync:
    def test_rgb_image(self):
        img = Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        result = _process_image_sync(buf.getvalue())
        assert result.mode in ("RGB", "L")

    def test_rgba_image_converted(self):
        img = Image.fromarray(np.zeros((50, 50, 4), dtype=np.uint8), mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = _process_image_sync(buf.getvalue())
        assert result.mode == "RGB"

    def test_grayscale_image(self):
        img = Image.fromarray(np.zeros((50, 50), dtype=np.uint8), mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = _process_image_sync(buf.getvalue())
        assert result.mode == "L"

    def test_invalid_bytes_raises(self):
        with pytest.raises(Exception):
            _process_image_sync(b"not an image")

    def test_palette_image_converted_to_rgb(self):
        # Palette-mode ("P") images are neither RGB nor L; must be converted to RGB.
        img = Image.new("P", (50, 50))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = _process_image_sync(buf.getvalue())
        assert result.mode == "RGB"

    def test_la_image_converted_to_rgb(self):
        # LA (grayscale + alpha) also falls outside {RGB, L} and must be converted.
        img = Image.new("LA", (50, 50))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = _process_image_sync(buf.getvalue())
        assert result.mode == "RGB"

    def test_1x1_pixel_image(self):
        # Boundary: smallest valid image. Must not crash; predictor will resize later.
        img = Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = _process_image_sync(buf.getvalue())
        assert result.size == (1, 1)

    def test_corrupted_header_raises(self):
        # Valid PNG magic bytes followed by garbage — Pillow should fail on decode.
        with pytest.raises(Exception):
            _process_image_sync(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
