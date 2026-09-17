"""Unit tests for src.services.crop_helpers (OpenCV-backed)."""

import base64

import pytest

# crop_helpers needs cv2 + numpy; skip the whole module if either is unavailable
# (this text-only service may not install them).
np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from src.services.crop_helpers import crop_image, order_points  # noqa: E402


class TestOrderPoints:
    def test_orders_to_tl_tr_br_bl(self):
        # Shuffled corners of a 10x10 square.
        pts = np.array([[10, 10], [0, 0], [0, 10], [10, 0]], dtype=np.float32)
        rect = order_points(pts)
        assert rect[0].tolist() == [0, 0]      # top-left
        assert rect[1].tolist() == [10, 0]     # top-right
        assert rect[2].tolist() == [10, 10]    # bottom-right
        assert rect[3].tolist() == [0, 10]     # bottom-left


class TestCropImage:
    def _image(self):
        return np.full((20, 20, 3), 127, dtype=np.uint8)

    def test_returns_base64_jpeg(self):
        bbox = [[0, 0], [10, 0], [10, 10], [0, 10]]
        result = crop_image(self._image(), bbox)
        assert isinstance(result, str)
        # Valid base64 that decodes to a non-empty JPEG buffer.
        assert len(base64.b64decode(result)) > 0

    def test_none_image_raises(self):
        with pytest.raises(ValueError, match="empty or None"):
            crop_image(None, [[0, 0], [1, 0], [1, 1], [0, 1]])

    def test_empty_image_raises(self):
        with pytest.raises(ValueError, match="empty or None"):
            crop_image(np.array([], dtype=np.uint8), [[0, 0], [1, 0], [1, 1], [0, 1]])

    def test_wrong_point_count_raises(self):
        with pytest.raises(ValueError, match="exactly 4 points"):
            crop_image(self._image(), [[0, 0], [1, 1], [2, 2]])

    def test_bad_point_dimensions_raises(self):
        with pytest.raises(ValueError, match="2 coordinates"):
            crop_image(self._image(), [[0, 0], [1], [2, 2], [3, 3]])
