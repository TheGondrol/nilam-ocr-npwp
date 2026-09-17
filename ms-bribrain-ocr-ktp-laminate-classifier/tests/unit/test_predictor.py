"""Unit tests for src.services.predictor module."""

from unittest.mock import patch, MagicMock
import pytest
import torch
import numpy as np
from PIL import Image

from src.services.predictor import PredictorService


@pytest.fixture(autouse=True)
def _stub_threshold_provider():
    """threshold is now a read-only property backed by the ThresholdProvider;
    stub it to 0.5 so predict() can resolve the threshold without lifespan init."""
    provider = MagicMock()
    provider.get.return_value = 0.5
    with patch("src.services.predictor.get_provider", return_value=provider):
        yield


def _mock_transform(image):
    """Return a tensor without using torchvision (avoids Windows access violation)."""
    return torch.randn(1, 76, 76)


def _make_predictor_with_mock_model(logit_value=0.0):
    """Create a PredictorService with a mocked model (no real conv ops)."""
    ps = PredictorService()
    mock_model = MagicMock()
    # Model emits 2-class logits; softmax([0, x])[1] == sigmoid(x), so the
    # logit_value drives prob(unlaminated) exactly as the sigmoid comments expect.
    mock_model.return_value = torch.tensor([[0.0, logit_value]])
    ps.model = mock_model
    ps.device = "cpu"
    ps.device_info = {"type": "cpu", "name": "CPU", "cuda_available": False}
    ps.transform = _mock_transform
    return ps


class TestPredictorService:
    def test_initial_state(self):
        ps = PredictorService()
        assert ps.model is None
        assert ps.is_loaded() is False
        assert ps.get_device_string() == "unknown"
        assert ps.get_device_info() == {}

    def test_predict_laminated(self):
        # logit -2.0 -> sigmoid ~0.12 -> below 0.5 -> LAMINATED
        ps = _make_predictor_with_mock_model(logit_value=-2.0)
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        prediction, prob = ps.predict(img, "test.jpg")
        assert prediction == "LAMINATED"
        assert 0 <= prob <= 1

    def test_predict_unlaminated(self):
        # logit 2.0 -> sigmoid ~0.88 -> above 0.5 -> UNLAMINATED
        ps = _make_predictor_with_mock_model(logit_value=2.0)
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        prediction, prob = ps.predict(img, "test.jpg")
        assert prediction == "UNLAMINATED"
        assert 0 <= prob <= 1

    def test_predict_model_not_loaded(self):
        ps = PredictorService()
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        import pytest
        with pytest.raises(RuntimeError, match="Model not loaded"):
            ps.predict(img, "test.jpg")

    def test_predict_transform_not_initialized(self):
        ps = PredictorService()
        ps.model = MagicMock()
        ps.device = "cpu"
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        import pytest
        with pytest.raises(RuntimeError, match="Transform not initialized"):
            ps.predict(img, "test.jpg")

    def test_is_loaded_true(self):
        ps = _make_predictor_with_mock_model()
        assert ps.is_loaded() is True

    def test_get_device_string(self):
        ps = _make_predictor_with_mock_model()
        assert ps.get_device_string() == "cpu"

    def test_get_device_info(self):
        ps = _make_predictor_with_mock_model()
        info = ps.get_device_info()
        assert info["type"] == "cpu"

    def test_cleanup(self):
        ps = _make_predictor_with_mock_model()
        ps.cleanup()
        assert ps.model is None
        assert ps.is_loaded() is False

    def test_cleanup_when_no_model(self):
        ps = PredictorService()
        ps.cleanup()  # Should not raise

    def test_warmup_failure_non_critical(self):
        ps = PredictorService()
        ps.model = MagicMock(side_effect=RuntimeError("fail"))
        ps.device = "cpu"
        ps.transform = MagicMock(side_effect=RuntimeError("fail"))
        ps.warmup()  # Should not raise (non-critical)

    def test_warmup_before_load_model_is_noop(self):
        # If warmup is called on a fresh PredictorService (no model/transform),
        # it must early-return silently instead of NoneType-calling.
        ps = PredictorService()
        assert ps.model is None
        assert ps.transform is None
        ps.warmup()  # Must not raise
        assert ps.model is None  # State unchanged

    def test_predict_at_threshold_boundary_below(self):
        # sigmoid(0) = 0.5 exactly. predict() uses strict `prob > threshold`,
        # so prob == threshold (0.5) must classify as LAMINATED.
        ps = _make_predictor_with_mock_model(logit_value=0.0)
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        prediction, prob = ps.predict(img, "boundary.jpg")
        assert prob == 0.5
        assert prediction == "LAMINATED"

    def test_predict_just_above_threshold(self):
        # logit 0.001 -> sigmoid ~0.50025 > 0.5 -> UNLAMINATED.
        ps = _make_predictor_with_mock_model(logit_value=0.001)
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        prediction, prob = ps.predict(img, "boundary.jpg")
        assert prob > 0.5
        assert prediction == "UNLAMINATED"

    def test_predict_just_below_threshold(self):
        # logit -0.001 -> sigmoid ~0.49975 < 0.5 -> LAMINATED.
        ps = _make_predictor_with_mock_model(logit_value=-0.001)
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        prediction, prob = ps.predict(img, "boundary.jpg")
        assert prob < 0.5
        assert prediction == "LAMINATED"
