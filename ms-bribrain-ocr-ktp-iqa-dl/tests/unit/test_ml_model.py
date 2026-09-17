"""Unit tests for src.models.ml_model module."""

from unittest.mock import patch, MagicMock

import pytest
import torch

from src.models.ml_model import get_transform, warmup_model


class TestGetTransform:
    def test_returns_compose(self):
        from torchvision import transforms
        t = get_transform()
        assert isinstance(t, transforms.Compose)


class TestWarmupModel:
    def test_warmup_success(self):
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.5, 0.5]])
        device = torch.device("cpu")
        # Use a mock transform to avoid torchvision crash
        mock_transform = MagicMock(return_value=torch.randn(3, 64, 320))
        warmup_model(mock_model, device, mock_transform)
        mock_model.assert_called_once()

    def test_warmup_handles_error(self):
        mock_model = MagicMock(side_effect=RuntimeError("fail"))
        device = torch.device("cpu")
        mock_transform = MagicMock(return_value=torch.randn(3, 64, 320))
        # Should not raise
        warmup_model(mock_model, device, mock_transform)


class TestLoadMobilenetModel:
    def test_file_not_found(self):
        from src.models.ml_model import load_mobilenet_model
        with pytest.raises(FileNotFoundError):
            load_mobilenet_model("/nonexistent/model.pth", 2, torch.device("cpu"))
