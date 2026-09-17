"""Unit tests for src.services.model_architecture module"""

import torch

from src.services.model_architecture import (
    ResNet50GraycopyDetector,
    create_resnet50_graycopy_model,
)


class TestResNet50GraycopyDetector:
    def test_default_initialization(self):
        model = ResNet50GraycopyDetector()
        assert isinstance(model, torch.nn.Module)

    def test_custom_classes(self):
        model = ResNet50GraycopyDetector(num_classes=3, dropout_rate=0.3)
        dummy = torch.randn(1, 3, 224, 224)
        model.eval()
        with torch.inference_mode():
            out = model(dummy)
        assert out.shape == (1, 3)

    def test_forward_default(self):
        model = ResNet50GraycopyDetector()
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.inference_mode():
            out = model(dummy)
        assert out.shape == (1, 2)

    def test_batch_forward(self):
        model = ResNet50GraycopyDetector()
        model.eval()
        dummy = torch.randn(4, 3, 224, 224)
        with torch.inference_mode():
            out = model(dummy)
        assert out.shape == (4, 2)


class TestCreateModel:
    def test_factory_function(self):
        model = create_resnet50_graycopy_model()
        assert isinstance(model, ResNet50GraycopyDetector)

    def test_factory_with_args(self):
        model = create_resnet50_graycopy_model(num_classes=5, dropout_rate=0.2)
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        with torch.inference_mode():
            out = model(dummy)
        assert out.shape == (1, 5)
