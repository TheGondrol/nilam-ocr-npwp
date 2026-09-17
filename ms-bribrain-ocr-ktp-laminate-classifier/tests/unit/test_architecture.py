"""Unit tests for src.models.architecture module."""

import torch
import torch.nn as nn

from src.models.architecture import UnlaminatedCopyDetector


class TestUnlaminatedCopyDetector:
    def test_is_nn_module(self):
        model = UnlaminatedCopyDetector()
        assert isinstance(model, nn.Module)

    def test_mobilenetv3_backbone(self):
        # MobileNetV3-small backbone exposes `features` and `classifier`.
        model = UnlaminatedCopyDetector()
        assert hasattr(model, "features")
        assert hasattr(model, "classifier")

    def test_input_channels_rgb(self):
        # First conv consumes 3-channel RGB input.
        model = UnlaminatedCopyDetector()
        first_conv = model.features[0][0]
        assert isinstance(first_conv, nn.Conv2d)
        assert first_conv.in_channels == 3

    def test_custom_head_outputs_two_classes(self):
        # classifier ends in the custom head whose final Linear emits 2 logits.
        model = UnlaminatedCopyDetector()
        head = model.classifier[-1]
        assert isinstance(head, nn.Sequential)
        last_linear = [m for m in head if isinstance(m, nn.Linear)][-1]
        assert last_linear.out_features == 2

    def test_head_has_dropout(self):
        model = UnlaminatedCopyDetector()
        head = model.classifier[-1]
        assert any(isinstance(m, nn.Dropout) for m in head)

    def test_forward_output_shape(self):
        # 3x224x224 RGB input -> (N, 2) logits.
        model = UnlaminatedCopyDetector().eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 2)
