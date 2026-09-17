"""
UnlaminatedCopyDetector Model Architecture

MobileNetV3-small backbone with a custom 2-class classification head.
Mirrors the model produced by the model_training pipeline with config:
    model_name: mobilenet_v3_small
    pretrained: true
    num_classes: 2
    custom_head: { hidden_dim: 256, dropout: 0.2, use_batchnorm: true }

State-dict keys are exactly ``features.*`` and ``classifier.*`` so the
trained checkpoint loads with strict=True.
"""

import torch.nn as nn
import torchvision.models as tv_models


def _build_custom_head(in_features: int, num_classes: int = 2,
                       hidden_dim: int = 256, dropout: float = 0.2,
                       use_batchnorm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(in_features, hidden_dim))
    layers.append(nn.ReLU())
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(hidden_dim))
    if dropout > 0:
        layers.append(nn.Dropout(dropout * 0.5))
    layers.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*layers)


def UnlaminatedCopyDetector(num_classes: int = 2, hidden_dim: int = 256,
                            dropout: float = 0.2, use_batchnorm: bool = True
                            ) -> nn.Module:
    """
    Input:  3×224×224 RGB tensor (ImageNet-normalized)
    Output: 2-class logits — index 0 = LAMINATED, index 1 = UNLAMINATED
    """
    model = tv_models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = _build_custom_head(
        in_features, num_classes=num_classes,
        hidden_dim=hidden_dim, dropout=dropout,
        use_batchnorm=use_batchnorm,
    )
    return model
