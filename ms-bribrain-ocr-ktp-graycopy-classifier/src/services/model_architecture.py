"""
ResNet-50 Model Architecture for Graycopy Detection
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet50GraycopyDetector(nn.Module):
    """
    ResNet-50 model for graycopy/photocopy detection.
    
    This model detects whether an ID document image is:
    - Class 0: GRAYCOPY - Photocopied or scanned document (suspicious)
    - Class 1: ORIGINAL - Real/legitimate document
    
    Note: Class indices are determined by alphabetically sorted directory names.
    
    Architecture:
    - Backbone: ResNet-50 pretrained on ImageNet
    - Custom FC head with dropout, batch normalization
    - Input: RGB image 224×224
    - Output: Binary classification (graycopy vs original)
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.6):
        super().__init__()
        # Load base ResNet-50 and copy layers directly (no wrapper)
        base_model = models.resnet50(weights=None)
        
        # Copy all ResNet layers directly to self
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        
        # Replace FC layer with custom head
        num_features = base_model.fc.in_features
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard ResNet-50 forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def create_resnet50_graycopy_model(
    num_classes: int = 2,
    dropout_rate: float = 0.6
) -> ResNet50GraycopyDetector:
    """
    Create ResNet-50 model for graycopy/photocopy detection.
    
    Args:
        num_classes: Number of output classes (default: 2)
        dropout_rate: Dropout rate for regularization (default: 0.6)
    
    Returns:
        ResNet50GraycopyDetector model instance
    """
    return ResNet50GraycopyDetector(num_classes=num_classes, dropout_rate=dropout_rate)
