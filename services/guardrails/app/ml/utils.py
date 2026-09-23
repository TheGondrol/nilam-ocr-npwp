"""Preprocessing shared by the in-process classifiers."""

from typing import Any

from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_to_tensor(image: Image.Image, image_size: int) -> Any:
    """RGB, resized to the square the network was trained on, scaled to [0, 1] and normalised with the
    ImageNet statistics; returns a CHW float tensor."""
    import numpy as np
    import torch

    resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std
