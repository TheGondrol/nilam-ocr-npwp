"""MobileNet model loading and preprocessing."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms

from src.core.config import config
import logging

logger = logging.getLogger("quality")


def load_mobilenet_model(
    model_path: str,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """
    Load pretrained MobileNetV2 model with custom classifier.

    Args:
        model_path: Path to model weights file
        num_classes: Number of output classes
        device: Device to load model on (cuda/cpu)

    Returns:
        Loaded PyTorch model

    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading MobileNetV2 model from {model_path}")

    # Load MobileNetV2 architecture
    model = models.mobilenet_v2(weights=None)

    # Modify classifier head for binary classification
    # MobileNetV2 has classifier with in_features=1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(model.last_channel, num_classes),
    )

    # Load trained weights
    try:
        # Try loading as TorchScript archive first
        try:
            jit_model = torch.jit.load(model_path, map_location=device)
            logger.info("Model loaded as TorchScript archive successfully")
            jit_model.eval()
            return jit_model
        except Exception:
            pass

        # Fall back to regular checkpoint loading. Prefer weights_only=True (no
        # arbitrary-object unpickling) and allowlist numpy's data-only array
        # globals so a trusted checkpoint that embeds numpy arrays still loads;
        # if it holds other non-tensor objects weights_only can't handle, fall
        # back to a full load of the (bucket-sourced, trusted) artifact (BUG-06).
        try:
            import numpy as _np
            import torch.serialization as _ts
            _allow = [_np.ndarray, _np.dtype]
            for _mod_name in ("numpy._core.multiarray", "numpy.core.multiarray"):
                try:
                    _ma = __import__(_mod_name, fromlist=["_reconstruct", "scalar"])
                    _allow += [g for g in (getattr(_ma, "_reconstruct", None),
                                           getattr(_ma, "scalar", None)) if g is not None]
                    break
                except Exception:
                    continue
            _ts.add_safe_globals(_allow)
        except Exception:
            pass
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except Exception as _wo_err:
            logger.warning(
                "weights_only load rejected the checkpoint (%s); falling back to a "
                "full load of the trusted artifact — keep the model bucket "
                "access-controlled and downloads over TLS", _wo_err)
            state_dict = torch.load(model_path, map_location=device, weights_only=False)

        # Handle different state dict formats
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        elif "state_dict" in state_dict:
            model.load_state_dict(state_dict["state_dict"])
        else:
            model.load_state_dict(state_dict)

        logger.info("Model weights loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")

    model = model.to(device)
    model.eval()

    # Optimize model based on device
    if device.type == "cuda" and config.use_compile:
        logger.info(f"Compiling model with mode: {config.compile_mode}")
        try:
            model = torch.compile(model, mode=config.compile_mode)
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Using uncompiled model.")
    elif device.type == "cpu":
        logger.info("Using TorchScript optimization for CPU")
        try:
            # Create dummy input for tracing (batch, channels, height, width)
            dummy_input = torch.randn(1, 3, config.img_height, config.img_width).to(device)
            model = torch.jit.trace(model, dummy_input)
            logger.info("Model traced with TorchScript successfully")
        except Exception as e:
            logger.warning(f"TorchScript tracing failed: {e}. Using standard model.")

    logger.info(f"Model loaded on device: {device}")
    return model


def get_transform() -> transforms.Compose:
    """
    Get image preprocessing transform pipeline.

    Returns:
        Composed transforms for preprocessing crops
    """
    img_height = config.img_height
    img_width = config.img_width
    mean = config.normalize_mean
    std = config.normalize_std

    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),  # Resize to exact dimensions (H, W)
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    logger.info(f"Transform pipeline created: Resize({img_width}x{img_height}) -> Normalize")
    return transform


def warmup_model(
    model: nn.Module,
    device: torch.device,
    transform: transforms.Compose,
) -> None:
    """
    Warm up model with dummy inference to trigger compilation.

    Args:
        model: PyTorch model
        device: Device model is on
        transform: Preprocessing transforms
    """
    logger.info("Warming up model with dummy inference...")

    try:
        # Create dummy image with correct dimensions (width, height)
        from PIL import Image
        dummy_image = Image.new("RGB", (config.img_width, config.img_height), color=(128, 128, 128))

        # Preprocess
        dummy_tensor = transform(dummy_image).unsqueeze(0).to(device)

        # Run inference
        with torch.inference_mode():
            _ = model(dummy_tensor)

        logger.info("Model warmup completed successfully")
    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")
