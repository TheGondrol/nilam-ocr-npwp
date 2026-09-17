"""
ResNet-50 Model Architecture for Screen Recapture Detection
"""

from functools import partial
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import models, transforms

from ..core.config import config
from ..core.logging import logger


def resize_with_padding(img: Image.Image, size: int = 224, fill: int = 255) -> Image.Image:
    """Aspect-preserving resize to (size, size) with constant-value padding.

    Matches the training pipeline's ``resize_mode: pad`` step exactly — the same
    function lives at ``model_training/shared/utils/class_specific_augmentation.py:
    resize_with_padding``. The image is scaled so its longest side equals ``size``,
    then the shorter side is padded equally with ``fill`` (default 255 = white).

    Keeping a local copy avoids a runtime dependency on the training repo.
    """
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = TF.resize(img, (new_h, new_w))
    pad_w = size - new_w
    pad_h = size - new_h
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    return TF.pad(img, padding, fill=fill)


def create_resnet50_model(num_classes: int = 2) -> nn.Module:
    """
    Create ResNet-50 model for screen recapture detection.
    
    IMPORTANT: This model was trained with:
    - Frozen backbone EXCEPT layer3 + layer4 (last 2 residual blocks)
    - Trainable FC head (2048 → 2 classes)
    - ImageNet pretrained weights as initialization
    
    Input: RGB image 224×224 (center cropped from 512px normalized)
    Output: Binary classification (alphabetical ImageFolder ordering)
        - Class 0: `good` folder (ORIGINAL - legitimate capture, normal)
        - Class 1: `recaptured` folder (RECAPTURED - screen photo, suspicious)
    
    Args:
        num_classes: Number of output classes (default: 2)
    
    Returns:
        ResNet-50 model with custom classifier head
    
    Note: For inference, we load the full architecture (no freezing needed since
    we're in eval mode). The trained weights already contain the learned parameters
    from the unfrozen layers (layer3, layer4, FC).
    """
    # Load ResNet-50 architecture (no pretrained weights, will load trained checkpoint)
    model = models.resnet50(weights=None)
    
    # Replace final fully connected layer for binary classification
    in_features = model.fc.in_features  # 2048 for ResNet-50
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


def load_model(
    model_path: str,
    device: torch.device,
    num_classes: int = 2
) -> nn.Module:
    """
    Load trained ResNet-50 model from checkpoint
    
    Args:
        model_path: Path to model weights file (.pth)
        device: Device to load model on (cuda or cpu)
        num_classes: Number of output classes
        
    Returns:
        Loaded model in eval mode
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    model_file = Path(model_path)
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        # Create model architecture
        model = create_resnet50_model(num_classes=num_classes)
        
        # Load weights with map_location for CPU compatibility. Prefer
        # weights_only=True and allowlist numpy's data-only array globals so a
        # trusted checkpoint that embeds numpy arrays still loads; if it holds
        # other non-tensor objects weights_only can't handle, fall back to a full
        # load of the (bucket-sourced, trusted) artifact (BUG-06).
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
        model.load_state_dict(state_dict)
        
        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Model device: {device}")
        
        # Apply device-specific model optimization
        if config.get('model.use_compile', False):
            try:
                if device.type == "cuda":
                    # Use torch.compile for GPU - better runtime optimization
                    compile_mode = config.get('model.compile_mode', 'reduce-overhead')
                    model = torch.compile(model, mode=compile_mode)
                    logger.info(f"Model compiled with torch.compile (mode='{compile_mode}') for GPU")
                else:
                    # Use JIT tracing for CPU - better static graph optimization
                    crop_size = config.get('model.crop_size', 224)
                    dummy_input = torch.randn(1, 3, crop_size, crop_size).to(device)
                    model = torch.jit.trace(model, dummy_input)
                    model = torch.jit.optimize_for_inference(model)
                    logger.info("Model optimized with torch.jit.trace for CPU inference")
            except Exception as e:
                logger.warning(f"Model optimization failed, using eager mode: {str(e)}")

        return cast(nn.Module, model)
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")


def get_transform(crop_size: int = 224) -> transforms.Compose:
    """
    Get image preprocessing transform pipeline.

    Must match training preprocessing in
    ``model_training/shared/utils/data_loader.py:get_transforms``. The resize
    strategy is config-driven via ``model.resize_mode`` (default ``pad``):

    - ``pad`` (current default): aspect-preserving resize so longest side
      equals ``crop_size``, then pad the shorter side with ``model.pad_fill``
      (default 255 = white). The full image is preserved — no crop, no
      aspect-ratio distortion. This is what the latest pad-mode training
      checkpoints expect.
    - ``crop``: resize short side to ``crop_size * 1.14`` and center-crop
      ``crop_size``. Matches the historical no-augment training branch.
    - ``squash``: resize directly to ``(crop_size, crop_size)`` (distorts
      aspect ratio). Matches the older val-transform behaviour.

    The trailing ToTensor + ImageNet-normalize stage is identical across modes
    and identical to training.
    """
    normalize_mean = config.get('model.normalize_mean', [0.485, 0.456, 0.406])
    normalize_std = config.get('model.normalize_std', [0.229, 0.224, 0.225])
    resize_mode = str(config.get('model.resize_mode', 'pad')).lower()
    pad_fill = int(config.get('model.pad_fill', 255))

    if resize_mode == 'pad':
        resize_step = transforms.Lambda(
            partial(resize_with_padding, size=crop_size, fill=pad_fill)
        )
    elif resize_mode == 'crop':
        resize_step = transforms.Compose([
            transforms.Resize(int(crop_size * 1.14)),
            transforms.CenterCrop(crop_size),
        ])
    elif resize_mode == 'squash':
        resize_step = transforms.Resize((crop_size, crop_size))
    else:
        raise ValueError(
            f"Unknown model.resize_mode={resize_mode!r}; expected one of "
            "'pad' | 'crop' | 'squash'"
        )

    logger.info(f"Image preprocessing: resize_mode={resize_mode}, crop_size={crop_size}")

    return transforms.Compose([
        resize_step,
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])
