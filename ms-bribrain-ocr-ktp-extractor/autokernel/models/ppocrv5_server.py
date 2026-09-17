"""
PP-OCRv5 server detection and recognition wrappers for AutoKernel.

These classes keep AutoKernel's existing model-loading contract:
  - no-arg constructors
  - a single tensor input
  - plain tensor outputs

Weights are expected to be prepared ahead of time with
`scripts/prepare_ppocrv5_server.py`.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
AUTOKERNEL_ROOT = SCRIPT_DIR.parent
WORKSPACE_DIR = AUTOKERNEL_ROOT / "workspace" / "ppocrv5"

DET_YAML_RELATIVE = Path("configs/det/PP-OCRv5/PP-OCRv5_server_det.yml")
REC_YAML_RELATIVE = Path("configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml")

DET_PTH_ENV = "AUTOKERNEL_PPOCRV5_SERVER_DET_PTH"
REC_PTH_ENV = "AUTOKERNEL_PPOCRV5_SERVER_REC_PTH"


def get_ppocr_root() -> Path:
    """Resolve the sibling PaddleOCR2Pytorch checkout."""
    env_root = os.getenv("AUTOKERNEL_PPOCR_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
    else:
        root = (AUTOKERNEL_ROOT.parent / "PaddleOCR2Pytorch").resolve()

    if not root.exists():
        raise FileNotFoundError(
            "Could not locate PaddleOCR2Pytorch. Set AUTOKERNEL_PPOCR_ROOT "
            f"or place the repo at {root}."
        )
    return root


def ensure_ppocr_on_path() -> Path:
    """Make PaddleOCR2Pytorch importable for sibling-repo integration."""
    _ensure_stdlib_profile_module()
    ppocr_root = get_ppocr_root()
    ppocr_root_str = str(ppocr_root)
    if ppocr_root_str not in sys.path:
        sys.path.insert(0, ppocr_root_str)
    return ppocr_root


def default_server_det_weights_path() -> Path:
    return WORKSPACE_DIR / "server_det.pth"


def default_server_rec_weights_path() -> Path:
    return WORKSPACE_DIR / "server_rec.pth"


def _ppocr_path(relative_path: Path) -> Path:
    return ensure_ppocr_on_path() / relative_path


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    cleaned = path_str[2:] if path_str.startswith("./") else path_str
    return ensure_ppocr_on_path() / cleaned


def _load_yaml(relative_path: Path) -> Dict[str, Any]:
    yaml_path = _ppocr_path(relative_path)
    with yaml_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_state_dict(weights_path: Path) -> Dict[str, torch.Tensor]:
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing weights at {weights_path}. Run "
            "scripts/prepare_ppocrv5_server.py or set the relevant "
            "AUTOKERNEL_PPOCRV5_SERVER_*_PTH environment variable."
        )
    state_dict = torch.load(weights_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unexpected weight format in {weights_path}: {type(state_dict)}")
    return state_dict


def _ensure_stdlib_profile_module() -> None:
    """No-op — profile.py was renamed to ak_profile.py to avoid shadowing."""
    pass


def _import_base_model():
    ensure_ppocr_on_path()
    from pytorchocr.modeling.architectures.base_model import BaseModel

    return BaseModel


def _character_count(global_config: Dict[str, Any]) -> int:
    dict_path = _resolve_repo_path(global_config["character_dict_path"])
    if not dict_path.exists():
        raise FileNotFoundError(f"Character dictionary not found: {dict_path}")

    characters = []
    with dict_path.open("rb") as handle:
        for line in handle.readlines():
            characters.append(line.decode("utf-8").strip("\n").strip("\r"))

    if global_config.get("use_space_char", False):
        characters.append(" ")

    # CTC uses a leading blank token.
    return len(characters) + 1


def server_det_architecture() -> Dict[str, Any]:
    config = _load_yaml(DET_YAML_RELATIVE)
    return copy.deepcopy(config["Architecture"])


def server_rec_architecture() -> Tuple[Dict[str, Any], int]:
    config = _load_yaml(REC_YAML_RELATIVE)
    architecture = copy.deepcopy(config["Architecture"])
    char_num = _character_count(config["Global"])
    architecture["Head"]["out_channels_list"] = {
        "CTCLabelDecode": char_num,
        "SARLabelDecode": char_num + 2,
        "NRTRLabelDecode": char_num + 3,
    }
    return architecture, char_num


def _build_base_model(architecture: Dict[str, Any], **kwargs: Any) -> nn.Module:
    base_model_cls = _import_base_model()
    return base_model_cls(copy.deepcopy(architecture), **kwargs)


def _build_server_det_net(weights_path: Path | None = None) -> nn.Module:
    path = Path(weights_path) if weights_path is not None else Path(
        os.getenv(DET_PTH_ENV, default_server_det_weights_path())
    )
    net = _build_base_model(server_det_architecture())
    net.load_state_dict(_load_state_dict(path))
    net.eval()
    return net


def _build_server_rec_net(weights_path: Path | None = None) -> nn.Module:
    path = Path(weights_path) if weights_path is not None else Path(
        os.getenv(REC_PTH_ENV, default_server_rec_weights_path())
    )
    architecture, char_num = server_rec_architecture()
    net = _build_base_model(architecture, out_channels=char_num)
    net.load_state_dict(_load_state_dict(path))
    net.eval()
    return net


class PPOCRv5ServerDetModel(nn.Module):
    """AutoKernel wrapper that exposes the server detector's shrink map."""

    def __init__(self) -> None:
        super().__init__()
        self.net = _build_server_det_net()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.net(x)
        if not isinstance(outputs, dict) or "maps" not in outputs:
            raise TypeError(f"Unexpected detector output type: {type(outputs)}")
        return outputs["maps"]


class PPOCRv5ServerRecModel(nn.Module):
    """AutoKernel wrapper that exposes the server recognizer's CTC output."""

    def __init__(self) -> None:
        super().__init__()
        self.net = _build_server_rec_net()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.net(x)
        if not isinstance(outputs, torch.Tensor):
            raise TypeError(f"Unexpected recognizer output type: {type(outputs)}")
        return outputs
