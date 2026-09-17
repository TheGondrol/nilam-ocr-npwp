"""PP-OCRv5 server Paddle-to-PyTorch conversion helpers.

Single conversion path: Paddle PIR inference model (``.pdiparams`` +
``.json``) → PyTorch ``.pth``. The PP-OCRv5 server checkpoints we ship are
all PIR-format, and the shape-bucket matcher in ``_convert_pir_to_pytorch``
has been verified to reproduce Paddle's outputs to FP32 noise (see
``scripts/verify_conversion.py``).
"""

from __future__ import annotations

import copy
import logging
import re
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml

logger = logging.getLogger(__name__)


DET_YAML_RELATIVE = Path("configs/det/PP-OCRv5/PP-OCRv5_server_det.yml")
REC_YAML_RELATIVE = Path("configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml")
DEFAULT_DICT_RELATIVE = Path("pytorchocr/utils/dict/ppocrv5_dict.txt")
Component = Literal["det", "rec"]


@dataclass(frozen=True)
class PPOCRv5ServerConversionConfig:
    """Resolved inputs and outputs for PP-OCRv5 server weight conversion."""

    ppocr_root: Path
    det_source_path: Path | None
    rec_source_path: Path | None
    det_output_path: Path
    rec_output_path: Path
    components: tuple[Component, ...] = ("det", "rec")
    force: bool = False


def prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def ensure_ppocr_root(ppocr_root: str | Path) -> Path:
    root = Path(ppocr_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"PaddleOCR2Pytorch root not found: {root}")
    prepend_sys_path(root)
    return root


def resolve_optional_path(path_value: str | Path | None, base_dir: str | Path) -> Path | None:
    if path_value is None:
        return None
    value = str(path_value).strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base_dir).expanduser().resolve() / path
    return path.resolve()


def infer_paddle_source_from_model_dir(model_dir: str | Path | None, base_dir: str | Path) -> Path | None:
    model_path = resolve_optional_path(model_dir, base_dir)
    if model_path is None:
        return None
    if model_path.is_dir() or model_path.suffix != ".pdiparams":
        return model_path / "inference.pdiparams"
    return model_path


def load_paddlex_config_model_dir(config_path: str | Path, submodule_name: str) -> str | None:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return (
        data.get("SubModules", {})
        .get(submodule_name, {})
        .get("model_dir")
    )


def _load_yaml(ppocr_root: Path, relative_path: Path) -> dict[str, Any]:
    path = ppocr_root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"PP-OCRv5 YAML not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_ppocr_path(ppocr_root: Path, path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    cleaned = path_str[2:] if path_str.startswith("./") else path_str
    return ppocr_root / cleaned


def _character_count(ppocr_root: Path, global_config: dict[str, Any]) -> int:
    dict_path = _resolve_ppocr_path(ppocr_root, global_config["character_dict_path"])
    if not dict_path.exists():
        fallback = ppocr_root / DEFAULT_DICT_RELATIVE
        dict_path = fallback if fallback.exists() else dict_path
    if not dict_path.exists():
        raise FileNotFoundError(f"PP-OCRv5 character dictionary not found: {dict_path}")

    characters = []
    with dict_path.open("rb") as handle:
        for line in handle.readlines():
            characters.append(line.decode("utf-8").strip("\n").strip("\r"))
    if global_config.get("use_space_char", False):
        characters.append(" ")
    return len(characters) + 1


def server_det_architecture(ppocr_root: Path) -> dict[str, Any]:
    config = _load_yaml(ppocr_root, DET_YAML_RELATIVE)
    return copy.deepcopy(config["Architecture"])


def server_rec_architecture(ppocr_root: Path) -> dict[str, Any]:
    config = _load_yaml(ppocr_root, REC_YAML_RELATIVE)
    architecture = copy.deepcopy(config["Architecture"])
    char_num = _character_count(ppocr_root, config["Global"])
    architecture["Head"]["out_channels_list"] = {
        "CTCLabelDecode": char_num,
        "SARLabelDecode": char_num + 2,
        "NRTRLabelDecode": char_num + 3,
    }
    return architecture


def _require_source(path: Path | None, label: str) -> Path:
    if path is None:
        raise FileNotFoundError(
            f"Missing PP-OCRv5 {label} Paddle checkpoint. Set the source path "
            f"in config.yaml, pass --{label}-src, or set "
            f"AUTOKERNEL_PPOCRV5_SERVER_{label.upper()}_PDPARAMS."
        )
    if not path.exists():
        raise FileNotFoundError(f"PP-OCRv5 {label} Paddle checkpoint not found: {path}")
    return path


# Paddle PIR internal suffix → (PyTorch param name, needs_transpose)
_PIR_BN_MAP = {
    "w_0": ("weight", False),
    "b_0": ("bias", False),
    "w_1": ("running_mean", False),
    "w_2": ("running_var", False),
}
_PIR_CONV_MAP = {
    "w_0": ("weight", False),
    "b_0": ("bias", False),
}
_PIR_LINEAR_MAP = {
    "w_0": ("weight", True),  # Paddle [in, out] → PyTorch [out, in]
    "b_0": ("bias", False),
}
_PIR_PARAM_MAP: dict[str, dict[str, tuple[str, bool]]] = {
    "conv2d": _PIR_CONV_MAP,
    "conv2d_transpose": _PIR_CONV_MAP,
    "batch_norm2d": _PIR_BN_MAP,
    "batch_norm": _PIR_BN_MAP,
    "linear": _PIR_LINEAR_MAP,
    "layer_norm": {"w_0": ("weight", False), "b_0": ("bias", False)},
}

_TORCH_TO_PADDLE_TYPE: dict[type, str] = {}


def _torch_to_paddle_type() -> dict[type, str]:
    """Lazy import torch types for mapping."""
    if not _TORCH_TO_PADDLE_TYPE:
        import torch.nn as nn

        _TORCH_TO_PADDLE_TYPE.update(
            {
                nn.Conv2d: "conv2d",
                nn.ConvTranspose2d: "conv2d_transpose",
                nn.BatchNorm2d: "batch_norm2d",
                nn.BatchNorm1d: "batch_norm",
                nn.Linear: "linear",
                nn.LayerNorm: "layer_norm",
            }
        )
    return _TORCH_TO_PADDLE_TYPE


# PyTorch op types that are semantically equivalent to Paddle's generic
# `batch_norm` label (Paddle PIR export sometimes uses it for 2D BN).
_PIR_TYPE_EQUIV = {"batch_norm": "batch_norm2d"}

# PyTorch submodule name patterns that are training-only (pruned from the
# PIR inference model by Paddle's export). Leaving these in the matching
# pool shifts shape-bucket entries onto the wrong Paddle op.
#   - head.thresh.*     →  DB detector's threshold branch (training only)
#   - backbone.last_conv →  PPHGNetV2 classification tail (inert under det=True)
_PIR_INFERENCE_DEAD_NAME_PATTERNS = (".thresh.", "backbone.last_conv")


def _is_inference_dead_module_name(name: str) -> bool:
    for pat in _PIR_INFERENCE_DEAD_NAME_PATTERNS:
        if pat in name or name.endswith(pat.lstrip(".")):
            return True
    return False


def _shape_key(pd_type: str, suffixes: dict[str, Any]) -> tuple:
    """Canonical shape-tuple for a Paddle op's (weight or bias) params."""
    if "w_0" in suffixes:
        shape = list(suffixes["w_0"].shape)
        if pd_type == "linear" and len(shape) == 2:
            shape = [shape[1], shape[0]]
        return tuple(shape)
    if "b_0" in suffixes:
        return tuple(list(suffixes["b_0"].shape))
    return ()


def _pt_shape_key(module: Any) -> tuple:
    if hasattr(module, "weight") and module.weight is not None:
        return tuple(module.weight.shape)
    if hasattr(module, "bias") and module.bias is not None:
        return tuple(module.bias.shape)
    return ()


def _convert_pir_to_pytorch(
    source: Path, architecture: dict[str, Any], destination: Path
) -> None:
    """Convert a PIR inference model directly to a PyTorch ``.pth`` file.

    Matches Paddle PIR params to PyTorch modules by a shape-bucket algorithm:
    group both sides by ``(canonical_op_type, tensor_shape)`` and zip each
    bucket's Paddle ops (sorted by numeric index → forward-graph order) with
    PyTorch modules (``named_modules()`` construction order). Training-only
    PyTorch submodules (DB thresh branch, classification tail) are removed
    from the pool so their construction-order slots don't bump later
    same-shape modules onto the wrong Paddle op.

    Bucketing is required because LKPAN ops are emitted interleaved by
    forward pass while PyTorch ``named_modules()`` groups them by submodule;
    a positional-greedy zip mis-pairs ~34% of the det weights.
    """
    import paddle
    import torch
    from pytorchocr.base_ocr_v20 import BaseOCRV20

    class _Model(BaseOCRV20):
        def __init__(self, config: dict[str, Any]) -> None:
            super().__init__(config)

    model = _Model(architecture)
    type_map = _torch_to_paddle_type()

    # Collect PyTorch modules by canonical Paddle op type, skipping names
    # known to be training-only (not present in the PIR export).
    pt_by_type: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    skipped_training_only: list[str] = []
    for name, module in model.net.named_modules():
        for torch_type, paddle_type in type_map.items():
            if isinstance(module, torch_type):
                if _is_inference_dead_module_name(name):
                    skipped_training_only.append(name)
                else:
                    pt_by_type[paddle_type].append((name, module))
                break

    # Load Paddle PIR model and group params by (op_type, op_index).
    paddle.disable_static()
    layer = paddle.jit.load(str(source.with_suffix("")))
    pd_params: dict[tuple[str, int], dict[str, Any]] = {}
    pd_op_indices: dict[str, set[int]] = defaultdict(set)
    for p in layer.parameters():
        match = re.match(r"^(.+?)_(\d+)\.(.+?)_\d+$", p.name)
        if match:
            op_type, idx, suffix = match.group(1), int(match.group(2)), match.group(3)
            pd_params.setdefault((op_type, idx), {})[suffix] = p
            pd_op_indices[op_type].add(idx)

    pd_sorted = {t: sorted(indices) for t, indices in pd_op_indices.items()}

    # Merge equivalent op types (e.g. batch_norm → batch_norm2d) so buckets line up.
    pd_by_canonical: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for pd_type, pd_indices in pd_sorted.items():
        canonical = _PIR_TYPE_EQUIV.get(pd_type, pd_type)
        for idx in pd_indices:
            pd_by_canonical[canonical].append((idx, pd_type))

    new_sd: OrderedDict[str, torch.Tensor] = OrderedDict()
    unmatched_pd: list[tuple[str, int]] = []
    unmatched_pt: list[str] = []

    for canonical, entries in pd_by_canonical.items():
        suffix_map = _PIR_PARAM_MAP.get(canonical)
        if suffix_map is None:
            unmatched_pd.extend((t, i) for i, t in entries)
            continue

        # Bucket Paddle ops by canonical shape.
        pd_buckets: dict[tuple, list[tuple[int, str]]] = defaultdict(list)
        for pd_idx, pd_type in entries:
            key = _shape_key(pd_type, pd_params.get((pd_type, pd_idx), {}))
            pd_buckets[key].append((pd_idx, pd_type))

        # Bucket PyTorch modules by shape.
        pt_buckets: dict[tuple, list[tuple[str, Any]]] = defaultdict(list)
        for pt_name, pt_module in pt_by_type.get(canonical, []):
            pt_buckets[_pt_shape_key(pt_module)].append((pt_name, pt_module))

        for shape, pd_list in pd_buckets.items():
            pt_list = pt_buckets.get(shape, [])
            n = min(len(pd_list), len(pt_list))
            for (pd_idx, pd_type), (pt_name, _pt_module) in zip(pd_list[:n], pt_list[:n]):
                suffixes = pd_params.get((pd_type, pd_idx), {})
                for pd_suffix, pd_param in suffixes.items():
                    mapping = suffix_map.get(pd_suffix)
                    if mapping is None:
                        continue
                    pt_param_name, needs_transpose = mapping
                    full_key = f"{pt_name}.{pt_param_name}"
                    arr = pd_param.numpy()
                    if needs_transpose and len(arr.shape) == 2:
                        arr = arr.T
                    new_sd[full_key] = torch.from_numpy(arr.copy())
            for pd_idx, pd_type in pd_list[n:]:
                unmatched_pd.append((pd_type, pd_idx))
            for pt_name, _ in pt_list[n:]:
                unmatched_pt.append(pt_name)

    model.net.load_state_dict(new_sd, strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save_pytorch_weights(str(destination))

    logger.info(
        "PIR conversion matched %d parameters; %d training-only PT modules skipped; "
        "%d Paddle ops unmatched; %d PT modules unmatched",
        len(new_sd),
        len(skipped_training_only),
        len(unmatched_pd),
        len(unmatched_pt),
    )
    if unmatched_pd:
        logger.warning("Unmatched Paddle ops: %s", unmatched_pd[:20])
    if unmatched_pt:
        logger.warning("Unmatched PyTorch modules: %s", unmatched_pt[:20])


def _convert_detector(ppocr_root: Path, source: Path, destination: Path) -> None:
    _convert_pir_to_pytorch(source, server_det_architecture(ppocr_root), destination)


def _convert_recognizer(ppocr_root: Path, source: Path, destination: Path) -> None:
    _convert_pir_to_pytorch(source, server_rec_architecture(ppocr_root), destination)


def convert_ppocrv5_server_weights(
    config: PPOCRv5ServerConversionConfig,
) -> dict[str, Path]:
    """Convert selected PP-OCRv5 server Paddle checkpoints into `.pth` files."""

    ppocr_root = ensure_ppocr_root(config.ppocr_root)
    written: dict[str, Path] = {}

    if "det" in config.components:
        if config.det_output_path.exists() and not config.force:
            logger.info("Detector weights already exist: %s", config.det_output_path)
        else:
            source = _require_source(config.det_source_path, "det")
            logger.info("Converting PP-OCRv5 server detector: %s -> %s", source, config.det_output_path)
            _convert_detector(ppocr_root, source, config.det_output_path)
        written["det"] = config.det_output_path

    if "rec" in config.components:
        if config.rec_output_path.exists() and not config.force:
            logger.info("Recognizer weights already exist: %s", config.rec_output_path)
        else:
            source = _require_source(config.rec_source_path, "rec")
            logger.info("Converting PP-OCRv5 server recognizer: %s -> %s", source, config.rec_output_path)
            _convert_recognizer(ppocr_root, source, config.rec_output_path)
        written["rec"] = config.rec_output_path

    return written


def ensure_ppocrv5_server_weights(
    config: PPOCRv5ServerConversionConfig,
    required_components: Iterable[Component] = ("det", "rec"),
) -> dict[str, Path]:
    """Convert only missing required outputs and return the output paths."""

    missing_components: list[Component] = []
    for component in required_components:
        output_path = config.det_output_path if component == "det" else config.rec_output_path
        if config.force or not output_path.exists():
            missing_components.append(component)

    if not missing_components:
        return {
            "det": config.det_output_path,
            "rec": config.rec_output_path,
        }

    conversion_config = PPOCRv5ServerConversionConfig(
        ppocr_root=config.ppocr_root,
        det_source_path=config.det_source_path,
        rec_source_path=config.rec_source_path,
        det_output_path=config.det_output_path,
        rec_output_path=config.rec_output_path,
        components=tuple(missing_components),
        force=config.force,
    )
    return convert_ppocrv5_server_weights(conversion_config)
