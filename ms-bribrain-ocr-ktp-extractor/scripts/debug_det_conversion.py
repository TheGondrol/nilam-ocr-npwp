#!/usr/bin/env python3
"""Run the det PIR conversion with heavy instrumentation to find coverage gaps.

Prints:
- Total PyTorch model parameter keys
- Total Paddle PIR parameters
- Matched count + coverage %
- List of PyTorch keys that would remain uninitialised (got random init)
- List of Paddle op indices that were not consumed
"""

from __future__ import annotations

import os
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
os.chdir(PROJECT_ROOT)
for _p in (PROJECT_ROOT, SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DET_SOURCE = PROJECT_ROOT / "src/models/server_models/ppocrv5_server_det_source/inference.pdiparams"
PPOCR_ROOT = PROJECT_ROOT / "PaddleOCR2Pytorch"

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
    "w_0": ("weight", True),
    "b_0": ("bias", False),
}
_PIR_PARAM_MAP = {
    "conv2d": _PIR_CONV_MAP,
    "conv2d_transpose": _PIR_CONV_MAP,
    "batch_norm2d": _PIR_BN_MAP,
    "batch_norm": _PIR_BN_MAP,
    "linear": _PIR_LINEAR_MAP,
    "layer_norm": {"w_0": ("weight", False), "b_0": ("bias", False)},
}


def main() -> int:
    sys.path.insert(0, str(PPOCR_ROOT))
    from pytorchocr.base_ocr_v20 import BaseOCRV20  # noqa: E402

    from src.services.ppocrv5_conversion import server_det_architecture  # noqa: E402

    import paddle
    import torch
    import torch.nn as nn

    arch = server_det_architecture(PPOCR_ROOT)

    class _Model(BaseOCRV20):
        def __init__(self, config):
            super().__init__(config)

    pt_model = _Model(arch)
    pt_sd_keys = set(pt_model.net.state_dict().keys())
    print(f"PyTorch model: {len(pt_sd_keys)} state-dict keys")

    type_map: dict[type, str] = {
        nn.Conv2d: "conv2d",
        nn.ConvTranspose2d: "conv2d_transpose",
        nn.BatchNorm2d: "batch_norm2d",
        nn.BatchNorm1d: "batch_norm",
        nn.Linear: "linear",
        nn.LayerNorm: "layer_norm",
    }
    pt_by_type: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for name, module in pt_model.net.named_modules():
        for torch_type, paddle_type in type_map.items():
            if isinstance(module, torch_type):
                pt_by_type[paddle_type].append((name, module))
                break
    for k, v in pt_by_type.items():
        print(f"  pt_by_type[{k}] = {len(v)} modules")

    # Load Paddle params.
    paddle.disable_static()
    layer = paddle.jit.load(str(DET_SOURCE.with_suffix("")))
    pd_params: dict[tuple[str, int], dict[str, Any]] = {}
    pd_op_indices: dict[str, set[int]] = defaultdict(set)
    for p in layer.parameters():
        match = re.match(r"^(.+?)_(\d+)\.(.+?)_\d+$", p.name)
        if match:
            op_type, idx, suffix = match.group(1), int(match.group(2)), match.group(3)
            pd_params.setdefault((op_type, idx), {})[suffix] = p
            pd_op_indices[op_type].add(idx)
    for k, v in pd_op_indices.items():
        print(f"  pd_op_indices[{k}] = {len(v)} ops")

    pd_sorted = {t: sorted(indices) for t, indices in pd_op_indices.items()}

    new_sd: OrderedDict[str, torch.Tensor] = OrderedDict()
    unmatched_pd: list[tuple[str, int]] = []
    skipped_pt: list[tuple[str, str, list]] = []  # (op_type, pt_name, pt_shape)

    for pd_type, pd_indices in pd_sorted.items():
        suffix_map = _PIR_PARAM_MAP.get(pd_type)
        if suffix_map is None:
            unmatched_pd.extend([(pd_type, idx) for idx in pd_indices])
            continue
        pt_modules = pt_by_type.get(pd_type, [])
        pd_cursor = 0
        for pt_name, pt_module in pt_modules:
            if pd_cursor >= len(pd_indices):
                skipped_pt.append(
                    (pd_type, pt_name, list(pt_module.weight.shape) if hasattr(pt_module, "weight") else [])
                )
                continue
            pd_idx = pd_indices[pd_cursor]
            suffixes = pd_params.get((pd_type, pd_idx), {})

            shapes_ok = True
            for pd_suffix, pd_param in suffixes.items():
                mapping = suffix_map.get(pd_suffix)
                if mapping is None:
                    continue
                pt_param_name, needs_transpose = mapping
                full_key = f"{pt_name}.{pt_param_name}"
                pt_tensor = pt_model.net.state_dict().get(full_key)
                if pt_tensor is None:
                    continue
                pd_shape = list(pd_param.shape)
                if needs_transpose and len(pd_shape) == 2:
                    pd_shape = [pd_shape[1], pd_shape[0]]
                if list(pt_tensor.shape) != pd_shape:
                    shapes_ok = False
                    break

            if shapes_ok:
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
                pd_cursor += 1
            else:
                skipped_pt.append(
                    (pd_type, pt_name, list(pt_module.weight.shape) if hasattr(pt_module, "weight") else [])
                )

        while pd_cursor < len(pd_indices):
            unmatched_pd.append((pd_type, pd_indices[pd_cursor]))
            pd_cursor += 1

    missing_pt_keys = pt_sd_keys - set(new_sd.keys())
    print(f"\n[Positional greedy]")
    print(f"Matched: {len(new_sd)} keys assigned into PyTorch state dict")
    print(f"Missing PyTorch keys (would stay random-init): {len(missing_pt_keys)}")
    print(f"Unmatched Paddle ops: {len(unmatched_pd)}")
    print(f"Skipped PyTorch modules (could not match Paddle by shape): {len(skipped_pt)}")

    # ------------------------------------------------------------------
    # Prototype: shape-bucket matching
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Shape-bucket matching prototype]")

    def shape_key(pd_type: str, suffixes: dict) -> tuple:
        # Prefer weight shape; fall back to bias shape or full suffix map shape.
        if "w_0" in suffixes:
            arr = suffixes["w_0"]
            shape = list(arr.shape)
            if pd_type == "linear" and len(shape) == 2:
                shape = [shape[1], shape[0]]
            return tuple(shape)
        if "b_0" in suffixes:
            return tuple(list(suffixes["b_0"].shape))
        return ()

    def pt_shape_key(pd_type: str, pt_module) -> tuple:
        if hasattr(pt_module, "weight") and pt_module.weight is not None:
            return tuple(pt_module.weight.shape)
        if hasattr(pt_module, "bias") and pt_module.bias is not None:
            return tuple(pt_module.bias.shape)
        return ()

    new_sd_bucket: OrderedDict[str, torch.Tensor] = OrderedDict()
    total_assigned = 0
    total_pd_remaining = 0
    total_pt_remaining = 0

    # Merge semantically-equivalent op types so buckets line up.
    # Paddle labels some 2D BNs generically as `batch_norm` — fold into `batch_norm2d`.
    type_equiv = {"batch_norm": "batch_norm2d"}

    # PyTorch module name patterns that are training-only (not exported to the
    # PIR inference model). Keeping them in the matching pool shifts later
    # shape-bucket entries onto wrong ops. Known training-dead branches:
    #   - head.thresh.*  →  DB detector's threshold branch
    #   - backbone.last_conv  →  PPHGNetV2 classification tail (pruned under det=True)
    skip_pt_patterns = (".thresh.", "backbone.last_conv")
    skip_pt_name = lambda n: any(p in n or n.endswith(p.lstrip(".")) for p in skip_pt_patterns)
    pt_by_type_filtered: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    skipped_training_only: list[str] = []
    for k, modules in pt_by_type.items():
        for pt_name, pt_module in modules:
            if skip_pt_name(pt_name):
                skipped_training_only.append(pt_name)
            else:
                pt_by_type_filtered[k].append((pt_name, pt_module))
    print(f"Skipped {len(skipped_training_only)} training-only PyTorch modules:")
    for n in skipped_training_only:
        print(f"  - {n}")
    pt_by_type = pt_by_type_filtered

    pd_canonical: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for pd_type, pd_indices in pd_sorted.items():
        canonical = type_equiv.get(pd_type, pd_type)
        for idx in pd_indices:
            pd_canonical[canonical].append((idx, pd_type))

    for canonical, entries in pd_canonical.items():
        suffix_map = _PIR_PARAM_MAP.get(canonical)
        if suffix_map is None:
            continue
        # Group Paddle ops by shape (across merged types)
        pd_buckets: dict[tuple, list[tuple[int, str]]] = defaultdict(list)
        for idx, original_type in entries:
            key = shape_key(original_type, pd_params.get((original_type, idx), {}))
            pd_buckets[key].append((idx, original_type))

        # Group PyTorch modules by shape
        pt_buckets: dict[tuple, list[tuple[str, Any]]] = defaultdict(list)
        for pt_name, pt_module in pt_by_type.get(canonical, []):
            key = pt_shape_key(canonical, pt_module)
            pt_buckets[key].append((pt_name, pt_module))

        for key, pd_list in pd_buckets.items():
            pt_list = pt_buckets.get(key, [])
            n = min(len(pd_list), len(pt_list))
            for (pd_idx, pd_type), (pt_name, pt_module) in zip(pd_list[:n], pt_list[:n]):
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
                    new_sd_bucket[full_key] = torch.from_numpy(arr.copy())
                total_assigned += 1
            total_pd_remaining += max(0, len(pd_list) - n)
            total_pt_remaining += max(0, len(pt_list) - n)

    missing_bucket = pt_sd_keys - set(new_sd_bucket.keys())
    print(f"Assigned: {total_assigned} Paddle ops")
    print(f"Paddle ops unmatched (no shape-matching pt bucket): {total_pd_remaining}")
    print(f"PyTorch modules unmatched (no shape-matching pd bucket): {total_pt_remaining}")
    print(f"Keys assigned into new_sd_bucket: {len(new_sd_bucket)}")
    print(f"Missing PyTorch keys (random-init): {len(missing_bucket)}")
    real_missing = sorted(k for k in missing_bucket if "num_batches_tracked" not in k)
    print(f"Real missing count (excluding num_batches_tracked): {len(real_missing)}")
    print("\nAll real missing PyTorch keys:")
    for k in real_missing:
        print(f"  {k}: shape={list(pt_model.net.state_dict()[k].shape)}")

    # Show paddle ops with shape buckets that have NO pt match
    print("\nPaddle ops with no pt bucket match:")
    for pd_type in pd_sorted:
        suffix_map = _PIR_PARAM_MAP.get(pd_type)
        if suffix_map is None:
            continue
        pd_buckets_2: dict[tuple, list[int]] = defaultdict(list)
        for idx in pd_sorted[pd_type]:
            key = shape_key(pd_type, pd_params.get((pd_type, idx), {}))
            pd_buckets_2[key].append(idx)
        pt_buckets_2: dict[tuple, int] = defaultdict(int)
        for pt_name, pt_module in pt_by_type.get(pd_type, []):
            pt_buckets_2[pt_shape_key(pd_type, pt_module)] += 1
        for key, pd_list in pd_buckets_2.items():
            pt_count = pt_buckets_2.get(key, 0)
            if len(pd_list) > pt_count:
                print(f"  [{pd_type}] shape={key}: {len(pd_list)} paddle, {pt_count} pt "
                      f"(paddle ops: {pd_list})")

    print("\nPyTorch modules with no pd bucket match:")
    for pd_type in pt_by_type:
        suffix_map = _PIR_PARAM_MAP.get(pd_type)
        if suffix_map is None:
            continue
        pd_buckets_2: dict[tuple, int] = defaultdict(int)
        for idx in pd_sorted.get(pd_type, []):
            key = shape_key(pd_type, pd_params.get((pd_type, idx), {}))
            pd_buckets_2[key] += 1
        pt_buckets_2: dict[tuple, list[str]] = defaultdict(list)
        for pt_name, pt_module in pt_by_type.get(pd_type, []):
            pt_buckets_2[pt_shape_key(pd_type, pt_module)].append(pt_name)
        for key, pt_list in pt_buckets_2.items():
            pd_count = pd_buckets_2.get(key, 0)
            if len(pt_list) > pd_count:
                print(f"  [{pd_type}] shape={key}: {pd_count} paddle, {len(pt_list)} pt "
                      f"(pt modules: {pt_list})")

    print("\n--- First 30 missing PyTorch keys ---")
    for k in sorted(missing_pt_keys)[:30]:
        print(f"  {k}: shape={list(pt_model.net.state_dict()[k].shape)}")

    print("\n--- First 30 unmatched Paddle ops (op, idx) ---")
    for op_type, idx in unmatched_pd[:30]:
        suffixes = pd_params.get((op_type, idx), {})
        shapes = {k: list(v.shape) for k, v in suffixes.items()}
        print(f"  {op_type}_{idx}: {shapes}")

    print("\n--- First 30 skipped PyTorch modules ---")
    for op_type, pt_name, shape in skipped_pt[:30]:
        print(f"  [{op_type}] {pt_name}: shape={shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
