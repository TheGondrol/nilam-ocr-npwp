#!/usr/bin/env python3
"""Recognizer conversion parity and hybrid backend latency probe."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
SRC_ROOT = PROJECT_ROOT / "src"
AUTOKERNEL_ROOT = REPO_ROOT / "autokernel"
PPOCR_ROOT = PROJECT_ROOT / "PaddleOCR2Pytorch"

for path in (PROJECT_ROOT, SRC_ROOT, AUTOKERNEL_ROOT, PPOCR_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _source_prefix(source_dir: Path) -> Path:
    return source_dir / "inference"


def _load_paddle_layer(source_dir: Path) -> Any:
    import paddle

    paddle.set_device("cpu")
    return paddle.jit.load(str(_source_prefix(source_dir)))


def _load_torch_rec(weights_path: Path, device: str = "cpu") -> Any:
    import torch

    os.environ["AUTOKERNEL_PPOCR_ROOT"] = str(PPOCR_ROOT)
    os.environ["AUTOKERNEL_PPOCRV5_SERVER_REC_PTH"] = str(weights_path)

    from models.ppocrv5_server import PPOCRv5ServerRecModel

    return PPOCRv5ServerRecModel().to(device=device, dtype=torch.float32).eval()


def _paddle_logits(layer: Any, batch: np.ndarray) -> np.ndarray:
    import paddle

    with paddle.no_grad():
        out = layer(paddle.to_tensor(batch.astype("float32", copy=False)))
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out.numpy()


def _torch_logits(model: Any, batch: np.ndarray) -> np.ndarray:
    import torch

    with torch.inference_mode():
        out = model(torch.from_numpy(batch.astype("float32", copy=False)))
    return out.detach().cpu().numpy()


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    diff = np.abs(reference.astype("float32") - candidate.astype("float32"))
    denom = np.maximum(np.maximum(np.abs(reference), np.abs(candidate)), 1e-12)
    ref_idx = reference.argmax(axis=2)
    cand_idx = candidate.argmax(axis=2)
    return {
        "shape": list(reference.shape),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "p99_abs": float(np.percentile(diff, 99)),
        "max_rel": float((diff / denom).max()),
        "top1_mismatch_rate": float((ref_idx != cand_idx).mean()),
        "top1_mismatches": int((ref_idx != cand_idx).sum()),
        "top1_total": int(ref_idx.size),
    }


def _print_metrics(label: str, metrics: dict[str, Any]) -> None:
    print(f"{label}:", flush=True)
    for key, value in metrics.items():
        print(f"  {key}: {value}", flush=True)


def _make_random_batch(width: int, batch_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(batch_size, 3, 48, width)).astype("float32")


def _crop_batch_from_image(image_path: Path, max_crops: int) -> tuple[np.ndarray, int]:
    import torch
    import torch.nn.functional as F
    from src.services.ocr_backends import _crop_text_region

    try:
        from paddleocr import TextDetection

        detector = TextDetection(model_name="PP-OCRv5_server_det")
        image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
        det_results = detector.predict(image_rgb)
        boxes = []
        for result in det_results:
            for poly in result.get("dt_polys", result.get("rec_polys", [])):
                p = np.asarray(poly, dtype=np.float64)
                if p.ndim == 1:
                    p = p.reshape(-1, 2)
                boxes.append(p)
    except Exception as exc:
        raise RuntimeError(f"Could not detect crops for {image_path}: {exc}") from exc

    boxes = boxes[:max_crops]
    image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    crops = [_crop_text_region(image_bgr, box) for box in boxes]
    if not crops:
        raise RuntimeError(f"No text crops detected in {image_path}")

    img_c, img_h, default_w = 3, 48, 320
    limited_min, limited_max = 16, 4000
    width_list = [c.shape[1] / float(c.shape[0]) for c in crops]
    max_wh_ratio = max(max(width_list), default_w / img_h)
    target_w = max(min(int(img_h * max_wh_ratio), limited_max), limited_min)

    batch = torch.zeros(len(crops), img_c, img_h, target_w, dtype=torch.float32)
    for i, crop in enumerate(crops):
        h, w = crop.shape[:2]
        resized_w = max(min(int(np.ceil(img_h * w / float(h))), target_w), limited_min)
        t = torch.from_numpy(crop.copy()).float()
        t = t.permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=(img_h, resized_w), mode="bilinear", align_corners=False)
        t = t.squeeze(0)
        t = (t / 255.0 - 0.5) / 0.5
        batch[i, :, :, :resized_w] = t
    return batch.numpy(), len(crops)


def compare_logits(args: argparse.Namespace) -> None:
    source_dir = args.source_dir.resolve()
    current_weight = args.current_weight.resolve()
    old_weight = args.old_weight.resolve() if args.old_weight else None

    print("=== Logit parity: Paddle source vs converted PyTorch ===", flush=True)
    print(f"source_dir: {source_dir}", flush=True)
    print(f"current_weight: {current_weight}", flush=True)
    if old_weight:
        print(f"old_weight: {old_weight}", flush=True)

    paddle_layer = _load_paddle_layer(source_dir)
    current_model = _load_torch_rec(current_weight, device="cpu")
    old_model = _load_torch_rec(old_weight, device="cpu") if old_weight and old_weight.exists() else None

    for width in args.widths:
        batch = _make_random_batch(width, args.batch_size, args.seed + width)
        ref = _paddle_logits(paddle_layer, batch)
        cur = _torch_logits(current_model, batch)
        _print_metrics(f"random width={width} current", _metrics(ref, cur))
        if old_model is not None:
            old = _torch_logits(old_model, batch)
            _print_metrics(f"random width={width} old", _metrics(ref, old))

    if args.image:
        batch, n_crops = _crop_batch_from_image(args.image.resolve(), args.max_crops)
        print(
            f"\nreal image crops: {args.image} ({n_crops} crops, tensor={list(batch.shape)})",
            flush=True,
        )
        ref = _paddle_logits(paddle_layer, batch)
        cur = _torch_logits(current_model, batch)
        _print_metrics("real crops current", _metrics(ref, cur))
        if old_model is not None:
            old = _torch_logits(old_model, batch)
            _print_metrics("real crops old", _metrics(ref, old))


def profile_hybrid(args: argparse.Namespace) -> None:
    if not args.image:
        return

    import torch
    from src.services.ocr_backends import HybridOCRBackend, _crop_text_region

    print("\n=== Hybrid stage latency profile ===", flush=True)
    print(f"image: {args.image.resolve()}", flush=True)

    backend = HybridOCRBackend(
        paddle_config_path="PaddleOCR_hybrid.yaml",
        autokernel_root=AUTOKERNEL_ROOT,
        ppocr_root=PPOCR_ROOT,
        rec_weights_path=args.current_weight.resolve(),
        rec_source_path=args.source_dir.resolve() / "inference.pdiparams",
        workspace_path=AUTOKERNEL_ROOT / "workspace/graph_capture_eval",
        use_gpu=True,
        optimize_recognizer=True,
        rec_batch_size=1,
        rec_image_shape="3,48,320",
        dtype="float16",
    )

    image_rgb = np.asarray(Image.open(args.image).convert("RGB"))

    def sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    try:
        for i in range(args.profile_iters):
            t0 = time.perf_counter()
            boxes = backend._detect_boxes(image_rgb)
            t1 = time.perf_counter()

            image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
            crops = [_crop_text_region(image_bgr, box) for box in boxes]
            t2 = time.perf_counter()

            rec_results = backend._fast_recognize(crops) if crops else []
            sync()
            t3 = time.perf_counter()

            default_results = []
            default_ms = 0.0
            if crops and args.compare_default_processing:
                t_default0 = time.perf_counter()
                default_results, _default_elapsed = backend.recognizer(crops)
                sync()
                default_ms = (time.perf_counter() - t_default0) * 1000

            t_full0 = time.perf_counter()
            full = backend.predict(image_rgb)
            sync()
            t4 = time.perf_counter()

            mismatches = 0
            if default_results:
                fast_texts = [str(item[0]) for item in rec_results]
                default_texts = [str(item[0]) for item in default_results]
                mismatches = sum(a != b for a, b in zip(fast_texts, default_texts))

            print(
                "iter={i} boxes={boxes} rec={rec} "
                "detect_ms={detect:.2f} crop_ms={crop:.2f} "
                "fast_recognize_ms={recognize:.2f} "
                "default_recognize_ms={default:.2f} "
                "processing_text_mismatches={mismatches} "
                "full_predict_ms={full_ms:.2f}".format(
                    i=i,
                    boxes=len(boxes),
                    rec=len(rec_results),
                    detect=(t1 - t0) * 1000,
                    crop=(t2 - t1) * 1000,
                    recognize=(t3 - t2) * 1000,
                    default=default_ms,
                    mismatches=mismatches,
                    full_ms=(t4 - t_full0) * 1000,
                ),
                flush=True,
            )
            if default_results and mismatches:
                for idx, (fast_item, default_item) in enumerate(
                    zip(rec_results, default_results)
                ):
                    if str(fast_item[0]) != str(default_item[0]):
                        print(
                            "  mismatch[{idx}]: fast={fast!r} default={default!r}".format(
                                idx=idx,
                                fast=fast_item,
                                default=default_item,
                            ),
                            flush=True,
                        )
    finally:
        backend.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/home/jupyter/gisa_playground/OCR/deploy/rec_model/051025_data_additional_7900_v2_latest"),
    )
    parser.add_argument(
        "--current-weight",
        type=Path,
        default=REPO_ROOT / "autokernel/workspace/ppocrv5/server_rec.pth",
    )
    parser.add_argument(
        "--old-weight",
        type=Path,
        default=REPO_ROOT / "autokernel/workspace/ppocrv5/server_rec.pth.bak-20260416T040622Z",
    )
    parser.add_argument("--image", type=Path, default=PROJECT_ROOT / "tests/data/good_data.png")
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--max-crops", type=int, default=32)
    parser.add_argument("--widths", type=int, nargs="+", default=[320, 512, 640])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260416)
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--skip-logits", action="store_true")
    parser.add_argument("--compare-default-processing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.no_image:
        args.image = None
    if not args.skip_logits:
        compare_logits(args)
    if not args.skip_profile:
        profile_hybrid(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
