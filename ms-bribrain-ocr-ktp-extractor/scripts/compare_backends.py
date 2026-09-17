#!/usr/bin/env python3
"""Compare recogniser outputs: PyTorch fp32 baseline vs AutoKernel fp16.

Strategy:
  1. Use PaddleOCR (default models, no HPI) just for **detection** — get boxes.
  2. Crop text regions from the original image.
  3. Feed the *same* crops to both the fp32 and fp16 recognisers.
  4. Compare recognised text character-by-character.

This isolates the recogniser (which AutoKernel optimises) from the detector
(which is not optimised and has conversion gaps).

Usage:
    uv run python scripts/compare_backends.py
    uv run python scripts/compare_backends.py --image-dir /path/to/images
    uv run python scripts/compare_backends.py --output report.txt
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for name in ("ppocr", "paddle", "paddleocr", "torch", "triton"):
    logging.getLogger(name).setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Pre-import torch (autokernel/profile.py was renamed to ak_profile.py so
# the stdlib shadowing issue is fixed, but we still import early for clarity)
# ---------------------------------------------------------------------------
import torch  # noqa: E402

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RecResult:
    """Recognition result for a single text crop."""
    text: str
    score: float


@dataclass
class CropComparison:
    """Side-by-side comparison of one crop."""
    crop_idx: int
    paddle: RecResult       # PaddleOCR native recognition
    baseline: RecResult     # PyTorch fp32
    autokernel: RecResult   # PyTorch fp16
    paddle_vs_fp32_match: bool
    fp32_vs_fp16_match: bool
    paddle_vs_fp32_sim: float
    fp32_vs_fp16_sim: float


@dataclass
class ImageReport:
    """Comparison report for a single image."""
    image_path: Path
    n_boxes: int
    comparisons: list[CropComparison] = field(default_factory=list)
    baseline_time_ms: float = 0.0
    autokernel_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _char_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return []


# ---------------------------------------------------------------------------
# Build recogniser from PaddleOCR2Pytorch
# ---------------------------------------------------------------------------

def _resolve(val: str) -> Path:
    p = Path(val).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _get_config():
    from src.core.config import Settings
    return Settings(str(PROJECT_ROOT / "config.yaml"))


def _build_recogniser(
    *,
    optimize: bool,
    dtype_name: str,
) -> Any:
    """Build a PaddleOCR2Pytorch TextRecognizer with the converted rec weights.

    Args:
        optimize: whether to apply AutoKernel kernel replacements.
        dtype_name: "float32" or "float16".
    """
    cfg = _get_config()

    ppocr_root = _resolve(cfg.ocr_autokernel_ppocr_root)
    autokernel_root = _resolve(cfg.ocr_autokernel_root)
    rec_weights = _resolve(cfg.ocr_autokernel_rec_weights_path)
    workspace = _resolve(cfg.ocr_autokernel_workspace_path)

    # Add PaddleOCR2Pytorch to sys.path
    ppocr_str = str(ppocr_root)
    if ppocr_str not in sys.path:
        sys.path.insert(0, ppocr_str)

    # Add autokernel root to sys.path (needed for verify.py / support.py)
    ak_str = str(autokernel_root)
    if ak_str not in sys.path:
        sys.path.insert(0, ak_str)

    # Set env vars before importing autokernel models
    os.environ["AUTOKERNEL_PPOCR_ROOT"] = str(ppocr_root)
    os.environ["AUTOKERNEL_PPOCRV5_SERVER_REC_PTH"] = str(rec_weights)

    ppocr_model = importlib.import_module("models.ppocrv5_server")
    pytorchocr_utility = importlib.import_module("tools.infer.pytorchocr_utility")
    predict_rec = importlib.import_module("tools.infer.predict_rec")

    # Build args — use_gpu=False for the TextRecognizer so it keeps its
    # internal default model on CPU.  We replace net with our own GPU model.
    # This avoids GPU memory pollution that corrupts CUDA graph captures.
    parser = pytorchocr_utility.init_args()
    args = parser.parse_args([])
    args.use_gpu = False
    args.rec_algorithm = "SVTR_PPOCRv5"
    args.rec_yaml_path = str(ppocr_root / ppocr_model.REC_YAML_RELATIVE)
    args.rec_model_path = str(rec_weights)
    args.rec_image_shape = cfg.ocr_autokernel_rec_image_shape
    args.rec_char_dict_path = str(ppocr_root / "pytorchocr/utils/dict/ppocrv5_dict.txt")
    args.rec_batch_num = cfg.ocr_autokernel_rec_batch_size
    args.image_dir = ""

    recogniser = predict_rec.TextRecognizer(args)
    # Override use_gpu so the __call__ method sends tensors to CUDA
    recogniser.use_gpu = True

    # Build rec model wrapper
    rec_wrapper = ppocr_model.PPOCRv5ServerRecModel()
    dtype = torch.float16 if dtype_name == "float16" else torch.float32
    device = "cuda"

    model = rec_wrapper.to(device=device, dtype=dtype).eval()

    if optimize:
        from src.services.ocr_backends import load_verified_replacement_specs

        specs = load_verified_replacement_specs(workspace, autokernel_root)

        verify_path = autokernel_root / "verify.py"
        spec_obj = importlib.util.spec_from_file_location("dgc_ext_autokernel_verify", verify_path)
        verify_mod = importlib.util.module_from_spec(spec_obj)
        sys.modules["dgc_ext_autokernel_verify"] = verify_mod
        spec_obj.loader.exec_module(verify_mod)

        support_mod = importlib.import_module("support")

        replacements = []
        for s in specs:
            support_stage = support_mod.build_support_stage(s.kernel_type, {s.kernel_type})
            replacements.append(verify_mod.KernelReplacement(
                kernel_type=s.kernel_type,
                rank=s.rank,
                speedup=s.speedup,
                optimized_path=str(s.optimized_path),
                reinsert_supported=support_stage["reinsert_supported"],
                status=s.status,
            ))

        context = verify_mod.OptimizedModelContext(model, replacements)
        patched = context.__enter__()

        class _Adapter(torch.nn.Module):
            def __init__(self, m, ctx):
                super().__init__()
                self.m = m
                self.ctx = ctx
            def forward(self, x):
                x = x.to(device=device, dtype=dtype)
                if self.ctx is not None:
                    x = self.ctx.prepare_input(x)
                return self.m(x)

        recogniser.net = _Adapter(patched, context).eval()
        recogniser._ak_context = context  # keep ref for cleanup
    else:
        # Wrap with dtype/device casting for fp16
        class _CastAdapter(torch.nn.Module):
            def __init__(self, m, dt, dev):
                super().__init__()
                self.m = m
                self.dt = dt
                self.dev = dev
            def forward(self, x):
                return self.m(x.to(device=self.dev, dtype=self.dt))

        recogniser.net = _CastAdapter(model, dtype, device).eval()

    return recogniser


# ---------------------------------------------------------------------------
# Detection using PaddleOCR default models (no HPI)
# ---------------------------------------------------------------------------

_paddle_ocr_instance = None


def _get_paddle_ocr():
    """Lazy-init PaddleOCR with the finetuned rec model (caches instance)."""
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        from paddleocr import PaddleOCR

        # Use the finetuned rec model (same weights as PyTorch conversion)
        rec_model_dir = str(
            PROJECT_ROOT
            / "src/models/server_models/ppocrv5_server_rec_source"
        )
        _paddle_ocr_instance = PaddleOCR(
            ocr_version="PP-OCRv5",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_hpi=False,
            text_recognition_model_dir=rec_model_dir,
        )
    return _paddle_ocr_instance


def _detect_boxes_paddle(image_bgr: np.ndarray) -> list[np.ndarray]:
    """Use PaddleOCR default PP-OCRv5 det (no HPI) to find text boxes."""
    ocr = _get_paddle_ocr()
    results = ocr.predict(image_bgr)
    boxes = []
    for result in results:
        for item in result.get("rec_polys", result.get("dt_polys", [])):
            poly = np.asarray(item, dtype=np.float64)
            if poly.ndim == 1:
                poly = poly.reshape(-1, 2)
            boxes.append(poly)
    return boxes


def paddle_recognise_crops(
    image_bgr: np.ndarray,
    boxes: list[np.ndarray],
) -> list[RecResult]:
    """Run PaddleOCR's own full pipeline on the same image and extract
    recognition results ordered by the given box list.

    Since PaddleOCR runs det+rec together, we run predict() and then match
    the returned polys back to our box order.
    """
    ocr = _get_paddle_ocr()
    results = ocr.predict(image_bgr)

    # Build lookup: (centroid) → (text, score)
    paddle_regions: list[tuple[np.ndarray, str, float]] = []
    for result in results:
        texts = result.get("rec_texts", [])
        scores = result.get("rec_scores", [])
        polys = result.get("rec_polys", result.get("dt_polys", []))
        for text, score, poly in zip(texts, scores, polys):
            p = np.asarray(poly, dtype=np.float64)
            if p.ndim == 1:
                p = p.reshape(-1, 2)
            paddle_regions.append((p, str(text), float(score)))

    # Match each input box to the nearest PaddleOCR poly by centroid distance
    matched: list[RecResult] = []
    used = set()
    for box in boxes:
        box_center = box.mean(axis=0)
        best_idx, best_dist = -1, float("inf")
        for j, (p, _, _) in enumerate(paddle_regions):
            if j in used:
                continue
            dist = float(np.linalg.norm(p.mean(axis=0) - box_center))
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        if best_idx >= 0 and best_dist < 50:  # 50px tolerance
            used.add(best_idx)
            _, text, score = paddle_regions[best_idx]
            matched.append(RecResult(text=text, score=score))
        else:
            matched.append(RecResult(text="<unmatched>", score=0.0))
    return matched


def _detect_boxes_ppocr2pt(image_bgr: np.ndarray) -> list[np.ndarray]:
    """Fallback: use PaddleOCR2Pytorch's TextDetector with converted weights."""
    cfg = _get_config()
    ppocr_root = _resolve(cfg.ocr_autokernel_ppocr_root)
    det_weights = _resolve(cfg.ocr_autokernel_det_weights_path)

    ppocr_str = str(ppocr_root)
    if ppocr_str not in sys.path:
        sys.path.insert(0, ppocr_str)

    ppocr_model = importlib.import_module("models.ppocrv5_server")
    pytorchocr_utility = importlib.import_module("tools.infer.pytorchocr_utility")
    predict_det = importlib.import_module("tools.infer.predict_det")

    parser = pytorchocr_utility.init_args()
    args = parser.parse_args([])
    args.use_gpu = True
    args.det_algorithm = "DB"
    args.det_yaml_path = str(ppocr_root / ppocr_model.DET_YAML_RELATIVE)
    args.det_model_path = str(det_weights)
    args.image_dir = ""

    detector = predict_det.TextDetector(args)

    # Replace net with the loaded model
    det_wrapper = ppocr_model.PPOCRv5ServerDetModel()
    detector.net = det_wrapper.net.to("cuda").float().eval()

    boxes, _ = detector(image_bgr)
    return [np.asarray(b, dtype=np.float64) for b in boxes] if boxes is not None else []


def detect_boxes(image_bgr: np.ndarray) -> list[np.ndarray]:
    """Try PaddleOCR first, fall back to PaddleOCR2Pytorch."""
    try:
        boxes = _detect_boxes_paddle(image_bgr)
        if boxes:
            print(f"    Detected {len(boxes)} boxes via PaddleOCR")
            return boxes
    except Exception as e:
        print(f"    PaddleOCR detection failed ({e}), trying PaddleOCR2Pytorch...")

    boxes = _detect_boxes_ppocr2pt(image_bgr)
    print(f"    Detected {len(boxes)} boxes via PaddleOCR2Pytorch")
    return boxes


# ---------------------------------------------------------------------------
# Crop text regions
# ---------------------------------------------------------------------------

def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float64)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def crop_text_region(image: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Crop and perspective-correct a text region from the image."""
    import cv2

    pts = _order_points(poly)
    tl, tr, br, bl = pts

    w = max(int(np.linalg.norm(tr - tl)), int(np.linalg.norm(br - bl)))
    h = max(int(np.linalg.norm(bl - tl)), int(np.linalg.norm(br - tr)))
    w = max(w, 1)
    h = max(h, 1)

    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
    crop = cv2.warpPerspective(image, M, (w, h))
    return crop


# ---------------------------------------------------------------------------
# Run recogniser on a list of crops
# ---------------------------------------------------------------------------

def recognise_crops(
    recogniser: Any,
    crops: list[np.ndarray],
) -> list[RecResult]:
    """Run the recogniser on a list of BGR image crops."""
    if not crops:
        return []
    rec_results, _ = recogniser(crops)
    return [RecResult(text=str(r[0]), score=float(r[1])) for r in rec_results]


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare_one_image(
    image_path: Path,
    baseline_rec: Any,
    autokernel_rec: Any,
) -> ImageReport:
    """Detect text in image, crop regions, compare recogniser outputs."""
    img_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    img_bgr = img_rgb[:, :, ::-1].copy()

    # Detect
    boxes = detect_boxes(img_bgr)
    if not boxes:
        return ImageReport(image_path=image_path, n_boxes=0)

    # Crop
    crops = [crop_text_region(img_bgr, box) for box in boxes]

    # PaddleOCR native recognition (runs det+rec, matched by centroid)
    paddle_results = paddle_recognise_crops(img_bgr, boxes)

    # Recognise with baseline (fp32) — must run BEFORE AK to avoid
    # CUDA graph memory corruption (two models can't share GPU safely
    # when one uses CUDA graphs).
    if crops:
        recognise_crops(baseline_rec, crops[:1])  # warmup
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    baseline_results = recognise_crops(baseline_rec, crops)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    baseline_ms = (time.perf_counter() - t0) * 1000

    # Free fp32 model from GPU before running AK (CUDA graphs need
    # stable memory addresses — a second model can corrupt them).
    baseline_rec.net = None
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Recognise with AK (fp16 + graph capture)
    if crops:
        recognise_crops(autokernel_rec, crops[:1])  # warmup + first graph capture
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    ak_results = recognise_crops(autokernel_rec, crops)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ak_ms = (time.perf_counter() - t0) * 1000

    # Compare
    comparisons = []
    for i, (pr, br, ar) in enumerate(zip(paddle_results, baseline_results, ak_results)):
        comparisons.append(CropComparison(
            crop_idx=i,
            paddle=pr,
            baseline=br,
            autokernel=ar,
            paddle_vs_fp32_match=pr.text == br.text,
            fp32_vs_fp16_match=br.text == ar.text,
            paddle_vs_fp32_sim=_char_similarity(pr.text, br.text),
            fp32_vs_fp16_sim=_char_similarity(br.text, ar.text),
        ))

    return ImageReport(
        image_path=image_path,
        n_boxes=len(boxes),
        comparisons=comparisons,
        baseline_time_ms=baseline_ms,
        autokernel_time_ms=ak_ms,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def format_report(reports: list[ImageReport]) -> str:
    lines: list[str] = []
    w = lines.append

    w("=" * 76)
    w("  Recogniser Comparison: PaddleOCR  vs  PyTorch fp32  vs  AutoKernel fp16")
    w("  (Same text crops from PaddleOCR detection)")
    w("=" * 76)
    w("")

    total_crops = 0
    total_paddle_fp32 = 0
    total_fp32_fp16 = 0
    paddle_fp32_mismatches: list[tuple[Path, CropComparison]] = []
    fp32_fp16_mismatches: list[tuple[Path, CropComparison]] = []

    for r in reports:
        w(f"Image: {r.image_path.name}")
        w(f"  Detected boxes : {r.n_boxes}")

        if not r.comparisons:
            w("  (no text regions detected)")
            w("")
            continue

        n_total = len(r.comparisons)
        n_p32 = sum(1 for c in r.comparisons if c.paddle_vs_fp32_match)
        n_3216 = sum(1 for c in r.comparisons if c.fp32_vs_fp16_match)

        w(f"  PyTorch fp32   : {r.baseline_time_ms:7.1f} ms")
        w(f"  PyTorch fp16   : {r.autokernel_time_ms:7.1f} ms")
        w(f"  Paddle vs fp32 : {n_p32}/{n_total} exact ({100*n_p32/n_total:.0f}%)")
        w(f"  fp32 vs fp16   : {n_3216}/{n_total} exact ({100*n_3216/n_total:.0f}%)")

        # Detailed per-crop table
        w("")
        w(f"  {'#':>3}  {'Paddle':30s}  {'fp32':30s}  {'fp16':30s}  Match")
        w(f"  {'─'*3}  {'─'*30}  {'─'*30}  {'─'*30}  {'─'*5}")
        for c in r.comparisons:
            p_t = c.paddle.text[:28] if c.paddle.text != "<unmatched>" else "—"
            b_t = c.baseline.text[:28]
            a_t = c.autokernel.text[:28]
            flags = ""
            if not c.paddle_vs_fp32_match and c.paddle.text != "<unmatched>":
                flags += "P≠32 "
            if not c.fp32_vs_fp16_match:
                flags += "32≠16"
            if not flags:
                flags = "✓"
            w(f"  {c.crop_idx:3d}  {p_t:30s}  {b_t:30s}  {a_t:30s}  {flags}")

            if not c.paddle_vs_fp32_match and c.paddle.text != "<unmatched>":
                paddle_fp32_mismatches.append((r.image_path, c))
            if not c.fp32_vs_fp16_match:
                fp32_fp16_mismatches.append((r.image_path, c))

        total_crops += n_total
        total_paddle_fp32 += n_p32
        total_fp32_fp16 += n_3216
        w("")

    # Summary
    w("-" * 76)
    w("  SUMMARY")
    w("-" * 76)
    w(f"  Images tested       : {len(reports)}")
    w(f"  Total text crops    : {total_crops}")
    if total_crops:
        w(f"  Paddle vs fp32 match: {total_paddle_fp32}/{total_crops}"
          f"  ({100*total_paddle_fp32/total_crops:.1f}%)"
          f"   ← weight conversion correctness")
        w(f"  fp32 vs fp16 match  : {total_fp32_fp16}/{total_crops}"
          f"  ({100*total_fp32_fp16/total_crops:.1f}%)"
          f"   ← precision safety")
    w("")

    if fp32_fp16_mismatches:
        w("  VERDICT: fp16 PRECISION CAUSES TEXT CHANGES — review before deployment")
    elif paddle_fp32_mismatches:
        n = len(paddle_fp32_mismatches)
        w(f"  VERDICT: fp32↔fp16 SAFE, but {n} Paddle↔fp32 differences"
          f" (weight conversion gaps)")
    elif total_crops == 0:
        w("  VERDICT: NO TEXT DETECTED — add more test images")
    else:
        w("  VERDICT: ALL THREE PATHS PRODUCE IDENTICAL TEXT")
    w("=" * 76)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare recogniser outputs: PyTorch fp32 vs AutoKernel fp16"
    )
    parser.add_argument(
        "--image-dir", type=Path,
        default=PROJECT_ROOT / "tests" / "data",
        help="Directory of images or single image path",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save report to file",
    )
    args = parser.parse_args()

    images = collect_images(args.image_dir)
    if not images:
        print(f"No images found in {args.image_dir}")
        sys.exit(1)
    print(f"Found {len(images)} image(s) to compare\n")

    # CUDA graphs capture memory addresses for ALL intermediate tensors.
    # A second model on the same GPU can corrupt graph replay by reusing
    # those addresses.  We run each backend in its own pass with full GPU
    # cleanup between them.
    import gc

    # --- Pass 1: fp32 baseline ---
    print("Pass 1/2: PyTorch fp32 baseline...")
    baseline_rec = _build_recogniser(optimize=False, dtype_name="float32")
    baseline_cache: dict[Path, tuple[list, list[np.ndarray], list[RecResult], float]] = {}
    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img_path.name}...")
        img_rgb = np.asarray(Image.open(img_path).convert("RGB"))
        img_bgr = img_rgb[:, :, ::-1].copy()
        boxes = detect_boxes(img_bgr)
        crops = [crop_text_region(img_bgr, box) for box in boxes]
        if crops:
            recognise_crops(baseline_rec, crops[:1])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = recognise_crops(baseline_rec, crops)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        baseline_cache[img_path] = (boxes, crops, results, ms)
    # Aggressively free the fp32 model and reset torch state.
    del baseline_rec
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # Reset torch.compile / dynamo caches to avoid cross-model interference
    torch._dynamo.reset()

    # --- Pass 2: AutoKernel fp16 ---
    print("Pass 2/2: AutoKernel fp16 (kernel optimisation + multi-width graph capture)...")
    autokernel_rec = _build_recogniser(optimize=True, dtype_name="float16")
    ak_cache: dict[Path, tuple[list[RecResult], float]] = {}
    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img_path.name}...")
        _, crops, _, _ = baseline_cache[img_path]
        if crops:
            recognise_crops(autokernel_rec, crops[:1])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = recognise_crops(autokernel_rec, crops)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        ak_cache[img_path] = (results, ms)

    # --- Build reports ---
    reports: list[ImageReport] = []
    for img_path in images:
        boxes, crops, baseline_results, baseline_ms = baseline_cache[img_path]
        ak_results, ak_ms = ak_cache[img_path]
        img_rgb = np.asarray(Image.open(img_path).convert("RGB"))
        img_bgr = img_rgb[:, :, ::-1].copy()
        paddle_results = paddle_recognise_crops(img_bgr, boxes)
        comparisons = []
        for i, (pr, br, ar) in enumerate(zip(paddle_results, baseline_results, ak_results)):
            comparisons.append(CropComparison(
                crop_idx=i, paddle=pr, baseline=br, autokernel=ar,
                paddle_vs_fp32_match=pr.text == br.text,
                fp32_vs_fp16_match=br.text == ar.text,
                paddle_vs_fp32_sim=_char_similarity(pr.text, br.text),
                fp32_vs_fp16_sim=_char_similarity(br.text, ar.text),
            ))
        reports.append(ImageReport(
            image_path=img_path, n_boxes=len(boxes),
            comparisons=comparisons,
            baseline_time_ms=baseline_ms, autokernel_time_ms=ak_ms,
        ))

    # Report
    print()
    report_text = format_report(reports)
    print(report_text)

    if args.output:
        args.output.write_text(report_text, encoding="utf-8")
        print(f"\nReport saved to {args.output}")

    # Cleanup
    ctx = getattr(autokernel_rec, "_ak_context", None)
    if ctx:
        ctx.__exit__(None, None, None)

    total_mismatches = sum(
        1 for r in reports for c in r.comparisons if not c.fp32_vs_fp16_match
    )
    sys.exit(1 if total_mismatches > 0 else 0)


if __name__ == "__main__":
    main()
