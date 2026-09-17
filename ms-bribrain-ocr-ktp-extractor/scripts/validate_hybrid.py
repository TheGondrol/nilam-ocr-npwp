#!/usr/bin/env python3
"""End-to-end validation of the HybridOCRBackend.

Runs the hybrid backend (PaddleOCR detection + AutoKernel PyTorch recognition)
on test images and compares with pure PaddleOCR results.

Usage:
    .venv/bin/python -u scripts/validate_hybrid.py
    .venv/bin/python -u scripts/validate_hybrid.py --image /path/to/image.jpg
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for name in ("ppocr", "paddle", "paddleocr", "torch", "triton"):
    logging.getLogger(name).setLevel(logging.ERROR)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate hybrid OCR backend")
    parser.add_argument(
        "--image", type=Path, default=None,
        help="Single image or directory (defaults to tests/data)",
    )
    args = parser.parse_args()

    image_path = args.image or PROJECT_ROOT / "tests" / "data"
    images = collect_images(image_path)
    if not images:
        print(f"No images found at {image_path}")
        sys.exit(1)
    print(f"Found {len(images)} image(s)\n")

    # ------------------------------------------------------------------
    # 1. Build PaddleOCR backend (baseline)
    # ------------------------------------------------------------------
    print("Building PaddleOCR baseline...")
    from src.services.ocr_backends import PaddleOCRBackend

    try:
        paddle_backend = PaddleOCRBackend("PaddleOCR_server.yaml")
    except Exception:
        print("  Server config failed, trying PP-OCRv5 without HPI...")
        from paddleocr import PaddleOCR

        class _WrappedPaddleOCR:
            def __init__(self, **kw):
                self.engine = PaddleOCR(**kw)
            def predict(self, img):
                return self.engine.predict(img)
            def close(self):
                pass

        paddle_backend = type("PaddleOCRBackend", (), {
            "name": "paddle",
            "predict": lambda self, img: self._engine.predict(img),
            "warmup": lambda self: None,
            "close": lambda self: None,
        })()
        paddle_backend._engine = PaddleOCR(
            ocr_version="PP-OCRv5",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_hpi=False,
        )
        paddle_backend.predict = lambda img: paddle_backend._engine.predict(img)
    print("  PaddleOCR ready\n")

    # ------------------------------------------------------------------
    # 2. Build Hybrid backend
    # ------------------------------------------------------------------
    print("Building Hybrid backend (PaddleOCR det + AutoKernel rec)...")
    from src.core.config import Settings
    from src.services.ocr_backends import HybridOCRBackend, _resolve_path, _settings_base_dir

    cfg = Settings(str(PROJECT_ROOT / "config.yaml"))
    base_dir = _settings_base_dir(cfg)

    hybrid_backend = HybridOCRBackend(
        paddle_config_path=cfg.ocr_server_config_path,
        autokernel_root=_resolve_path(cfg.ocr_autokernel_root, base_dir),
        ppocr_root=_resolve_path(cfg.ocr_autokernel_ppocr_root, base_dir),
        rec_weights_path=_resolve_path(cfg.ocr_autokernel_rec_weights_path, base_dir),
        rec_source_path=(
            _resolve_path(cfg.ocr_autokernel_rec_source_path, base_dir)
            if cfg.ocr_autokernel_rec_source_path
            else None
        ),
        workspace_path=_resolve_path(cfg.ocr_autokernel_workspace_path, base_dir),
        use_gpu=True,
        optimize_recognizer=cfg.ocr_autokernel_optimize_recognizer,
        rec_batch_size=cfg.ocr_autokernel_rec_batch_size,
        rec_image_shape=cfg.ocr_autokernel_rec_image_shape,
        dtype=cfg.ocr_autokernel_dtype,
    )
    print("  Hybrid backend ready\n")

    # ------------------------------------------------------------------
    # 3. Run both on each image and compare
    # ------------------------------------------------------------------
    total_paddle_texts = 0
    total_hybrid_texts = 0
    total_matches = 0
    total_diffs = 0

    for img_path in images:
        print(f"Image: {img_path.name}")
        img_rgb = np.asarray(Image.open(img_path).convert("RGB"))

        # PaddleOCR
        t0 = time.perf_counter()
        paddle_results = paddle_backend.predict(img_rgb)
        paddle_ms = (time.perf_counter() - t0) * 1000

        # Hybrid
        t0 = time.perf_counter()
        hybrid_results = hybrid_backend.predict(img_rgb)
        hybrid_ms = (time.perf_counter() - t0) * 1000

        # Extract texts
        p_res = paddle_results[0] if paddle_results else {}
        h_res = hybrid_results[0] if hybrid_results else {}

        p_texts = p_res.get("rec_texts", [])
        h_texts = h_res.get("rec_texts", [])
        p_scores = p_res.get("rec_scores", [])
        h_scores = h_res.get("rec_scores", [])

        print(f"  PaddleOCR  : {len(p_texts):3d} texts, {paddle_ms:7.1f} ms")
        print(f"  Hybrid     : {len(h_texts):3d} texts, {hybrid_ms:7.1f} ms")

        # Compare texts — the detection boxes may differ slightly in ordering,
        # so we compare by matching texts rather than positional alignment.
        p_set = set(p_texts)
        h_set = set(h_texts)
        common = p_set & h_set
        only_paddle = p_set - h_set
        only_hybrid = h_set - p_set

        total_paddle_texts += len(p_texts)
        total_hybrid_texts += len(h_texts)
        total_matches += len(common)
        total_diffs += len(only_paddle) + len(only_hybrid)

        # Per-crop detail table
        max_rows = max(len(p_texts), len(h_texts))
        if max_rows > 0:
            print(f"\n  {'#':>3}  {'PaddleOCR':35s}  {'Hybrid':35s}  Match")
            print(f"  {'─'*3}  {'─'*35}  {'─'*35}  {'─'*5}")
            for i in range(max_rows):
                pt = p_texts[i][:33] if i < len(p_texts) else "—"
                ht = h_texts[i][:33] if i < len(h_texts) else "—"
                ps = f"{p_scores[i]:.3f}" if i < len(p_scores) else ""
                hs = f"{h_scores[i]:.3f}" if i < len(h_scores) else ""
                match = "✓" if (i < len(p_texts) and i < len(h_texts) and p_texts[i] == h_texts[i]) else "≠"
                print(f"  {i:3d}  {pt:35s}  {ht:35s}  {match}")
        print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 76)
    print("  SUMMARY")
    print("=" * 76)
    print(f"  Images tested      : {len(images)}")
    print(f"  PaddleOCR texts    : {total_paddle_texts}")
    print(f"  Hybrid texts       : {total_hybrid_texts}")
    print(f"  Common texts       : {total_matches}")
    if total_diffs == 0 and total_matches > 0:
        print(f"\n  VERDICT: HYBRID PRODUCES IDENTICAL TEXTS TO PADDLEOCR")
    elif total_diffs > 0:
        pct = 100 * total_matches / max(total_paddle_texts, 1)
        print(f"  Differing texts    : {total_diffs}")
        print(f"\n  VERDICT: {pct:.1f}% text overlap — differences may come from")
        print(f"           recognition model (finetuned PyTorch vs PaddleOCR default)")
    else:
        print(f"\n  VERDICT: NO TEXT DETECTED")
    print("=" * 76)

    # Cleanup
    hybrid_backend.close()

    paddle_close = getattr(paddle_backend, "close", None)
    if callable(paddle_close):
        paddle_close()


if __name__ == "__main__":
    main()
