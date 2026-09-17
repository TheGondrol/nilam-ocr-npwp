#!/usr/bin/env python3
"""End-to-end line-by-line OCR text comparison: paddle vs hybrid vs autokernel.

Each backend runs its own det+rec pipeline via ``.predict()`` — no shared
crops or shared detector. Boxes are aligned across backends by centroid
distance, using the Paddle result as the reference. Lines that appear in
only one backend are listed as orphans.

Usage:
    .venv/bin/python -u scripts/compare_ocr_lines.py
    .venv/bin/python -u scripts/compare_ocr_lines.py --image path/to/img.png
    .venv/bin/python -u scripts/compare_ocr_lines.py --output report.txt
    .venv/bin/python -u scripts/compare_ocr_lines.py --only paddle,hybrid
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for name in ("ppocr", "paddle", "paddleocr", "torch", "triton"):
    logging.getLogger(name).setLevel(logging.ERROR)

import torch  # noqa: E402  — load early to avoid cuda/paddle init-order issues

try:
    # Force CUDA initialization before PaddleOCR has a chance to fall back to CPU.
    # A plain import is not always enough; Paddle can leave later
    # torch.cuda.is_available() checks false in the same process.
    if torch.cuda.is_available():
        torch.cuda.init()
except Exception:
    pass

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
CENTROID_MATCH_PX = 40.0  # max centroid distance to consider same text line


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Line:
    text: str
    score: float
    poly: np.ndarray  # shape (N, 2)

    @property
    def centroid(self) -> np.ndarray:
        return self.poly.mean(axis=0)


@dataclass
class BackendResult:
    backend: str
    lines: list[Line] = field(default_factory=list)
    elapsed_ms: float = 0.0
    error: str | None = None


@dataclass
class ImageReport:
    image_path: Path
    results: dict[str, BackendResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Image IO
# ---------------------------------------------------------------------------

def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return []


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


# ---------------------------------------------------------------------------
# Backend builders — each returns an object with .predict(rgb) and .close()
# ---------------------------------------------------------------------------

def _make_backend_settings(backend_name: str) -> Any:
    """Build the SimpleNamespace-shaped settings object consumed by
    ``create_ocr_backend``. Mirrors ``ocr_service._build_backend_settings``
    so we can reuse the same env-var-driven configuration path.
    """
    from types import SimpleNamespace
    import os
    from src.core.config import settings as base_settings

    def env_bool(name: str, default: bool = False) -> bool:
        v = os.getenv(name)
        return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}

    # Sensible local defaults so the script works without env setup if the
    # standard repo layout is present.
    ak_root_default = str((PROJECT_ROOT / "autokernel").resolve())
    ppocr_root_default = str((PROJECT_ROOT / "PaddleOCR2Pytorch").resolve())
    workspace_default = str((PROJECT_ROOT / "autokernel" / "workspace" / "graph_capture_eval").resolve())
    rec_pth_default = str((PROJECT_ROOT / "autokernel" / "workspace" / "ppocrv5" / "server_rec.pth").resolve())
    det_pth_default = str((PROJECT_ROOT / "autokernel" / "workspace" / "ppocrv5" / "server_det.pth").resolve())

    exclude = os.getenv("OCR_AUTOKERNEL_EXCLUDE_KERNEL_TYPES")
    ns = SimpleNamespace(
        ocr_backend=backend_name,
        ocr_autokernel_enabled=True,  # force-enabled for this script
        ocr_server_config_path=os.getenv(
            "OCR_SERVER_CONFIG_PATH", base_settings.ocr_server_config_path
        ),
        ocr_mobile_config_path=base_settings.ocr_mobile_config_path,
        ocr_autokernel_root=os.getenv("AUTOKERNEL_ROOT", ak_root_default),
        ocr_autokernel_ppocr_root=os.getenv("AUTOKERNEL_PPOCR_ROOT", ppocr_root_default),
        ocr_autokernel_workspace_path=os.getenv("AUTOKERNEL_WORKSPACE", workspace_default),
        ocr_autokernel_det_weights_path=os.getenv("AUTOKERNEL_PPOCRV5_SERVER_DET_PTH", det_pth_default),
        ocr_autokernel_rec_weights_path=os.getenv("AUTOKERNEL_PPOCRV5_SERVER_REC_PTH", rec_pth_default),
        ocr_autokernel_det_source_path=os.getenv("AUTOKERNEL_PPOCRV5_SERVER_DET_SOURCE", ""),
        ocr_autokernel_rec_source_path=os.getenv("AUTOKERNEL_PPOCRV5_SERVER_REC_SOURCE", ""),
        ocr_autokernel_auto_convert_weights=env_bool("OCR_AUTOKERNEL_AUTO_CONVERT_WEIGHTS", False),
        ocr_autokernel_optimize_recognizer=env_bool("OCR_AUTOKERNEL_OPTIMIZE_RECOGNIZER", True),
        ocr_autokernel_optimize_detector=env_bool("OCR_AUTOKERNEL_OPTIMIZE_DETECTOR", False),
        ocr_autokernel_rec_batch_size=int(os.getenv("OCR_AUTOKERNEL_REC_BATCH_SIZE", "1")),
        ocr_autokernel_rec_image_shape=os.getenv("OCR_AUTOKERNEL_REC_IMAGE_SHAPE", "3,48,320"),
        ocr_autokernel_rec_bucket_max_width_ratio=float(
            os.getenv("OCR_AUTOKERNEL_REC_BUCKET_MAX_WIDTH_RATIO", "1.30")
        ),
        ocr_autokernel_det_limit_side_len=int(os.getenv("OCR_AUTOKERNEL_DET_LIMIT_SIDE_LEN", "1280")),
        ocr_autokernel_det_limit_type=os.getenv("OCR_AUTOKERNEL_DET_LIMIT_TYPE", "max"),
        ocr_autokernel_torch_compile=env_bool("OCR_AUTOKERNEL_TORCH_COMPILE", False),
        ocr_autokernel_torch_compile_det=env_bool(
            "OCR_AUTOKERNEL_TORCH_COMPILE_DET",
            env_bool("OCR_AUTOKERNEL_TORCH_COMPILE", False),
        ),
        ocr_autokernel_torch_compile_rec=env_bool(
            "OCR_AUTOKERNEL_TORCH_COMPILE_REC",
            env_bool("OCR_AUTOKERNEL_TORCH_COMPILE", False),
        ),
        ocr_autokernel_torch_compile_mode=os.getenv("OCR_AUTOKERNEL_TORCH_COMPILE_MODE", "default"),
        ocr_autokernel_torch_compile_dynamic=env_bool("OCR_AUTOKERNEL_TORCH_COMPILE_DYNAMIC", True),
        ocr_autokernel_warmup_image_path=os.getenv("OCR_AUTOKERNEL_WARMUP_IMAGE_PATH", ""),
        ocr_autokernel_dtype=os.getenv("OCR_AUTOKERNEL_DTYPE", "float16"),
        ocr_autokernel_exclude_kernel_types=(
            [s.strip() for s in exclude.split(",") if s.strip()] if exclude else None
        ),
        # required by _settings_base_dir
        config_path=str(PROJECT_ROOT / "config.yaml"),
    )
    return ns


PADDLE_REC_MODEL_DIR: str | None = None  # set from CLI; points to the same
# pdiparams that was converted to .pth for the PyTorch side. Ensures apples-
# to-apples: identical weights, only the runtime format differs.
PADDLE_CONFIG_PATH: str | None = None  # set from CLI. When provided, baseline
# Paddle loads this paddlex_config so its det preprocessing/thresholds match
# the project's shipping configs (e.g. PaddleOCR_server_nohpi.yaml).


def _build_paddle_backend() -> Any:
    """Build PaddleOCR with the **finetuned** rec weights explicitly set.

    The default ``PaddleOCR_hybrid.yaml`` has ``TextRecognition.model_dir: null``
    which silently loads the stock PP-OCRv5 weights from paddlex cache — NOT
    the finetuned model that was converted to .pth. Passing
    ``text_recognition_model_dir`` overrides that so both sides run the same
    underlying weights.
    """
    from paddleocr import PaddleOCR

    rec_dir = PADDLE_REC_MODEL_DIR
    if rec_dir is None or not Path(rec_dir).is_dir():
        raise RuntimeError(
            f"Paddle rec model dir not found: {rec_dir!r}. Pass --paddle-rec-dir."
        )

    # enable_hpi=False → native Paddle framework inference. With HPI enabled,
    # PaddleOCR auto-converts pdiparams→ONNX via the ``paddlex`` CLI, which
    # isn't on PATH here. Native inference is also the cleaner apples-to-
    # apples against PyTorch — no intermediate ONNX conversion on either side.
    if PADDLE_CONFIG_PATH:
        engine = PaddleOCR(
            paddlex_config=PADDLE_CONFIG_PATH,
            text_recognition_model_dir=rec_dir,
            enable_hpi=False,
        )
    else:
        engine = PaddleOCR(
            ocr_version="PP-OCRv5",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_hpi=False,
            text_recognition_model_dir=rec_dir,
        )

    class _PaddleFinetuned:
        name = "paddle"
        def predict(self, img): return engine.predict(img)
        def close(self):
            closer = getattr(engine, "close", None)
            if callable(closer):
                closer()

    return _PaddleFinetuned()


def _build_hybrid_backend() -> Any:
    from src.services.ocr_backends import create_ocr_backend
    return create_ocr_backend(_make_backend_settings("hybrid"), use_gpu=True)


def _build_autokernel_backend() -> Any:
    from src.services.ocr_backends import create_ocr_backend
    return create_ocr_backend(_make_backend_settings("autokernel"), use_gpu=True)


def _build_fullpytorch_backend() -> Any:
    from src.services.ocr_backends import create_ocr_backend
    settings = _make_backend_settings("fullpytorch")
    return create_ocr_backend(settings, use_gpu=True)


BACKEND_BUILDERS = {
    "paddle": _build_paddle_backend,
    "hybrid": _build_hybrid_backend,
    "autokernel": _build_autokernel_backend,
    "fullpytorch": _build_fullpytorch_backend,
}


# ---------------------------------------------------------------------------
# Shared-crops mode: run Paddle det+rec once, reuse polys + crops + paddle
# texts as ground-truth, then call each other backend's *recognizer only*
# on the exact same crops.  Isolates recognizer diffs from detector and
# crop-extraction differences.
# ---------------------------------------------------------------------------

def _extract_shared_crops(
    image_rgb: np.ndarray, polys: list[np.ndarray]
) -> list[np.ndarray]:
    """Extract crops from polys using the same helper Hybrid uses in prod.

    NOTE: pass RGB (not BGR) — the rec model was trained on RGB, and both
    backends have been updated to match. Feeding BGR costs ~3% accuracy on
    ambiguous chars (e.g. Darah↔Daran).
    """
    from src.services.ocr_backends import _crop_text_region
    image_rgb = np.ascontiguousarray(image_rgb)
    return [_crop_text_region(image_rgb, p) for p in polys]


def _recognize_crops_on_backend(backend: Any, crops: list[np.ndarray]) -> list[Line]:
    """Call the backend's recognizer directly on pre-extracted crops.

    For Hybrid: uses ``_fast_recognize`` (its production rec path).
    For AutoKernel: uses ``system.text_recognizer`` (predict_rec.TextRecognizer).
    """
    if not crops:
        return []
    fast = getattr(backend, "_fast_recognize", None)
    if callable(fast):
        results = fast(crops)
        # hybrid's _fast_recognize returns list[(text, score)]
        return [
            Line(text=str(t), score=float(s), poly=np.zeros((4, 2)))
            for (t, s) in results
        ]
    system = getattr(backend, "system", None)
    if system is not None and hasattr(system, "text_recognizer"):
        rec_results, _ = system.text_recognizer(crops)
        return [
            Line(text=str(r[0]), score=float(r[1]), poly=np.zeros((4, 2)))
            for r in rec_results
        ]
    raise RuntimeError(f"Backend {backend.__class__.__name__} has no rec-only entry point")


def run_shared_mode(images: list[Path], backends: list[str]) -> list[ImageReport]:
    """Shared-crops pipeline: Paddle does det+rec; other backends rec-only."""
    if "paddle" not in backends:
        raise RuntimeError("Shared mode requires paddle in --only (it provides crops)")

    prebuilt: dict[str, Any] = {}
    for b in [x for x in backends if x != "paddle"]:
        print(f"\n=== Prebuild: {b} ===")
        try:
            prebuilt[b] = BACKEND_BUILDERS[b]()
        except Exception as exc:
            print(f"  FAILED to build {b}: {exc}")
            prebuilt[b] = exc

    # ---- Pass 1: Paddle — collect polys + texts + cache crops per image ----
    print("\n=== Pass 1/3: paddle (det + rec; polys will be reused) ===")
    paddle = _build_paddle_backend()
    paddle_data: dict[Path, tuple[list[np.ndarray], list[Line], float, np.ndarray]] = {}
    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img_path.name}...", end=" ", flush=True)
        img_rgb = load_rgb(img_path)
        if i == 1:
            try: paddle.predict(img_rgb)
            except Exception: pass
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        raw = paddle.predict(img_rgb)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        lines = _result_to_lines(raw)
        polys = [l.poly for l in lines]
        paddle_data[img_path] = (polys, lines, ms, img_rgb)
        print(f"{len(lines):3d} lines, {ms:.1f} ms")
    close = getattr(paddle, "close", None)
    if callable(close):
        try: close()
        except Exception: pass
    del paddle
    gc.collect(); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    try: torch._dynamo.reset()
    except Exception: pass

    # ---- Pass 2/3: run hybrid/autokernel recognizer-only on same crops ----
    other_results: dict[str, dict[Path, tuple[list[Line], float]]] = {}
    for b in [x for x in backends if x != "paddle"]:
        print(f"\n=== Pass: {b} (rec-only on Paddle crops) ===")
        backend = prebuilt.get(b)
        if isinstance(backend, Exception):
            exc = backend
            print(f"  FAILED to build {b}: {exc}")
            other_results[b] = {p: ([], 0.0) for p in images}
            continue
        cache: dict[Path, tuple[list[Line], float]] = {}
        for i, img_path in enumerate(images, 1):
            polys, _paddle_lines, _pms, img_rgb = paddle_data[img_path]
            crops = _extract_shared_crops(img_rgb, polys)
            print(f"  [{i}/{len(images)}] {img_path.name} ({len(crops)} crops)...", end=" ", flush=True)
            if i == 1 and crops:
                try: _recognize_crops_on_backend(backend, crops[:1])
                except Exception: pass
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                rec_lines = _recognize_crops_on_backend(backend, crops)
            except Exception as exc:
                print(f"rec failed: {exc}")
                cache[img_path] = ([], 0.0)
                continue
            if torch.cuda.is_available(): torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            # reattach polys so the report keeps bbox info
            for rl, poly in zip(rec_lines, polys):
                rl.poly = poly
            cache[img_path] = (rec_lines, ms)
            print(f"{len(rec_lines):3d} lines, {ms:.1f} ms")
        other_results[b] = cache
        closer = getattr(backend, "close", None)
        if callable(closer):
            try: closer()
            except Exception: pass
        del backend
        gc.collect(); gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.synchronize()
        try: torch._dynamo.reset()
        except Exception: pass

    # ---- Assemble reports with positional alignment (crops are identical) ----
    reports: list[ImageReport] = []
    for img_path in images:
        polys, paddle_lines, paddle_ms, _ = paddle_data[img_path]
        results: dict[str, BackendResult] = {
            "paddle": BackendResult(backend="paddle", lines=paddle_lines, elapsed_ms=paddle_ms)
        }
        for b in [x for x in backends if x != "paddle"]:
            rec_lines, ms = other_results[b].get(img_path, ([], 0.0))
            results[b] = BackendResult(backend=b, lines=rec_lines, elapsed_ms=ms)
        reports.append(ImageReport(image_path=img_path, results=results))
    return reports


def format_shared_image(report: ImageReport, backends: list[str]) -> list[str]:
    """Positional comparison (same crop index) for shared-crops mode."""
    out: list[str] = []
    w = out.append
    w(f"Image: {report.image_path.name}")
    for b in backends:
        r = report.results.get(b)
        if r and not r.error:
            w(f"  {b:<11}: {len(r.lines):3d} lines,  {r.elapsed_ms:7.1f} ms")
        else:
            w(f"  {b:<11}: ERROR {r.error if r else '<missing>'}")

    ref = report.results["paddle"].lines
    others = [b for b in backends if b != "paddle"]
    col_w = 28

    header = [f"{'#':>3}", f"{'paddle (ref)':<{col_w}}"]
    for b in others:
        header.append(f"{b:<{col_w}}")
    header.append("diff")
    w("  " + "  ".join(header))
    w("  " + "  ".join(["─" * 3, *(["─" * col_w] * (1 + len(others))), "─" * 6]))

    tallies = {b: 0 for b in others}
    n = len(ref)
    for i, rl in enumerate(ref):
        row = [f"{i:3d}", f"{_abbrev(rl.text, col_w):<{col_w}}"]
        flags: list[str] = []
        for b in others:
            r = report.results[b]
            if i < len(r.lines):
                other_text = r.lines[i].text
            else:
                other_text = "<missing>"
            row.append(f"{_abbrev(other_text, col_w):<{col_w}}")
            if other_text == rl.text:
                tallies[b] += 1
            else:
                flags.append(f"{b}≠")
        row.append(" ".join(flags) if flags else "✓")
        w("  " + "  ".join(row))

    w("")
    w(f"  match vs paddle (ref, {n} lines):")
    for b in others:
        pct = 100.0 * tallies[b] / n if n else 0.0
        w(f"      {b:<11}: {tallies[b]:3d}/{n} exact  ({pct:5.1f}%)")
    return out


def format_shared_report(reports: list[ImageReport], backends: list[str]) -> str:
    out: list[str] = []
    w = out.append
    w("=" * 90)
    w("  OCR line-by-line comparison — SHARED-CROPS mode")
    w("  Paddle runs det+rec once; its polys + cv2 crops are reused by all other")
    w("  backends' recognizer-only paths. Detector, crop extraction, and")
    w("  post-processing are identical across columns.")
    w(f"  Backends: {'  vs  '.join(backends)}")
    w("=" * 90)
    w("")
    for r in reports:
        out.extend(format_shared_image(r, backends))
        w("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _result_to_lines(result: Any) -> list[Line]:
    """Normalise PaddleOCR-compatible result dict(s) into a list of Line."""
    if not result:
        return []
    first = result[0]
    if first is None:
        return []

    # Some PaddleOCR variants return a Result object where items are accessed
    # via __getitem__ rather than dict.get. Try .get, fall back to subscript.
    def _get(obj, key, default=None):
        getter = getattr(obj, "get", None)
        if callable(getter):
            return getter(key, default)
        try:
            return obj[key]
        except (KeyError, TypeError):
            return default

    texts = _get(first, "rec_texts", []) or []
    scores = _get(first, "rec_scores", []) or []
    polys = _get(first, "rec_polys", None)
    if polys is None:
        polys = _get(first, "dt_polys", []) or []

    lines: list[Line] = []
    for t, s, p in zip(texts, scores, polys):
        p_np = np.asarray(p, dtype=np.float64)
        if p_np.ndim == 1:
            p_np = p_np.reshape(-1, 2)
        lines.append(Line(text=str(t), score=float(s), poly=p_np))
    return lines


def run_backend(backend_name: str, images: list[Path]) -> dict[Path, BackendResult]:
    """Build a backend, run it on every image, tear down, free GPU."""
    print(f"\n=== Pass: {backend_name} ===")
    print(f"  Building backend...")
    if backend_name in {"hybrid", "autokernel", "fullpytorch"}:
        print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    t0 = time.perf_counter()
    try:
        backend = BACKEND_BUILDERS[backend_name]()
    except Exception as exc:
        print(f"  FAILED to build {backend_name}: {exc}")
        return {p: BackendResult(backend=backend_name, error=str(exc)) for p in images}
    print(f"  Ready in {time.perf_counter() - t0:.1f}s")

    cache: dict[Path, BackendResult] = {}
    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img_path.name}...", end=" ", flush=True)
        img_rgb = load_rgb(img_path)
        # Warm up once per image on the first image only
        if i == 1:
            try:
                backend.predict(img_rgb)
            except Exception as exc:
                print(f"warmup failed: {exc}")
                cache[img_path] = BackendResult(backend=backend_name, error=str(exc))
                continue

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            raw = backend.predict(img_rgb)
        except Exception as exc:
            print(f"predict failed: {exc}")
            cache[img_path] = BackendResult(backend=backend_name, error=str(exc))
            continue
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        lines = _result_to_lines(raw)
        cache[img_path] = BackendResult(backend=backend_name, lines=lines, elapsed_ms=ms)
        print(f"{len(lines):3d} lines, {ms:.1f} ms")

    # Teardown with aggressive GPU cleanup (critical before loading a backend
    # that uses CUDA graph capture — stale memory corrupts graph replay).
    close = getattr(backend, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            print(f"  close() raised: {exc}")
    del backend
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    try:
        torch._dynamo.reset()
    except Exception:
        pass

    return cache


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _match_by_centroid(
    ref: list[Line],
    other: list[Line],
    tol: float = CENTROID_MATCH_PX,
) -> list[int | None]:
    """For each ref line, return index into `other` (or None)."""
    used: set[int] = set()
    matches: list[int | None] = []
    ref_cs = np.array([l.centroid for l in ref]) if ref else np.empty((0, 2))
    oth_cs = np.array([l.centroid for l in other]) if other else np.empty((0, 2))
    for i, rc in enumerate(ref_cs):
        best_j, best_d = None, float("inf")
        for j, oc in enumerate(oth_cs):
            if j in used:
                continue
            d = float(np.linalg.norm(rc - oc))
            if d < best_d:
                best_d, best_j = d, j
        if best_j is not None and best_d <= tol:
            used.add(best_j)
            matches.append(best_j)
        else:
            matches.append(None)
    return matches


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _abbrev(s: str, w: int) -> str:
    return s if len(s) <= w else s[: w - 1] + "…"


def format_image(report: ImageReport, backends: list[str]) -> list[str]:
    lines: list[str] = []
    w = lines.append

    w(f"Image: {report.image_path.name}")
    for b in backends:
        r = report.results.get(b)
        if r is None or r.error:
            w(f"  {b:<11}: ERROR {r.error if r else '<missing>'}")
        else:
            w(f"  {b:<11}: {len(r.lines):3d} lines,  {r.elapsed_ms:7.1f} ms")

    # Choose reference backend: first one without error with lines
    ref_name = None
    for b in backends:
        r = report.results.get(b)
        if r and not r.error and r.lines:
            ref_name = b
            break
    if ref_name is None:
        w("  (no backend produced lines)")
        return lines

    ref = report.results[ref_name].lines
    others = [b for b in backends if b != ref_name]

    # Match other backends to ref
    match_map: dict[str, list[int | None]] = {}
    for b in others:
        r = report.results.get(b)
        if r is None or r.error:
            match_map[b] = [None] * len(ref)
        else:
            match_map[b] = _match_by_centroid(ref, r.lines)

    # Column width budgeting
    col_w = 28
    header_cols = [f"{'#':>3}", f"{ref_name+' (ref)':<{col_w}}"]
    for b in others:
        header_cols.append(f"{b:<{col_w}}")
    header_cols.append("diff")
    w("  " + "  ".join(header_cols))
    w("  " + "  ".join(["─" * 3, *(["─" * col_w] * (1 + len(others))), "─" * 6]))

    total_cmp = 0
    total_match = {b: 0 for b in others}
    for i, rl in enumerate(ref):
        row = [f"{i:3d}", f"{_abbrev(rl.text, col_w):<{col_w}}"]
        flags: list[str] = []
        for b in others:
            r = report.results.get(b)
            j = match_map[b][i]
            if r is None or r.error:
                row.append(f"{'<err>':<{col_w}}")
                flags.append(f"{b}!")
                continue
            if j is None:
                row.append(f"{'<no-match>':<{col_w}}")
                flags.append(f"{b}∅")
                continue
            ol = r.lines[j]
            row.append(f"{_abbrev(ol.text, col_w):<{col_w}}")
            if ol.text != rl.text:
                flags.append(f"{b}≠")
            else:
                total_match[b] += 1
        total_cmp += 1
        diff = " ".join(flags) if flags else "✓"
        row.append(diff)
        w("  " + "  ".join(row))

    # Orphans: lines in non-ref backends that weren't matched to any ref line
    for b in others:
        r = report.results.get(b)
        if r is None or r.error:
            continue
        matched_j = {j for j in match_map[b] if j is not None}
        orphans = [(j, l) for j, l in enumerate(r.lines) if j not in matched_j]
        if orphans:
            w(f"  [only in {b}] {len(orphans)} orphan line(s):")
            for j, ol in orphans:
                w(f"      - idx {j:3d}  {_abbrev(ol.text, 60)}")

    # Per-image summary
    w("")
    w(f"  match vs {ref_name} (ref, {total_cmp} lines):")
    for b in others:
        r = report.results.get(b)
        if r is None or r.error:
            continue
        pct = 100.0 * total_match[b] / total_cmp if total_cmp else 0.0
        w(f"      {b:<11}: {total_match[b]:3d}/{total_cmp} exact  ({pct:5.1f}%)")

    return lines


def format_report(reports: list[ImageReport], backends: list[str]) -> str:
    out: list[str] = []
    w = out.append
    w("=" * 90)
    w("  OCR line-by-line comparison (end-to-end; each backend runs own det+rec)")
    w(f"  Backends: {'  vs  '.join(backends)}")
    w(f"  Alignment: centroid distance ≤ {CENTROID_MATCH_PX:.0f} px")
    w("=" * 90)
    w("")
    for r in reports:
        out.extend(format_image(r, backends))
        w("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None,
                        help="Single image or directory (defaults to tests/data)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write report to file")
    parser.add_argument(
        "--only", type=str, default="paddle,hybrid,autokernel",
        help="Comma-separated backends to run (subset of: paddle, hybrid, autokernel)",
    )
    parser.add_argument(
        "--mode", choices=("shared", "e2e"), default="shared",
        help="shared: Paddle det+crops once, other backends rec-only on same crops "
             "(apples-to-apples). e2e: each backend runs full det+rec independently.",
    )
    parser.add_argument(
        "--paddle-rec-dir", type=str,
        default="/home/jupyter/gisa_playground/OCR/deploy/rec_model/051025_data_additional_7900_v2_latest",
        help="Directory holding the finetuned inference.pdiparams that the "
             "PyTorch .pth was converted from. Forces Paddle to use the same weights.",
    )
    parser.add_argument(
        "--paddle-config", type=str, default=None,
        help="Path to a paddlex_config YAML for the baseline Paddle. Use this "
             "to match the project's shipping det preprocessing/thresholds "
             "(e.g. PaddleOCR_server_nohpi.yaml) instead of the stock defaults.",
    )
    args = parser.parse_args()
    global PADDLE_REC_MODEL_DIR, PADDLE_CONFIG_PATH
    PADDLE_REC_MODEL_DIR = args.paddle_rec_dir
    PADDLE_CONFIG_PATH = args.paddle_config

    backends = [b.strip().lower() for b in args.only.split(",") if b.strip()]
    for b in backends:
        if b not in BACKEND_BUILDERS:
            sys.exit(f"Unknown backend: {b}")

    src = args.image or PROJECT_ROOT / "tests" / "data"
    images = collect_images(src)
    if not images:
        sys.exit(f"No images found at {src}")
    print(f"Found {len(images)} image(s); running {len(backends)} backend(s): {backends}")

    if args.mode == "shared":
        reports = run_shared_mode(images, backends)
        text = format_shared_report(reports, backends)
    else:
        # Order passes: cheapest-to-cleanup first, autokernel (CUDA graphs) last.
        order = {"paddle": 0, "hybrid": 1, "autokernel": 2, "fullpytorch": 3}
        passes = sorted(backends, key=lambda b: order[b])

        per_backend: dict[str, dict[Path, BackendResult]] = {}
        for b in passes:
            per_backend[b] = run_backend(b, images)

        reports = [
            ImageReport(
                image_path=p,
                results={b: per_backend[b].get(p, BackendResult(backend=b, error="<missing>")) for b in backends},
            )
            for p in images
        ]

        text = format_report(reports, backends)
    print()
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"\nReport saved to {args.output}")

    # Exit 1 if any non-paddle backend has any mismatch against paddle (if paddle was run)
    if "paddle" in backends and len(backends) > 1:
        had_diff = False
        for r in reports:
            ref = r.results.get("paddle")
            if ref is None or ref.error or not ref.lines:
                continue
            for b in backends:
                if b == "paddle":
                    continue
                other = r.results.get(b)
                if other is None or other.error:
                    had_diff = True
                    break
                m = _match_by_centroid(ref.lines, other.lines)
                for i, j in enumerate(m):
                    if j is None or other.lines[j].text != ref.lines[i].text:
                        had_diff = True
                        break
        sys.exit(1 if had_diff else 0)


if __name__ == "__main__":
    main()
