#!/usr/bin/env python3
"""Pinpoint where the PyTorch rec output drifts from the Paddle rec output.

Strategy:
  1. Pick a crop that Paddle recognises correctly but PyTorch gets wrong.
  2. Run three preprocessing paths (Paddle cv2, PyTorch cv2 via TextRecognizer,
     Hybrid F.interpolate); diff the tensors element-wise.
  3. Load the converted .pth and (when available) the original .pdiparams;
     diff every shared-name tensor element-wise.
  4. Run the PyTorch model on each preprocessed tensor and report text +
     logits-argmax at the diverging timestep.

Default crop: image 2, box index 10 ("Ggl Darah").
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "PaddleOCR2Pytorch"))

logging.basicConfig(level=logging.WARNING)
for n in ("ppocr", "paddle", "paddleocr", "torch", "triton"):
    logging.getLogger(n).setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Paddle-side preprocessing (exactly mirrors paddlex.processors.resize_norm_img)
# ---------------------------------------------------------------------------

def paddle_preprocess(crop_bgr: np.ndarray, imgH: int = 48, imgW: int = 320) -> np.ndarray:
    """Return (C, H, W) float32 tensor padded to imgW."""
    imgC = 3
    h, w = crop_bgr.shape[:2]
    max_wh_ratio = max(imgW / imgH, w / h)
    target_w = int(imgH * max_wh_ratio)
    target_w = max(min(target_w, 4000), 16)

    ratio = w / float(h)
    resized_w = min(int(np.ceil(imgH * ratio)), target_w)
    resized_w = max(resized_w, 16)

    resized = cv2.resize(crop_bgr, (resized_w, imgH))
    resized = resized.astype("float32").transpose((2, 0, 1)) / 255.0
    resized = (resized - 0.5) / 0.5
    out = np.zeros((imgC, imgH, target_w), dtype=np.float32)
    out[:, :, :resized_w] = resized
    return out


def hybrid_preprocess(crop_bgr: np.ndarray, imgH: int = 48, imgW: int = 320,
                      dtype: torch.dtype = torch.float32) -> np.ndarray:
    """Replicate HybridOCRBackend._fast_recognize's F.interpolate path."""
    h, w = crop_bgr.shape[:2]
    max_wh_ratio = max(imgW / imgH, w / h)
    target_w = int(imgH * max_wh_ratio)
    target_w = max(min(target_w, 4000), 16)

    ratio = w / float(h)
    resized_w = min(int(np.ceil(imgH * ratio)), target_w)
    resized_w = max(resized_w, 16)

    t = torch.from_numpy(crop_bgr.copy()).to(dtype=torch.float32)
    t = t.permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(imgH, resized_w), mode="bilinear", align_corners=False)
    t = t.squeeze(0)
    t = (t / 255.0 - 0.5) / 0.5
    out = torch.zeros(3, imgH, target_w, dtype=torch.float32)
    out[:, :, :resized_w] = t
    return out.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Detection (use PaddleOCR once to get polys + Paddle's own texts)
# ---------------------------------------------------------------------------

_CAPTURED: dict[str, list] = {"inputs": [], "crops": []}


def _install_paddle_preprocess_spy() -> None:
    """Monkey-patch paddlex rec preprocessing to capture every tensor that
    goes into the model and every raw crop.
    """
    try:
        from paddlex.inference.models.text_recognition import processors as proc
    except Exception as exc:
        print(f"  spy install failed: {exc}")
        return

    # Spy on resize_norm_img — captures the preprocessed (C,H,W) float32 tensor
    original_rni = proc.OCRReisizeNormImg.resize_norm_img
    def _spy_rni(self, img, max_wh_ratio):
        out = original_rni(self, img, max_wh_ratio)
        _CAPTURED["inputs"].append({"input_tensor": np.asarray(out).copy(),
                                    "max_wh_ratio": float(max_wh_ratio),
                                    "crop_hw": tuple(img.shape[:2])})
        _CAPTURED["crops"].append(np.asarray(img).copy())
        return out
    proc.OCRReisizeNormImg.resize_norm_img = _spy_rni


def detect_and_paddle_texts(
    image_rgb: np.ndarray, rec_model_dir: str
) -> tuple[list[np.ndarray], list[str]]:
    from paddleocr import PaddleOCR

    _install_paddle_preprocess_spy()

    engine = PaddleOCR(
        ocr_version="PP-OCRv5",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_hpi=False,
        text_recognition_model_dir=rec_model_dir,
    )
    result = engine.predict(image_rgb)
    first = result[0]
    texts = list(first["rec_texts"])
    polys = [np.asarray(p, dtype=np.float64) for p in first["rec_polys"]]
    close = getattr(engine, "close", None)
    if callable(close):
        try: close()
        except Exception: pass
    return polys, texts


def crop_perspective(image_bgr: np.ndarray, poly: np.ndarray) -> np.ndarray:
    from src.services.ocr_backends import _crop_text_region
    return _crop_text_region(image_bgr, poly)


# ---------------------------------------------------------------------------
# PyTorch rec model
# ---------------------------------------------------------------------------

def _default_torch_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_pt_recognizer(
    rec_weights_path: Path,
    ppocr_root: Path,
    device: str | None = None,
):
    device = device or _default_torch_device()
    os.environ.setdefault("AUTOKERNEL_PPOCR_ROOT", str(ppocr_root))
    os.environ.setdefault("AUTOKERNEL_PPOCRV5_SERVER_REC_PTH", str(rec_weights_path))
    sys.path.insert(0, str(PROJECT_ROOT / "autokernel"))

    import importlib
    ppocr_model = importlib.import_module("models.ppocrv5_server")
    pytorchocr_utility = importlib.import_module("tools.infer.pytorchocr_utility")
    predict_rec = importlib.import_module("tools.infer.predict_rec")

    parser = pytorchocr_utility.init_args()
    args = parser.parse_args([])
    args.use_gpu = False
    args.rec_algorithm = "SVTR_PPOCRv5"
    args.rec_yaml_path = str(ppocr_root / ppocr_model.REC_YAML_RELATIVE)
    args.rec_model_path = str(rec_weights_path)
    args.rec_image_shape = "3,48,320"
    args.rec_char_dict_path = str(ppocr_root / "pytorchocr/utils/dict/ppocrv5_dict.txt")
    args.rec_batch_num = 1
    args.image_dir = ""

    rec = predict_rec.TextRecognizer(args)
    rec.use_gpu = device == "cuda"

    rec_wrapper = ppocr_model.PPOCRv5ServerRecModel()
    rec.net = rec_wrapper.to(device=device, dtype=torch.float32).eval()
    rec._debug_device = device
    return rec


def run_pt_model_on_tensor(rec, padded_nchw: np.ndarray):
    """Skip rec.__call__'s batching and just run the model on a (C,H,W) tensor."""
    device = getattr(rec, "_debug_device", _default_torch_device())
    t = torch.from_numpy(padded_nchw).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        logits = rec.net(t)
    logits_np = logits.float().cpu().numpy()
    preds_idx = logits_np.argmax(axis=-1)
    preds_prob = logits_np.max(axis=-1)
    # reuse its postprocess_op
    texts = rec.postprocess_op.decode(preds_idx, preds_prob, is_remove_duplicate=True)
    return texts[0][0], texts[0][1], logits_np


# ---------------------------------------------------------------------------
# Weight drift
# ---------------------------------------------------------------------------

def _load_paddle_inference_params(model_dir: Path) -> dict:
    """Load a Paddle *combined-inference* model via jit.load, returning a
    name→ndarray dict of parameters. Works for the modern `inference.json +
    inference.pdiparams` format.
    """
    import paddle
    # jit.load expects the prefix (no extension). New-format prefix is the
    # directory containing "inference.*".
    prefix = str(model_dir / "inference")
    layer = paddle.jit.load(prefix)
    state = {}
    for name, tensor in layer.state_dict().items():
        state[name] = tensor.numpy()
    return state


def _paddle_to_pytorch_key(pd_key: str) -> str:
    """Mirror the name mapping used by PaddleOCR2Pytorch converters."""
    k = pd_key
    k = k.replace("._mean", ".running_mean")
    k = k.replace("._variance", ".running_var")
    return k


def compare_weights(pth_path: Path, model_dir: Path):
    """Compare PyTorch .pth against Paddle inference params, pairing by name."""
    print(f"\n=== Weight drift ===")
    print(f"  pth         : {pth_path}")
    print(f"  paddle dir  : {model_dir}")

    pth_sd = torch.load(pth_path, map_location="cpu", weights_only=False)
    if isinstance(pth_sd, dict) and "state_dict" in pth_sd:
        pth_sd = pth_sd["state_dict"]
    print(f"  pth tensors : {len(pth_sd)}")

    try:
        pd_sd = _load_paddle_inference_params(model_dir)
    except Exception as exc:
        print(f"  paddle.jit.load failed: {exc}")
        return
    print(f"  paddle tensors : {len(pd_sd)}")

    pt_keys = set(pth_sd.keys())
    mapped = {_paddle_to_pytorch_key(k): k for k in pd_sd.keys()}
    shared = sorted(pt_keys & mapped.keys())
    missing_in_pt = sorted(set(mapped.keys()) - pt_keys)
    missing_in_pd = sorted(pt_keys - set(mapped.keys()))
    print(f"  matched keys     : {len(shared)}")
    print(f"  only in paddle   : {len(missing_in_pt)}"
          + (f"  ({missing_in_pt[:3]}…)" if missing_in_pt else ""))
    print(f"  only in pytorch  : {len(missing_in_pd)}"
          + (f"  ({missing_in_pd[:3]}…)" if missing_in_pd else ""))

    if not shared:
        return

    diffs = []
    for pt_k in shared:
        pd_k = mapped[pt_k]
        pt_np = pth_sd[pt_k].detach().cpu().numpy()
        pd_np = np.asarray(pd_sd[pd_k])
        # Handle Paddle→PyTorch linear weight transpose (Paddle stores in×out,
        # PyTorch stores out×in). Only swap if it would make shapes match.
        if pt_np.shape != pd_np.shape and pt_np.ndim == 2 and pd_np.ndim == 2 \
                and pt_np.shape == pd_np.shape[::-1]:
            pd_np = pd_np.T
        if pt_np.shape != pd_np.shape:
            diffs.append((pt_k, None, None, f"shape mismatch {pt_np.shape} vs {pd_np.shape}"))
            continue
        d = np.abs(pt_np.astype(np.float64) - pd_np.astype(np.float64))
        diffs.append((pt_k, float(d.max()), float(d.mean()), None))

    # Summarize
    ok = [d for d in diffs if d[3] is None]
    shape_mismatch = [d for d in diffs if d[3] is not None]
    print(f"  exact-shape     : {len(ok)}")
    print(f"  shape-mismatch  : {len(shape_mismatch)}")
    if shape_mismatch:
        for n, _, _, msg in shape_mismatch[:5]:
            print(f"    {n}: {msg}")

    if ok:
        max_diffs = sorted(ok, key=lambda x: x[1], reverse=True)
        print(f"\n  Top-20 tensors by max abs diff:")
        print(f"    {'tensor':60s}  {'max':>10s}  {'mean':>10s}")
        for n, mx, mn, _ in max_diffs[:20]:
            print(f"    {n[:60]:60s}  {mx:10.3e}  {mn:10.3e}")

        all_max = np.array([d[1] for d in ok])
        all_mean = np.array([d[2] for d in ok])
        print(f"\n  Overall : max-abs-diff max={all_max.max():.3e}  "
              f"mean={all_max.mean():.3e}   mean-abs-diff max={all_mean.max():.3e}  "
              f"mean={all_mean.mean():.3e}")
        print(f"  tensors with max-diff > 1e-5 : "
              f"{int((all_max > 1e-5).sum())} / {len(ok)}")
        print(f"  tensors with max-diff > 1e-3 : "
              f"{int((all_max > 1e-3).sum())} / {len(ok)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", default=str(PROJECT_ROOT / "tests/data/good_data_2.png"))
    ap.add_argument("--crop-idx", type=int, default=10,
                    help="Index of the bad crop (default 10 = 'Ggl Darah' on good_data_2)")
    ap.add_argument("--rec-dir", default="/home/jupyter/gisa_playground/OCR/deploy/rec_model/051025_data_additional_7900_v2_latest")
    ap.add_argument("--pth", default=str(PROJECT_ROOT / "autokernel/workspace/ppocrv5/server_rec.pth"))
    ap.add_argument("--ppocr-root", default=str(PROJECT_ROOT / "PaddleOCR2Pytorch"))
    args = ap.parse_args()

    img_path = Path(args.image)
    rec_dir = args.rec_dir
    pth = Path(args.pth)
    ppocr = Path(args.ppocr_root)

    print(f"Image    : {img_path}")
    print(f"Crop idx : {args.crop_idx}")

    img_rgb = np.asarray(Image.open(img_path).convert("RGB"))
    img_bgr = np.ascontiguousarray(img_rgb[:, :, ::-1])

    polys, paddle_texts = detect_and_paddle_texts(img_rgb, rec_dir)
    if args.crop_idx >= len(polys):
        sys.exit(f"crop_idx {args.crop_idx} out of range ({len(polys)} polys)")

    poly = polys[args.crop_idx]
    paddle_text = paddle_texts[args.crop_idx]
    print(f"\nPaddle says (crop {args.crop_idx}): {paddle_text!r}")

    crop = crop_perspective(img_bgr, poly)
    print(f"Crop shape: {crop.shape}")

    # ---- What did PaddleOCR actually feed to its rec model for this crop? ----
    print(f"\n=== Captured from PaddleOCR internal pipeline ===")
    print(f"  total crops seen by rec : {len(_CAPTURED['inputs'])}")
    # Best heuristic to pair: match by crop H/W (Paddle's crops may be cropped
    # differently than _crop_text_region but should be close for a given poly).
    # Pick the one closest in (h, w).
    best = None
    best_j = None
    for j, rec_info in enumerate(_CAPTURED["inputs"]):
        h, w = rec_info["crop_hw"]
        score = abs(h - crop.shape[0]) + abs(w - crop.shape[1])
        if best is None or score < best[0]:
            best = (score, rec_info)
            best_j = j
    paddle_internal_tensor = None
    paddle_internal_crop = None
    if best is not None:
        info = best[1]
        paddle_internal_tensor = info["input_tensor"]
        paddle_internal_crop = _CAPTURED["crops"][best_j]
        print(f"  matched internal crop: orig hw={info['crop_hw']}, input_tensor"
              f" shape={paddle_internal_tensor.shape}, max_wh_ratio={info['max_wh_ratio']:.3f}")
        print(f"\n  ==> RAW CROP diff (paddle's vs ours):")
        print(f"      paddle raw crop shape : {paddle_internal_crop.shape}")
        print(f"      ours raw crop shape   : {crop.shape}")
        if paddle_internal_crop.shape == crop.shape:
            # Note: both should be RGB after our fix
            d_bgr = np.abs(paddle_internal_crop.astype(np.float64) - crop.astype(np.float64))
            d_rgb = np.abs(paddle_internal_crop.astype(np.float64) - crop[:, :, ::-1].astype(np.float64))
            print(f"      |paddle_crop - ours_BGR| max={d_bgr.max():.1f}  mean={d_bgr.mean():.2f}")
            print(f"      |paddle_crop - ours_RGB| max={d_rgb.max():.1f}  mean={d_rgb.mean():.2f}")
            # Per-channel means of raw crop (before any normalization)
            print(f"      paddle raw per-ch mean: {paddle_internal_crop.reshape(-1,3).mean(axis=0)}")
            print(f"      ours   raw per-ch mean: {crop.reshape(-1,3).mean(axis=0)}")
        else:
            print(f"      SHAPE MISMATCH in raw crop")

    # ---- Our own preprocessing ----
    t_paddle = paddle_preprocess(crop)
    t_hybrid = hybrid_preprocess(crop)

    print(f"\n=== Preprocessing tensor shapes ===")
    print(f"  paddle cv2 : {t_paddle.shape}  dtype={t_paddle.dtype}  "
          f"min={t_paddle.min():.3f} max={t_paddle.max():.3f}")
    print(f"  hybrid F.i : {t_hybrid.shape}  dtype={t_hybrid.dtype}  "
          f"min={t_hybrid.min():.3f} max={t_hybrid.max():.3f}")

    if t_paddle.shape == t_hybrid.shape:
        diff = np.abs(t_paddle - t_hybrid)
        print(f"  |paddle-hybrid| tensor: max={diff.max():.4e}  mean={diff.mean():.4e}  "
              f"frac>1e-2={(diff>1e-2).mean():.3f}")
    else:
        print(f"  shape mismatch — cannot diff directly")

    if paddle_internal_tensor is not None:
        print(f"\n  paddle internal vs paddle_preprocess (ours):")
        if paddle_internal_tensor.shape == t_paddle.shape:
            d = np.abs(paddle_internal_tensor - t_paddle)
            print(f"    shape OK, |diff| max={d.max():.4e}  mean={d.mean():.4e}")
            # Per-channel means
            print(f"    internal  per-ch mean: "
                  f"{paddle_internal_tensor.mean(axis=(1,2))}")
            print(f"    ours(BGR) per-ch mean: {t_paddle.mean(axis=(1,2))}")
            # Try channel-reversed version
            t_paddle_rgb = paddle_preprocess(crop[:, :, ::-1])  # swap channels first
            d_rgb = np.abs(paddle_internal_tensor - t_paddle_rgb)
            print(f"    ours(RGB) per-ch mean: {t_paddle_rgb.mean(axis=(1,2))}")
            print(f"    |internal - ours(RGB)| max={d_rgb.max():.4e}  mean={d_rgb.mean():.4e}")
        else:
            print(f"    SHAPE DIFFERS: internal={paddle_internal_tensor.shape}  ours={t_paddle.shape}")

    # ---- Run the PyTorch model on each preprocessed tensor ----
    print(f"\n=== PyTorch model output on each preprocessed tensor ===")
    rec = build_pt_recognizer(pth, ppocr)
    device = getattr(rec, "_debug_device", _default_torch_device())

    # Since the real backends now crop from RGB (post-fix), test the RGB
    # variants too — that mirrors the actual production path.
    crop_rgb = crop[:, :, ::-1].copy()
    t_paddle_rgb = paddle_preprocess(crop_rgb)
    t_hybrid_rgb_cpu = hybrid_preprocess(crop_rgb)

    txt_a, scr_a, logits_a = run_pt_model_on_tensor(rec, t_paddle)
    txt_b, scr_b, logits_b = run_pt_model_on_tensor(rec, t_hybrid)
    txt_a_rgb, scr_a_rgb, _ = run_pt_model_on_tensor(rec, t_paddle_rgb)
    txt_b_rgb, scr_b_rgb, _ = run_pt_model_on_tensor(rec, t_hybrid_rgb_cpu)

    # Also test hybrid's CUDA F.interpolate path — this is what `_fast_recognize`
    # actually runs in production.
    def _hybrid_preprocess_cuda(crop_bgr_or_rgb, imgH=48, imgW=320):
        if device != "cuda":
            raise RuntimeError("CUDA unavailable")
        h, w = crop_bgr_or_rgb.shape[:2]
        max_wh_ratio = max(imgW / imgH, w / h)
        target_w = int(imgH * max_wh_ratio)
        target_w = max(min(target_w, 4000), 16)
        ratio = w / float(h)
        resized_w = min(int(np.ceil(imgH * ratio)), target_w)
        resized_w = max(resized_w, 16)
        t = torch.from_numpy(crop_bgr_or_rgb.copy()).to(device="cuda", dtype=torch.float32)
        t = t.permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=(imgH, resized_w), mode="bilinear", align_corners=False)
        t = t.squeeze(0)
        t = (t / 255.0 - 0.5) / 0.5
        out = torch.zeros(3, imgH, target_w, device="cuda", dtype=torch.float32)
        out[:, :, :resized_w] = t
        return out.cpu().numpy().astype(np.float32)

    if device == "cuda":
        t_hybrid_rgb_cuda = _hybrid_preprocess_cuda(crop_rgb)
        txt_b_cuda, scr_b_cuda, _ = run_pt_model_on_tensor(rec, t_hybrid_rgb_cuda)
    else:
        txt_b_cuda, scr_b_cuda = "<cuda unavailable>", 0.0

    if paddle_internal_tensor is not None:
        txt_c, scr_c, logits_c = run_pt_model_on_tensor(rec, paddle_internal_tensor)
        print(f"  paddle-INTERNAL tensor → PyTorch model : {txt_c!r}  (score {scr_c:.3f})")

    print(f"  cv2 on BGR (ours)  → PT model : {txt_a!r}  (score {scr_a:.3f})")
    print(f"  F.i CPU on BGR     → PT model : {txt_b!r}  (score {scr_b:.3f})")
    print(f"  cv2 on RGB (ours)  → PT model : {txt_a_rgb!r}  (score {scr_a_rgb:.3f})")
    print(f"  F.i CPU on RGB     → PT model : {txt_b_rgb!r}  (score {scr_b_rgb:.3f})")
    print(f"  F.i CUDA on RGB    → PT model : {txt_b_cuda!r}  (score {scr_b_cuda:.3f})  ← full hybrid uses this")
    print(f"  Paddle-native (ground truth)  : {paddle_text!r}")

    # Diff logits
    if logits_a.shape == logits_b.shape:
        d = np.abs(logits_a - logits_b)
        print(f"\n  logits diff (cv2 vs F.i): max={d.max():.3e}  mean={d.mean():.3e}")
        # Which time-step's argmax differs?
        a = logits_a.argmax(axis=-1).squeeze()
        b = logits_b.argmax(axis=-1).squeeze()
        diff_positions = np.where(a != b)[0]
        print(f"  argmax differs at {len(diff_positions)} timesteps: {diff_positions.tolist()}")

    # ---- Weight drift check ----
    compare_weights(pth, Path(rec_dir))

    # ---- Direct logits comparison: Paddle model vs PyTorch model on the
    # *exact same* preprocessed tensor.  If logits differ, the divergence is
    # on the model side (weights or architecture).
    print(f"\n=== Paddle model vs PyTorch model on identical preprocessed tensor ===")
    try:
        import paddle
        paddle_layer = paddle.jit.load(str(Path(rec_dir) / "inference"))
        paddle_layer.eval()
        with paddle.no_grad():
            pd_input = paddle.to_tensor(t_paddle[None, ...])  # (1,C,H,W)
            pd_out = paddle_layer(pd_input)
        # Could be a tuple; take the first output
        if isinstance(pd_out, (tuple, list)):
            pd_out = pd_out[0]
        pd_logits = pd_out.numpy()
        # Paddle may return (B, T, V) or (B, V, T). Normalise to (B, T, V).
        if pd_logits.ndim == 3 and pd_logits.shape[1] != logits_a.shape[1]:
            if pd_logits.shape[2] == logits_a.shape[1]:
                pd_logits = pd_logits.transpose(0, 2, 1)

        print(f"  paddle logits shape: {pd_logits.shape}   pytorch shape: {logits_a.shape}")
        if pd_logits.shape == logits_a.shape:
            d = np.abs(pd_logits.astype(np.float64) - logits_a.astype(np.float64))
            print(f"  |paddle - pytorch| logits   max={d.max():.4e}  mean={d.mean():.4e}")
            a = pd_logits.argmax(axis=-1).squeeze()
            b = logits_a.argmax(axis=-1).squeeze()
            mismatch = np.where(a != b)[0]
            print(f"  argmax mismatches at {len(mismatch)} timesteps: {mismatch.tolist()}")
            if len(mismatch) > 0:
                # For the first few mismatches, show the competing chars + logits gap
                chars = rec.postprocess_op.character
                print(f"  Per-mismatch detail (first 10):")
                for t in mismatch[:10]:
                    pa, pb = int(a[t]), int(b[t])
                    pad_ch = chars[pa] if 0 <= pa < len(chars) else "?"
                    pt_ch = chars[pb] if 0 <= pb < len(chars) else "?"
                    pad_logit = float(pd_logits[0, t, pa])
                    pt_logit = float(logits_a[0, t, pa])
                    # Top-2 on each side
                    top_pd = pd_logits[0, t].argsort()[-3:][::-1]
                    top_pt = logits_a[0, t].argsort()[-3:][::-1]
                    print(f"    t={t:3d}  pd_pick={pa}({pad_ch!r})  pt_pick={pb}({pt_ch!r})")
                    print(f"             pd top3: {[(int(i), chars[int(i)], float(pd_logits[0,t,int(i)])) for i in top_pd]}")
                    print(f"             pt top3: {[(int(i), chars[int(i)], float(logits_a[0,t,int(i)])) for i in top_pt]}")
    except Exception as exc:
        print(f"  paddle forward failed: {exc}")


if __name__ == "__main__":
    main()
