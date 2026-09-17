"""Parity tests for the latency-win refactors.

Covers three changes that touch the OCR hot path and could silently regress
accuracy if the math drifts:

  1. _process_image_sync swapped PIL → cv2.imdecode.
  2. _stage_recognition_batch_cuda collapsed N per-crop H2D into one.
  3. rec post no longer casts the (B,T,V) logits to fp32 before argmax/max;
     it casts only the (B,T) prob output instead.

torch is an optional dep here (production container has it; host venv
typically does not) so torch-dependent tests skip cleanly via importorskip.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from PIL import Image

from src.services.ocr_backends import _resize_norm_rec_crop, _resize_rec_crop
from src.services.ocr_service import _process_image_sync


# ---------------------------------------------------------------------------
# Change 1: PIL → cv2 image decode
# ---------------------------------------------------------------------------


def _pil_encode(arr: np.ndarray, *, fmt: str, **kwargs) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def _pil_decode(image_bytes: bytes) -> np.ndarray:
    """The decode path we replaced — kept here as the parity reference."""
    return np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))


def test_process_image_sync_png_decode_exact_parity():
    """PNG is lossless; cv2 and PIL must produce byte-identical arrays."""
    rng = np.random.default_rng(seed=42)
    src = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
    png_bytes = _pil_encode(src, fmt="PNG")

    new_img, h, w = _process_image_sync(png_bytes)
    old_img = _pil_decode(png_bytes)

    assert new_img.shape == (64, 96, 3)
    assert new_img.dtype == np.uint8
    assert (h, w) == (64, 96)
    np.testing.assert_array_equal(new_img, old_img)


def test_process_image_sync_jpeg_decode_close_to_pil():
    """JPEG decode can differ by a few values between libjpeg variants. The
    OCR pipeline tolerates that; we only need the means and shapes to align
    so downstream preprocessing produces equivalent crops."""
    rng = np.random.default_rng(seed=7)
    src = rng.integers(0, 256, size=(128, 192, 3), dtype=np.uint8)
    jpeg_bytes = _pil_encode(src, fmt="JPEG", quality=95)

    new_img, _, _ = _process_image_sync(jpeg_bytes)
    old_img = _pil_decode(jpeg_bytes)

    assert new_img.shape == old_img.shape
    assert new_img.dtype == old_img.dtype
    diff = np.abs(new_img.astype(np.int16) - old_img.astype(np.int16))
    # Per-pixel: any single channel within 5 levels keeps downstream
    # normalize-then-resize stable. Mean across the frame should be <1.
    assert diff.max() <= 5, f"per-pixel max diff {diff.max()} exceeds 5"
    assert diff.mean() < 1.0, f"mean diff {diff.mean():.3f} too high"


def test_process_image_sync_rejects_invalid_bytes():
    """cv2.imdecode returns None on garbage; the helper must raise so the
    route handler returns a 400 instead of crashing on a None array."""
    with pytest.raises(ValueError, match="decode"):
        _process_image_sync(b"not an image")


# ---------------------------------------------------------------------------
# Change 2: per-crop H2D → single H2D for rec staging
# ---------------------------------------------------------------------------


def _old_stage_batch_cpu(
    torch,
    crops,
    *,
    batch_indices,
    resized_widths,
    target_w,
    img_c,
    img_h,
    rec_batch_size,
):
    """Reproduces the pre-refactor per-crop staging math on CPU.

    The original GPU branch cast uint8→fp32, mul/sub/div in fp32, then copied
    into a pre-zeroed workspace slice. CPU-side this is byte-identical because
    the integer→fp32 cast is exact for [0,255] and the affine has no
    fp16-specific quirks at fp32 precision.
    """
    batch = torch.zeros(
        (rec_batch_size, img_c, img_h, target_w),
        dtype=torch.float32,
    )
    for batch_pos, crop_index in enumerate(batch_indices):
        crop = crops[int(crop_index)]
        resized_w = resized_widths[int(crop_index)]
        resized = _resize_rec_crop(crop, resized_w=resized_w, img_h=img_h)
        t = torch.from_numpy(np.ascontiguousarray(resized))  # (H, W, C) uint8
        t = t.permute(2, 0, 1).to(dtype=torch.float32)
        t.mul_(1.0 / 255.0)
        t.sub_(0.5)
        t.div_(0.5)
        batch[batch_pos, :, :, :resized_w].copy_(t)
    return batch


def _new_stage_batch_cpu(
    torch,
    crops,
    *,
    batch_indices,
    resized_widths,
    target_w,
    img_c,
    img_h,
    rec_batch_size,
):
    """Reproduces the new single-H2D staging math on CPU."""
    host_buf = np.zeros(
        (rec_batch_size, img_c, img_h, target_w),
        dtype=np.float32,
    )
    for batch_pos, crop_index in enumerate(batch_indices):
        crop = crops[int(crop_index)]
        resized_w = resized_widths[int(crop_index)]
        host_buf[batch_pos] = _resize_norm_rec_crop(
            crop,
            img_c=img_c,
            img_h=img_h,
            target_w=target_w,
            resized_w=resized_w,
        )
    return torch.from_numpy(host_buf)


def _make_crop(rng, h, w):
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_stage_recognition_batch_single_h2d_matches_per_crop():
    """Single padded fp32 buffer + one transfer must be byte-identical to the
    old per-crop GPU normalize math (modulo dtype, which we hold at fp32)."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(seed=13)
    img_c, img_h = 3, 48
    target_w = 416
    rec_batch_size = 8

    # Three valid crops at varying widths; pad rows beyond fill must stay 0.
    crops = [
        _make_crop(rng, h=32, w=300),
        _make_crop(rng, h=40, w=180),
        _make_crop(rng, h=48, w=410),
    ]
    batch_indices = [0, 1, 2]
    resized_widths = {0: 320, 1: 200, 2: 410}

    old = _old_stage_batch_cpu(
        torch,
        crops,
        batch_indices=batch_indices,
        resized_widths=resized_widths,
        target_w=target_w,
        img_c=img_c,
        img_h=img_h,
        rec_batch_size=rec_batch_size,
    )
    new = _new_stage_batch_cpu(
        torch,
        crops,
        batch_indices=batch_indices,
        resized_widths=resized_widths,
        target_w=target_w,
        img_c=img_c,
        img_h=img_h,
        rec_batch_size=rec_batch_size,
    )

    assert old.shape == new.shape == (rec_batch_size, img_c, img_h, target_w)
    # ULP-level fp32 rounding differs between (x * (1/255)) — old GPU path —
    # and (x / 255.0) — new path, which matches PaddleOCR's CPU reference.
    # The gap is ~1e-7 absolute, far below fp16 model precision; we hold the
    # test to 1e-5 / 1e-4 so any future *math* change still fails loudly.
    torch.testing.assert_close(new, old, rtol=1e-4, atol=1e-5)

    # And pad regions are zero in both: rows beyond len(batch_indices) and
    # columns beyond resized_w within each active row.
    assert bool(torch.all(new[len(batch_indices):] == 0))
    for batch_pos, crop_index in enumerate(batch_indices):
        rw = resized_widths[crop_index]
        assert bool(torch.all(new[batch_pos, :, :, rw:] == 0)), (
            f"pad columns of crop {crop_index} not zero"
        )


def test_stage_recognition_batch_full_batch_no_padding_rows():
    """When batch_indices fills rec_batch_size completely, no pad rows; both
    paths must agree on the entire tensor."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(seed=99)
    img_c, img_h = 3, 48
    target_w = 320
    rec_batch_size = 4  # smaller so the test is cheap
    crops = [_make_crop(rng, h=48, w=300) for _ in range(rec_batch_size)]
    batch_indices = list(range(rec_batch_size))
    resized_widths = {i: 320 for i in range(rec_batch_size)}

    old = _old_stage_batch_cpu(
        torch,
        crops,
        batch_indices=batch_indices,
        resized_widths=resized_widths,
        target_w=target_w,
        img_c=img_c,
        img_h=img_h,
        rec_batch_size=rec_batch_size,
    )
    new = _new_stage_batch_cpu(
        torch,
        crops,
        batch_indices=batch_indices,
        resized_widths=resized_widths,
        target_w=target_w,
        img_c=img_c,
        img_h=img_h,
        rec_batch_size=rec_batch_size,
    )
    # See ULP-tolerance rationale in the test above.
    torch.testing.assert_close(new, old, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Change 3: drop .float() before argmax — verify fp16 argmax/max parity
# ---------------------------------------------------------------------------


def test_rec_argmax_fp16_matches_fp32_on_softmax_outputs():
    """For PP-OCRv5-shaped logits run through softmax, fp16 argmax must give
    identical indices to fp32 argmax. Two probabilities being within fp16
    quantization (~1e-3) implies both are near-1/V uniform, which never
    happens in well-trained CTC heads on real crops; we verify on synthetic
    softmax outputs that resemble production confidence."""
    torch = pytest.importorskip("torch")
    rng = torch.Generator().manual_seed(2024)
    B, T, V = 8, 64, 18385

    # Sharp logits: one big peak per (b, t), like real CTC predictions.
    logits = torch.randn(B, T, V, generator=rng) * 0.5
    peak_idx = torch.randint(0, V, (B, T), generator=rng)
    logits.scatter_(2, peak_idx.unsqueeze(-1), 20.0)
    probs_fp32 = torch.softmax(logits, dim=2)

    probs_fp16 = probs_fp32.to(dtype=torch.float16)
    prob_fp32, idx_fp32 = probs_fp32.max(dim=2)
    prob_fp16, idx_fp16 = probs_fp16.max(dim=2)

    # Argmax indices must be identical for sharp softmax peaks.
    assert torch.equal(idx_fp16, idx_fp32), (
        "fp16 argmax diverged from fp32 — peaks too soft for fp16 precision"
    )
    # Prob values agree up to fp16 representable precision (~1e-3 relative).
    torch.testing.assert_close(
        prob_fp16.float(), prob_fp32, rtol=1e-2, atol=1e-3
    )


def test_rec_argmax_fp16_matches_fp32_with_compressed_dynamic_range():
    """Same as above but with logits scaled so the max-min gap is small.
    Validates that even when the second-best probability is meaningfully
    close to the best, fp16 still tracks the winning class."""
    torch = pytest.importorskip("torch")
    rng = torch.Generator().manual_seed(7)
    B, T, V = 4, 16, 18385
    # Logits in a tight range, with one explicit peak of +3 — covers the
    # near-tie regime without crossing fp16 quantization.
    logits = torch.randn(B, T, V, generator=rng) * 0.1
    peak_idx = torch.randint(0, V, (B, T), generator=rng)
    logits.scatter_(2, peak_idx.unsqueeze(-1), 3.0)
    probs_fp32 = torch.softmax(logits, dim=2)
    probs_fp16 = probs_fp32.to(dtype=torch.float16)

    _, idx_fp32 = probs_fp32.max(dim=2)
    _, idx_fp16 = probs_fp16.max(dim=2)
    assert torch.equal(idx_fp16, idx_fp32)
