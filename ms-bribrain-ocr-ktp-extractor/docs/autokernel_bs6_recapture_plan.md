# Plan: Recapture AutoKernel CUDA Graphs at batch_size = 6

## Goal

Match Hybrid's text parity (96.4% / 100% vs Paddle) on the AutoKernel backend
without losing AutoKernel's latency advantage. Currently AutoKernel is stuck at
`rec_batch_size=1` because its CUDA graphs were captured assuming a fixed
batch dimension of 1; running at `batch_size=6` breaks graph replay.

Target after this change:
- AutoKernel text parity ≈ Hybrid text parity (≥96% vs Paddle)
- AutoKernel latency: same or better than current (batched GPU utilization)

## Why batch matters for text accuracy

Paddle's `ToBatch` pads every crop in a batch to the batch's max width with
zeros. CNN boundary effects near the zero-padding edge shift borderline
argmaxes. Hybrid at `bs=1` sees each crop padded to its own width only — a
different padding context than Paddle, which causes ~1-3 text diffs per image
on low-confidence crops (e.g. `KAwiN` vs `KAwIN`, `wNi` vs `WNiI`). Matching
Paddle's `bs=6` closed the gap for Hybrid (30/30 on good_data_2).

## Current implementation — where the batch is baked in

`autokernel/verify.py:1707-1829` — `_replace_cuda_graph_strategy`:

- **Graph cache key**: `graph_cache: Dict[int, ...]` — keyed **only** by
  `bucket_w` (line 1728). Batch dimension is implicit in whatever first
  call's `template` tensor had (typically `B=1`).
- **Width buckets**: `GRAPH_WIDTH_BUCKETS = [320, 480, 640, 800, 960, 1280]`
  (line 1705). Six graphs total at `B=1`.
- **Padding**: `_pad_to_width` pads only dim 3 (W). Never touches dim 0 (B).
- **Capture**: `_capture_for_width(bucket_w, template)` at line 1750:
  `static_in = _pad_to_width(template, bucket_w).clone()` — `static_in.shape`
  is `(template.B, C, H, bucket_w)` which is `(1, 3, 48, W)` in practice.
- **Replay**: `static_in.copy_(padded)` at line 1793 requires the incoming
  tensor to have the **same batch size** as the captured graph. `B=6` input
  into a `B=1` static buffer either truncates silently (wrong) or errors.

## Known gotcha (from memory)

`torch.compile` + CUDA graph interaction: when `torch.compile` wraps the
capture path, its internal allocator creates stale input copies that the
graph references. `static_in.copy_(padded)` updates the outer buffer but
graph replays from the stale compiled copy → subsequent crops at the same
bucket return the first crop's output. HybridOCRBackend already sets
`context._suppress_compile = True` (`ocr_backends.py:706`). AutoKernel does
not. Any batch-aware rework must keep `torch.compile` suppressed when
graph_capture is active, or explicitly use
`torch.compiler.cudagraph_mark_step_begin()`.

## Design options

| Option | Cache keyed by | End-of-run crop handling | Graphs captured | GPU mem cost |
|---|---|---|---|---|
| **A. (B, W) pairs** | actual `(B, bucket_w)` | Capture a new graph on first encounter | up to 6×6 = 36 | High (36 × per-graph ~30-100MB) |
| **B. Fixed B=6, pad trailing** | `bucket_w` only | Pad with dummy crops to always feed `B=6` | 6 (one per width) | Same as today |
| **C. Batch buckets** | `(bucket_b, bucket_w)` with `bucket_b ∈ {1, 6}` | Snap actual batch to nearest bucket, pad | 2×6 = 12 | Moderate (12 × ~30-100MB) |

### Recommendation: **Option C (batch buckets: 1 and 6)**

Rationale:
- **Option A**: worst GPU memory footprint; production images often have
  crop-counts that aren't multiples of 6 → we'd cache many rare batch
  sizes for no benefit
- **Option B**: cleanest cache, but dummy-padding wastes ~15% compute on
  the trailing sub-batch for typical KTP images (~30 crops → 5 full + 1
  partial). Also complicates correctness — must mask dummy outputs
- **Option C**: bucket ∈ {1, 6} covers the common case. `B=6` batch for
  full batches (accuracy + throughput). `B=1` for the trailing remainder.
  This matches what Paddle does: the trailing partial batch is processed
  with whatever batch size remains, not padded to a fixed size

## Implementation steps

Work in `autokernel/verify.py`, `_replace_cuda_graph_strategy` method.

### Step 1 — Add batch bucketing alongside width bucketing

Add a class-level constant next to `GRAPH_WIDTH_BUCKETS`:
```python
GRAPH_BATCH_BUCKETS: List[int] = [1, 6]
```

Extend `_snap_to_bucket` to return `(bucket_b, bucket_w)`:
```python
def _snap_batch_to_bucket(b: int) -> int:
    for bb in sorted(self.GRAPH_BATCH_BUCKETS):
        if bb >= b:
            return bb
    return self.GRAPH_BATCH_BUCKETS[-1]
```

### Step 2 — Change cache key

Line 1728: `graph_cache: Dict[Tuple[int, int], Tuple[Any, Any, Any]] = {}`.

### Step 3 — Update `_capture_for_width` → `_capture_for_shape`

Accept both `bucket_b` and `bucket_w`. Expand or tile the template tensor's
batch dim to `bucket_b` before cloning into `static_in`:
```python
def _capture_for_shape(bucket_b, bucket_w, template):
    padded = _pad_to_width(template, bucket_w)
    if padded.shape[0] < bucket_b:
        # Repeat the first crop to fill the dummy batch slots
        padded = padded[:1].expand(bucket_b, -1, -1, -1).contiguous()
    elif padded.shape[0] > bucket_b:
        padded = padded[:bucket_b].contiguous()
    static_in = padded.clone()
    # ... rest identical to current _capture_for_width
```

### Step 4 — Update dispatch in `forward_with_multi_width_cuda_graph`

Lines 1782-1802:
```python
if isinstance(runtime_input, torch.Tensor) and runtime_input.ndim == 4:
    actual_b, _, _, actual_w = runtime_input.shape
    bucket_w = _snap_to_bucket(actual_w)
    bucket_b = _snap_batch_to_bucket(actual_b)
    padded = _pad_to_width(runtime_input, bucket_w)

    # Pad batch if under-sized (only for trailing batches with actual_b<bucket_b)
    if padded.shape[0] < bucket_b:
        padding = torch.zeros(
            bucket_b - padded.shape[0], *padded.shape[1:],
            dtype=padded.dtype, device=padded.device,
        )
        padded_full = torch.cat([padded, padding], dim=0)
    else:
        padded_full = padded

    key = (bucket_b, bucket_w)
    if key not in graph_cache:
        graph_cache[key] = _capture_for_shape(bucket_b, bucket_w, padded_full)
    graph, static_in, static_out = graph_cache[key]
    # ... copy_, replay ...

    # Slice back to actual_b so caller sees the right number of outputs
    if isinstance(static_out, torch.Tensor):
        result = static_out[:actual_b]
    else:
        result = tuple(t[:actual_b] for t in static_out)
    return _clone_graph_value(result) if output_mode != "static" else result
```

### Step 5 — Re-verify and re-record optimized kernels

Run `autokernel/verify.py` end-to-end to re-record `verification_result.json`.
The multi-bucket cache must be warmed for both `B=1` and `B=6` per width.
Expect first-run compile time to roughly 2× (12 graphs instead of 6) — ~30s.

### Step 6 — Update `AutoKernelPPOCRv5Backend` default `rec_batch_size`

Once graphs support `B=6`, mirror the Hybrid fix: in
`src/services/ocr_backends.py:1058-1068` (the `if backend_mode == "hybrid"`
block), apply the same `rec_batch_size = max(rec_batch_size, 6)` override to
the `"autokernel"` branch as well.

### Step 7 — Remove the `graph_capture` skip in Hybrid

`src/services/ocr_backends.py:806-815` currently force-excludes
`graph_capture` from Hybrid's kernel stack because the old single-batch
graphs were incompatible with Hybrid's batched eager mode. With multi-batch
graph capture landed, Hybrid can re-enable graph_capture to claim the same
latency win AutoKernel has. This is a follow-on; land it after Step 6 is
green.

## Testing plan

After each step, run:

```bash
OCR_AUTOKERNEL_ENABLED=1 .venv/bin/python -u scripts/compare_ocr_lines.py \
    --mode shared --only paddle,autokernel --output /tmp/ak_bs6_<step>.txt
```

Acceptance gates:

1. **Functional**: AutoKernel returns 28/28 + 30/30 line counts
   (no graph replay errors, no zero-output regressions).
2. **Accuracy**: AutoKernel matches Paddle on ≥ 96% / ≥ 96% (i.e. parity
   with Hybrid after its batched fix).
3. **Latency**: Steady-state predict() on good_data.png < 500 ms
   after warmup. Compare against current baseline
   (~1.2 s first image, <200 ms steady).
4. **Graph correctness sweep**: run the full `autokernel/verify.py` logits
   check (`verify_autokernel_ppocrv5_logits.py`) — argmax mismatch rate
   across synthetic + real crops must stay at the current baseline
   (see `memory/project_autokernel_graph_capture.md`: 28/28 fp32↔fp16
   match on KTP, 25/28 Paddle↔fp32).

End-to-end regression: run Hybrid at the same time to confirm Steps 6-7
didn't regress its parity (still 27/28 + 30/30 expected).

## Rollback

All changes are in `autokernel/verify.py` + two call sites in
`ocr_backends.py`. Revert: re-add the skip of `graph_capture` in
`HybridOCRBackend._load_kernel_replacements`, leave AutoKernel's
`rec_batch_size` at 1, and revert the graph_cache keying. No on-disk
artifacts change.

## Open questions (flag before implementation)

1. **Output slicing vs. static-mode caller**: current `output_mode: "static"`
   hands back the static tensor directly. Slicing it to `actual_b` breaks
   that contract. Either (a) force `output_mode="clone"` when batch padding
   is active, or (b) document that static mode returns the full
   `(bucket_b, ...)` tensor and require callers to slice.
2. **Capture warmup cost**: we currently warm `warmup_iters=3` before
   capture. For 12 buckets, total warmup = 36 forward passes. Acceptable
   (~5 s on H100); flag for other GPUs.
3. **GPU memory**: 12 static input buffers at `(6, 3, 48, 1280) fp16` =
   6·3·48·1280·2 = ~2.2 MB per input buffer. Output buffers similar.
   Graphs themselves carry captured kernels' workspace. Expect a total
   GPU memory bump of ~50-150 MB vs. current 6-graph state — budget-safe
   for current deployments but worth measuring after Step 5.
