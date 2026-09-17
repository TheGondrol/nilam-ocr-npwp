# Latency Optimization Guide

This repository has evolved from a mostly framework-default OCR serving path into a set of GPU-first backends that reduce end-to-end latency by keeping more of the pipeline on the GPU, shrinking CPU<->GPU transfer volume, using mixed precision where it is numerically safe, and compiling stable hot paths with `torch.compile`.

The implementation lives mainly in [src/services/ocr_backends.py](../src/services/ocr_backends.py) and [src/services/ocr_service.py](../src/services/ocr_service.py). Supporting benchmark and validation scripts live in [scripts/](../scripts/).

## Goals

The latency work in this repo is built around a few practical goals:

- Move work off the CPU when the GPU can do it faster.
- Avoid bouncing large intermediate tensors between CPU and GPU.
- Batch recognition work to increase GPU utilization.
- Use `fp16` where it speeds up inference without breaking accuracy.
- Use `torch.compile` on the paths that are stable enough to benefit from it.
- Warm the runtime up at startup so real requests do not pay the cold-start cost.
- Keep correctness close to PaddleOCR behavior while still changing the runtime implementation.

## Main Optimization Themes

### 1. Replacing CPU work with GPU work

The repo does not only run the model forward pass on CUDA. It also moves parts of preprocessing and postprocessing to the GPU when that reduces total request time.

Examples:

- FullPyTorch detection uses a GPU-side preprocessing path in `FullPyTorchBackend._detect_boxes()`: image upload, normalization, and detector forward happen on CUDA before only the detector map is copied back for DB postprocess.
- FullPyTorch recognition uses `_stage_recognition_batch_cuda()` to resize, normalize, and stage recognition batches directly on the GPU.
- Hybrid recognition replaces PaddleOCR's default per-crop recognizer path with a batched GPU path in `HybridOCRBackend._fast_recognize()`.

The general pattern is: upload once, do more work while the tensor is already on device, and only bring back compact outputs.

### 2. Minimizing CPU<->GPU transfers

The largest avoidable overhead in OCR serving is often not only model compute. It is the amount of intermediate data crossing the PCIe boundary.

This repo reduces that in several places:

- Recognition no longer transfers full `B x T x vocab` logits to CPU just to decode CTC. Instead, it computes `argmax` and confidence on GPU first, then copies only compact indices/probabilities back for decode.
- Hybrid mode skips the unused recognition pass inside Paddle detection by building a detection-only path and running recognition separately.
- FullPyTorch detection keeps preprocess and forward on GPU and copies back only the probability maps required by DB postprocessing.
- The recognizer batch planner groups crops by width and caps padding expansion with `OCR_AUTOKERNEL_REC_BUCKET_MAX_WIDTH_RATIO`, which reduces both wasted compute and useless tensor movement.

In practice, the repo is trying to move "small answers" back to CPU rather than "large activations".

### 3. Using mixed precision carefully

The repo uses `fp16` aggressively where it helps, but not blindly.

Recognizer:

- The PyTorch recognizer adapters run in configurable dtype via `OCR_AUTOKERNEL_DTYPE`.
- On GPU, the common fast path is `float16`.
- On CPU, the code falls back to `float32`, because CPU `fp16` is typically slower and more fragile.

Detector:

- Full detector `fp16` is not safe for PP-OCRv5 server detection.
- `FullPyTorchBackend` therefore supports a mixed-precision detector where the backbone runs in `fp16` or `bf16`, but the neck and head stay in `fp32`.
- `_build_mixed_precision_detector_net()` exists specifically because the detector neck can emit activations above `fp16` range, which can produce `NaN` maps if the entire network is cast down.

This is an important design choice in the repo: use lower precision for speed, but only in the subgraphs that are numerically stable.

### 4. Compiling the stable hot paths

`torch.compile` is used in the FullPyTorch backend for the model paths that benefit from ahead-of-time graph capture and kernel fusion.

Key points from the current implementation:

- `FullPyTorchBackend._maybe_compile_models()` can compile the detector, recognizer, or both.
- The detector stem is rewritten in `_rewrite_det_stem_for_compile()` so Inductor can compile it more reliably.
- The recognizer runtime model can be compiled independently with `OCR_AUTOKERNEL_TORCH_COMPILE_REC=1`.
- `src/services/ocr_service.py` uses a dedicated single-thread OCR executor because compiled CUDA graph state is thread-local; serving compiled models from multiple worker threads can assert or replay incorrectly.
- Warmup intentionally exercises multiple detector and recognizer shapes in `_warmup_compiled_shapes()` so the first real request does not trigger expensive recompilation.

The repo treats `torch.compile` as an optimization for specific steady-state paths, not as a global switch that is always safe.

### 5. Warming and caching to remove cold-start pain

Several optimizations improve steady-state latency but add first-run cost. The repo addresses that explicitly.

- `warmup_ocr()` runs startup inference before traffic arrives.
- FullPyTorch warmup can use a real image via `OCR_AUTOKERNEL_WARMUP_IMAGE_PATH`.
- The deployment path in [deploy_fullpytorch.sh](../deploy_fullpytorch.sh) persists `TORCHINDUCTOR_CACHE_DIR`, which allows compiled artifacts to survive container restarts.
- The deployment script also enables `TORCHINDUCTOR_FX_GRAPH_CACHE=1` and `TORCHINDUCTOR_AUTOGRAD_CACHE=1`.

This matters because a fast steady-state path is not enough if every pod restart pays the full compile cost again.

### 6. Batching and width bucketing

Text recognition is naturally variable-shape. This repo improves utilization without padding every crop to the worst-case width.

The main mechanism is in `FullPyTorchBackend._recognition_batch_plan()`:

- Crops are sorted by width ratio.
- They are grouped into batches up to `OCR_AUTOKERNEL_REC_BATCH_SIZE`.
- Padding is bounded by `OCR_AUTOKERNEL_REC_BUCKET_MAX_WIDTH_RATIO`.
- Each batch is padded only to its local target width, not the global max width of the image.

Benefits:

- better GPU occupancy than per-crop inference
- less wasted padding than naive fixed-width batching
- less memory traffic than sending very over-padded tensors through the model

### 7. Overlapping staging and compute

FullPyTorch recognition uses a small pipeline to overlap batch staging with model execution:

- `_stage_recognition_batch_cuda()` prepares the next batch on a dedicated CUDA stream
- `_fast_recognize()` uses double buffering so the next batch can be staged while the current batch is being consumed

This is a classic latency optimization: keep the GPU busy and hide prep work behind compute where possible.

## Backend Strategy in This Repo

The repo now has several backend modes selected through `OCR_BACKEND`.

### `paddle`

Baseline PaddleOCR runtime. Useful for compatibility and as a reference, but it leaves more performance on the table.

### `hybrid`

Implemented by `HybridOCRBackend`.

Design:

- Paddle-based detection
- PyTorch recognizer loaded from converted PP-OCRv5 weights
- AutoKernel recognizer replacements when enabled
- GPU-side batched recognition path

Why it is faster than a default end-to-end Paddle path:

- avoids an unnecessary recognition pass inside the detector-side pipeline
- batches crops for recognition
- computes CTC argmax on GPU
- transfers only compact decode inputs back to CPU

Why it still exists:

- it keeps Paddle's detector behavior while accelerating the recognition side
- it is a good intermediate point when FullPyTorch detection parity or detector compile behavior still needs validation

### `fullpytorch`

Implemented by `FullPyTorchBackend`.

Design:

- converted PyTorch detector
- converted PyTorch recognizer
- GPU-side detector preprocess
- GPU-side recognizer preprocessing and argmax
- optional `torch.compile`
- mixed-precision detector and `fp16` recognizer

This is the most complete latency-oriented path in the repo because it gives full control over both stages of OCR and avoids framework handoff overhead between Paddle and PyTorch.

### `autokernel`

Implemented by `AutoKernelPPOCRv5Backend`.

Design:

- converted PP-OCRv5 detector and recognizer
- verified AutoKernel replacements loaded from workspace artifacts
- optional graph-capture and kernel-level optimization stack

This path is optimized around pre-verified kernel replacements rather than only general PyTorch runtime tuning.

## Concrete Improvements by Area

### Detection

Current latency-oriented changes include:

- keeping detector net on CUDA
- GPU-side normalize and tensor creation in FullPyTorch `_detect_boxes()`
- mixed precision for detector backbone only
- optional detector `torch.compile`
- detector warmup across multiple shapes to avoid first-request recompiles

Important constraint:

- detector DB postprocess still expects CPU/numpy-friendly inputs, so the final map is copied back before box extraction

### Recognition

Current latency-oriented changes include:

- crop sorting and local-width batching
- GPU-side batch staging
- `fp16` recognizer inference
- GPU-side `argmax` and confidence extraction
- compact D2H transfer for decode
- optional `torch.compile` on the recognizer runtime model
- AutoKernel replacement stack for supported backends

Recognition is where most of the repo's "remove CPU work and reduce transfer volume" improvements are concentrated.

### Service Runtime

Outside the model code, the service also contains latency-relevant decisions:

- OCR warmup at process startup
- dedicated OCR executor for compiled CUDA graph stability
- separation between CPU-bound image loading and model execution executors
- deployment scripts that configure persistent compile cache and CUDA MPS

## Runtime Knobs

The backend is controlled primarily through env vars assembled in `src/services/ocr_service.py::_build_backend_settings()`.

Most important variables:

- `OCR_BACKEND`: `auto`, `paddle`, `autokernel`, `hybrid`, `fullpytorch`
- `OCR_AUTOKERNEL_ENABLED`: enables the converted-weight infrastructure used by non-Paddle backends
- `OCR_AUTOKERNEL_DTYPE`: recognizer dtype, usually `float16`
- `OCR_AUTOKERNEL_DET_DTYPE`: detector dtype for FullPyTorch, often `float16` for mixed-precision backbone mode
- `OCR_AUTOKERNEL_REC_BATCH_SIZE`: maximum recognition batch size
- `OCR_AUTOKERNEL_REC_BUCKET_MAX_WIDTH_RATIO`: padding expansion cap per recognition batch
- `OCR_AUTOKERNEL_TORCH_COMPILE`: global compile toggle
- `OCR_AUTOKERNEL_TORCH_COMPILE_DET`: detector compile toggle
- `OCR_AUTOKERNEL_TORCH_COMPILE_REC`: recognizer compile toggle
- `OCR_AUTOKERNEL_TORCH_COMPILE_MODE`: compile mode such as `default` or `reduce-overhead`
- `OCR_AUTOKERNEL_TORCH_COMPILE_DYNAMIC`: whether to compile with dynamic shape support
- `OCR_AUTOKERNEL_WARMUP_IMAGE_PATH`: image used for realistic startup warmup
- `OCR_AUTOKERNEL_DET_LIMIT_SIDE_LEN`
- `OCR_AUTOKERNEL_DET_LIMIT_TYPE`

Common FullPyTorch setup from the repo's own scripts:

```bash
export OCR_BACKEND=fullpytorch
export OCR_AUTOKERNEL_ENABLED=1
export OCR_AUTOKERNEL_DTYPE=float16
export OCR_AUTOKERNEL_DET_DTYPE=float16
export OCR_AUTOKERNEL_TORCH_COMPILE_REC=1
export OCR_AUTOKERNEL_REC_BATCH_SIZE=8
export OCR_AUTOKERNEL_DET_LIMIT_SIDE_LEN=1280
export OCR_AUTOKERNEL_DET_LIMIT_TYPE=max
export OCR_AUTOKERNEL_WARMUP_IMAGE_PATH="${PWD}/tests/data/good_data_3.png"
```

See [scripts/run_server_fullpytorch.sh](../scripts/run_server_fullpytorch.sh) and [deploy_fullpytorch.sh](../deploy_fullpytorch.sh) for the full operational setup.

## Benchmarking and Validation

The repo already includes several scripts for measuring and validating the latency work.

### End-to-end backend comparison

[scripts/compare_ocr_lines.py](../scripts/compare_ocr_lines.py)

Use this to compare output quality and coarse end-to-end timing across backends:

```bash
uv run python scripts/compare_ocr_lines.py --only paddle,hybrid,fullpytorch
```

### FullPyTorch stage breakdown

[scripts/profile_fullpytorch_stages.py](../scripts/profile_fullpytorch_stages.py)

Breaks `predict()` into:

- detection
- crop extraction
- recognition preprocessing
- recognition forward
- recognition postprocess
- total

Useful for confirming whether a change reduced preprocessing cost, model cost, or transfer cost.

### Detector-only stage breakdown

[scripts/profile_fullpytorch_det_stages.py](../scripts/profile_fullpytorch_det_stages.py)

Breaks detection into:

- `pre`
- `h2d`
- `fwd`
- `d2h`
- `post`
- `total`

This is the most direct script for analyzing CPU/GPU transfer reductions on the detector path.

### Hybrid stage breakdown

[scripts/profile_hybrid_stages.py](../scripts/profile_hybrid_stages.py)

Useful for quantifying the recognition-side speedup when moving from Paddle's default recognition path to the custom GPU batched path.

### Detector precision / compile experiments

The repo also includes focused scripts such as:

- [scripts/profile_det_fp16_vs_fp32.py](../scripts/profile_det_fp16_vs_fp32.py)
- [scripts/debug_det_fp16.py](../scripts/debug_det_fp16.py)
- [scripts/debug_det_fp16_mixed.py](../scripts/debug_det_fp16_mixed.py)
- [scripts/debug_det_autocast_compare.py](../scripts/debug_det_autocast_compare.py)
- [scripts/_bench_det_compile.py](../scripts/_bench_det_compile.py)

These are useful when changing detector precision or compile settings, because detector speed gains can easily regress correctness if precision is pushed too far.

## Known Tradeoffs and Constraints

The current design is intentionally conservative in a few places.

### Full detector `fp16` is not always safe

The PP-OCRv5 server detector can overflow in the neck/head when fully cast to `fp16`. That is why the repo keeps neck and head in `fp32` for the mixed detector path.

### `torch.compile` is beneficial but shape-sensitive

Compile works best when:

- shapes are warmed up in advance
- the hot path is stable
- runtime calls happen on the same execution thread

That is why the service pins OCR model calls to a dedicated single-thread executor.

### Some optimizations help one backend and hurt another

Examples already documented in code:

- Hybrid suppresses `torch.compile` in its AutoKernel recognizer path because variable batched eager execution is preferred there.
- AutoKernel graph-capture behavior has tighter shape and batch assumptions than the more flexible FullPyTorch path.

### Correctness still matters more than isolated kernel speedups

Several scripts and comments in the repo show that some apparently faster settings can break parity:

- detector all-`fp16`
- width/padding behavior that diverges from Paddle batching
- compile behavior on dynamic detector modules

The repo therefore validates with output comparison and stage profilers, not just raw throughput.

## Recommended Reading Order in the Code

If you want to understand the latency work from source, read in this order:

1. [src/services/ocr_service.py](../src/services/ocr_service.py)
2. [src/services/ocr_backends.py](../src/services/ocr_backends.py)
3. [scripts/profile_fullpytorch_stages.py](../scripts/profile_fullpytorch_stages.py)
4. [scripts/profile_fullpytorch_det_stages.py](../scripts/profile_fullpytorch_det_stages.py)
5. [scripts/profile_hybrid_stages.py](../scripts/profile_hybrid_stages.py)
6. [deploy_fullpytorch.sh](../deploy_fullpytorch.sh)
7. [docs/autokernel_bs6_recapture_plan.md](./autokernel_bs6_recapture_plan.md)

## Summary

The latency improvement strategy in this repo is not a single optimization. It is a stack:

- keep more preprocessing and postprocessing on the GPU
- move less data back to CPU
- batch recognition intelligently
- use `fp16` where it is safe
- keep numerically fragile subgraphs in `fp32`
- compile stable hot paths
- warm and cache aggressively
- isolate backend-specific tradeoffs instead of forcing one runtime strategy everywhere

That combination is what turns the OCR service from "GPU inference only" into a genuinely GPU-first serving pipeline.
