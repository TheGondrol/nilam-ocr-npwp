# Converting PaddleOCR `.pdiparams` → PyTorch `.pth`

This document explains how this repo turns Paddle PP-OCRv5 server checkpoints
(`inference.pdiparams` + `inference.json`) into PyTorch `state_dict` files
(`server_det.pth`, `server_rec.pth`) that are loaded by the PyTorch backends
(autokernel / hybrid / fullpytorch).

The pipeline lives in [`src/services/ppocrv5_conversion.py`](../src/services/ppocrv5_conversion.py)
and is exercised by the two thin runner scripts in [`scripts/`](../scripts/).
There is **one** conversion path — the PIR direct converter
(`_convert_pir_to_pytorch`). Older fallbacks (legacy static-graph extraction,
the `PPOCRv5{Det,Rec}Converter` classes) have been removed; correctness vs.
the source Paddle weights is verified by `scripts/verify_conversion.py`.

---

## 1. Quick runbook

The det and rec stock PP-OCRv5 server `pdiparams` files are expected at:

```
src/models/server_models/ppocrv5_server_det_source/inference.pdiparams   (84M)
src/models/server_models/ppocrv5_server_rec_source/inference.pdiparams   (81M)
```

**They are gitignored (`*.pdiparams` in `.gitignore`)** — only the metadata
(`inference.json`, `inference.yml`, `config.json`) is committed. On a fresh
checkout you must download them yourself; see §2 for how.

Once both `inference.pdiparams` files are in place, re-convert them to
PyTorch `.pth`:

```bash
# IMPORTANT: use the project venv (system python's libstdc++ is too old to
# import paddle on this host).
.venv/bin/python scripts/reconvert_det.py
.venv/bin/python scripts/reconvert_rec.py
```

Outputs:

```
autokernel/workspace/ppocrv5/server_det.pth   (~101M, 642 tensors)
autokernel/workspace/ppocrv5/server_rec.pth   (~105M, 547 tensors)
```

These are raw PyTorch state dicts; downstream code loads them with
`torch.load(...)` + `net.load_state_dict(...)` (no extra wrapping).

To verify a converted file numerically matches the source Paddle model on a
fixed input:

```bash
.venv/bin/python scripts/verify_conversion.py
```

A healthy result is max-abs diff ≲ 1e-4 on both detector (probability map)
and recognizer (CTC logits). On the stock checkpoints we see:

```
=== Detector ===
  paddle.shape=(1, 1, 320, 320)  torch.shape=(1, 1, 320, 320)
  max-abs diff = 4.66e-08         # pure FP32 noise
=== Recognizer ===
  paddle.shape=(1, 40, 18385)  torch.shape=(1, 40, 18385)
  max-abs diff = 6.01e-05         # FP32 noise accumulated over 40 timesteps
```

If `import paddle` fails with `GLIBCXX_3.4.30' not found`, you are using the
system `python` instead of the project venv — run `.venv/bin/python ...`.

---

## 2. Obtaining the stock PP-OCRv5 server checkpoints

PaddleOCR / PaddleX publish the official PP-OCRv5 server checkpoints to
several mirrors. Pick whichever you can reach from this host.

### Option A — Let PaddleX download into its cache (preferred when paddlex is installed)

Any first-time call to a `PaddleOCR` or `paddlex` pipeline that names
`PP-OCRv5_server_det` / `PP-OCRv5_server_rec` will auto-download the model
into `~/.paddlex/official_models/<model_name>/`. The smallest one-liner that
triggers downloads for both:

```bash
.venv/bin/python -c "
from paddlex.inference.utils.official_models import official_models
for m in ('PP-OCRv5_server_det', 'PP-OCRv5_server_rec'):
    print(m, '->', official_models[m])
"
```

PaddleX's hoster ladder is `huggingface → aistudio → modelscope → bos`
(see `paddlex/inference/utils/official_models.py:_ModelManager`); the first
reachable mirror wins. To force a specific mirror, set
`PADDLE_PDX_MODEL_SOURCE=bos|huggingface|modelscope|aistudio` before running.

After download the cache layout is:

```
~/.paddlex/official_models/PP-OCRv5_server_det/
  ├── inference.pdiparams        ← the file we convert
  ├── inference.json             ← PIR program graph (required alongside .pdiparams)
  ├── inference.yml              ← preprocessing/postprocessing config
  ├── inference.onnx             ← unused by us
  ├── config.json
  └── README.md
~/.paddlex/official_models/PP-OCRv5_server_rec/    (same layout)
```

You can then either point the converter at the cache directly or copy the
two files into the repo's `src/models/.../` source dirs:

```bash
SRC_DET=~/.paddlex/official_models/PP-OCRv5_server_det
SRC_REC=~/.paddlex/official_models/PP-OCRv5_server_rec
DST_DET=src/models/server_models/ppocrv5_server_det_source
DST_REC=src/models/server_models/ppocrv5_server_rec_source

cp "$SRC_DET"/{inference.pdiparams,inference.json} "$DST_DET"/
cp "$SRC_REC"/{inference.pdiparams,inference.json} "$DST_REC"/
```

(`inference.yml` and `config.json` are already committed; don't overwrite
them unless PaddleOCR has bumped the model.)

### Option B — Direct BOS download (no paddlex needed)

The Paddle Object Storage URLs are stable and don't require any auth or
extra Python deps:

```bash
BOS=https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0
mkdir -p /tmp/ppocrv5 && cd /tmp/ppocrv5

curl -fLO  "$BOS/PP-OCRv5_server_det_infer.tar"
curl -fLO  "$BOS/PP-OCRv5_server_rec_infer.tar"

tar -xf PP-OCRv5_server_det_infer.tar   # → PP-OCRv5_server_det/{inference.pdiparams, inference.json, ...}
tar -xf PP-OCRv5_server_rec_infer.tar   # → PP-OCRv5_server_rec/{inference.pdiparams, inference.json, ...}
```

Then copy the two files of interest into the repo source dirs as in
Option A.

### Option C — Hugging Face / ModelScope mirror

If BOS is blocked but you have access to Hugging Face:

```bash
# repo IDs: PaddlePaddle/PP-OCRv5_server_det and PaddlePaddle/PP-OCRv5_server_rec
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('PaddlePaddle/PP-OCRv5_server_det',
                  local_dir='src/models/server_models/ppocrv5_server_det_source',
                  allow_patterns=['inference.pdiparams', 'inference.json'])
snapshot_download('PaddlePaddle/PP-OCRv5_server_rec',
                  local_dir='src/models/server_models/ppocrv5_server_rec_source',
                  allow_patterns=['inference.pdiparams', 'inference.json'])
"
```

ModelScope (`modelscope.snapshot_download`) and AIStudio
(`paddlex.utils.download.aistudio_download`) work the same way — the repo IDs
match the names used by `_HuggingFaceModelHoster` /
`_ModelScopeModelHoster` / `_AIStudioModelHoster` in
`paddlex/inference/utils/official_models.py`.

### Sanity check after downloading

```bash
ls -lh src/models/server_models/ppocrv5_server_det_source/inference.pdiparams \
       src/models/server_models/ppocrv5_server_rec_source/inference.pdiparams
# Expect ~84M for det, ~81M for rec.
```

Both `inference.pdiparams` AND `inference.json` must exist side by side —
the `.json` is the PIR program graph, and the converter requires it.

### Verifying you got the right model version

`scripts/verify_conversion.py` is the ultimate check: convert and verify on
the same machine. If the max-abs diff is in the FP32-noise band
(< 1e-3 for det, < 1e-3 for rec), the source `.pdiparams` is the canonical
PP-OCRv5 server checkpoint that the architecture YAML was written for.

If you see a much larger diff, you probably grabbed a different variant
(e.g. mobile rather than server, or a fine-tuned checkpoint). Re-download
from a different mirror to confirm.

---

## 3. Supported checkpoint format

Only **Paddle PIR inference models** are supported (`.pdiparams` +
`.json`). The stock PP-OCRv5 server checkpoints in this repo are PIR; if you
ever need to convert a `.pdparams` state-dict or a legacy static-graph
inference model, you'll need to first re-export it as PIR (or extend the
converter).

---

## 4. Architecture comes from YAML, not flags

The converter does **not** auto-detect the model architecture. It reads it
from hard-coded YAML configs inside `PaddleOCR2Pytorch/`:

| Component | YAML | Built model |
|---|---|---|
| `det` | `PaddleOCR2Pytorch/configs/det/PP-OCRv5/PP-OCRv5_server_det.yml` | DB head, backbone `PPHGNetV2_B4` (det=True), neck `LKPAN`, head `PFHeadLocal` (large) |
| `rec` | `PaddleOCR2Pytorch/configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml` | `SVTR_HGNet`, backbone `PPHGNetV2_B4` (text_rec=True), `MultiHead` = CTCHead(SVTR neck) + NRTRHead |

Helpers:

- `server_det_architecture(ppocr_root)` → returns the `Architecture` block.
- `server_rec_architecture(ppocr_root)` → same, plus injects `out_channels_list`
  for the multi-head decoders, computed from the character dictionary
  (`pytorchocr/utils/dict/ppocrv5_dict.txt`, +1 for the blank token, +2/+3 for
  SAR/NRTR special tokens, plus 1 for the space char if `use_space_char: true`).

If you want to convert a *different* PP-OCRv5 variant (e.g. mobile), swap the
YAML constants at the top of `ppocrv5_conversion.py` (`DET_YAML_RELATIVE`,
`REC_YAML_RELATIVE`) or add new wrapper functions that point at
`PP-OCRv5_mobile_*.yml`.

---

## 5. The PIR direct converter (`_convert_pir_to_pytorch`)

This is the only conversion path. It:

1. Builds the empty PyTorch model from the YAML architecture (`BaseOCRV20`).
2. Walks `model.net.named_modules()` and groups every Conv2d / ConvTranspose2d
   / BN / Linear / LayerNorm by its **canonical Paddle op type**.
3. Loads the Paddle PIR model with `paddle.jit.load`, parses each parameter
   name with a regex `^(.+?)_(\d+)\.(.+?)_\d+$` — e.g.
   `conv2d_17.w_0_0` → op_type `conv2d`, op_index `17`, suffix `w_0`.
4. Buckets both Paddle ops and PyTorch modules by `(op_type, weight_shape)`,
   then zips bucket-by-bucket.
5. Maps Paddle suffixes to PyTorch param names and writes the new state dict.

### Suffix → param-name map

| Paddle op | `w_0` | `b_0` | `w_1` | `w_2` | Notes |
|---|---|---|---|---|---|
| `conv2d`, `conv2d_transpose` | `weight` | `bias` | — | — | |
| `batch_norm`, `batch_norm2d` | `weight` (γ) | `bias` (β) | `running_mean` | `running_var` | Paddle’s `_mean` / `_variance` are mapped here. |
| `linear` | `weight` (transposed) | `bias` | — | — | Paddle stores `[in, out]`; we transpose to PyTorch’s `[out, in]`. |
| `layer_norm` | `weight` | `bias` | — | — | |

### Why a shape bucket and not positional zip

`named_modules()` yields PyTorch modules in **construction order** (grouped by
submodule), but Paddle PIR emits ops in **forward-graph order**. For a model
like LKPAN (which interleaves convs across pyramid branches), a naive
positional zip mis-pairs ~34% of params and silently leaves them random-init.

Bucketing by `(op_type, shape)` recovers the correct correspondence inside
each bucket, because within a single shape class the construction order and
forward order do agree. `_PIR_TYPE_EQUIV = {"batch_norm": "batch_norm2d"}`
folds Paddle's generic BN type into our 2D BN bucket.

### Training-only modules are skipped

PIR export prunes branches that aren't on the inference path. We must
mirror that on the PyTorch side or shape-bucket positions shift. The
patterns excluded (in `_PIR_INFERENCE_DEAD_NAME_PATTERNS`):

- `*.thresh.*` — DB detector’s threshold branch (only used during training).
- `backbone.last_conv` — PPHGNetV2 classification tail (inert when `det=True`
  or `text_rec=True`).

These names are skipped when collecting PyTorch modules, then their full
names are logged at INFO as `... training-only PT modules skipped`.

### Diagnostic output

After conversion the helper logs:

```
PIR conversion matched N parameters; M training-only PT modules skipped;
P Paddle ops unmatched; Q PT modules unmatched
```

You want **`P == 0` and `Q == 0`**. Anything non-zero means buckets didn't
align and some weights were left random-init — investigate before shipping
the resulting `.pth`. Then re-run `scripts/verify_conversion.py` to confirm
the numerical match against Paddle.

---

## 6. Output format

The PIR converter saves the model with:

```python
torch.save(self.net.state_dict(), weights_path,
           _use_new_zipfile_serialization=False)
```

(via `BaseOCRV20.save_pytorch_weights`). The file is a flat
`OrderedDict[str, torch.Tensor]` keyed by the PyTorch model's submodule
paths:

- Detector keys begin with `backbone.`, `neck.`, `head.`.
- Recognizer keys begin with `backbone.`, `head.ctc_head.`,
  `head.gtc_head.`, `head.before_gtc.`, `head.ctc_encoder.`.

Loading is a one-liner — see `autokernel/models/ppocrv5_server.py`:

```python
net.load_state_dict(torch.load(weights_path, map_location="cpu"))
```

(Use `strict=False` if you've turned off optional sub-heads.)

---

## 7. Where conversion is wired up at deploy time

`convert_ppocrv5_server_weights` and `ensure_ppocrv5_server_weights` (bottom
of `ppocrv5_conversion.py`) are the entry points used by the deploy /
bootstrap code paths. They take a `PPOCRv5ServerConversionConfig` containing:

- `ppocr_root` — path to `PaddleOCR2Pytorch/`
- `det_source_path` / `rec_source_path` — `.pdiparams` (or directory)
- `det_output_path` / `rec_output_path` — `.pth` destinations
- `components` — subset of `("det", "rec")`
- `force` — re-run even if the output already exists

`ensure_*` skips components whose output already exists (unless `force=True`).
`convert_*` always runs the requested components.

Source paths are typically derived from `config.yaml`'s
`paddleocr.text_detection.model_dir` /
`paddleocr.text_recognition.model_dir` via
`infer_paddle_source_from_model_dir`, which appends `inference.pdiparams` to a
directory.

The standalone `scripts/reconvert_{det,rec}.py` runners hard-code the stock
paths and call `_convert_{detector,recognizer}` directly — handy for forcing
a re-conversion after touching the matcher logic.

---

## 8. Verifying correctness

`scripts/verify_conversion.py` is the canonical correctness check:

1. Loads the source Paddle PIR model with `paddle.jit.load`.
2. Loads the converted `.pth` into a freshly built PyTorch model.
3. Runs the same fixed `np.random.default_rng(...)` input through both.
4. Selects the inference-relevant output key (`maps` for det,
   `ctc` for rec — the auxiliary `cbn_maps` / `nrtr` outputs are not used
   by the deployed pipeline) and reports max-abs / mean-abs diff.

Treat anything beyond ~1e-3 max-abs diff as a bug in the matcher — typical
healthy values are `< 1e-4` (CTC logit space) and `< 1e-7` (sigmoid det
output).

If you change the matcher, always re-run this script before committing the
new `.pth` files.

---

## 9. Common gotchas

- **`GLIBCXX_3.4.30' not found` when importing paddle** — use
  `.venv/bin/python`, not the system `python`. The project virtualenv ships a
  newer libstdc++ that paddle needs.
- **Unmatched Paddle ops in the log** — usually means a new submodule was
  added to the YAML architecture but its op type isn't in `_PIR_PARAM_MAP` /
  `type_map`, or a training-only branch isn't being skipped. Inspect the
  warning, then either add the op type or add a name pattern to
  `_PIR_INFERENCE_DEAD_NAME_PATTERNS`.
- **Recognizer character count** — `out_channels_list` is computed from the
  dict file. If you swap the dict (e.g. monolingual → multilingual) you must
  re-run conversion; the rec head's final FC dims depend on it.
- **Linear weight shape** — Paddle `linear` weights are transposed in the PIR
  path (`[in, out]` → `[out, in]`). If you add a custom op that internally
  calls `paddle.matmul` with stored weights, you may need to extend the
  suffix map.
- **Det dict has two heads** — `model.net(x)` returns
  `{"maps": ..., "cbn_maps": ...}`. Only `maps` is the inference output;
  `cbn_maps` is a training-time auxiliary that ends up at zeros after
  conversion (its source op is pruned from the PIR export). The verifier
  picks `maps` explicitly.

---

## 10. File map

| Path | Role |
|---|---|
| `src/services/ppocrv5_conversion.py` | Single-source-of-truth conversion logic (PIR matcher + entry points). |
| `scripts/reconvert_det.py` | One-shot runner for the stock det checkpoint. |
| `scripts/reconvert_rec.py` | One-shot runner for the stock rec checkpoint. |
| `scripts/verify_conversion.py` | Numerical correctness check vs. source Paddle model. |
| `scripts/prepare_ppocrv5_server_weights.py` | Deploy-time CLI wrapper around `convert_ppocrv5_server_weights`. |
| `PaddleOCR2Pytorch/configs/det/PP-OCRv5/PP-OCRv5_server_det.yml` | Det architecture spec. |
| `PaddleOCR2Pytorch/configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml` | Rec architecture spec. |
| `PaddleOCR2Pytorch/pytorchocr/base_ocr_v20.py` | `BaseOCRV20` — builds the empty PyTorch model and saves the state dict. |
| `src/models/server_models/ppocrv5_server_det_source/` | Stock det PIR checkpoint. |
| `src/models/server_models/ppocrv5_server_rec_source/` | Stock rec PIR checkpoint. |
| `autokernel/workspace/ppocrv5/server_{det,rec}.pth` | Conversion outputs consumed by the runtime. |
