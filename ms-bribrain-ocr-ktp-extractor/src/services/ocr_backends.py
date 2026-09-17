"""OCR backend implementations used by the OCR service."""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol

import numpy as np

from src.services.ppocrv5_conversion import (
    PPOCRv5ServerConversionConfig,
    ensure_ppocrv5_server_weights,
)

logger = logging.getLogger(__name__)


# Recognizer width buckets. Crop target_w is snapped to the smallest bucket
# that fits; crops wider than the top bucket get truncated, not split. Closed
# enumeration of post-snap shapes is what makes torch.compile recompiles zero
# at steady state — each bucket compiles once during warmup.
REC_W_BUCKETS: tuple[int, ...] = (320, 416, 512, 640, 800, 1024, 1280)


def _snap_rec_width_to_bucket(desired_w: int) -> int:
    for bucket in REC_W_BUCKETS:
        if desired_w <= bucket:
            return bucket
    return REC_W_BUCKETS[-1]


# Per-batch bucketing summary log (off by default). Enable with
# OCR_AUTOKERNEL_BUCKET_DEBUG_LOG=1 to evaluate whether REC_W_BUCKETS is too
# coarse for prod traffic — log line includes batch fill_pct percentiles and
# truncation count.
_BUCKET_DEBUG_LOG = os.getenv(
    "OCR_AUTOKERNEL_BUCKET_DEBUG_LOG", "0"
).strip().lower() in {"1", "true", "yes", "on"}


class OCRBackend(Protocol):
    """Common interface for OCR engines used by the service."""

    name: str

    def predict(self, image_np: np.ndarray) -> list:
        """Return PaddleOCR-compatible prediction results."""

    def warmup(self) -> None:
        """Run a small inference to initialize runtime kernels."""

    def close(self) -> None:
        """Release backend resources."""


class PaddleOCRBackend:
    """Existing PaddleOCR runtime wrapped behind the backend interface."""

    name = "paddle"

    def __init__(
        self,
        config_path: str,
        paddle_ocr_cls: Optional[Callable[..., Any]] = None,
    ) -> None:
        if paddle_ocr_cls is None:
            from paddleocr import PaddleOCR as paddle_ocr_cls

        self.config_path = config_path
        self.engine = paddle_ocr_cls(paddlex_config=config_path)

    def predict(self, image_np: np.ndarray) -> list:
        return self.engine.predict(image_np)

    def warmup(self) -> None:
        self.predict(np.zeros((540, 856, 3), dtype=np.uint8))

    def close(self) -> None:
        close = getattr(self.engine, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class KernelReplacementSpec:
    """Serializable description of a verified AutoKernel replacement."""

    kernel_type: str
    rank: int
    speedup: float
    optimized_path: Path
    status: str = "done"


@dataclass(frozen=True)
class DetectorPreprocessState:
    """Cached detector preprocess metadata for the strict-parity GPU path."""

    resize_op: Any
    scale: float
    mean: Any
    std: Any


def _settings_base_dir(app_settings: Any) -> Path:
    config_path = getattr(app_settings, "config_path", None)
    if isinstance(config_path, (str, Path)):
        return Path(config_path).expanduser().resolve().parent
    return Path.cwd()


def _resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _minarea_box_ordered(pts: np.ndarray) -> np.ndarray:
    """Mirror of paddlex.inference.pipelines.components.common.crop_image_regions
    ``get_minarea_rect_crop`` corner selection: fit minimum-area rect, take its
    4 corners, and sort into TL, TR, BR, BL based on x/y — ensures the crop
    uses a proper rectangle even when the detector poly is slightly skewed.
    """
    import cv2

    bounding_box = cv2.minAreaRect(np.asarray(pts).astype(np.int32))
    corners = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
    # `corners` is now sorted by x-ascending. The first two are the LEFT pair,
    # last two are the RIGHT pair. Within each pair, the smaller-y is TOP.
    if corners[1][1] > corners[0][1]:
        index_a, index_d = 0, 1   # a=TL, d=BL
    else:
        index_a, index_d = 1, 0
    if corners[3][1] > corners[2][1]:
        index_b, index_c = 2, 3   # b=TR, c=BR
    else:
        index_b, index_c = 3, 2
    # Paddle returns the box in order [TL, TR, BR, BL].
    box = np.array(
        [corners[index_a], corners[index_b], corners[index_c], corners[index_d]],
        dtype=np.float32,
    )
    return box


def _crop_text_region(image: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Crop and perspective-correct a text region.

    Byte-for-byte mirrors PaddleOCR's ``get_minarea_rect_crop`` →
    ``get_rotate_crop_image`` flow: min-area rect corner selection,
    perspective warp with INTER_CUBIC + BORDER_REPLICATE, and a rot90
    when the result is >= 1.5× taller than wide. Matching these three
    defaults closed the last preprocessing gap vs Paddle-native rec.
    """
    import cv2

    points = _minarea_box_ordered(np.asarray(poly))
    # Width = max of the two horizontal sides; height = max of the two vertical.
    img_crop_width = int(
        max(np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]))
    )
    img_crop_height = int(
        max(np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]))
    )
    pts_std = np.float32([
        [0, 0],
        [img_crop_width, 0],
        [img_crop_width, img_crop_height],
        [0, img_crop_height],
    ])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        image, M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    # Tall crops rotated 90° CCW (matches PaddleOCR post-warp rule).
    if dst_img.shape[0] * 1.0 / max(dst_img.shape[1], 1) >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def _resize_norm_rec_crop(
    crop: np.ndarray,
    *,
    img_c: int,
    img_h: int,
    target_w: int,
    resized_w: int,
) -> np.ndarray:
    """Mirror PaddleOCR2Pytorch TextRecognizer.resize_norm_img for one crop."""
    import cv2

    resized = cv2.resize(crop, (int(resized_w), int(img_h)))
    resized = resized.astype("float32")
    resized = resized.transpose((2, 0, 1)) / 255.0
    resized -= 0.5
    resized /= 0.5
    out = np.zeros((img_c, int(img_h), int(target_w)), dtype=np.float32)
    out[:, :, : int(resized_w)] = resized
    return out


def _resize_rec_crop(crop: np.ndarray, *, resized_w: int, img_h: int) -> np.ndarray:
    """Resize one recognizer crop with the exact OpenCV path used upstream."""
    import cv2

    return cv2.resize(crop, (int(resized_w), int(img_h)))


def _compute_det_resize_shape(
    height: int,
    width: int,
    *,
    limit_side_len: float,
    limit_type: str,
) -> tuple[int, int, float, float]:
    """Mirror PaddleOCR2Pytorch DetResizeForTest.resize_image_type0."""
    if limit_type == "max":
        if max(height, width) > limit_side_len:
            ratio = float(limit_side_len) / float(max(height, width))
        else:
            ratio = 1.0
    elif limit_type == "min":
        if min(height, width) < limit_side_len:
            ratio = float(limit_side_len) / float(min(height, width))
        else:
            ratio = 1.0
    elif limit_type == "resize_long":
        ratio = float(limit_side_len) / float(max(height, width))
    else:
        raise ValueError(f"Unsupported detector limit_type: {limit_type}")

    resize_h = int(height * ratio)
    resize_w = int(width * ratio)
    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)
    ratio_h = resize_h / float(height)
    ratio_w = resize_w / float(width)
    return resize_h, resize_w, ratio_h, ratio_w


def _sort_text_boxes(boxes: list[np.ndarray]) -> list[np.ndarray]:
    """Sort detector boxes in PaddleOCR reading order.

    PaddleOCR's full TextSystem sorts detector output before cropping. The
    detector-only API can return bottom-up boxes, so hybrid must apply the same
    top-to-bottom, left-to-right ordering before recognition.
    """
    if not boxes:
        return []

    ordered = sorted(
        boxes,
        key=lambda box: (
            float(np.asarray(box)[0][1]),
            float(np.asarray(box)[0][0]),
        ),
    )
    ordered = list(ordered)

    for i in range(len(ordered) - 1):
        for j in range(i, -1, -1):
            current = np.asarray(ordered[j])
            next_box = np.asarray(ordered[j + 1])
            if (
                abs(float(next_box[0][1]) - float(current[0][1])) < 10
                and float(next_box[0][0]) < float(current[0][0])
            ):
                ordered[j], ordered[j + 1] = ordered[j + 1], ordered[j]
            else:
                break

    return ordered


def _build_mixed_precision_detector_net(det_net: Any, torch: Any, dtype: Any) -> Any:
    """Run backbone in ``dtype`` (fp16/bf16), keep neck+head in fp32.

    PP-OCRv5's server detector emits ~90k activations from the neck and relies
    on the head's BatchNorm (running_std ≈ 5.9e5) to normalise them. fp16 tops
    out at 65504, so the neck output saturates and BN then produces NaN for
    the entire probability map. The backbone itself stays well within fp16
    range, so casting only the backbone recovers most of the speedup without
    the overflow.
    """

    det_net.backbone.to(dtype=dtype)
    det_net.neck.float()
    det_net.head.float()

    class _MixedPrecisionDetNet(torch.nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x: Any) -> Any:
            # PPHGNetV2 backbone always returns 4 feature maps (c2..c5). Dynamo
            # mangles ``type(x)(generator)`` / ``[t for t in x]`` patterns here
            # (produces an empty container inside the compiled graph), so we
            # unpack explicitly before casting back to fp32 for the fp32 neck.
            c2, c3, c4, c5 = self.wrapped.backbone(x.to(dtype=dtype))
            feats = [c2.float(), c3.float(), c4.float(), c5.float()]
            return self.wrapped.head(self.wrapped.neck(feats))

    return _MixedPrecisionDetNet(det_net).eval()


def _import_module_from_path(module_name: str, module_path: Path) -> Any:
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def resolve_optimized_kernel_path(
    path_value: str | Path,
    autokernel_root: str | Path,
    workspace_path: str | Path,
) -> Path:
    """
    Resolve optimized kernel paths from verification files.

    AutoKernel artifacts may contain absolute paths from the machine that produced
    them. For deployment we keep the basename stable and search the configured
    workspace and AutoKernel workspace before failing.
    """

    raw_path = Path(path_value).expanduser()
    root = Path(autokernel_root).expanduser().resolve()
    workspace = Path(workspace_path).expanduser().resolve()

    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                (Path.cwd() / raw_path).resolve(),
                (root / raw_path).resolve(),
                (workspace / raw_path).resolve(),
            ]
        )

    if raw_path.name:
        candidates.extend(
            [
                (workspace / raw_path.name).resolve(),
                (root / "workspace" / raw_path.name).resolve(),
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Optimized kernel {path_value!s} was not found in configured AutoKernel paths"
    )


def load_verified_replacement_specs(
    workspace_path: str | Path,
    autokernel_root: str | Path,
) -> List[KernelReplacementSpec]:
    """
    Load the verified AutoKernel replacement set from deployment artifacts.

    `verification_result.json` is preferred because it records the final stack
    that passed correctness. `orchestration_state.json` is a fallback for older
    workspaces that do not have the verifier output.
    """

    workspace = Path(workspace_path).expanduser().resolve()
    root = Path(autokernel_root).expanduser().resolve()
    verification_path = workspace / "verification_result.json"
    state_path = workspace / "orchestration_state.json"

    specs: list[KernelReplacementSpec] = []
    if verification_path.exists():
        data = json.loads(verification_path.read_text(encoding="utf-8"))
        correctness = str(
            data.get("verification", {}).get("correctness", "")
        ).upper()
        if correctness and correctness != "PASS":
            raise RuntimeError(
                f"AutoKernel verification did not pass: {verification_path}"
            )

        for item in data.get("optimized", {}).get("kernels_replaced", []):
            kernel_type = item.get("type") or item.get("op_type") or item.get("kernel_type")
            raw_path = item.get("path") or item.get("optimized_path")
            if not kernel_type or not raw_path:
                continue
            specs.append(
                KernelReplacementSpec(
                    kernel_type=str(kernel_type),
                    rank=int(item.get("rank", 0)),
                    speedup=float(item.get("speedup", 1.0)),
                    optimized_path=resolve_optimized_kernel_path(raw_path, root, workspace),
                    status="done",
                )
            )
        return specs

    if not state_path.exists():
        return []

    data = json.loads(state_path.read_text(encoding="utf-8"))
    for item in data.get("kernels", []):
        status = str(item.get("status", "")).strip().lower()
        speedup = float(
            item.get("end_to_end_speedup", item.get("speedup", item.get("best_speedup", 1.0)))
        )
        include_candidate = (
            status in {"done", "optimizing"}
            if status
            else speedup > 1.0
        )
        if not include_candidate:
            continue

        kernel_type = item.get("op_type") or item.get("type") or item.get("kernel_type")
        raw_path = item.get("optimized_path")
        if not raw_path:
            source_file = item.get("file")
            if source_file:
                raw_path = str(Path(source_file).with_name(Path(source_file).stem + "_optimized.py"))
        if not kernel_type or not raw_path:
            continue

        specs.append(
            KernelReplacementSpec(
                kernel_type=str(kernel_type),
                rank=int(item.get("rank", 0)),
                speedup=speedup,
                optimized_path=resolve_optimized_kernel_path(raw_path, root, workspace),
                status=status or "done",
            )
        )
    return specs


class AutoKernelPPOCRv5Backend:
    """
    PP-OCRv5 server backend using converted PyTorch models and AutoKernel kernels.

    The detector and recognizer are both loaded from the converted PP-OCRv5 server
    weights. The verified optimized kernel stack is applied to the recognizer
    when configured, matching the artifact that passed AutoKernel verification.
    """

    name = "autokernel"

    def __init__(
        self,
        *,
        autokernel_root: str | Path,
        ppocr_root: str | Path,
        det_weights_path: str | Path,
        rec_weights_path: str | Path,
        det_source_path: str | Path | None,
        rec_source_path: str | Path | None,
        workspace_path: str | Path,
        use_gpu: bool,
        auto_convert_weights: bool = True,
        optimize_recognizer: bool = True,
        optimize_detector: bool = False,
        rec_batch_size: int = 1,
        rec_image_shape: str = "3,48,320",
        dtype: str = "float16",
        exclude_kernel_types: Optional[List[str]] = None,
    ) -> None:
        self.autokernel_root = Path(autokernel_root).expanduser().resolve()
        self.ppocr_root = Path(ppocr_root).expanduser().resolve()
        self.det_weights_path = Path(det_weights_path).expanduser().resolve()
        self.rec_weights_path = Path(rec_weights_path).expanduser().resolve()
        self.exclude_kernel_types: set[str] = set(exclude_kernel_types or [])
        self.det_source_path = (
            Path(det_source_path).expanduser().resolve()
            if det_source_path
            else None
        )
        self.rec_source_path = (
            Path(rec_source_path).expanduser().resolve()
            if rec_source_path
            else None
        )
        self.workspace_path = Path(workspace_path).expanduser().resolve()
        self.use_gpu = use_gpu
        self.auto_convert_weights = auto_convert_weights
        self.optimize_recognizer = optimize_recognizer
        self.optimize_detector = optimize_detector
        self.rec_batch_size = rec_batch_size
        self.rec_image_shape = rec_image_shape
        self.dtype_name = dtype
        self._contexts: list[Any] = []

        self._ensure_converted_weights()
        self._validate_paths()
        self._configure_environment()
        self.system = self._build_system()

    def _ensure_converted_weights(self) -> None:
        if not self.auto_convert_weights:
            return
        if self.det_weights_path.exists() and self.rec_weights_path.exists():
            return

        ensure_ppocrv5_server_weights(
            PPOCRv5ServerConversionConfig(
                ppocr_root=self.ppocr_root,
                det_source_path=self.det_source_path,
                rec_source_path=self.rec_source_path,
                det_output_path=self.det_weights_path,
                rec_output_path=self.rec_weights_path,
                components=("det", "rec"),
            )
        )

    def _validate_paths(self) -> None:
        required_paths = {
            "AutoKernel root": self.autokernel_root,
            "PaddleOCR2Pytorch root": self.ppocr_root,
            "detector weights": self.det_weights_path,
            "recognizer weights": self.rec_weights_path,
        }
        for label, path in required_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")
        if self.optimize_recognizer and not self.workspace_path.exists():
            raise FileNotFoundError(
                f"AutoKernel workspace not found: {self.workspace_path}"
            )

    def _configure_environment(self) -> None:
        _prepend_sys_path(self.autokernel_root)
        _prepend_sys_path(self.ppocr_root)
        os.environ["AUTOKERNEL_PPOCR_ROOT"] = str(self.ppocr_root)
        os.environ["AUTOKERNEL_PPOCRV5_SERVER_DET_PTH"] = str(self.det_weights_path)
        os.environ["AUTOKERNEL_PPOCRV5_SERVER_REC_PTH"] = str(self.rec_weights_path)

    def _build_system(self) -> Any:
        torch = importlib.import_module("torch")
        if self.use_gpu and not torch.cuda.is_available():
            raise RuntimeError("AutoKernel OCR backend requires CUDA when GPU mode is enabled")

        ppocr_model = importlib.import_module("models.ppocrv5_server")
        predict_system = importlib.import_module("tools.infer.predict_system")
        pytorchocr_utility = importlib.import_module("tools.infer.pytorchocr_utility")

        parser = pytorchocr_utility.init_args()
        args = parser.parse_args([])
        args.use_gpu = self.use_gpu
        args.use_angle_cls = False
        args.det_algorithm = "DB"
        args.det_yaml_path = str(self.ppocr_root / ppocr_model.DET_YAML_RELATIVE)
        args.det_model_path = str(self.det_weights_path)
        args.rec_yaml_path = str(self.ppocr_root / ppocr_model.REC_YAML_RELATIVE)
        args.rec_model_path = str(self.rec_weights_path)
        args.rec_image_shape = self.rec_image_shape
        args.rec_char_dict_path = str(
            self.ppocr_root / "pytorchocr/utils/dict/ppocrv5_dict.txt"
        )
        args.rec_batch_num = self.rec_batch_size
        args.image_dir = ""

        system = predict_system.TextSystem(args)
        det_wrapper = ppocr_model.PPOCRv5ServerDetModel()
        rec_wrapper = ppocr_model.PPOCRv5ServerRecModel()

        device = "cuda" if self.use_gpu else "cpu"
        system.text_detector.net = det_wrapper.net.to(device).eval()
        if self.use_gpu:
            system.text_detector.net = system.text_detector.net.float()

        system.text_recognizer.net = self._build_recognizer_net(rec_wrapper, torch, device)
        if self.optimize_detector:
            logger.warning(
                "AutoKernel detector optimization requested, but no detector "
                "kernel stack is verified. The converted detector is used without "
                "kernel replacement."
            )

        return system

    def _build_recognizer_net(self, rec_wrapper: Any, torch: Any, device: str) -> Any:
        dtype = torch.float32 if device == "cpu" else self._torch_dtype(torch)
        model = rec_wrapper.to(device=device, dtype=dtype).eval()
        context = None
        patched_model = model

        if self.optimize_recognizer:
            if device != "cuda":
                raise RuntimeError("AutoKernel recognizer optimization requires CUDA")
            replacements = self._load_kernel_replacements()
            if not replacements:
                raise RuntimeError(
                    f"No verified AutoKernel recognizer kernels found in {self.workspace_path}"
                )
            verify_mod = self._load_verify_module()
            context = verify_mod.OptimizedModelContext(model, replacements)
            patched_model = context.__enter__()
            self._contexts.append(context)
            logger.info(
                "Applied AutoKernel recognizer replacements: %s applied, %s skipped",
                len(getattr(context, "_applied_replacements", [])),
                len(getattr(context, "_skipped_replacements", [])),
            )

        class _RecognizerNetAdapter(torch.nn.Module):
            def __init__(self, runtime_model: Any, optimized_context: Any) -> None:
                super().__init__()
                self.runtime_model = runtime_model
                self.optimized_context = optimized_context

            def forward(self, x: Any) -> Any:
                x = x.to(device=device, dtype=dtype)
                if self.optimized_context is not None:
                    x = self.optimized_context.prepare_input(x)
                return self.runtime_model(x)

        adapter = _RecognizerNetAdapter(patched_model, context).eval()
        logger.info(
            "Recognizer adapter ready (device=%s, dtype=%s, optimized=%s)",
            device, dtype, self.optimize_recognizer,
        )
        return adapter

    def _torch_dtype(self, torch: Any) -> Any:
        normalized = self.dtype_name.lower()
        if normalized in {"float16", "fp16", "half"}:
            return torch.float16
        if normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if normalized in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"Unsupported AutoKernel OCR dtype: {self.dtype_name}")

    def _load_verify_module(self) -> Any:
        return _import_module_from_path(
            "dgc_ext_autokernel_verify",
            self.autokernel_root / "verify.py",
        )

    def _load_kernel_replacements(self) -> list[Any]:
        specs = load_verified_replacement_specs(self.workspace_path, self.autokernel_root)
        verify_mod = self._load_verify_module()
        support_mod = importlib.import_module("support")

        replacements = []
        for spec in specs:
            if spec.kernel_type in self.exclude_kernel_types:
                logger.info(
                    "Skipping excluded kernel type %s (rank %d)",
                    spec.kernel_type, spec.rank,
                )
                continue
            support_stage = support_mod.build_support_stage(
                spec.kernel_type,
                {spec.kernel_type},
            )
            replacements.append(
                verify_mod.KernelReplacement(
                    kernel_type=spec.kernel_type,
                    rank=spec.rank,
                    speedup=spec.speedup,
                    optimized_path=str(spec.optimized_path),
                    reinsert_supported=support_stage["reinsert_supported"],
                    status=spec.status,
                )
            )
        return replacements

    def predict(self, image_np: np.ndarray) -> list:
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError(f"Expected RGB image array with shape HxWx3, got {image_np.shape}")

        image_rgb = np.ascontiguousarray(image_np)
        boxes, rec_res, _timings = self.system(image_rgb)
        return [self._to_paddle_result(boxes, rec_res)]

    @staticmethod
    def _to_paddle_result(boxes: Any, rec_res: Any) -> dict:
        boxes = [] if boxes is None else boxes
        rec_res = [] if rec_res is None else rec_res

        rec_texts: list[str] = []
        rec_scores: list[float] = []
        rec_polys: list[np.ndarray] = []

        for box, rec in zip(boxes, rec_res):
            if rec is None:
                continue
            text, score = rec[0], rec[1]
            rec_texts.append(str(text))
            rec_scores.append(float(score))
            rec_polys.append(np.asarray(box))

        return {
            "rec_texts": rec_texts,
            "rec_scores": rec_scores,
            "rec_polys": rec_polys,
        }

    def warmup(self) -> None:
        self.predict(np.zeros((540, 856, 3), dtype=np.uint8))

    def close(self) -> None:
        for context in reversed(self._contexts):
            context.__exit__(None, None, None)
        self._contexts.clear()

        try:
            torch = importlib.import_module("torch")
        except Exception:
            return
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


class FullPyTorchBackend:
    """PP-OCRv5 server with PyTorch detector + PyTorch recognizer, no AutoKernel.

    Loads the same converted `.pth` weights as `AutoKernelPPOCRv5Backend` but
    runs them in plain eager-mode PyTorch — no kernel replacement, no
    torch.compile, no workspace dependency. Intended as:

      * a reference implementation for Paddle→PyTorch parity verification
        (see `scripts/verify_det_logits.py` and
        `tests/test_det_polygon_parity.py`),
      * a profiling baseline before applying further optimizations.

    The detector stays fp32 because DB post-processing is sensitive to
    shrink-map precision (fp16 loses polygons entirely on KTP images). The
    recognizer honours the configured dtype (default fp16, verified bit-exact
    vs Paddle on good_data_{1,2,3}).
    """

    name = "fullpytorch"

    def __init__(
        self,
        *,
        autokernel_root: str | Path,
        ppocr_root: str | Path,
        det_weights_path: str | Path,
        rec_weights_path: str | Path,
        det_source_path: str | Path | None,
        rec_source_path: str | Path | None,
        use_gpu: bool,
        auto_convert_weights: bool = True,
        rec_batch_size: int = 8,
        rec_image_shape: str = "3,48,320",
        rec_bucket_max_width_ratio: float = 1.30,
        det_limit_side_len: int = 1280,
        det_limit_type: str = "max",
        torch_compile_det: bool = False,
        torch_compile_rec: bool = False,
        torch_compile_mode: str = "default",
        torch_compile_dynamic: bool = True,
        warmup_image_path: str | Path | None = None,
        dtype: str = "float16",
        det_dtype: str = "float32",
    ) -> None:
        self.autokernel_root = Path(autokernel_root).expanduser().resolve()
        self.ppocr_root = Path(ppocr_root).expanduser().resolve()
        self.det_weights_path = Path(det_weights_path).expanduser().resolve()
        self.rec_weights_path = Path(rec_weights_path).expanduser().resolve()
        self.det_source_path = (
            Path(det_source_path).expanduser().resolve() if det_source_path else None
        )
        self.rec_source_path = (
            Path(rec_source_path).expanduser().resolve() if rec_source_path else None
        )
        self.use_gpu = use_gpu
        self.auto_convert_weights = auto_convert_weights
        self.rec_batch_size = rec_batch_size
        self.rec_image_shape = rec_image_shape
        self.rec_bucket_max_width_ratio = rec_bucket_max_width_ratio
        self.det_limit_side_len = det_limit_side_len
        self.det_limit_type = det_limit_type
        self.torch_compile_det = torch_compile_det
        self.torch_compile_rec = torch_compile_rec
        self.torch_compile_mode = torch_compile_mode
        self.torch_compile_dynamic = torch_compile_dynamic
        self.warmup_image_path = (
            Path(warmup_image_path).expanduser().resolve() if warmup_image_path else None
        )
        self.dtype_name = dtype
        self.det_dtype_name = det_dtype

        self._ensure_converted_weights()
        self._validate_paths()
        self._configure_environment()
        self.system = self._build_system()

    def _ensure_converted_weights(self) -> None:
        if not self.auto_convert_weights:
            return
        if self.det_weights_path.exists() and self.rec_weights_path.exists():
            return
        ensure_ppocrv5_server_weights(
            PPOCRv5ServerConversionConfig(
                ppocr_root=self.ppocr_root,
                det_source_path=self.det_source_path,
                rec_source_path=self.rec_source_path,
                det_output_path=self.det_weights_path,
                rec_output_path=self.rec_weights_path,
                components=("det", "rec"),
            )
        )

    def _validate_paths(self) -> None:
        required = {
            "PaddleOCR2Pytorch root": self.ppocr_root,
            "detector weights": self.det_weights_path,
            "recognizer weights": self.rec_weights_path,
            "autokernel models dir": self.autokernel_root / "models",
        }
        for label, path in required.items():
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

    def _configure_environment(self) -> None:
        _prepend_sys_path(self.autokernel_root)
        _prepend_sys_path(self.ppocr_root)
        os.environ["AUTOKERNEL_PPOCR_ROOT"] = str(self.ppocr_root)
        os.environ["AUTOKERNEL_PPOCRV5_SERVER_DET_PTH"] = str(self.det_weights_path)
        os.environ["AUTOKERNEL_PPOCRV5_SERVER_REC_PTH"] = str(self.rec_weights_path)

    def _torch_dtype(self, torch: Any) -> Any:
        return self._resolve_dtype(torch, self.dtype_name)

    def _torch_det_dtype(self, torch: Any) -> Any:
        return self._resolve_dtype(torch, self.det_dtype_name)

    @staticmethod
    def _resolve_dtype(torch: Any, name: str) -> Any:
        normalized = name.lower()
        if normalized in {"float16", "fp16", "half"}:
            return torch.float16
        if normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if normalized in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"Unsupported FullPyTorch OCR dtype: {name}")

    def _build_system(self) -> Any:
        torch = importlib.import_module("torch")
        if self.use_gpu and not torch.cuda.is_available():
            raise RuntimeError("FullPyTorch OCR backend requires CUDA when GPU mode is enabled")

        ppocr_model = importlib.import_module("models.ppocrv5_server")
        predict_system = importlib.import_module("tools.infer.predict_system")
        pytorchocr_utility = importlib.import_module("tools.infer.pytorchocr_utility")

        parser = pytorchocr_utility.init_args()
        args = parser.parse_args([])
        args.use_gpu = self.use_gpu
        args.use_angle_cls = False
        args.det_algorithm = "DB"
        args.det_yaml_path = str(self.ppocr_root / ppocr_model.DET_YAML_RELATIVE)
        args.det_model_path = str(self.det_weights_path)
        args.det_limit_side_len = self.det_limit_side_len
        args.det_limit_type = self.det_limit_type
        args.rec_yaml_path = str(self.ppocr_root / ppocr_model.REC_YAML_RELATIVE)
        args.rec_model_path = str(self.rec_weights_path)
        args.rec_image_shape = self.rec_image_shape
        args.rec_char_dict_path = str(
            self.ppocr_root / "pytorchocr/utils/dict/ppocrv5_dict.txt"
        )
        args.rec_batch_num = self.rec_batch_size
        args.image_dir = ""

        system = predict_system.TextSystem(args)
        device = "cuda" if self.use_gpu else "cpu"
        rec_dtype = self._torch_dtype(torch)

        det_wrapper = ppocr_model.PPOCRv5ServerDetModel()
        det_net = det_wrapper.net.to(device).float().eval()
        det_dtype = self._torch_det_dtype(torch)
        if device == "cuda" and det_dtype is not torch.float32:
            system.text_detector.net = _build_mixed_precision_detector_net(
                det_net, torch, det_dtype
            )
            logger.info(
                "FullPyTorch detector running mixed precision "
                "(backbone=%s, neck=fp32, head=fp32) — PP-OCRv5's neck emits "
                "activations above fp16 range so full fp16 produces NaN maps.",
                det_dtype,
            )
        else:
            system.text_detector.net = det_net

        rec_wrapper = ppocr_model.PPOCRv5ServerRecModel()
        system.text_recognizer.net = self._build_recognizer_adapter(
            rec_wrapper, torch, device, rec_dtype
        )
        self._maybe_compile_models(system, torch)
        self._rec_imgC, self._rec_imgH, self._rec_imgW = (
            int(v) for v in self.rec_image_shape.split(",")
        )
        return system

    def _maybe_compile_models(self, system: Any, torch: Any) -> None:
        if not (self.torch_compile_det or self.torch_compile_rec):
            return
        if not hasattr(torch, "compile"):
            raise RuntimeError("FullPyTorch torch.compile requested but unavailable")
        # Default cache_size_limit is 8. With bucketed rec we expect up to
        # len(REC_W_BUCKETS) compiled graphs plus headroom; once exceeded,
        # dynamo silently falls back to eager AND keeps retracing on each new
        # shape — the exact steady-state failure mode we're trying to avoid.
        torch._dynamo.config.cache_size_limit = max(
            64, len(REC_W_BUCKETS) * 4
        )
        compile_kwargs = {
            "mode": self.torch_compile_mode,
            "dynamic": self.torch_compile_dynamic,
        }
        if self.torch_compile_det:
            # PPHGNetV2 stem uses Conv2d(kernel=2, stride=1, padding="same") and a
            # dynamic-pad MaxPool wrapper. Both produce Max(0, CeilToInt(...))
            # shape math that Inductor's FloorDiv simplifier can't evaluate
            # under dynamic=True. Rewrite them to static F.pad((0,1,0,1)) +
            # padding=0, which is bit-exact for the k=2,s=1 config in use.
            self._rewrite_det_stem_for_compile(system.text_detector.net, torch)
            system.text_detector.net = torch.compile(
                system.text_detector.net, **compile_kwargs
            )
            logger.info(
                "FullPyTorch detector compiled with torch.compile(mode=%s, dynamic=%s)",
                self.torch_compile_mode,
                self.torch_compile_dynamic,
            )
        if self.torch_compile_rec:
            system.text_recognizer.net.runtime_model = torch.compile(
                system.text_recognizer.net.runtime_model, **compile_kwargs
            )
            logger.info(
                "FullPyTorch recognizer compiled with torch.compile(mode=%s, dynamic=%s)",
                self.torch_compile_mode,
                self.torch_compile_dynamic,
            )

    @staticmethod
    def _rewrite_det_stem_for_compile(det_net: Any, torch: Any) -> None:
        import torch.nn as nn
        import torch.nn.functional as F

        class _StaticPadConvBNAct(nn.Module):
            def __init__(self, src: Any) -> None:
                super().__init__()
                c = src.conv
                new_conv = nn.Conv2d(
                    c.in_channels,
                    c.out_channels,
                    kernel_size=2,
                    stride=1,
                    padding=0,
                    groups=c.groups,
                    bias=c.bias is not None,
                )
                new_conv.weight.data.copy_(c.weight.data)
                if c.bias is not None:
                    new_conv.bias.data.copy_(c.bias.data)
                new_conv.to(c.weight.device, c.weight.dtype)
                self.conv = new_conv
                self.bn = src.bn
                self.use_act = src.use_act
                self.use_lab = src.use_lab
                if self.use_act:
                    self.act = src.act
                    if self.use_lab:
                        self.lab = src.lab

            def forward(self, x: Any) -> Any:
                x = F.pad(x, (0, 1, 0, 1))
                x = self.conv(x)
                x = self.bn(x)
                if self.use_act:
                    x = self.act(x)
                    if self.use_lab:
                        x = self.lab(x)
                return x

        class _StaticPadMaxPoolK2S1(nn.Module):
            def __init__(self, src: Any) -> None:
                super().__init__()
                self.pool = src.pool

            def forward(self, x: Any) -> Any:
                return self.pool(F.pad(x, (0, 1, 0, 1)))

        for parent in list(det_net.modules()):
            for child_name, child in list(parent.named_children()):
                conv = getattr(child, "conv", None)
                if (
                    isinstance(conv, nn.Conv2d)
                    and conv.padding == "same"
                    and conv.kernel_size == (2, 2)
                    and conv.stride == (1, 1)
                ):
                    setattr(parent, child_name, _StaticPadConvBNAct(child))
                    continue
                if (
                    type(child).__name__ == "PaddingSameAsPaddleMaxPool2d"
                    and getattr(child, "kernel_size", None) == 2
                    and getattr(child, "stride", None) == 1
                ):
                    setattr(parent, child_name, _StaticPadMaxPoolK2S1(child))

        # PPHGNetV2.forward builds its det output as ``out = []; out.append(...)
        # in a loop; return out``. Under torch.compile Dynamo traces the list
        # as empty (the appended items don't reach the graph output), so the
        # neck receives ``[]`` and ``c2,c3,c4,c5 = x`` fails. Rebind forward to
        # an unrolled version that returns an explicit 4-tuple. Assumes
        # out_indices == [0,1,2,3] (all four stages), which matches the det
        # config in use.
        for m in det_net.modules():
            if type(m).__name__ == "PPHGNetV2" and getattr(m, "det", False):
                if getattr(m, "out_indices", None) != [0, 1, 2, 3]:
                    continue
                if len(m.stages) != 4:
                    continue

                def _forward_4tuple(self: Any, x: Any) -> Any:
                    x = self.stem(x)
                    c2 = self.stages[0](x)
                    c3 = self.stages[1](c2)
                    c4 = self.stages[2](c3)
                    c5 = self.stages[3](c4)
                    return (c2, c3, c4, c5)

                m.forward = _forward_4tuple.__get__(m, type(m))

    def _build_detector_adapter(
        self, det_wrapper: Any, torch: Any, device: str, dtype: Any
    ) -> Any:
        """Cast det net to ``dtype`` on GPU, cast input tensor at forward time.

        TextDetector's shrink-map post-process runs on the fp32 output of this
        adapter (``preds.float()`` inside the DB head is not required — we cast
        output back to fp32 here so downstream numpy conversion is identical).
        """
        runtime_dtype = torch.float32 if device == "cpu" else dtype
        model = det_wrapper.net.to(device=device, dtype=runtime_dtype).eval()

        class _DetectorAdapter(torch.nn.Module):
            def __init__(self, runtime_model: Any) -> None:
                super().__init__()
                self.runtime_model = runtime_model

            def forward(self, x: Any) -> Any:
                x = x.to(device=device, dtype=runtime_dtype)
                torch._dynamo.maybe_mark_dynamic(x, 2)
                torch._dynamo.maybe_mark_dynamic(x, 3)
                out = self.runtime_model(x)
                if isinstance(out, dict):
                    return {k: v.float() for k, v in out.items()}
                if isinstance(out, (list, tuple)):
                    return type(out)(v.float() for v in out)
                return out.float()

        logger.info(
            "FullPyTorch detector adapter ready (device=%s, dtype=%s)",
            device,
            runtime_dtype,
        )
        return _DetectorAdapter(model).eval()

    def _build_recognizer_adapter(
        self, rec_wrapper: Any, torch: Any, device: str, dtype: Any
    ) -> Any:
        """Wrap the rec net so TextRecognizer's fp32 input is cast to ``dtype``.

        At device=cpu, always use fp32 (fp16 CPU forward is slow and error-prone
        under some PyTorch versions).
        """
        runtime_dtype = torch.float32 if device == "cpu" else dtype
        model = rec_wrapper.to(device=device, dtype=runtime_dtype).eval()

        class _RecognizerAdapter(torch.nn.Module):
            def __init__(self, runtime_model: Any) -> None:
                super().__init__()
                self.runtime_model = runtime_model

            def forward(self, x: Any) -> Any:
                # Bucketed input means the model sees exactly len(REC_W_BUCKETS)
                # distinct (B=rec_batch_size, C, H, W) shapes — each compiles
                # once during warmup. No dynamic marking; let Inductor fully
                # specialize per bucket. .contiguous() kills stride-guard
                # recompiles (L['x'].stride()[0] == C*H*W and friends) that
                # otherwise fire on workspace re-views.
                x = x.to(device=device, dtype=runtime_dtype).contiguous()
                return self.runtime_model(x)

        logger.info(
            "FullPyTorch recognizer adapter ready (device=%s, dtype=%s)",
            device,
            runtime_dtype,
        )
        return _RecognizerAdapter(model).eval()

    def _get_detector_preprocess_state(self, torch: Any) -> DetectorPreprocessState:
        state = getattr(self, "_detector_preprocess_state", None)
        if state is not None:
            return state

        detector = self.system.text_detector
        resize_op = next(
            op for op in detector.preprocess_op if type(op).__name__ == "DetResizeForTest"
        )
        norm_op = next(
            op for op in detector.preprocess_op if type(op).__name__ == "NormalizeImage"
        )
        state = DetectorPreprocessState(
            resize_op=resize_op,
            scale=float(getattr(norm_op, "scale", 1.0 / 255.0)),
            mean=torch.as_tensor(
                np.asarray(norm_op.mean, dtype=np.float32).reshape(-1),
                device="cuda",
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            std=torch.as_tensor(
                np.asarray(norm_op.std, dtype=np.float32).reshape(-1),
                device="cuda",
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )
        self._detector_preprocess_state = state
        return state

    def _get_rec_batch_workspace(
        self,
        torch: Any,
        *,
        slot: int,
        batch_size: int,
        target_w: int,
        dtype: Any,
        device: str,
    ) -> Any:
        workspaces = getattr(self, "_rec_batch_workspaces", None)
        if workspaces is None:
            workspaces = [None, None]
            self._rec_batch_workspaces = workspaces

        workspace = workspaces[slot]
        if (
            workspace is None
            or workspace.device.type != device
            or workspace.dtype != dtype
            or workspace.shape[0] < batch_size
            or workspace.shape[3] < target_w
        ):
            workspace = torch.empty(
                (
                    max(int(self.rec_batch_size), batch_size),
                    self._rec_imgC,
                    self._rec_imgH,
                    target_w,
                ),
                dtype=dtype,
                device=device,
            )
            workspaces[slot] = workspace
        return workspace[:batch_size, :, :, :target_w]

    def _get_rec_stage_stream(self, torch: Any) -> Any:
        stream = getattr(self, "_rec_stage_stream", None)
        if stream is None:
            stream = torch.cuda.Stream()
            self._rec_stage_stream = stream
        return stream

    def _stage_recognition_batch_cuda(
        self,
        torch: Any,
        *,
        slot: int,
        crops: list[np.ndarray],
        batch_indices: list[int],
        resized_widths: dict[int, int],
        target_w: int,
        imgH: int,
        model_dtype: Any,
    ) -> tuple[Any, Any]:
        stage_stream = self._get_rec_stage_stream(torch)
        # Always materialize a full rec_batch_size batch so torch.compile sees
        # a closed set of (B, C, H, W_bucket) shapes. Pad rows beyond
        # len(batch_indices) stay zero from np.zeros below and are sliced
        # off the model output before postprocess. See REC_W_BUCKETS comment.
        batch = self._get_rec_batch_workspace(
            torch,
            slot=slot,
            batch_size=self.rec_batch_size,
            target_w=target_w,
            dtype=model_dtype,
            device="cuda",
        )

        # Build a single zero-padded normalized fp32 batch on CPU. Previously
        # this loop issued one H2D per crop (8 transfers per batch); a single
        # contiguous H2D removes ~7 launch overheads per batch.
        host_buf = np.zeros(
            (self.rec_batch_size, self._rec_imgC, imgH, target_w),
            dtype=np.float32,
        )
        for batch_pos, crop_index in enumerate(batch_indices):
            crop = crops[int(crop_index)]
            resized_w = resized_widths[int(crop_index)]
            host_buf[batch_pos] = _resize_norm_rec_crop(
                crop,
                img_c=self._rec_imgC,
                img_h=imgH,
                target_w=target_w,
                resized_w=resized_w,
            )

        with torch.cuda.stream(stage_stream), torch.inference_mode():
            # copy_(src) handles cross-device + cross-dtype in one transfer.
            batch.copy_(torch.from_numpy(host_buf))
            ready = torch.cuda.Event()
            ready.record(stage_stream)

        return batch, ready

    def _detect_boxes(self, image_rgb: np.ndarray) -> list[np.ndarray]:
        if not self.use_gpu:
            boxes, _elapsed = self.system.text_detector(image_rgb)
            if boxes is None:
                return []
            return _sort_text_boxes([np.asarray(box) for box in boxes])

        import torch
        import cv2

        detector = self.system.text_detector
        preprocess = self._get_detector_preprocess_state(torch)

        src_h, src_w = image_rgb.shape[:2]
        resize_h, resize_w, ratio_h, ratio_w = _compute_det_resize_shape(
            src_h,
            src_w,
            limit_side_len=float(getattr(preprocess.resize_op, "limit_side_len")),
            limit_type=str(getattr(preprocess.resize_op, "limit_type", "min")),
        )
        shape_list = np.asarray([[src_h, src_w, ratio_h, ratio_w]], dtype=np.float32)
        resized = cv2.resize(image_rgb, (int(resize_w), int(resize_h)))

        with torch.inference_mode():
            inp = torch.from_numpy(np.ascontiguousarray(resized)).to(
                device="cuda", dtype=torch.uint8
            )
            inp = inp.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
            inp.mul_(preprocess.scale)
            inp.sub_(preprocess.mean)
            inp.div_(preprocess.std)
            outputs = detector.net(inp)

        if detector.det_algorithm not in {"DB", "DB++", "PSE"}:
            raise NotImplementedError(
                f"GPU detector preprocess path only supports DB-like outputs, got {detector.det_algorithm}"
            )
        preds = {"maps": outputs["maps"].float().cpu().numpy()}
        post_result = detector.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]["points"]
        if dt_boxes is None:
            return []
        dt_boxes = detector.filter_tag_det_res(dt_boxes, image_rgb.shape)
        return _sort_text_boxes([np.asarray(box) for box in dt_boxes])

    def _recognition_batch_plan(
        self, crops: list[np.ndarray]
    ) -> tuple[list[tuple[int, list[int]]], dict[int, int], set[int]]:
        """Plan recognizer batches by sorted width with bounded padding.

        Exact-width grouping preserves CTC parity but creates too many small
        launches on KTP crops. PaddleOCR batches sorted widths and pads each
        batch to the widest crop. This planner keeps that launch reduction, but
        splits a batch when padding would exceed ``rec_bucket_max_width_ratio``.

        Returns (batches, resized_widths, truncated_indices). truncated_indices
        contains crop indices whose natural resized width would have exceeded
        REC_W_BUCKETS[-1] and thus lose right-edge content after bucket capping.
        """
        import math

        if not crops:
            return [], {}, set()

        imgH = self._rec_imgH
        base_ratio = self._rec_imgW / imgH
        recognizer = self.system.text_recognizer
        limited_max = getattr(recognizer, "limited_max_width", 4000)
        limited_min = getattr(recognizer, "limited_min_width", 16)
        batch_num = max(int(self.rec_batch_size), 1)
        max_pad_ratio = max(float(self.rec_bucket_max_width_ratio), 1.0)
        top_bucket = REC_W_BUCKETS[-1]

        target_widths: dict[int, int] = {}
        resized_widths: dict[int, int] = {}
        truncated_indices: set[int] = set()
        items: list[tuple[int, float, int]] = []
        for crop_index, crop in enumerate(crops):
            h, w = crop.shape[:2]
            wh_ratio = w / float(h)
            desired_w = max(
                min(int(imgH * max(wh_ratio, base_ratio)), limited_max),
                limited_min,
            )
            target_w = _snap_rec_width_to_bucket(desired_w)
            natural_w = int(math.ceil(imgH * wh_ratio))
            if natural_w > top_bucket:
                truncated_indices.add(crop_index)
            resized_w = max(
                min(natural_w, target_w),
                limited_min,
            )
            target_widths[crop_index] = target_w
            resized_widths[crop_index] = resized_w
            items.append((target_w, wh_ratio, crop_index))

        items.sort(key=lambda item: (item[0], item[1], item[2]))
        batches: list[tuple[int, list[int]]] = []
        pos = 0
        while pos < len(items):
            min_target_w = items[pos][0]
            batch_indices: list[int] = []
            while pos < len(items) and len(batch_indices) < batch_num:
                target_w, _wh_ratio, crop_index = items[pos]
                if batch_indices and target_w > min_target_w * max_pad_ratio:
                    break
                batch_indices.append(crop_index)
                pos += 1

            target_w = max(target_widths[int(i)] for i in batch_indices)
            batches.append((target_w, batch_indices))

        return batches, resized_widths, truncated_indices

    def _fast_recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Recognise crops with Paddle-compatible preprocessing and GPU-side argmax.

        PaddleOCR2Pytorch's default recognizer path preprocesses crops on CPU
        and copies the full ``B x T x vocab`` logits tensor back to CPU for
        CTC decode. This path keeps Paddle-compatible cv2 resize/normalise,
        then batches model/argmax on GPU and transfers only the compact argmax
        indices and probabilities.

        Whole body runs under torch.inference_mode() so workspace allocations,
        slicing, .contiguous() copies, and model forwards all produce tensors
        with identical dispatch key sets — otherwise the compiled rec graphs
        captured during warmup (in inference_mode) don't match prod tensors
        (allocated in default autograd-on context), and every bucket
        recompiles on first real request.
        """
        import torch

        if not crops:
            return []

        imgC, imgH = self._rec_imgC, self._rec_imgH
        device = "cuda" if self.use_gpu else "cpu"
        model_dtype = torch.float32 if device == "cpu" else self._torch_dtype(torch)
        rec_res: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        batches, resized_widths, truncated_indices = self._recognition_batch_plan(crops)

        if self.use_gpu:
            current_stream = torch.cuda.current_stream()
            prepared_batch, prepared_ready = self._stage_recognition_batch_cuda(
                torch,
                slot=0,
                crops=crops,
                batch_indices=batches[0][1],
                resized_widths=resized_widths,
                target_w=batches[0][0],
                imgH=imgH,
                model_dtype=model_dtype,
            )

        for batch_idx, (target_w, batch_indices) in enumerate(batches):
            if self.use_gpu:
                current_stream.wait_event(prepared_ready)
                batch = prepared_batch
                next_prepared: tuple[Any, Any] | None = None
                if batch_idx + 1 < len(batches):
                    next_target_w, next_batch_indices = batches[batch_idx + 1]
                    next_prepared = self._stage_recognition_batch_cuda(
                        torch,
                        slot=(batch_idx + 1) % 2,
                        crops=crops,
                        batch_indices=next_batch_indices,
                        resized_widths=resized_widths,
                        target_w=next_target_w,
                        imgH=imgH,
                        model_dtype=model_dtype,
                    )
            else:
                batch_np = np.zeros(
                    (self.rec_batch_size, imgC, imgH, target_w),
                    dtype=np.float32,
                )
                for batch_pos, crop_index in enumerate(batch_indices):
                    crop = crops[int(crop_index)]
                    resized_w = resized_widths[int(crop_index)]
                    batch_np[batch_pos] = _resize_norm_rec_crop(
                        crop,
                        img_c=imgC,
                        img_h=imgH,
                        target_w=target_w,
                        resized_w=resized_w,
                    )
                batch = torch.from_numpy(batch_np).to(device=device, dtype=model_dtype)

            if _BUCKET_DEBUG_LOG:
                fills = sorted(
                    resized_widths[int(i)] / float(target_w) for i in batch_indices
                )
                n = len(fills)
                p50 = fills[n // 2]
                p99 = fills[max(0, int(round(0.99 * (n - 1))))]
                truncated_in_batch = sum(
                    1 for i in batch_indices if int(i) in truncated_indices
                )
                logger.info(
                    "rec_bucket batch crops=%d target_w=%d fill_pct min=%.2f p50=%.2f p99=%.2f max=%.2f truncated=%d",
                    n,
                    target_w,
                    fills[0],
                    p50,
                    p99,
                    fills[-1],
                    truncated_in_batch,
                )

            with torch.inference_mode():
                preds = self.system.text_recognizer.net(batch)

            preds = preds[: len(batch_indices)]
            # max() in fp16 yields identical argmax to fp32 (CTC softmax peaks
            # are well-separated). Cast only the (B, T) prob output to fp32,
            # not the (B, T, V) logits — saves a vocab-sized cast kernel.
            preds_prob, preds_idx = preds.max(dim=2)
            batch_results = self.system.text_recognizer.postprocess_op.decode(
                preds_idx.cpu().numpy(),
                preds_prob.float().cpu().numpy(),
                is_remove_duplicate=True,
            )
            for batch_pos, result in enumerate(batch_results):
                rec_res[int(batch_indices[batch_pos])] = result
            if self.use_gpu and next_prepared is not None:
                prepared_batch, prepared_ready = next_prepared

        return rec_res

    def predict(self, image_np: np.ndarray) -> list:
        import torch

        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError(f"Expected RGB image array with shape HxWx3, got {image_np.shape}")
        # Wrap the entire OCR call in inference_mode so workspace allocations,
        # tensor slicing, .contiguous() copies, and model forwards all produce
        # tensors with identical dispatch key sets between warmup (which also
        # runs inside inference_mode) and prod. Without this, the rec
        # compiled graphs captured during warmup don't match prod tensors
        # (extra ADInplaceOrView key in default autograd-on context) and every
        # bucket recompiles ~25s on first real request.
        with torch.inference_mode():
            image_rgb = np.ascontiguousarray(image_np)
            boxes = self._detect_boxes(image_rgb)
            if not boxes:
                return [{"rec_texts": [], "rec_scores": [], "rec_polys": []}]

            crops = [_crop_text_region(image_rgb, box) for box in boxes]
            rec_res = self._fast_recognize(crops)
            return [AutoKernelPPOCRv5Backend._to_paddle_result(boxes, rec_res)]

    def warmup(self) -> None:
        if self.torch_compile_det or self.torch_compile_rec:
            if self.warmup_image_path and self.warmup_image_path.exists():
                self._warmup_with_image(self.warmup_image_path)
            self._warmup_compiled_shapes()
        else:
            self.predict(np.zeros((607, 1080, 3), dtype=np.uint8))

    def _warmup_with_image(self, image_path: Path) -> None:
        from PIL import Image
        import torch

        image = np.asarray(Image.open(image_path).convert("RGB"))
        for _ in range(3):
            self.predict(image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _warmup_compiled_shapes(self) -> None:
        import torch

        # Detector shape parity sweep, only if det compile is enabled. Det
        # is normally NOT compiled in prod; this path stays for future use.
        # Use inference_mode (NOT no_grad) to match the prod request path.
        # The two contexts produce tensors with different dispatch key sets
        # (no_grad keeps ADInplaceOrView, inference_mode strips it). After
        # _RecognizerAdapter.forward calls .contiguous() — which copies and
        # stamps current-context dispatch keys onto the new tensor — a warmup
        # done under no_grad bakes the wrong key set into the compiled graph,
        # forcing a recompile of every bucket on the first real request
        # (~25s/bucket gap visible in TORCH_LOGS=recompiles).
        if self.torch_compile_det:
            detector_shapes = [
                (640, 640), (640, 608), (608, 640), (608, 608),
                (736, 1280), (1280, 736), (1088, 608), (608, 1088),
            ]
            with torch.inference_mode():
                for _ in range(3):
                    for h, w in detector_shapes:
                        self._detect_boxes(np.zeros((h, w, 3), dtype=np.uint8))

        if self.torch_compile_rec:
            # One pass per width bucket. _recognition_batch_plan snaps target_w
            # to the bucket containing the crop's wh_ratio-derived width, and
            # _stage_recognition_batch_cuda always materialises a full
            # rec_batch_size batch — so a single dummy crop per bucket compiles
            # every shape the rec model will ever see in prod.
            for bucket_w in REC_W_BUCKETS:
                dummy = np.zeros(
                    (self._rec_imgH, bucket_w, self._rec_imgC), dtype=np.uint8
                )
                with torch.inference_mode():
                    for _ in range(3):
                        self._fast_recognize([dummy])

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def close(self) -> None:
        try:
            torch = importlib.import_module("torch")
        except Exception:
            return
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


class HybridOCRBackend:
    """PaddleOCR detection + AutoKernel PyTorch recognition.

    Uses PaddleOCR's ``TextDetection`` model for detection (skipping the
    unused recognition pass) and replaces recognition with the
    AutoKernel-optimized PyTorch recognizer using GPU preprocessing,
    batched inference, and GPU-side argmax to minimise CPU↔GPU transfers.
    """

    name = "hybrid"

    def __init__(
        self,
        *,
        paddle_config_path: str,
        autokernel_root: str | Path,
        ppocr_root: str | Path,
        rec_weights_path: str | Path,
        rec_source_path: str | Path | None,
        workspace_path: str | Path,
        use_gpu: bool,
        auto_convert_weights: bool = True,
        optimize_recognizer: bool = True,
        rec_batch_size: int = 1,
        rec_image_shape: str = "3,48,320",
        dtype: str = "float16",
        exclude_kernel_types: Optional[List[str]] = None,
        paddle_ocr_cls: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.autokernel_root = Path(autokernel_root).expanduser().resolve()
        self.ppocr_root = Path(ppocr_root).expanduser().resolve()
        self.rec_weights_path = Path(rec_weights_path).expanduser().resolve()
        self.rec_source_path = (
            Path(rec_source_path).expanduser().resolve()
            if rec_source_path
            else None
        )
        self.workspace_path = Path(workspace_path).expanduser().resolve()
        self.use_gpu = use_gpu
        self.optimize_recognizer = optimize_recognizer
        self.rec_batch_size = rec_batch_size
        self.rec_image_shape = rec_image_shape
        self.dtype_name = dtype
        self.exclude_kernel_types: set[str] = set(exclude_kernel_types or [])
        self._contexts: list[Any] = []

        self._ensure_converted_weights(auto_convert_weights)
        self._validate_paths()

        # PyTorch recognizer with AutoKernel optimizations
        self._configure_environment()
        self.recognizer = self._build_recognizer()

        # Detection-only PaddleOCR — skips the recognition pass entirely.
        # Build this after Torch has initialized CUDA. Some PaddleOCR detector
        # initialization paths can leave Torch unable to discover CUDA if they
        # run first in a fresh process.
        self._paddle_det = self._build_detector(paddle_config_path, paddle_ocr_cls)

        # Parse rec_image_shape once for the fast-path.
        self._rec_imgC, self._rec_imgH, self._rec_imgW = (
            int(v) for v in self.rec_image_shape.split(",")
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _ensure_converted_weights(self, auto_convert: bool) -> None:
        if not auto_convert or self.rec_weights_path.exists():
            return
        ensure_ppocrv5_server_weights(
            PPOCRv5ServerConversionConfig(
                ppocr_root=self.ppocr_root,
                det_source_path=None,
                rec_source_path=self.rec_source_path,
                det_output_path=self.rec_weights_path.parent / "server_det.pth",
                rec_output_path=self.rec_weights_path,
                components=("rec",),
            )
        )

    def _validate_paths(self) -> None:
        required = {
            "AutoKernel root": self.autokernel_root,
            "PaddleOCR2Pytorch root": self.ppocr_root,
            "recognizer weights": self.rec_weights_path,
        }
        for label, path in required.items():
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")
        if self.optimize_recognizer and not self.workspace_path.exists():
            raise FileNotFoundError(
                f"AutoKernel workspace not found: {self.workspace_path}"
            )

    def _configure_environment(self) -> None:
        _prepend_sys_path(self.autokernel_root)
        _prepend_sys_path(self.ppocr_root)
        os.environ["AUTOKERNEL_PPOCR_ROOT"] = str(self.ppocr_root)
        os.environ["AUTOKERNEL_PPOCRV5_SERVER_REC_PTH"] = str(self.rec_weights_path)

    # ------------------------------------------------------------------
    # Recognizer building (mirrors AutoKernelPPOCRv5Backend)
    # ------------------------------------------------------------------

    def _build_recognizer(self) -> Any:
        torch = importlib.import_module("torch")
        if self.use_gpu and not torch.cuda.is_available():
            raise RuntimeError("Hybrid OCR backend requires CUDA when GPU mode is enabled")

        ppocr_model = importlib.import_module("models.ppocrv5_server")
        pytorchocr_utility = importlib.import_module("tools.infer.pytorchocr_utility")
        predict_rec = importlib.import_module("tools.infer.predict_rec")

        parser = pytorchocr_utility.init_args()
        args = parser.parse_args([])
        args.use_gpu = False  # build on CPU, replace net manually
        args.rec_algorithm = "SVTR_PPOCRv5"
        args.rec_yaml_path = str(self.ppocr_root / ppocr_model.REC_YAML_RELATIVE)
        args.rec_model_path = str(self.rec_weights_path)
        args.rec_image_shape = self.rec_image_shape
        args.rec_char_dict_path = str(
            self.ppocr_root / "pytorchocr/utils/dict/ppocrv5_dict.txt"
        )
        args.rec_batch_num = self.rec_batch_size
        args.image_dir = ""

        recognizer = predict_rec.TextRecognizer(args)
        recognizer.use_gpu = True

        rec_wrapper = ppocr_model.PPOCRv5ServerRecModel()
        device = "cuda" if self.use_gpu else "cpu"
        recognizer.net = self._build_recognizer_net(rec_wrapper, torch, device)
        return recognizer

    def _build_recognizer_net(self, rec_wrapper: Any, torch: Any, device: str) -> Any:
        dtype = torch.float32 if device == "cpu" else self._torch_dtype(torch)
        model = rec_wrapper.to(device=device, dtype=dtype).eval()
        context = None
        patched_model = model

        if self.optimize_recognizer:
            if device != "cuda":
                raise RuntimeError("AutoKernel recognizer optimization requires CUDA")
            replacements = self._load_kernel_replacements()
            if not replacements:
                raise RuntimeError(
                    f"No verified AutoKernel recognizer kernels found in {self.workspace_path}"
                )
            verify_mod = self._load_verify_module()
            context = verify_mod.OptimizedModelContext(model, replacements)
            # Force-suppress torch.compile — hybrid uses batched eager mode
            # instead of CUDA graph, and compile adds overhead on variable
            # batch sizes.
            context._suppress_compile = True
            patched_model = context.__enter__()
            self._contexts.append(context)
            logger.info(
                "Hybrid: applied AutoKernel recognizer replacements: %s applied, %s skipped",
                len(getattr(context, "_applied_replacements", [])),
                len(getattr(context, "_skipped_replacements", [])),
            )

        class _RecognizerNetAdapter(torch.nn.Module):
            def __init__(self, runtime_model: Any, optimized_context: Any) -> None:
                super().__init__()
                self.runtime_model = runtime_model
                self.optimized_context = optimized_context

            def forward(self, x: Any) -> Any:
                x = x.to(device=device, dtype=dtype)
                if self.optimized_context is not None:
                    x = self.optimized_context.prepare_input(x)
                return self.runtime_model(x)

        adapter = _RecognizerNetAdapter(patched_model, context).eval()
        logger.info(
            "Hybrid recognizer adapter ready (device=%s, dtype=%s, optimized=%s)",
            device, dtype, self.optimize_recognizer,
        )
        return adapter

    def _torch_dtype(self, torch: Any) -> Any:
        normalized = self.dtype_name.lower()
        if normalized in {"float16", "fp16", "half"}:
            return torch.float16
        if normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if normalized in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"Unsupported AutoKernel OCR dtype: {self.dtype_name}")

    def _load_verify_module(self) -> Any:
        return _import_module_from_path(
            "dgc_ext_autokernel_verify",
            self.autokernel_root / "verify.py",
        )

    def _load_kernel_replacements(self) -> list[Any]:
        specs = load_verified_replacement_specs(self.workspace_path, self.autokernel_root)
        verify_mod = self._load_verify_module()
        support_mod = importlib.import_module("support")

        # Hybrid mode uses batched forward — CUDA graph_capture requires
        # fixed batch_size=1, so always exclude it here.
        skip = self.exclude_kernel_types | {"graph_capture"}

        replacements = []
        for spec in specs:
            if spec.kernel_type in skip:
                logger.info(
                    "Skipping kernel type %s (rank %d)%s",
                    spec.kernel_type,
                    spec.rank,
                    " (incompatible with batched mode)"
                    if spec.kernel_type == "graph_capture"
                    else "",
                )
                continue
            support_stage = support_mod.build_support_stage(
                spec.kernel_type,
                {spec.kernel_type},
            )
            replacements.append(
                verify_mod.KernelReplacement(
                    kernel_type=spec.kernel_type,
                    rank=spec.rank,
                    speedup=spec.speedup,
                    optimized_path=str(spec.optimized_path),
                    reinsert_supported=support_stage["reinsert_supported"],
                    status=spec.status,
                )
            )
        return replacements

    @staticmethod
    def _build_detector(
        paddle_config_path: str,
        paddle_ocr_cls: Optional[Callable[..., Any]],
    ) -> Any:
        """Build a detection-only model, skipping recognition entirely."""
        try:
            from paddleocr import TextDetection

            return TextDetection(model_name="PP-OCRv5_server_det")
        except Exception as exc:
            logger.warning(
                "TextDetection init failed (%s); falling back to full PaddleOCR pipeline",
                exc,
            )
            if paddle_ocr_cls is None:
                from paddleocr import PaddleOCR as paddle_ocr_cls
            try:
                return paddle_ocr_cls(paddlex_config=paddle_config_path)
            except Exception:
                return paddle_ocr_cls(
                    ocr_version="PP-OCRv5",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_hpi=False,
                )

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _detect_boxes(self, image_np: np.ndarray) -> list[np.ndarray]:
        """Run detection-only model and extract polygons."""
        results = self._paddle_det.predict(image_np)
        boxes: list[np.ndarray] = []
        for result in results:
            for poly in result.get("dt_polys", result.get("rec_polys", [])):
                p = np.asarray(poly, dtype=np.float64)
                if p.ndim == 1:
                    p = p.reshape(-1, 2)
                boxes.append(p)
        return _sort_text_boxes(boxes)

    # ------------------------------------------------------------------
    # Fast GPU recognition
    # ------------------------------------------------------------------

    def _fast_recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Paddle-compatible preprocessing with batched GPU recognition.

        Replaces the default ``TextRecognizer.__call__`` which does per-crop
        CPU preprocessing and transfers 80+ MB of logits to CPU.  The fast
        path mirrors Paddle's cv2 resize / normalise, runs a single batched
        forward pass, computes argmax on GPU, and decodes characters on CPU.
        """
        import math
        import torch

        imgC, imgH = self._rec_imgC, self._rec_imgH
        device = "cuda"
        model_dtype = self._torch_dtype(torch)
        limited_max = getattr(self.recognizer, "limited_max_width", 4000)
        limited_min = getattr(self.recognizer, "limited_min_width", 16)

        width_list = [c.shape[1] / float(c.shape[0]) for c in crops]
        indices = np.argsort(np.asarray(width_list))
        batch_num = max(int(self.rec_batch_size), 1)
        rec_res: list[tuple[str, float]] = [("", 0.0)] * len(crops)

        for beg in range(0, len(crops), batch_num):
            end = min(len(crops), beg + batch_num)
            batch_indices = indices[beg:end]

            # Match TextRecognizer.__call__: sort by width and pad each batch
            # only to that batch's widest crop, not the widest crop globally.
            max_wh_ratio = max(width_list[int(i)] for i in batch_indices)
            max_wh_ratio = max(max_wh_ratio, self._rec_imgW / imgH)
            target_w = max(min(int(imgH * max_wh_ratio), limited_max), limited_min)

            batch_np = np.zeros(
                (len(batch_indices), imgC, imgH, target_w),
                dtype=np.float32,
            )

            for batch_pos, crop_index in enumerate(batch_indices):
                crop = crops[int(crop_index)]
                h, w = crop.shape[:2]
                resized_w = max(
                    min(int(math.ceil(imgH * w / float(h))), target_w), limited_min
                )
                batch_np[batch_pos] = _resize_norm_rec_crop(
                    crop,
                    img_c=imgC,
                    img_h=imgH,
                    target_w=target_w,
                    resized_w=resized_w,
                )

            batch = torch.from_numpy(batch_np).to(device=device, dtype=model_dtype)

            with torch.no_grad():
                preds = self.recognizer.net(batch)

            preds_f = preds.float()
            preds_idx = preds_f.argmax(dim=2).cpu().numpy()
            preds_prob = preds_f.max(dim=2).values.cpu().numpy()

            batch_results = self.recognizer.postprocess_op.decode(
                preds_idx, preds_prob, is_remove_duplicate=True
            )
            for batch_pos, result in enumerate(batch_results):
                rec_res[int(batch_indices[batch_pos])] = result

        return rec_res

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, image_np: np.ndarray) -> list:
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError(f"Expected RGB image array with shape HxWx3, got {image_np.shape}")

        # Detect text boxes (detection-only — no wasted recognition pass).
        boxes = self._detect_boxes(image_np)
        if not boxes:
            return [{"rec_texts": [], "rec_scores": [], "rec_polys": []}]

        # Crop text regions. The rec model was trained on RGB (same as Paddle's
        # own pipeline); feeding BGR costs ~3% accuracy on ambiguous chars.
        image_rgb = np.ascontiguousarray(image_np)
        crops = [_crop_text_region(image_rgb, box) for box in boxes]

        # Recognise with GPU-accelerated fast path.
        rec_results = self._fast_recognize(crops)

        # Assemble PaddleOCR-compatible result.
        rec_texts: list[str] = []
        rec_scores: list[float] = []
        rec_polys: list[np.ndarray] = []
        for box, rec in zip(boxes, rec_results):
            if rec is None:
                continue
            rec_texts.append(str(rec[0]))
            rec_scores.append(float(rec[1]))
            rec_polys.append(box)

        return [{"rec_texts": rec_texts, "rec_scores": rec_scores, "rec_polys": rec_polys}]

    def warmup(self) -> None:
        self.predict(np.zeros((540, 856, 3), dtype=np.uint8))

    def close(self) -> None:
        for context in reversed(self._contexts):
            context.__exit__(None, None, None)
        self._contexts.clear()

        paddle_close = getattr(self._paddle_det, "close", None)
        if callable(paddle_close):
            paddle_close()

        try:
            torch = importlib.import_module("torch")
        except Exception:
            return
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_ocr_backend(
    app_settings: Any,
    *,
    use_gpu: bool,
    paddle_ocr_cls: Optional[Callable[..., Any]] = None,
) -> OCRBackend:
    """Create the configured OCR backend without changing the service API."""

    backend_mode = str(getattr(app_settings, "ocr_backend", "auto")).strip().lower()
    if backend_mode not in {"auto", "paddle", "autokernel", "hybrid", "fullpytorch"}:
        raise ValueError(f"Unsupported OCR backend: {backend_mode}")

    config_path = (
        app_settings.ocr_server_config_path
        if use_gpu
        else app_settings.ocr_mobile_config_path
    )

    if backend_mode == "paddle":
        return PaddleOCRBackend(config_path, paddle_ocr_cls=paddle_ocr_cls)
    if not getattr(app_settings, "ocr_autokernel_enabled", False):
        if backend_mode in {"autokernel", "hybrid", "fullpytorch"}:
            raise RuntimeError(
                f"OCR backend '{backend_mode}' requires ocr_autokernel_enabled "
                "(it shares the converted .pth weight-loading infrastructure)"
            )
        return PaddleOCRBackend(config_path, paddle_ocr_cls=paddle_ocr_cls)
    if backend_mode == "auto" and not use_gpu:
        return PaddleOCRBackend(config_path, paddle_ocr_cls=paddle_ocr_cls)

    base_dir = _settings_base_dir(app_settings)
    rec_source = (
        _resolve_path(app_settings.ocr_autokernel_rec_source_path, base_dir)
        if getattr(app_settings, "ocr_autokernel_rec_source_path", "")
        else None
    )
    autokernel_kwargs = {
        "autokernel_root": _resolve_path(app_settings.ocr_autokernel_root, base_dir),
        "ppocr_root": _resolve_path(app_settings.ocr_autokernel_ppocr_root, base_dir),
        "det_weights_path": _resolve_path(
            app_settings.ocr_autokernel_det_weights_path, base_dir
        ),
        "rec_weights_path": _resolve_path(
            app_settings.ocr_autokernel_rec_weights_path, base_dir
        ),
        "det_source_path": _resolve_path(
            app_settings.ocr_autokernel_det_source_path, base_dir
        ) if getattr(app_settings, "ocr_autokernel_det_source_path", "") else None,
        "rec_source_path": rec_source,
        "workspace_path": _resolve_path(
            app_settings.ocr_autokernel_workspace_path, base_dir
        ),
        "use_gpu": use_gpu,
        "auto_convert_weights": app_settings.ocr_autokernel_auto_convert_weights,
        "optimize_recognizer": app_settings.ocr_autokernel_optimize_recognizer,
        "optimize_detector": app_settings.ocr_autokernel_optimize_detector,
        "rec_batch_size": app_settings.ocr_autokernel_rec_batch_size,
        "rec_image_shape": app_settings.ocr_autokernel_rec_image_shape,
        "dtype": app_settings.ocr_autokernel_dtype,
        "exclude_kernel_types": getattr(
            app_settings, "ocr_autokernel_exclude_kernel_types", None
        ),
    }

    if backend_mode == "hybrid":
        hybrid_kwargs = {
            k: v
            for k, v in autokernel_kwargs.items()
            if k not in {"det_weights_path", "det_source_path", "optimize_detector"}
        }
        # Hybrid has no CUDA graph constraint (graph_capture is skipped in
        # HybridOCRBackend._load_kernel_replacements), so use Paddle's default
        # batch_size=6 to match Paddle's ToBatch padding exactly. This closed
        # the last text-parity gap vs Paddle-native in validation (30/30 on
        # good_data_2). AutoKernel stays at the user-configured value (default
        # 1) because its graph_capture kernels are compiled for a fixed batch.
        if hybrid_kwargs.get("rec_batch_size", 1) < 6:
            hybrid_kwargs["rec_batch_size"] = 6
        return HybridOCRBackend(
            paddle_config_path=config_path,
            paddle_ocr_cls=paddle_ocr_cls,
            **hybrid_kwargs,
        )

    if backend_mode == "autokernel":
        return AutoKernelPPOCRv5Backend(**autokernel_kwargs)

    if backend_mode == "fullpytorch":
        fullpytorch_kwargs = {
            k: v
            for k, v in autokernel_kwargs.items()
            if k
            not in {
                "workspace_path",
                "optimize_recognizer",
                "optimize_detector",
                "exclude_kernel_types",
            }
        }
        if fullpytorch_kwargs.get("rec_batch_size", 1) < 8:
            fullpytorch_kwargs["rec_batch_size"] = 8
        fullpytorch_kwargs["rec_bucket_max_width_ratio"] = float(
            getattr(app_settings, "ocr_autokernel_rec_bucket_max_width_ratio", 1.30)
        )
        fullpytorch_kwargs["det_limit_side_len"] = int(
            getattr(app_settings, "ocr_autokernel_det_limit_side_len", 1280)
        )
        fullpytorch_kwargs["det_limit_type"] = str(
            getattr(app_settings, "ocr_autokernel_det_limit_type", "max")
        )
        compile_all = bool(getattr(app_settings, "ocr_autokernel_torch_compile", False))
        fullpytorch_kwargs["torch_compile_det"] = bool(
            getattr(app_settings, "ocr_autokernel_torch_compile_det", compile_all)
        )
        fullpytorch_kwargs["torch_compile_rec"] = bool(
            getattr(app_settings, "ocr_autokernel_torch_compile_rec", compile_all)
        )
        fullpytorch_kwargs["torch_compile_mode"] = str(
            getattr(app_settings, "ocr_autokernel_torch_compile_mode", "default")
        )
        fullpytorch_kwargs["torch_compile_dynamic"] = bool(
            getattr(app_settings, "ocr_autokernel_torch_compile_dynamic", True)
        )
        fullpytorch_kwargs["warmup_image_path"] = getattr(
            app_settings, "ocr_autokernel_warmup_image_path", ""
        )
        fullpytorch_kwargs["det_dtype"] = str(
            getattr(app_settings, "ocr_autokernel_det_dtype", "float32")
        )
        # Default to fp32 for the reference path; user can still override by
        # setting OCR_AUTOKERNEL_DTYPE (shared env var with the other backends).
        return FullPyTorchBackend(**fullpytorch_kwargs)

    if use_gpu:
        try:
            return AutoKernelPPOCRv5Backend(**autokernel_kwargs)
        except Exception as exc:
            logger.warning(
                "AutoKernel OCR backend unavailable; falling back to PaddleOCR: %s",
                exc,
                exc_info=True,
            )

    return PaddleOCRBackend(config_path, paddle_ocr_cls=paddle_ocr_cls)
