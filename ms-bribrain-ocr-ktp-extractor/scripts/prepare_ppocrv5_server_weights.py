#!/usr/bin/env python3
"""Convert PP-OCRv5 server Paddle checkpoints into PyTorch `.pth` files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.core.config import settings  # noqa: E402
from src.services.ppocrv5_conversion import (  # noqa: E402
    PPOCRv5ServerConversionConfig,
    Component,
    convert_ppocrv5_server_weights,
    infer_paddle_source_from_model_dir,
    load_paddlex_config_model_dir,
    resolve_optional_path,
)


def _resolve(path_value: str | None, base_dir: Path) -> Path | None:
    return resolve_optional_path(path_value, base_dir)


def _configured_source(
    *,
    explicit: str | None,
    env_name: str,
    configured: str,
    paddlex_config_path: Path,
    paddlex_submodule: str,
    base_dir: Path,
) -> Path | None:
    if explicit:
        return _resolve(explicit, base_dir)

    env_value = os.getenv(env_name)
    if env_value:
        return _resolve(env_value, base_dir)

    configured_path = _resolve(configured, base_dir)
    if configured_path is not None:
        return configured_path

    model_dir = load_paddlex_config_model_dir(paddlex_config_path, paddlex_submodule)
    return infer_paddle_source_from_model_dir(model_dir, paddlex_config_path.parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        choices=("both", "det", "rec"),
        default="both",
        help="Which converted weights to produce.",
    )
    parser.add_argument("--ppocr-root", default=None)
    parser.add_argument("--det-src", default=None)
    parser.add_argument("--rec-src", default=None)
    parser.add_argument("--det-dst", default=None)
    parser.add_argument("--rec-dst", default=None)
    parser.add_argument("--paddlex-config", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PPOCRv5ServerConversionConfig:
    base_dir = settings.config_path.resolve().parent
    paddlex_config_path = _resolve(
        args.paddlex_config or settings.ocr_server_config_path,
        base_dir,
    )
    if paddlex_config_path is None:
        paddlex_config_path = base_dir / "PaddleOCR_server.yaml"

    components: tuple[Component, ...]
    if args.component == "both":
        components = ("det", "rec")
    else:
        components = (args.component,)

    det_source = _configured_source(
        explicit=args.det_src,
        env_name="AUTOKERNEL_PPOCRV5_SERVER_DET_PDPARAMS",
        configured=settings.ocr_autokernel_det_source_path,
        paddlex_config_path=paddlex_config_path,
        paddlex_submodule="TextDetection",
        base_dir=base_dir,
    )
    rec_source = _configured_source(
        explicit=args.rec_src,
        env_name="AUTOKERNEL_PPOCRV5_SERVER_REC_PDPARAMS",
        configured=settings.ocr_autokernel_rec_source_path,
        paddlex_config_path=paddlex_config_path,
        paddlex_submodule="TextRecognition",
        base_dir=base_dir,
    )

    ppocr_root = _resolve(args.ppocr_root or settings.ocr_autokernel_ppocr_root, base_dir)
    det_output = _resolve(args.det_dst or settings.ocr_autokernel_det_weights_path, base_dir)
    rec_output = _resolve(args.rec_dst or settings.ocr_autokernel_rec_weights_path, base_dir)
    if ppocr_root is None or det_output is None or rec_output is None:
        raise ValueError("PP-OCR root and output paths must be configured")

    return PPOCRv5ServerConversionConfig(
        ppocr_root=ppocr_root,
        det_source_path=det_source,
        rec_source_path=rec_source,
        det_output_path=det_output,
        rec_output_path=rec_output,
        components=components,
        force=args.force,
    )


def main() -> int:
    args = parse_args()
    try:
        config = build_config(args)
        outputs = convert_ppocrv5_server_weights(config)
    except Exception as exc:
        print(f"PP-OCRv5 server weight conversion failed: {exc}", file=sys.stderr)
        return 2

    result = {
        "components": list(config.components),
        "ppocr_root": str(config.ppocr_root),
        "det_source_path": str(config.det_source_path) if config.det_source_path else None,
        "rec_source_path": str(config.rec_source_path) if config.rec_source_path else None,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.json_output:
        Path(args.json_output).write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
