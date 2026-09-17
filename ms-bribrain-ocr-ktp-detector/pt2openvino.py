"""
PyTorch to OpenVINO Model Conversion Script
Converts YOLO .pt model to OpenVINO format for optimized CPU inference.

Usage:
    python pt2openvino.py
    python pt2openvino.py --model ./src/models/best.pt --half --int8
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def convert_to_openvino(
    model_path: str = "./src/models/best.pt",
    half: bool = True,
    int8: bool = False,
    dynamic: bool = False,
    simplify: bool = True,
):
    """
    Convert YOLO PyTorch model to OpenVINO format.

    Args:
        model_path: Path to the source .pt model
        half: Enable FP16 precision (recommended for CPU speedup)
        int8: Enable INT8 quantization (maximum speed, slight accuracy trade-off)
        dynamic: Enable dynamic input shapes
        simplify: Simplify the model graph

    Returns:
        Path to the exported OpenVINO model directory
    """
    model_file = Path(model_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    print(f"Loading model from: {model_file}")
    model = YOLO(str(model_file))

    print("Exporting to OpenVINO format...")
    print(f"  - Half precision (FP16): {half}")
    print(f"  - INT8 quantization: {int8}")
    print(f"  - Dynamic shapes: {dynamic}")
    print(f"  - Simplify: {simplify}")

    # Export to OpenVINO
    export_path = model.export(
        format="openvino",
        half=half,
        int8=int8,
        dynamic=dynamic,
        simplify=simplify,
    )

    print("\n✓ Export complete!")
    print(f"  OpenVINO model saved to: {export_path}")
    print("\nTo use the OpenVINO model, update config.yaml:")
    print("  model:")
    print("    export_format: openvino")

    return export_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO PyTorch model to OpenVINO format"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./src/models/best.pt",
        help="Path to source YOLO model (.pt file)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="Enable FP16 half precision (default: True)",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Disable FP16 half precision",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Enable INT8 quantization for maximum CPU speed",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic input shapes",
    )

    args = parser.parse_args()

    half = not args.no_half

    try:
        convert_to_openvino(
            model_path=args.model,
            half=half,
            int8=args.int8,
            dynamic=args.dynamic,
        )
    except Exception as e:
        print(f"✗ Export failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
