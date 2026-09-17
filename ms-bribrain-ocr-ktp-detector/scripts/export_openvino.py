"""
OpenVINO Model Export Script
Exports the YOLO model to OpenVINO format for optimized CPU inference.

Usage:
    python scripts/export_openvino.py [--half] [--int8]

Options:
    --half  Enable FP16 half precision (recommended for speed)
    --int8  Enable INT8 quantization (maximum speed, may reduce accuracy slightly)
"""

import argparse
import sys
from pathlib import Path


from ultralytics import YOLO


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def export_to_openvino(
    model_path: str = "./src/models/best.pt",
    output_dir: str = "./src/models",
    half: bool = True,
    int8: bool = False,
):
    """
    Export YOLO model to OpenVINO format.

    Args:
        model_path: Path to the source .pt model
        output_dir: Directory to save the exported model
        half: Enable FP16 precision
        int8: Enable INT8 quantization
    """
    model_file = Path(model_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    print(f"Loading model from: {model_file}")
    model = YOLO(str(model_file))

    print("Exporting to OpenVINO format...")
    print(f"  - Half precision (FP16): {half}")
    print(f"  - INT8 quantization: {int8}")

    # Export to OpenVINO
    export_path = model.export(
        format="openvino",
        half=half,
        int8=int8,
        dynamic=False,  # Static shapes for better optimization
        simplify=True,
    )

    print("\nExport complete!")
    print(f"OpenVINO model saved to: {export_path}")
    print("\nTo use the OpenVINO model, update config.yaml:")
    print("  model:")
    print(f"    path: {export_path}")
    print("    export_format: openvino")

    return export_path


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLO model to OpenVINO format for CPU optimization"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./src/models/best.pt",
        help="Path to source YOLO model (.pt file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./src/models",
        help="Output directory for exported model",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="Enable FP16 half precision (default: True)",
    )
    parser.add_argument(
        "--no-half", action="store_true", help="Disable FP16 half precision"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Enable INT8 quantization for maximum CPU speed",
    )

    args = parser.parse_args()

    half = not args.no_half

    try:
        export_to_openvino(
            model_path=args.model, output_dir=args.output, half=half, int8=args.int8
        )
    except Exception as e:
        print(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
