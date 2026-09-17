#!/usr/bin/env python3
"""Force re-conversion of the rec PIR source to PyTorch using the updated matcher."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
os.chdir(PROJECT_ROOT)
for _p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.services.ppocrv5_conversion import _convert_recognizer, ensure_ppocr_root  # noqa: E402


SOURCE = PROJECT_ROOT / "src/models/server_models/ppocrv5_server_rec_source/inference.pdiparams"
DEST = REPO_ROOT / "autokernel/workspace/ppocrv5/server_rec.pth"
PPOCR_ROOT = PROJECT_ROOT / "PaddleOCR2Pytorch"


def main() -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ppocr_root = ensure_ppocr_root(PPOCR_ROOT)
    print(f"source: {SOURCE}")
    print(f"dest:   {DEST}")
    print(f"ppocr_root: {ppocr_root}")

    _convert_recognizer(ppocr_root, SOURCE, DEST)
    print(f"wrote {DEST} ({DEST.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
