#!/usr/bin/env python3
"""Inspect PIR JSON 'p' op attributes to see whether semantic parameter names
are preserved (e.g. 'backbone.stem.conv.weight') or lost to generic form
(e.g. 'conv2d_0.w_0').
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIR_JSON = PROJECT_ROOT / "src/models/server_models/ppocrv5_server_det_source/inference.json"


def main() -> int:
    data = json.loads(PIR_JSON.read_text(encoding="utf-8"))
    ops = data["program"]["regions"][0]["blocks"][0]["ops"]
    p_ops = [op for op in ops if op.get("#") == "p"]
    print(f"Total ops: {len(ops)}")
    print(f"Parameter ops: {len(p_ops)}")
    print()
    print("First 25 parameter op attribute lists (attrs = op['A']):")
    for i, op in enumerate(p_ops[:25]):
        print(f"  [{i}] {op.get('A')}")
    print()
    print("Last 10 parameter ops:")
    for i, op in enumerate(p_ops[-10:]):
        print(f"  [{len(p_ops) - 10 + i}] {op.get('A')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
