"""Tulis ulang openapi.yaml dari aplikasinya sendiri.

    make openapi

Spec yang ditulis tangan selalu kalah cepat dari kodenya. Karena itu file ini
turunan, bukan sumber: yang diedit adalah decorator route, dan `make openapi`
yang memindahkannya. tests/test_openapi.py menjaga keduanya tidak berpisah.
Format dump sama dengan scripts/check_openapi.py di nilam-ocr-orchestration.
"""

from pathlib import Path

import yaml
from src.main import app

TARGET = Path(__file__).resolve().parents[1] / "openapi.yaml"


def spec_text() -> str:
    return yaml.safe_dump(app.openapi(), sort_keys=False, allow_unicode=True, width=100)


if __name__ == "__main__":
    TARGET.write_text(spec_text(), encoding="utf-8")
    print(f"ditulis: {TARGET}")
