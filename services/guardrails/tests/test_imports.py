"""This service has no database and no SQLAlchemy: its app must import without it. The 0.2.0 deploy of
2026-09-23 crashed at startup because a shared module pulled SQLAlchemy in; this keeps that from
coming back."""

import os
import subprocess
import sys
from pathlib import Path

from ocr_common.testing import TEST_API_KEY

SERVICE = Path(__file__).resolve().parents[1]

CODE = """
import importlib.abc, sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "sqlalchemy" or name.startswith("sqlalchemy."):
            raise ModuleNotFoundError("sqlalchemy is not installed in this service")
        return None

sys.meta_path.insert(0, Block())
import app.main
"""


def test_the_app_imports_without_sqlalchemy():
    env = {
        **os.environ,
        "API_KEY": TEST_API_KEY,
        "ENVIRONMENT": "local",
        "GUARDRAILS_BACKEND": "mock",
        "PYTHONPATH": str(SERVICE),
    }
    result = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, cwd=SERVICE, env=env)
    assert result.returncode == 0, result.stderr
