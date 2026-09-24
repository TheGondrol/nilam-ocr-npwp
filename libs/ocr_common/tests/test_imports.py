"""The orchestrator and guardrails run without SQLAlchemy (no `db` extra): everything a service imports from
ocr_common must load without it. Each case runs in a fresh interpreter where importing sqlalchemy raises."""

import subprocess
import sys

import pytest

BLOCK_SQLALCHEMY = """
import importlib.abc, sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "sqlalchemy" or name.startswith("sqlalchemy."):
            raise ModuleNotFoundError("sqlalchemy is not installed in this service")
        return None

sys.meta_path.insert(0, Block())
"""

MODULES = [
    "ocr_common.pipeline",
    "ocr_common.pipeline.factory",
    "ocr_common.web.app",
    "ocr_common.web.intake",
    "ocr_common.clients.remote",
    "ocr_common.npwp",
    "ocr_common.types",
    "ocr_common.simulation",
    "ocr_common.testing_endpoints",
    "ocr_common.web.testing_routes",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_without_sqlalchemy(module):
    code = BLOCK_SQLALCHEMY + f"\nimport {module}\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_sql_modules_are_the_only_ones_that_need_it():
    code = BLOCK_SQLALCHEMY + "\nimport ocr_common.pipeline.results_sql\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode != 0 and "sqlalchemy" in result.stderr
