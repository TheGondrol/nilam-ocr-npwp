import importlib
import sys
from pathlib import Path

import yaml
from fastapi import FastAPI


def spec_text(app: FastAPI) -> str:
    return yaml.safe_dump(app.openapi(), sort_keys=False, allow_unicode=True, width=100)


def main(module: str = "src.main", target: str = "openapi.yaml") -> None:
    sys.path.insert(0, str(Path.cwd()))
    app = importlib.import_module(module).app
    path = Path.cwd() / target
    path.write_text(spec_text(app), encoding="utf-8")
    print(f"ditulis: {path}")


if __name__ == "__main__":
    main()
