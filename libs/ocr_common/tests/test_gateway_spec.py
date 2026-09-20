"""api/gateway.openapi.yaml adalah turunan openapi.yaml keempat service; kalau berbeda, jalankan `make openapi`."""

import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def _builder():
    script = ROOT / "scripts" / "build_gateway_openapi.py"
    spec = importlib.util.spec_from_file_location("build_gateway_openapi", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gateway_spec_is_up_to_date():
    builder = _builder()
    assert yaml.safe_load(builder.TARGET.read_text(encoding="utf-8")) == builder.build()


def test_gateway_spec_has_no_dangling_refs_and_documents_the_callback():
    built = _builder().build()
    text = yaml.safe_dump(built)
    names = set(_builder().REF.findall(text))
    assert names <= set(built["components"]["schemas"])
    bodies = built["webhooks"]["stageCallback"]["post"]["requestBody"]["content"]["application/json"]["schema"]["oneOf"]
    assert {b["$ref"].rsplit("/", 1)[-1] for b in bodies} == {"StageCallback", "ScoringStageCallback"}


def test_every_gateway_operation_names_the_service_that_owns_it():
    for path, item in _builder().build()["paths"].items():
        for method, operation in item.items():
            assert operation["servers"], f"{method.upper()} {path} has no servers"
            assert operation["operationId"], f"{method.upper()} {path} has no operationId"
