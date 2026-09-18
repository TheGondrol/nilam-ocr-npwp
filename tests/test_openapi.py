"""openapi.yaml adalah turunan kode; kalau berbeda, jalankan `make openapi`."""

from pathlib import Path

import yaml
from scripts.export_openapi import spec_text

ROOT = Path(__file__).resolve().parents[1]


def test_openapi_yaml_is_up_to_date():
    disk = yaml.safe_load((ROOT / "openapi.yaml").read_text(encoding="utf-8"))
    live = yaml.safe_load(spec_text())
    assert live == disk, "openapi.yaml ketinggalan dari kode; jalankan `make openapi`"


def test_extract_ocr_form_has_file_and_file_url():
    spec = yaml.safe_load(spec_text())
    operation = next(iter(spec["paths"]["/v1/extract-ocr"].values()))
    form = next(iter(operation["requestBody"]["content"].values()))
    fields = set(spec["components"]["schemas"][form["schema"]["$ref"].split("/")[-1]]["properties"])
    assert fields == {"request_id", "file", "file_url"}


def test_error_responses_have_their_own_examples():
    spec = yaml.safe_load(spec_text())
    for path, methods in spec["paths"].items():
        if path == "/health":
            continue
        for operation in methods.values():
            for code, response in operation["responses"].items():
                if code.startswith(("4", "5")):
                    example = response.get("content", {}).get("application/json", {}).get("example")
                    assert example, f"{path} {code}: tidak punya contoh sendiri"
                    assert str(example["status_code"]) == code
