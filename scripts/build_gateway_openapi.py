import copy
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from ocr_common.app import API_CONVENTIONS

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "api" / "gateway.openapi.yaml"
SERVICES = ("guardrails", "ekstraksi", "structuring", "scoring")

OPERATIONS: list[tuple[str, str, str, str]] = [
    ("guardrails", "post", "/v1/extract-ocr", "1. Start the pipeline"),
    ("ekstraksi", "get", "/v1/ekstraksi/jobs/{request_id}", "3. Reconciliation"),
    ("structuring", "get", "/v1/structuring/jobs/{request_id}", "3. Reconciliation"),
    ("scoring", "get", "/v1/scoring/jobs/{request_id}", "3. Reconciliation"),
    ("ekstraksi", "post", "/v1/generate-request-id", "Legacy synchronous contract (deprecated)"),
    ("ekstraksi", "get", "/v1/get-ocr-result/{request_id}", "Legacy synchronous contract (deprecated)"),
]
CALLBACK_TAG = "2. Callbacks (you implement this)"

TAGS = [
    {
        "name": "1. Start the pipeline",
        "description": (
            "The only call you make: guardrails check, then OCR -> structuring -> scoring. "
            "Always 200: rejected -> the reason, nothing runs; accepted -> `job`, the rest arrives by callback."
        ),
    },
    {
        "name": CALLBACK_TAG,
        "description": "Listed under **Webhooks**: the request each stage sends to you when it finishes.",
    },
    {
        "name": "3. Reconciliation",
        "description": "Read a stage's status and result directly, e.g. after a missed callback or a timeout.",
    },
    {
        "name": "Legacy synchronous contract (deprecated)",
        "description": (
            "The old synchronous flow on the ekstraksi service. Its `POST /v1/extract-ocr` (port 8030) is left out "
            "here: that path now belongs to the guardrails service above."
        ),
    },
]

DESCRIPTION = """
Everything the gateway / orchestrator needs to integrate the NPWP OCR pipeline, merged from the four
services' own specs. Each service also serves its full Swagger UI at `/docs`.

## The flow

1. **`POST /v1/extract-ocr`** on the *guardrails* service (port `8031`) with your `request_id` and the document
   (`file` or `file_url`). This is the only call you make. The guardrails check runs synchronously and
   always answers **200**:
   - `data.passed: false`, `data.job: null` -> nothing runs and no callback follows; answer the client
     with `data.reason`.
   - `data.passed: true` -> the guardrails service has already handed the document to the OCR stage;
     `data.job` is that stage's answer. You are done calling.
2. **Receive callbacks** (see *Webhooks*): `OCR` -> `STRUCTURING` -> `SCORING`, each `DONE` or
   `FAILED`. The `SCORING` / `DONE` callback carries the **final result**. A `FAILED` callback of any
   stage ends the request.
3. **Reconcile when needed** with `GET /v1/<stage>/jobs/{request_id}` on the stage's own service.

Time a request out on your side: a stage that crashes mid-job cannot send its callback.

## What the result is, and is not

`fields` holds `nomor_npwp`, `nama`, `nama_badan`. `scoring` holds `npwp_confidence` and
`name_confidence`: the probability, from the ML team's trust model, that each extracted field is
correct. There is **no document-level score and no approve / reject decision**; thresholds are yours.

## Addresses

Each operation lists the servers of the service that owns it. Inside GKE the four services share one
Service; from another namespace: `http://nilam-ocr-npwp.nilam-ocr-npwp.svc.cluster.local:<port>` with
guardrails `8031`, ekstraksi `8030`, structuring `8032`, scoring `8033`. Nothing is exposed outside
the cluster.
"""

REF = re.compile(r"#/components/schemas/([A-Za-z0-9_]+)")


def _load(service: str) -> dict[str, Any]:
    return yaml.safe_load((ROOT / "services" / service / "openapi.yaml").read_text(encoding="utf-8"))


def _refs(node: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                found.update(REF.findall(value))
            else:
                found |= _refs(value)
    elif isinstance(node, list):
        for item in node:
            found |= _refs(item)
    return found


def _rewrite(node: Any, rename: dict[str, str]) -> Any:
    if isinstance(node, dict):
        return {
            key: (
                REF.sub(lambda m: f"#/components/schemas/{rename.get(m.group(1), m.group(1))}", value)
                if key == "$ref" and isinstance(value, str)
                else _rewrite(value, rename)
            )
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_rewrite(item, rename) for item in node]
    return node


def _closure(names: set[str], schemas: dict[str, Any]) -> set[str]:
    pending, seen = list(names), set()
    while pending:
        name = pending.pop()
        if name in seen or name not in schemas:
            continue
        seen.add(name)
        pending.extend(_refs(schemas[name]))
    return seen


def build() -> dict[str, Any]:
    specs = {service: _load(service) for service in SERVICES}
    merged_schemas: dict[str, Any] = {}
    paths: dict[str, Any] = {}
    sent_when: list[str] = []
    callback_bodies: list[str] = []

    for service in SERVICES:
        spec = specs[service]
        schemas = spec["components"]["schemas"]
        wanted = [(m, p, tag) for s, m, p, tag in OPERATIONS if s == service]
        nodes: list[Any] = [spec["paths"][p][m] for m, p, _ in wanted]
        webhook = (spec.get("webhooks") or {}).get("stageCallback", {}).get("post")
        if webhook:
            nodes.append(webhook)
        needed = _closure(set().union(*(_refs(node) for node in nodes)) if nodes else set(), schemas)

        rename = {
            name: f"{service.capitalize()}{name}"
            for name in needed
            if name in merged_schemas and merged_schemas[name] != _rewrite(schemas[name], {})
        }
        for name in needed:
            merged_schemas[rename.get(name, name)] = _rewrite(copy.deepcopy(schemas[name]), rename)

        servers = [server for server in spec.get("servers", []) if server["url"] != "/"]
        for method, path, tag in wanted:
            operation = _rewrite(copy.deepcopy(spec["paths"][path][method]), rename)
            operation["tags"] = [tag]
            operation["servers"] = servers
            paths.setdefault(path, {})[method] = operation

        if webhook:
            description = webhook["description"]
            when = re.search(r"\*\*When\.\*\* (.*?)\n\n", description, re.S)
            sent_when.append(f"- **{service}**: {when.group(1) if when else ''}")
            body = _rewrite(webhook["requestBody"]["content"]["application/json"]["schema"], rename)
            callback_bodies.append(body["$ref"])
            shared_description = description

    webhook_template = copy.deepcopy(specs["ekstraksi"]["webhooks"]["stageCallback"]["post"])
    webhook_template["tags"] = [CALLBACK_TAG]
    webhook_template["description"] = re.sub(
        r"\*\*When\.\*\* .*?\n\n",
        "**When.** Once per stage, from the service that ran it:\n\n" + "\n".join(sent_when) + "\n\n",
        shared_description,
        flags=re.S,
    )
    unique_bodies = list(dict.fromkeys(callback_bodies))
    webhook_template["requestBody"]["content"]["application/json"]["schema"] = {
        "oneOf": [{"$ref": ref} for ref in unique_bodies],
        "description": "`stage: SCORING` carries the final result in `result`; for the other stages `result` is null.",
    }

    return {
        "openapi": "3.1.0",
        "info": {
            "title": "NILAM OCR NPWP: gateway integration API",
            "version": specs["ekstraksi"]["info"]["version"],
            "description": DESCRIPTION.strip() + "\n" + API_CONVENTIONS,
        },
        "tags": TAGS,
        "paths": paths,
        "webhooks": {"stageCallback": {"post": webhook_template}},
        "components": {
            "schemas": dict(sorted(merged_schemas.items())),
            "securitySchemes": specs["ekstraksi"]["components"]["securitySchemes"],
        },
    }


def main() -> int:
    text = yaml.safe_dump(build(), sort_keys=False, allow_unicode=True, width=100)
    if "--check" in sys.argv:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if yaml.safe_load(current or "{}") != yaml.safe_load(text):
            print(f"{TARGET.relative_to(ROOT)} ketinggalan dari spec service; jalankan `make openapi`")
            return 1
        print(f"{TARGET.relative_to(ROOT)} sesuai")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(text, encoding="utf-8")
    print(f"ditulis: {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
