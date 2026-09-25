import copy
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from ocr_common.web.app import API_CONVENTIONS

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "api" / "gateway.openapi.yaml"
# The orchestrator owns every operation; the stages are listed for the callback webhook they send.
SERVICES = ("orchestrator", "ekstraksi", "structuring", "scoring")

OPERATIONS: list[tuple[str, str, str, str]] = [
    ("orchestrator", "post", "/v1/extract-ocr", "1. Start the pipeline"),
    ("orchestrator", "get", "/v1/extract-ocr/{request_id}", "3. Status"),
]
CALLBACK_TAG = "2. Callbacks (you implement this)"

TAGS = [
    {
        "name": "1. Start the pipeline",
        "description": (
            "The call you make: guardrails check, then OCR -> structuring -> scoring, waited on for up to "
            "PIPELINE_WAIT_SECONDS; answers in the central orchestrator's extract-ocr contract. 200: `completed` "
            "with the fields. 202: still `processing`, the result arrives by callback. 400: rejected. 422: a stage "
            "failed."
        ),
    },
    {
        "name": CALLBACK_TAG,
        "description": "Listed under **Webhooks**: the request each stage sends to you when it finishes.",
    },
    {
        "name": "3. Status",
        "description": (
            "Where a request is now, in the same contract, without waiting: e.g. after a 202 whose callback did "
            "not arrive."
        ),
    },
]

DESCRIPTION = """
Everything the central orchestrator / gateway needs to integrate the NPWP OCR pipeline, merged from the
services' own specs. Each service also serves its full Swagger UI at `/docs`.

## The flow

1. **`POST /v1/extract-ocr`** on the *orchestrator* service (port `8034`) with your `request_id` and the
   document (`file` or `file_url`, optional `params`). It answers in the central orchestrator's
   `extract-ocr` contract (`job_status`, `data`, `guardrails`, `params`):
   - rejected by the guardrails model -> **400**, `errors: DOWNSTREAM_VALIDATION_ERROR`, `guardrails: 1`;
     nothing runs and no callback follows.
   - passed -> the document goes on to the OCR stage and the service waits for the pipeline for up to
     `PIPELINE_WAIT_SECONDS` (15 s by default, counted from the request's arrival). Finished in time ->
     **200**, `job_status: completed`, `data` = `nomor_npwp` and `nama` as `{value, confidence}` with
     confidence 0/1; a stage failed -> **422** `OCR_FAILED` / `STRUCTURING_FAILED` / `SCORING_FAILED`; still
     running -> **202**, `job_status: processing`. Give this call an HTTP timeout well above the wait.
2. **Receive callbacks** (see *Webhooks*), sent by the pipeline stages themselves after a hand-off. The
   `SCORING` / `DONE` callback carries the **final result** (richer than `data`: OCR scores, trust
   probabilities, the guardrails report). A `FAILED` callback of any stage ends the request.
3. **Read the status when needed** with `GET /v1/extract-ocr/{request_id}` on the same service: the same
   contract, without waiting.

Time a request out on your side: a stage that crashes mid-job cannot send its callback.

## What the result is, and is not

`fields` holds `nomor_npwp`, `nama`, `nama_badan`. `scoring` holds `npwp_confidence` and
`name_confidence`: the probability, from the ML team's trust model, that each extracted field is
correct. There is **no document-level score and no approve / reject decision**; thresholds are yours.

## Addresses

The orchestrator is the only service you call. From another namespace in GKE:
`http://nilam-ocr-npwp.nilam-ocr-npwp.svc.cluster.local:8034` (the release's entry Service). The
guardrails and pipeline stage services are internal. Nothing is exposed outside the cluster.
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
    shared_description: str | None = None

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

    if shared_description is None:
        raise SystemExit("no service spec publishes the stageCallback webhook: regenerate the service specs first")
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
