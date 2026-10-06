"""Downloads the model files from GCS into the services' weights/, where `docker build` bakes them into the image.
deploy/helm/deploy.sh runs it before every guardrails / scoring build, so an image always carries the model that was
in GCS when it was built (`make weights` does the same by hand).

    python scripts/fetch_weights.py [guardrails] [scoring]

Source: GUARDRAILS_MODEL_GCS_URI / SCORING_MODEL_GCS_URI, by default the fixed paths in
gs://gc-bribrain-dev-gcs-ocr-nilam-01/nilam-ocr-npwp/. A file already there with the object's SHA-256 is kept; any
other is replaced. The download is checked against GCS's MD5 and the SHA-256 recorded at upload (or
GUARDRAILS_/SCORING_MODEL_SHA256 when set).

Credentials (ocr_common.clients.gcp.google_credentials): the AZURE_* / GCP_* variables of --env-file (default
wif.gcs.env, never committed) or of the environment; without them Application Default Credentials (a Compute
Engine VM's or CI runner's service account, `gcloud auth application-default login`).

Prints `<name> sha256 <hex>` per model; exits 1 when a model cannot be fetched.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libs" / "ocr_common"))

from ocr_common.clients import gcs  # noqa: E402
from ocr_common.clients.gcp import credentials_from_env  # noqa: E402

BUCKET = "gs://gc-bribrain-dev-gcs-ocr-nilam-01/nilam-ocr-npwp"
MODELS = {
    "guardrails": {
        "uri_env": "GUARDRAILS_MODEL_GCS_URI",
        "sha_env": "GUARDRAILS_MODEL_SHA256",
        "default_uri": f"{BUCKET}/guardrails/best_model.pt",
        "target": ROOT / "services" / "guardrails" / "weights" / "best_model.pt",
    },
    "scoring": {
        "uri_env": "SCORING_MODEL_GCS_URI",
        "sha_env": "SCORING_MODEL_SHA256",
        "default_uri": f"{BUCKET}/scoring/trust_model.joblib",
        "target": ROOT / "services" / "scoring" / "weights" / "trust_model.joblib",
    },
}


def fetch(name: str, credentials, *, force: bool) -> str:
    """Brings `name`'s file up to date with GCS; returns its SHA-256."""
    spec = MODELS[name]
    uri = os.environ.get(spec["uri_env"]) or spec["default_uri"]
    target: Path = spec["target"]
    if force:
        target.unlink(missing_ok=True)
    print(f"{name}: {uri} -> {target.relative_to(ROOT)}", file=sys.stderr)
    gcs.download(uri, target, credentials, sha256=os.environ.get(spec["sha_env"]) or None)
    return gcs.file_digests(target)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", default=list(MODELS), help=f"{' / '.join(MODELS)} (default: both)")
    parser.add_argument("--force", action="store_true", help="download even when the local file is already current")
    parser.add_argument("--env-file", type=Path, default=ROOT / "wif.gcs.env")
    args = parser.parse_args()
    if unknown := [name for name in args.models if name not in MODELS]:
        parser.error(f"unknown model {unknown[0]!r}: {', '.join(MODELS)}")

    credentials, who = credentials_from_env(args.env_file)
    print(f"GCS as {who}", file=sys.stderr)
    for name in args.models:
        try:
            sha256 = fetch(name, credentials, force=args.force)
        except (gcs.GcsError, ValueError, OSError) as exc:
            print(f"{name}: {exc}", file=sys.stderr)
            return 1
        print(f"{name} sha256 {sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
