"""Downloads the model files into the services' weights/ (`make weights`), for running them on a laptop or
building an image with the model baked in. A deployed service does not need this: with *_MODEL_GCS_URI set it
downloads its model itself at start.

    GUARDRAILS_MODEL_GCS_URI=gs://... GUARDRAILS_MODEL_SHA256=... python scripts/fetch_weights.py [guardrails|scoring]

gs:// URIs use GCP Workload Identity Federation with Entra ID: the AZURE_* / GCP_* variables of --env-file
(default wif.gcs.env, never committed) or of the environment. https:// and s3:// (MinIO `mc`) still work.
Exit code 2 when a model has no URI set (its file is left as it is).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libs" / "ocr_common"))

MODELS = {
    "guardrails": {
        "uri_env": "GUARDRAILS_MODEL_GCS_URI",
        "sha_env": "GUARDRAILS_MODEL_SHA256",
        "target": ROOT / "services" / "guardrails" / "weights" / "best_model.pt",
    },
    "scoring": {
        "uri_env": "SCORING_MODEL_GCS_URI",
        "sha_env": "SCORING_MODEL_SHA256",
        "target": ROOT / "services" / "scoring" / "weights" / "trust_model.joblib",
    },
}


def _download(uri: str, target: Path, sha256: str | None, env_file: Path) -> None:
    if uri.startswith("gs://"):
        from ocr_common.clients import gcs
        from ocr_common.clients.gcp import credentials_from_env

        credentials, _ = credentials_from_env(env_file)
        gcs.download(uri, target, credentials, sha256=sha256)  # verified against MD5 and SHA-256
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        tmp = Path(handle.name)
    try:
        if uri.startswith("s3://"):
            subprocess.run(["mc", "cp", uri[len("s3://") :], str(tmp)], check=True)
        elif uri.startswith(("http://", "https://")):
            with urllib.request.urlopen(uri, timeout=120) as response, open(tmp, "wb") as out:
                shutil.copyfileobj(response, out)
        else:
            raise SystemExit(f"skema URI tidak dikenal: {uri!r} (pakai gs://, s3://, atau https://)")
        if sha256:
            from ocr_common.clients.gcs import file_digests

            actual = file_digests(tmp)[0]
            if actual != sha256.lower():
                raise SystemExit(f"sha256 tidak cocok (dapat {actual}, harap {sha256})")
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def fetch(name: str, *, force: bool, env_file: Path) -> int:
    spec = MODELS[name]
    uri = os.environ.get(spec["uri_env"])
    sha256 = os.environ.get(spec["sha_env"])
    target: Path = spec["target"]
    if not uri:
        state = "sudah ada" if target.is_file() else "TIDAK ADA"
        print(f"{name}: {spec['uri_env']} tidak di-set; lewati unduhan ({target} {state})")
        return 2
    if target.is_file() and not force and not sha256:
        print(f"{name}: {target} sudah ada, lewati (pakai --force untuk unduh ulang)")
        return 0
    if force:
        target.unlink(missing_ok=True)
    print(f"{name}: {uri} -> {target}")
    _download(uri, target, sha256, env_file)
    print(f"{name}: siap ({target.stat().st_size // 1024} KB)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", default=list(MODELS), help="nama model (default: semua)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--env-file", type=Path, default=ROOT / "wif.gcs.env")
    args = parser.parse_args()
    return max(fetch(name, force=args.force, env_file=args.env_file) for name in args.models)


if __name__ == "__main__":
    sys.exit(main())
