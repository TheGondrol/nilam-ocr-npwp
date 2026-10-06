"""Uploads a model file to GCS for a service to download where it starts (GUARDRAILS_MODEL_GCS_URI,
SCORING_MODEL_GCS_URI), then downloads it back and checks it.

    python scripts/upload_model.py services/guardrails/weights/best_model.pt \\
        gs://gc-bribrain-dev-gcs-ocr-nilam-01/nilam-ocr-npwp/guardrails/best_model.pt

Credentials: GCP Workload Identity Federation with Entra ID (Tim SEA), the variables of `--env-file`
(default wif.gcs.env, never committed) or of the environment. The object is overwritten: the services take
the new model when their pods restart (kubectl rollout restart). Keep the previous file to roll back.
"""

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libs" / "ocr_common"))

from ocr_common.clients import gcs  # noqa: E402
from ocr_common.clients.gcp import credentials_from_env  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path, help="the model file")
    parser.add_argument("uri", help="gs://bucket/path/to/file (overwritten when it exists)")
    parser.add_argument("--env-file", type=Path, default=ROOT / "wif.gcs.env")
    args = parser.parse_args()
    if not args.file.is_file():
        raise SystemExit(f"{args.file} does not exist")

    credentials, identity = credentials_from_env(args.env_file)
    print(f"as {identity}\nupload {args.file} -> {args.uri}")
    stored = gcs.upload(args.file, args.uri, credentials)
    sha256 = stored["metadata"][gcs.SHA256_METADATA]
    print(f"stored: {stored['size']} bytes, md5 {stored['md5Hash']}, generation {stored['generation']}")

    with tempfile.TemporaryDirectory() as scratch:
        back = gcs.download(args.uri, Path(scratch) / args.file.name, credentials, sha256=sha256)
        print(f"downloaded back and verified ({back.stat().st_size} bytes)")
    print(f"\nsha256: {sha256}\nuri:    {args.uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
