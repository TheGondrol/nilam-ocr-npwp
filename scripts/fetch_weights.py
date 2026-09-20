import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "guardrails": {
        "uri_env": "GUARDRAILS_MODEL_URI",
        "sha_env": "GUARDRAILS_MODEL_SHA256",
        "target": ROOT / "services" / "guardrails" / "weights" / "best_model.pt",
    },
}


def _download(uri: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        tmp = Path(handle.name)
    try:
        if uri.startswith("gs://"):
            subprocess.run(["gsutil", "cp", uri, str(tmp)], check=True)
        elif uri.startswith("s3://"):
            subprocess.run(["mc", "cp", uri[len("s3://") :], str(tmp)], check=True)
        elif uri.startswith(("http://", "https://")):
            with urllib.request.urlopen(uri, timeout=120) as response, open(tmp, "wb") as out:
                shutil.copyfileobj(response, out)
        else:
            raise SystemExit(f"skema URI tidak dikenal: {uri!r} (pakai gs://, s3://, atau https://)")
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name: str, *, force: bool) -> int:
    spec = MODELS[name]
    uri = os.environ.get(spec["uri_env"])
    target: Path = spec["target"]
    if not uri:
        state = "sudah ada" if target.is_file() else "TIDAK ADA"
        print(f"{name}: {spec['uri_env']} tidak di-set; lewati unduhan ({target} {state})")
        return 2
    if target.is_file() and not force:
        print(f"{name}: {target} sudah ada, lewati (pakai --force untuk unduh ulang)")
    else:
        print(f"{name}: mengunduh {uri} -> {target}")
        _download(uri, target)

    expected = os.environ.get(spec["sha_env"])
    if expected:
        actual = _sha256(target)
        if actual != expected.lower():
            target.unlink()
            print(f"{name}: sha256 tidak cocok (dapat {actual}, harap {expected}); file dihapus", file=sys.stderr)
            return 1
        print(f"{name}: sha256 cocok")
    print(f"{name}: siap ({target.stat().st_size // 1024} KB)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", default=list(MODELS), help="nama model (default: semua)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    return max(fetch(name, force=args.force) for name in args.models)


if __name__ == "__main__":
    sys.exit(main())
