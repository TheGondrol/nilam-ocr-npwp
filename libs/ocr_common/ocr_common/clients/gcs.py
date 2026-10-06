"""Model files in Google Cloud Storage: upload once, download where a service starts, verified both ways.

An object is never overwritten (`ifGenerationMatch=0`): a new model goes under a new path, so the one a
deployment pins stays exactly what it was and a rollback is a config change. Each upload records the file's
SHA-256 in the object's metadata; a download checks the object's MD5 (GCS's own) always, and the SHA-256 the
caller pins (`*_MODEL_SHA256`) or, without one, the one recorded at upload.

Only `google-auth[requests]` (the `gcs` extra of ocr-common): the JSON API is called directly.
"""

import base64
import hashlib
import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

logger = logging.getLogger(__name__)

API = "https://storage.googleapis.com/storage/v1"
UPLOAD_API = "https://storage.googleapis.com/upload/storage/v1"
CHUNK = 1 << 20
SHA256_METADATA = "sha256"


class GcsError(RuntimeError):
    """A GCS call failed or a file did not match its checksum; the message says which and why."""


@dataclass(frozen=True)
class GcsObject:
    bucket: str
    name: str

    @classmethod
    def parse(cls, uri: str) -> "GcsObject":
        """`gs://bucket/path/to/file`."""
        if not uri.startswith("gs://") or "/" not in uri[5:]:
            raise ValueError(f"not a gs://bucket/object URI: {uri!r}")
        bucket, name = uri[5:].split("/", 1)
        if not bucket or not name or name.endswith("/"):
            raise ValueError(f"not a gs://bucket/object URI: {uri!r}")
        return cls(bucket, name)

    @property
    def uri(self) -> str:
        return f"gs://{self.bucket}/{self.name}"

    @property
    def path(self) -> str:
        return f"/b/{self.bucket}/o/{quote(self.name, safe='')}"


def _session(credentials: Any) -> Any:
    from google.auth.transport.requests import AuthorizedSession

    return AuthorizedSession(credentials)


def _check(response: Any, action: str, obj: GcsObject) -> None:
    if response.status_code >= 400:
        try:
            message = response.json().get("error", {}).get("message", "")
        except ValueError:
            message = response.text[:200]
        raise GcsError(f"{action} {obj.uri} failed ({response.status_code}): {message}")


def file_digests(path: Path) -> tuple[str, str]:
    """(sha256 hex, md5 base64, as GCS reports it) of a file."""
    sha256, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), base64.b64encode(md5.digest()).decode()


def metadata(uri: str, credentials: Any) -> dict[str, Any]:
    """The object's metadata (size, md5Hash, metadata.sha256, ...); `GcsError` when it does not exist."""
    obj = GcsObject.parse(uri)
    response = _session(credentials).get(API + obj.path, timeout=60)
    _check(response, "reading", obj)
    return response.json()


def upload(path: Path, uri: str, credentials: Any) -> dict[str, Any]:
    """Uploads `path` as `uri`, refusing to overwrite an existing object, with its SHA-256 in the metadata;
    checks the MD5 GCS computed against the file's. Returns the new object's metadata."""
    obj = GcsObject.parse(uri)
    sha256, md5 = file_digests(path)
    boundary = uuid.uuid4().hex
    head = json.dumps({"name": obj.name, "metadata": {SHA256_METADATA: sha256}}).encode()
    body = b"".join(
        [
            f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n".encode(),
            head,
            f"\r\n--{boundary}\r\nContent-Type: application/octet-stream\r\n\r\n".encode(),
            path.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    response = _session(credentials).post(
        f"{UPLOAD_API}/b/{obj.bucket}/o",
        params={"uploadType": "multipart", "ifGenerationMatch": "0"},
        data=body,
        headers={"Content-Type": f"multipart/related; boundary={boundary}"},
        timeout=300,
    )
    if response.status_code == 412:
        raise GcsError(f"{obj.uri} already exists: objects are never overwritten, upload under a new path")
    _check(response, "uploading", obj)
    stored = response.json()
    if stored.get("md5Hash") != md5:
        raise GcsError(f"{obj.uri}: MD5 after upload {stored.get('md5Hash')} differs from the file's {md5}")
    return stored


def download(uri: str, target: Path, credentials: Any, *, sha256: str | None = None) -> Path:
    """Downloads `uri` to `target` (atomically: a partial file never appears there) and verifies it: the MD5
    of GCS, then `sha256` or, without it, the SHA-256 recorded at upload. A `target` that already has the
    expected SHA-256 is kept as it is. Returns `target`."""
    obj = GcsObject.parse(uri)
    session = _session(credentials)
    response = session.get(API + obj.path, timeout=60)
    _check(response, "reading", obj)
    meta = response.json()
    expected = (sha256 or meta.get("metadata", {}).get(SHA256_METADATA) or "").lower() or None
    if expected and target.is_file() and file_digests(target)[0] == expected:
        logger.info("model %s already at %s (sha256 matches)", obj.uri, target)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    temp = Path(temp_name)
    try:
        with os.fdopen(handle, "wb") as out:
            with session.get(API + obj.path, params={"alt": "media"}, stream=True, timeout=300) as media:
                _check(media, "downloading", obj)
                for chunk in media.iter_content(CHUNK):
                    out.write(chunk)
        actual_sha256, actual_md5 = file_digests(temp)
        if meta.get("md5Hash") and actual_md5 != meta["md5Hash"]:
            raise GcsError(f"{obj.uri}: downloaded MD5 {actual_md5} differs from the object's {meta['md5Hash']}")
        if expected and actual_sha256 != expected:
            raise GcsError(f"{obj.uri}: SHA-256 {actual_sha256} differs from the expected {expected}")
        temp.replace(target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    logger.info("model %s downloaded to %s (%d KB)", obj.uri, target, target.stat().st_size // 1024)
    return target
