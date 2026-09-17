"""Field-level encryption for sensitive log columns (DPIA).

AES-256-GCM authenticated encryption. The symmetric key is read from the
``LOG_ENCRYPTION_KEY`` environment variable (base64 of 32 bytes), consistent with
how ``DATABASE_URL`` / ``API_KEY`` are provided. Stored value layout (BYTEA):

    version(1) || key_id(1) || nonce(12) || ciphertext+tag

Values are JSON-encoded before encryption, so dicts/lists (``payload``/``result``)
and strings (``error_message``) are handled uniformly. ``encrypt()`` returns
``None`` for ``None``/empty input so the column stays NULL; ``decrypt()`` reverses
both. This module is intentionally self-contained and identical across services.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_VERSION = 0x01
_KEY_ID = 0x01          # reserved for future key rotation
_NONCE_LEN = 12
_HEADER_LEN = 2         # version + key_id

_ENV_VAR = "LOG_ENCRYPTION_KEY"

_key: Optional[bytes] = None


def _load_key() -> bytes:
    """Load and cache the 32-byte AES key from the environment."""
    global _key
    if _key is None:
        raw = os.getenv(_ENV_VAR)
        if not raw or not raw.strip():
            raise RuntimeError(
                f"{_ENV_VAR} is not set; cannot encrypt/decrypt log data"
            )
        try:
            key = base64.b64decode(raw.strip())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"{_ENV_VAR} is not valid base64: {exc}") from exc
        if len(key) != 32:
            raise RuntimeError(
                f"{_ENV_VAR} must decode to 32 bytes (AES-256); got {len(key)}"
            )
        _key = key
    return _key


def encrypt(value: Any) -> Optional[bytes]:
    """Encrypt a JSON-serializable value into a BYTEA blob.

    Returns ``None`` for ``None`` or empty string so the DB column stays NULL.
    """
    if value is None or value == "":
        return None
    plaintext = json.dumps(value, default=str, ensure_ascii=False).encode("utf-8")
    nonce = os.urandom(_NONCE_LEN)
    ciphertext = AESGCM(_load_key()).encrypt(nonce, plaintext, None)
    return bytes([_VERSION, _KEY_ID]) + nonce + ciphertext


def decrypt(blob: Optional[bytes]) -> Any:
    """Decrypt a BYTEA blob produced by :func:`encrypt`. ``None`` -> ``None``."""
    if blob is None:
        return None
    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    if len(blob) < _HEADER_LEN + _NONCE_LEN:
        raise ValueError("ciphertext too short / not an encrypted log value")
    version = blob[0]
    if version != _VERSION:
        raise ValueError(f"unsupported log-encryption version: {version}")
    nonce = blob[_HEADER_LEN:_HEADER_LEN + _NONCE_LEN]
    ciphertext = blob[_HEADER_LEN + _NONCE_LEN:]
    plaintext = AESGCM(_load_key()).decrypt(nonce, ciphertext, None)
    return json.loads(plaintext.decode("utf-8"))
