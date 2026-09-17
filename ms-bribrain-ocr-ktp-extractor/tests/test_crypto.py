"""Tests for src.core.crypto (AES-256-GCM log-field encryption).

The crypto helper is byte-identical across the encrypting services (dgc_ext,
dgc_irl, dgc_oct, dgc_pps), so this suite validates the shared behaviour: JSON
round-trip for dicts/lists/strings, NULL handling for empty input, authenticated-
encryption tamper detection, and wrong-key rejection.
"""

import base64
import os

import pytest

# A deterministic 32-byte key, set before the module is first imported.
os.environ.setdefault("LOG_ENCRYPTION_KEY", base64.b64encode(b"k" * 32).decode())

from src.core import crypto  # noqa: E402


def test_roundtrip_dict():
    pii = {"nik": "1234567890123456", "nama": "JOHN DOE", "alamat": "JL X"}
    blob = crypto.encrypt(pii)
    assert isinstance(blob, bytes)
    assert blob != pii  # not stored as plaintext
    assert crypto.decrypt(blob) == pii


def test_roundtrip_list_and_string():
    assert crypto.decrypt(crypto.encrypt([1, "a", {"b": 2}])) == [1, "a", {"b": 2}]
    assert crypto.decrypt(crypto.encrypt("error message")) == "error message"


def test_empty_and_none_are_null():
    assert crypto.encrypt(None) is None
    assert crypto.encrypt("") is None
    assert crypto.decrypt(None) is None


def test_header_layout():
    blob = crypto.encrypt({"a": 1})
    assert blob[0] == 0x01  # version
    assert blob[1] == 0x01  # key_id
    assert len(blob) > 2 + 12  # header + nonce + ciphertext


def test_unique_nonce_per_encryption():
    a = crypto.encrypt({"a": 1})
    b = crypto.encrypt({"a": 1})
    assert a != b  # random nonce -> different ciphertext for identical input


def test_tamper_is_rejected():
    blob = bytearray(crypto.encrypt({"a": 1}))
    blob[-1] ^= 0x01
    with pytest.raises(Exception):
        crypto.decrypt(bytes(blob))


def test_wrong_key_rejected():
    blob = crypto.encrypt({"a": 1})
    saved = crypto._key
    try:
        crypto._key = b"x" * 32  # different key
        with pytest.raises(Exception):
            crypto.decrypt(blob)
    finally:
        crypto._key = saved


def test_unsupported_version_rejected():
    blob = bytearray(crypto.encrypt({"a": 1}))
    blob[0] = 0x02
    with pytest.raises(ValueError):
        crypto.decrypt(bytes(blob))
