"""Unit tests for src.core.crypto (AES-256-GCM field encryption)."""

import base64

import pytest

import src.core.crypto as crypto

VALID_KEY = base64.b64encode(b"0" * 32).decode()


@pytest.fixture
def valid_key(monkeypatch):
    """Set a valid 32-byte key and reset the module-level key cache."""
    monkeypatch.setenv("LOG_ENCRYPTION_KEY", VALID_KEY)
    crypto._key = None
    yield
    crypto._key = None


class TestLoadKey:
    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("LOG_ENCRYPTION_KEY", raising=False)
        crypto._key = None
        with pytest.raises(RuntimeError, match="is not set"):
            crypto._load_key()
        crypto._key = None

    def test_blank_key_raises(self, monkeypatch):
        monkeypatch.setenv("LOG_ENCRYPTION_KEY", "   ")
        crypto._key = None
        with pytest.raises(RuntimeError, match="is not set"):
            crypto._load_key()
        crypto._key = None

    def test_invalid_base64_raises(self, monkeypatch):
        monkeypatch.setenv("LOG_ENCRYPTION_KEY", "a")  # invalid length for base64
        crypto._key = None
        with pytest.raises(RuntimeError, match="not valid base64"):
            crypto._load_key()
        crypto._key = None

    def test_wrong_length_raises(self, monkeypatch):
        monkeypatch.setenv("LOG_ENCRYPTION_KEY", base64.b64encode(b"short").decode())
        crypto._key = None
        with pytest.raises(RuntimeError, match="32 bytes"):
            crypto._load_key()
        crypto._key = None

    def test_valid_key_cached(self, valid_key):
        k1 = crypto._load_key()
        assert len(k1) == 32
        # Cached: a second call returns the same object.
        assert crypto._load_key() is k1


class TestEncrypt:
    def test_none_returns_none(self, valid_key):
        assert crypto.encrypt(None) is None

    def test_empty_string_returns_none(self, valid_key):
        assert crypto.encrypt("") is None

    def test_encrypts_dict_to_bytes(self, valid_key):
        blob = crypto.encrypt({"nama": "JOHN"})
        assert isinstance(blob, bytes)
        assert blob[0] == crypto._VERSION
        assert blob[1] == crypto._KEY_ID
        # version + key_id + nonce + ciphertext(+tag)
        assert len(blob) > crypto._HEADER_LEN + crypto._NONCE_LEN


class TestDecrypt:
    def test_none_returns_none(self, valid_key):
        assert crypto.decrypt(None) is None

    def test_too_short_raises(self, valid_key):
        with pytest.raises(ValueError, match="too short"):
            crypto.decrypt(b"\x01\x01")

    def test_unsupported_version_raises(self, valid_key):
        blob = bytes([0x02, crypto._KEY_ID]) + b"\x00" * (crypto._NONCE_LEN + 16)
        with pytest.raises(ValueError, match="unsupported"):
            crypto.decrypt(blob)

    @pytest.mark.parametrize(
        "value",
        [{"nama": "JOHN", "nik": "123"}, "a plain string", ["a", "b", 1], 0],
    )
    def test_roundtrip(self, valid_key, value):
        assert crypto.decrypt(crypto.encrypt(value)) == value

    def test_roundtrip_via_memoryview(self, valid_key):
        blob = crypto.encrypt({"k": "v"})
        assert crypto.decrypt(memoryview(blob)) == {"k": "v"}
