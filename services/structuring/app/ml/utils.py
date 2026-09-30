import re

_BADAN_PREFIX = re.compile(r"^\s*(PT|CV|UD|PD|KOPERASI|YAYASAN|FIRMA)\b", re.IGNORECASE)


def normalize_npwp(value: str) -> str:
    """15 digits, in any punctuation, become plain digits (the dots and dash are stripped); anything
    else is returned trimmed."""
    digits = re.sub(r"\D", "", value)
    if len(digits) == 15:
        return digits
    return value.strip()


def is_badan(name: str) -> bool:
    """A name that starts with a legal-entity prefix (PT, CV, ...) belongs in `nama_badan`, not `nama`."""
    return _BADAN_PREFIX.match(name) is not None
