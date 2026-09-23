import re

_BADAN_PREFIX = re.compile(r"^\s*(PT|CV|UD|PD|KOPERASI|YAYASAN|FIRMA)\b", re.IGNORECASE)


def normalize_npwp(value: str) -> str:
    """15 digits, in any punctuation, become the printed form XX.XXX.XXX.X-XXX.XXX; anything else is
    returned trimmed (a 16-digit NPWP has no printed punctuation)."""
    digits = re.sub(r"\D", "", value)
    if len(digits) == 15:
        return f"{digits[0:2]}.{digits[2:5]}.{digits[5:8]}.{digits[8]}-{digits[9:12]}.{digits[12:15]}"
    return value.strip()


def is_badan(name: str) -> bool:
    """A name that starts with a legal-entity prefix (PT, CV, ...) belongs in `nama_badan`, not `nama`."""
    return _BADAN_PREFIX.match(name) is not None
