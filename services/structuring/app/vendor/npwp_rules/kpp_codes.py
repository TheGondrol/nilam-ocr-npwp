"""Loads the KPP (Kantor Pelayanan Pajak) code table (dataset/Kode
Wilayah/kpp_codes.json, generated once by ocr_npwp/app/build_kpp_codes.py
from the source PDF) for npwp.has_invalid_kpp_prefix's exact-match
validation of a legacy 15-digit NPWP's digits 10-12 (the registering KPP
office's code).

Not cached, same as wilayah_codes.py - re-reads and re-parses the (much
smaller, ~170-entry) JSON on every call. Revisit with @lru_cache(maxsize=1)
if this ever gets called somewhere hotter than once or twice per request.

Ported from ocr_npwp/app/kpp_codes.py unchanged apart from the path
default - see wilayah_codes.py's docstring for why (env var override /
local-dev fallback, matching name_master.py's convention here)."""

import json
import os
from pathlib import Path

# nilam-ocr-npwp: same lazy resolution as wilayah_codes.py (KPP_CODES_PATH, fallback data/).
_DATA_DIR = Path(__file__).resolve().parent / "data"


def default_data_path() -> Path:
    return Path(os.environ.get("KPP_CODES_PATH") or str(_DATA_DIR / "kpp_codes.json"))


def _load_valid_kpp_codes(data_path: Path | None = None) -> frozenset[str] | None:
    """Returns the set of valid 3-digit KPP codes, or None if the data
    file is missing/unreadable/malformed - callers treat None as "no
    signal", the same stance wilayah_codes.py and name_master.py take when
    their own reference file can't be loaded, rather than flagging every
    number as invalid just because the lookup itself failed."""
    data_path = data_path or default_data_path()
    try:
        data = json.loads(data_path.read_text(encoding="utf-8"))
        return frozenset(data.keys())
    except (FileNotFoundError, OSError, ValueError):
        return None


def is_valid_kpp_code(code: str) -> bool | None:
    """True/False for whether `code` (expected to be a 3-digit string) is
    a real KPP code, or None if the reference data couldn't be loaded at
    all (see _load_valid_kpp_codes)."""
    codes = _load_valid_kpp_codes()
    if codes is None:
        return None
    return code in codes
