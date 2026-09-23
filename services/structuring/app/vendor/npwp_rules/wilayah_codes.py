"""Loads the Depdagri kecamatan code table (dataset/Kode Wilayah/
kode_wilayah.json, generated once by ocr_npwp/app/build_wilayah_codes.py
from the source PDF) for npwp.has_invalid_kecamatan_prefix's exact-match
validation of a 16-digit NIK-based NPWP's first six digits (province+
kabupaten/kota+kecamatan).

Only the kecamatan level is exposed: a code existing there already implies
its 2-digit province and 4-digit kabupaten/kota prefixes are real too
(build_wilayah_codes.py's own sanity checks guarantee every parsed
kecamatan's parent chain resolves), so nothing downstream needs the
province/kabkota tables separately today. Kept as a set of codes, not a
{code: name} dict, since nothing currently reads the place names back out.

Ported from ocr_npwp/app/wilayah_codes.py unchanged apart from the path
default, which follows name_master.py's convention here (env var override,
falling back to the local dev copy under BRI/dataset/) rather than
ocr_npwp's hardcoded-relative-path approach - a deployed container gets
this via a mounted volume/WILAYAH_CODES_PATH, same as NAME_MASTER_PATH.
"""

import json
import os
from pathlib import Path

# nilam-ocr-npwp: the path is read when called, not at import, so WILAYAH_CODES_PATH set by the
# service's settings (app/dependencies.py exports it) is seen; the fallback is data/ next to this file.
_DATA_DIR = Path(__file__).resolve().parent / "data"


def default_data_path() -> Path:
    return Path(os.environ.get("WILAYAH_CODES_PATH") or str(_DATA_DIR / "kode_wilayah.json"))


def _load_valid_kecamatan_codes(data_path: Path | None = None) -> frozenset[str] | None:
    """Returns the set of valid 6-digit kecamatan codes, or None if the
    data file is missing/unreadable/malformed - callers treat None as "no
    signal", the same stance name_master.py takes when its own reference
    file can't be loaded, rather than flagging every number as invalid
    just because the lookup itself failed.

    Not cached (re-reads and re-parses the ~700KB JSON on every call) -
    measured at ~4-5ms per call, negligible next to OCR inference time, but
    revisit with @lru_cache(maxsize=1) if this ever gets called somewhere
    hotter than once or twice per request."""
    data_path = data_path or default_data_path()
    try:
        data = json.loads(data_path.read_text(encoding="utf-8"))
        return frozenset(data["kecamatan"].keys())
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return None


def is_valid_kecamatan_code(code: str) -> bool | None:
    """True/False for whether `code` (expected to be a 6-digit string) is a
    real kecamatan code, or None if the reference data couldn't be loaded
    at all (see _load_valid_kecamatan_codes)."""
    codes = _load_valid_kecamatan_codes()
    if codes is None:
        return None
    return code in codes
