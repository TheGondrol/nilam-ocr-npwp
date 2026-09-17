"""Constants and mappings for OCR processing.

This module loads all constants, mappings, and character corrections
from config.yaml for easier maintenance and configuration.

Threshold constants below are *live* — comparisons read the current value
from the threshold provider (refreshed hourly from the management DB).
"""

from typing import Any, List, Tuple

from src.core.config import get_config
from src.services.threshold_provider import get_provider

# Get configuration
config = get_config()

# Load all configs
constants = config.get_constants()
mappings = config.get_mappings()
messages = config.get_messages()
character_mappings = config.get_character_mappings()
thresholds = config.get_thresholds()
regex_patterns = config.get_regex()


class _LiveThreshold:
    """Numeric-comparable proxy that reads from the threshold provider.

    Imports like `from src.services.constants import THRESHOLD_CONFIDENCE`
    bind to a single instance per process; every comparison goes through
    `_live()` so updates from the management service take effect within the
    provider's refresh interval.
    """

    __slots__ = ("_key", "_default")

    def __init__(self, key: str, default: float) -> None:
        self._key = key
        self._default = float(default)

    def _live(self) -> float:
        try:
            return float(get_provider().get(self._key))
        except Exception:
            return self._default

    def __lt__(self, other: Any) -> bool: return self._live() < other
    def __le__(self, other: Any) -> bool: return self._live() <= other
    def __gt__(self, other: Any) -> bool: return self._live() > other
    def __ge__(self, other: Any) -> bool: return self._live() >= other
    def __eq__(self, other: Any) -> bool: return self._live() == other
    def __ne__(self, other: Any) -> bool: return self._live() != other
    def __float__(self) -> float: return self._live()
    def __int__(self) -> int: return int(self._live())
    def __hash__(self) -> int: return hash(("live", self._key))
    def __repr__(self) -> str: return f"<LiveThreshold {self._key}={self._live()}>"


# ============================================================================
# Threshold Constants (centralized for all modules) — live from provider
# ============================================================================
THRESHOLD_CONFIDENCE = _LiveThreshold("confidence", thresholds.get("confidence", 0.8))
THRESHOLD_PARTIAL = _LiveThreshold("partial", thresholds.get("partial", 75))
THRESHOLD_RATIO = _LiveThreshold("ratio", thresholds.get("ratio", 80))

# ============================================================================
# Regex Patterns (centralized for all modules)
# ============================================================================
REGEX = regex_patterns

# ============================================================================
# Type Aliases for OCR Data
# ============================================================================
# OCR data format: [text, confidence, box]
# Box format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
OCRBox = List[List[int]]
OCRItem = Tuple[str, float, OCRBox]  # (text, confidence, bounding_box)

# ============================================================================
# Key Data Tuples for Field Identification
# ============================================================================
keydata = tuple(constants.get("keydata", []))
keyid = tuple(constants.get("keyid", []))
keydata_status = tuple(constants.get("keydata_status", []))
keydata_agama = tuple(constants.get("keydata_agama", []))
keydata_jeniskelamin = tuple(constants.get("keydata_jeniskelamin", []))
data_key_unittest = tuple(constants.get("data_key_unittest", []))

# ============================================================================
# Mapping Tuples for Field Values
# ============================================================================
mapping_pekerjaan = tuple(mappings.get("pekerjaan", []))

# Pekerjaan key list mirrors the status_perkawinan pattern: the label plus
# every known value, so the remap pass can match a value cell directly even
# when the "pekerjaan" label itself was missed by the OCR.
keydata_pekerjaan = ("pekerjaan",) + tuple(v.lower() for v in mapping_pekerjaan)

keydata_repeat = tuple(constants.get("keydata_repeat", [])) + keydata_pekerjaan
mapping_status = tuple(mappings.get("status", []))
mapping_agama = tuple(mappings.get("agama", []))
mapping_jeniskelamin = tuple(mappings.get("jenis_kelamin", []))

# ============================================================================
# Month Mapping Dictionary
# ============================================================================
bulan_dict = config.get_bulan_dict()

# ============================================================================
# Error Messages
# ============================================================================
message_glare = messages.get("glare", "glare")
message_blur = messages.get("blur", "blur")
message_rotated = messages.get("rotated", "tidak sejajar")

# ============================================================================
# Character Correction Mappings
# ============================================================================
alphabet_mapping = character_mappings.get("alphabet_to_digit", {})
digit_mapping = character_mappings.get("digit_to_alphabet", {})
symbol_mapping = character_mappings.get("symbol_to_digit", {})
