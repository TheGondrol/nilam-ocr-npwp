"""Correct an OCR'd name that dropped inter-word spacing (e.g. "HASTATIABBAS"
for "HASTATI ABBAS") by matching it against a reference master list of real,
correctly-spaced Indonesian names (see NAME_MASTER_PATH env var below).

Active behavior is exact-match only: the OCR'd name is squished (all spaces
removed, uppercased) and looked up directly in a dict built the same way
from the master list. This is deliberately not comparing spaced text to
spaced text - despacing *both* sides first is exactly what makes it not
matter which space(s) OCR actually dropped, or how many: a fully collapsed
name ("MUHAMADRINOSUKIRMA"), a partially-merged one ("MUHAMAD
RINOSUKIRMA"), and the already-correct one ("MUHAMAD RINO SUKIRMA") all
squish down to the same key and resolve to the same correct master entry.
Effectively instant once the index is built (one-time cost, see
_load_index). A name the master doesn't contain at all - e.g. a customer
who isn't in the reference list yet - is returned unchanged: there's
nothing to look up, so nothing is guessed.

Master entries whose despaced form is ambiguous - two or more different
real names collapse to the same squished string, e.g. "APRIYADI" and
"APRI YADI" both squish to "APRIYADI" - are excluded from the index.
With no reliable way to pick between them, leaving the OCR text untouched
is safer than guessing, since guessing wrong would actively break a name
that was already extracted correctly (e.g. a genuine single-word
"APRIYADI" getting wrongly split just because the master also happens to
list a same-spelled two-word name).

Second step: truncated-prefix match. The master caps every entry at 20
raw characters - a legacy fixed-width field limit, confirmed by a sharp
spike of 70k+ entries sitting at exactly length 20 versus a smooth falloff
below it - so a real name longer than that (e.g. "I KETUT NONIK SUHERMAN")
is only ever stored truncated ("I KETUT NONIK SUHERM"), and can never
exact-match a full, correctly-spaced query. For queries the exact step
misses, this second step checks whether a 20-char (i.e. likely-truncated)
master entry's despaced form is a *prefix* of the query's despaced form,
picking the longest such prefix available. On a hit, it trusts the master
only for that verified prefix span and appends whatever the OCR text
already had past that point completely unchanged - the master has zero
information beyond its own cutoff, so nothing past it is ever guessed.
Tested against real production data: fixes real truncation-blocked cases,
but - because it only needs partial (prefix) agreement rather than full
agreement - it is more exposed than the exact-match step to two different
real people coincidentally sharing the same opening letters (e.g. despaced
"ANISAPUTRIWULANDARI", one continuous real name, wrongly matched a
different truncated entry "ANISA PUTRI WULANDAR" that happens to share the
same first 18 letters). Measured trade-off: 8 additional names fixed for
3 additional regressions, versus the exact-match step's own 31-for-3 - a
knowingly worse ratio, accepted here because the residual gain was judged
worth it.

Both steps run through the same word-count guard before being accepted:
if the corrected form has *fewer* words than the original OCR text, the
correction is discarded and the original is kept. This costs nothing
against the fixes either step finds (every genuine space-recovery fix
*adds* words, never removes them) while blocking the specific failure mode
where the master wrongly *merges* an already-correctly-spaced name (e.g.
"FEBRI YANSYAH" -> "FEBRIYANSYAH", because a different, single-word real
person shares that despaced key).

A fuzzy fallback (for a squished name that also has a misread letter, e.g.
"HASTAT1ABBAS") was prototyped and deliberately left disabled - see
correct_name_spacing's `use_fuzzy_fallback` param. Testing against the real
~270k-name master showed it isn't safe even after two rounds of
restrictions (same despaced length as the query; result must contain a
space): with this many names, a genuinely correct single-word name
routinely has an unrelated same-length, high-similarity neighbor elsewhere
in the list (e.g. despaced "APRIYADI" matched "M APRIADI", a different
real person, at a score and shape that passed every filter tried). The
code is kept for a future attempt at a safer heuristic, not wired into the
default path.

correct_name_spacing (the general despaced-master-list guess) is no longer
the active correction name_extraction.extract_name applies - it proved too
regression-prone dataset-wide and was superseded there by
correct_name_with_npwp_list below, a narrower and safer correction that
only fires when this specific document's own registered name is already
known (see that function's docstring). correct_name_spacing/is_recognized_name
stay here and in active use regardless: is_recognized_name is still
extract_name's candidate-selection tie-breaker (a stray OCR fragment vs.
the real name line - a different job from "guess the correction"), and
correct_name_spacing itself is kept, unused by default, for future
experimentation the same way its own use_fuzzy_fallback param already is.
"""

import os
import re
from functools import lru_cache
from pathlib import Path

import openpyxl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

# Configurable so a deployed container can point this at wherever the
# reference file actually lives there (baked into the image, a mounted
# volume, a path fetched from GCS at startup, ...) instead of assuming the
# local dev machine's directory layout. Default below is the local dev
# path only - BRI/dataset sits two levels above GEN/, which is three more
# levels above this file (ocr/name_master.py -> nilamnpwp/ -> GEN/ -> BRI/).
# nilam-ocr-npwp: the paths are read when called, not at import, so NAME_MASTER_PATH /
# NPWP_NAME_LIST_PATH set by the service's settings (app/dependencies.py exports them) are seen; the
# fallback is data/ next to this file. Both spreadsheets are internal data: not in git, mounted at deploy.
_DATA_DIR = Path(__file__).resolve().parent / "data"


def default_master_path() -> Path:
    return Path(os.environ.get("NAME_MASTER_PATH") or str(_DATA_DIR / "name_lnmast.xlsx"))


def default_list_path() -> Path:
    return Path(os.environ.get("NPWP_NAME_LIST_PATH") or str(_DATA_DIR / "list_name_npwp.xlsx"))

# Below this rapidfuzz.fuzz.ratio score (0-100), a fuzzy-fallback candidate
# is considered too dissimilar to trust as "the same name with one OCR
# error". Candidates are already restricted to the query's own despaced
# length (see _keys_by_length), so this only needs to tolerate a single
# substituted character - for an 8-character name that's a ratio of
# 100 * 7/8 = 87.5, so 85 comfortably covers one substitution even on short
# names while still rejecting names that differ by two or more characters.
FUZZY_SCORE_CUTOFF = 85

# Reference list of each known document's own registered taxpayer name
# (refno;name_dh pairs, refno matching the file_id a caller of this service
# supplies - see doc_service/main.py's extract_ocr). Ported from
# ocr_npwp/app/name_master.py, whose own default path is a hardcoded
# relative BRI/dataset/list_name_npwp.xlsx - kept configurable here instead,
# same NAME_MASTER_PATH-style convention as default_master_path() above.

# Jaro-Winkler similarity (0-100) a name candidate must meet or exceed
# against this file_id's own reference name (name_dh) before it's trusted
# as a correction - see correct_name_with_npwp_list.
JARO_WINKLER_THRESHOLD = 98.0


def _despace(name: str) -> str:
    return re.sub(r"\s+", "", name.upper())


def _normalize_slash_spacing(text: str) -> str:
    """Cards that print maiden/married names on one line join them with a
    bare "/" and inconsistent (or no) surrounding whitespace (e.g. "A/B",
    "A/ B") - normalize every "/" to exactly one space on each side, so a
    name candidate's own slash spacing (already normalized the same way by
    name_extraction.normalize_slash_spacing before this ever sees it)
    can't cause a spurious Jaro-Winkler mismatch against name_dh's spacing.
    Private, separate copy of name_extraction.normalize_slash_spacing -
    this module can't import that one back without a circular import."""
    return re.sub(r"\s*/\s*", " / ", text)


@lru_cache(maxsize=1)
def _load_index(master_path: Path) -> dict[str, str]:
    """Build {despaced_name: spaced_name} from the master list, once per
    process (subsequent calls with the same path reuse the cached result).
    Despaced keys with more than one distinct spaced form in the master are
    dropped - see module docstring for why that ambiguity isn't guessable."""
    wb = openpyxl.load_workbook(master_path, read_only=True)
    ws = wb.worksheets[0]

    variants: dict[str, set[str]] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        raw = row[0]
        if not raw:
            continue
        name = str(raw).strip()
        if not name:
            continue
        variants.setdefault(_despace(name), set()).add(name)

    return {key: next(iter(names)) for key, names in variants.items() if len(names) == 1}


@lru_cache(maxsize=1)
def _keys_by_length(master_path: Path) -> dict[int, list[str]]:
    """Index keys grouped by their despaced length, so the fuzzy fallback
    only ever searches candidates the same length as the query - see the
    module docstring for why cross-length matches are unsafe. Cached
    separately from _load_index so this grouping isn't redone per call."""
    by_length: dict[int, list[str]] = {}
    for key in _load_index(master_path):
        by_length.setdefault(len(key), []).append(key)
    return by_length


@lru_cache(maxsize=1)
def _load_truncated_index(master_path: Path) -> dict[str, str]:
    """Build {despaced_prefix: spaced_20char_entry} from master rows whose
    raw (spaced) length is exactly 20 - the signature of a name cut off by
    the master's field-length cap (see module docstring). Ambiguous
    despaced prefixes (two different 20-char entries sharing the same
    despaced form) are dropped, same reasoning as _load_index - this index
    is deliberately separate from it since a 20-char entry is only a
    partial name, never a valid exact-match target on its own."""
    wb = openpyxl.load_workbook(master_path, read_only=True)
    ws = wb.worksheets[0]

    variants: dict[str, set[str]] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        raw = row[0]
        if not raw:
            continue
        name = str(raw).strip()
        if len(name) != 20:
            continue
        variants.setdefault(_despace(name), set()).add(name)

    return {key: next(iter(names)) for key, names in variants.items() if len(names) == 1}


@lru_cache(maxsize=1)
def _truncated_keys_by_length(master_path: Path) -> dict[int, set[str]]:
    """Truncated-index keys grouped by despaced length, so the longest
    matching prefix can be found with a handful of dict lookups (one per
    candidate length) instead of scanning every truncated entry per
    query."""
    by_length: dict[int, set[str]] = {}
    for key in _load_truncated_index(master_path):
        by_length.setdefault(len(key), set()).add(key)
    return by_length


def _split_at_nonspace_count(original: str, n: int) -> tuple[str, str]:
    """Split `original` right after its Nth non-space character, keeping
    every space `original` already had on both sides of the cut - so the
    untouched tail past a truncated-prefix match preserves whatever
    spacing OCR actually produced there, rather than assuming any."""
    count = 0
    for i, ch in enumerate(original):
        if not ch.isspace():
            count += 1
            if count == n:
                return original[: i + 1], original[i + 1 :]
    return original, ""


def _match_truncated_prefix(original: str, key: str, master_path: Path) -> str | None:
    """Try the longest despaced truncated-master prefix that matches the
    start of `key`, and splice the master's verified spacing for that span
    onto the query's own untouched tail - see module docstring. Returns
    None if no truncated entry's despaced form prefixes this query."""
    trunc_index = _load_truncated_index(master_path)
    by_length = _truncated_keys_by_length(master_path)

    for length in range(min(len(key) - 1, 20), 0, -1):
        candidates = by_length.get(length)
        if not candidates:
            continue
        prefix = key[:length]
        if prefix in candidates:
            _head, tail = _split_at_nonspace_count(original, length)
            return trunc_index[prefix] + tail

    return None


def is_recognized_name(name: str, master_path: Path | None = None) -> bool:
    """True if `name`'s despaced form is an exact-match entry in the master
    list (same index correct_name_spacing's exact step uses - not the
    truncated-prefix or fuzzy steps, which are name_correction heuristics
    rather than a plausibility signal). False - never raises - if the
    master list is missing/unreadable or `name` is empty, same best-effort
    fallback correct_name_spacing already uses.

    Meant as a tie-breaker among multiple already-plausible name
    candidates (see name_extraction.extract_name), not a filter on its
    own: the master doesn't cover every real name (a customer who isn't in
    the reference list yet is a normal case, not a red flag), so treat
    "not recognized" as "no extra signal either way," never as "this must
    be wrong." """
    if not name:
        return False
    master_path = master_path or default_master_path()
    try:
        index = _load_index(master_path)
    except (FileNotFoundError, OSError):
        return False
    return _despace(name) in index


def _apply_word_count_guard(original: str, corrected: str) -> str:
    """Reject a correction that would reduce word count versus the
    original - every genuine space-recovery fix adds words, so a
    word-count *decrease* only ever means the master matched a different,
    already-merged real name (see module docstring's "FEBRI YANSYAH"
    example) rather than fixing anything."""
    if len(corrected.split()) < len(original.split()):
        return original
    return corrected


def correct_name_spacing(name: str, master_path: Path | None = None, use_fuzzy_fallback: bool = False) -> str:
    """Return `name` with inter-word spacing corrected against the master
    list, or `name` unchanged if no confident match is found - including
    when the master list itself is missing/unreadable, since this is a
    best-effort improvement layered on top of OCR, never a hard
    requirement.

    Tries, in order: exact despaced match, then a truncated-master-prefix
    match (for names longer than the master's 20-char cap - see module
    docstring), then - only if `use_fuzzy_fallback` is passed True, off by
    default - a same-length fuzzy match. Every step's result passes through
    the word-count guard (_apply_word_count_guard) before being accepted.

    `use_fuzzy_fallback` stays off by default - see module docstring for
    why that step isn't considered safe to run yet. Left as a parameter,
    rather than deleted, so it stays easy to experiment with a future,
    stricter version of it without redoing this plumbing."""
    if not name:
        return name
    original = name.strip()
    master_path = master_path or default_master_path()

    try:
        index = _load_index(master_path)
    except (FileNotFoundError, OSError):
        return name

    key = _despace(original)
    if not key:
        return name

    exact = index.get(key)
    if exact is not None:
        return _apply_word_count_guard(original, exact)

    try:
        prefixed = _match_truncated_prefix(original, key, master_path)
    except (FileNotFoundError, OSError):
        prefixed = None
    if prefixed is not None:
        return _apply_word_count_guard(original, prefixed)

    if not use_fuzzy_fallback:
        return original

    same_length_candidates = _keys_by_length(master_path).get(len(key))
    if not same_length_candidates:
        return original

    # extract (not extractOne) + a small limit: the single best-scoring
    # candidate can be an unrelated single-word name (see module docstring),
    # so this looks at a handful of top candidates and takes the
    # highest-scoring one that actually recovers a space, rather than
    # giving up just because #1 happens to be single-word.
    matches = process.extract(key, same_length_candidates, scorer=fuzz.ratio, score_cutoff=FUZZY_SCORE_CUTOFF, limit=5)
    for matched_key, _score, _idx in matches:
        corrected = index[matched_key]
        if " " in corrected:
            return _apply_word_count_guard(original, corrected)

    return original


@lru_cache(maxsize=1)
def _load_npwp_name_index(list_path: Path) -> dict[str, str]:
    """Build {refno: name_dh} from list_name_npwp.xlsx, once per process.
    Each data row is a single "refno;name_dh" cell. A refno can appear more
    than once in the source file (sometimes once with a name and once
    blank) - the first non-empty name_dh seen for a refno wins, and a
    refno with no non-empty name_dh anywhere maps to "" (meaning "no
    correction available", not "not found"). Ported from
    ocr_npwp/app/name_master.py unchanged."""
    wb = openpyxl.load_workbook(list_path, read_only=True)
    ws = wb.worksheets[0]

    index: dict[str, str] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        raw = row[0]
        if not raw:
            continue
        refno, _, name_dh = str(raw).partition(";")
        refno = refno.strip()
        name_dh = name_dh.strip()
        if not refno:
            continue
        if name_dh and not index.get(refno):
            index[refno] = name_dh
        elif refno not in index:
            index[refno] = name_dh

    return index


def score_name_against_npwp_list(
    name_base: str,
    file_id: str | None,
    list_path: Path | None = None,
) -> tuple[float, str] | None:
    """Return (jaro_winkler_score, name_dh) comparing name_base against
    this file_id's reference name, or None if there's nothing to compare -
    name_base/file_id missing, the reference list is missing/unreadable, or
    this file_id has no (non-empty) entry in it. Split out from
    correct_name_with_npwp_list so a caller that wants the raw score/match
    for reporting doesn't have to recompute the lookup or the similarity
    itself. Ported from ocr_npwp/app/name_master.py unchanged."""
    if not name_base or not file_id:
        return None
    list_path = list_path or default_list_path()

    try:
        index = _load_npwp_name_index(list_path)
    except (FileNotFoundError, OSError):
        return None

    name_dh = index.get(file_id)
    if not name_dh:
        return None

    name_dh = _normalize_slash_spacing(name_dh)
    score = JaroWinkler.normalized_similarity(
        _normalize_slash_spacing(name_base).strip().upper(), name_dh.upper()
    ) * 100
    return score, name_dh


def correct_name_with_npwp_list(
    name_base: str,
    file_id: str | None,
    list_path: Path | None = None,
    threshold: float = JARO_WINKLER_THRESHOLD,
) -> str:
    """Return name_dh in place of name_base if this file_id has a reference
    name and name_base is at least `threshold` Jaro-Winkler similar to it;
    otherwise return name_base unchanged - including when the reference
    list is missing/unreadable, this file_id has no entry, or its entry is
    empty, since this is a best-effort improvement layered on top of OCR,
    never a hard requirement. This is name_extraction.extract_name's active
    master-correction step (see module docstring for why it replaced
    correct_name_spacing there). Ported from ocr_npwp/app/name_master.py
    unchanged."""
    result = score_name_against_npwp_list(name_base, file_id, list_path)
    if result is None:
        return name_base

    score, name_dh = result
    if score >= threshold:
        return name_dh
    return _normalize_slash_spacing(name_base).strip()
