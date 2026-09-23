import re

from .name_master import correct_name_spacing, is_recognized_name
from .npwp import NPWP_PATTERN, find_document_region, poly_center

# A name candidate line: letters/spaces only, no digits - excludes anything
# that's clearly a number, date, or address with a house/RT-RW number. "/"
# is allowed since some cards print maiden-name/married-name as one line
# (e.g. "RATNA SUSANTI/SYAMARIS MULYODI SASTRO"). No comma here deliberately
# - see NAME_TITLE_SUFFIX_PATTERN below for the one place a comma is let
# through, on a much narrower shape than "any comma anywhere": a blanket
# allowance here would also validate genuine comma-separated address lines
# ("BREBES,JAWA TENGAH"), which on a card where the real name line failed
# OCR badly enough to miss NAME_MIN_SCORE would otherwise win by default -
# turning a safe "nothing found" (which correctly guardrails for review)
# into a confidently wrong answer.
NAME_LINE_PATTERN = re.compile(r"^[A-Za-z ./'-]+$")

# A card can glue an academic/professional title straight onto the name
# with a comma and no space (e.g. "UMIMAESAROH, S.PD") - without some
# comma exception that whole line fails NAME_LINE_PATTERN outright and is
# discarded as a candidate, not merely out-ranked by something else nearer.
# Indonesian title abbreviations are reliably short, single-token, and
# dotted ("S.PD", "S.H", "M.M", "S.KOM", "DR") - unlike a comma-separated
# address fragment, which is always multi-word ("JAWA TENGAH") or itself
# strung with further commas ("SAYUNG, DEMAK, JAWA TENGAH"). Matching that
# specific shape, rather than allowing any comma, is what keeps the two
# apart - see NAME_LINE_PATTERN's comment above for why that distinction
# matters.
NAME_TITLE_SUFFIX_PATTERN = re.compile(r"^([A-Za-z ./'-]+), ?([A-Za-z]{1,4}(?:\.[A-Za-z]{1,4}){0,2}\.?)$")

# Card boilerplate that can still match NAME_LINE_PATTERN (all-caps, no
# digits) but isn't a person's name - excluded by word, not by full line,
# since label text sometimes shares a line with real content. Deliberately
# excludes "PRATAMA": it's meant to catch the "KPP PRATAMA <city>" office
# line, but "Pratama" is also a very common real Indonesian given/middle
# name, so blocking it by itself throws out genuine name lines (e.g.
# "ADITYA ARDELLO PRATAMA"). "KPP" alone already excludes that office line,
# so "PRATAMA" is redundant for its original purpose and only adds a
# collision risk.
NAME_LABEL_WORDS = {
    "NPWP", "NOMOR", "POKOK", "WAJIB", "PAJAK", "KPP", "NAMA",
    "ALAMAT", "TERDAFTAR", "DIREKTORAT", "JENDERAL", "KEMENTERIAN",
    "KEUANGAN", "REPUBLIK", "INDONESIA", "ORANG", "PRIBADI", "NIK",
    "TANGGAL", "KANTOR", "PELAYANAN", "KARTU", "KAB", "KAB.", "JL",
    "KOTA", "PROVINSI", "KEC", "KEC.", "KEL", "KEL.", "WWW.PAJAK.GO.ID",
    "FASKES", "CAPTCHA", "DJP", "REFERAL", "REFERRAL", "DEVELOPER"
}

# The "NPWP" label itself, printed on its own OCR line right next to (or
# glued onto) the number, with four twists NAME_LABEL_WORDS' exact-string
# membership check can't catch on its own: a trailing period with nothing
# after it ("NPWP.", vs. e.g. "NPWP:12.345..." which stays on the number's
# own line and never becomes a name candidate), OCR reading the "W" as a
# "V" (visually similar in condensed/blurry fonts, and not covered by
# npwp.py's digit-homoglyph table since this is the letter label, not the
# digit run), OCR dropping the trailing "P" entirely ("npw" - a truncated
# read, not a misread character), and OCR inserting a stray space in the
# middle ("np vp", seen on PI2409D8SR - splits what NAME_LABEL_WORDS or a
# single-word check would otherwise catch into two harmless-looking
# "words"). The optional " ?" absorbs that last case; the final "P?" stays
# optional for the dropped-P case. Anchored full-string match against the
# whole candidate (not per-word) - a real name that's just
# "NPW"/"NPV"/"NPWP"/"NP VP" is not a thing.
NAME_LABEL_NPWP_PATTERN = re.compile(r"^NP ?[VW]P?\.?$")

# A subset of NAME_LABEL_WORDS that only ever appears as a prefix directly
# followed by a place/street name on the card (e.g. "KOTA TEGAL", "KAB
# BREBES", "JL SUDIRMAN", "KEC BREBES", "KEL KALIGANGSA WETAN"). When OCR
# drops the space, the merged token ("KOTATEGAL", "KELKALIGANGSA") no
# longer matches NAME_LABEL_WORDS exactly, so it's caught separately by
# prefix instead. Deliberately excludes short/ambiguous words like "NIK"
# and "KPP", which collide with real Indonesian names (e.g. "NIKITA",
# "NIKO") if matched by prefix.
NAME_LABEL_PREFIXES = ("KOTA", "KAB", "KAB.", "PROVINSI", "JL", "KEC", "KEC.", "KEL", "KEL.")
NAME_LABEL_PREFIX_PATTERN = re.compile(
    "^(?:" + "|".join(re.escape(prefix) for prefix in NAME_LABEL_PREFIXES) + ")[A-Z]"
)

# Older card layouts print the "NAMA" label glued onto the same OCR line as
# the value itself (e.g. "NAMA:HESTI DIAN ERLINDA", "NAMA : HESTI..."). Some
# split label and value into two separate OCR boxes instead, but still only
# print the value box as ": IM OKA MAHENDRA NR" - the colon stays glued to
# the value even though "NAMA" itself is a whole separate line/candidate
# (already excluded via NAME_LABEL_WORDS). Either way the colon isn't in
# NAME_LINE_PATTERN's character class, so without stripping it first the
# whole line would fail the pattern and get discarded - falling through to
# whatever letters-only line is next nearest, which is often an address
# fragment instead of the real name. The leading "NAMA" is optional here for
# exactly that reason: strip it when present, but a bare leading colon on
# its own is just as much a label artifact and never part of a real name.
NAME_LABEL_PREFIX_STRIP_PATTERN = re.compile(r"^(?:NAMA\s*)?[:.]\s*", re.IGNORECASE)

# Scanned cards often carry a diagonal watermark (e.g. "CamScanner") whose
# fragments get OCR'd into short, letters-only, non-boilerplate lines that
# otherwise look exactly like a single-word name. Recognition confidence is
# the one signal that reliably tells them apart - watermark fragments score
# well below genuine printed text.
NAME_MIN_SCORE = 0.9

# A candidate this short is essentially never a real name, even a
# mononym - crosschecked against every ground-truth name_gt value across
# the full dataset (6256 rows): none is 2 characters or fewer other than a
# handful of corrupted ground-truth entries ("NK", "RW", "MU" - fragments,
# not real names), while genuine short names ("ADI") run 3+. A 1-2
# character OCR fragment (e.g. "da") can still score high confidence and
# sit closer to the NPWP anchor than the real name line, winning purely on
# distance - excluding it from candidacy lets the next-nearest, actual
# name line win instead.
MIN_NAME_LENGTH = 3


def _is_all_uppercase(text: str) -> bool:
    """True if `text` has no lowercase ASCII letter - the printed name on
    every NPWP card template seen is uppercase, so a lowercase letter is a
    signal the line is something else entirely (e.g. a watermark fragment,
    see NAME_MIN_SCORE) rather than the genuine name, misread. Punctuation
    like "." or "'" (allowed in NAME_LINE_PATTERN for title suffixes and
    apostrophized names) carries no case, so it never affects this check
    either way."""
    return not any(ch.islower() for ch in text)


def normalize_slash_spacing(text: str) -> str:
    """Cards that print maiden/married names on one line join them with a
    bare "/" and inconsistent (or no) surrounding whitespace (e.g. "A/B",
    "A/ B") - normalize every "/" to exactly one space on each side."""
    return re.sub(r"\s*/\s*", " / ", text)


# Balinese given names are conventionally written as two words - a
# birth-order title ("I" for men) plus the name itself (e.g. "I GEDE",
# "I WAYAN", "I KETUT", "I MADE", "I NYOMAN") - but OCR sometimes fails to
# pick up the card's own printed space between them, merging the two into
# one token ("IGEDE", "IWAYAN", "IKETUT", "IMADE", "INYOMAN"). Restoring the
# space only for these five specific titles (rather than any word starting
# with "I") avoids incorrectly splitting real single words that happen to
# start with I, e.g. "IRWAN", "INDRA", "ISKANDAR".
_BALINESE_TITLE_PATTERN = re.compile(r"(?<![A-Za-z])I(GEDE|WAYAN|KETUT|MADE|KADEK|PUTU|NYOMAN)(?![A-Za-z])")


def normalize_balinese_title_spacing(text: str) -> str:
    """Split a merged Balinese birth-order title back off the name that
    follows it (e.g. "IGEDE HENDRA" -> "I GEDE HENDRA") - see
    _BALINESE_TITLE_PATTERN for why only these five titles are handled."""
    return _BALINESE_TITLE_PATTERN.sub(r"I \1", text)


# A company (rather than individual) taxpayer's name starts with the "PT"
# (Perseroan Terbatas / limited company) entity marker. Only the specific
# case of a period with no following space ("PT.COMPANY") gets normalized,
# by inserting the missing space - "PT COMPANY" (no period at all) is left
# alone rather than having a period inserted, since that's not something OCR
# dropped, just a formatting variant some cards print unpunctuated.
_PT_PREFIX_PATTERN = re.compile(r"^PT\.(?=\S)")


def normalize_pt_prefix_spacing(text: str) -> str:
    """Insert the missing space after "PT." when a company name's leading
    entity marker has a period glued directly onto the rest of the name
    (e.g. "PT.ABC INDUSTRI" -> "PT. ABC INDUSTRI") - see
    _PT_PREFIX_PATTERN. Leaves "PT COMPANY" (no period) and "PT. COMPANY"
    (already spaced) unchanged."""
    return _PT_PREFIX_PATTERN.sub("PT. ", text, count=1)


# A company (rather than individual) taxpayer's registered name always
# starts with a legal-entity marker printed right on the name line itself -
# "PT" (Perseroan Terbatas / limited company) or "CV" (Commanditaire
# Vennootschap / limited partnership). That makes it a stronger signal than
# mere proximity to the NPWP number: an address or KPP-office line
# elsewhere in the region can otherwise win on distance alone (see
# extract_name's docstring) even though the PT/CV line is unambiguously the
# actual name. Requires a following word (not just "PT" on its own, which
# would be indistinguishable from the two-letter fragment OCR noise
# NAME_MIN_SCORE already exists to filter).
_COMPANY_PREFIX_PATTERN = re.compile(r"^(?:PT|CV)\.?\s+\S", re.IGNORECASE)


def _clean_name_candidate(text: str) -> str | None:
    """Strip label/bullet noise off a candidate line and return what's left
    only if it still looks like a name - letters/spaces only, no
    card-boilerplate words. Returns None if the line doesn't qualify at
    all. Used by extract_name's candidate scan."""
    stripped = NAME_LABEL_PREFIX_STRIP_PATTERN.sub("", text.strip(), count=1)
    # Stray middle-dot/bullet marks (e.g. "KUSMULYONO·") show up on some
    # scans as OCR noise glued onto the end of an otherwise-clean name -
    # not part of NAME_LINE_PATTERN's character class, so left in place
    # they'd fail the whole line instead of just being ignored.
    stripped = stripped.strip("·• ")

    # A title-suffixed line ("UMIMAESAROH, S.PD") validates its name
    # portion only - see NAME_TITLE_SUFFIX_PATTERN for why this shape (and
    # not a comma anywhere) is safe to let through.
    title_match = NAME_TITLE_SUFFIX_PATTERN.match(stripped)
    name_part = title_match.group(1) if title_match else stripped

    words = name_part.split()
    if not words or not NAME_LINE_PATTERN.match(name_part):
        return None
    if len(name_part) < MIN_NAME_LENGTH:
        return None
    # A PT/CV line is unambiguously the registered company name by virtue
    # of its explicit legal-entity marker - skip the label-word/prefix
    # checks below for it, since a real company name can legitimately
    # contain a word that's boilerplate everywhere else on the card (e.g.
    # "PT ... INDONESIA", a very common company-name suffix, but
    # "INDONESIA" is also blacklisted to filter out the letterhead's own
    # "REPUBLIK INDONESIA" line).
    if _COMPANY_PREFIX_PATTERN.match(stripped):
        return stripped
    if any(word.upper() in NAME_LABEL_WORDS for word in words):
        return None
    if any(NAME_LABEL_PREFIX_PATTERN.match(word.upper()) for word in words):
        return None
    if NAME_LABEL_NPWP_PATTERN.match(name_part.upper()):
        return None
    return stripped


def extract_name(
    res,
    apply_balinese_normalization: bool = False,
    apply_master_correction: bool = True,
    apply_formatting_normalization: bool = True,
) -> tuple[str, float] | None:
    """Find the taxpayer name line in an OCR result: it's consistently the
    nearest name-shaped line to the NPWP number, within the bounded region
    of the page that's actually the NPWP card (see find_document_region).
    Position still works across rotated/zoomed/cropped photos because
    rec_polys are in doc-preprocessor-corrected space
    (use_doc_orientation_classify + use_textline_orientation normalize
    up/down before this ever sees the result), so we don't need the
    original photo to be upright to begin with. Candidates are filtered to
    letters-only lines with no card-boilerplate words (single-word names,
    e.g. Indonesian mononyms, are allowed), since position alone would also
    match address or KPP office lines further down the card.

    Uploads sometimes bundle a second document (KTP/KK/Akta Nikah) on the
    same page as the real NPWP card - e.g. a KTP cropped without its own
    "KARTU TANDA PENDUDUK" title, which the page-level guardrail can't
    catch. find_document_region bounds the search to the NPWP card's own
    region so a bundled document's "Nama"/"Nama Kepala Keluarga" field
    can't outrank the real name just for being closer in reading order.

    The nearest candidate is picked by absolute distance to the NPWP
    number, not "nearest below" - some templates print the name above the
    number rather than below it (e.g. logo, then name, then number), and a
    below-only rule would skip a same-region, correctly-anchored name in
    favor of an unrelated line further down.

    When the NPWP number itself wasn't detected on this page, there's no
    anchor to measure distance from - rather than give up on the name too
    (detection is best-effort per field, not all-or-nothing), fall back to
    the topmost name-shaped candidate in the region. That's a safe stand-in
    because every observed card layout prints the name above the address
    block.

    Newer card layouts print the 16-digit NIK (national ID) number a few
    lines below the name - which also matches NPWP_PATTERN's digit-run
    shape (see npwp.py: NPWP_PATTERN recognizes both the legacy 15-digit
    and the newer 16-digit NPWP formats, and the two happen to look the
    same). The printed order is consistently NPWP number -> name -> NIK
    number -> address -> KPP office, so when a second NPWP-shaped line
    turns up below the anchor, candidates at or past it (address
    fragments, the KPP office name) are excluded rather than left to
    compete on distance alone - which otherwise lets a farther office/city
    line win outright when the real name gets disqualified for some other
    reason (e.g. a title suffix briefly failing NAME_LINE_PATTERN).

    Returns the matched line's own rec_score alongside its text, so callers
    can surface OCR confidence for the name the same way they already do
    for the NPWP number.

    `apply_balinese_normalization` gates only the Balinese title-spacing
    fix (see normalize_balinese_title_spacing) on the winning candidate.
    `apply_master_correction` gates only the name-master spacing
    correction (see name_master.correct_name_spacing). Both default to
    what the primary/"modified" output uses.

    `apply_formatting_normalization` gates the slash/PT-prefix spacing
    fixes (normalize_slash_spacing, normalize_pt_prefix_spacing) as one
    group - these are deterministic, rule-based formatting fixes rather
    than reference-list corrections, so they default to on even for an
    otherwise "base" (uncorrected) call."""
    rec_texts = res.get("rec_texts", [])
    rec_polys = res.get("rec_polys", [])
    rec_scores = res.get("rec_scores", [])
    if not rec_texts or not rec_polys:
        return None

    y_min, y_max = find_document_region(rec_texts, rec_polys)

    npwp_idx = _select_npwp_anchor(rec_texts, rec_polys, y_min, y_max)
    npwp_y = poly_center(rec_polys[npwp_idx])[1] if npwp_idx is not None else None

    # The nearest other NPWP-shaped line below the anchor, if any - see the
    # NIK explanation above. Only looks below (not above) since the NIK
    # always prints after the name, never before it.
    nik_y = None
    if npwp_y is not None:
        below_anchor_matches = [
            poly_center(poly)[1]
            for i, (text, poly) in enumerate(zip(rec_texts, rec_polys))
            if i != npwp_idx
            and NPWP_PATTERN.search(text)
            and (y_min is None or poly_center(poly)[1] >= y_min)
            and (y_max is None or poly_center(poly)[1] <= y_max)
            and poly_center(poly)[1] > npwp_y
        ]
        if below_anchor_matches:
            nik_y = min(below_anchor_matches)

    candidates = []
    for i, (text, poly) in enumerate(zip(rec_texts, rec_polys)):
        if i == npwp_idx:
            continue
        stripped = _clean_name_candidate(text)
        if stripped is None:
            continue
        score = rec_scores[i] if i < len(rec_scores) else None
        if score is not None and score < NAME_MIN_SCORE:
            continue

        _, center_y = poly_center(poly)
        if y_min is not None and center_y < y_min:
            continue
        if y_max is not None and center_y > y_max:
            continue

        distance = abs(center_y - npwp_y) if npwp_y is not None else center_y
        candidates.append((distance, stripped, score, poly, center_y))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    # A PT/CV company-name line outranks anything else in the region
    # regardless of distance to the anchor - see _COMPANY_PREFIX_PATTERN.
    # Still broken by distance among themselves, for the rare case of more
    # than one (e.g. a second PT/CV-prefixed line elsewhere on the card).
    company_candidates = [c for c in candidates if _COMPANY_PREFIX_PATTERN.match(c[1])]
    if company_candidates:
        pool = company_candidates
    else:
        # Prefer candidates strictly between the NPWP anchor and the NIK
        # line (see the NIK explanation above) when that band exists and
        # actually contains at least one candidate - falls back to the
        # unrestricted region otherwise, so a layout without a detected
        # NIK line (or where the real name doesn't fall in that band for
        # some other reason) isn't newly broken by this restriction.
        bounded = [c for c in candidates if nik_y is not None and npwp_y < c[4] < nik_y]
        pool = bounded if bounded else candidates
        # Among multiple remaining candidates, an all-uppercase one
        # outranks one with a lowercase letter - see _is_all_uppercase for
        # why the printed name itself is never mixed-case, so a lowercase
        # letter marks a candidate as more likely OCR noise than the
        # genuine name. Checked before the recognized-name tie-break below
        # since it doesn't depend on the master list covering this name.
        # Only narrows the pool when at least one candidate qualifies, same
        # reasoning as the recognized-name check.
        if len(pool) > 1:
            uppercase_only = [c for c in pool if _is_all_uppercase(c[1])]
            if uppercase_only:
                pool = uppercase_only
        # Among multiple remaining candidates, one recognized by the
        # name-master reference list (see name_master.is_recognized_name)
        # outranks one that isn't - a stray OCR fragment (background
        # bleed-through, watermark noise) sitting closer to the anchor
        # than the real name is otherwise indistinguishable from it by
        # position alone. Only narrows the pool when at least one
        # candidate is actually recognized - the master doesn't cover
        # every real name, so "none recognized" carries no signal and
        # distance-to-anchor stays the deciding factor, same as before.
        if len(pool) > 1:
            recognized = [c for c in pool if is_recognized_name(c[1])]
            if recognized:
                pool = recognized
    _, name, score, _, _ = pool[0]

    if apply_formatting_normalization:
        name = normalize_slash_spacing(name)
        if apply_balinese_normalization:
            name = normalize_balinese_title_spacing(name)
        name = normalize_pt_prefix_spacing(name)
    if apply_master_correction:
        name = correct_name_spacing(name)
    return name, score


def _select_npwp_anchor(rec_texts, rec_polys, y_min: float | None, y_max: float | None) -> int | None:
    """Pick which NPWP-shaped match to anchor name search on: the first one
    that falls inside the document region, so a bundled second document's
    own NPWP-shaped number (e.g. a KTP's 16-digit NIK) can't get used as
    the anchor just for appearing earlier in reading order. Falls back to
    the very first match overall if none are in-region - e.g. when the
    region itself couldn't be determined (see find_document_region)."""
    first = None
    for i, (text, poly) in enumerate(zip(rec_texts, rec_polys)):
        if not NPWP_PATTERN.search(text):
            continue
        if first is None:
            first = i
        y = poly_center(poly)[1]
        if y_min is not None and y < y_min:
            continue
        if y_max is not None and y > y_max:
            continue
        return i
    return first
