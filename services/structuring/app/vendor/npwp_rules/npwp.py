import itertools
import re

import numpy as np

from .kpp_codes import is_valid_kpp_code
from .wilayah_codes import is_valid_kecamatan_code

# OCR occasionally misreads a single digit as a visually similar letter in
# an otherwise digit-shaped run (e.g. "1Z30" for "1230", "9O" for "90") -
# this class matches "digit-or-confusable-letter" so NPWP_PATTERN still
# recognizes the shape despite the typo; normalize_npwp() maps each
# confusable letter back to its digit afterwards. Kept deliberately small
# (only letters that are genuinely easy to confuse with one digit each) to
# avoid matching ordinary text that merely contains some digits and letters.
#
# "T" is included in the matching class but deliberately left OUT of the
# blind translation table below: unlike every other entry here (each reads
# as exactly one digit, in any position), "T" is genuinely two-way
# confusable - it can be a misread "1" or a misread "7" depending on
# font/scan quality, so guessing either one unconditionally would be wrong
# roughly half the time. See PROVINCE_CODES / _resolve_province_prefix for
# how it's actually resolved: positionally, only where the number's own
# structure pins it down, never as a blanket rule.
_DIGIT_LIKE = "0-9OoSsIlBZzT"
_DIGIT_HOMOGLYPHS = str.maketrans({
    "O": "0", "o": "0",
    "S": "5", "s": "5",
    "I": "1", "l": "1",
    "B": "8",
    "Z": "2", "z": "2",
})

# Indonesian NIK numbers - and by extension the newer 16-digit NIK-based
# NPWP format, which reuses the same numbering - open with a 2-digit
# province code. Source: Kemendagri's official Kode Wilayah table. Used
# only to disambiguate "T" (see _DIGIT_LIKE above): if "T" lands in one of
# a 16-digit match's first two digits, only one of "T"->"1" or "T"->"7"
# completes an existing province code, and _resolve_province_prefix uses
# that to pick the right one instead of guessing.
PROVINCE_CODES = {
    "11": "Aceh", "12": "Sumatera Utara", "13": "Sumatera Barat", "14": "Riau",
    "15": "Jambi", "16": "Sumatera Selatan", "17": "Bengkulu", "18": "Lampung",
    "19": "Bangka Belitung", "21": "Kepulauan Riau", "31": "DKI Jakarta",
    "32": "Jawa Barat", "33": "Jawa Tengah", "34": "DI Yogyakarta", "35": "Jawa Timur",
    "36": "Banten", "51": "Bali", "52": "Nusa Tenggara Barat", "53": "Nusa Tenggara Timur",
    "61": "Kalimantan Barat", "62": "Kalimantan Tengah", "63": "Kalimantan Selatan",
    "64": "Kalimantan Timur", "65": "Kalimantan Utara", "71": "Sulawesi Utara",
    "72": "Sulawesi Tengah", "73": "Sulawesi Selatan", "74": "Sulawesi Tenggara",
    "75": "Gorontalo", "76": "Sulawesi Barat", "81": "Maluku", "82": "Maluku Utara",
    "91": "Papua", "92": "Papua Barat",
}

_AMBIGUOUS_DIGIT = "T"


def _resolve_province_prefix(digit_like: str) -> str:
    """Given a digit-like string (letters not yet translated) whose first
    two characters are meant to be a NIK/16-digit-NPWP province code,
    replace an ambiguous "T" in EITHER of those two positions with
    whichever of "1"/"7" completes a real code from PROVINCE_CODES. Any
    other homoglyph letter in the prefix is translated first (e.g. the "S"
    in "ST71" -> "5", so "5T" resolves to "51" Bali, not "57"). Position 1
    is confirmed against a real scan ("T701..." on PK24055W7U, true leading
    digit "1"); position 2 against "ST71 0302 0880 0005" on PN2510K1RI (Kota
    Denpasar, Bali). Leaves the string untouched (for normalize_npwp's
    final \\D-strip to drop the "T", same as any other unresolvable
    character) if there's no "T" in the prefix, or if zero or more than one
    substitution yields a valid code - no signal means no guess, same
    stance name_master.py takes on ambiguous name matches."""
    prefix = digit_like[:2].translate(_DIGIT_HOMOGLYPHS)
    if _AMBIGUOUS_DIGIT not in prefix:
        return digit_like
    valid = [
        code
        for code in (
            "".join(choice)
            for choice in itertools.product(*(("1", "7") if ch == _AMBIGUOUS_DIGIT else (ch,) for ch in prefix))
        )
        if code in PROVINCE_CODES
    ]
    if len(valid) != 1:
        return digit_like
    return valid[0] + digit_like[2:]

# Two valid NPWP shapes: the legacy 15-digit XX.XXX.XXX.X-XXX.XXX, and the
# newer 16-digit NIK-based format - printed as four space-separated groups
# of four digits (XXXX XXXX XXXX XXXX). OCR doesn't reliably preserve all
# three of those inter-group spaces - it drops whichever ones it likes
# (e.g. "000000000000 0000", first three groups merged but the last space
# survives), including sometimes all of them (a solid 16-digit run), so each
# of the three gaps is optional independently rather than requiring exactly
# zero or exactly all three spaces. The lookaround keeps it from matching
# inside a longer, unrelated digit run (e.g. a 20-digit QR/barcode payload
# OCR'd as one token) - it uses the same digit-or-confusable-letter class as
# the groups themselves, so a homoglyph typo right at the boundary can't
# hide a 17-digit run from the lookaround and cause a false match.
NPWP_PATTERN = re.compile(
    rf"[{_DIGIT_LIKE}]{{2}}\.[{_DIGIT_LIKE}]{{3}}\.[{_DIGIT_LIKE}]{{3}}\.[{_DIGIT_LIKE}]-[{_DIGIT_LIKE}]{{3}}\.[{_DIGIT_LIKE}]{{3}}"
    rf"|(?<![{_DIGIT_LIKE}])[{_DIGIT_LIKE}]{{4}} ?[{_DIGIT_LIKE}]{{4}} ?[{_DIGIT_LIKE}]{{4}} ?[{_DIGIT_LIKE}]{{4}}(?![{_DIGIT_LIKE}])"
)


# Boilerplate printed on other document types (KTP, KK) that sometimes get
# bundled into the same upload as an NPWP card. If any page shows this, the
# upload contains more than one document and must be rejected outright.
OTHER_DOCUMENT_KEYWORDS = (
    "KARTU KELUARGA",
    "KARTU TANDA PENDUDUK",
    "Kartu Indonesia Sehat",
    "KUTIPAN AKTA NIKAH",
    "Berlaku Hingga"
)

# A genuine NPWP upload is at most 2 pages (front+back, or husband+wife on
# separate pages). More than that is very likely a bundled second document
# that landed on its own page instead of sharing a page with the real NPWP
# card - confirmed empirically against the npwp_merged dataset: of the
# files with more than 2 pages, 65/78 (83%) carried an actual
# OTHER_DOCUMENT_KEYWORDS boilerplate match on at least one page (see
# GEN/ocr_npwp/app/audit_multi_page.py). Checked purely on page count so it
# still catches the remaining cases where OCR fails to legibly read that
# bundled document's own boilerplate.
MAX_EXPECTED_PAGES = 2


def find_other_document_keyword(rec_texts: list[str]) -> str | None:
    """Return the first OTHER_DOCUMENT_KEYWORDS boilerplate phrase found in
    this page's OCR text (e.g. KTP, KK), or None if none is present. A
    match means the upload bundles more than one document and should be
    rejected - the returned keyword is the reason to report for that."""
    joined = " ".join(rec_texts).upper()
    return next((keyword for keyword in OTHER_DOCUMENT_KEYWORDS if keyword.upper() in joined), None)


# A CAPTCHA challenge in the OCR text means the page isn't an NPWP card at
# all (e.g. the source system intercepted the request with a verification
# screen) - not a standard NPWP layout, so it must be rejected outright.
CAPTCHA_KEYWORDS = ("CAPTCHA",)


def contains_captcha(rec_texts: list[str]) -> bool:
    """True if this page's OCR text shows a CAPTCHA/verification challenge
    instead of actual NPWP card content."""
    joined = " ".join(rec_texts).upper()
    return any(keyword in joined for keyword in CAPTCHA_KEYWORDS)


# Screenshots of the DJP "Cek NPWP" web lookup tool aren't a photographed
# physical card - they render the same fields in a table layout and mask
# the actual name ("disamarkan" = obscured), so no real name can ever be
# recovered from this layout. Reject outright rather than returning
# whatever nearby table cell/header the position heuristic happens to pick.
WEB_LOOKUP_KEYWORDS = ("CEK NPWP", "DISAMARKAN")


def contains_web_lookup_screenshot(rec_texts: list[str]) -> bool:
    """True if this page's OCR text shows a screenshot of the DJP web NPWP
    lookup tool instead of an actual NPWP card."""
    joined = " ".join(rec_texts).upper()
    return any(keyword in joined for keyword in WEB_LOOKUP_KEYWORDS)


def poly_center(poly) -> tuple[float, float]:
    center = np.asarray(poly, dtype=float).mean(axis=0)
    return float(center[0]), float(center[1])


# Boilerplate that identifies the DJP-issued NPWP content itself, present on
# every card template seen so far - though never the same one twice, since
# templates vary (older cards spell out "DIREKTORAT JENDERAL PAJAK", newer
# ones just show a "npwp."/"djp" logo + KPP office line). Any single hit is
# enough to mark where the real NPWP content starts; that's what makes this
# useful as a boundary against a bundled KTP/KK/Akta Nikah sharing the same
# page/upload, which prints its own unrelated letterhead instead.
DOCUMENT_ANCHOR_KEYWORDS = (
    "NPWP", "DJP", "KPP", "KANTOR PELAYANAN PAJAK",
    "DIREKTORAT JENDERAL PAJAK", "KEMENTERIAN KEUANGAN",
)


# "PROVINSI" opens a KTP's own header line (e.g. "PROVINSI JAWA BARAT") on
# every bundled-KTP sample seen, including ones cropped without a
# "KARTU TANDA PENDUDUK" title - which OTHER_DOCUMENT_KEYWORDS can't catch,
# since it only fires on that title text. But it's not safe to reject a
# whole upload on "PROVINSI" the way OTHER_DOCUMENT_KEYWORDS does: some
# genuine single-document NPWP cards spell their own address out as the
# full Desa/Kec/Kab/Provinsi administrative hierarchy, so the word shows up
# there too (see PG2404LGRU: "...KAB. MOROWALI" / "PROVINSI SULAWESI
# TENGAH" as the card's own address, no bundling at all). What makes it
# safe here specifically is that it's *only* used to trim the lower bound
# of the search region, never to reject the page outright - the NPWP
# number and name are always printed above the address block on every
# template seen, "PROVINSI" included, so even a false hit on a genuine
# card's own address only trims off trailing content (KPP line, logo,
# registration date) that name/number extraction was never going to pick
# anyway.
REGION_LOWER_BOUND_KEYWORDS = OTHER_DOCUMENT_KEYWORDS + ("PROVINSI",)


def find_document_region(rec_texts: list[str], rec_polys) -> tuple[float | None, float | None]:
    """Return (y_min, y_max) bounding the actual NPWP-card content on a page
    that might also carry a bundled second document (see
    OTHER_DOCUMENT_KEYWORDS) - e.g. a KTP/KK printed above or below the real
    NPWP card in the same upload. y_min is the topmost line matching a DJP
    boilerplate anchor; y_max is the topmost *other-document* keyword found
    below y_min, marking where a second bundled document begins. Either
    bound comes back None when nothing matches, meaning "no restriction on
    that side" - about 5% of real cards (tight crops, heavy OCR noise) never
    legibly show any anchor phrase, so callers should treat a None bound as
    today's whole-page behavior rather than excluding everything.

    An anchor only counts toward y_min if it sits at or above the
    bottommost NPWP-shaped number on the page (not the topmost - a page
    bundling a KTP above the real NPWP card has its earliest NPWP-shaped
    match, the KTP's own NIK, in the wrong document entirely, so comparing
    against that would reject the real card's own anchors too). Anchor
    position isn't consistent across templates - most print KPP/DJP
    branding as a header, but some (e.g. a "djp" watermark tucked in a
    corner) print it *below* the number and name instead, the same way
    INDAH NATALIA's card layout printed "KPP PRATAMA CIKUPA" after the
    address rather than before it: real content, just not at the top.
    Letting an anchor that trails every number on the page push y_min
    downward would exclude the very name it's supposed to help bound."""
    npwp_ys = [poly_center(poly)[1] for text, poly in zip(rec_texts, rec_polys) if NPWP_PATTERN.search(text)]
    last_npwp_y = max(npwp_ys) if npwp_ys else None

    y_min = None
    for text, poly in zip(rec_texts, rec_polys):
        upper = text.upper()
        if any(keyword in upper for keyword in DOCUMENT_ANCHOR_KEYWORDS):
            y = poly_center(poly)[1]
            if last_npwp_y is not None and y > last_npwp_y:
                continue
            if y_min is None or y < y_min:
                y_min = y

    y_max = None
    for text, poly in zip(rec_texts, rec_polys):
        upper = text.upper()
        if any(keyword.upper() in upper for keyword in REGION_LOWER_BOUND_KEYWORDS):
            y = poly_center(poly)[1]
            if y_min is not None and y <= y_min:
                continue
            if y_max is None or y < y_max:
                y_max = y

    return y_min, y_max


def _normalize_npwp_run(digit_like: str) -> str:
    """Resolve province-prefix ambiguity (if this run is 16 digit-like
    characters) and translate the safe homoglyphs, on a single already-
    isolated digit-like run - the shared final step both normalize_npwp
    call paths below use, so a 16-char match gets identical treatment
    whether it arrived alone or embedded in a larger blob."""
    if len(digit_like) == 16:
        digit_like = _resolve_province_prefix(digit_like)
    return re.sub(r"\D", "", digit_like.translate(_DIGIT_HOMOGLYPHS))


def normalize_npwp(text: str) -> str:
    """Strip the formatting punctuation/spaces off a matched NPWP-shaped
    string (dots, dashes, the 4-group space layout) and map any OCR
    digit/letter homoglyphs (see _DIGIT_HOMOGLYPHS) back to the digit they
    stand for, leaving only digits - the canonical form callers should
    store/return.

    Finds every NPWP_PATTERN-shaped match in `text` first and normalizes
    each independently, rather than isolating digit-like characters across
    the whole input as one run - the latter only works when the caller has
    already handed in a single pre-isolated match (a lone 16-char run),
    which is true for most existing callers but not for one that searches
    a whole raw OCR text dump for a value's presence: there, an ambiguous
    "T" (see PROVINCE_CODES/_resolve_province_prefix) can sit anywhere in
    a 100+-character blob of unrelated digits from dates, addresses, and
    other numbers on the page, so the whole-input digit-like run is never
    exactly 16 characters and _resolve_province_prefix silently never
    fires. Normalizing each match separately keeps that check correctly
    scoped to just the 16 characters it's meant to apply to, regardless of
    how much surrounding text there is. Multiple matches are joined with a
    space (not concatenated bare) so a substring search downstream can't
    accidentally straddle two unrelated numbers at their boundary. Falls
    back to normalizing the whole input as a single run only when no
    NPWP_PATTERN match is found in it at all (e.g. `text` is itself
    already just a bare, non-standard-shaped digit fragment)."""
    matches = [m.group() for m in NPWP_PATTERN.finditer(text)]
    if not matches:
        return _normalize_npwp_run(re.sub(rf"[^{_DIGIT_LIKE}]", "", text))
    return " ".join(_normalize_npwp_run(re.sub(rf"[^{_DIGIT_LIKE}]", "", m)) for m in matches)


def normalize_npwp_raw(text: str) -> str:
    """Same as normalize_npwp but WITHOUT the digit-homoglyph translation -
    strips formatting punctuation only, so a misread letter (e.g. the "O"
    in "63.48O.341...") is simply dropped as non-digit rather than
    corrected back to "0". Lets a caller compare what the homoglyph
    correction step actually changes for a given match."""
    return re.sub(r"\D", "", text)


def normalize_npwp_literal(text: str) -> str:
    """Find every NPWP_PATTERN-shaped match in `text` (the same
    homoglyph-tolerant shape recognition normalize_npwp uses to decide
    what counts as "a 15/16-digit run" in the first place) and strip only
    formatting punctuation/spaces from each - unlike normalize_npwp
    (translates a homoglyph letter to the digit it stands for) and
    normalize_npwp_raw (drops it as non-digit), this keeps any misread
    letter exactly as OCR produced it, untouched.

    A letter left in place can never equal the ground truth's pure-digit
    string, so this is what a strict "was this exact digit sequence
    captured" check needs: a digit OCR'd as a letter - or any other
    captured-wrong digit - correctly reads as not-extracted, rather than
    the letter being silently dropped first (normalize_npwp_raw), which
    could coincidentally leave behind a shorter run that still
    substring-matches something it shouldn't, or silently corrected
    (normalize_npwp) - that tolerance belongs to the loose check, not this
    one. Matches are joined with a space, same reasoning as normalize_npwp
    (so a substring search downstream can't straddle two unrelated
    numbers at their boundary). Falls back to normalizing the whole input
    as a single run only when no NPWP_PATTERN match is found in it at
    all."""
    def _preserve(raw: str) -> str:
        return re.sub(r"[^0-9A-Za-z]", "", raw)

    matches = [m.group() for m in NPWP_PATTERN.finditer(text)]
    if not matches:
        return _preserve(text)
    return " ".join(_preserve(m) for m in matches)


def contains_digit_homoglyph(text: str) -> bool:
    """True if this matched NPWP-shaped string has at least one digit
    position OCR read as a letter (e.g. 'O' for '0', see _DIGIT_HOMOGLYPHS)
    rather than a plain digit - i.e. normalize_npwp() had to correct
    something to arrive at the final number. Checked against the raw match
    text (before normalize_npwp's translation), since the corrected/
    normalized string never contains letters by construction."""
    return text.translate(_DIGIT_HOMOGLYPHS) != text


def has_invalid_province_prefix(text: str) -> bool:
    """True if this matched NPWP-shaped string is the 16-digit NIK-based
    format and its first two digits, after the same homoglyph correction
    and "T" disambiguation normalize_npwp() applies, don't correspond to
    any real code in PROVINCE_CODES - e.g. a cleanly-OCR'd "77..." prefix,
    which involves no ambiguous letter and so gives
    _resolve_province_prefix nothing to act on. This is a review flag, not
    a correction: unlike _resolve_province_prefix (which only substitutes
    a digit when exactly one candidate is valid), there's no safe guess to
    make here - the mismatch might be a genuine misread, or a real code
    this table is missing. Always False for the legacy 15-digit format,
    whose first two digits are a taxpayer-type code, not a province code.
    Also False when a "T" in the prefix was left unresolved (zero or more
    than one valid completion) - that ambiguity is already the caller's
    signal via the "T" surviving in the raw match text, and normalize_npwp
    itself will end up dropping it, so this check doesn't pile a second,
    misleading flag on the same 16th-character shortfall."""
    digit_like = re.sub(rf"[^{_DIGIT_LIKE}]", "", text)
    if len(digit_like) != 16:
        return False
    normalized = _normalize_npwp_run(digit_like)
    if len(normalized) != 16:
        return False
    return normalized[:2] not in PROVINCE_CODES


def has_invalid_kecamatan_prefix(text: str) -> bool:
    """True if this matched NPWP-shaped string is the 16-digit NIK-based
    format and its first six digits (province+kabupaten/kota+kecamatan),
    after the same normalization has_invalid_province_prefix applies,
    don't correspond to any real kecamatan code in the Depdagri reference
    table (see wilayah_codes.is_valid_kecamatan_code) - a strictly broader
    check than has_invalid_province_prefix's, since a code that exists at
    the kecamatan level necessarily has a valid province and kabupaten/kota
    prefix too (build_wilayah_codes.py's own parse-time sanity checks
    guarantee that). Kept as a separate flag rather than replacing
    has_invalid_province_prefix so the reference-data-unavailable case
    (is_valid_kecamatan_code returns None) degrades independently of the
    always-available 2-digit check: False here (like that function) means
    "no signal", not "confirmed valid" - see is_valid_kecamatan_code."""
    digit_like = re.sub(rf"[^{_DIGIT_LIKE}]", "", text)
    if len(digit_like) != 16:
        return False
    normalized = _normalize_npwp_run(digit_like)
    if len(normalized) != 16:
        return False
    return is_valid_kecamatan_code(normalized[:6]) is False


def has_invalid_birthdate_digits(text: str) -> bool:
    """True if this matched NPWP-shaped string is the 16-digit NIK-based
    format and digits 7-12 (the NIK's embedded birthdate, DDMMYY) don't
    form a plausible date - day (digits 7-8) outside 01-31, or outside
    41-71 for a female NIK (the standard NIK convention adds 40 to the day
    of month for women), or month (digits 9-10) outside 01-12. The year
    (digits 11-12) is deliberately left unchecked: a bare two-digit year is
    genuinely ambiguous between 19xx/20xx with nothing else on the card to
    resolve it, so unlike day/month there's no value in that pair that's
    actually out of range - every 00-99 is a plausible birth year. Same
    normalization as has_invalid_province_prefix/has_invalid_kecamatan_prefix;
    always False for the legacy 15-digit format, which has no NIK-style
    birthdate encoded in it at all."""
    digit_like = re.sub(rf"[^{_DIGIT_LIKE}]", "", text)
    if len(digit_like) != 16:
        return False
    normalized = _normalize_npwp_run(digit_like)
    if len(normalized) != 16:
        return False
    day = int(normalized[6:8])
    month = int(normalized[8:10])
    valid_day = 1 <= day <= 31 or 41 <= day <= 71
    valid_month = 1 <= month <= 12
    return not (valid_day and valid_month)


def has_invalid_kpp_prefix(text: str) -> bool:
    """True if this matched NPWP-shaped string is the legacy 15-digit
    format (AA.BBB.CCC.D-EEE.FFF) and its digits 10-12 (EEE, the
    registering KPP office's code) don't correspond to any real code in
    the Depdagri/DJP reference table (see kpp_codes.is_valid_kpp_code) -
    unlike has_invalid_province_prefix/has_invalid_kecamatan_prefix, this
    only applies to the 15-digit format: the 16-digit NIK-based format has
    no KPP code encoded in it at all, its digits 10-12 are part of the
    NIK's own serial number instead. Same normalization approach as the
    other has_invalid_* checks - always False for a 16-digit match, and
    False (not a confirmed-valid signal) if a "T" in the digit run was
    left unresolved (dropped by the final \\D-strip, shortening the
    normalized string below 15) or if the reference data couldn't be
    loaded (see is_valid_kpp_code)."""
    digit_like = re.sub(rf"[^{_DIGIT_LIKE}]", "", text)
    if len(digit_like) != 15:
        return False
    normalized = _normalize_npwp_run(digit_like)
    if len(normalized) != 15:
        return False
    return is_valid_kpp_code(normalized[9:12]) is False


# Counts digit-LIKE characters (see _DIGIT_LIKE) rather than str.isdigit(),
# which returns False for the confusable letters NPWP_PATTERN itself
# already treats as digits when matching. Using isdigit() here used to
# silently undercount any 16-digit match that still had an untranslated
# homoglyph letter in it (e.g. "3329 1Z30 1001 0006" -> 15, not 16),
# making npwp_priority() lose that match's rightful 16-digit priority tier
# to a 15-digit legacy match on OCR-confidence tie-break alone - even
# though normalize_npwp() would have translated it to the correct 16-digit
# number just fine. See npwp_priority's docstring for why the tier itself
# matters.
_DIGIT_LIKE_PATTERN = re.compile(f"[{_DIGIT_LIKE}]")


def npwp_priority(text: str, score: float) -> tuple[bool, float]:
    """Sort key for picking the canonical NPWP when a card yields multiple
    NPWP-shaped matches: the 16-digit NIK-based format always outranks the
    legacy 15-digit one, since a card showing both (the 15-digit number in
    large print, the 16-digit one below it under an "NPWP16" label) means
    the 16-digit one is the current official number superseding the other.
    Some cards have already switched to the 16-digit format as their only/
    main number, in which case it's the sole candidate and still wins.
    OCR confidence only breaks ties between matches of the same format."""
    digit_count = len(_DIGIT_LIKE_PATTERN.findall(text))
    return (digit_count == 16, score)
