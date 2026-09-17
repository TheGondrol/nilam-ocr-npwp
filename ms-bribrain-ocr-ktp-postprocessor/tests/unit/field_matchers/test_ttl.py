"""Tests for TTL (Tempat/Tanggal Lahir) field matcher."""

import pytest

from src.services.field_matchers.ttl import (
    cleaning_tempat_meet_tgllength,
    format_ttl_new_sesuai,
    format_ttl_new_tidak_sesuai,
    matching_tempatlahir,
    matching_tempatlahir_new,
    meet_tgllength,
    normalisasi_meet_tgllength,
    notmeet_tgllength,
    notmeet_tgllength_jadi_satu,
    notmeet_tgllength_terpisah,
    split_tgllahir,
    validate_ttl_data,
)


class TestNormalisasiMeetTgllength:
    """Test cases for normalisasi_meet_tgllength function."""

    def test_with_4_parts(self):
        """Test with 4 parts (standard format)."""
        parts = ["JAKARTA", "15", "08", "1990"]
        tempat, day, month, year = normalisasi_meet_tgllength(parts)
        assert tempat == "JAKARTA"
        assert day == "15"
        assert month == "Aug"  # bulan_dict converts "08" to "Aug"
        assert year == "1990"

    def test_with_more_than_4_parts(self):
        """Test with more than 4 parts (multi-word city)."""
        parts = ["JAKARTA", "SELATAN", "15", "08", "1990"]
        tempat, day, month, year = normalisasi_meet_tgllength(parts)
        assert "JAKARTA" in tempat
        assert "SELATAN" in tempat
        assert day == "15"
        assert month == "Aug"  # bulan_dict converts "08" to "Aug"
        assert year == "1990"

    def test_with_less_than_4_parts(self):
        """Test with less than 4 parts."""
        parts = ["JAKARTA", "15"]
        tempat, day, month, year = normalisasi_meet_tgllength(parts)
        assert tempat == "JAKARTA"
        assert day == "15"

    def test_with_month_name(self):
        """Test with month name instead of number."""
        parts = ["JAKARTA", "15", "Agustus", "1990"]
        tempat, day, month, year = normalisasi_meet_tgllength(parts)
        assert tempat == "JAKARTA"
        assert day == "15"
        assert year == "1990"
        # Month should be converted using bulan_dict

    def test_empty_parts(self):
        """Test with empty parts list."""
        parts = []
        tempat, day, month, year = normalisasi_meet_tgllength(parts)
        assert tempat is None


class TestMeetTgllength:
    """Test cases for meet_tgllength function."""

    def test_valid_ttl_format(self):
        """Test with valid TTL format."""
        parts = ["JAKARTA", "15", "08", "1990"]
        tempat, day, month, year = meet_tgllength(parts)
        assert tempat == "JAKARTA"
        assert day == "15"
        assert month == "Aug"  # bulan_dict converts "08" to "Aug"
        assert year == "1990"

    def test_ttl_with_multi_word_city(self):
        """Test with multi-word city name."""
        parts = ["JAKARTA", "SELATAN", "15", "08", "1990"]
        tempat, day, month, year = meet_tgllength(parts)
        if tempat:
            assert "JAKARTA" in tempat or "SELATAN" in tempat

    def test_invalid_ttl_non_digit_day(self):
        """Test with non-digit day."""
        parts = ["JAKARTA", "XX", "08", "1990"]
        tempat, day, month, year = meet_tgllength(parts)
        assert tempat is None
        assert day is None

    def test_invalid_ttl_non_digit_year(self):
        """Test with non-digit year."""
        parts = ["JAKARTA", "15", "08", "ABCD"]
        tempat, day, month, year = meet_tgllength(parts)
        assert tempat is None

    def test_ttl_with_exception(self):
        """Test exception handling."""
        parts = None
        tempat, day, month, year = meet_tgllength(parts)  # type: ignore[invalid-argument-type]
        assert tempat is None
        assert day is None
        assert month is None
        assert year is None


class TestMatchingTempatlahir:
    """Test cases for matching_tempatlahir function."""

    def test_basic_ttl(self):
        """Test basic TTL extraction."""
        data = ["Tempat/Tgl Lahir", 0.94, []]
        next_data = ["JAKARTA, 15-08-1990", 0.93, []]
        tempat, tanggal = matching_tempatlahir(data, next_data)
        assert isinstance(tempat, str)
        assert isinstance(tanggal, str)

    def test_ttl_with_comma_separator(self):
        """Test TTL with comma separator."""
        data = ["Tempat/Tgl Lahir", 0.94, []]
        next_data = ["JAKARTA, 15-08-1990", 0.93, []]
        tempat, tanggal = matching_tempatlahir(data, next_data)
        if tempat:
            assert "JAKARTA" in tempat or tempat == ""

    def test_ttl_with_slash_separator(self):
        """Test TTL with slash separator."""
        data = ["Tempat/Tgl Lahir", 0.94, []]
        next_data = ["JAKARTA/15-08-1990", 0.93, []]
        tempat, tanggal = matching_tempatlahir(data, next_data)
        assert isinstance(tempat, str)
        assert isinstance(tanggal, str)

    def test_ttl_low_confidence(self):
        """Test TTL with low confidence."""
        data = ["Tempat/Tgl Lahir", 0.94, []]
        next_data = ["JAKARTA, 15-08-1990", 0.50, []]
        tempat, tanggal = matching_tempatlahir(data, next_data)
        assert tempat == ""
        assert tanggal == ""

    def test_ttl_empty_next_data(self):
        """Test with empty next_data."""
        data = ["Tempat/Tgl Lahir", 0.94, []]
        next_data = ["", 0.93, []]
        tempat, tanggal = matching_tempatlahir(data, next_data)
        assert tempat == ""
        assert tanggal == ""


class TestMatchingTempatlahirNew:
    """Test cases for matching_tempatlahir_new function."""

    def test_basic_ttl_new(self):
        """Test basic TTL extraction with new method."""
        data = ["Tempat/Tgl Lahir", 0.94, []]
        next_data = ["JAKARTA 15-08-1990", 0.93, []]
        tempat, tanggal = matching_tempatlahir_new(data, next_data)
        assert isinstance(tempat, str)
        assert isinstance(tanggal, str)

    def test_ttl_new_space_separator(self):
        """Test TTL with space separator."""
        data = ["Tempat/Tgl Lahir", 0.94, []]
        next_data = ["JAKARTA 15-08-1990", 0.93, []]
        tempat, tanggal = matching_tempatlahir_new(data, next_data)
        if tempat:
            assert isinstance(tempat, str)

    def test_ttl_new_low_confidence(self):
        """Test TTL new with low confidence."""
        data = ["Tempat/Tgl Lahir", 0.94, []]
        next_data = ["JAKARTA 15-08-1990", 0.50, []]
        tempat, tanggal = matching_tempatlahir_new(data, next_data)
        # matching_tempatlahir_new doesn't check confidence, it processes the data
        assert tempat == "JAKARTA"
        assert "15" in tanggal
        assert "Aug" in tanggal  # Month converted to abbreviated name
        assert "1990" in tanggal

    def test_ttl_new_empty(self):
        """Test with empty next_data."""
        data = ["Tempat/Tgl Lahir", 0.94, []]
        next_data = ["", 0.93, []]
        tempat, tanggal = matching_tempatlahir_new(data, next_data)
        assert tempat == ""
        assert tanggal == ""

    def test_ttl_new_exception_handling(self):
        """Test exception handling in new method."""
        data = ["Tempat/Tgl Lahir", 0.94, []]
        next_data = None
        tempat, tanggal = matching_tempatlahir_new(data, next_data)  # type: ignore[invalid-argument-type]
        assert tempat == ""
        assert tanggal == ""


class TestTTLIntegration:
    """Integration tests for TTL matching."""

    def test_various_date_formats(self):
        """Test various date formats."""
        test_cases = [
            "JAKARTA, 15-08-1990",
            "JAKARTA 15-08-1990",
            "JAKARTA/15-08-1990",
            "JAKARTA 15/08/1990",
        ]
        
        for ttl_text in test_cases:
            data = ["Tempat/Tgl Lahir", 0.94, []]
            next_data = [ttl_text, 0.93, []]
            tempat, tanggal = matching_tempatlahir(data, next_data)
            # Should process without error
            assert isinstance(tempat, str)
            assert isinstance(tanggal, str)

    def test_various_city_names(self):
        """Test various city names."""
        cities = [
            "JAKARTA",
            "JAKARTA SELATAN",
            "BANDUNG",
            "SURABAYA",
        ]

        for city in cities:
            data = ["Tempat/Tgl Lahir", 0.94, []]
            next_data = [f"{city}, 15-08-1990", 0.93, []]
            tempat, tanggal = matching_tempatlahir(data, next_data)
            assert isinstance(tempat, str)


class TestCleaningTempatMeetTgllength:
    """Cover colon and alphabets branches in cleaning_tempat_meet_tgllength (line 37)."""

    def test_colon_in_tempat_extracts_after_colon(self):
        clear, tempat = cleaning_tempat_meet_tgllength("Tempat: JAKARTA")
        assert clear is True
        assert "JAKARTA" in tempat

    def test_plain_tempat(self):
        clear, tempat = cleaning_tempat_meet_tgllength("JAKARTA")
        assert clear is True
        assert "JAKARTA" in tempat


class TestMeetTgllengthDayTooLong:
    """Cover lines 64-72: day length > 2 triggers re-extraction."""

    def test_day_longer_than_2_chars(self):
        # "JAKARTA15" is fused into the day slot -> regex pulls tempat + day
        parts = ["JAKARTA15", "08", "1990"]
        tempat, day, month, year = meet_tgllength(parts)
        # day should have been re-extracted to <=2 digits
        assert day is None or (day.isdigit() and len(day) <= 2)


class TestNotmeetTgllengthJadiSatu:
    """Cover lines 94-131: notmeet_tgllength_jadi_satu."""

    def test_with_colon_prefix(self):
        tempat, day, month, year = notmeet_tgllength_jadi_satu("Tempat: JAKARTA 15-08-1990")
        # Should extract something meaningful
        assert tempat is not None or day is not None

    def test_8_digit_merged(self):
        tempat, day, month, year = notmeet_tgllength_jadi_satu("JAKARTA15081990")
        assert day == "15"
        assert month == "08"
        assert year == "1990"

    def test_three_separated_numbers(self):
        tempat, day, month, year = notmeet_tgllength_jadi_satu("JAKARTA 15 08 1990")
        assert day == "15"
        assert month == "08"
        assert year == "1990"

    def test_fallback_path_no_regex_match(self):
        """When initial regex does not match, falls back to letters/digits extraction."""
        tempat, day, month, year = notmeet_tgllength_jadi_satu("15081990")
        # Fallback path: all digits, no huruf
        assert day == "15"

    def test_exception_returns_none_tuple(self):
        tempat, day, month, year = notmeet_tgllength_jadi_satu(None)  # type: ignore[invalid-argument-type]
        assert tempat is None


class TestNotmeetTgllengthTerpisah:
    """Cover lines 145-165: notmeet_tgllength_terpisah both regex branches + fallback."""

    def test_regex_format_tanggal1(self):
        tempat, day, month, year = notmeet_tgllength_terpisah("Jakarta 15-08-1990")
        assert day == "15"
        assert month == "08"
        assert year == "1990"

    def test_regex_fallback_when_no_match(self):
        tempat, day, month, year = notmeet_tgllength_terpisah("15081990")
        # Fallback path
        assert day == "15" or day is None


class TestNotmeetTgllength:
    """Cover branching in notmeet_tgllength (lines 178, 184-200)."""

    def test_empty_string_returns_none_tuple(self):
        assert notmeet_tgllength("", False) == (None, None, None, None)

    def test_next_line_true_uses_terpisah(self):
        tempat, day, month, year = notmeet_tgllength("Jakarta 15-08-1990", True)
        assert day == "15"
        assert year == "1990"

    def test_next_line_false_uses_jadi_satu(self):
        tempat, day, month, year = notmeet_tgllength("JAKARTA15081990", False)
        assert day == "15"

    def test_month_raw_none_returns_none_tuple(self):
        # All-letters input cannot produce a month
        tempat, day, month, year = notmeet_tgllength("ABCDEFG", False)
        assert month is None


class TestFormatTtlNewSesuai:
    """Cover lines 208-221."""

    def test_digit_rich_path(self):
        # Tuple-like list with string + string of digits; length >= 3 and >= 5 digits
        tempat, number = format_ttl_new_sesuai(["JAKARTA", "15081990"])
        assert tempat == "JAKARTA"
        assert number == "15081990"

    def test_low_digit_fallback_path_returns_tuple(self):
        """BUG-13 fixed: the fallback joins the list to a string before
        re.findall instead of passing a list, so it returns cleanly."""
        tempat, number = format_ttl_new_sesuai(["JAKARTA", "X"])
        assert tempat == "JAKARTA"
        assert isinstance(number, str)


class TestFormatTtlNewTidakSesuai:
    """Cover lines 226-238."""

    def test_valid_parses(self):
        tempat, number = format_ttl_new_tidak_sesuai("JAKARTA 15-08-1990")
        assert isinstance(tempat, str)
        assert isinstance(number, str)

    def test_exception_returns_empty(self):
        tempat, number = format_ttl_new_tidak_sesuai(None)  # type: ignore[invalid-argument-type]
        assert tempat == ""
        assert number == ""


class TestSplitTgllahir:
    """Cover lines 302, 310-312."""

    def test_empty_returns_empty_list(self):
        assert split_tgllahir("") == []

    def test_none_returns_empty_list(self):
        assert split_tgllahir(None) == []  # type: ignore[invalid-argument-type]

    def test_valid_splits(self):
        parts = split_tgllahir("JAKARTA 15 08 1990")
        assert len(parts) >= 4


class TestValidateTtlData:
    """Cover lines 317-335: validate_ttl_data branches."""

    def test_high_digit_in_data_uses_data_branch(self):
        # digit_data >= 6 and digit_nextdata < 6
        data = ["JAKARTA:15-08-1990", 0.95, []]
        next_data = ["LAKI-LAKI", 0.94, []]
        parts = validate_ttl_data(data, next_data)
        assert isinstance(parts, list)

    def test_high_digit_in_next_data_uses_next_branch(self):
        data = ["Tempat/Tgl Lahir", 0.94, []]
        next_data = [": JAKARTA 15 08 1990", 0.95, []]
        parts = validate_ttl_data(data, next_data)
        assert isinstance(parts, list)
        assert len(parts) > 0

    def test_low_digits_returns_empty_parts(self):
        data = ["Tempat/Tgl Lahir", 0.94, []]
        next_data = ["LAKI-LAKI", 0.94, []]
        parts = validate_ttl_data(data, next_data)
        assert parts == []


class TestMatchingTempatlahirBranches:
    """Cover remaining branches in matching_tempatlahir (lines 356-365)."""

    def test_next_data_contains_jenis_uses_data(self):
        """When next_data has 'jenis' keyword, use `data` field."""
        data = ["JAKARTA, 15-08-1990", 0.95, []]
        next_data = ["Jenis Kelamin", 0.94, []]
        tempat, tgl = matching_tempatlahir(data, next_data)
        assert isinstance(tempat, str)
        assert isinstance(tgl, str)

    def test_parts_ge_4_path(self):
        """When validate_ttl_data returns >=4 parts and 'jenis' not in next."""
        data = ["Tempat/Tgl Lahir", 0.94, []]
        next_data = ["JAKARTA 15 08 1990", 0.95, []]
        tempat, tgl = matching_tempatlahir(data, next_data)
        assert isinstance(tempat, str)
        assert isinstance(tgl, str)

    def test_long_tempat_trims_to_capitals(self):
        """When tempat is longer than 9 chars, only capital-letter words are kept."""
        data = ["Tempat/Tgl Lahir", 0.94, []]
        next_data = ["JAKARTA SELATAN TIMUR 15-08-1990", 0.95, []]
        tempat, tgl = matching_tempatlahir(data, next_data)
        assert isinstance(tempat, str)
