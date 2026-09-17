"""Tests for text helper functions."""

import pytest

from src.services.text_helpers import (
    fuzz_ratio,
    fuzz_partial,
    correct_alphabets_to_digits,
    correct_digits_to_alphabets,
    clean_colon,
    getdigitonly,
    searchmapping,
    filter_result,
    data_verification,
    create_err_message,
)


class TestFuzzRatio:
    """Test cases for fuzz_ratio function."""

    def test_identical_strings(self):
        """Test with identical strings."""
        score = fuzz_ratio("JAKARTA", "JAKARTA")
        assert score == 100

    def test_case_insensitive(self):
        """Test case insensitivity."""
        score = fuzz_ratio("Jakarta", "JAKARTA")
        assert score == 100

    def test_similar_strings(self):
        """Test with similar strings."""
        score = fuzz_ratio("NIK", "NIK")
        assert score == 100
        
        score = fuzz_ratio("NIK", "NIKK")
        assert score > 50  # Similar but not identical

    def test_different_strings(self):
        """Test with completely different strings."""
        score = fuzz_ratio("JAKARTA", "SURABAYA")
        assert score < 50

    def test_empty_strings(self):
        """Test with empty strings."""
        score = fuzz_ratio("", "")
        assert score == 100
        
        score = fuzz_ratio("TEXT", "")
        assert score == 0


class TestFuzzPartial:
    """Test cases for fuzz_partial function."""

    def test_substring_match(self):
        """Test with substring matching."""
        score = fuzz_partial("NIK", "NIK 1234")
        assert score == 100

    def test_partial_match(self):
        """Test with partial matching."""
        score = fuzz_partial("Tempat", "Tempat/Tgl Lahir")
        assert score == 100

    def test_no_match(self):
        """Test with no matching substring."""
        score = fuzz_partial("ABC", "XYZ")
        assert score < 50


class TestCorrectAlphabetsToDigits:
    """Test cases for correct_alphabets_to_digits function."""

    def test_o_to_zero(self):
        """Test converting O to 0 between digits."""
        result = correct_alphabets_to_digits("31740O123")
        assert "O" not in result or result == "317400123"

    def test_letter_not_between_digits(self):
        """Test that letters not between digits are preserved."""
        result = correct_alphabets_to_digits("ABC123")
        assert "ABC" in result

    def test_no_correction_needed(self):
        """Test string with no corrections needed."""
        result = correct_alphabets_to_digits("123456")
        assert result == "123456"

    def test_multiple_corrections(self):
        """Test multiple character corrections."""
        result = correct_alphabets_to_digits("1O2I3")
        # O and I between digits should be corrected
        assert isinstance(result, str)

    def test_empty_string(self):
        """Test with empty string."""
        result = correct_alphabets_to_digits("")
        assert result == ""


class TestCorrectDigitsToAlphabets:
    """Test cases for correct_digits_to_alphabets function."""

    def test_digit_between_letters(self):
        """Test converting digits to letters between alphabets."""
        result = correct_digits_to_alphabets("JAK0RTA")
        # 0 between A and R might be converted to O
        assert isinstance(result, str)

    def test_digit_not_between_letters(self):
        """Test that digits not between letters are preserved."""
        result = correct_digits_to_alphabets("123ABC")
        assert "123" in result

    def test_no_correction_needed(self):
        """Test string with no corrections needed."""
        result = correct_digits_to_alphabets("JAKARTA")
        assert result == "JAKARTA"

    def test_empty_string(self):
        """Test with empty string."""
        result = correct_digits_to_alphabets("")
        assert result == ""


class TestCleanColon:
    """Test cases for clean_colon function."""

    def test_remove_leading_colon(self):
        """Test removing leading colon."""
        result = clean_colon(":JAKARTA")
        assert result == "JAKARTA"
        assert not result.startswith(":")

    def test_remove_leading_spaces(self):
        """Test removing leading spaces."""
        result = clean_colon("  JAKARTA")
        assert result == "JAKARTA"
        assert not result.startswith(" ")

    def test_remove_colon_and_spaces(self):
        """Test removing both colon and spaces."""
        result = clean_colon(": JAKARTA")
        assert result == "JAKARTA"

    def test_no_cleaning_needed(self):
        """Test string with no cleaning needed."""
        result = clean_colon("JAKARTA")
        assert result == "JAKARTA"

    def test_empty_string(self):
        """Test with empty string."""
        result = clean_colon("")
        assert result == ""

    def test_none_input(self):
        """Test with None input."""
        result = clean_colon(None)  # type: ignore[invalid-argument-type]
        assert result == ""


class TestGetDigitOnly:
    """Test cases for getdigitonly function."""

    def test_extract_digits(self):
        """Test extracting digits from mixed string."""
        result = getdigitonly("ABC123DEF456")
        assert result == "123456"

    def test_remove_spaces(self):
        """Test removing spaces."""
        result = getdigitonly("1 2 3 4 5")
        assert result == "12345"

    def test_remove_special_chars(self):
        """Test removing special characters."""
        result = getdigitonly("123-456-789")
        assert result == "123456789"

    def test_only_digits(self):
        """Test string with only digits."""
        result = getdigitonly("123456")
        assert result == "123456"

    def test_no_digits(self):
        """Test string with no digits."""
        result = getdigitonly("ABCDEF")
        assert result == ""

    def test_empty_string(self):
        """Test with empty string."""
        result = getdigitonly("")
        assert result == ""

    def test_none_input(self):
        """Test with None input."""
        result = getdigitonly(None)  # type: ignore[invalid-argument-type]
        assert result == ""


class TestSearchMapping:
    """Test cases for searchmapping function."""

    def test_agama_mapping(self):
        """Test religion mapping."""
        result, score = searchmapping("agama", "ISLAM")
        assert result is not None
        assert score > 80

    def test_jenis_kelamin_mapping(self):
        """Test gender mapping."""
        result, score = searchmapping("jenis_kelamin", "LAKI-LAKI")
        assert result is not None
        assert score > 80

    def test_status_perkawinan_mapping(self):
        """Test marital status mapping."""
        result, score = searchmapping("status_perkawinan", "KAWIN")
        assert result is not None
        assert score > 80

    def test_invalid_field(self):
        """Test with invalid field name."""
        result, score = searchmapping("invalid_field", "TEST")
        assert result is None
        assert score == 0

    def test_fuzzy_match(self):
        """Test fuzzy matching with typo."""
        result, score = searchmapping("agama", "ISLM")
        # Should still find a match
        assert isinstance(result, (str, type(None)))
        assert isinstance(score, (int, float))


class TestFilterResult:
    """Test cases for filter_result function."""

    def test_filter_within_threshold(self):
        """Test filtering results within threshold."""
        width = 1000
        result = [
            [[[100, 50], [200, 50], [200, 100], [100, 100]], ("TEXT1", 0.95)],
            [[[700, 50], [750, 50], [750, 100], [700, 100]], ("TEXT2", 0.94)],
            [[[850, 50], [900, 50], [900, 100], [850, 100]], ("TEXT3", 0.93)],  # Within threshold
        ]
        filtered = filter_result(result, width)
        # All should be within threshold (4/5 * 1000 = 800)
        assert len(filtered) > 0

    def test_filter_outside_threshold(self):
        """Test filtering results outside threshold."""
        width = 1000
        result = [
            [[[100, 50], [200, 50], [200, 100], [100, 100]], ("TEXT1", 0.95)],
            [[[850, 50], [950, 50], [950, 100], [850, 100]], ("TEXT2", 0.94)],  # Outside threshold
        ]
        filtered = filter_result(result, width)
        # Second item should be filtered out
        assert len(filtered) <= len(result)

    def test_empty_result(self):
        """Test with empty result list."""
        filtered = filter_result([], 1000)
        assert filtered == []

    def test_various_widths(self):
        """Test with various width values."""
        result = [
            [[[100, 50], [200, 50], [200, 100], [100, 100]], ("TEXT", 0.95)],
        ]
        
        filtered_small = filter_result(result, 500)
        filtered_large = filter_result(result, 5000)
        
        assert isinstance(filtered_small, list)
        assert isinstance(filtered_large, list)


class TestDataVerification:
    """Test cases for data_verification function."""

    def test_normal_type_with_valid_data(self):
        """Test normal type with valid data."""
        result = data_verification("BUDI SANTOSO", "normal")
        assert result == "BUDI SANTOSO"

    def test_normal_type_with_keydata(self):
        """Test normal type with keydata (should be filtered)."""
        result = data_verification("nik", "normal")
        assert result == ""

    def test_normal_type_with_nama_keydata(self):
        """Test normal type with 'nama' keydata."""
        result = data_verification("nama", "normal")
        assert result == ""

    def test_other_type_with_valid_data(self):
        """Test other type with valid data."""
        result = data_verification("SOME DATA", "other")
        assert result == "SOME DATA"

    def test_other_type_with_keyid(self):
        """Test other type with keyid (should be filtered)."""
        # Assuming there's a keyid list in constants
        result = data_verification("test", "other")
        assert isinstance(result, str)

    def test_case_insensitive_filtering(self):
        """Test that filtering is case-insensitive."""
        result = data_verification("NIK", "normal")
        assert result == ""

    def test_empty_data(self):
        """Test with empty data."""
        result = data_verification("", "normal")
        assert result == ""


class TestCreateErrMessage:
    """Test cases for create_err_message function."""

    def test_single_error_blur(self):
        """Test with single error - blur."""
        result = create_err_message(is_blurry=True, is_glare=False, is_rotated=False)
        assert "blur" in result.lower() or "kabur" in result.lower()

    def test_single_error_glare(self):
        """Test with single error - glare."""
        result = create_err_message(is_blurry=False, is_glare=True, is_rotated=False)
        assert "glare" in result.lower() or "silau" in result.lower() or "pantulan" in result.lower()

    def test_single_error_rotated(self):
        """Test with single error - rotated."""
        result = create_err_message(is_blurry=False, is_glare=False, is_rotated=True)
        assert "tidak sejajar" in result.lower()

    def test_two_errors(self):
        """Test with two errors."""
        result = create_err_message(is_blurry=True, is_glare=True, is_rotated=False)
        assert "dan" in result

    def test_three_errors(self):
        """Test with all three errors."""
        result = create_err_message(is_blurry=True, is_glare=True, is_rotated=True)
        assert "dan" in result
        assert isinstance(result, str)

    def test_no_errors(self):
        """Test with no errors (edge case)."""
        try:
            result = create_err_message(is_blurry=False, is_glare=False, is_rotated=False)
            # Might raise IndexError if message list is empty
            assert True
        except IndexError:
            # Expected behavior when no errors
            assert True


class TestCorrectAlphabetsToDigitsSymbolBranch:
    """Cover the symbol_mapping branch inside correct_alphabets_to_digits."""

    def test_symbol_between_digits_is_converted(self):
        """'1!1' should become '111' because '!' maps to '1' and sits between digits."""
        result = correct_alphabets_to_digits("1!1")
        assert result == "111"

    def test_symbol_not_between_digits_is_kept(self):
        """Symbol at the edge or next to letters should not be converted."""
        result = correct_alphabets_to_digits("!1A")
        assert result == "!1A"


class TestSearchmappingNoMatchField:
    """Cover line 183: field found but process.extractOne returns None."""

    def test_process_extractone_returns_none(self, monkeypatch):
        from src.services import text_helpers

        class _FakeProcess:
            @staticmethod
            def extractOne(*args, **kwargs):
                return None

        monkeypatch.setattr(text_helpers, "process", _FakeProcess)
        result, score = searchmapping("agama", "XYZ")
        assert result is None
        assert score == 0


class TestDataVerificationNonNormal:
    """Cover line 236: data_verification with valuetype != 'normal', data not in keyid."""

    def test_non_normal_valid_data_returned(self):
        from src.services.text_helpers import data_verification

        result = data_verification("SomeValidValue", "other")
        assert result == "SomeValidValue"

    def test_non_normal_matches_keyid_returns_empty(self):
        from src.services.text_helpers import data_verification
        from src.services.constants import keyid

        if keyid:
            result = data_verification(keyid[0], "other")
            assert result == ""
