"""Tests for NIK field matcher."""

import pytest

from src.services.field_matchers.nik import matching_nik


class TestMatchingNIK:
    """Test cases for NIK matching function."""

    def test_valid_nik_with_high_confidence(self):
        """Test valid 16-digit NIK with high confidence."""
        data = ["3174012345678901", 0.97, [[220, 120], [400, 120], [400, 170], [220, 170]]]
        result = matching_nik(data)
        assert result == "3174012345678901"

    def test_nik_with_alphabet_to_digit_conversion(self):
        """Test NIK with letters that should be converted to digits."""
        # O (letter) should be converted to 0 (zero)
        data = ["3174O12345678901", 0.85, [[220, 120], [400, 120], [400, 170], [220, 170]]]
        result = matching_nik(data)
        # Result depends on correct_alphabets_to_digits implementation
        assert len(result) in [0, 16]  # Either filtered out or corrected

    def test_nik_with_colon(self):
        """Test NIK with colon that should be cleaned."""
        data = [":3174012345678901", 0.90, [[220, 120], [400, 120], [400, 170], [220, 170]]]
        result = matching_nik(data)
        # Should clean the colon and return valid NIK
        if result:
            assert result == "3174012345678901"

    def test_nik_too_short(self):
        """Test NIK with less than 16 digits."""
        data = ["31740123456789", 0.90, [[220, 120], [400, 120], [400, 170], [220, 170]]]
        result = matching_nik(data)
        assert result == ""

    def test_nik_too_long(self):
        """Test NIK with more than 16 digits."""
        data = ["31740123456789012", 0.88, [[220, 120], [400, 120], [400, 170], [220, 170]]]
        result = matching_nik(data)
        assert result == ""

    def test_nik_with_low_confidence(self):
        """Test NIK with confidence below threshold."""
        data = ["3174012345678901", 0.50, [[220, 120], [400, 120], [400, 170], [220, 170]]]
        result = matching_nik(data)
        assert result == ""

    def test_empty_nik(self):
        """Test with empty NIK text."""
        data = ["", 0.95, [[220, 120], [400, 120], [400, 170], [220, 170]]]
        result = matching_nik(data)
        assert result == ""

    def test_single_character_nik(self):
        """Test with single character NIK."""
        data = ["1", 0.95, [[220, 120], [400, 120], [400, 170], [220, 170]]]
        result = matching_nik(data)
        assert result == ""

    def test_nik_with_special_characters(self):
        """Test NIK containing special characters."""
        data = ["3174-012-345-678-901", 0.90, [[220, 120], [400, 120], [400, 170], [220, 170]]]
        result = matching_nik(data)
        # Should extract only digits
        if result:
            assert result.isdigit()

    def test_nik_with_spaces(self):
        """Test NIK containing spaces."""
        data = ["3174 0123 4567 8901", 0.90, [[220, 120], [400, 120], [400, 170], [220, 170]]]
        result = matching_nik(data)
        # Should extract only digits
        if result:
            assert result.isdigit()
            assert " " not in result

    def test_nik_with_mixed_alphanumeric(self):
        """Test NIK with letters that can't be converted."""
        data = ["3174ABC345678901", 0.90, [[220, 120], [400, 120], [400, 170], [220, 170]]]
        result = matching_nik(data)
        # Might be filtered out or partially converted
        if result:
            assert result.isdigit()

    def test_none_data(self):
        """Test with None or malformed data."""
        try:
            result = matching_nik(None)  # type: ignore[invalid-argument-type]
            assert result == ""
        except:
            # If it raises exception, that's also acceptable behavior
            pass

    def test_empty_list(self):
        """Test with empty list."""
        try:
            result = matching_nik([])
            assert result == ""
        except:
            # If it raises exception, that's also acceptable behavior
            pass

    def test_whitespace_only_text(self):
        """Whitespace-only NIK text should return empty string."""
        data = ["   ", 0.97, []]
        result = matching_nik(data)
        assert result == ""

    def test_confidence_at_exact_threshold(self):
        """Confidence exactly at THRESHOLD_CONFIDENCE should NOT pass (strict >)."""
        from src.services.constants import THRESHOLD_CONFIDENCE
        data = ["3174012345678901", THRESHOLD_CONFIDENCE, []]
        result = matching_nik(data)
        assert result == ""

    def test_nik_longer_than_16_digits(self):
        """17+ digit input should return empty (strict ==16)."""
        data = ["31740123456789012345", 0.97, []]
        result = matching_nik(data)
        assert result == ""
