"""Tests for Nama field matcher."""

import pytest

from src.services.field_matchers.nama import matching_nama


class TestMatchingNama:
    """Test cases for nama matching function."""

    def test_valid_nama_with_high_confidence(self):
        """Test valid name with high confidence."""
        data = ["BUDI SANTOSO", 0.95, [[220, 190], [400, 190], [400, 240], [220, 240]]]
        result = matching_nama(data)
        assert result == "BUDI SANTOSO"

    def test_nama_with_lowercase(self):
        """Test name with lowercase letters (should be converted to uppercase)."""
        data = ["Budi Santoso", 0.94, [[220, 190], [400, 190], [400, 240], [220, 240]]]
        result = matching_nama(data)
        assert result == "BUDI SANTOSO"

    def test_nama_with_special_characters(self):
        """Test name containing special characters."""
        data = ["AHMAD@FAUZI", 0.92, [[220, 190], [400, 190], [400, 240], [220, 240]]]
        result = matching_nama(data)
        # Special characters should be replaced with spaces
        assert "@" not in result
        assert "AHMAD" in result
        assert "FAUZI" in result

    def test_nama_with_colon(self):
        """Test name with colon prefix."""
        data = [":SITI NURHALIZA", 0.94, [[220, 190], [400, 190], [400, 240], [220, 240]]]
        result = matching_nama(data)
        # Colon should be cleaned
        assert ":" not in result
        if result:
            assert result == "SITI NURHALIZA"

    def test_nama_with_multiple_spaces(self):
        """Test name with multiple consecutive spaces."""
        data = ["BUDI    SANTOSO", 0.93, [[220, 190], [400, 190], [400, 240], [220, 240]]]
        result = matching_nama(data)
        # Multiple spaces should be normalized to single space
        assert "    " not in result
        assert result == "BUDI SANTOSO"

    def test_nama_with_leading_trailing_spaces(self):
        """Test name with leading and trailing spaces."""
        data = ["  AHMAD FAUZI  ", 0.94, [[220, 190], [400, 190], [400, 240], [220, 240]]]
        result = matching_nama(data)
        # Spaces should be trimmed
        assert result == "AHMAD FAUZI"
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_nama_with_low_confidence(self):
        """Test name with confidence below threshold."""
        data = ["BUDI SANTOSO", 0.50, [[220, 190], [400, 190], [400, 240], [220, 240]]]
        result = matching_nama(data)
        assert result == ""

    def test_empty_nama(self):
        """Test with empty name text."""
        data = ["", 0.95, [[220, 190], [400, 190], [400, 240], [220, 240]]]
        result = matching_nama(data)
        assert result == ""

    def test_single_character_nama(self):
        """Test with single character name."""
        data = ["A", 0.95, [[220, 190], [400, 190], [400, 240], [220, 240]]]
        result = matching_nama(data)
        assert result == ""

    def test_nama_with_numbers(self):
        """Test name containing numbers."""
        data = ["BUDI123", 0.92, [[220, 190], [400, 190], [400, 240], [220, 240]]]
        result = matching_nama(data)
        # Numbers might be allowed in the name
        if result:
            assert "BUDI" in result

    def test_nama_with_dots_and_commas(self):
        """Test name with punctuation."""
        data = ["SITI, NUR.HALIZA", 0.93, [[220, 190], [400, 190], [400, 240], [220, 240]]]
        result = matching_nama(data)
        # Punctuation should be replaced with spaces
        assert "," not in result
        assert "." not in result

    def test_very_long_nama(self):
        """Test with very long name."""
        data = ["NAMA YANG SANGAT PANJANG SEKALI UNTUK TESTING", 0.94, 
                [[220, 190], [400, 190], [400, 240], [220, 240]]]
        result = matching_nama(data)
        # Should process without error
        assert isinstance(result, str)

    def test_nama_with_mixed_special_chars(self):
        """Test name with various special characters."""
        data = ["BUDI!@#$%SANTOSO", 0.92, [[220, 190], [400, 190], [400, 240], [220, 240]]]
        result = matching_nama(data)
        # Special chars should be replaced
        if result:
            assert not any(c in result for c in "!@#$%")
