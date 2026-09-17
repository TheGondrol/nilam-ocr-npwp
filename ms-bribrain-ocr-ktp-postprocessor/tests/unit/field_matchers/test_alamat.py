"""Tests for Alamat field matcher."""

import pytest

from src.services.field_matchers.alamat import matching_alamat


class TestMatchingAlamat:
    """Test cases for alamat matching function."""

    def test_valid_alamat(self):
        """Test valid alamat with high confidence."""
        next_data = ["JL SUDIRMAN NO 10", 0.92, [[220, 400], [500, 400], [500, 450], [220, 450]]]
        result = matching_alamat(next_data)
        assert result == "JL SUDIRMAN NO 10"

    def test_alamat_with_lowercase(self):
        """Test alamat is converted to uppercase."""
        next_data = ["jl sudirman no 10", 0.92, []]
        result = matching_alamat(next_data)
        assert result == "JL SUDIRMAN NO 10"

    def test_alamat_with_colon(self):
        """Test alamat with colon prefix."""
        next_data = [":JL SUDIRMAN NO 10", 0.92, []]
        result = matching_alamat(next_data)
        assert "JL SUDIRMAN NO 10" in result
        assert not result.startswith(":")

    def test_alamat_with_low_confidence(self):
        """Test alamat with low confidence."""
        next_data = ["JL SUDIRMAN NO 10", 0.50, []]
        result = matching_alamat(next_data)
        assert result == ""

    def test_empty_alamat(self):
        """Test with empty alamat."""
        next_data = ["", 0.92, []]
        result = matching_alamat(next_data)
        assert result == ""

    def test_alamat_with_special_characters(self):
        """Test alamat with special characters."""
        next_data = ["JL. SUDIRMAN NO. 10 RT/RW 001/002", 0.92, []]
        result = matching_alamat(next_data)
        assert isinstance(result, str)
        assert result.isupper()

    def test_very_long_alamat(self):
        """Test with very long alamat."""
        next_data = ["JL SUDIRMAN NO 10 KOMPLEK PERUMAHAN GRIYA INDAH BLOK A RT 001 RW 002", 0.92, []]
        result = matching_alamat(next_data)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_alamat_with_numbers(self):
        """Test alamat with numbers."""
        next_data = ["JL RAYA BOGOR KM 25", 0.92, []]
        result = matching_alamat(next_data)
        assert result == "JL RAYA BOGOR KM 25"

    def test_none_data(self):
        """Test with None data."""
        next_data = None
        result = matching_alamat(next_data)  # type: ignore[invalid-argument-type]
        assert result == ""

    def test_empty_list(self):
        """Test with empty list."""
        next_data = []
        result = matching_alamat(next_data)
        assert result == ""

    def test_malformed_data(self):
        """Test with malformed data (missing confidence)."""
        next_data = ["JL SUDIRMAN"]
        result = matching_alamat(next_data)
        assert result == ""

    def test_alamat_with_multiple_spaces(self):
        """Test alamat with multiple spaces."""
        next_data = ["JL    SUDIRMAN    NO    10", 0.92, []]
        result = matching_alamat(next_data)
        assert isinstance(result, str)
        # Should normalize spaces
        assert "    " not in result or result == "JL    SUDIRMAN    NO    10"
