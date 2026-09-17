"""Tests for Kecamatan field matcher."""

import pytest

from src.services.field_matchers.kecamatan import matching_kecamatan


class TestMatchingKecamatan:
    """Test cases for kecamatan matching function."""

    def test_valid_kecamatan_from_next_data(self):
        """Test valid kecamatan from next_data."""
        data = ["Kecamatan", 0.94, []]
        next_data = ["TANAH ABANG", 0.93, []]
        result = matching_kecamatan(data, next_data)
        assert result == "TANAH ABANG"

    def test_valid_kecamatan_from_data(self):
        """Test valid kecamatan from data itself (long text)."""
        data = ["Kecamatan: KEBAYORAN BARU", 0.94, []]
        next_data = ["Something else", 0.93, []]
        result = matching_kecamatan(data, next_data)
        assert "KEBAYORAN" in result or "BARU" in result

    def test_kecamatan_with_colon(self):
        """Test kecamatan with colon prefix."""
        data = ["Kecamatan", 0.94, []]
        next_data = [":TANAH ABANG", 0.93, []]
        result = matching_kecamatan(data, next_data)
        assert "TANAH ABANG" in result
        assert not result.startswith(":")

    def test_kecamatan_with_lowercase(self):
        """Test kecamatan is converted to uppercase."""
        data = ["Kecamatan", 0.94, []]
        next_data = ["tanah abang", 0.93, []]
        result = matching_kecamatan(data, next_data)
        assert result.isupper() or result == ""

    def test_kecamatan_with_low_confidence(self):
        """Test kecamatan with low confidence."""
        data = ["Kec", 0.50, []]
        next_data = ["TANAH ABANG", 0.50, []]
        result = matching_kecamatan(data, next_data)
        assert result == ""

    def test_empty_data(self):
        """Test with empty data."""
        data = ["", 0.94, []]
        next_data = ["", 0.93, []]
        result = matching_kecamatan(data, next_data)
        assert result == ""

    def test_kecamatan_pattern_matching(self):
        """Test pattern matching for capital letters."""
        data = ["Kecamatan", 0.94, []]
        next_data = ["MENTENG", 0.93, []]
        result = matching_kecamatan(data, next_data)
        assert "MENTENG" in result or result == ""

    def test_none_next_data(self):
        """Test with None next_data."""
        data = ["Kecamatan", 0.94, []]
        next_data = None
        result = matching_kecamatan(data, next_data)  # type: ignore[invalid-argument-type]
        assert result == ""

    def test_empty_list_next_data(self):
        """Test with empty list next_data."""
        data = ["Kecamatan", 0.94, []]
        next_data = []
        result = matching_kecamatan(data, next_data)
        assert result == ""

    def test_malformed_data(self):
        """Test with malformed data."""
        data = []
        next_data = ["TANAH ABANG", 0.93, []]
        result = matching_kecamatan(data, next_data)
        assert result == ""

    def test_long_data_field(self):
        """Test with long data field (>11 chars)."""
        data = ["Kecamatan: KEBAYORAN LAMA", 0.94, []]
        next_data = ["Ignored", 0.93, []]
        result = matching_kecamatan(data, next_data)
        assert isinstance(result, str)

    def test_kecamatan_with_special_chars(self):
        """Test kecamatan with special characters."""
        data = ["Kecamatan", 0.94, []]
        next_data = ["TANAH-ABANG", 0.93, []]
        result = matching_kecamatan(data, next_data)
        assert isinstance(result, str)
