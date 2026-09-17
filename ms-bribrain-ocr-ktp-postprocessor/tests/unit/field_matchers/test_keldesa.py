"""Tests for Kelurahan/Desa field matcher."""

import pytest

from src.services.field_matchers.keldesa import matching_keldesa


class TestMatchingKeldesa:
    """Test cases for kelurahan/desa matching function."""

    def test_valid_keldesa(self):
        """Test valid kelurahan/desa with high confidence."""
        next_data = ["TANAH ABANG", 0.92, [[220, 540], [350, 540], [350, 590], [220, 590]]]
        result = matching_keldesa(next_data)
        assert result == "TANAH ABANG"

    def test_keldesa_with_lowercase(self):
        """Test kelurahan/desa is converted to uppercase."""
        next_data = ["tanah abang", 0.92, []]
        result = matching_keldesa(next_data)
        assert result == "TANAH ABANG"

    def test_keldesa_with_colon(self):
        """Test kelurahan/desa with colon prefix."""
        next_data = [":TANAH ABANG", 0.92, []]
        result = matching_keldesa(next_data)
        assert "TANAH ABANG" in result
        assert not result.startswith(":")

    def test_keldesa_with_low_confidence(self):
        """Test kelurahan/desa with low confidence."""
        next_data = ["TANAH ABANG", 0.50, []]
        result = matching_keldesa(next_data)
        assert result == ""

    def test_empty_keldesa(self):
        """Test with empty kelurahan/desa."""
        next_data = ["", 0.92, []]
        result = matching_keldesa(next_data)
        assert result == ""

    def test_keldesa_with_special_characters(self):
        """Test kelurahan/desa with special characters."""
        next_data = ["KELURAHAN-MENTENG", 0.92, []]
        result = matching_keldesa(next_data)
        assert isinstance(result, str)
        assert result.isupper()

    def test_long_keldesa_name(self):
        """Test with long kelurahan/desa name."""
        next_data = ["KELURAHAN KAMPUNG MELAYU KECIL", 0.92, []]
        result = matching_keldesa(next_data)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_keldesa_with_numbers(self):
        """Test kelurahan/desa with numbers."""
        next_data = ["KELURAHAN 5 ULU", 0.92, []]
        result = matching_keldesa(next_data)
        assert isinstance(result, str)

    def test_none_data(self):
        """Test with None data."""
        next_data = None
        result = matching_keldesa(next_data)  # type: ignore[invalid-argument-type]
        assert result == ""

    def test_empty_list(self):
        """Test with empty list."""
        next_data = []
        result = matching_keldesa(next_data)
        assert result == ""

    def test_malformed_data(self):
        """Test with malformed data (missing confidence)."""
        next_data = ["TANAH ABANG"]
        result = matching_keldesa(next_data)
        assert result == ""

    def test_keldesa_with_multiple_spaces(self):
        """Test kelurahan/desa with multiple spaces."""
        next_data = ["TANAH    ABANG", 0.92, []]
        result = matching_keldesa(next_data)
        assert isinstance(result, str)
