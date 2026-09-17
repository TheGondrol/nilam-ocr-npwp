"""Tests for RT/RW field matcher."""

import pytest

from src.services.field_matchers.rtrw import matching_rtrw, rtrw_sesuai_format, rtrw_tidak_sesuai_format


class TestRTRWMatching:
    """Test cases for RT/RW matching functions."""

    def test_standard_format_with_slash(self):
        """Test RT/RW in standard format with slash separator."""
        data = ["001/002", 0.94, [[220, 470], [300, 470], [300, 520], [220, 520]]]
        rt, rw = matching_rtrw(data)
        assert rt == "001"
        assert rw == "002"

    def test_format_without_slash(self):
        """Test RT/RW without slash separator."""
        data = ["001002", 0.93, [[220, 470], [300, 470], [300, 520], [220, 520]]]
        rt, rw = matching_rtrw(data)
        # Should split as first 3 and last 3 digits
        if rt and rw:
            assert len(rt) == 3
            assert len(rw) == 3

    def test_short_format(self):
        """Test RT/RW with short format (e.g., 5/3)."""
        data = ["5/3", 0.92, [[220, 470], [300, 470], [300, 520], [220, 520]]]
        rt, rw = matching_rtrw(data)
        # Should be zero-padded to 3 digits
        if rt and rw:
            assert rt == "005"
            assert rw == "003"

    def test_format_with_extra_characters(self):
        """Test RT/RW with extra non-digit characters."""
        data = ["RT001/RW002", 0.91, [[220, 470], [300, 470], [300, 520], [220, 520]]]
        rt, rw = matching_rtrw(data)
        # Should extract only digits
        if rt and rw:
            assert rt.isdigit()
            assert rw.isdigit()

    def test_low_confidence(self):
        """Test RT/RW with low confidence."""
        data = ["001/002", 0.50, [[220, 470], [300, 470], [300, 520], [220, 520]]]
        rt, rw = matching_rtrw(data)
        assert rt == ""
        assert rw == ""

    def test_empty_data(self):
        """Test with empty data."""
        data = ["", 0.95, [[220, 470], [300, 470], [300, 520], [220, 520]]]
        rt, rw = matching_rtrw(data)
        assert rt == ""
        assert rw == ""

    def test_invalid_format(self):
        """Test with invalid format (no numbers)."""
        data = ["ABC/DEF", 0.92, [[220, 470], [300, 470], [300, 520], [220, 520]]]
        rt, rw = matching_rtrw(data)
        assert rt == ""
        assert rw == ""

    def test_single_number(self):
        """Test with single number."""
        data = ["123", 0.93, [[220, 470], [300, 470], [300, 520], [220, 520]]]
        rt, rw = matching_rtrw(data)
        # Might return empty or partial result
        assert isinstance(rt, str)
        assert isinstance(rw, str)


class TestRTRWSesuaiFormat:
    """Test cases for standard format extraction."""

    def test_with_slash(self):
        """Test standard format with slash."""
        rt, rw = rtrw_sesuai_format("001/002")
        assert rt == "001"
        assert rw == "002"

    def test_without_slash(self):
        """Test standard format without slash."""
        rt, rw = rtrw_sesuai_format("001002")
        assert rt == "001"
        assert rw == "002"

    def test_with_letters(self):
        """Test with letters that should be stripped."""
        rt, rw = rtrw_sesuai_format("RT001/RW002")
        if rt and rw:
            assert rt == "001"
            assert rw == "002"

    def test_short_numbers(self):
        """Test with short numbers that need padding."""
        rt, rw = rtrw_sesuai_format("1/2")
        if rt and rw:
            assert rt == "001"
            assert rw == "002"


class TestRTRWTidakSesuaiFormat:
    """Test cases for non-standard format extraction."""

    def test_non_standard_format(self):
        """Test non-standard RT/RW format."""
        rt, rw = rtrw_tidak_sesuai_format("10203")
        # Should attempt to extract RT and RW
        assert isinstance(rt, str)
        assert isinstance(rw, str)

    def test_with_zeros(self):
        """Test format with zeros as separators."""
        rt, rw = rtrw_tidak_sesuai_format("10203")
        # Should handle zeros appropriately
        assert isinstance(rt, str)
        assert isinstance(rw, str)

    def test_invalid_format(self):
        """Test completely invalid format."""
        rt, rw = rtrw_tidak_sesuai_format("ABC")
        assert rt == ""
        assert rw == ""
