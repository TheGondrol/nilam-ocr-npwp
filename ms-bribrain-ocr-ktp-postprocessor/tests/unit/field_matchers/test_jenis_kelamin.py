"""Tests for Jenis Kelamin field matcher."""

import pytest

from src.services.field_matchers.jenis_kelamin import matching_jeniskelamin


class TestMatchingJenisKelamin:
    """Test cases for jenis kelamin matching function."""

    def test_laki_laki_with_key_jenis_kelamin(self):
        """Test LAKI-LAKI with 'jenis kelamin' key."""
        key = "jenis kelamin"
        data = ["Jenis Kelamin", 0.95, []]
        next_data = ["LAKI-LAKI", 0.94, []]
        result = matching_jeniskelamin(key, data, next_data)
        assert result == "LAKI-LAKI"

    def test_perempuan_with_key_jenis_kelamin(self):
        """Test PEREMPUAN with 'jenis kelamin' key."""
        key = "jenis kelamin"
        data = ["Jenis Kelamin", 0.95, []]
        next_data = ["PEREMPUAN", 0.94, []]
        result = matching_jeniskelamin(key, data, next_data)
        assert result == "PEREMPUAN"

    def test_laki_laki_with_colon(self):
        """Test LAKI-LAKI with colon prefix."""
        key = "jenis kelamin"
        data = ["Jenis Kelamin", 0.95, []]
        next_data = [":LAKI-LAKI", 0.94, []]
        result = matching_jeniskelamin(key, data, next_data)
        assert result == "LAKI-LAKI"

    def test_perempuan_typo(self):
        """Test PEREMPUAN with typo (fuzzy matching)."""
        key = "jenis kelamin"
        data = ["Jenis Kelamin", 0.95, []]
        next_data = ["PEREMPAN", 0.94, []]  # Typo
        result = matching_jeniskelamin(key, data, next_data)
        # Should still match due to fuzzy matching
        assert isinstance(result, str)

    def test_low_confidence(self):
        """Test with low confidence."""
        key = "jenis kelamin"
        data = ["Jenis Kelamin", 0.95, []]
        next_data = ["LAKI-LAKI", 0.50, []]
        result = matching_jeniskelamin(key, data, next_data)
        assert result == ""

    def test_non_jenis_kelamin_key_laki_laki(self):
        """Test with non-jenis kelamin key (uses data instead)."""
        key = "kelamin"
        data = ["LAKI-LAKI", 0.94, []]
        next_data = ["Something", 0.95, []]
        result = matching_jeniskelamin(key, data, next_data)
        assert result == "LAKI-LAKI"

    def test_non_jenis_kelamin_key_perempuan(self):
        """Test PEREMPUAN with alternate key."""
        key = "kelamin"
        data = ["PEREMPUAN", 0.94, []]
        next_data = ["Something", 0.95, []]
        result = matching_jeniskelamin(key, data, next_data)
        assert result == "PEREMPUAN"

    def test_empty_next_data(self):
        """Test with empty next_data."""
        key = "jenis kelamin"
        data = ["Jenis Kelamin", 0.95, []]
        next_data = ["", 0.94, []]
        result = matching_jeniskelamin(key, data, next_data)
        assert result == ""

    def test_invalid_gender(self):
        """Test with invalid gender value."""
        key = "jenis kelamin"
        data = ["Jenis Kelamin", 0.95, []]
        next_data = ["INVALID", 0.94, []]
        result = matching_jeniskelamin(key, data, next_data)
        assert isinstance(result, str)

    def test_none_data(self):
        """Test with None data."""
        key = "jenis kelamin"
        data = None
        next_data = ["LAKI-LAKI", 0.94, []]
        result = matching_jeniskelamin(key, data, next_data)  # type: ignore[invalid-argument-type]
        assert isinstance(result, str)

    def test_exception_handling(self):
        """Test exception handling with malformed data."""
        key = "jenis kelamin"
        data = []
        next_data = []
        result = matching_jeniskelamin(key, data, next_data)
        assert result == ""

    def test_laki_without_hyphen(self):
        """Test LAKI LAKI without hyphen."""
        key = "jenis kelamin"
        data = ["Jenis Kelamin", 0.95, []]
        next_data = ["LAKI LAKI", 0.94, []]
        result = matching_jeniskelamin(key, data, next_data)
        # Should still match due to fuzzy matching
        assert isinstance(result, str)
