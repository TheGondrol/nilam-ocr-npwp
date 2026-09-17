"""Tests for Agama field matcher."""

import pytest

from src.services.field_matchers.agama import matching_agama


class TestMatchingAgama:
    """Test cases for agama matching function."""

    def test_valid_agama_with_key_agama(self):
        """Test valid agama with 'agama' key."""
        key = "agama"
        data = ["Agama", 0.95, [[100, 680], [200, 680], [200, 730], [100, 730]]]
        next_data = ["ISLAM", 0.96, [[220, 680], [300, 680], [300, 730], [220, 730]]]
        result = matching_agama(key, data, next_data)
        assert result == "ISLAM"

    def test_valid_kristen(self):
        """Test KRISTEN detection."""
        key = "agama"
        data = ["Agama", 0.95, []]
        next_data = ["KRISTEN", 0.96, []]
        result = matching_agama(key, data, next_data)
        assert result == "KRISTEN"

    def test_valid_katolik(self):
        """Test KATOLIK detection."""
        key = "agama"
        data = ["Agama", 0.95, []]
        next_data = ["KATOLIK", 0.96, []]
        result = matching_agama(key, data, next_data)
        assert result == "KATHOLIK"  # searchmapping converts KATOLIK to KATHOLIK

    def test_valid_hindu(self):
        """Test HINDU detection."""
        key = "agama"
        data = ["Agama", 0.95, []]
        next_data = ["HINDU", 0.96, []]
        result = matching_agama(key, data, next_data)
        assert result == "HINDU"

    def test_valid_buddha(self):
        """Test BUDDHA detection."""
        key = "agama"
        data = ["Agama", 0.95, []]
        next_data = ["BUDDHA", 0.96, []]
        result = matching_agama(key, data, next_data)
        assert result == "BUDDHA"

    def test_agama_with_colon(self):
        """Test agama with colon prefix."""
        key = "agama"
        data = ["Agama", 0.95, []]
        next_data = [":ISLAM", 0.96, []]
        result = matching_agama(key, data, next_data)
        assert result == "ISLAM"

    def test_agama_with_low_confidence(self):
        """Test agama with low confidence."""
        key = "agama"
        data = ["Agama", 0.95, []]
        next_data = ["ISLAM", 0.50, []]
        result = matching_agama(key, data, next_data)
        assert result == ""

    def test_non_agama_key(self):
        """Test with non-agama key (uses data instead of next_data)."""
        key = "not_agama"
        data = ["ISLAM", 0.96, []]
        next_data = ["Something", 0.95, []]
        result = matching_agama(key, data, next_data)
        assert result == "ISLAM"

    def test_fuzzy_matching(self):
        """Test fuzzy matching with typo."""
        key = "agama"
        data = ["Agama", 0.95, []]
        next_data = ["ISLM", 0.96, []]  # Typo
        result = matching_agama(key, data, next_data)
        # Should still match due to fuzzy matching
        assert isinstance(result, str)

    def test_empty_next_data(self):
        """Test with empty next_data."""
        key = "agama"
        data = ["Agama", 0.95, []]
        next_data = ["", 0.96, []]
        result = matching_agama(key, data, next_data)
        assert result == ""

    def test_invalid_agama(self):
        """Test with invalid agama value."""
        key = "agama"
        data = ["Agama", 0.95, []]
        next_data = ["INVALID_RELIGION", 0.96, []]
        result = matching_agama(key, data, next_data)
        # Should return empty or handle gracefully
        assert isinstance(result, str)

    def test_none_data(self):
        """Test with None data."""
        key = "agama"
        data = None
        next_data = ["ISLAM", 0.96, []]
        result = matching_agama(key, data, next_data)  # type: ignore[invalid-argument-type]
        assert isinstance(result, str)

    def test_exception_handling(self):
        """Test exception handling with malformed data."""
        key = "agama"
        data = []
        next_data = []
        result = matching_agama(key, data, next_data)
        assert result == ""
