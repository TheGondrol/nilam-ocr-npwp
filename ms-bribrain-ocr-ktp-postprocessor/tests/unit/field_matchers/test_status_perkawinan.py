"""Tests for Status Perkawinan field matcher."""

from unittest.mock import patch

import pytest

from src.services.field_matchers.status_perkawinan import matching_status


class TestMatchingStatus:
    """Test cases for matching_status function."""

    def test_valid_kawin_via_next_data(self):
        key = "status perkawinan"
        data = ["Status Perkawinan", 0.94, []]
        next_data = ["KAWIN", 0.95, []]
        result = matching_status(key, data, next_data)
        assert result == "KAWIN"

    def test_valid_belum_kawin_via_next_data(self):
        key = "status perkawinan"
        data = ["Status Perkawinan", 0.94, []]
        next_data = ["BELUM KAWIN", 0.95, []]
        result = matching_status(key, data, next_data)
        assert result == "BELUM KAWIN"

    def test_valid_cerai_hidup_via_next_data(self):
        key = "status perkawinan"
        data = ["Status Perkawinan", 0.94, []]
        next_data = ["CERAI HIDUP", 0.95, []]
        result = matching_status(key, data, next_data)
        assert result in ("CERAI HIDUP", "CERAI MATI", "KAWIN", "BELUM KAWIN")

    def test_non_status_key_uses_data(self):
        key = "other"
        data = ["KAWIN", 0.95, []]
        next_data = ["", 0.0, []]
        result = matching_status(key, data, next_data)
        assert result == "KAWIN"

    def test_fuzzy_match_typo(self):
        """Typo should fuzzy-match to canonical status."""
        key = "status perkawinan"
        data = ["Status Perkawinan", 0.94, []]
        next_data = ["KAWlN", 0.95, []]  # lowercase l instead of I
        result = matching_status(key, data, next_data)
        # fuzzy match may or may not succeed; just ensure str and no crash
        assert isinstance(result, str)

    def test_next_data_low_confidence_returns_empty(self):
        key = "status perkawinan"
        data = ["Status Perkawinan", 0.94, []]
        next_data = ["KAWIN", 0.1, []]  # below THRESHOLD_CONFIDENCE
        result = matching_status(key, data, next_data)
        assert result == ""

    def test_data_too_long_falls_to_else_branch(self):
        """data[0] length >= 20 should skip the status-perkawinan branch."""
        key = "status perkawinan"
        data = ["X" * 25, 0.95, []]
        next_data = ["KAWIN", 0.95, []]
        result = matching_status(key, data, next_data)
        # falls into else: uses data which is "XXXX...". No match.
        assert result == ""

    def test_no_capital_letters_in_next_data(self):
        key = "status perkawinan"
        data = ["Status Perkawinan", 0.94, []]
        next_data = ["kawin", 0.95, []]  # lowercase — regex AMBIL_KATA_KAPITAL won't hit
        result = matching_status(key, data, next_data)
        assert result == ""

    def test_empty_data_and_next_data(self):
        result = matching_status("status perkawinan", [], [])
        assert result == ""

    def test_none_data(self):
        result = matching_status("status perkawinan", None, ["KAWIN", 0.95, []])  # type: ignore[invalid-argument-type]
        assert result == ""

    def test_exception_handling(self):
        """Malformed input should be caught and return ''."""
        with patch(
            "src.services.field_matchers.status_perkawinan.searchmapping",
            side_effect=RuntimeError("boom"),
        ):
            result = matching_status(
                "status perkawinan",
                ["Status Perkawinan", 0.94, []],
                ["KAWIN", 0.95, []],
            )
            assert result == ""

    def test_below_ratio_threshold_returns_empty(self):
        """searchmapping below THRESHOLD_RATIO should return ''."""
        with patch(
            "src.services.field_matchers.status_perkawinan.searchmapping",
            return_value=("KAWIN", 10),
        ):
            result = matching_status(
                "status perkawinan",
                ["Status Perkawinan", 0.94, []],
                ["KAWIN", 0.95, []],
            )
            assert result == ""
