"""Cover the `except Exception` branches in each field matcher."""

from unittest.mock import patch

import pytest

from src.services.field_matchers.agama import matching_agama
from src.services.field_matchers.alamat import matching_alamat
from src.services.field_matchers.jenis_kelamin import matching_jeniskelamin
from src.services.field_matchers.keldesa import matching_keldesa
from src.services.field_matchers.nama import matching_nama
from src.services.field_matchers.nik import matching_nik
from src.services.field_matchers.rtrw import rtrw_sesuai_format


def test_matching_agama_exception_returns_empty():
    with patch(
        "src.services.field_matchers.agama.clean_colon",
        side_effect=RuntimeError("boom"),
    ):
        assert matching_agama("agama", ["Agama", 0.95, []], ["ISLAM", 0.95, []]) == ""


def test_matching_alamat_exception_returns_empty():
    with patch(
        "src.services.field_matchers.alamat.clean_colon",
        side_effect=RuntimeError("boom"),
    ):
        assert matching_alamat(["JL", 0.95, []]) == ""


def test_matching_jeniskelamin_exception_returns_empty():
    with patch(
        "src.services.field_matchers.jenis_kelamin.clean_colon",
        side_effect=RuntimeError("boom"),
    ):
        assert matching_jeniskelamin("jenis kelamin", ["JK", 0.95, []], ["LAKI-LAKI", 0.95, []]) == ""


def test_matching_keldesa_exception_returns_empty():
    with patch(
        "src.services.field_matchers.keldesa.clean_colon",
        side_effect=RuntimeError("boom"),
    ):
        assert matching_keldesa(["TANAH ABANG", 0.95, []]) == ""


def test_matching_nama_exception_returns_empty():
    with patch(
        "src.services.field_matchers.nama.clean_colon",
        side_effect=RuntimeError("boom"),
    ):
        assert matching_nama(["BUDI", 0.95, []]) == ""


def test_matching_nik_exception_returns_empty():
    with patch(
        "src.services.field_matchers.nik.clean_colon",
        side_effect=RuntimeError("boom"),
    ):
        assert matching_nik(["3174012345678901", 0.95, []]) == ""


def test_rtrw_sesuai_format_exception_returns_empty_tuple():
    """Pass a non-string to force a TypeError inside the try block."""
    assert rtrw_sesuai_format(None) == ("", "")  # type: ignore[invalid-argument-type]
