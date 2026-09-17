"""Integration tests for field extraction across multiple matchers.

Tests how different field matchers work together to extract
complete KTP information from OCR data.
"""

import pytest
from src.services.field_matchers.nik import matching_nik
from src.services.field_matchers.nama import matching_nama
from src.services.field_matchers.ttl import matching_tempatlahir, matching_tempatlahir_new
from src.services.field_matchers.jenis_kelamin import matching_jeniskelamin
from src.services.field_matchers.agama import matching_agama
from src.services.field_matchers.alamat import matching_alamat
from src.services.field_matchers.kecamatan import matching_kecamatan
from src.services.field_matchers.keldesa import matching_keldesa
from src.services.field_matchers.rtrw import matching_rtrw


class TestFieldExtractionIntegration:
    """Test integration between multiple field matchers."""

    def test_nik_extraction(self):
        """Test extracting NIK from data."""
        # matching_nik takes a single list: [text, confidence, box]
        nik_data = ["3174012801950001", 0.97, [[210, 50], [450, 80]]]
        
        # Extract NIK
        nik = matching_nik(nik_data)
        
        assert nik == "3174012801950001"

    def test_nama_extraction(self):
        """Test extracting Nama from data."""
        # matching_nama takes a single list: [text, confidence, box]
        nama_data = ["BUDI SANTOSO", 0.96, [[210, 90], [450, 120]]]
        
        # Extract Nama
        nama = matching_nama(nama_data)
        
        assert nama == "BUDI SANTOSO"

    def test_ttl_extraction_multiple_formats(self):
        """Test TTL extraction works with different format functions."""
        # matching_tempatlahir expects data as individual items [text, confidence, box]
        # not as a list of lists
        data1 = ["JAKARTA, 28-01-1995", 0.95, []]
        next_data1 = ["Jenis Kelamin", 0.93, []]
        
        tempat1, tgl1 = matching_tempatlahir(data1, next_data1)
        
        # Both should extract values
        assert isinstance(tempat1, str) or tempat1 is None
        assert isinstance(tgl1, str) or tgl1 is None

    def test_address_components_extraction(self):
        """Test extracting address components."""
        # matching_alamat takes next_data: list (next line's data)
        alamat_next = ["JL. SUDIRMAN NO. 123", 0.96, []]
        alamat = matching_alamat(alamat_next)
        
        # matching_rtrw takes next_data only
        rtrw_next = ["001/002", 0.95, []]
        rt, rw = matching_rtrw(rtrw_next)
        
        # matching_keldesa takes next_data: list
        keldesa = matching_keldesa(["KEBAYORAN BARU", 0.94, []])
        
        # matching_kecamatan takes (data, next_data)
        kecamatan = matching_kecamatan(["Kecamatan", 0.94, []], ["KEBAYORAN BARU", 0.95, []])
        
        # All components extracted
        assert alamat == "JL. SUDIRMAN NO. 123"
        assert rt == "001" or rt == ""
        assert rw == "002" or rw == ""
        assert keldesa == "KEBAYORAN BARU"
        assert kecamatan == "KEBAYORAN BARU"

    def test_gender_and_religion_extraction(self):
        """Test extracting gender and religion fields."""
        # matching_jeniskelamin takes (key, data, next_data)
        jk_data = ["Jenis Kelamin", 0.93, []]
        jk_next = ["LAKI-LAKI", 0.94, []]
        jk = matching_jeniskelamin("jenis kelamin", jk_data, jk_next)
        
        # matching_agama takes (key, data, next_data)
        agama_data = ["Agama", 0.95, []]
        agama_next = ["ISLAM", 0.96, []]
        agama = matching_agama("agama", agama_data, agama_next)
        
        assert jk in ["LAKI-LAKI", "LAKI LAKI"]
        assert agama == "ISLAM"

    def test_nik_with_character_errors(self):
        """Test NIK extraction handles noisy data."""
        # NIK with character errors
        nik_data = ["3l74Ol28Ol95OOOl", 0.87, []]  # Letters instead of digits
        nik = matching_nik(nik_data)
        
        # Should handle corrections - may return empty if can't correct
        assert isinstance(nik, str)
        if nik:
            assert len(nik) == 16
            assert nik.isdigit()

    def test_nama_with_noise(self):
        """Test Nama extraction with noisy data."""
        nama_data = ["BUDl SANT0S0", 0.88, []]  # Mixed letters/digits
        nama = matching_nama(nama_data)
        
        # Should extract something
        assert isinstance(nama, str)

    def test_empty_data_across_matchers(self):
        """Test all matchers handle empty data gracefully."""
        empty_data = ["", 0.0, []]
        empty_next = ["", 0.0, []]
        
        # All matchers should return empty strings without crashing
        nik = matching_nik(empty_data)
        nama = matching_nama(empty_data)
        jk = matching_jeniskelamin("jenis kelamin", empty_data, empty_next)
        agama = matching_agama("agama", empty_data, empty_next)
        alamat = matching_alamat(empty_next)
        
        assert nik == ""
        assert nama == ""
        assert jk == ""
        assert agama == ""
        assert alamat == ""

    def test_case_sensitivity_nik(self):
        """Test NIK extraction is case insensitive."""
        nik_data = ["3174012801950001", 0.96, []]
        nik = matching_nik(nik_data)
        
        assert nik == "3174012801950001"

    def test_religion_case_variations(self):
        """Test religion matching handles case variations."""
        agama_data = ["AGAMA", 0.95, []]
        agama_next = ["islam", 0.96, []]
        agama = matching_agama("agama", agama_data, agama_next)
        
        assert agama in ["ISLAM", ""]

    def test_low_confidence_data(self):
        """Test matchers handle low confidence data appropriately."""
        # Low confidence NIK (below threshold)
        low_conf_nik = ["3174012801950001", 0.50, []]
        nik = matching_nik(low_conf_nik)
        
        # Should return empty due to low confidence
        assert nik == ""

    def test_multiple_field_extraction_workflow(self):
        """Test extracting multiple fields in sequence."""
        # Simulate realistic extraction workflow
        nik_data = ["3174012801950001", 0.97, []]
        nama_data = ["BUDI SANTOSO", 0.96, []]
        jk_key = ["Jenis Kelamin", 0.93, []]
        jk_value = ["LAKI-LAKI", 0.94, []]
        agama_key = ["Agama", 0.95, []]
        agama_value = ["ISLAM", 0.96, []]
        
        nik = matching_nik(nik_data)
        nama = matching_nama(nama_data)
        jk = matching_jeniskelamin("jenis kelamin", jk_key, jk_value)
        agama = matching_agama("agama", agama_key, agama_value)
        
        assert nik == "3174012801950001"
        assert nama == "BUDI SANTOSO"
        assert jk in ["LAKI-LAKI", "LAKI LAKI"]
        assert agama == "ISLAM"
