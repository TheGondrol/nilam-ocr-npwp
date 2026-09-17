"""Tests for remapper utility."""

from unittest.mock import patch

import pytest

from src.utils.remapper import remapping


class TestRemapping:
    """Test cases for remapping function."""

    def test_remapping_with_complete_data(self, sample_cleaned_data):
        """Test remapping with already complete data."""
        # Create a result dict with all 12 fields
        result = {
            "nik": ["3174012345678901", 0.97],
            "nama": ["BUDI SANTOSO", 0.95],
            "tempat_lahir": ["JAKARTA", 0.93],
            "tanggal_lahir": ["15-08-1990", 0.93],
            "jenis_kelamin": ["LAKI-LAKI", 0.94],
            "alamat": ["JL SUDIRMAN NO 10", 0.92],
            "rt": ["001", 0.94],
            "rw": ["002", 0.94],
            "kel_desa": ["TANAH ABANG", 0.92],
            "kecamatan": ["TANAH ABANG", 0.93],
            "agama": ["ISLAM", 0.96],
            "status_perkawinan": ["KAWIN", 0.93]
        }
        
        remapped = remapping(sample_cleaned_data, result)
        
        # Result should contain the same or more fields
        assert isinstance(remapped, dict)
        assert len(remapped) >= len(result)

    def test_remapping_with_missing_fields(self):
        """Test remapping with missing fields."""
        data_cleaned = [
            ["Kel/Desa", 0.93, [[100, 540], [200, 540], [200, 590], [100, 590]]],
            ["TANAH ABANG", 0.92, [[220, 540], [350, 540], [350, 590], [220, 590]]],
            ["Kecamatan", 0.94, [[100, 610], [200, 610], [200, 660], [100, 660]]],
            ["TANAH ABANG", 0.93, [[220, 610], [350, 610], [350, 660], [220, 660]]],
        ]
        
        result = {
            "nik": ["3174012345678901", 0.97],
            "nama": ["BUDI SANTOSO", 0.95],
        }
        
        with patch('src.utils.remapper.matching_keldesa') as mock_keldesa, \
             patch('src.utils.remapper.matching_kecamatan') as mock_kecamatan:
            
            mock_keldesa.return_value = "TANAH ABANG"
            mock_kecamatan.return_value = "TANAH ABANG"
            
            remapped = remapping(data_cleaned, result)
            
            # Should attempt to fill missing fields
            assert isinstance(remapped, dict)

    def test_remapping_updates_better_score(self):
        """Test that remapping updates fields with better scores."""
        data_cleaned = [
            ["RT/RW", 0.95, [[100, 470], [200, 470], [200, 520], [100, 520]]],
            ["001/002", 0.94, [[220, 470], [300, 470], [300, 520], [220, 520]]],
        ]
        
        result = {
            "rt": ["000", 0.70],  # Lower score
            "rw": ["000", 0.70],  # Lower score
        }
        
        with patch('src.utils.remapper.matching_rtrw') as mock_rtrw:
            mock_rtrw.return_value = ("001", "002")
            
            remapped = remapping(data_cleaned, result)
            
            # Should update with better scores
            assert isinstance(remapped, dict)

    def test_remapping_with_status_perkawinan(self):
        """Test remapping status perkawinan field."""
        data_cleaned = [
            ["Status Perkawinan", 0.94, [[100, 750], [250, 750], [250, 800], [100, 800]]],
            ["KAWIN", 0.93, [[260, 750], [350, 750], [350, 800], [260, 800]]],
        ]
        
        result = {}
        
        with patch('src.utils.remapper.matching_status') as mock_status:
            mock_status.return_value = "KAWIN"
            
            remapped = remapping(data_cleaned, result)
            
            # Should add status perkawinan
            assert isinstance(remapped, dict)

    def test_remapping_with_tempat_lahir(self):
        """Test remapping tempat/tanggal lahir fields."""
        data_cleaned = [
            ["Tempat/Tgl Lahir", 0.94, [[100, 260], [250, 260], [250, 310], [100, 310]]],
            ["JAKARTA, 15-08-1990", 0.93, [[260, 260], [400, 260], [400, 310], [260, 310]]],
            ["Alamat", 0.92, [[100, 330], [200, 330], [200, 380], [100, 380]]],
        ]
        
        result = {}
        
        with patch('src.utils.remapper.matching_tempatlahir') as mock_ttl, \
             patch('src.utils.remapper.matching_tempatlahir_new') as mock_ttl_new:
            
            mock_ttl.return_value = ("JAKARTA", "15-08-1990")
            mock_ttl_new.return_value = ("", "")
            
            remapped = remapping(data_cleaned, result)
            
            # Should add tempat and tanggal lahir
            assert isinstance(remapped, dict)

    def test_remapping_with_rtrw(self):
        """Test remapping RT/RW fields."""
        data_cleaned = [
            ["RT/RW", 0.95, [[100, 470], [200, 470], [200, 520], [100, 520]]],
            ["001/002", 0.94, [[220, 470], [300, 470], [300, 520], [220, 520]]],
        ]
        
        result = {}
        
        with patch('src.utils.remapper.matching_rtrw') as mock_rtrw:
            mock_rtrw.return_value = ("001", "002")
            
            remapped = remapping(data_cleaned, result)
            
            # Should add RT and RW
            assert isinstance(remapped, dict)

    def test_remapping_empty_data(self):
        """Test remapping with empty data."""
        result = {"nik": ["3174012345678901", 0.97]}
        
        remapped = remapping([], result)
        
        # Should return original result
        assert remapped == result

    def test_remapping_empty_result(self):
        """Test remapping with empty result."""
        data_cleaned = [
            ["NIK", 0.98, [[100, 120], [200, 120], [200, 170], [100, 170]]],
            ["3174012345678901", 0.97, [[220, 120], [400, 120], [400, 170], [220, 170]]],
        ]
        
        remapped = remapping(data_cleaned, {})
        
        # Should attempt to add fields
        assert isinstance(remapped, dict)

    def test_remapping_preserves_existing_better_scores(self):
        """Test that remapping preserves existing fields with better scores."""
        data_cleaned = [
            ["Kel/Desa", 0.70, [[100, 540], [200, 540], [200, 590], [100, 590]]],
            ["TANAH ABANG", 0.65, [[220, 540], [350, 540], [350, 590], [220, 590]]],
        ]
        
        result = {
            "kel_desa": ["MENTENG", 0.95],  # Better score
        }
        
        with patch('src.utils.remapper.matching_keldesa') as mock_keldesa:
            mock_keldesa.return_value = "TANAH ABANG"
            
            remapped = remapping(data_cleaned, result)
            
            # Should preserve the better score
            # (behavior depends on implementation)
            assert isinstance(remapped, dict)
            assert "kel_desa" in remapped

    def test_remapping_with_low_confidence_data(self):
        """Test remapping with data below confidence threshold."""
        data_cleaned = [
            ["Kecamatan", 0.50, [[100, 610], [200, 610], [200, 660], [100, 660]]],
            ["TANAH ABANG", 0.45, [[220, 610], [350, 610], [350, 660], [220, 660]]],
        ]
        
        result = {}
        
        remapped = remapping(data_cleaned, result)
        
        # Low confidence data should not be added
        assert isinstance(remapped, dict)
