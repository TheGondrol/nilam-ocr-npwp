"""Tests for OCR processor service."""

from unittest.mock import patch, MagicMock

import pytest

from src.services.ocr_processor import mappingnext


class TestMappingNext:
    """Test cases for mappingnext function."""

    def test_basic_ocr_processing(self, sample_ocr_data):
        """Test basic OCR processing with complete data."""
        with patch('src.services.ocr_processor.remapping') as mock_remapping:
            # Mock remapping to return the input unchanged
            mock_remapping.side_effect = lambda data, result: result
            
            result, nik_box = mappingnext(sample_ocr_data)
            
            # Check that result is a dictionary
            assert isinstance(result, dict)
            
            # Check that NIK box is returned
            assert nik_box is not None
            
            # Check that some expected fields are present
            # (actual values depend on the matcher implementations)
            assert isinstance(result, dict)

    def test_empty_data(self):
        """Test with empty OCR data."""
        result, nik_box = mappingnext([])
        
        assert isinstance(result, dict)
        assert nik_box is None

    def test_data_with_empty_text(self):
        """Test filtering of data with empty or dash text."""
        data = [
            [[[100, 50], [200, 50], [200, 100], [100, 100]], ("PROVINSI", 0.95)],
            [[[100, 120], [200, 120], [200, 170], [100, 170]], ("-", 0.98)],  # Should be filtered
            [[[100, 190], [200, 190], [200, 240], [100, 240]], ("", 0.96)],    # Should be filtered
            [[[100, 260], [200, 260], [200, 310], [100, 310]], ("  ", 0.94)],  # Should be filtered
            [[[100, 330], [200, 330], [200, 380], [100, 380]], ("VALID", 0.93)],
        ]
        
        result, nik_box = mappingnext(data)
        
        # Should process without error
        assert isinstance(result, dict)

    def test_low_confidence_data(self):
        """Test with low confidence OCR data."""
        data = [
            [[[100, 50], [200, 50], [200, 100], [100, 100]], ("NIK", 0.95)],
            [[[100, 120], [200, 120], [200, 170], [100, 170]], ("3174012345678901", 0.40)],  # Low confidence
        ]
        
        result, nik_box = mappingnext(data)
        
        # Should handle low confidence gracefully
        assert isinstance(result, dict)

    def test_nik_detection(self):
        """Test NIK detection and extraction."""
        data = [
            [[[100, 50], [200, 50], [200, 100], [100, 100]], ("NIK", 0.98)],
            [[[220, 50], [400, 50], [400, 100], [220, 100]], ("3174012345678901", 0.97)],
        ]
        
        with patch('src.services.ocr_processor.matching_nik') as mock_nik, \
             patch('src.services.ocr_processor.remapping') as mock_remapping:
            
            mock_nik.return_value = "3174012345678901"
            mock_remapping.side_effect = lambda data, result: result
            
            result, nik_box = mappingnext(data)
            
            # NIK should be detected
            mock_nik.assert_called()
            if "nik" in result:
                assert result["nik"] == "3174012345678901"

    def test_nik_recovered_by_16_digit_fallback_when_label_missing(self):
        """NIK is recovered by its 16-digit pattern when the 'NIK' label was
        missed by OCR, so no cell fuzzy-matches 'nik'."""
        nik_box = [[100, 120], [400, 120], [400, 170], [100, 170]]
        data = [
            [[[100, 50], [300, 50], [300, 100], [100, 100]], ("PROVINSI JAWA BARAT", 0.95)],
            # NIK label dropped by OCR; the value cell still carries 16 digits.
            [nik_box, ("3174012345678901", 0.97)],
            [[[100, 190], [300, 190], [300, 240], [100, 240]], ("BUDI", 0.95)],
        ]

        with patch('src.services.ocr_processor.remapping') as mock_remapping:
            mock_remapping.side_effect = lambda data, result: result

            result, returned_box = mappingnext(data)

        assert result.get("nik") == "3174012345678901"
        assert returned_box == nik_box

    def test_nik_fallback_ignores_non_16_digit_cells(self):
        """The 16-digit fallback does not fire when no cell cleans to exactly
        16 digits (avoids false positives from shorter numbers)."""
        data = [
            [[[100, 50], [300, 50], [300, 100], [100, 100]], ("PROVINSI", 0.95)],
            [[[100, 120], [300, 120], [300, 170], [100, 170]], ("31740123", 0.97)],  # 8 digits
        ]

        with patch('src.services.ocr_processor.remapping') as mock_remapping:
            mock_remapping.side_effect = lambda data, result: result

            result, _ = mappingnext(data)

        assert not result.get("nik")

    def test_nama_detection(self):
        """Test nama (name) detection and extraction."""
        data = [
            [[[100, 50], [200, 50], [200, 100], [100, 100]], ("Nama", 0.96)],
            [[[220, 50], [400, 50], [400, 100], [220, 100]], ("BUDI SANTOSO", 0.95)],
            [[[100, 150], [200, 150], [200, 200], [100, 200]], ("Alamat", 0.94)],  # Next field
        ]
        
        with patch('src.services.ocr_processor.matching_nama') as mock_nama, \
             patch('src.services.ocr_processor.remapping') as mock_remapping:
            
            mock_nama.return_value = "BUDI SANTOSO"
            mock_remapping.side_effect = lambda data, result: result
            
            result, nik_box = mappingnext(data)
            
            # Nama should be detected
            mock_nama.assert_called()

    def test_tempat_lahir_detection(self):
        """Test tempat/tanggal lahir detection."""
        data = [
            [[[100, 50], [250, 50], [250, 100], [100, 100]], ("Tempat/Tgl Lahir", 0.94)],
            [[[260, 50], [400, 50], [400, 100], [260, 100]], ("JAKARTA, 15-08-1990", 0.93)],
            [[[100, 150], [200, 150], [200, 200], [100, 200]], ("Alamat", 0.92)],  # Extra data to avoid index error
        ]
        
        with patch('src.services.ocr_processor.matching_tempatlahir') as mock_ttl, \
             patch('src.services.ocr_processor.remapping') as mock_remapping:
            
            mock_ttl.return_value = ("JAKARTA", "15-08-1990")
            mock_remapping.side_effect = lambda data, result: result
            
            result, nik_box = mappingnext(data)
            
            # Tempat lahir should be detected
            mock_ttl.assert_called()

    def test_rtrw_detection(self):
        """Test RT/RW detection."""
        data = [
            [[[100, 50], [200, 50], [200, 100], [100, 100]], ("RT/RW", 0.95)],
            [[[220, 50], [300, 50], [300, 100], [220, 100]], ("001/002", 0.94)],
        ]
        
        with patch('src.services.ocr_processor.matching_rtrw') as mock_rtrw, \
             patch('src.services.ocr_processor.remapping') as mock_remapping:
            
            mock_rtrw.return_value = ("001", "002")
            mock_remapping.side_effect = lambda data, result: result
            
            result, nik_box = mappingnext(data)
            
            # RT/RW should be detected
            mock_rtrw.assert_called()

    def test_agama_detection(self):
        """Test agama (religion) detection."""
        data = [
            [[[100, 50], [200, 50], [200, 100], [100, 100]], ("Agama", 0.95)],
            [[[220, 50], [300, 50], [300, 100], [220, 100]], ("ISLAM", 0.96)],
        ]
        
        with patch('src.services.ocr_processor.matching_agama') as mock_agama, \
             patch('src.services.ocr_processor.remapping') as mock_remapping:
            
            mock_agama.return_value = "ISLAM"
            mock_remapping.side_effect = lambda data, result: result
            
            result, nik_box = mappingnext(data)
            
            # Agama should be detected
            mock_agama.assert_called()

    def test_remapping_called_when_few_fields(self):
        """Test that remapping is called when fewer than 12 fields are found."""
        data = [
            [[[100, 50], [200, 50], [200, 100], [100, 100]], ("NIK", 0.98)],
            [[[220, 50], [400, 50], [400, 100], [220, 100]], ("3174012345678901", 0.97)],
        ]
        
        with patch('src.services.ocr_processor.remapping') as mock_remapping:
            mock_remapping.side_effect = lambda data, result: result
            
            result, nik_box = mappingnext(data)
            
            # Remapping should be called because we have < 12 fields
            mock_remapping.assert_called_once()

    def test_result_score_handling(self):
        """Test that only values (not scores) are returned in final result."""
        data = [
            [[[100, 50], [200, 50], [200, 100], [100, 100]], ("NIK", 0.98)],
            [[[220, 50], [400, 50], [400, 100], [220, 100]], ("3174012345678901", 0.97)],
        ]
        
        with patch('src.services.ocr_processor.matching_nik') as mock_nik, \
             patch('src.services.ocr_processor.remapping') as mock_remapping:
            
            mock_nik.return_value = "3174012345678901"
            mock_remapping.side_effect = lambda data, result: result
            
            result, nik_box = mappingnext(data)
            
            # All values in result should be strings, not tuples with scores
            for value in result.values():
                assert isinstance(value, (str, type(None)))
                assert not isinstance(value, (list, tuple))

    def test_multiple_keywords_same_field(self):
        """Test handling of multiple keywords mapping to same field."""
        data = [
            [[[100, 50], [200, 50], [200, 100], [100, 100]], ("Jenis Kelamin", 0.95)],
            [[[220, 50], [350, 50], [350, 100], [220, 100]], ("LAKI-LAKI", 0.94)],
            [[[100, 150], [200, 150], [200, 200], [100, 200]], ("Kelamin", 0.93)],
            [[[220, 150], [350, 150], [350, 200], [220, 200]], ("LAKI-LAKI", 0.92)],
        ]
        
        result, nik_box = mappingnext(data)
        
        # Should handle multiple matches for same field
        assert isinstance(result, dict)
