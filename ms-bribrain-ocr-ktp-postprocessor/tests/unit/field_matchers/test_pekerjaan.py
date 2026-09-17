"""Tests for Pekerjaan field matcher."""

import pytest

from src.services.field_matchers.pekerjaan import matching_pekerjaan


# Common reusable fixtures
EMPTY_PREV = ["", 0, []]
LABEL_CELL = ["Pekerjaan", 0.95, []]
KEWARGA_NEXT = ["Kewarganegaraan", 0.95, []]
KEWARGA_NEXT_NOISY = ["Kewargamegaraan WiIr", 0.88, []]
EMPTY_NEXT = ["", 0, []]


class TestMatchingPekerjaanSeparateCell:
    """Skenario value berada di cell terpisah dari label."""

    def test_value_only_uppercase(self):
        next_data = ["GURU", 0.96, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, LABEL_CELL, next_data) == "GURU"

    def test_value_with_colon_prefix(self):
        next_data = [":GURU", 0.96, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, LABEL_CELL, next_data) == "GURU"

    def test_value_with_colon_and_space(self):
        next_data = [": GURU", 0.96, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, LABEL_CELL, next_data) == "GURU"

    def test_multi_word_value(self):
        next_data = ["KARYAWAN SWASTA", 0.95, []]
        assert (
            matching_pekerjaan("pekerjaan", EMPTY_PREV, LABEL_CELL, next_data) == "KARYAWAN SWASTA"
        )

    def test_multi_word_value_with_colon(self):
        next_data = [":KARYAWAN SWASTA", 0.95, []]
        assert (
            matching_pekerjaan("pekerjaan", EMPTY_PREV, LABEL_CELL, next_data) == "KARYAWAN SWASTA"
        )

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("PNS", "PNS"),
            ("TNI", "TNI"),
            ("POLRI", "POLRI"),
            ("WIRASWASTA", "WIRASWASTA"),
            ("PETANI/PEKEBUN", "PETANI/PEKEBUN"),
            ("IBU RUMAH TANGGA", "IBU RUMAH TANGGA"),
            ("PELAJAR/MAHASISWA", "PELAJAR/MAHASISWA"),
            ("PENSIUNAN", "PENSIUNAN"),
            ("DOKTER", "DOKTER"),
            ("DOSEN", "DOSEN"),
        ],
    )
    def test_known_pekerjaan_values(self, value, expected):
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, LABEL_CELL, [value, 0.96, []]) == expected


class TestMatchingPekerjaanCombinedCell:
    """Skenario label dan value digabung di satu cell — next_data berisi label
    field berikutnya (kewarganegaraan), sehingga matcher mengekstrak dari `data`.
    """

    def test_label_and_value_with_colon(self):
        data = ["Pekerjaan:GURU", 0.95, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, data, KEWARGA_NEXT) == "GURU"

    def test_label_and_value_with_colon_space(self):
        data = ["Pekerjaan: GURU", 0.95, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, data, KEWARGA_NEXT) == "GURU"

    def test_label_and_value_lowercase_with_colon(self):
        data = ["pekerjaan:guru", 0.95, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, data, KEWARGA_NEXT) == "GURU"

    def test_label_typo_missing_a(self):
        data = ["pekerjan:guru", 0.95, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, data, KEWARGA_NEXT) == "GURU"

    def test_label_typo_missing_p(self):
        data = ["ekerjaan:guru", 0.95, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, data, KEWARGA_NEXT) == "GURU"

    def test_label_and_value_no_separator_lowercase(self):
        data = ["pekerjaanguru", 0.95, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, data, KEWARGA_NEXT) == "GURU"

    def test_label_lowercase_value_uppercase_no_separator(self):
        data = ["PekerjaanGURU", 0.95, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, data, KEWARGA_NEXT) == "GURU"

    def test_combined_cell_multi_word_value(self):
        data = ["Pekerjaan: KARYAWAN SWASTA", 0.95, []]
        assert (
            matching_pekerjaan("pekerjaan", EMPTY_PREV, data, KEWARGA_NEXT) == "KARYAWAN SWASTA"
        )

    def test_combined_cell_no_separator_multi_word(self):
        data = ["PekerjaanKARYAWAN SWASTA", 0.95, []]
        assert (
            matching_pekerjaan("pekerjaan", EMPTY_PREV, data, KEWARGA_NEXT) == "KARYAWAN SWASTA"
        )

    def test_label_typo_swapped_chars(self):
        data = ["Peerkjaan: GURU", 0.95, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, data, KEWARGA_NEXT) == "GURU"

    def test_label_with_extra_char(self):
        data = ["Pekerjaaan: GURU", 0.95, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, data, KEWARGA_NEXT) == "GURU"

    def test_combined_with_different_pekerjaan(self):
        assert (
            matching_pekerjaan("pekerjaan", EMPTY_PREV, ["Pekerjaan:PNS", 0.95, []], KEWARGA_NEXT)
            == "PNS"
        )
        assert (
            matching_pekerjaan(
                "pekerjaan", EMPTY_PREV, ["pekerjaan:wiraswasta", 0.95, []], KEWARGA_NEXT
            )
            == "WIRASWASTA"
        )


class TestMatchingPekerjaanKewarganegaraanDetection:
    """Skenario di mana next_data berisi label kewarganegaraan."""

    def test_real_ocr_user_sample(self):
        data = ["PekerjaanGURU", 0.99, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, data, KEWARGA_NEXT_NOISY) == "GURU"

    def test_kewarganegaraan_clean(self):
        data = ["Pekerjaan:GURU", 0.95, []]
        next_data = ["Kewarganegaraan", 0.95, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, data, next_data) == "GURU"

    def test_kewarganegaraan_with_value_appended(self):
        data = ["PekerjaanGURU", 0.99, []]
        next_data = ["Kewarganegaraan WNI", 0.95, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, data, next_data) == "GURU"

    def test_kewarganegaraan_typo(self):
        data = ["PekerjaanGURU", 0.99, []]
        next_data = ["Kewargnegaraan", 0.92, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, data, next_data) == "GURU"

    def test_next_data_is_value_not_kewarganegaraan(self):
        data = ["Pekerjaan", 0.95, []]
        next_data = ["GURU", 0.96, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, data, next_data) == "GURU"


class TestMatchingPekerjaanFuzzyValue:
    """Skenario nilai pekerjaan dengan OCR error pada bagian value."""

    def test_value_with_trailing_punctuation(self):
        assert (
            matching_pekerjaan("pekerjaan", EMPTY_PREV, LABEL_CELL, ["GURU.", 0.95, []]) == "GURU"
        )

    def test_value_with_typo_extra_char(self):
        assert (
            matching_pekerjaan("pekerjaan", EMPTY_PREV, LABEL_CELL, ["GURUU", 0.95, []]) == "GURU"
        )

    def test_karyawan_swasta_with_typo(self):
        assert (
            matching_pekerjaan("pekerjaan", EMPTY_PREV, LABEL_CELL, ["KARYAWAN SWASTAA", 0.95, []])
            == "KARYAWAN SWASTA"
        )

    def test_value_below_ratio_threshold_returns_empty(self):
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, LABEL_CELL, ["XYZ123", 0.95, []]) == ""


class TestMatchingPekerjaanEdgeCases:
    """Edge case dan defensive handling."""

    def test_low_confidence_next_data_returns_empty(self):
        next_data = ["GURU", 0.50, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, LABEL_CELL, next_data) == ""

    def test_low_confidence_data_with_kewarganegaraan_next_returns_empty(self):
        data = ["PekerjaanGURU", 0.50, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, data, KEWARGA_NEXT) == ""

    def test_empty_next_text(self):
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, LABEL_CELL, ["", 0.96, []]) == ""

    def test_whitespace_only(self):
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, LABEL_CELL, ["   ", 0.96, []]) == ""

    def test_invalid_pekerjaan_value(self):
        assert (
            matching_pekerjaan("pekerjaan", EMPTY_PREV, LABEL_CELL, ["INVALID_OCCUPATION", 0.96, []])
            == ""
        )

    def test_none_next_data(self):
        result = matching_pekerjaan("pekerjaan", EMPTY_PREV, LABEL_CELL, None)  # type: ignore[arg-type]
        assert result == ""

    def test_all_empty(self):
        assert matching_pekerjaan("pekerjaan", [], [], []) == ""

    def test_confidence_at_threshold_boundary(self):
        next_data = ["GURU", 0.80, []]
        assert matching_pekerjaan("pekerjaan", EMPTY_PREV, LABEL_CELL, next_data) == ""


class TestMatchingPekerjaanCatA_PrevCellFallback:
    """Cat A: value muncul di cell SEBELUM label (anomali urutan OCR)."""

    def test_prev_cell_contains_value(self):
        # Real failing case: ["PELAJAR/MAHASISWA", "Pekerjaan", "WNI", "Kewarganegaraan"]
        prev_data = ["PELAJAR/MAHASISWA", 0.95, []]
        next_data = ["WNI", 0.95, []]
        assert (
            matching_pekerjaan("pekerjaan", prev_data, LABEL_CELL, next_data) == "PELAJAR/MAHASISWA"
        )

    def test_prev_cell_with_colon_prefix(self):
        # Real failing case: [":KARYAWAN SWASTA", "Pekerjaan", ":WNI", ...]
        prev_data = [":KARYAWAN SWASTA", 0.95, []]
        next_data = [":WNI", 0.95, []]
        assert (
            matching_pekerjaan("pekerjaan", prev_data, LABEL_CELL, next_data) == "KARYAWAN SWASTA"
        )

    def test_prev_cell_belum_tidak_bekerja(self):
        # Real failing case: ["BELUM/TIDAK BEKERJA", "Pekerjaan", "WNI", ...]
        prev_data = ["BELUM/TIDAK BEKERJA", 0.95, []]
        next_data = ["WNI", 0.95, []]
        assert (
            matching_pekerjaan("pekerjaan", prev_data, LABEL_CELL, next_data)
            == "BELUM/TIDAK BEKERJA"
        )

    def test_prev_cell_concat_no_space(self):
        # Real failing case: [":MENGURUSRUMAH TANGGA", ...]
        prev_data = [":MENGURUSRUMAH TANGGA", 0.95, []]
        next_data = ["Kewarganegaraan:WNI", 0.95, []]
        # Even with kewarganegaraan in next_data (which would normally pull from data),
        # the data cell ("Pekerjaan") yields nothing, so it falls back to prev_data.
        assert (
            matching_pekerjaan("pekerjaan", prev_data, LABEL_CELL, next_data)
            == "MENGURUS RUMAH TANGGA"
        )

    def test_prev_cell_belum_tidak_bekerja_concat(self):
        # Real failing case: [":BELUM/TIDAKBEKERJA", "Pekerjaan", "Kewarganegaraan:WNI", ...]
        prev_data = [":BELUM/TIDAKBEKERJA", 0.95, []]
        next_data = ["Kewarganegaraan:WNI", 0.95, []]
        assert (
            matching_pekerjaan("pekerjaan", prev_data, LABEL_CELL, next_data)
            == "BELUM/TIDAK BEKERJA"
        )

    def test_prev_cell_ignored_when_next_has_value(self):
        # Sanity: jika next_data valid, prev_data harus tidak digunakan
        prev_data = ["BURUH", 0.95, []]
        next_data = ["GURU", 0.95, []]
        assert matching_pekerjaan("pekerjaan", prev_data, LABEL_CELL, next_data) == "GURU"

    def test_prev_cell_with_status_perkawinan_does_not_match(self):
        # Sanity: prev_data yang bukan pekerjaan tidak boleh false-match
        prev_data = ["Status Perkawinan KAWIN", 0.95, []]
        next_data = ["XYZ", 0.95, []]
        # Tidak ada match valid di mana pun → return ""
        assert matching_pekerjaan("pekerjaan", prev_data, LABEL_CELL, next_data) == ""


class TestMatchingPekerjaanOfficialDukcapilList:
    """106 jenis pekerjaan resmi dari Dukcapil — semua harus resolve ke dirinya
    sendiri ketika dibaca dari OCR sebagai value murni."""

    OFFICIAL_106 = [
        "BELUM/TIDAK BEKERJA", "MENGURUS RUMAH TANGGA", "PELAJAR/MAHASISWA",
        "PENSIUNAN", "APARATUR SIPIL NEGARA (ASN)", "TENTARA NASIONAL INDONESIA",
        "KEPOLISIAN RI (POLRI)", "PERDAGANGAN", "PETANI/PEKEBUN", "PETERNAK",
        "NELAYAN/PERIKANAN", "INDUSTRI", "KONSTRUKSI", "TRANSPORTASI",
        "KARYAWAN SWASTA", "KARYAWAN BUMN", "KARYAWAN BUMD", "KARYAWAN HONORER",
        "BURUH HARIAN LEPAS", "BURUH TANI/PERKEBUNAN", "BURUH NELAYAN/PERIKANAN",
        "BURUH PETERNAKAN", "PEMBANTU RUMAH TANGGA", "TUKANG CUKUR",
        "TUKANG LISTRIK", "TUKANG BATU", "TUKANG KAYU", "TUKANG SOL SEPATU",
        "TUKANG LAS/PANDAI BESI", "TUKANG JAHIT", "TUKANG GIGI", "PENATA RIAS",
        "PENATA BUSANA", "PENATA RAMBUT", "MEKANIK", "SENIMAN", "TABIB",
        "PARAJI", "PERANCANG BUSANA", "PENERJEMAH", "IMAM MASJID", "PENDETA",
        "PASTOR", "WARTAWAN", "USTADZ/MUBALIGH", "JURU MASAK", "PROMOTOR ACARA",
        "ANGGOTA DPR-RI", "ANGGOTA DPD", "ANGGOTA BPK", "PRESIDEN",
        "WAKIL PRESIDEN", "ANGGOTA MAHKAMAH KONSTITUSI",
        "ANGGOTA KABINET/KEMENTERIAN", "DUTA BESAR/KEPALA PERWAKILAN",
        "GUBERNUR", "WAKIL GUBERNUR", "BUPATI", "WAKIL BUPATI", "WALIKOTA",
        "WAKIL WALIKOTA", "ANGGOTA DPRD PROVINSI", "ANGGOTA DPRD KAB/KOTA",
        "DOSEN", "GURU", "PILOT", "PENGACARA", "NOTARIS", "ARSITEK", "AKUNTAN",
        "KONSULTAN", "DOKTER", "BIDAN", "PERAWAT", "APOTEKER",
        "PSIKIATER/PSIKOLOG", "PENYIAR TELEVISI", "PENYIAR RADIO", "PELAUT",
        "PENELITI", "SOPIR", "PIALANG", "PARANORMAL", "PEDAGANG",
        "PERANGKAT DESA", "KEPALA DESA", "BIARAWATI", "WIRASWASTA", "ARTIS",
        "ATLET", "CHEF", "MANAJER", "TENAGA TATA USAHA", "OPERATOR",
        "PEKERJA PENGOLAHAN, KERAJINAN", "TEKNISI", "ASISTEN AHLI", "GEMBALA",
        "USKUP", "BIARAWAN", "PANDITA", "PINANDITA", "BHIKKHU", "XUESHI",
        "WENSHI", "JIAOSHENG",
    ]

    def test_count_is_106(self):
        assert len(self.OFFICIAL_106) == 106

    @pytest.mark.parametrize("value", OFFICIAL_106)
    def test_each_official_value_resolves_to_itself(self, value):
        assert (
            matching_pekerjaan("pekerjaan", EMPTY_PREV, LABEL_CELL, [value, 0.95, []]) == value
        )


class TestMatchingPekerjaanCatB_AddedMappings:
    """Cat B: value yang sebelumnya tidak ada di mapping_pekerjaan."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("BURUH HARIAN LEPAS", "BURUH HARIAN LEPAS"),
            (":BURUH HARIAN LEPAS", "BURUH HARIAN LEPAS"),
            ("BURUHHARIAN LEPAS", "BURUH HARIAN LEPAS"),  # OCR concat
            ("BURUH TANI/PERKEBUNAN", "BURUH TANI/PERKEBUNAN"),
            (":BURUH TANI/PERKEBUNAN", "BURUH TANI/PERKEBUNAN"),
            ("PENDETA", "PENDETA"),
            ("TRANSPORTASI", "TRANSPORTASI"),
            (":TRANSPORTASI", "TRANSPORTASI"),
        ],
    )
    def test_newly_added_mappings(self, value, expected):
        assert (
            matching_pekerjaan("pekerjaan", EMPTY_PREV, LABEL_CELL, [value, 0.95, []]) == expected
        )


class TestMatchingPekerjaanValueKeyDirect:
    """Saat keydat yang match adalah VALUE pekerjaan (bukan label "pekerjaan"),
    value sudah ada di `data` itu sendiri — mirroring pola status_perkawinan."""

    def test_value_key_resolves_from_data(self):
        # keydat "nelayan/perikanan" match → value ada langsung di `data`.
        data = ["NELAYAN/PERIKANAN", 0.95, []]
        assert (
            matching_pekerjaan("nelayan/perikanan", EMPTY_PREV, data, EMPTY_NEXT)
            == "NELAYAN/PERIKANAN"
        )

    def test_value_key_with_colon_prefix(self):
        data = [":NELAYAN/PERIKANAN", 0.95, []]
        assert (
            matching_pekerjaan("nelayan/perikanan", EMPTY_PREV, data, EMPTY_NEXT)
            == "NELAYAN/PERIKANAN"
        )

    def test_value_key_concat_no_space(self):
        data = ["PELAJARMAHASISWA", 0.95, []]
        assert (
            matching_pekerjaan("pelajar/mahasiswa", EMPTY_PREV, data, EMPTY_NEXT)
            == "PELAJAR/MAHASISWA"
        )

    def test_value_key_low_confidence_returns_empty(self):
        data = ["NELAYAN/PERIKANAN", 0.50, []]
        assert (
            matching_pekerjaan("nelayan/perikanan", EMPTY_PREV, data, EMPTY_NEXT)
            == ""
        )

    def test_value_key_ignores_next_data(self):
        # Saat key adalah value, prev/next harus diabaikan.
        data = ["GURU", 0.95, []]
        next_data = ["BERLAKU HINGGA", 0.95, []]
        prev_data = ["RANDOM", 0.95, []]
        assert matching_pekerjaan("guru", prev_data, data, next_data) == "GURU"


class TestMatchingPekerjaanPipelineRemapping:
    """Pipeline integration untuk kasus di mana label "pekerjaan" gagal terdeteksi
    di mappingnext (OCR error pada label) dan harus di-recover oleh remapping."""

    def test_remap_recovers_value_when_label_below_ratio(self):
        """File 1 user: label "Pokorjaan" ratio<80, value "NELAYAN/PERIKANAN" ada
        di token berikutnya. Remap harus pick up via value-key match."""
        from src.services.ocr_processor import mappingnext

        ocr = [
            [[[57, 128], [143, 169]], ("NIK", 0.99)],
            [[[235, 132], [687, 169]], ("3523162404010005", 0.98)],
            [[[59, 188], [134, 217]], ("Pokorjaan", 0.89)],
            [[[266, 191], [409, 216]], (":NELAYAN/PERIKANAN", 0.95)],
            [[[57, 442], [175, 468]], ("Kewarganegaraan:WNI", 0.90)],
            [[[55, 495], [228, 526]], ("Berlaku Hingga", 0.91)],
            [[[262, 498], [455, 525]], ("SEUMUR HIDUP", 0.99)],
        ]
        result, _ = mappingnext(ocr)
        assert result.get("pekerjaan") == "NELAYAN/PERIKANAN"

    def test_remap_recovers_value_when_label_below_confidence(self):
        """File 2 user: label "Pekerjasn" conf 0.75<0.80, value "PELAJARMAHASISWA"
        di token sebelumnya (Cat A). Remap harus pick up via value-key match."""
        from src.services.ocr_processor import mappingnext

        ocr = [
            [[[57, 128], [143, 169]], ("NIK", 0.99)],
            [[[235, 132], [687, 169]], ("6303050907950009", 0.98)],
            [[[59, 188], [134, 217]], ("Status Perkawinan:BELUM KAWIN", 0.93)],
            [[[266, 191], [409, 216]], ("PELAJARMAHASISWA", 0.97)],
            [[[57, 442], [175, 468]], ("Pekerjasn", 0.75)],
            [[[262, 498], [455, 525]], ("Berlaku Hingga", 0.89)],
            [[[262, 498], [455, 525]], ("SEUMUR HIDUP", 0.99)],
        ]
        result, _ = mappingnext(ocr)
        assert result.get("pekerjaan") == "PELAJAR/MAHASISWA"


class TestMatchingPekerjaanPipelineIntegration:
    """Integrasi via pipeline mappingnext untuk skenario realistis."""

    def _build_ocr(self, label_text: str, value_text: str | None = None):
        items = [
            [[[100, 50], [200, 80]], ("NIK", 0.96)],
            [[[210, 50], [450, 80]], ("3174012801950001", 0.97)],
            [[[100, 90], [200, 120]], (label_text, 0.95)],
        ]
        if value_text is not None:
            items.append([[[210, 90], [450, 120]], (value_text, 0.95)])
        items.append([[[100, 130], [200, 160]], ("Berlaku Hingga", 0.94)])
        items.append([[[210, 130], [450, 160]], ("SEUMUR HIDUP", 0.94)])
        return items

    def test_pipeline_separate_cell_normal(self):
        from src.services.ocr_processor import mappingnext

        data = self._build_ocr("Pekerjaan", "KARYAWAN SWASTA")
        result, _ = mappingnext(data)
        assert result.get("pekerjaan") == "KARYAWAN SWASTA"

    def test_pipeline_separate_cell_with_colon_value(self):
        from src.services.ocr_processor import mappingnext

        data = self._build_ocr("Pekerjaan", ":GURU")
        result, _ = mappingnext(data)
        assert result.get("pekerjaan") == "GURU"

    def test_pipeline_label_typo_missing_letter(self):
        from src.services.ocr_processor import mappingnext

        data = self._build_ocr("Pekerjan", "GURU")
        result, _ = mappingnext(data)
        assert result.get("pekerjaan") == "GURU"

    def test_pipeline_label_extra_letter(self):
        from src.services.ocr_processor import mappingnext

        data = self._build_ocr("Pekerjaann", "GURU")
        result, _ = mappingnext(data)
        assert result.get("pekerjaan") == "GURU"

    def test_pipeline_user_real_ocr_sample(self):
        """Sample asli user: cell pekerjaan ter-merge + cell berikutnya kewarganegaraan."""
        from src.services.ocr_processor import mappingnext

        ocr = [
            [[[57, 128], [143, 128], [143, 169], [57, 169]], ["NIK", 0.99]],
            [[[235, 132], [687, 135], [687, 169], [235, 165]], ["3175015805850007", 0.99]],
            [[[59, 188], [134, 190], [133, 217], [58, 214]], ["Nama", 0.99]],
            [[[266, 191], [409, 191], [409, 216], [266, 216]], ["ARIA SANTI", 0.99]],
            [[[57, 442], [175, 442], [175, 468], [57, 468]], ["PekerjaanGURU", 0.99]],
            [[[264, 442], [513, 442], [513, 466], [264, 466]], ["Kewargamegaraan WiIr", 0.88]],
            [[[55, 495], [228, 498], [228, 526], [54, 524]], ["Berlaku Hingga", 0.91]],
            [[[262, 498], [455, 496], [456, 523], [262, 525]], ["SEUMUR HIDUP", 0.99]],
        ]
        result, _ = mappingnext(ocr)
        assert result.get("pekerjaan") == "GURU"

    def test_pipeline_cat_a_value_before_label(self):
        """Cat A: value cell muncul SEBELUM label cell di urutan OCR."""
        from src.services.ocr_processor import mappingnext

        ocr = [
            [[[100, 50], [200, 80]], ("NIK", 0.96)],
            [[[210, 50], [450, 80]], ("3174012801950001", 0.97)],
            # Value cell muncul DULUAN
            [[[100, 90], [300, 120]], ("PELAJAR/MAHASISWA", 0.95)],
            [[[100, 130], [200, 160]], ("Pekerjaan", 0.95)],
            [[[210, 130], [300, 160]], ("WNI", 0.95)],
            [[[100, 170], [300, 200]], ("Kewarganegaraan", 0.94)],
            [[[100, 210], [200, 240]], ("Berlaku Hingga", 0.94)],
            [[[210, 210], [400, 240]], ("SEUMUR HIDUP", 0.94)],
        ]
        result, _ = mappingnext(ocr)
        assert result.get("pekerjaan") == "PELAJAR/MAHASISWA"

    def test_pipeline_cat_b_buruh_harian_lepas(self):
        """Cat B: value baru di mapping_pekerjaan."""
        from src.services.ocr_processor import mappingnext

        data = self._build_ocr("Pekerjaan", "BURUH HARIAN LEPAS")
        result, _ = mappingnext(data)
        assert result.get("pekerjaan") == "BURUH HARIAN LEPAS"

    def test_pipeline_cat_b_pendeta(self):
        from src.services.ocr_processor import mappingnext

        data = self._build_ocr("Pekerjaan", "PENDETA")
        result, _ = mappingnext(data)
        assert result.get("pekerjaan") == "PENDETA"

    def test_pipeline_cat_c_colon_only_skip(self):
        """Cat C: next_data hanya berisi ":" — pipeline harus lompat ke i+2."""
        from src.services.ocr_processor import mappingnext

        ocr = [
            [[[100, 50], [200, 80]], ("NIK", 0.96)],
            [[[210, 50], [450, 80]], ("3174012801950001", 0.97)],
            [[[100, 90], [200, 120]], ("Pekerjaan", 0.95)],
            [[[210, 90], [220, 120]], (":", 0.95)],
            [[[230, 90], [400, 120]], ("KARYAWAN SWASTA", 0.95)],
            [[[100, 130], [200, 160]], ("Berlaku Hingga", 0.94)],
            [[[210, 130], [400, 160]], ("SEUMUR HIDUP", 0.94)],
        ]
        result, _ = mappingnext(ocr)
        assert result.get("pekerjaan") == "KARYAWAN SWASTA"
