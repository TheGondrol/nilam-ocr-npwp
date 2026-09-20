"""
Backend `npwp_rules`: aturan regex + posisi dari ML engineer (src/vendor/npwp_rules), dirangkai ke
kontrak structurer kita. Data ujinya dua hasil OCR kartu NPWP ASLI:

- kartu model baru: response service model ekstraksi (tests/fixtures/remote_npwp_response.json),
  nama tanpa label, nomor 15 digit DAN "NPWP16" 16 digit;
- kartu model lama: hasil PaddleOCR atas foto kartu kuning, "NPWP:" berlabel tapi nama tetap tanpa label.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from ocr_common.errors import ServiceError
from src.models.structuring import NpwpRulesStructurer, RuleBasedNpwpStructurer, get_structurer

_SAMPLE = json.loads((Path(__file__).parent / "fixtures" / "remote_npwp_response.json").read_text(encoding="utf-8"))[0]


def _bbox(poly) -> dict[str, float]:
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}


NEW_CARD: list[dict[str, Any]] = [
    {"text": text, "confidence": round(score, 4), "bbox": _bbox(poly), "page": 0}
    for text, score, poly in zip(_SAMPLE["rec_texts"], _SAMPLE["rec_scores"], _SAMPLE["rec_polys"], strict=True)
]
OLD_CARD: list[dict[str, Any]] = [
    {"text": text, "confidence": conf, "bbox": {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}, "page": 0}
    for text, conf, b in [
        ("DIREKTORAT JENDERAL PAJAK", 0.9985, (175, 247, 887, 308)),
        ("NPWP:48.903.841.4-722.000", 0.9992, (35, 389, 637, 437)),
        ("TOTOK WIJAYANTO, SSI.", 0.9773, (33, 480, 483, 523)),
        ("JL. SUBULUS SALAM NO.017 RT.29/00", 0.9747, (31, 520, 632, 563)),
        ("SIDOMULYO, SAMARINDAILIR", 0.9765, (31, 558, 522, 601)),
        ("SAMARINDA-KALIMANTANTIMUR 75116", 0.9887, (28, 590, 691, 638)),
        ("TANGGAL TERDAFTAR: 07/11/2007", 0.9831, (23, 678, 493, 715)),
    ]
]


def _line(text: str, top: int, confidence: float = 0.98, page: int = 0) -> dict[str, Any]:
    return {
        "text": text,
        "confidence": confidence,
        "bbox": {"x1": 40, "y1": top, "x2": 900, "y2": top + 40},
        "page": page,
    }


def structure(lines, **kwargs):
    return NpwpRulesStructurer(**kwargs).structure(lines)


def test_default_backend_is_npwp_rules():
    assert get_structurer().name == "npwp_rules"


def test_new_card_name_without_label_is_found_by_position():
    fields = structure(NEW_CARD)
    assert fields["nama"] == {
        "value": "RAHMAT HIDAYAT",
        "confidence": 0.9931,
        "source": "RAHMAT HIDAYAT",
        "signals": {"corrected": False},
    }
    assert fields["nama_badan"] == {"value": None, "confidence": 0.0, "source": None, "signals": None}


def test_new_card_prefers_the_16_digit_number_over_the_legacy_15_digit_one():
    """Kartu yang mencetak keduanya: nomor 16 digit (label NPWP16) adalah nomor resmi yang berlaku."""
    number = structure(NEW_CARD)["nomor_npwp"]
    assert number["value"] == "4318085607040052"
    assert number["source"] == "NPWP16:4318 0856 07040052"
    assert number["confidence"] == 0.9762


def test_signals_for_the_scoring_payload_come_from_the_ml_rules():
    """npwp_has_homoglyph, npwp_candidate_count, name_corrected di payload scoring ML engineer."""
    number = structure(NEW_CARD)["nomor_npwp"]["signals"]
    assert number == {"has_homoglyph": False, "candidate_count": 2}  # 15 digit + NPWP16 di kartu yang sama
    assert structure(OLD_CARD)["nomor_npwp"]["signals"] == {"has_homoglyph": False, "candidate_count": 1}

    corrected = structure(
        [_line("KPP PRATAMA TEGAL", 40), _line("63.48O.341.5-5O1.000", 120), _line("SRI WAHYUNI", 200)]
    )
    assert corrected["nomor_npwp"]["signals"]["has_homoglyph"] is True
    # name_master.py masih pengganti sementara: koreksi nama tidak pernah terjadi.
    assert corrected["nama"]["signals"] == {"corrected": False}


def test_old_card_keeps_the_dotted_15_digit_format_and_title_suffix():
    fields = structure(OLD_CARD)
    assert fields["nomor_npwp"]["value"] == "48.903.841.4-722.000"
    assert fields["nama"]["value"] == "TOTOK WIJAYANTO, SSI."  # gelar menempel dengan koma tetap lolos


def test_this_is_exactly_what_the_label_based_backend_misses():
    """Alasan backend ini dipasang: pada kedua kartu asli, rule_based tidak menemukan nama."""
    for card in (NEW_CARD, OLD_CARD):
        assert RuleBasedNpwpStructurer().structure(card)["nama"]["value"] is None
        assert structure(card)["nama"]["value"] is not None


def test_without_bbox_position_falls_back_to_reading_order():
    lines = [{"text": line["text"], "confidence": line["confidence"]} for line in NEW_CARD]
    fields = structure(lines)
    assert (fields["nomor_npwp"]["value"], fields["nama"]["value"]) == ("4318085607040052", "RAHMAT HIDAYAT")


def test_company_name_goes_to_nama_badan():
    fields = structure(
        [
            _line("KPP PRATAMA JAKARTA MENTENG", 40),
            _line("01.234.567.8-091.000", 120),
            _line("PT CIPTA KARYA INDONESIA", 200),  # "INDONESIA" kata boilerplate, tapi baris PT/CV dikecualikan
            _line("JL. MERDEKA NO. 12", 280),
        ]
    )
    assert fields["nama_badan"]["value"] == "PT CIPTA KARYA INDONESIA"
    assert fields["nama"]["value"] is None


def test_ocr_homoglyphs_in_the_number_are_corrected():
    fields = structure([_line("KPP PRATAMA TEGAL", 40), _line("63.48O.341.5-5O1.000", 120), _line("SRI WAHYUNI", 200)])
    assert fields["nomor_npwp"]["value"] == "63.480.341.5-501.000"


def test_number_with_an_unresolvable_character_is_not_reported_as_a_shorter_valid_number():
    """'T' di luar posisi pertama tidak bisa dipastikan 1 atau 7, jadi dibuang aturan ML engineer. Nomor 16
    digit yang tinggal 15 digit akan LOLOS validasi sebagai nomor 15 digit: lebih baik tidak dilaporkan."""
    fields = structure([_line("KPP PRATAMA TEGAL", 40), _line("3329 1T30 1001 0006", 120), _line("SRI WAHYUNI", 200)])
    assert fields["nomor_npwp"]["value"] is None
    assert fields["nama"]["value"] == "SRI WAHYUNI"  # deteksi per field, bukan semua-atau-tidak


def test_address_and_office_lines_never_become_the_name():
    fields = structure(
        [
            _line("npwp", 40),
            _line("12.345.678.9-012.345", 120),
            _line("KOTA TEGAL", 200),
            _line("KPP PRATAMA TEGAL", 260),
            _line("ADITYA ARDELLO PRATAMA", 330),  # "PRATAMA" adalah nama orang yang sah
        ]
    )
    assert fields["nama"]["value"] == "ADITYA ARDELLO PRATAMA"


def test_low_confidence_fragment_near_the_number_does_not_win():
    fields = structure(
        [
            _line("12.345.678.9-012.345", 120),
            _line("CamScan", 150, confidence=0.55),  # watermark: lebih dekat, tapi di bawah NAME_MIN_SCORE
            _line("BUDI SANTOSO", 220),
        ]
    )
    assert fields["nama"]["value"] == "BUDI SANTOSO"


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (
            [_line("PROVINSI JAWA BARAT", 40), _line("KARTU TANDA PENDUDUK", 90)],
            "another document (KARTU TANDA PENDUDUK)",
        ),
        ([_line("KARTU KELUARGA", 40)], "another document (KARTU KELUARGA)"),
        ([_line("Masukkan kode CAPTCHA", 40)], "CAPTCHA challenge"),
        ([_line("Cek NPWP", 40), _line("Nama: (disamarkan)", 90)], "screenshot of the DJP NPWP lookup"),
        ([_line("npwp", 40, page=p) for p in range(3)], "Upload has 3 pages"),
    ],
)
def test_page_guardrails_reject_non_npwp_uploads(lines, message):
    with pytest.raises(ServiceError) as exc:
        structure(lines)
    assert exc.value.status_code == 400
    assert message in exc.value.message


def test_page_guardrails_can_be_turned_off():
    lines = [_line("KARTU KELUARGA", 40), _line("12.345.678.9-012.345", 120), _line("BUDI SANTOSO", 200)]
    assert structure(lines, page_guardrails=False)["nomor_npwp"]["value"] == "12.345.678.9-012.345"


def test_two_page_upload_takes_number_and_name_across_pages():
    fields = structure(
        [_line("12.345.678.9-012.345", 120, page=0), _line("BUDI SANTOSO", 200, page=0), _line("djp", 40, page=1)]
    )
    assert (fields["nomor_npwp"]["value"], fields["nama"]["value"]) == ("12.345.678.9-012.345", "BUDI SANTOSO")


def test_nothing_recognisable_returns_all_fields_null_not_an_error():
    fields = structure([_line("lorem ipsum 123", 40)])
    assert [f["value"] for f in fields.values()] == [None, None, None]


def test_http_structure_with_real_card(client, auth):
    response = client.post("/v1/structuring/structure", json={"lines": NEW_CARD}, headers=auth)
    assert response.status_code == 200, response.text
    fields = response.json()["data"]["fields"]
    assert fields["nomor_npwp"]["value"] == "4318085607040052"
    assert fields["nama"]["value"] == "RAHMAT HIDAYAT"


def test_http_bundled_document_is_400(client, auth):
    lines = [*NEW_CARD, _line("KARTU TANDA PENDUDUK", 1200)]
    response = client.post("/v1/structuring/structure", json={"lines": lines}, headers=auth)
    assert response.status_code == 400
    assert "another document (KARTU TANDA PENDUDUK)" in response.json()["message"]
