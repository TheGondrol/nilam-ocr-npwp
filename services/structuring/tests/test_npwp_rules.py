import json
from pathlib import Path

import pytest

from ocr_common.types import BoundingBox, OcrBlock

from app.dependencies import get_structurer
from app.ml import npwp_rules
from app.ml.npwp_rules import NpwpRulesStructurer
from app.ml.rule_based import RuleBasedNpwpStructurer

_SAMPLE = json.loads((Path(__file__).parent / "fixtures" / "remote_npwp_response.json").read_text(encoding="utf-8"))[0]


def _bbox(poly) -> BoundingBox:
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}


NEW_CARD: list[OcrBlock] = [
    {"text": text, "confidence": round(score, 4), "bbox": _bbox(poly), "page": 0}
    for text, score, poly in zip(_SAMPLE["rec_texts"], _SAMPLE["rec_scores"], _SAMPLE["rec_polys"], strict=True)
]
OLD_CARD: list[OcrBlock] = [
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
NO_NUMBER_SIGNAL = {
    "has_homoglyph": False,
    "invalid_province_prefix": False,
    "invalid_kecamatan_prefix": False,
    "invalid_birthdate": False,
    "invalid_kpp_prefix": False,
}


def _line(text: str, top: int, confidence: float = 0.98, page: int = 0) -> OcrBlock:
    return {
        "text": text,
        "confidence": confidence,
        "bbox": {"x1": 40, "y1": top, "x2": 900, "y2": top + 40},
        "page": page,
    }


def _card(number: str, name: str = "SRI WAHYUNI") -> list[OcrBlock]:
    return [_line("KPP PRATAMA TEGAL", 40), _line(number, 120), _line(name, 200)]


def structure(lines):
    return NpwpRulesStructurer().structure(lines)


def fields(lines):
    return structure(lines)["fields"]


def test_default_backend_is_npwp_rules():
    assert get_structurer().name == "npwp_rules"


def test_new_card_name_without_label_is_found_by_position():
    result = fields(NEW_CARD)
    assert result["nama"] == {
        "value": "RAHMAT HIDAYAT",
        "confidence": 0.9931,
        "source": "RAHMAT HIDAYAT",
        "signals": {"name_base": "RAHMAT HIDAYAT", "corrected": False},
    }
    assert result["nama_badan"] == {"value": None, "confidence": 0.0, "source": None, "signals": None}


def test_new_card_prefers_the_16_digit_number_over_the_legacy_15_digit_one():
    number = fields(NEW_CARD)["nomor_npwp"]
    assert number["value"] == "4318085607040052"
    assert number["source"] == "NPWP16:4318 0856 07040052"
    assert number["confidence"] == 0.9762
    assert number["signals"]["candidate_count"] == 2


def test_old_card_keeps_the_dotted_15_digit_format_and_title_suffix_and_is_not_flagged():
    document = structure(OLD_CARD)
    assert document["fields"]["nomor_npwp"]["value"] == "48.903.841.4-722.000"
    assert document["fields"]["nomor_npwp"]["signals"] == {"candidate_count": 1, **NO_NUMBER_SIGNAL}
    assert document["fields"]["nama"]["value"] == "TOTOK WIJAYANTO, SSI."
    assert (document["flag"], document["flag_reason"]) == (False, None)


def test_this_is_exactly_what_the_label_based_backend_misses():
    for card in (NEW_CARD, OLD_CARD):
        assert RuleBasedNpwpStructurer().structure(card)["fields"]["nama"]["value"] is None
        assert fields(card)["nama"]["value"] is not None


def test_without_bbox_position_falls_back_to_reading_order():
    lines = [{"text": line["text"], "confidence": line["confidence"]} for line in NEW_CARD]
    result = fields(lines)
    assert (result["nomor_npwp"]["value"], result["nama"]["value"]) == ("4318085607040052", "RAHMAT HIDAYAT")


def test_company_name_goes_to_nama_badan():
    result = fields(
        [
            _line("KPP PRATAMA JAKARTA MENTENG", 40),
            _line("01.234.567.8-091.000", 120),
            _line("PT CIPTA KARYA INDONESIA", 200),
            _line("JL. MERDEKA NO. 12", 280),
        ]
    )
    assert result["nama_badan"]["value"] == "PT CIPTA KARYA INDONESIA"
    assert result["nama"]["value"] is None


def test_name_value_is_normalised_but_name_base_is_the_raw_read():
    name = fields([_line("12.345.678.9-012.345", 120), _line("RATNA SUSANTI/SYAMARIS MULYODI", 200)])["nama"]
    assert name["value"] == "RATNA SUSANTI / SYAMARIS MULYODI"
    assert name["source"] == "RATNA SUSANTI/SYAMARIS MULYODI"
    assert name["signals"] == {"name_base": "RATNA SUSANTI/SYAMARIS MULYODI", "corrected": True}


# --- the number is not corrected; what OCR misread is flagged ----------------------------------


def test_homoglyph_is_dropped_not_corrected_and_flagged():
    document = structure(_card("63.48O.341.5-5O1.000"))
    number = document["fields"]["nomor_npwp"]
    assert number["value"] == "6348341551000", "the two O are dropped, not read as 0 (correction is off)"
    assert number["signals"]["has_homoglyph"] is True
    assert (document["flag"], document["flag_reason"]) == (True, npwp_rules.FLAG_HOMOGLYPH)
    assert document["fields"]["nama"]["value"] == "SRI WAHYUNI"


def test_an_unresolved_t_is_dropped_and_flagged_not_reported_as_a_valid_15_digit_number():
    document = structure(_card("3329 1T30 1001 0006"))
    number = document["fields"]["nomor_npwp"]
    assert number["value"] == "332913010010006", "15 digits left, but not the printed 15-digit shape: no dots"
    assert number["signals"]["has_homoglyph"] is True
    assert document["flag_reason"] == npwp_rules.FLAG_HOMOGLYPH


def test_a_t_in_the_province_prefix_is_dropped_too_while_correction_is_off():
    # The rules can resolve "T7.." to "17.." (Bengkulu), but the ML team wires normalize_npwp_raw, not
    # normalize_npwp, so the T is dropped like any other letter and the number is flagged.
    number = fields(_card("T701 0130 1001 0006"))["nomor_npwp"]
    assert number["value"] == "701013010010006"
    assert number["signals"]["has_homoglyph"] is True


def test_invalid_province_code_is_flagged_for_a_16_digit_number():
    document = structure(_card("7729 0130 1001 0006"))
    assert document["fields"]["nomor_npwp"]["signals"]["invalid_province_prefix"] is True
    assert (document["flag"], document["flag_reason"]) == (True, npwp_rules.FLAG_INVALID_PROVINCE)


def test_invalid_birthdate_is_flagged_for_a_16_digit_number():
    document = structure(_card("3329 0199 3001 0006"))
    assert document["fields"]["nomor_npwp"]["signals"]["invalid_birthdate"] is True
    assert document["flag_reason"] == npwp_rules.FLAG_INVALID_BIRTHDATE


def test_birthdate_of_a_woman_adds_40_to_the_day():
    assert structure(_card("3329 0157 0190 0006"))["flag"] is False


def test_kecamatan_check_needs_the_kode_wilayah_table(tmp_path, monkeypatch):
    number = "3329 0130 1001 0006"
    monkeypatch.setenv("WILAYAH_CODES_PATH", str(tmp_path / "tidak-ada.json"))
    assert fields(_card(number))["nomor_npwp"]["signals"]["invalid_kecamatan_prefix"] is False, "no table: no signal"

    table = tmp_path / "kode_wilayah.json"
    table.write_text(json.dumps({"kecamatan": {"330101": "Contoh"}}), encoding="utf-8")
    monkeypatch.setenv("WILAYAH_CODES_PATH", str(table))
    document = structure(_card(number))
    assert document["fields"]["nomor_npwp"]["signals"]["invalid_kecamatan_prefix"] is True
    assert document["flag_reason"] == npwp_rules.FLAG_INVALID_KECAMATAN
    assert structure(_card("3301 0130 1001 0006"))["flag"] is False


def test_kpp_check_needs_the_kpp_table(tmp_path, monkeypatch):
    number = "48.903.841.4-722.000"
    monkeypatch.setenv("KPP_CODES_PATH", str(tmp_path / "tidak-ada.json"))
    assert fields(_card(number))["nomor_npwp"]["signals"]["invalid_kpp_prefix"] is False, "no table: no signal"

    table = tmp_path / "kpp_codes.json"
    table.write_text(json.dumps({"012": "KPP Pratama Contoh"}), encoding="utf-8")
    monkeypatch.setenv("KPP_CODES_PATH", str(table))
    document = structure(_card(number))
    assert document["fields"]["nomor_npwp"]["signals"]["invalid_kpp_prefix"] is True
    assert document["flag_reason"] == npwp_rules.FLAG_INVALID_KPP
    assert structure(_card("48.903.841.4-012.000"))["flag"] is False


def test_the_delivered_reference_tables_are_used_by_default():
    """kode_wilayah.json (7230 kecamatan) and kpp_codes.json (173 offices) of the ML team live in data/."""
    assert structure(_card("3301 0130 1001 0006"))["flag"] is False
    assert structure(_card("3399 9930 1001 0006"))["flag_reason"] == npwp_rules.FLAG_INVALID_KECAMATAN
    assert structure(_card("48.903.841.4-012.000"))["flag"] is False
    assert structure(_card("48.903.841.4-999.000"))["flag_reason"] == npwp_rules.FLAG_INVALID_KPP


def test_recognised_name_from_the_name_master_wins_the_tie_break(tmp_path, monkeypatch):
    from app.vendor.npwp_rules import name_master

    lines = [_line("12.345.678.9-012.345", 120), _line("BUDI SANTOSA", 150), _line("BUDI SANTOSO", 180)]
    assert fields(lines)["nama"]["value"] == "BUDI SANTOSA", "without the master the nearest line wins"

    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    workbook.active.append(["name"])
    workbook.active.append(["BUDI SANTOSO"])
    master = tmp_path / "name_lnmast.xlsx"
    workbook.save(master)
    monkeypatch.setenv("NAME_MASTER_PATH", str(master))
    name_master._load_index.cache_clear()
    try:
        assert fields(lines)["nama"]["value"] == "BUDI SANTOSO"
    finally:
        name_master._load_index.cache_clear()


# --- every check flags; all but two also reject (ML team, 23 Sep 2026) --------------------------


@pytest.mark.parametrize(
    ("lines", "reason"),
    [
        (
            [_line("PROVINSI JAWA BARAT", 40), _line("KARTU TANDA PENDUDUK", 90), *_card("12.345.678.9-012.345")],
            "Dokumen lain terdeteksi: 'KARTU TANDA PENDUDUK' pada halaman 1",
        ),
        ([_line("Masukkan kode CAPTCHA", 40)], npwp_rules.FLAG_CAPTCHA),
        ([_line("Cek NPWP", 40), _line("Nama: (disamarkan)", 90)], npwp_rules.FLAG_WEB_LOOKUP),
        ([_line("npwp", 40, page=p) for p in range(3)], "Dokumen memiliki 3 halaman (lebih dari 2)"),
        ([_line("lorem ipsum 123", 40)], npwp_rules.FLAG_BLANK),
        (_card("7729 0130 1001 0006"), npwp_rules.FLAG_INVALID_PROVINCE),
        (_card("3329 0199 3001 0006"), npwp_rules.FLAG_INVALID_BIRTHDATE),
        (_card("3399 9930 1001 0006"), npwp_rules.FLAG_INVALID_KECAMATAN),
        (_card("48.903.841.4-999.000"), npwp_rules.FLAG_INVALID_KPP),
    ],
)
def test_rejecting_checks_flag_and_reject(lines, reason):
    document = structure(lines)
    assert document["flag"] is True
    assert reason in document["flag_reason"]
    assert reason in document["reject_reason"]
    assert set(document["fields"]) == {"nomor_npwp", "nama", "nama_badan"}, "the fields are still read"


@pytest.mark.parametrize(
    ("lines", "reason"),
    [
        ([_line("12.345.678.9-012.345", 120), _line("SUKIRMAN", 200)], npwp_rules.FLAG_SINGLE_WORD_NAME),
        (_card("63.48O.341.5-5O1.000"), npwp_rules.FLAG_HOMOGLYPH),
    ],
)
def test_single_word_name_and_letter_in_the_number_are_tolerated(lines, reason):
    document = structure(lines)
    assert (document["flag"], document["flag_reason"]) == (True, reason), "still an input of the trust model"
    assert document["reject_reason"] is None


def test_a_clean_card_is_neither_flagged_nor_rejected():
    document = structure(_card("12.345.678.9-012.345", "BUDI SANTOSO"))
    assert (document["flag"], document["flag_reason"], document["reject_reason"]) == (False, None, None)


def test_a_tolerated_reason_that_outranks_a_rejecting_one_does_not_hide_it():
    # flag_reason keeps the rules' priority (single-word name first), but the document is still rejected
    # for its invalid province code, with that message.
    document = structure(_card("7729 0130 1001 0006", "SUKIRMAN"))
    assert document["flag_reason"] == npwp_rules.FLAG_SINGLE_WORD_NAME
    assert document["reject_reason"] == npwp_rules.FLAG_INVALID_PROVINCE


def test_the_first_reason_in_priority_order_is_reported():
    lines = [_line("KARTU KELUARGA", 40), *_card("63.48O.341.5-5O1.000", "SUKIRMAN")]
    document = structure(lines)
    assert document["flag_reason"].startswith("Dokumen lain terdeteksi")
    assert document["reject_reason"].startswith("Dokumen lain terdeteksi")


def test_address_and_office_lines_never_become_the_name():
    result = fields(
        [
            _line("npwp", 40),
            _line("12.345.678.9-012.345", 120),
            _line("KOTA TEGAL", 200),
            _line("KPP PRATAMA TEGAL", 260),
            _line("ADITYA ARDELLO PRATAMA", 330),
        ]
    )
    assert result["nama"]["value"] == "ADITYA ARDELLO PRATAMA"


def test_low_confidence_fragment_near_the_number_does_not_win():
    result = fields(
        [
            _line("12.345.678.9-012.345", 120),
            _line("CamScan", 150, confidence=0.55),
            _line("BUDI SANTOSO", 220),
        ]
    )
    assert result["nama"]["value"] == "BUDI SANTOSO"


def test_two_page_upload_takes_number_and_name_from_the_first_page_that_has_them():
    result = fields(
        [_line("12.345.678.9-012.345", 120, page=0), _line("BUDI SANTOSO", 200, page=0), _line("djp", 40, page=1)]
    )
    assert (result["nomor_npwp"]["value"], result["nama"]["value"]) == ("12.345.678.9-012.345", "BUDI SANTOSO")


def test_http_structure_with_real_card(client, auth):
    response = client.post("/v1/structuring/structure", json={"lines": NEW_CARD}, headers=auth)
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["fields"]["nomor_npwp"]["value"] == "4318085607040052"
    assert data["fields"]["nama"]["value"] == "RAHMAT HIDAYAT"
    assert {"flag", "flag_reason", "reject_reason"} <= set(data)


def test_http_bundled_document_is_200_with_its_reject_reason(client, auth):
    # The synchronous debugging endpoint reports the verdict; only the pipeline turns it into a 400.
    lines = [*NEW_CARD, _line("KARTU TANDA PENDUDUK", 1200)]
    response = client.post("/v1/structuring/structure", json={"lines": lines}, headers=auth)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["flag"] is True
    assert "KARTU TANDA PENDUDUK" in data["flag_reason"]
    assert "KARTU TANDA PENDUDUK" in data["reject_reason"]
    assert data["fields"]["nama"]["value"] == "RAHMAT HIDAYAT"
