"""Laporan PDF satu request (report.py) untuk setiap akhir request yang bisa terjadi: dari event stream yang
bentuknya sama dengan yang dipancarkan app.py. Jalankan: python -m pytest tools/tracker/backend"""

import io

import pytest
import report
from PIL import Image

RID = "REQ_0123456789ab"
T0 = 1_760_000_000.0

OCR = {
    "engine": "paddle",
    "model": "PP-OCRv5",
    "elapsed_ms": 812.4,
    "full_text": "NPWP\n01.234.567.8-901.000\nBUDI SANTOSO",
    "blocks": [
        {"text": "NPWP", "confidence": 0.99, "bbox": {"x1": 10, "y1": 10, "x2": 80, "y2": 30}, "page": 0},
        {"text": "01.234.567.8-901.000", "confidence": 0.97, "bbox": {"x1": 10, "y1": 40, "x2": 190, "y2": 60}},
        {"text": "BUDI SANTOSO → 中", "confidence": 0.62, "bbox": {"x1": 10, "y1": 70, "x2": 150, "y2": 90}},
    ],
}
STRUCTURING = {
    "fields": {
        "nomor_npwp": {
            "value": "01.234.567.8-901.000",
            "confidence": 0.97,
            "source": "regex",
            "signals": {"candidate_count": 1},
        },
        "nama": {
            "value": "BUDI SANTOSO",
            "confidence": 0.62,
            "source": "line_after_npwp",
            "signals": {"name_base": "BUDI SANTOSO"},
        },
        "nama_badan": {"value": None, "confidence": 0.0},
    },
    "flag": False,
    "flag_reason": None,
    "reject_reason": None,
}
SCORING = {
    "npwp_confidence": 0.9312,
    "name_confidence": 0.4123,
    "fields": {
        "nomor_npwp": {"value": "01.234.567.8-901.000", "confidence": 1, "threshold": 0.8},
        "nama": {"value": "BUDI SANTOSO", "confidence": 0.4123, "threshold": None},
    },
    "payload": {
        "npwp": "012345678901000",
        "npwp_score": 0.97,
        "npwp_candidate_count": 1,
        "name_base": "BUDI SANTOSO",
        "name_score": 0.62,
        "avg_doc_score": 0.86,
        "min_doc_score": 0.62,
        "flag": False,
        "guardrail_probability": 0.91,
    },
}
GUARDRAILS_ROW = {
    "passed": True,
    "verdict": "accepted",
    "confidence": 0.91,
    "threshold": 0.5,
    "threshold_source": "service",
    "n_pages": 1,
    "reason": None,
    "report": {
        "document": {
            "verdict": "accepted",
            "confidence": 0.91,
            "threshold": 0.5,
            "n_pages": 1,
            "n_approve": 1,
            "n_reject": 0,
        },
        "pages": [{"page_index": 0, "proba_approve": 0.91, "proba_reject": 0.09, "verdict": "accepted"}],
    },
}


def ev(dt: float, type_: str, stage: str, status: str, **fields):
    return {"type": type_, "request_id": RID, "stage": stage, "status": status, "ts": T0 + dt, **fields}


def client(sequence=None):
    return ev(
        0,
        "client",
        "CLIENT",
        "SUBMITTED",
        filename="npwp.jpg",
        size=48_000,
        sequence=sequence,
        source="upload",
        callback_mode="ok",
        slow=0,
    )


def http(status, *, errors=None, message="OCR extraction completed successfully", data=None, **fields):
    body = {"status_code": status, "message": message, "data": data, "errors": errors, "request_id": RID}
    return ev(
        1.5,
        "http",
        "GUARDRAILS",
        "RESPONSE",
        http_status=status,
        errors=errors,
        message=message,
        elapsed_ms=1500,
        wait_seconds=15,
        body=body,
        **fields,
    )


def completed_events():
    return [
        client(),
        ev(0.3, "stage", "OCR", "PROCESSING", source="db", attempt=1),
        ev(
            0.4,
            "outbox",
            "OUTBOX",
            "QUEUED",
            message={
                "id": 1,
                "owner": "OCR",
                "kind": "handoff",
                "target": "STRUCTURING",
                "message": "POST /v1/structuring/jobs",
                "attempts": 0,
                "last_error": None,
            },
        ),
        ev(1.1, "stage", "OCR", "DONE", source="db", attempt=1, result=OCR),
        ev(1.2, "stage", "STRUCTURING", "DONE", source="db", attempt=1, result=STRUCTURING),
        ev(1.4, "stage", "SCORING", "DONE", source="db", attempt=1, result=SCORING),
        http(200, data={"nomor_npwp": {"value": "01.234.567.8-901.000", "confidence": 1}}),
        ev(1.5, "stage", "GUARDRAILS", "DONE", elapsed_ms=1500),
        ev(
            1.6,
            "callback",
            "SCORING",
            "DONE",
            attempt=1,
            accepted=True,
            duplicate=False,
            http_status=200,
            format="result",
            final=True,
            contract={"verdict": "WARN", "checks": [{"rule": "field tambahan", "level": "WARN", "detail": "x"}]},
        ),
        ev(2.0, "pipeline", "PIPELINE", "END"),
    ]


def jpeg(width=200, height=120) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), (240, 240, 240)).save(out, format="JPEG")
    return out.getvalue()


def data(events, **overrides):
    return {
        "request_id": RID,
        "generated_at": T0 + 60,
        "target": "local",
        "summary": {},
        "events": events,
        "guardrails_rows": [GUARDRAILS_ROW],
        "ocr_results_rows": [],
        "document": jpeg(),
        "db_note": None,
        **overrides,
    }


def test_completed_request_renders_every_section():
    pdf = report.render(data(completed_events()))
    assert pdf.startswith(b"%PDF")
    assert (
        report.outcome(completed_events(), {s: report.stage_view(completed_events(), s, None) for s in report.STAGES})[
            0
        ]
        == "ok"
    )


@pytest.mark.parametrize(
    ("events", "tone", "title"),
    [
        (
            [
                client(),
                http(
                    400,
                    errors="DOWNSTREAM_VALIDATION_ERROR",
                    message="Document rejected by guardrails",
                    rejected_by="guardrails",
                    pipeline_last_stage="guardrails",
                ),
                ev(1.6, "stage", "GUARDRAILS", "REJECTED", error_message="Document rejected by guardrails"),
            ],
            "bad",
            "DITOLAK oleh model guardrails",
        ),
        (
            [
                client(),
                http(202),
                ev(1.6, "stage", "GUARDRAILS", "DONE"),
                ev(20, "stage", "OCR", "DONE", source="db", result=OCR),
                ev(
                    21,
                    "stage",
                    "STRUCTURING",
                    "REJECTED",
                    source="db",
                    result={**STRUCTURING, "reject_reason": "nama 1 kata"},
                    error_message="nama 1 kata",
                ),
            ],
            "bad",
            "DITOLAK oleh aturan structuring",
        ),
        (
            [client(), http(202), ev(30, "callback", "SCORING", "DONE", accepted=True, final=True, http_status=200)],
            "ok",
            "SELESAI lewat callback",
        ),
        ([client(), http(202)], "warn", "DIPROSES (202)"),
        (
            [
                client(),
                http(422, errors="OCR_FAILED", message="engine error"),
                ev(2, "stage", "OCR", "FAILED", source="db", error_message="engine error"),
            ],
            "bad",
            "GAGAL di tahap Extraction (OCR)",
        ),
        (
            [client(["extraction", "scoring"]), http(422, errors="INVALID_PIPELINE_SEQUENCE", message="bad sequence")],
            "bad",
            "TIDAK DIPROSES (HTTP 422)",
        ),
        ([client()], "warn", "Belum ada jawaban orchestrator"),
    ],
)
def test_outcome_and_render_for_every_ending(events, tone, title):
    sequence = events[0].get("sequence")
    views = {stage: report.stage_view(events, stage, sequence) for stage in report.STAGES}
    assert report.outcome(events, views)[:2] == (tone, title)
    assert report.render(data(events, guardrails_rows=None, ocr_results_rows=None, db_note="db mati")).startswith(
        b"%PDF"
    )


def test_guardrails_only_uses_the_answer_data_as_report():
    events = [client(["guardrails"]), http(200, data=GUARDRAILS_ROW["report"]), ev(2, "pipeline", "PIPELINE", "END")]
    views = {stage: report.stage_view(events, stage, ["guardrails"]) for stage in report.STAGES}
    assert views["OCR"]["status"] == "SKIPPED"
    assert report.render(data(events, guardrails_rows=[])).startswith(b"%PDF")


def test_document_image_draws_ocr_boxes_only_when_they_fit():
    _, note = report._document_image(jpeg(), OCR["blocks"])
    assert "Kotak merah" in note
    _, note = report._document_image(jpeg(100, 50), OCR["blocks"])
    assert "di luar ukuran foto" in note
    assert report._document_image(b"%PDF-1.7 ...", [])[0] is None
    assert report._document_image(b"not an image", [])[0] is None
    assert report._document_image(None, [])[0] is None


def test_text_outside_latin1_does_not_break_the_pdf():
    assert report._t("A → B 中") == "A -> B ?"
