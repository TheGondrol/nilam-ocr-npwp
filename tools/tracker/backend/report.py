"""
Laporan PDF satu request (GET /api/requests/{id}/report.pdf), untuk tim ML Engineer dan QC: apa yang dikirim,
keputusan akhirnya dengan bahasa biasa, lalu apa yang dilihat dan diputuskan tiap tahap (probabilitas guardrails
per halaman, baris OCR dengan confidence-nya, field structuring dan flag aturannya, input dan output trust model
scoring), jawaban yang sampai ke Orkestrasi pusat, dan timeline lengkapnya.

`render(data)` murni: semua yang dibutuhkan dikumpulkan app.py (stream event Redis, ringkasan, baris
nilam_guardrails_results dan nilam_ocr_results, dokumen yang diunggah), jadi bisa diuji tanpa Redis dan DB.
Font bawaan PDF (Helvetica / Courier) hanya latin-1: teks di luar itu diganti `?` oleh `_t`.
"""

import io
import json
from datetime import datetime
from typing import Any

from fpdf import FPDF, FontFace, XPos, YPos

STAGES = ["GUARDRAILS", "OCR", "STRUCTURING", "SCORING"]
SERVICE_OF_STAGE = {"GUARDRAILS": "guardrails", "OCR": "extraction", "STRUCTURING": "structuring", "SCORING": "scoring"}
STAGE_LABELS = {
    "GUARDRAILS": "Guardrails (model kelayakan dokumen)",
    "OCR": "Extraction (OCR)",
    "STRUCTURING": "Structuring (aturan NPWP)",
    "SCORING": "Scoring (trust model)",
}
# Baris OCR di bawah ini ditandai di tabel: kandidat salah baca yang perlu dilihat QC.
LOW_CONFIDENCE = 0.80
MAX_TIMELINE_ROWS = 400
MAX_JSON_CHARS = 6000

# Penjelasan tiap fitur payload scoring (ConfidenceService.payload_from_chain).
PAYLOAD_NOTES = {
    "npwp": "nomor NPWP yang dinilai (digit saja)",
    "npwp_score": "confidence OCR baris nomor NPWP",
    "npwp_candidate_count": "jumlah kandidat nomor yang ditemukan structuring",
    "name_base": "nama sebelum normalisasi",
    "name_score": "confidence OCR baris nama",
    "avg_doc_score": "rata-rata confidence semua baris OCR",
    "min_doc_score": "confidence baris OCR terendah",
    "flag": "aturan structuring menandai dokumen (lihat flag_reason)",
    "guardrail_probability": "probabilitas lolos guardrails (kosong kalau tidak lolos / tidak dijalankan)",
}

INK = (24, 24, 27)
MUTED = (113, 113, 122)
LINE = (212, 212, 216)
HEAD_FILL = (244, 244, 245)
TONES = {
    "ok": ((220, 252, 231), (21, 128, 61)),
    "warn": ((254, 243, 199), (161, 98, 7)),
    "bad": ((254, 226, 226), (185, 28, 28)),
    "info": ((219, 234, 254), (29, 78, 216)),
    "idle": ((244, 244, 245), (82, 82, 91)),
}
STATUS_TONE = {"DONE": "ok", "REJECTED": "bad", "FAILED": "bad", "PROCESSING": "warn", "SKIPPED": "idle"}

_REPLACE = {
    "→": "->",
    "←": "<-",
    "…": "...",
    "—": "-",
    "–": "-",
    "·": "|",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "≥": ">=",
    "≤": "<=",
    "×": "x",
}


def _t(value: Any) -> str:
    """Teks yang aman untuk font bawaan PDF (latin-1)."""
    text = "" if value is None else str(value)
    for old, new in _REPLACE.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def _ms(ms: Any) -> str:
    if ms is None:
        return "-"
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.2f} dtk"


def _num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "ya" if value else "tidak"
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".") if value != int(value) else f"{value:.1f}"
    return str(value)


def _when(ts: float | None) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts).astimezone().strftime("%d %b %Y %H:%M:%S (UTC%z)")


def _iso_when(value: Any) -> str:
    try:
        return datetime.fromisoformat(str(value)).astimezone().strftime("%d %b %H:%M:%S")
    except ValueError:
        return _t(value)


# --- membaca event -------------------------------------------------------------------


def _first(events: list[dict[str, Any]], type_: str) -> dict[str, Any] | None:
    return next((e for e in events if e.get("type") == type_), None)


def _last(events: list[dict[str, Any]], type_: str) -> dict[str, Any] | None:
    return next((e for e in reversed(events) if e.get("type") == type_), None)


def stage_view(events: list[dict[str, Any]], stage: str, sequence: list[str] | None) -> dict[str, Any]:
    """Keadaan satu tahap dari event-eventnya, sama dengan kartu tahap di UI (PipelineView.stageView)."""
    if sequence and SERVICE_OF_STAGE[stage] not in sequence:
        return {"status": "SKIPPED"}
    mine = [e for e in events if e.get("type") == "stage" and e.get("stage") == stage]
    if not mine:
        return {"status": "-"}
    last = mine[-1]
    status = "REJECTED" if any(e.get("status") == "REJECTED" for e in mine) else last.get("status")
    started = next((e["ts"] for e in mine if e.get("status") == "PROCESSING"), None)
    finished = last["ts"] if last.get("status") in ("DONE", "FAILED", "REJECTED") else None
    elapsed = last.get("elapsed_ms")
    if elapsed is None and started and finished:
        elapsed = (finished - started) * 1000
    with_result = next((e for e in reversed(mine) if e.get("result")), None)
    error = next((e.get("error_message") for e in reversed(mine) if e.get("error_message")), None)
    return {
        "status": status,
        "elapsed_ms": elapsed,
        "result": with_result.get("result") if with_result else None,
        "error": error,
        "attempt": last.get("attempt"),
    }


def outcome(events: list[dict[str, Any]], views: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
    """(nada, judul, penjelasan) keputusan akhir request, dalam bahasa yang bisa dibaca QC."""
    http = _last(events, "http")
    final_cb = next(
        (e for e in reversed(events) if e.get("type") == "callback" and e.get("final") and e.get("accepted")), None
    )
    rejected = next(((s, v) for s, v in views.items() if v.get("status") == "REJECTED"), None)
    failed = next(((s, v) for s, v in views.items() if v.get("status") == "FAILED"), None)
    if http is None:
        if failed:
            return "bad", "Gagal sebelum orchestrator menjawab", _t(failed[1].get("error"))
        return "warn", "Belum ada jawaban orchestrator", "Request masih berjalan atau orchestrator tidak terjangkau."
    status, message = http.get("http_status"), http.get("message") or ""
    if rejected:
        stage, view = rejected
        who = "model guardrails" if stage == "GUARDRAILS" else "aturan structuring"
        return "bad", f"DITOLAK oleh {who}", f"{view.get('error') or message}. Tidak ada data yang dikembalikan."
    if failed and status != 200:
        stage, view = failed
        return "bad", f"GAGAL di tahap {STAGE_LABELS[stage]}", _t(view.get("error") or message)
    if status == 200:
        return "ok", "SELESAI - data terekstrak", f"Orchestrator menjawab 200 setelah {_ms(http.get('elapsed_ms'))}."
    if status == 202:
        if final_cb and final_cb.get("status") == "DONE":
            return (
                "ok",
                "SELESAI lewat callback",
                (
                    f"Pipeline lebih lama dari batas tunggu {_num(http.get('wait_seconds'))} dtk, orchestrator "
                    "menjawab "
                    "202 dan hasilnya dikirim lewat callback."
                ),
            )
        if final_cb:
            return "bad", "GAGAL (dilaporkan lewat callback)", _t(final_cb.get("error_message"))
        return "warn", "DIPROSES (202)", "Batas tunggu habis; hasil belum diterima saat laporan dibuat."
    if status == 400 and http.get("errors") == "DOWNSTREAM_VALIDATION_ERROR":
        return "bad", f"DITOLAK oleh {http.get('rejected_by') or 'pipeline'}", message
    return "bad", f"TIDAK DIPROSES (HTTP {status})", f"{http.get('errors') or ''} {message}".strip()


# --- PDF -------------------------------------------------------------------------------


class ReportPDF(FPDF):
    def __init__(self, request_id: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.request_id = request_id
        self.set_margins(14, 16, 14)
        self.set_auto_page_break(True, margin=16)
        self.set_text_color(*INK)
        self.set_draw_color(*LINE)

    def header(self):
        if self.page_no() == 1:
            return
        self.font(8)
        self.set_text_color(*MUTED)
        self.cell(0, 5, _t(f"Laporan request {self.request_id}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)
        self.set_text_color(*INK)

    def footer(self):
        self.set_y(-11)
        self.font(7.5)
        self.set_text_color(*MUTED)
        self.cell(0, 5, "nilam-ocr-npwp tracker | berisi data dokumen asli: hanya untuk tim internal", align="L")
        self.cell(0, 5, f"hal. {self.page_no()}/{{nb}}", align="R")
        self.set_text_color(*INK)

    # fpdf2 menerima float dan tuple padding, anotasinya lebih sempit (int): opsi lewat dict supaya type checker diam.

    def font(self, size: float, style: str = "", family: str = "Helvetica"):
        options: dict[str, Any] = {"size": size}
        self.set_font(family, style, **options)

    def block(self, line_height: float, text: str, **options: Any):
        self.multi_cell(0, line_height, _t(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT, **options)

    def fpdf_table(self, **options: Any):
        return self.table(**options)

    # potongan tata letak

    def h1(self, text: str):
        self.font(17, "B")
        self.cell(0, 9, _t(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def h2(self, text: str, *, new_page: bool = False):
        if new_page or self.get_y() > self.h - 50:
            self.add_page()
        else:
            self.ln(4)
        self.font(12.5, "B")
        self.cell(0, 7, _t(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*INK)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_draw_color(*LINE)
        self.ln(2)

    def h3(self, text: str):
        if self.get_y() > self.h - 40:
            self.add_page()
        self.ln(1.5)
        self.font(10, "B")
        self.cell(0, 6, _t(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def para(self, text: str, *, size: float = 9, muted: bool = False):
        self.font(size)
        if muted:
            self.set_text_color(*MUTED)
        self.block(size * 0.48, text, align="L")
        self.set_text_color(*INK)
        self.ln(1)

    def callout(self, tone: str, title: str, body: str):
        fill, ink = TONES[tone]
        self.set_fill_color(*fill)
        self.set_draw_color(*ink)
        self.set_text_color(*ink)
        self.font(12, "B")
        self.block(7, title, border="LTR", fill=True, padding=(2, 3, 0, 3))
        self.set_text_color(*INK)
        self.font(9)
        self.block(4.6, body or " ", border="LBR", fill=True, padding=(1, 3, 2.5, 3))
        self.set_draw_color(*LINE)
        self.set_fill_color(255, 255, 255)  # tabel memakai warna isian aktif
        self.ln(2)

    def grid(
        self,
        head: list[str],
        rows: list[list[Any]],
        widths: tuple[float, ...],
        *,
        size: float = 8.5,
        mono: tuple[int, ...] = (),
        tones: list[str | None] | None = None,
    ):
        """Tabel dengan baris judul; `mono`: indeks kolom ber-font Courier, `tones`: warna latar per baris."""
        self.font(size)
        heading = FontFace(emphasis="BOLD", fill_color=HEAD_FILL)
        with self.fpdf_table(
            col_widths=widths,
            width=self.epw,
            text_align="LEFT",
            line_height=size * 0.5,
            headings_style=heading,
            padding=(1, 1.5),
            borders_layout="HORIZONTAL_LINES",
        ) as table:
            row = table.row()
            for name in head:
                row.cell(_t(name))
            for index, values in enumerate(rows):
                tone = tones[index] if tones else None
                row = table.row()
                for column, value in enumerate(values):
                    style = FontFace(
                        family="Courier" if column in mono else None,
                        fill_color=TONES[tone][0] if tone else None,
                        color=TONES[tone][1] if tone and column == len(values) - 1 else None,
                    )
                    row.cell(_t(value if value not in (None, "") else "-"), style=style)
        self.ln(1.5)

    def kv(self, pairs: list[tuple[str, Any]], *, key_width: float = 0.3):
        rows = [[k, v] for k, v in pairs]
        self.font(8.5)
        with self.fpdf_table(
            col_widths=(key_width, 1 - key_width),
            width=self.epw,
            text_align="LEFT",
            line_height=4.3,
            first_row_as_headings=False,
            padding=(0.8, 1.5),
            borders_layout="HORIZONTAL_LINES",
        ) as table:
            for key, value in rows:
                row = table.row()
                row.cell(_t(key), style=FontFace(color=MUTED))
                row.cell(_t(value if value not in (None, "") else "-"))
        self.ln(1.5)

    def code(self, value: Any):
        text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False, default=str)
        if len(text) > MAX_JSON_CHARS:
            text = text[:MAX_JSON_CHARS] + f"\n... (dipotong, {len(text) - MAX_JSON_CHARS} karakter lagi)"
        self.font(7.2, family="Courier")
        self.set_fill_color(*HEAD_FILL)
        self.block(3.3, text, fill=True, padding=2)
        self.set_fill_color(255, 255, 255)
        self.ln(1.5)


def _document_image(content: bytes | None, blocks: list[dict[str, Any]]) -> tuple[Any, str]:
    """(gambar PIL untuk PDF atau None, catatan). Kotak bbox OCR bernomor digambar di atasnya kalau semuanya
    muat di dalam gambar (bbox ada di koordinat gambar yang dibaca model OCR)."""
    if not content:
        return None, "Dokumen tidak tersimpan di tracker (request dibuat sebelum fitur laporan, atau sudah dihapus)."
    if content[:4] == b"%PDF":
        return None, "Dokumen berupa PDF: tidak dirender di laporan ini."
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps

        image = ImageOps.exif_transpose(Image.open(io.BytesIO(content))).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - file rusak tetap menghasilkan laporan
        return None, f"Dokumen tidak bisa dibaca sebagai gambar ({type(exc).__name__})."
    boxes = [(i, b["bbox"]) for i, b in enumerate(blocks, 1) if isinstance(b.get("bbox"), dict) and not b.get("page")]
    note = "Foto dokumen seperti yang dikirim."
    fits = boxes and all(
        0 <= bb["x1"] < bb["x2"] <= image.width + 2 and 0 <= bb["y1"] < bb["y2"] <= image.height + 2 for _, bb in boxes
    )
    if fits:
        draw = ImageDraw.Draw(image)
        stroke = max(2, image.width // 400)
        font = ImageFont.load_default(size=max(12, image.width // 55))
        for number, bb in boxes:
            draw.rectangle((bb["x1"], bb["y1"], bb["x2"], bb["y2"]), outline=(220, 38, 38), width=stroke)
            draw.text((bb["x1"], bb["y1"]), str(number), fill=(220, 38, 38), font=font, anchor="ld")
        note = "Kotak merah = baris yang dibaca OCR; nomornya sama dengan kolom # di tabel Extraction."
    elif boxes:
        note = "Foto dokumen seperti yang dikirim (bbox OCR di luar ukuran foto: model membaca versi yang diubah)."
    image.thumbnail((1600, 1600))
    return image, note


def _fields_rows(fields: dict[str, Any]) -> tuple[list[list[Any]], list[str | None]]:
    rows, tones = [], []
    for name, field in fields.items():
        field = field or {}
        value, conf, threshold = field.get("value"), field.get("confidence"), field.get("threshold", "n/a")
        if value is None:
            decision, tone = "tidak ditemukan", "bad"
        elif threshold in (None, "n/a"):
            decision, tone = "tanpa threshold: confidence = probabilitas trust model", None
        elif conf == 1:
            decision, tone = f"lolos (probabilitas >= {_num(threshold)})", "ok"
        else:
            decision, tone = f"di bawah threshold {_num(threshold)}", "warn"
        rows.append([name, value, _num(conf), "-" if threshold in (None, "n/a") else _num(threshold), decision])
        tones.append(tone)
    return rows, tones


def render(data: dict[str, Any]) -> bytes:
    """PDF laporan satu request. `data`: request_id, generated_at, target, summary, events (urut stream),
    guardrails_rows / ocr_results_rows (None = database tidak tersedia), document (bytes atau None), db_note."""
    request_id = data["request_id"]
    events = data.get("events") or []
    summary = data.get("summary") or {}
    client = _first(events, "client") or {}
    http = _last(events, "http")
    sequence = client.get("sequence") or summary.get("sequence")
    if http and http.get("errors") == "INVALID_PIPELINE_SEQUENCE":
        sequence = None
    views = {stage: stage_view(events, stage, sequence) for stage in STAGES}
    t0 = client.get("ts") or (events[0]["ts"] if events else None)
    end = _last(events, "pipeline")
    total_ms = (end["ts"] - t0) * 1000 if end and t0 else None
    submissions = sum(1 for e in events if e.get("type") == "client")

    ocr = views["OCR"].get("result") or {}
    structuring = views["STRUCTURING"].get("result") or {}
    scoring = views["SCORING"].get("result") or {}
    blocks = ocr.get("blocks") or []
    body = (http or {}).get("body") or {}
    guardrails_rows = data.get("guardrails_rows")
    guardrails_row = guardrails_rows[-1] if guardrails_rows else None
    guardrails_report = (guardrails_row or {}).get("report")
    if not guardrails_report and sequence == ["guardrails"] and isinstance(body.get("data"), dict):
        guardrails_report = body["data"]  # "guardrails saja": data jawaban adalah report-nya

    pdf = ReportPDF(request_id)
    pdf.set_title(f"Laporan request {request_id}")
    pdf.set_creator("nilam-ocr-npwp tracker")
    pdf.add_page()

    # --- halaman 1: ringkasan ---
    pdf.font(8.5)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 5, "LAPORAN REQUEST OCR NPWP", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*INK)
    pdf.h1(request_id)
    pdf.ln(1)
    tone, title, explanation = outcome(events, views)
    pdf.callout(tone, title, explanation)

    pdf.kv(
        [
            (
                "Dokumen",
                f"{client.get('filename') or summary.get('filename') or '-'}"
                f" ({(client.get('size') or 0) / 1024:.0f} KB, dikirim sebagai {client.get('source') or 'upload'})",
            ),
            (
                "Dikirim",
                _when(t0) + (f"  |  dikirim {submissions}x dengan request_id yang sama" if submissions > 1 else ""),
            ),
            ("pipeline_name_sequence", ", ".join(sequence) if sequence else "pipeline penuh"),
            (
                "Jawaban orchestrator",
                f"HTTP {http.get('http_status')} setelah {_ms(http.get('elapsed_ms'))}"
                f" (batas tunggu {_num(http.get('wait_seconds'))} dtk)"
                if http
                else "belum ada",
            ),
            ("Total sampai selesai", _ms(total_ms) if total_ms is not None else "belum selesai saat laporan dibuat"),
            (
                "Target / simulasi",
                f"{data.get('target')}  |  callback {client.get('callback_mode') or 'ok'}"
                + (f"  |  OCR ditunda {client.get('slow')} dtk" if client.get("slow") else ""),
            ),
            ("Laporan dibuat", _when(data.get("generated_at"))),
        ]
    )

    pdf.h3("Tahap per tahap")
    stage_rows, stage_tones = [], []
    for stage in STAGES:
        view = views[stage]
        note = view.get("error") or ""
        if stage == "OCR" and blocks:
            note = note or f"{len(blocks)} baris terbaca"
        if stage == "STRUCTURING" and structuring:
            note = note or (f"flag: {structuring.get('flag_reason')}" if structuring.get("flag") else "tanpa flag")
        if stage == "SCORING" and scoring:
            note = note or (
                f"npwp {_num(scoring.get('npwp_confidence'))} | nama {_num(scoring.get('name_confidence'))}"
            )
        if stage == "GUARDRAILS" and guardrails_row:
            note = note or f"{guardrails_row.get('verdict')} (confidence {_num(guardrails_row.get('confidence'))})"
        stage_rows.append([STAGE_LABELS[stage], view["status"], _ms(view.get("elapsed_ms")), note])
        stage_tones.append(STATUS_TONE.get(view["status"]))
    # Warna hanya di kolom status: kolom terakhir tabel dipakai `table` sebagai kolom berwarna, jadi tukar urutannya.
    pdf.grid(
        ["Tahap", "Durasi", "Catatan", "Status"],
        [[r[0], r[2], r[3], r[1]] for r in stage_rows],
        (0.3, 0.12, 0.43, 0.15),
        tones=stage_tones,
    )

    # Field akhir dengan threshold-nya dari hasil scoring; tanpa itu (mis. scoring belum terbaca) data jawaban 200.
    final_fields = scoring.get("fields")
    if not final_fields and http and http.get("http_status") == 200 and (not sequence or sequence[-1] == "scoring"):
        final_fields = body.get("data")
    if isinstance(final_fields, dict) and final_fields:
        pdf.h3("Hasil akhir yang dikirim ke Orkestrasi pusat")
        rows, tones = _fields_rows(final_fields)
        pdf.grid(
            ["Field", "Nilai", "Confidence", "Threshold", "Keputusan"],
            rows,
            (0.15, 0.32, 0.12, 0.11, 0.30),
            mono=(1,),
            tones=tones,
        )
        if scoring:
            pdf.para(
                f"Probabilitas trust model: nomor_npwp {_num(scoring.get('npwp_confidence'))}, nama "
                f"{_num(scoring.get('name_confidence'))}. Dengan threshold dari pusat, confidence menjadi 1 (lolos) "
                "atau 0; tanpa threshold, confidence adalah probabilitas itu sendiri.",
                size=8,
                muted=True,
            )

    image, note = _document_image(data.get("document"), blocks)
    pdf.h3("Dokumen")
    if image is not None:
        max_w, max_h = pdf.epw * 0.75, max(60.0, min(150.0, pdf.h - pdf.get_y() - 30))
        scale = min(max_w / image.width, max_h / image.height)
        if pdf.get_y() + image.height * scale > pdf.h - 20:
            pdf.add_page()
        pdf.image(image, x=pdf.l_margin, w=image.width * scale, h=image.height * scale)
        pdf.ln(1)
    pdf.para(note, size=8, muted=True)

    # --- tiap tahap ---
    pdf.h2("1. Guardrails: apakah dokumen layak diproses", new_page=True)
    pdf.para(
        "Model guardrails menilai tiap halaman: proba_approve = peluang halaman itu dokumen NPWP yang layak. "
        "Halaman ditolak kalau proba_approve di bawah threshold; dokumen lolos sesuai kebijakan service "
        "(default: semua halaman harus lolos).",
        size=8.5,
        muted=True,
    )
    if views["GUARDRAILS"]["status"] == "SKIPPED":
        pdf.para("Dilewati: guardrails tidak ada di pipeline_name_sequence.")
    elif guardrails_report:
        document = guardrails_report.get("document") or {}
        passed = (guardrails_row or {}).get("passed", document.get("verdict") == "accepted")
        pdf.kv(
            [
                (
                    "Keputusan",
                    f"{'LOLOS' if passed else 'DITOLAK'}" f" (verdict model: {document.get('verdict')})",
                ),
                (
                    "Confidence dokumen",
                    f"{_num(document.get('confidence'))}  (lolos: proba_approve terendah; ditolak: "
                    "proba_reject tertinggi)",
                ),
                (
                    "Threshold",
                    f"{_num(document.get('threshold'))}"
                    + (f"  |  sumber: {guardrails_row.get('threshold_source')}" if guardrails_row else ""),
                ),
                (
                    "Halaman",
                    f"{document.get('n_pages')} ({document.get('n_approve')} lolos, "
                    f"{document.get('n_reject')} ditolak)",
                ),
                ("Alasan", (guardrails_row or {}).get("reason") or guardrails_report.get("reason")),
            ]
        )
        pages = guardrails_report.get("pages") or []
        if pages:
            pdf.grid(
                ["Halaman", "proba_approve", "proba_reject", "Verdict"],
                [
                    [p.get("page_index"), _num(p.get("proba_approve")), _num(p.get("proba_reject")), p.get("verdict")]
                    for p in pages
                ],
                (0.2, 0.25, 0.25, 0.3),
                mono=(1, 2),
                tones=["ok" if p.get("verdict") == "accepted" else "bad" for p in pages],
            )
    else:
        payload = scoring.get("payload") or {}
        pdf.para(
            "Report guardrails tidak terbaca: "
            + (data.get("db_note") or "tidak ada baris nilam_guardrails_results.")
            + (
                f" Probabilitas lolos yang dipakai scoring: {_num(payload.get('guardrail_probability'))}."
                if payload
                else ""
            )
            + (f" Nilai guardrails di jawaban: {_num(body.get('guardrails'))}." if "guardrails" in body else "")
        )

    pdf.h2("2. Extraction: teks yang dibaca OCR")
    if views["OCR"]["status"] == "SKIPPED":
        pdf.para("Dilewati: extraction tidak ada di pipeline_name_sequence.")
    elif not ocr:
        pdf.para(views["OCR"].get("error") or "Belum ada hasil OCR.")
    else:
        scores = [float(b["confidence"]) for b in blocks if b.get("confidence") is not None]
        low = [s for s in scores if s < LOW_CONFIDENCE]
        pdf.kv(
            [
                ("Engine / model", f"{ocr.get('engine')} / {ocr.get('model') or '-'}"),
                ("Waktu inferensi", _ms(ocr.get("elapsed_ms"))),
                ("Baris terbaca", len(blocks)),
                (
                    "Confidence rata-rata / terendah",
                    f"{sum(scores) / len(scores):.4f} / {min(scores):.4f}" if scores else "-",
                ),
                (f"Baris < {LOW_CONFIDENCE:.2f}", f"{len(low)} (ditandai kuning di tabel)"),
            ]
        )
        pdf.grid(
            ["#", "Hal.", "Teks", "Confidence"],
            [[i, b.get("page", 0), b.get("text"), _num(b.get("confidence"))] for i, b in enumerate(blocks, 1)],
            (0.06, 0.07, 0.69, 0.18),
            mono=(2,),
            tones=["warn" if (b.get("confidence") or 0) < LOW_CONFIDENCE else None for b in blocks],
        )

    pdf.h2("3. Structuring: field yang diambil dari teks")
    if views["STRUCTURING"]["status"] == "SKIPPED":
        pdf.para("Dilewati: structuring tidak ada di pipeline_name_sequence.")
    elif not structuring:
        pdf.para(views["STRUCTURING"].get("error") or "Belum ada hasil structuring.")
    else:
        fields = structuring.get("fields") or {}
        pdf.grid(
            ["Field", "Nilai", "Conf. OCR", "Sumber", "Sinyal"],
            [
                [
                    name,
                    (f or {}).get("value"),
                    _num((f or {}).get("confidence")),
                    (f or {}).get("source"),
                    ", ".join(f"{k}={v}" for k, v in ((f or {}).get("signals") or {}).items()),
                ]
                for name, f in fields.items()
            ],
            (0.15, 0.27, 0.11, 0.17, 0.30),
            mono=(1,),
            tones=[None if (f or {}).get("value") is not None else "bad" for f in fields.values()],
        )
        pdf.kv(
            [
                (
                    "Flag aturan",
                    f"{'YA' if structuring.get('flag') else 'tidak'}"
                    + (f" - {structuring.get('flag_reason')}" if structuring.get("flag_reason") else ""),
                ),
                ("Alasan penolakan", structuring.get("reject_reason") or "tidak ditolak"),
            ]
        )
        pdf.para(
            "Flag ikut menjadi input trust model. Penolakan (reject_reason) menghentikan pipeline di tahap ini "
            "dan pusat menerima 400 DOWNSTREAM_VALIDATION_ERROR.",
            size=8,
            muted=True,
        )

    pdf.h2("4. Scoring: seberapa yakin hasilnya benar")
    if views["SCORING"]["status"] == "SKIPPED":
        pdf.para("Dilewati: scoring tidak ada di pipeline_name_sequence.")
    elif not scoring:
        pdf.para(views["SCORING"].get("error") or "Belum ada hasil scoring.")
    else:
        pdf.kv(
            [
                ("npwp_confidence", f"{_num(scoring.get('npwp_confidence'))}  (peluang nomor NPWP benar)"),
                ("name_confidence", f"{_num(scoring.get('name_confidence'))}  (peluang nama benar)"),
            ]
        )
        payload = scoring.get("payload") or {}
        if payload:
            pdf.h3("Input trust model (payload)")
            pdf.grid(
                ["Fitur", "Nilai", "Arti"],
                [[k, v if isinstance(v, str) else _num(v), PAYLOAD_NOTES.get(k, "")] for k, v in payload.items()],
                (0.24, 0.26, 0.50),
                mono=(0, 1),
            )

    # --- jawaban ke pusat ---
    pdf.h2("5. Yang diterima Orkestrasi pusat")
    if http:
        pdf.kv(
            [
                ("HTTP", f"{http.get('http_status')}  {http.get('errors') or ''}"),
                ("message", http.get("message")),
                ("pipeline_last_stage", http.get("pipeline_last_stage") or "null (tidak ada error / penolakan)"),
                (
                    "Lama menjawab",
                    f"{_ms(http.get('elapsed_ms'))} dari batas tunggu {_num(http.get('wait_seconds'))} dtk",
                ),
            ]
        )
        pdf.h3("Body jawaban POST /v1/extract-ocr")
        pdf.code(body)
    callbacks = [e for e in events if e.get("type") == "callback"]
    if callbacks:
        pdf.h3("Callback yang tiba di tracker")
        pdf.grid(
            ["t+", "Tahap / status", "Format", "Jawaban", "Attempt", "Kontrak"],
            [
                [
                    _ms((e["ts"] - t0) * 1000) if t0 else "-",
                    f"{e.get('stage')} {e.get('status')}",
                    e.get("format"),
                    f"{e.get('http_status')}{' DUPLIKAT' if e.get('duplicate') else ''}",
                    e.get("attempt"),
                    ((e.get("contract") or {}).get("verdict") or "-")
                    + "".join(
                        f"; {c['rule']}"
                        for c in (e.get("contract") or {}).get("checks", [])
                        if c.get("level") != "PASS"
                    ),
                ]
                for e in callbacks
            ],
            (0.11, 0.2, 0.09, 0.14, 0.09, 0.37),
            tones=[
                {"PASS": "ok", "WARN": "warn", "FAIL": "bad"}.get((e.get("contract") or {}).get("verdict") or "")
                for e in callbacks
            ],
        )
    rows = data.get("ocr_results_rows")
    pdf.h3("nilam_ocr_results (log setiap jawaban ke pusat)")
    if rows:
        pdf.grid(
            ["Dicatat", "HTTP", "status_desc", "guardrails", "last_stage", "message"],
            [
                [
                    _iso_when(r.get("created_at")),
                    r.get("status_code"),
                    r.get("status_desc"),
                    _num(r.get("guardrails")),
                    r.get("pipeline_last_stage"),
                    r.get("message"),
                ]
                for r in rows
            ],
            (0.22, 0.08, 0.14, 0.11, 0.13, 0.32),
        )
    else:
        pdf.para(
            (data.get("db_note") or "Database tidak tersedia.") if rows is None else "Tidak ada baris.", muted=True
        )

    # --- timeline ---
    pdf.h2("6. Timeline semua event", new_page=True)
    pdf.para(
        "t+ dihitung dari saat dokumen dikirim tracker. Event berasal dari jawaban orchestrator (http), "
        "tabel job dan outbox di database (stage / outbox), callback ke tracker, dan gangguan (chaos).",
        size=8,
        muted=True,
    )
    timeline = []
    for e in events[:MAX_TIMELINE_ROWS]:
        detail = e.get("error_message") or ""
        if e.get("type") == "http":
            detail = f"HTTP {e.get('http_status')} {e.get('errors') or ''}"
        elif e.get("type") == "outbox" and isinstance(e.get("message"), dict):
            m = e["message"]
            detail = f"{m.get('owner')} -> {m.get('target')}: {m.get('message')} (attempt {m.get('attempts')})"
            if m.get("last_error"):
                detail += f" | {m['last_error']}"
        elif e.get("type") == "callback":
            detail = f"dijawab {e.get('http_status')}, attempt {e.get('attempt')}"
        elif e.get("type") == "stage" and e.get("source"):
            detail = (f"[{e['source']}] " + detail).strip()
        timeline.append(
            [_ms((e["ts"] - t0) * 1000) if t0 else "-", e.get("type"), e.get("stage"), e.get("status"), detail]
        )
    pdf.grid(["t+", "Jenis", "Tahap", "Status", "Detail"], timeline, (0.11, 0.1, 0.15, 0.14, 0.5), size=7.5)
    if len(events) > MAX_TIMELINE_ROWS:
        pdf.para(f"... {len(events) - MAX_TIMELINE_ROWS} event lagi tidak ditampilkan.", muted=True)

    pdf.h2("Cara membaca laporan ini")
    pdf.para(
        "HTTP 200: pipeline selesai di dalam batas tunggu, data ada di jawaban. 202: batas tunggu habis, hasil "
        "dikirim lewat callback. 400 DOWNSTREAM_VALIDATION_ERROR: dokumen ditolak guardrails atau aturan structuring. "
        "422 *_FAILED: sebuah tahap gagal; 422 lain: request tidak valid.\n"
        "Confidence OCR: keyakinan model OCR membaca satu baris teks (0-1). Probabilitas trust model (scoring): "
        "peluang field akhir benar, dihitung dari sinyal OCR, structuring, dan guardrails. Threshold: batas dari "
        "Orkestrasi pusat; tanpa threshold confidence yang dikirim adalah probabilitasnya.\n"
        "Untuk QC: bandingkan Nilai di 'Hasil akhir' dengan foto. Kalau salah, lihat baris OCR asalnya (bagian 2) "
        "dan apakah structuring mengambil baris yang benar (bagian 3).",
        size=8.5,
    )

    return bytes(pdf.output())
