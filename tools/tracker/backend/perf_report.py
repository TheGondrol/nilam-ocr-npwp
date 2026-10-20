"""Laporan PDF performance test (GET /api/perftest/{run}/report.pdf, juga ditulis otomatis saat run selesai ke
load-tester/out/perf-<run>.pdf), untuk tim: kesimpulan lulus/gagal per SLO, temuan, grafik laju/latensi/status/
resource per jendela waktu, hasil per fase dan per tahap pipeline, perbandingan dengan run sebelumnya, lalu
konfigurasi dan metodologinya supaya hasilnya bisa diulang.

`render(meta, result)` murni: `result` dari perf_stats.compute (+ `baseline`), jadi bisa diuji tanpa cluster.
Gaya dan potongan tata letaknya sama dengan laporan per request (report.ReportPDF).
"""

import math
from datetime import datetime
from typing import Any

from fpdf.enums import XPos, YPos
from report import INK, LINE, MUTED, TONES, ReportPDF, _ms, _t

BLUE = (37, 99, 235)
SKY = (56, 189, 248)
GREEN = (22, 163, 74)
AMBER = (217, 119, 6)
RED = (220, 38, 38)
GRAY = (113, 113, 122)
SERVICE_COLORS = {
    "orchestrator": (37, 99, 235),
    "guardrails": (147, 51, 234),
    "extraction": (220, 38, 38),
    "structuring": (217, 119, 6),
    "scoring": (22, 163, 74),
    "generator": (113, 113, 122),
}
BUCKET_COLORS = {
    "200": (16, 185, 129),
    "rejected": (245, 158, 11),
    "202": (56, 189, 248),
    "4xx": (248, 113, 113),
    "5xx": (220, 38, 38),
    "timeout": (100, 116, 139),
}
BUCKET_LABELS = {
    "200": "200 selesai",
    "rejected": "400 ditolak",
    "202": "202 menyusul",
    "4xx": "4xx",
    "5xx": "5xx",
    "timeout": "timeout",
}
STATUS_LABELS = {
    "finished": "selesai",
    "aborted": "dihentikan otomatis",
    "stopped": "dihentikan pengguna",
    "failed": "gagal",
    "interrupted": "terputus",
    "running": "berjalan (laporan sementara)",
    "draining": "menunggu antrean habis (laporan sementara)",
    "starting": "menyiapkan (laporan sementara)",
}
TEST_TYPES = [
    (
        "Smoke",
        "Beban sangat kecil dalam waktu singkat. Memastikan semua service hidup, endpoint menjawab, dan alat "
        "ukur bekerja, sebelum tes yang lebih berat.",
    ),
    (
        "Load (baseline)",
        "Beban normal yang diharapkan, ditahan beberapa menit. Hasilnya menjadi baseline yang "
        "dibandingkan antar rilis: apakah rilis baru lebih lambat atau lebih boros resource.",
    ),
    (
        "Stress",
        "Laju dinaikkan bertahap per step sampai SLO dilanggar. Menjawab berapa kapasitas berkelanjutan dan "
        "komponen mana yang jenuh lebih dulu. Berhenti otomatis begitu satu step jelas melanggar SLO.",
    ),
    (
        "Spike",
        "Lonjakan mendadak dari laju dasar, lalu kembali. Menjawab apakah sistem tetap melayani saat lonjakan "
        "dan berapa lama sampai pulih.",
    ),
    (
        "Soak",
        "Laju sedang ditahan lama (30-120 menit). Mencari degradasi yang hanya muncul seiring waktu: kebocoran "
        "memori, latensi yang memburuk, antrean yang pelan-pelan menumpuk.",
    ),
]
DEFINITIONS = [
    (
        "Pintu masuk",
        "Lama POST /v1/extract-ocr-test ditahan orchestrator. Orchestrator menunggu pipeline sampai "
        "PIPELINE_WAIT_SECONDS; lebih lama dari itu dijawab 202 dan hasilnya menyusul.",
    ),
    (
        "End-to-end",
        "Dari request dikirim sampai tahap terakhir sequence berstatus DONE di tabel nilam_testing_*. "
        "Diukur di database, karena endpoint -test tidak mengirim callback.",
    ),
    (
        "Error server",
        "Jawaban 5xx dan timeout (tidak ada jawaban dalam WAIT + 30 detik). Dokumen yang ditolak model "
        "atau aturan (400 DOWNSTREAM_VALIDATION_ERROR) dan 4xx lain tidak dihitung sebagai error beban.",
    ),
    (
        "Rasio tuntas",
        "Request yang tuntas dibanding semua yang masuk pipeline dan tidak ditolak (tuntas + gagal + "
        "belum selesai), setelah antrean diberi waktu habis.",
    ),
    (
        "Durasi tahap / jeda",
        "Durasi tahap: dibuat sampai selesainya job tahap itu. Jeda: dari tahap sebelumnya "
        "selesai sampai job tahap ini dibuat (serah-terima lewat outbox).",
    ),
    (
        "Resource",
        "kubectl top pods tiap ~15 detik: nilai tertinggi satu pod dibanding limit per pod. CPU mendekati "
        "limit berarti pod di-throttle; memori mendekati limit berisiko OOMKilled.",
    ),
    (
        "Utilisasi CPU (target 75%)",
        "CPU pod tersibuk tiap service dibanding limit CPU-nya, dirata-rata selama satu fase tahan. Menurut "
        "teori antrean, waktu tunggu tumbuh sebanding rho / (1 - rho): 3x pada utilisasi 75%, 9x pada 90%, "
        "19x pada 95%. Di atas 75% sedikit lonjakan beban sudah berubah jadi antrean panjang, jadi kapasitas "
        "berkelanjutan di stress test dihitung dengan semua service <= target ini. Catatan: HPA Kubernetes "
        "mengukur terhadap requests, bukan limit, jadi angkanya tidak sama dengan target HPA.",
    ),
]


def _rps(value: Any) -> str:
    return "-" if value is None else f"{value:.2f} rps"


def _pct(value: Any) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _when(ts: Any) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(float(ts)).astimezone().strftime("%d %b %Y %H:%M:%S (UTC%z)")


def _duration(seconds: Any) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    return f"{seconds // 60} mnt {seconds % 60} dtk" if seconds >= 60 else f"{seconds} dtk"


PERCENT_SLOS = ("error_rate", "min_completion", "max_cpu_utilization")


def _slo_value(key: str, value: Any) -> str:
    if value is None:
        return "-"
    if key in PERCENT_SLOS:
        return _pct(value)
    if key.endswith("_ms"):
        return _ms(value)
    return str(int(value))


def _slo_target(key: str, value: Any) -> str:
    if key == "error_rate":
        return f"< {_pct(value)}"
    if key == "min_completion":
        return f">= {_pct(value)}"
    if key == "max_cpu_utilization":
        return f"<= {_pct(value)}"
    if key.endswith("_ms"):
        return f"< {_ms(value)}"
    return f"<= {int(value)}"


class PerfPDF(ReportPDF):
    def __init__(self, title: str):
        super().__init__(title)
        self.title_text = title

    def header(self):
        if self.page_no() == 1:
            return
        self.font(8)
        self.set_text_color(*MUTED)
        self.cell(0, 5, _t(self.title_text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)
        self.set_text_color(*INK)

    def footer(self):
        self.set_y(-11)
        self.font(7.5)
        self.set_text_color(*MUTED)
        self.cell(0, 5, "nilam-ocr-npwp tracker | performance test GKE | internal", align="L")
        self.cell(0, 5, f"hal. {self.page_no()}/{{nb}}", align="R")
        self.set_text_color(*INK)

    def bullet(self, tone: str, text: str):
        fill, ink = TONES.get(tone, TONES["info"])
        y = self.get_y()
        self.set_fill_color(*ink)
        self.rect(self.l_margin + 0.5, y + 1.2, 2, 2, style="F")
        self.set_fill_color(255, 255, 255)
        self.set_x(self.l_margin + 4.5)
        self.font(8.8)
        self.multi_cell(self.epw - 4.5, 4.4, _t(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(0.6)

    # --- grafik -----------------------------------------------------------------------

    def _frame(self, x: float, y: float, w: float, h: float, y_max: float, y_fmt, title: str, x_max: float):
        self.font(8.5, "B")
        self.set_xy(x, y - 5)
        self.cell(w, 4.5, _t(title))
        self.set_draw_color(*LINE)
        self.set_line_width(0.2)
        self.font(6.5)
        self.set_text_color(*MUTED)
        for i in range(5):
            gy = y + h - h * i / 4
            self.line(x, gy, x + w, gy)
            self.set_xy(x - 13, gy - 1.6)
            self.cell(12, 3.2, _t(y_fmt(y_max * i / 4)), align="R")
        ticks = 6
        for i in range(ticks + 1):
            gx = x + w * i / ticks
            self.set_xy(gx - 6, y + h + 0.6)
            self.cell(12, 3, _t(f"{x_max * i / ticks / 60:.1f}m"), align="C")
        self.set_text_color(*INK)

    def line_chart(
        self,
        title: str,
        series: list[tuple[str, tuple[int, int, int], list[tuple[float, float | None]], bool]],
        *,
        x_max: float,
        y_fmt=lambda v: f"{v:g}",
        hlines: list[tuple[float, str, tuple[int, int, int]]] | None = None,
        height: float = 46,
    ):
        """`series`: (label, warna, [(x detik, y)], putus-putus). y None memutus garis."""
        if self.get_y() > self.h - height - 30:
            self.add_page()
        x, y, w, h = self.l_margin + 14, self.get_y() + 7, self.epw - 16, height
        values = [v for _, _, pts, _ in series for _, v in pts if v is not None]
        # Garis batas jauh di atas data (mis. SLO 60 dtk untuk data 0.7 dtk) tidak ikut menentukan skala: datanya
        # akan menempel di dasar. Garis itu hanya disebut di legenda.
        data_max = max(values, default=0)

        def above(label: str, value: float) -> str:
            text = label if y_fmt(value) in label else f"{label} {y_fmt(value)}"
            return f"{text} (di atas skala)"

        hlines = [
            (value, label if data_max <= 0 or value <= 3 * data_max else above(label, value), c)
            for value, label, c in hlines or []
        ]
        shown = [value for value, label, _ in hlines if "(di atas skala)" not in label]
        top = max([data_max, *shown], default=0)
        y_max = _nice_max(top * 1.05) if top > 0 else 1
        self._frame(x, y, w, h, y_max, y_fmt, title, x_max or 1)

        def px(t: float, v: float) -> tuple[float, float]:
            return x + w * min(max(t / (x_max or 1), 0), 1), y + h - h * min(v / y_max, 1)

        for value, label, color in hlines:
            if "(di atas skala)" in label:
                continue
            self.set_draw_color(*color)
            self.set_line_width(0.25)
            self.set_dash_pattern(dash=1.2, gap=1.2)
            _, hy = px(0, value)
            self.line(x, hy, x + w, hy)
            self.set_dash_pattern()
        for _label, color, points, dashed in series:
            self.set_draw_color(*color)
            self.set_line_width(0.45)
            if dashed:
                self.set_dash_pattern(dash=1.5, gap=1)
            previous = None
            for t, v in points:
                if v is None:
                    previous = None
                    continue
                point = px(t, v)
                if previous is not None:
                    self.line(*previous, *point)
                elif len(points) == 1:
                    self.circle(point[0] - 0.5, point[1] - 0.5, 1, style="F")
                previous = point
            self.set_dash_pattern()
        self.set_line_width(0.2)
        self.set_draw_color(*LINE)
        legend = [(label, color) for label, color, _, _ in series] + [
            (label, color) for _, label, color in hlines or []
        ]
        self._legend(x, y + h + 5, legend)
        self.set_y(y + h + 11)

    def _legend(self, x: float, y: float, items: list[tuple[str, tuple[int, int, int]]]):
        self.font(7)
        cx = x
        for label, color in items:
            width = self.get_string_width(_t(label)) + 8
            if cx + width > self.w - self.r_margin:
                cx, y = x, y + 4
            self.set_fill_color(*color)
            self.rect(cx, y + 0.9, 3, 1.6, style="F")
            self.set_xy(cx + 4, y)
            self.cell(width - 4, 3.4, _t(label))
            cx += width
        self.set_fill_color(255, 255, 255)

    def status_bars(self, title: str, windows: list[dict[str, Any]], x_max: float, height: float = 36):
        if self.get_y() > self.h - height - 30:
            self.add_page()
        x, y, w, h = self.l_margin + 14, self.get_y() + 7, self.epw - 16, height
        self._frame(x, y, w, h, 1.0, lambda v: f"{v * 100:.0f}%", title, x_max or 1)
        slot = w / max(1, len(windows))
        for index, win in enumerate(windows):
            sent = win["sent"]
            if not sent:
                continue
            bottom = y + h
            for key, color in BUCKET_COLORS.items():
                share = win["counts"].get(key, 0) / sent
                if share <= 0:
                    continue
                self.set_fill_color(*color)
                self.rect(x + index * slot + slot * 0.1, bottom - h * share, slot * 0.8, h * share, style="F")
                bottom -= h * share
        self.set_fill_color(255, 255, 255)
        self._legend(x, y + h + 5, [(BUCKET_LABELS[k], c) for k, c in BUCKET_COLORS.items()])
        self.set_y(y + h + 11)


def render(meta: dict[str, Any], result: dict[str, Any]) -> bytes:
    profile = meta.get("profile", "-")
    from perf_stats import PARAM_LABELS, PROFILES

    info = PROFILES.get(profile, {})
    title = f"Performance test {info.get('label', profile)} | run {meta.get('run_id')}"
    pdf = PerfPDF(title)
    pdf.set_title(_t(title))
    pdf.alias_nb_pages()
    pdf.add_page()
    s = result["summary"]
    status = meta.get("status", "-")

    # --- halaman 1: kesimpulan ---------------------------------------------------------
    pdf.font(9)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 5, "LAPORAN PERFORMANCE TEST | NILAM OCR NPWP | GKE DEV", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*INK)
    pdf.h1(f"{info.get('label', profile)}: {meta.get('run_id')}")
    pdf.para(f"Pertanyaan yang dijawab: {info.get('question', '-')}", size=9.5)
    if status not in ("finished", "aborted", "stopped", "failed", "interrupted"):
        pdf.callout("warn", "LAPORAN SEMENTARA", f"Run masih {STATUS_LABELS.get(status, status)}; angka belum final.")
    v = result["verdict"]
    pdf.callout(v["tone"], v["title"], v["body"])
    if meta.get("error"):
        pdf.callout("bad", "Run gagal", meta["error"])

    started = meta.get("load_started_at") or meta.get("created_at")
    finished = meta.get("load_finished_at")
    pdf.kv(
        [
            ("Waktu", f"{_when(started)} s/d {_when(finished) if finished else '-'}"),
            ("Durasi beban", _duration((finished - started) if finished and started else meta.get("elapsed_seconds"))),
            ("Status run", STATUS_LABELS.get(status, status)),
            (
                "Target",
                f"{meta.get('target')} (cluster {meta.get('context', '-').split('_')[-1]}, ns {meta.get('namespace')})",
            ),
            (
                "Sequence",
                ", ".join(meta.get("pipeline_name_sequence") or [])
                or "pipeline penuh (guardrails, extraction, structuring, scoring)",
            ),
            ("File uji", ", ".join(meta.get("images", []))),
            ("Catatan", meta.get("note") or "-"),
        ]
    )

    pdf.h2("Kriteria lulus (SLO)")
    rows, tones = [], []
    for row in result["slo"]:
        state = "LULUS" if row["ok"] else ("GAGAL" if row["ok"] is False else "TIDAK TERUKUR")
        rows.append(
            [row["label"], _slo_target(row["key"], row["target"]), _slo_value(row["key"], row["actual"]), state]
        )
        tones.append("ok" if row["ok"] else ("bad" if row["ok"] is False else "warn"))
        if row.get("note"):
            rows[-1][0] += f" ({row['note']})"
    pdf.grid(["Kriteria", "Target", "Aktual", "Status"], rows, (0.46, 0.18, 0.18, 0.18), tones=tones)
    if profile == "stress":
        pdf.para(
            "Pada stress test SLO dinilai per step (lihat tabel per fase); hasil keseluruhan di atas mencakup step "
            "yang sudah melewati batas, jadi wajar gagal. Yang dicari adalah laju step terakhir yang masih lulus.",
            size=8,
            muted=True,
        )

    pdf.h2("Ringkasan angka")
    c = s["counts"]
    pdf.grid(
        ["Metrik", "Nilai", "Metrik", "Nilai"],
        [
            ["Request dikirim", s["sent"], "Laju tercapai", _rps(s["achieved_rate"])],
            ["200 selesai", c["200"], "202 menyusul", c["202"]],
            ["400 ditolak", c["rejected"], "4xx lain", c["4xx"]],
            ["5xx", c["5xx"], "Timeout", c["timeout"]],
            ["Tuntas (DONE)", s["completed"], "Gagal di tahap", s["failed"]],
            ["Ditolak (total)", s["rejected"], "Belum tuntas", s["in_flight"]],
            [
                "p50 / p95 pintu masuk",
                f"{_ms(s['entry']['p50'])} / {_ms(s['entry']['p95'])}",
                "maks pintu masuk",
                _ms(s["entry"]["max"]),
            ],
            [
                "p50 / p95 end-to-end",
                f"{_ms(s['e2e']['p50'])} / {_ms(s['e2e']['p95'])}",
                "maks end-to-end",
                _ms(s["e2e"]["max"]),
            ],
            [
                "Dokumen tuntas / menit",
                "-" if s["completed_per_minute"] is None else f"{s['completed_per_minute']:.1f}",
                "Rasio tuntas",
                _pct(s["completion_ratio"]),
            ],
        ],
        (0.27, 0.23, 0.27, 0.23),
    )

    pdf.h2("Temuan")
    for tone, text in result["findings"]:
        pdf.bullet(tone, text)

    # --- grafik -------------------------------------------------------------------------
    windows = result["windows"]
    x_max = max([w["t"] + 30 for w in windows], default=60)
    pdf.h2("Grafik per jendela 30 detik", new_page=True)
    pdf.line_chart(
        "Laju: rencana vs dikirim vs tuntas (rps)",
        [
            ("rencana", GRAY, [(w["t"] + 15, w["target_rate"]) for w in windows], True),
            ("dikirim", BLUE, [(w["t"] + 15, w["rate"]) for w in windows], False),
            ("tuntas", GREEN, [(w["t"] + 15, w["done_rate"]) for w in windows], False),
        ],
        x_max=x_max,
        y_fmt=lambda v: f"{v:.1f}",
    )
    slo = result["slo_targets"]
    pdf.line_chart(
        "Latensi per jendela kirim (detik)",
        [
            ("p50 pintu masuk", SKY, [(w["t"] + 15, _sec(w["entry_p50"])) for w in windows], False),
            ("p95 pintu masuk", BLUE, [(w["t"] + 15, _sec(w["entry_p95"])) for w in windows], False),
            ("p95 end-to-end", GREEN, [(w["t"] + 15, _sec(w["e2e_p95"])) for w in windows], False),
        ],
        x_max=x_max,
        y_fmt=lambda v: f"{v:.0f}s" if v >= 10 or v == 0 else f"{v:.1f}s",
        hlines=[
            (slo["entry_p95_ms"] / 1000, "SLO pintu masuk", BLUE),
            (slo["e2e_p95_ms"] / 1000, "SLO end-to-end", GREEN),
        ],
    )
    pdf.status_bars("Campuran jawaban per jendela kirim", windows, x_max)

    resources = result["resources"]
    start = meta.get("load_started_at") or meta.get("created_at") or 0
    cpu_series, mem_series = [], []
    for service, r in resources.items():
        if not r["series"]:
            continue
        color = SERVICE_COLORS.get(service, GRAY)
        cpu_limit, mem_limit = r.get("cpu_limit_m"), r.get("mem_limit_mi")
        if cpu_limit:
            cpu_series.append(
                (
                    service,
                    color,
                    [(ts - start, cpu / cpu_limit * 100) for ts, cpu, _ in r["series"]],
                    service == "generator",
                )
            )
        if mem_limit:
            mem_series.append(
                (
                    service,
                    color,
                    [(ts - start, mem / mem_limit * 100) for ts, _, mem in r["series"]],
                    service == "generator",
                )
            )
    if cpu_series:
        pdf.line_chart(
            "CPU pod tertinggi per service (% limit)",
            cpu_series,
            x_max=x_max,
            y_fmt=lambda v: f"{v:.0f}%",
            hlines=[(slo.get("max_cpu_utilization", 0.75) * 100, "target utilisasi (queueing)", AMBER)],
        )
    if mem_series:
        pdf.line_chart(
            "Memori pod tertinggi per service (% limit)",
            mem_series,
            x_max=x_max,
            y_fmt=lambda v: f"{v:.0f}%",
            hlines=[(85, "batas 85%", AMBER)],
        )

    # --- rincian -----------------------------------------------------------------------
    pdf.h2("Hasil per fase")
    rows, tones = [], []
    for p in result["phases"]:
        planned = (
            f"{p['start_rate']:g}->{p['end_rate']:g}" if p["start_rate"] != p["end_rate"] else f"{p['end_rate']:g}"
        )
        rows.append(
            [
                p["label"],
                planned,
                "-" if p["achieved_rate"] is None else f"{p['achieved_rate']:.2f}",
                p["sent"],
                _pct(p["error_rate"]),
                _pct(p["share_202"]),
                _ms(p["entry_p95"]),
                _ms(p["e2e_p95"]),
                f"{p['completed_of_sent']}/{p['sent']}",
                "-"
                if p.get("cpu_util_max") is None
                else f"{p['cpu_util_max'] * 100:.0f}% {_short(p.get('cpu_busiest'))}",
            ]
        )
        ok = _phase_ok(p, slo)
        tones.append(None if p["kind"] == "ramp" or not p["sent"] else ("ok" if ok else "bad"))
    pdf.grid(
        ["Fase", "Rencana", "Tercapai", "Kirim", "Error", "202", "p95 masuk", "p95 e2e", "Tuntas", "CPU maks"],
        rows,
        (0.17, 0.08, 0.08, 0.06, 0.07, 0.07, 0.1, 0.1, 0.1, 0.17),
        size=7.2,
        tones=tones,
    )
    pdf.para(
        "Hijau: fase tahan yang memenuhi SLO; merah: melanggar (termasuk CPU rata-rata di atas "
        f"{slo.get('max_cpu_utilization', 0.75) * 100:.0f}%). Fase naik (peralihan laju) tidak dinilai. CPU maks: "
        "utilisasi rata-rata pod tersibuk dari service paling sibuk selama fase.",
        size=7.5,
        muted=True,
    )

    capacity = result.get("capacity")
    if capacity:
        pdf.h3("Kapasitas (stress)")
        target = slo.get("max_cpu_utilization", 0.75) * 100
        pdf.kv(
            [
                (f"Kapasitas berkelanjutan (CPU <= {target:.0f}%)", _rps(capacity["sustainable_rate"])),
                (
                    "Step pertama yang melanggar",
                    f"{capacity['break_label']}: {', '.join(capacity.get('break_reasons') or [])}"
                    if capacity["break_label"]
                    else "tidak ada (batas belum tercapai)",
                ),
                ("Batas menurut latensi/error saja", _rps(capacity.get("latency_rate"))),
                ("Step diuji", capacity["steps_tested"]),
            ]
        )
        pdf.para(
            "Rencanakan beban produksi di bawah kapasitas berkelanjutan, dan setel batas bulkhead/admission dari angka "
            "ini (hukum Little: request di sistem = laju x waktu layanan).",
            size=7.5,
            muted=True,
        )
    recovery = result.get("recovery")
    if recovery:
        pdf.h3("Pemulihan (spike)")
        pdf.kv(
            [
                ("Melanggar SLO saat lonjakan", "ya" if recovery["degraded_during_spike"] else "tidak"),
                (
                    "Waktu pulih setelah lonjakan",
                    "tidak pulih selama masa pulih"
                    if recovery["recovery_seconds"] is None
                    else _duration(recovery["recovery_seconds"]),
                ),
            ]
        )
    drift = result.get("drift")
    if drift:
        pdf.h3("Stabilitas (soak): seperempat awal vs akhir")
        pdf.kv(
            [
                ("p95 pintu masuk", f"{_ms(drift['entry_p95_first'])} -> {_ms(drift['entry_p95_last'])}"),
                ("p95 end-to-end", f"{_ms(drift['e2e_p95_first'])} -> {_ms(drift['e2e_p95_last'])}"),
            ]
        )

    pdf.h2("Per tahap pipeline")
    stage_rows = [
        [
            name,
            st["count"],
            _sec_text(st["p50"]),
            _sec_text(st["p95"]),
            _sec_text(st["max"]),
            _sec_text(st["gap_p50"]),
            _sec_text(st["gap_p95"]),
            st["failed"],
        ]
        for name, st in result["stages"].items()
    ]
    if stage_rows:
        pdf.grid(
            ["Tahap", "n", "p50", "p95", "maks", "jeda p50", "jeda p95", "gagal"],
            stage_rows,
            (0.2, 0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12),
            size=8,
        )
        pdf.para(
            "Guardrails: dari request dikirim sampai vonis tercatat (sinkron di orchestrator). Tahap lain: dari job "
            "dibuat sampai selesai. Jeda: serah-terima dari tahap sebelumnya lewat outbox.",
            size=7.5,
            muted=True,
        )
    else:
        pdf.para("Tidak ada data per tahap (database tidak terbaca atau sequence tanpa tahap).", muted=True)

    pdf.h2("Resource per service")
    res_rows, res_tones = [], []
    for service, r in resources.items():
        if r["cpu_max_m"] is None and not r["restarts"]:
            continue
        res_rows.append(
            [
                service,
                r["replicas"] or "-",
                f"{_num0(r['cpu_max_m'])}m / {_num0(r['cpu_limit_m'])}m",
                _pct(r.get("cpu_pct_avg")),
                _pct(r["cpu_pct_max"]),
                f"{_num0(r['mem_max_mi'])} / {_num0(r['mem_limit_mi'])} MiB",
                _pct(r["mem_pct_max"]),
                f"{r['restarts']} / {r['oom']}",
            ]
        )
        hot = (r.get("cpu_pct_avg") or 0) > slo.get("max_cpu_utilization", 0.75) or (r["mem_pct_max"] or 0) >= 0.85
        res_tones.append("bad" if r["restarts"] or r["oom"] else ("warn" if hot else None))
    if res_rows:
        pdf.grid(
            [
                "Service",
                "Replika",
                "CPU maks / limit",
                "CPU rata2",
                "CPU maks",
                "Memori maks / limit",
                "Mem %",
                "Restart / OOM",
            ],
            res_rows,
            (0.14, 0.08, 0.17, 0.09, 0.09, 0.2, 0.09, 0.14),
            size=7.6,
            tones=res_tones,
        )
        pdf.para(
            "generator = pod k6 sendiri; kalau CPU-nya tinggi, generator yang membatasi, bukan pipeline.",
            size=7.5,
            muted=True,
        )
    else:
        pdf.para("Data resource tidak tersedia (kubectl top / metrics-server).", muted=True)

    baseline = result.get("baseline")
    pdf.h2("Perbandingan dengan run sebelumnya")
    if baseline:
        b = baseline["summary"]
        pdf.para(
            f"Baseline: run {baseline['run_id']} ({_when(baseline.get('created_at'))}), profil dan sequence sama.",
            size=8.5,
        )

        def delta(now: Any, before: Any, *, lower_better: bool = True, pct: bool = False) -> str:
            if now is None or before is None:
                return "-"
            diff = now - before
            if pct:
                return f"{diff * 100:+.1f} poin"
            if not before:
                return f"{diff:+.0f}"
            change = diff / before * 100
            mark = (
                "lebih baik"
                if (change < 0) == lower_better and abs(change) >= 5
                else ("lebih buruk" if abs(change) >= 5 else "setara")
            )
            return f"{change:+.0f}% ({mark})"

        pdf.grid(
            ["Metrik", "Run ini", "Baseline", "Selisih"],
            [
                [
                    "Laju tercapai",
                    _rps(s["achieved_rate"]),
                    _rps(b["achieved_rate"]),
                    delta(s["achieved_rate"], b["achieved_rate"], lower_better=False),
                ],
                [
                    "p95 pintu masuk",
                    _ms(s["entry"]["p95"]),
                    _ms(b["entry"]["p95"]),
                    delta(s["entry"]["p95"], b["entry"]["p95"]),
                ],
                ["p95 end-to-end", _ms(s["e2e"]["p95"]), _ms(b["e2e"]["p95"]), delta(s["e2e"]["p95"], b["e2e"]["p95"])],
                [
                    "Error server",
                    _pct(s["error_rate"]),
                    _pct(b["error_rate"]),
                    delta(s["error_rate"], b["error_rate"], pct=True),
                ],
                [
                    "Rasio tuntas",
                    _pct(s["completion_ratio"]),
                    _pct(b["completion_ratio"]),
                    delta(s["completion_ratio"], b["completion_ratio"], pct=True),
                ],
                [
                    "Dokumen tuntas / menit",
                    _num1(s["completed_per_minute"]),
                    _num1(b["completed_per_minute"]),
                    delta(s["completed_per_minute"], b["completed_per_minute"], lower_better=False),
                ],
            ],
            (0.3, 0.22, 0.22, 0.26),
        )
        changed = {
            k: v for k, v in (meta.get("versions") or {}).items() if (baseline.get("versions") or {}).get(k) != v
        }
        if changed:
            pdf.para("Image yang berbeda dari baseline: " + ", ".join(f"{k} {v}" for k, v in changed.items()), size=8)
    else:
        pdf.para(
            "Belum ada run sebelumnya dengan profil dan sequence yang sama. Run ini menjadi baseline berikutnya.",
            muted=True,
        )

    for heading, items in (
        ("Error yang paling sering", result.get("errors")),
        ("Gagal di tahap", result.get("failures")),
        ("Alasan penolakan", result.get("rejections")),
    ):
        if items:
            pdf.h3(heading)
            pdf.grid(["Pesan", "Jumlah"], [[text, count] for text, count in items], (0.85, 0.15), size=7.6)

    # --- konfigurasi & metodologi --------------------------------------------------------
    pdf.h2("Konfigurasi run")
    params = meta.get("params") or {}
    defaults = info.get("defaults", {})
    pdf.kv(
        [(PARAM_LABELS.get(k, k), params.get(k, d)) for k, d in defaults.items()]
        + [
            ("Batas tunggu orchestrator", f"{meta.get('wait_seconds')} dtk (PIPELINE_WAIT_SECONDS)"),
            ("Generator", f"{meta.get('k6_image')}, Job di namespace {meta.get('namespace')}, VU {meta.get('vus')}"),
            ("Penghentian otomatis", "ya" if meta.get("auto_abort") else "hanya batas keamanan (error > 50%)"),
            (
                "Alasan dihentikan",
                meta.get("aborted_reason") or ("dihentikan pengguna" if meta.get("stop_requested") else "-"),
            ),
        ]
    )
    pdf.h3("Rencana beban")
    pdf.grid(
        ["Fase", "Jenis", "Laju (rps)", "Mulai", "Lama"],
        [
            [
                p["label"],
                "naik/turun" if p["kind"] == "ramp" else "tahan",
                f"{p['start_rate']:g} -> {p['end_rate']:g}",
                _duration(p["offset"]),
                _duration(p["seconds"]),
            ]
            for p in meta.get("plan", [])
        ],
        (0.34, 0.14, 0.2, 0.16, 0.16),
        size=7.8,
    )
    pdf.h3("Versi yang diuji")
    pdf.kv([(k, v) for k, v in (meta.get("versions") or {}).items()] or [("-", "-")])
    k6 = result.get("k6") or {}
    if k6:
        pdf.h3("Ringkasan k6")
        pdf.kv(
            [
                ("Iterasi / request", f"{k6.get('iterations')} / {k6.get('http_reqs')}"),
                ("Iterasi terlewat (dropped)", k6.get("dropped_iterations")),
                ("VU maksimum", k6.get("vus_max")),
                (
                    "Data terkirim / diterima",
                    f"{(k6.get('data_sent') or 0) / 1e6:.1f} MB / {(k6.get('data_received') or 0) / 1e6:.1f} MB",
                ),
                (
                    "Threshold k6",
                    ", ".join(f"{k}: {'ok' if ok else 'terlewati'}" for k, ok in (k6.get("thresholds") or {}).items())
                    or "-",
                ),
            ]
        )
    pdf.h2("Metodologi")
    pdf.para(
        "Generator beban (k6) berjalan sebagai Job di namespace pipeline yang sama dan menembak Service orchestrator "
        "langsung, sehingga yang terukur adalah service, bukan jaringan laptop atau tunnel port-forward. Request "
        "dikirim ke endpoint -test (TESTING_ENDPOINTS): pipeline yang sama, tabel nilam_testing_*, tanpa callback ke "
        "Orkestrasi pusat, sehingga data produksi/dev tidak tercemar. Kedatangan request memakai open model "
        "(laju tetap per detik, tidak bergantung pada lama jawaban), seperti trafik sungguhan.",
        size=8.5,
    )
    pdf.para(
        "Selama run tracker mencatat CPU/memori tiap pod (kubectl top), restart dan OOMKilled, serta status job "
        "tiap tahap di database (lewat pod relay Cloud SQL) untuk end-to-end dan durasi per tahap. Setelah k6 "
        "selesai, tracker menunggu antrean pipeline habis (maks. 3 menit) sebelum menghitung hasil.",
        size=8.5,
    )
    for name, text in TEST_TYPES:
        pdf.h3(name)
        pdf.para(text, size=8.3)
    pdf.h2("Definisi metrik")
    for name, text in DEFINITIONS:
        pdf.h3(name)
        pdf.para(text, size=8.3)
    pdf.h2("Batasan")
    for text in (
        "Cluster dev dipakai bersama tim lain dan kapasitasnya terbatas; angka absolut tidak sama dengan produksi. "
        "Bandingkan run dengan baseline di cluster yang sama.",
        "Model OCR berjalan di VM terpisah (bukan di cluster): kapasitas extraction ikut dibatasi VM itu.",
        "Endpoint -test tidak mengirim callback, sehingga pengiriman callback ke Orkestrasi pusat tidak ikut terukur.",
        "Resource disampel tiap ~15 detik (metrics-server); lonjakan sangat singkat bisa tidak terlihat.",
    ):
        pdf.bullet("info", text)
    return bytes(pdf.output())


def _nice_max(value: float) -> float:
    """Batas atas sumbu dengan empat garis bantu berangka bulat (1, 2, 2.5, 5 x 10^n per garis)."""
    raw = value / 4
    exponent = 10 ** math.floor(math.log10(raw))
    step = next(n * exponent for n in (1, 2, 2.5, 5, 10) if raw <= n * exponent)
    return step * 4


def _sec(ms: Any) -> float | None:
    return None if ms is None else ms / 1000


def _sec_text(seconds: Any) -> str:
    return "-" if seconds is None else _ms(seconds * 1000)


def _num0(value: Any) -> str:
    return "-" if value is None else f"{value:.0f}"


def _num1(value: Any) -> str:
    return "-" if value is None else f"{value:.1f}"


def _phase_ok(p: dict[str, Any], slo: dict[str, float]) -> bool:
    from perf_stats import step_problems

    return not step_problems(p, slo) if p["sent"] else True


def _short(service: str | None) -> str:
    """Nama service yang muat di kolom sempit tabel fase."""
    return {"orchestrator": "orch", "guardrails": "guard", "extraction": "extr", "structuring": "struct"}.get(
        service or "", service or ""
    )
