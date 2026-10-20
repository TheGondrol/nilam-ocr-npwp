"""Performance test GKE: profil uji, rencana beban, dan perhitungan hasil (statistik, SLO, temuan). Murni: tanpa
Redis, kubectl, atau database, supaya bisa diuji sendiri (test_perf.py). perftest.py mengumpulkan datanya,
perf_report.py menulis PDF-nya.

Data mentah satu run:
  samples    request_id -> {"s": status HTTP (0 = timeout), "t": epoch kirim, "ms": lama jawaban, "e": errors, ...}
             (satu baris log `PT|...` k6 per request)
  db         snapshot tabel nilam_testing_* (perftest._fetch_db): {"stages": {service: {request_id: [status,
             created, updated, error]}}, "guard": {request_id: [passed, created]}, "dead": n, "fetched_at": ts}
  resources  sampel `kubectl top pods` tiap ~15 dtk: [{"ts", "pods": {pod: [cpu_m, mem_mi]}}]
  pods       restart per pod selama run: {pod: {"service", "restarts", "reasons": [...]}}
"""

import math
from collections import Counter
from typing import Any

PIPELINE_NAMES = ("guardrails", "extraction", "structuring", "scoring")
STAGE_SERVICES = ("extraction", "structuring", "scoring")
SERVICES = ("orchestrator", "guardrails", "extraction", "structuring", "scoring")
BUCKETS = ("200", "rejected", "202", "4xx", "5xx", "timeout")
REJECTED_CODE = "DOWNSTREAM_VALIDATION_ERROR"
WINDOW_SECONDS = 30
# Pesan job FAILED karena galat di service sendiri (ocr_common.pipeline.stage); FAILED structuring lainnya berisi
# alasan aturan yang menolak dokumen.
INTERNAL_ERROR_PREFIX = "Internal error in"

SLO_DEFAULTS = {
    "error_rate": 0.01,
    "entry_p95_ms": 15000.0,
    "e2e_p95_ms": 60000.0,
    "min_completion": 0.99,
    "max_restarts": 0,
    # Queueing theory: waiting time grows as rho / (1 - rho), 3x at 75% utilisation, 9x at 90%. A service run above
    # this on average has no headroom left for a burst.
    "max_cpu_utilization": 0.75,
}

SLO_LABELS = {
    "error_rate": "Error server (5xx + timeout)",
    "entry_p95_ms": "p95 jawaban /v1/extract-ocr",
    "e2e_p95_ms": "p95 end-to-end (kirim sampai tuntas)",
    "min_completion": "Rasio tuntas",
    "max_restarts": "Restart / OOMKilled pod pipeline",
    "max_cpu_utilization": "Utilisasi CPU rata-rata pod tersibuk (per fase)",
}

PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "label": "Smoke",
        "question": "Apakah semuanya hidup dan menjawab benar dengan beban sangat kecil?",
        "summary": "Beban sangat kecil dalam waktu singkat; dijalankan pertama, sebelum tes lain.",
        "defaults": {"base_rate": 0.2, "duration_seconds": 60},
        "heavy": False,
        "auto_abort": False,
    },
    "load": {
        "label": "Load (baseline)",
        "question": "Apakah beban normal yang diharapkan dilayani sesuai SLO?",
        "summary": "Naik perlahan ke laju normal, lalu ditahan. Hasilnya jadi baseline untuk dibandingkan antar rilis.",
        "defaults": {"base_rate": 0.5, "ramp_seconds": 30, "duration_seconds": 300},
        "heavy": False,
        "auto_abort": False,
    },
    "stress": {
        "label": "Stress",
        "question": "Sampai laju berapa SLO masih terpenuhi, dan bagian mana yang jenuh lebih dulu?",
        "summary": "Laju dinaikkan bertahap per step; berhenti otomatis begitu satu step jelas melanggar SLO.",
        "defaults": {"start_rate": 0.5, "peak_rate": 3.0, "step_rate": 0.5, "step_seconds": 120},
        "heavy": True,
        "auto_abort": True,
    },
    "spike": {
        "label": "Spike",
        "question": "Apa yang terjadi saat beban melonjak mendadak, dan berapa lama pulihnya?",
        "summary": "Laju dasar, lonjakan mendadak, lalu kembali ke laju dasar untuk mengukur waktu pulih.",
        "defaults": {
            "base_rate": 0.3,
            "peak_rate": 3.0,
            "baseline_seconds": 60,
            "spike_seconds": 60,
            "recovery_seconds": 180,
        },
        "heavy": True,
        "auto_abort": False,
    },
    "soak": {
        "label": "Soak",
        "question": "Apakah sistem stabil dalam waktu lama (kebocoran memori, latensi yang memburuk)?",
        "summary": "Laju sedang ditahan lama; membandingkan awal vs akhir dan tren memori tiap service.",
        "defaults": {"base_rate": 0.3, "ramp_seconds": 60, "duration_seconds": 1800},
        "heavy": True,
        "auto_abort": False,
    },
}

PARAM_LABELS = {
    "base_rate": "Laju dasar (rps)",
    "duration_seconds": "Durasi tahan (detik)",
    "ramp_seconds": "Naik bertahap (detik)",
    "start_rate": "Laju awal (rps)",
    "peak_rate": "Laju puncak (rps)",
    "step_rate": "Kenaikan per step (rps)",
    "step_seconds": "Lama tiap step (detik)",
    "baseline_seconds": "Laju dasar sebelum lonjakan (detik)",
    "spike_seconds": "Lama lonjakan (detik)",
    "recovery_seconds": "Masa pulih (detik)",
}

STEP_RAMP_SECONDS = 10
SPIKE_EDGE_SECONDS = 5
MAX_STEPS = 20


class PlanError(ValueError):
    pass


def _phase(name: str, label: str, kind: str, start: float, end: float, seconds: int) -> dict[str, Any]:
    return {"name": name, "label": label, "kind": kind, "start_rate": start, "end_rate": end, "seconds": seconds}


def build_plan(profile: str, params: dict[str, Any], *, max_rate: float, max_seconds: int) -> list[dict[str, Any]]:
    """Fase-fase run: {name, label, kind: ramp | hold, start_rate, end_rate, seconds, offset}. Fase `hold` yang dinilai
    per fase (step stress, lonjakan, masa pulih); `ramp` hanya peralihan laju."""
    if profile not in PROFILES:
        raise PlanError(f"profil tidak dikenal: {profile}")
    p = {**PROFILES[profile]["defaults"], **{k: v for k, v in params.items() if v not in (None, "")}}
    try:
        p = {k: float(v) for k, v in p.items() if k in PROFILES[profile]["defaults"]}
    except (TypeError, ValueError):
        raise PlanError("parameter profil harus angka") from None

    def rate(key: str) -> float:
        value = round(p[key], 1)
        if not 0.1 <= value <= max_rate:
            raise PlanError(f"{PARAM_LABELS[key]} harus 0.1..{max_rate:g} rps")
        return value

    def seconds(key: str, low: int = 5) -> int:
        value = int(p[key])
        if value < low:
            raise PlanError(f"{PARAM_LABELS[key]} minimal {low} detik")
        return value

    phases: list[dict[str, Any]] = []
    if profile == "smoke":
        base = rate("base_rate")
        phases.append(_phase("steady", f"Smoke {base:g} rps", "hold", base, base, seconds("duration_seconds")))
    elif profile in ("load", "soak"):
        base = rate("base_rate")
        phases.append(_phase("warmup", "Naik bertahap", "ramp", 0.0, base, seconds("ramp_seconds", 0) or 1))
        name = "soak" if profile == "soak" else "steady"
        phases.append(_phase(name, f"Tahan {base:g} rps", "hold", base, base, seconds("duration_seconds", 10)))
    elif profile == "stress":
        start, peak, step = rate("start_rate"), rate("peak_rate"), rate("step_rate")
        if peak < start:
            raise PlanError("laju puncak harus >= laju awal")
        levels = []
        level = start
        while level <= peak + 1e-9:
            levels.append(round(level, 1))
            level += step
        if len(levels) > MAX_STEPS:
            raise PlanError(f"terlalu banyak step ({len(levels)}); maksimal {MAX_STEPS}")
        hold = seconds("step_seconds", 30)
        previous = 0.0
        for index, level in enumerate(levels, start=1):
            phases.append(_phase(f"ramp{index}", f"Naik ke {level:g} rps", "ramp", previous, level, STEP_RAMP_SECONDS))
            phases.append(_phase(f"step{index}", f"Step {index}: {level:g} rps", "hold", level, level, hold))
            previous = level
    elif profile == "spike":
        base, peak = rate("base_rate"), rate("peak_rate")
        if peak <= base:
            raise PlanError("laju lonjakan harus > laju dasar")
        phases += [
            _phase("warmup", "Naik bertahap", "ramp", 0.0, base, 15),
            _phase("baseline", f"Dasar {base:g} rps", "hold", base, base, seconds("baseline_seconds", 30)),
            _phase("spike_up", "Lonjakan naik", "ramp", base, peak, SPIKE_EDGE_SECONDS),
            _phase("spike", f"Lonjakan {peak:g} rps", "hold", peak, peak, seconds("spike_seconds", 10)),
            _phase("spike_down", "Lonjakan turun", "ramp", peak, base, SPIKE_EDGE_SECONDS),
            _phase("recovery", f"Pulih di {base:g} rps", "hold", base, base, seconds("recovery_seconds", 30)),
        ]
    offset = 0
    for phase in phases:
        phase["offset"] = offset
        offset += phase["seconds"]
    if offset > max_seconds:
        raise PlanError(f"total durasi {offset} detik melebihi batas {max_seconds} detik")
    return phases


def plan_seconds(plan: list[dict[str, Any]]) -> int:
    return sum(p["seconds"] for p in plan)


def plan_peak(plan: list[dict[str, Any]]) -> float:
    return max((max(p["start_rate"], p["end_rate"]) for p in plan), default=0.0)


def k6_stages(plan: list[dict[str, Any]]) -> tuple[int, list[list[int]]]:
    """(START_RATE, STAGES) skrip perf.js: laju per 10 detik, jadi resolusinya 0.1 rps."""
    start = round(plan[0]["start_rate"] * 10) if plan else 0
    return start, [[round(p["end_rate"] * 10), int(p["seconds"])] for p in plan]


def target_rate(plan: list[dict[str, Any]], offset: float) -> float:
    for phase in plan:
        if phase["offset"] <= offset < phase["offset"] + phase["seconds"]:
            share = (offset - phase["offset"]) / phase["seconds"] if phase["seconds"] else 1.0
            return phase["start_rate"] + (phase["end_rate"] - phase["start_rate"]) * share
    return 0.0


def phase_at(plan: list[dict[str, Any]], offset: float) -> dict[str, Any] | None:
    for phase in plan:
        if phase["offset"] <= offset < phase["offset"] + phase["seconds"]:
            return phase
    return None


def vus_for(peak_rate: float, wait_seconds: float) -> tuple[int, int]:
    """(preAllocatedVUs, maxVUs): koneksi ditahan sampai WAIT detik, jadi VU sibuk ~ laju x (WAIT + 10)."""
    busy = math.ceil(peak_rate * (wait_seconds + 10))
    return max(5, busy), min(600, max(10, math.ceil(busy * 1.5)))


# --- statistik dasar ---------------------------------------------------------------


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def trend(values: list[float]) -> dict[str, float | None]:
    values = sorted(values)
    return {
        "count": len(values),
        "avg": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": values[-1] if values else None,
    }


def bucket(sample: dict[str, Any]) -> str:
    status = int(sample.get("s") or 0)
    if status == 400 and sample.get("e") == REJECTED_CODE:
        return "rejected"
    if status == 200:
        return "200"
    if status == 202:
        return "202"
    if status == 0:
        return "timeout"
    return "5xx" if status >= 500 else "4xx"


def slope_per_hour(points: list[tuple[float, float]]) -> float | None:
    """Kemiringan regresi linear (satuan y per jam) dari (epoch, y)."""
    if len(points) < 3:
        return None
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den * 3600


# --- hasil per request ---------------------------------------------------------------


def request_outcomes(
    samples: dict[str, dict[str, Any]], db: dict[str, Any] | None, sequence: list[str] | None
) -> dict[str, dict[str, Any]]:
    """Per request: state (done | rejected | failed | in_flight | not_started), waktu tuntas, durasi tiap tahap dan jeda
    antar tahap (detik). Tanpa snapshot database hanya jawaban pintu masuk yang diketahui."""
    names = sequence or list(PIPELINE_NAMES)
    stages = [s for s in names if s in STAGE_SERVICES]
    db_stages = (db or {}).get("stages", {})
    guard = (db or {}).get("guard", {})
    out: dict[str, dict[str, Any]] = {}
    for rid, sample in samples.items():
        b = bucket(sample)
        t0 = float(sample["t"])
        answered = t0 + float(sample.get("ms") or 0) / 1000
        row: dict[str, Any] = {"bucket": b, "t": t0, "stage_s": {}, "gap_s": {}, "done_at": None, "reason": None}
        if b == "rejected":
            row.update(state="rejected", reason=sample.get("m") or "ditolak")
        elif b in ("4xx", "5xx"):
            row["state"] = "not_started"
        elif not stages:
            # Sequence ["guardrails"]: jawaban 200 sudah final.
            row.update(state="done" if b == "200" else "in_flight", done_at=answered if b == "200" else None)
        elif db is None:
            row["state"] = "unknown"
        else:
            row["state"] = "in_flight"
            previous_end = None
            g = guard.get(rid)
            if g is not None:
                row["stage_s"]["guardrails"] = max(0.0, float(g[1]) - t0)
                previous_end = float(g[1])
            for stage in stages:
                job = db_stages.get(stage, {}).get(rid)
                if job is None:
                    break
                status, created, updated, error = job
                created, updated = float(created), float(updated)
                if previous_end is not None:
                    row["gap_s"][stage] = max(0.0, created - previous_end)
                if status in ("DONE", "FAILED"):
                    row["stage_s"][stage] = max(0.0, updated - created)
                previous_end = updated
                if status == "FAILED":
                    internal = str(error or "").startswith(INTERNAL_ERROR_PREFIX)
                    if stage == "structuring" and not internal:
                        row.update(state="rejected", reason=error or "ditolak aturan structuring")
                    else:
                        row.update(state="failed", reason=f"{stage}: {error or '-'}")
                    break
                if status != "DONE":
                    break
                if stage == stages[-1]:
                    row.update(state="done", done_at=updated)
        if row["done_at"] is not None:
            row["e2e_ms"] = max(0.0, (row["done_at"] - t0) * 1000)
        out[rid] = row
    return out


# --- jendela waktu dan fase ------------------------------------------------------------


def _window_stats(rows: list[dict[str, Any]], samples: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(bucket(s) for s in samples)
    sent = len(samples)
    entry = [float(s.get("ms") or 0) for s in samples]
    e2e = [r["e2e_ms"] for r in rows if r.get("e2e_ms") is not None]
    errors = counts.get("5xx", 0) + counts.get("timeout", 0)
    return {
        "sent": sent,
        "counts": {k: counts.get(k, 0) for k in BUCKETS},
        "error_rate": errors / sent if sent else None,
        "share_202": counts.get("202", 0) / sent if sent else None,
        "entry_p50": percentile(entry, 0.5),
        "entry_p95": percentile(entry, 0.95),
        "e2e_p95": percentile(e2e, 0.95),
        "completed_of_sent": sum(1 for r in rows if r["state"] == "done"),
        "unfinished": sum(1 for r in rows if r["state"] in ("in_flight", "failed")),
    }


def windows(
    samples: dict[str, dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
    plan: list[dict[str, Any]],
    start: float,
    end: float,
    size: int = WINDOW_SECONDS,
) -> list[dict[str, Any]]:
    """Per jendela `size` detik sejak `start`: request yang DIKIRIM di jendela itu (laju, campuran status, latensi,
    e2e-nya) dan request yang TUNTAS di jendela itu (throughput keluar)."""
    if end <= start:
        return []
    count = max(1, math.ceil((end - start) / size))
    by_window: list[list[str]] = [[] for _ in range(count)]
    for rid, s in samples.items():
        index = int((float(s["t"]) - start) // size)
        if 0 <= index < count:
            by_window[index].append(rid)
    finished = [0] * count
    for row in outcomes.values():
        if row.get("done_at") is not None:
            index = int((row["done_at"] - start) // size)
            if 0 <= index < count:
                finished[index] += 1
    result = []
    for index, rids in enumerate(by_window):
        t = index * size
        width = min(size, end - start - t) or size
        stats = _window_stats([outcomes[r] for r in rids], [samples[r] for r in rids])
        result.append(
            {
                "t": t,
                "target_rate": target_rate(plan, t + width / 2),
                "rate": stats["sent"] / width,
                "done_rate": finished[index] / width,
                "phase": (phase_at(plan, t + width / 2) or {}).get("name"),
                **stats,
            }
        )
    return result


def phase_stats(
    samples: dict[str, dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
    plan: list[dict[str, Any]],
    start: float,
    cpu: dict[str, list[tuple[float, float]]] | None = None,
) -> list[dict[str, Any]]:
    """Per fase: request yang dikirim di fase itu, dan (dengan `cpu`, dari `cpu_utilization`) utilisasi CPU rata-rata
    pod tersibuk tiap service selama fase: `cpu_util` {service: rasio}, `cpu_util_max` dan `cpu_busiest` (service
    pipeline dengan utilisasi tertinggi; generator k6 tidak ikut)."""
    rows = []
    for phase in plan:
        lo, hi = start + phase["offset"], start + phase["offset"] + phase["seconds"]
        rids = [rid for rid, s in samples.items() if lo <= float(s["t"]) < hi]
        stats = _window_stats([outcomes[r] for r in rids], [samples[r] for r in rids])
        util = {
            service: sum(values) / len(values)
            for service, points in (cpu or {}).items()
            if (values := [u for ts, u in points if lo <= ts < hi])
        }
        pipeline = {s: u for s, u in util.items() if s != "generator"}
        busiest = max(pipeline, key=pipeline.__getitem__) if pipeline else None
        rows.append(
            {
                **{k: phase[k] for k in ("name", "label", "kind", "start_rate", "end_rate", "seconds", "offset")},
                "achieved_rate": stats["sent"] / phase["seconds"] if phase["seconds"] else None,
                **stats,
                "cpu_util": util,
                "cpu_util_max": pipeline[busiest] if busiest else None,
                "cpu_busiest": busiest,
            }
        )
    return rows


# --- resource ------------------------------------------------------------------------


def service_of_pod(pod: str, release: str) -> str | None:
    if pod.startswith("nilam-perf-"):
        return "generator"
    for service in SERVICES:
        if pod.startswith(f"{release}-{service}-"):
            return service
    return None


def cpu_utilization(
    resources: list[dict[str, Any]], limits: dict[str, Any], release: str
) -> dict[str, list[tuple[float, float]]]:
    """Per service, (epoch, utilisasi) tiap sampel: CPU pod tersibuk dibanding limit CPU per pod. Limit, bukan request:
    di limit itulah pod mulai di-throttle (HPA memakai request, jadi angkanya tidak sama dengan target HPA)."""
    out: dict[str, list[tuple[float, float]]] = {}
    for sample in resources:
        busiest: dict[str, float] = {}
        for pod, (cpu, _mem) in sample.get("pods", {}).items():
            service = service_of_pod(pod, release)
            if service is not None:
                busiest[service] = max(busiest.get(service, 0.0), float(cpu))
        for service, cpu in busiest.items():
            limit = (limits.get(service) or {}).get("cpu_m")
            if limit:
                out.setdefault(service, []).append((float(sample["ts"]), cpu / limit))
    return out


def resource_stats(
    resources: list[dict[str, Any]], pods: dict[str, dict[str, Any]], limits: dict[str, Any], release: str
) -> dict[str, Any]:
    """Per service: CPU/memori tertinggi satu pod dibanding limit per pod, rata-rata, tren memori, restart."""
    per: dict[str, dict[str, Any]] = {}
    for sample in resources:
        ts = float(sample["ts"])
        totals: dict[str, list[float]] = {}
        for pod, (cpu, mem) in sample.get("pods", {}).items():
            service = service_of_pod(pod, release)
            if service is None:
                continue
            entry = per.setdefault(service, {"cpu": [], "mem": [], "mem_points": [], "series": []})
            entry["cpu"].append(float(cpu))
            entry["mem"].append(float(mem))
            t = totals.setdefault(service, [0.0, 0.0, 0.0])
            t[0] = max(t[0], float(cpu))
            t[1] = max(t[1], float(mem))
            t[2] += float(mem)
        for service, (cpu_max, mem_max, mem_sum) in totals.items():
            per[service]["series"].append([ts, cpu_max, mem_max])
            per[service]["mem_points"].append((ts, mem_sum))
    out = {}
    for service in [*SERVICES, "generator"]:
        entry = per.get(service)
        limit = limits.get(service, {})
        restarts = [p for p in pods.values() if p.get("service") == service]
        cpu_limit, mem_limit = limit.get("cpu_m"), limit.get("mem_mi")
        cpu_max = max(entry["cpu"]) if entry else None
        mem_max = max(entry["mem"]) if entry else None
        out[service] = {
            "replicas": limit.get("replicas"),
            "cpu_limit_m": cpu_limit,
            "mem_limit_mi": mem_limit,
            "cpu_max_m": cpu_max,
            "cpu_avg_m": sum(entry["cpu"]) / len(entry["cpu"]) if entry else None,
            "mem_max_mi": mem_max,
            "cpu_pct_max": cpu_max / cpu_limit if cpu_max is not None and cpu_limit else None,
            "cpu_pct_avg": (
                sum(s[1] for s in entry["series"]) / len(entry["series"]) / cpu_limit
                if entry and entry["series"] and cpu_limit
                else None
            ),
            "mem_pct_max": mem_max / mem_limit if mem_max is not None and mem_limit else None,
            "mem_slope_mi_per_h": slope_per_hour(entry["mem_points"]) if entry else None,
            "restarts": sum(int(p.get("restarts") or 0) for p in restarts),
            "oom": sum(1 for p in restarts for r in p.get("reasons", []) if r == "OOMKilled"),
            "series": entry["series"] if entry else [],
        }
    return out


# --- SLO dan kesimpulan ----------------------------------------------------------------


def _slo_row(key: str, target: Any, actual: Any, ok: bool | None, note: str = "") -> dict[str, Any]:
    return {"key": key, "label": SLO_LABELS[key], "target": target, "actual": actual, "ok": ok, "note": note}


def evaluate_slo(
    summary: dict[str, Any],
    slo: dict[str, float],
    resources: dict[str, Any],
    phases: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    err = summary["error_rate"]
    rows.append(_slo_row("error_rate", slo["error_rate"], err, None if err is None else err < slo["error_rate"]))
    p95 = summary["entry"]["p95"]
    rows.append(_slo_row("entry_p95_ms", slo["entry_p95_ms"], p95, None if p95 is None else p95 < slo["entry_p95_ms"]))
    e2e = summary["e2e"]["p95"]
    note = "" if summary["db_available"] else "database tidak terbaca: e2e tidak terukur"
    rows.append(_slo_row("e2e_p95_ms", slo["e2e_p95_ms"], e2e, None if e2e is None else e2e < slo["e2e_p95_ms"], note))
    ratio = summary["completion_ratio"]
    rows.append(
        _slo_row(
            "min_completion",
            slo["min_completion"],
            ratio,
            None if ratio is None else ratio >= slo["min_completion"],
            "" if summary["db_available"] else "database tidak terbaca",
        )
    )
    restarts = sum(r["restarts"] + r["oom"] for s, r in resources.items() if s != "generator")
    rows.append(_slo_row("max_restarts", slo["max_restarts"], restarts, restarts <= slo["max_restarts"]))
    measured = [p for p in phases or [] if p["kind"] == "hold" and p["sent"] and p.get("cpu_util_max") is not None]
    if measured:
        worst = max(measured, key=lambda p: p["cpu_util_max"])
        target = slo["max_cpu_utilization"]
        rows.append(
            _slo_row(
                "max_cpu_utilization",
                target,
                worst["cpu_util_max"],
                worst["cpu_util_max"] <= target,
                f"{worst['cpu_busiest']} di {worst['label']}",
            )
        )
    else:
        rows.append(
            _slo_row("max_cpu_utilization", slo["max_cpu_utilization"], None, None, "data CPU (kubectl top) tidak ada")
        )
    return rows


def step_problems(row: dict[str, Any], slo: dict[str, float], *, cpu: bool = True) -> list[str]:
    """Kriteria SLO yang dilanggar satu fase/jendela (kosong = memenuhi semuanya). `cpu=False`: utilisasi CPU tidak
    dinilai (jendela waktu tidak membawanya, atau kapasitas murni menurut latensi)."""
    if row["sent"] == 0:
        return ["tidak ada request"]
    problems = []
    if row["error_rate"] is not None and row["error_rate"] >= slo["error_rate"]:
        problems.append(f"error {row['error_rate'] * 100:.1f}%")
    if row["entry_p95"] is not None and row["entry_p95"] >= slo["entry_p95_ms"]:
        problems.append(f"p95 pintu masuk {row['entry_p95'] / 1000:.1f} dtk")
    if row.get("e2e_p95") is not None and row["e2e_p95"] >= slo["e2e_p95_ms"]:
        problems.append(f"p95 end-to-end {row['e2e_p95'] / 1000:.1f} dtk")
    target = slo.get("max_cpu_utilization")
    if cpu and target is not None and row.get("cpu_util_max") is not None and row["cpu_util_max"] > target:
        problems.append(f"CPU {row['cpu_busiest']} {row['cpu_util_max'] * 100:.0f}%")
    return problems


def _step_ok(row: dict[str, Any], slo: dict[str, float], *, cpu: bool = True) -> bool:
    return not step_problems(row, slo, cpu=cpu)


def _first_breach(steps: list[dict[str, Any]], slo: dict[str, float], *, cpu: bool) -> tuple[Any, Any]:
    """(step tertinggi yang lulus berturut-turut dari step pertama, step pertama yang melanggar)."""
    sustainable, breaking = None, None
    for step in steps:
        if _step_ok(step, slo, cpu=cpu):
            if breaking is None:
                sustainable = step
        elif breaking is None:
            breaking = step
    return sustainable, breaking


def stress_capacity(phases: list[dict[str, Any]], slo: dict[str, float]) -> dict[str, Any]:
    """Laju step tertinggi yang semua SLO-nya terpenuhi berturut-turut dari step pertama, termasuk utilisasi CPU <=
    target (queueing theory: kapasitas yang direncanakan), dan step pertama yang melanggar beserta alasannya. Juga
    laju tertinggi menurut latensi/error saja, tanpa CPU (batas sebelum pengguna merasakan)."""
    steps = [p for p in phases if p["kind"] == "hold" and p["sent"] > 0]
    sustainable, breaking = _first_breach(steps, slo, cpu=True)
    latency_ok, latency_break = _first_breach(steps, slo, cpu=False)
    return {
        "sustainable_rate": sustainable["end_rate"] if sustainable else None,
        "sustainable_label": sustainable["label"] if sustainable else None,
        "break_rate": breaking["end_rate"] if breaking else None,
        "break_label": breaking["label"] if breaking else None,
        "break_reasons": step_problems(breaking, slo) if breaking else [],
        "latency_rate": latency_ok["end_rate"] if latency_ok else None,
        "latency_break_label": latency_break["label"] if latency_break else None,
        "cpu_measured": any(s.get("cpu_util_max") is not None for s in steps),
        "steps_tested": len(steps),
    }


def spike_recovery(
    windows_: list[dict[str, Any]], plan: list[dict[str, Any]], slo: dict[str, float]
) -> dict[str, Any] | None:
    """Detik dari akhir lonjakan sampai jendela pertama yang kembali memenuhi SLO (error dan p95 pintu masuk)."""
    spike = next((p for p in plan if p["name"] == "spike"), None)
    if spike is None:
        return None
    end = spike["offset"] + spike["seconds"]
    during = [w for w in windows_ if spike["offset"] <= w["t"] < end and w["sent"]]
    degraded = any(not _step_ok(w | {"e2e_p95": None}, slo, cpu=False) for w in during)
    after = [w for w in windows_ if w["t"] >= end and w["sent"]]
    for w in after:
        if _step_ok(w | {"e2e_p95": None}, slo, cpu=False):
            return {"degraded_during_spike": degraded, "recovery_seconds": max(0, w["t"] - end)}
    return {"degraded_during_spike": degraded, "recovery_seconds": None}


def soak_drift(windows_: list[dict[str, Any]], plan: list[dict[str, Any]]) -> dict[str, Any] | None:
    soak = next((p for p in plan if p["name"] == "soak"), None)
    if soak is None:
        return None
    hold = [w for w in windows_ if soak["offset"] <= w["t"] < soak["offset"] + soak["seconds"] and w["sent"]]
    if len(hold) < 4:
        return None
    quarter = max(1, len(hold) // 4)

    def avg(rows: list[dict[str, Any]], key: str) -> float | None:
        values = [r[key] for r in rows if r[key] is not None]
        return sum(values) / len(values) if values else None

    first, last = hold[:quarter], hold[-quarter:]
    return {
        "entry_p95_first": avg(first, "entry_p95"),
        "entry_p95_last": avg(last, "entry_p95"),
        "e2e_p95_first": avg(first, "e2e_p95"),
        "e2e_p95_last": avg(last, "e2e_p95"),
    }


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _secs(ms: float | None) -> str:
    return "-" if ms is None else (f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} dtk")


def findings(result: dict[str, Any]) -> list[tuple[str, str]]:
    """Temuan otomatis, (nada ok | warn | bad | info, kalimat), urut dari yang paling penting."""
    out: list[tuple[str, str]] = []
    s = result["summary"]
    counts = s["counts"]
    sent = s["sent"] or 0
    if result.get("aborted_reason"):
        out.append(("bad", f"Run dihentikan otomatis: {result['aborted_reason']}."))
    errors = counts["5xx"] + counts["timeout"]
    if errors:
        out.append(("bad", f"{errors} request ({_pct(errors / sent)}) dijawab 5xx atau timeout."))
    for service, r in result["resources"].items():
        if service == "generator":
            continue
        if r["oom"]:
            out.append(("bad", f"Pod {service} OOMKilled {r['oom']}x selama run: limit memori terlampaui."))
        elif r["restarts"]:
            out.append(("bad", f"Pod {service} restart {r['restarts']}x selama run."))
    first_202 = next((w for w in result["windows"] if w["counts"]["202"]), None)
    if first_202:
        out.append(
            (
                "warn",
                f"Jawaban 202 mulai muncul pada menit {first_202['t'] / 60:.1f} (laju ~{first_202['target_rate']:.1f} "
                f"rps): pipeline lebih lama dari batas tunggu orchestrator, hasilnya menyusul lewat callback.",
            )
        )
    # Utilisasi CPU rata-rata fase tahan di atas target (queueing theory): fase pertama yang melewatinya, per service.
    target = result["slo_targets"].get("max_cpu_utilization", 0.75)
    over: dict[str, dict[str, Any]] = {}
    for phase in result["phases"]:
        if phase["kind"] != "hold" or not phase["sent"]:
            continue
        for service, util in (phase.get("cpu_util") or {}).items():
            if service != "generator" and util > target and service not in over:
                over[service] = {"util": util, "phase": phase}
    for service, hit in over.items():
        rate = hit["phase"]["end_rate"]
        out.append(
            (
                "warn",
                f"CPU {service} rata-rata {_pct(hit['util'])} limit di {hit['phase']['label']} (~{rate:g} rps), "
                f"di atas {_pct(target)}: antrean dan latensi naik tajam di atas titik ini; tambah replika/limit atau "
                f"turunkan beban yang direncanakan.",
            )
        )
    for service, r in result["resources"].items():
        if service == "generator":
            continue
        if service not in over and r["cpu_pct_max"] is not None and r["cpu_pct_max"] >= 0.9:
            out.append(
                (
                    "info",
                    f"CPU {service} sempat memuncak {_pct(r['cpu_pct_max'])} limit "
                    f"(rata-rata fase masih di bawah {_pct(target)}).",
                )
            )
        if r["mem_pct_max"] is not None and r["mem_pct_max"] >= 0.85:
            out.append(("warn", f"Memori {service} mencapai {_pct(r['mem_pct_max'])} limit."))
    stages = [(name, st) for name, st in result["stages"].items() if st["p95"] is not None]
    if stages:
        name, st = max(stages, key=lambda item: item[1]["p95"])
        e2e = s["e2e"]["p95"]
        share = f", ~{_pct(st['p95'] * 1000 / e2e)} dari p95 end-to-end" if e2e else ""
        out.append(("info", f"Tahap terlama: {name} (p95 {_secs(st['p95'] * 1000)}{share})."))
    for name, st in result["stages"].items():
        if st.get("gap_p95") is not None and st["gap_p95"] > 2:
            out.append(("warn", f"Jeda serah-terima ke {name} (outbox) p95 {st['gap_p95']:.1f} dtk: antrean menumpuk."))
    k6 = result.get("k6") or {}
    if k6.get("dropped_iterations"):
        out.append(
            (
                "warn",
                f"k6 melewatkan {k6['dropped_iterations']} iterasi (VU habis): laju yang tercapai di bawah rencana.",
            )
        )
    generator = result["resources"].get("generator", {})
    if generator.get("cpu_pct_max") is not None and generator["cpu_pct_max"] >= 0.8:
        out.append(("warn", "CPU generator k6 di atas 80% limit: hasil bisa meremehkan kapasitas sesungguhnya."))
    if s["in_flight"]:
        out.append(("warn", f"{s['in_flight']} request belum tuntas saat laporan dibuat (masih antre atau macet)."))
    if s.get("dead_letters"):
        out.append(("bad", f"{s['dead_letters']} pesan outbox menjadi dead letter."))
    if sent and counts["rejected"] / sent > 0.2:
        out.append(
            (
                "info",
                f"{_pct(counts['rejected'] / sent)} dokumen ditolak model/aturan: periksa apakah file uji mewakili.",
            )
        )
    if not s["db_available"]:
        out.append(("warn", "Database tidak terbaca: end-to-end, rasio tuntas, dan durasi per tahap tidak terukur."))
    capacity = result.get("capacity")
    if capacity:
        if capacity["sustainable_rate"] is not None:
            tail = (
                f"; mulai melanggar di {capacity['break_label']} ({', '.join(capacity['break_reasons'])})"
                if capacity["break_rate"] is not None
                else "; batas belum tercapai"
            )
            out.insert(0, ("info", f"Kapasitas berkelanjutan: {capacity['sustainable_rate']:g} rps{tail}."))
        else:
            out.insert(0, ("bad", "Step pertama pun sudah melanggar SLO: turunkan laju awal."))
        if capacity.get("latency_rate") != capacity["sustainable_rate"] and capacity.get("latency_rate") is not None:
            out.insert(
                1,
                (
                    "info",
                    f"Menurut latensi/error saja sistem masih lulus sampai {capacity['latency_rate']:g} rps, tapi "
                    f"di atas {capacity['sustainable_rate'] or 0:g} rps CPU sudah melewati target utilisasi: sisa "
                    f"ruang untuk lonjakan habis.",
                ),
            )
        if not capacity.get("cpu_measured"):
            out.append(("warn", "Utilisasi CPU tidak terukur: kapasitas hanya dinilai dari latensi dan error."))
    recovery = result.get("recovery")
    if recovery:
        if recovery["recovery_seconds"] is None:
            out.insert(0, ("bad", "Setelah lonjakan, sistem tidak kembali memenuhi SLO selama masa pulih."))
        elif recovery["degraded_during_spike"]:
            out.insert(0, ("info", f"Pulih {recovery['recovery_seconds']} detik setelah lonjakan berakhir."))
        else:
            out.insert(0, ("ok", "Lonjakan dilayani tanpa melanggar SLO."))
    drift = result.get("drift")
    if drift and drift["entry_p95_first"] and drift["entry_p95_last"]:
        growth = drift["entry_p95_last"] / drift["entry_p95_first"] - 1
        if growth > 0.25:
            out.append(("warn", f"p95 pintu masuk memburuk {_pct(growth)} dari awal ke akhir soak."))
    for service, r in result["resources"].items():
        slope, limit = r.get("mem_slope_mi_per_h"), r.get("mem_limit_mi")
        if result["profile"] == "soak" and slope and limit and slope > 0.1 * limit:
            out.append(("warn", f"Memori {service} naik ~{slope:.0f} MiB/jam: kemungkinan kebocoran."))
    if not out:
        out.append(("ok", "Tidak ada temuan: semua indikator dalam batas."))
    return out


def compute(
    meta: dict[str, Any],
    samples: dict[str, dict[str, Any]],
    db: dict[str, Any] | None,
    resources: list[dict[str, Any]],
    pods: dict[str, dict[str, Any]],
    k6: dict[str, Any] | None,
    now: float,
) -> dict[str, Any]:
    plan = meta["plan"]
    slo = {**SLO_DEFAULTS, **(meta.get("slo") or {})}
    start = float(meta.get("load_started_at") or meta.get("started_at") or now)
    end = float(meta.get("load_finished_at") or now)
    end = min(end, start + plan_seconds(plan) + 60) if plan else end
    outcomes = request_outcomes(samples, db, meta.get("pipeline_name_sequence"))
    counts = Counter(r["bucket"] for r in outcomes.values())
    sent = len(samples)
    states = Counter(r["state"] for r in outcomes.values())
    entry = [float(s.get("ms") or 0) for s in samples.values()]
    e2e = [r["e2e_ms"] for r in outcomes.values() if r.get("e2e_ms") is not None]
    db_available = db is not None or not any(
        s in STAGE_SERVICES for s in (meta.get("pipeline_name_sequence") or PIPELINE_NAMES)
    )
    eligible = states["done"] + states["failed"] + states["in_flight"]
    starts = sorted(float(s["t"]) for s in samples.values())
    done_at = sorted(r["done_at"] for r in outcomes.values() if r.get("done_at") is not None)

    stage_names = [
        "guardrails",
        *[s for s in (meta.get("pipeline_name_sequence") or PIPELINE_NAMES) if s in STAGE_SERVICES],
    ]
    if meta.get("pipeline_name_sequence") and "guardrails" not in meta["pipeline_name_sequence"]:
        stage_names.remove("guardrails")
    stages = {}
    for name in stage_names:
        durations = [r["stage_s"][name] for r in outcomes.values() if name in r["stage_s"]]
        gaps = [r["gap_s"][name] for r in outcomes.values() if name in r["gap_s"]]
        stages[name] = {
            "count": len(durations),
            "p50": percentile(durations, 0.5),
            "p95": percentile(durations, 0.95),
            "max": max(durations) if durations else None,
            "gap_p50": percentile(gaps, 0.5),
            "gap_p95": percentile(gaps, 0.95),
            "failed": sum(1 for r in outcomes.values() if r["state"] == "failed" and str(r["reason"]).startswith(name)),
        }

    summary = {
        "sent": sent,
        "counts": {k: counts.get(k, 0) for k in BUCKETS},
        "states": dict(states),
        "error_rate": (counts["5xx"] + counts["timeout"]) / sent if sent else None,
        "achieved_rate": (len(starts) - 1) / (starts[-1] - starts[0])
        if len(starts) > 1 and starts[-1] > starts[0]
        else None,
        "entry": trend(entry),
        "e2e": trend(e2e),
        "completed": states["done"],
        "rejected": states["rejected"],
        "failed": states["failed"],
        "in_flight": states["in_flight"],
        "completion_ratio": states["done"] / eligible if eligible and db_available else None,
        "completed_per_minute": len(done_at) / ((done_at[-1] - starts[0]) / 60)
        if done_at and starts and done_at[-1] > starts[0]
        else None,
        "db_available": db_available,
        "dead_letters": (db or {}).get("dead"),
    }
    win = windows(samples, outcomes, plan, start, max(end, start + 1))
    release = meta.get("release", "nilam-ocr-npwp")
    cpu = cpu_utilization(resources, meta.get("limits") or {}, release)
    phases = phase_stats(samples, outcomes, plan, start, cpu)
    res = resource_stats(resources, pods, meta.get("limits") or {}, release)
    result: dict[str, Any] = {
        "profile": meta["profile"],
        "summary": summary,
        "windows": win,
        "phases": phases,
        "stages": stages,
        "resources": res,
        "slo": evaluate_slo(summary, slo, res, phases),
        "slo_targets": slo,
        "k6": k6,
        "aborted_reason": meta.get("aborted_reason"),
        "capacity": stress_capacity(phases, slo) if meta["profile"] == "stress" else None,
        "recovery": spike_recovery(win, plan, slo) if meta["profile"] == "spike" else None,
        "drift": soak_drift(win, plan) if meta["profile"] == "soak" else None,
        "failures": Counter(r["reason"] for r in outcomes.values() if r["state"] == "failed").most_common(10),
        "rejections": Counter(r["reason"] for r in outcomes.values() if r["state"] == "rejected").most_common(10),
        "errors": Counter(
            f"HTTP {int(samples[rid].get('s') or 0)}: {samples[rid].get('e') or samples[rid].get('m') or '-'}"
            for rid, r in outcomes.items()
            if r["bucket"] in ("4xx", "5xx", "timeout")
        ).most_common(10),
    }
    result["verdict"] = verdict(result)
    result["findings"] = findings(result)
    return result


def verdict(result: dict[str, Any]) -> dict[str, str]:
    """{tone, title, body}: lulus / gagal / tidak dapat dinilai; stress melaporkan kapasitas, bukan lulus/gagal."""
    rows = result["slo"]
    failed = [r for r in rows if r["ok"] is False]
    unknown = [r for r in rows if r["ok"] is None]
    if result["summary"]["sent"] == 0:
        return {"tone": "bad", "title": "TIDAK ADA REQUEST", "body": "k6 tidak mengirim request apa pun."}
    if result["profile"] == "stress":
        capacity = result["capacity"] or {}
        if capacity.get("sustainable_rate") is None:
            return {
                "tone": "bad",
                "title": "JENUH SEJAK STEP PERTAMA",
                "body": "Laju awal sudah melanggar SLO; ulangi dengan laju awal lebih rendah.",
            }
        if capacity.get("break_rate") is None:
            return {
                "tone": "ok",
                "title": f"KAPASITAS >= {capacity['sustainable_rate']:g} RPS",
                "body": "Semua step memenuhi SLO; batas belum tercapai, naikkan laju puncak untuk menemukannya.",
            }
        return {
            "tone": "info",
            "title": f"KAPASITAS BERKELANJUTAN ~{capacity['sustainable_rate']:g} RPS",
            "body": (
                f"SLO mulai dilanggar di {capacity['break_label']}: {', '.join(capacity['break_reasons'])}. Kapasitas "
                f"dihitung dengan utilisasi CPU <= {result['slo_targets']['max_cpu_utilization'] * 100:.0f}% (ruang "
                f"untuk lonjakan)."
            ),
        }
    if failed:
        names = ", ".join(r["label"] for r in failed)
        return {
            "tone": "bad",
            "title": f"GAGAL: {len(failed)} dari {len(rows)} SLO",
            "body": f"Tidak terpenuhi: {names}.",
        }
    if unknown:
        names = ", ".join(r["label"] for r in unknown)
        return {"tone": "warn", "title": "BELUM DAPAT DINILAI PENUH", "body": f"Tidak terukur: {names}."}
    return {"tone": "ok", "title": "LULUS: semua SLO terpenuhi", "body": "Semua kriteria terpenuhi pada beban ini."}


MAX_BACKLOG_SECONDS = 300


def should_abort(
    windows_: list[dict[str, Any]],
    slo: dict[str, float],
    *,
    auto_abort: bool,
    now_offset: float,
    in_flight: int = 0,
    current_rate: float = 0.0,
) -> str | None:
    """Alasan menghentikan run, dari dua jendela penuh terakhir. Batas keamanan berlaku untuk semua profil (error
    > 50%); profil dengan auto_abort (stress) berhenti lebih awal: error >= 5x SLO (min. 10%), p95 pintu masuk
    >= 2x SLO di dua jendela berturut-turut, atau antrean belum tuntas setara lebih dari 5 menit kiriman (jenuh
    tanpa error: di endpoint -test kelebihan beban hanya terlihat sebagai 202 dan antrean yang terus bertambah)."""
    full = [w for w in windows_ if w["t"] + WINDOW_SECONDS <= now_offset and w["sent"] >= 3]
    if len(full) < 2:
        return None
    last = full[-2:]
    if all((w["error_rate"] or 0) > 0.5 for w in last):
        return "lebih dari 50% request 5xx/timeout di dua jendela berturut-turut (batas keamanan)"
    if not auto_abort:
        return None
    if current_rate > 0 and in_flight / current_rate > MAX_BACKLOG_SECONDS:
        return (
            f"{in_flight} request belum tuntas, setara lebih dari {MAX_BACKLOG_SECONDS // 60} menit kiriman pada "
            f"~{current_rate:.1f} rps: pipeline tidak mengimbangi laju"
        )
    limit = max(0.1, 5 * slo["error_rate"])
    if all((w["error_rate"] or 0) >= limit for w in last):
        return (
            f"error server >= {limit * 100:.0f}% di dua jendela berturut-turut "
            f"(laju ~{last[-1]['target_rate']:.1f} rps)"
        )
    if all((w["entry_p95"] or 0) >= 2 * slo["entry_p95_ms"] for w in last):
        return (
            f"p95 pintu masuk >= {2 * slo['entry_p95_ms'] / 1000:.0f} dtk di dua jendela berturut-turut "
            f"(laju ~{last[-1]['target_rate']:.1f} rps)"
        )
    return None
