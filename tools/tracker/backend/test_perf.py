"""Performance test GKE: rencana beban, klasifikasi hasil per request, SLO, penghentian otomatis, dan laporan PDF
(perf_stats.py, perf_report.py), dari data sintetis yang bentuknya sama dengan yang dikumpulkan perftest.py.
Jalankan: python -m pytest tools/tracker/backend"""

import perf_report
import perf_stats as ps
import pytest

T0 = 1_760_000_000.0
RELEASE = "nilam-ocr-npwp"
LIMITS = {
    "orchestrator": {"replicas": 2, "cpu_m": 1000.0, "mem_mi": 1024.0},
    "guardrails": {"replicas": 1, "cpu_m": 2000.0, "mem_mi": 2048.0},
    "extraction": {"replicas": 1, "cpu_m": 1000.0, "mem_mi": 512.0},
    "structuring": {"replicas": 1, "cpu_m": 1000.0, "mem_mi": 512.0},
    "scoring": {"replicas": 1, "cpu_m": 1000.0, "mem_mi": 512.0},
    "generator": {"replicas": 1, "cpu_m": 1000.0, "mem_mi": 1024.0},
}


def plan(profile: str, **params):
    return ps.build_plan(profile, params, max_rate=10, max_seconds=7200)


def meta(profile: str, the_plan, **extra):
    return {
        "run_id": "pt000001",
        "profile": profile,
        "plan": the_plan,
        "slo": dict(ps.SLO_DEFAULTS),
        "pipeline_name_sequence": None,
        "load_started_at": T0,
        "load_finished_at": T0 + ps.plan_seconds(the_plan),
        "created_at": T0 - 5,
        "limits": LIMITS,
        "release": RELEASE,
        "images": ["npwp1.jpg"],
        "target": "http://nilam-ocr-npwp-orchestrator:8034/v1/extract-ocr-test",
        "context": "gke_x_asia-southeast2_cluster",
        "namespace": "nilam-ocr-npwp",
        "wait_seconds": 15,
        "vus": [20, 30],
        "versions": {"orchestrator": "orch:abc123"},
        "status": "finished",
        **extra,
    }


def synth(the_plan, *, slow_after: float | None = None, error_after: float | None = None, rate_scale: float = 1.0):
    """Request per rencana; di atas `slow_after` rps pipeline melambat (202, e2e panjang), di atas `error_after` 5xx."""
    samples, stages, guard = {}, {s: {} for s in ps.STAGE_SERVICES}, {}
    n = 0
    t = 0.0
    total = ps.plan_seconds(the_plan)
    while t < total:
        rate = ps.target_rate(the_plan, t) * rate_scale
        if rate <= 0:
            t += 1
            continue
        rid = f"TEST_pt000001_{n:05d}"
        sent = T0 + t
        slow = slow_after is not None and rate > slow_after
        if error_after is not None and rate > error_after:
            samples[rid] = {"r": rid, "s": 503, "t": sent, "ms": 120, "e": "SERVICE_UNAVAILABLE", "m": "busy"}
        else:
            status = 202 if slow else 200
            samples[rid] = {"r": rid, "s": status, "t": sent, "ms": 15100 if slow else 4200, "e": None, "m": None}
            guard[rid] = [True, sent + 0.8]
            start = sent + 1.0
            for stage, dur in (("extraction", 2.5 if not slow else 25), ("structuring", 0.3), ("scoring", 0.4)):
                stages[stage][rid] = ["DONE", start, start + dur, None]
                start = start + dur + 0.2
        n += 1
        t += 1 / rate
    db = {"stages": stages, "guard": guard, "dead": 0, "fetched_at": T0 + total + 60}
    resources = [
        {
            "ts": T0 + i * 15,
            "pods": {
                f"{RELEASE}-orchestrator-a-1": [120.0, 150.0],
                f"{RELEASE}-extraction-b-1": [300.0 + i, 200.0 + i],
                f"{RELEASE}-structuring-c-1": [40.0, 110.0],
                f"{RELEASE}-scoring-d-1": [50.0, 150.0],
                f"{RELEASE}-guardrails-e-1": [400.0, 600.0],
                "nilam-perf-pt000001-x": [80.0, 90.0],
                "cloudsql-relay-ocr-nilam": [0.0, 1.0],
            },
        }
        for i in range(int(total / 15) + 1)
    ]
    return samples, db, resources


def test_plans_and_k6_stages():
    stress = plan("stress", start_rate=0.5, peak_rate=2, step_rate=0.5, step_seconds=60)
    holds = [p for p in stress if p["kind"] == "hold"]
    assert [p["end_rate"] for p in holds] == [0.5, 1.0, 1.5, 2.0]
    start, stages = ps.k6_stages(stress)
    assert start == 0 and stages[:2] == [[5, 10], [5, 60]]
    spike = plan("spike")
    assert [p["name"] for p in spike] == ["warmup", "baseline", "spike_up", "spike", "spike_down", "recovery"]
    assert ps.plan_peak(spike) == 3.0
    assert ps.plan_seconds(plan("smoke")) == 60
    with pytest.raises(ps.PlanError):
        plan("load", base_rate=50)
    with pytest.raises(ps.PlanError):
        plan("stress", start_rate=0.1, peak_rate=10, step_rate=0.1)
    with pytest.raises(ps.PlanError):
        plan("soak", duration_seconds=99999)


def test_request_outcomes_classify_rejections_and_failures():
    samples = {
        "a": {"s": 200, "t": T0, "ms": 3000},
        "b": {"s": 400, "t": T0, "ms": 500, "e": "DOWNSTREAM_VALIDATION_ERROR", "m": "bukan NPWP"},
        "c": {"s": 202, "t": T0, "ms": 15000},
        "d": {"s": 202, "t": T0, "ms": 15000},
        "e": {"s": 503, "t": T0, "ms": 100},
        "f": {"s": 0, "t": T0, "ms": 45000},
    }
    db = {
        "stages": {
            "extraction": {
                "a": ["DONE", T0 + 1, T0 + 2, None],
                "c": ["DONE", T0 + 1, T0 + 20, None],
                "d": ["FAILED", T0 + 1, T0 + 3, "Internal error in extraction stage"],
            },
            "structuring": {
                "a": ["DONE", T0 + 2.1, T0 + 2.4, None],
                "c": ["FAILED", T0 + 20.5, T0 + 21, "nomor NPWP tidak valid"],
            },
            "scoring": {"a": ["DONE", T0 + 2.5, T0 + 2.9, None]},
        },
        "guard": {"a": [True, T0 + 0.9]},
    }
    out = ps.request_outcomes(samples, db, None)
    assert out["a"]["state"] == "done" and out["a"]["e2e_ms"] == pytest.approx(2900)
    assert out["a"]["gap_s"]["structuring"] == pytest.approx(0.1)
    assert out["b"]["state"] == "rejected"
    assert out["c"]["state"] == "rejected" and "NPWP" in out["c"]["reason"]
    assert out["d"]["state"] == "failed" and out["d"]["reason"].startswith("extraction")
    assert out["e"]["state"] == "not_started"
    assert out["f"]["state"] == "in_flight"
    assert ps.request_outcomes(samples, None, None)["a"]["state"] == "unknown"
    # Sequence ["guardrails"]: jawaban 200 sudah final, tanpa database.
    assert ps.request_outcomes(samples, None, ["guardrails"])["a"]["state"] == "done"


def test_load_passes_all_slo():
    p = plan("load", base_rate=1, ramp_seconds=20, duration_seconds=240)
    samples, db, resources = synth(p)
    result = ps.compute(meta("load", p), samples, db, resources, {}, None, T0 + 400)
    assert result["verdict"]["tone"] == "ok", result["slo"]
    assert all(row["ok"] for row in result["slo"])
    assert result["summary"]["completion_ratio"] == 1.0
    assert result["stages"]["extraction"]["p95"] == pytest.approx(2.5)
    assert result["resources"]["extraction"]["cpu_pct_max"] > 0.3
    assert result["resources"]["generator"]["cpu_max_m"] == 80.0
    assert any("Tahap terlama: extraction" in text for _, text in result["findings"])


def test_stress_finds_capacity_and_breaking_step():
    p = plan("stress", start_rate=0.5, peak_rate=2.5, step_rate=0.5, step_seconds=60)
    samples, db, resources = synth(p, slow_after=1.6)
    result = ps.compute(meta("stress", p), samples, db, resources, {}, None, T0 + 600)
    assert result["capacity"]["sustainable_rate"] == 1.5
    assert result["capacity"]["break_rate"] == 2.0
    assert "1.5" in result["verdict"]["title"]
    assert any("202" in text for _, text in result["findings"])


def test_failed_slo_and_restarts():
    p = plan("load", base_rate=1, ramp_seconds=10, duration_seconds=120)
    samples, db, resources = synth(p, error_after=0.5)
    pods = {f"{RELEASE}-extraction-b-1": {"service": "extraction", "restarts": 1, "reasons": ["OOMKilled"]}}
    result = ps.compute(meta("load", p), samples, db, resources, pods, None, T0 + 300)
    failed = {row["key"] for row in result["slo"] if row["ok"] is False}
    assert {"error_rate", "max_restarts"} <= failed
    assert result["verdict"]["tone"] == "bad"
    assert any("OOMKilled" in text for _, text in result["findings"])


def test_without_database_is_inconclusive():
    p = plan("smoke")
    samples, _, resources = synth(p)
    result = ps.compute(meta("smoke", p), samples, None, resources, {}, None, T0 + 120)
    assert result["verdict"]["tone"] == "warn"
    assert {row["key"] for row in result["slo"] if row["ok"] is None} == {"e2e_p95_ms", "min_completion"}


def test_spike_recovery():
    p = plan("spike", base_rate=0.5, peak_rate=3, baseline_seconds=60, spike_seconds=60, recovery_seconds=120)
    samples, db, resources = synth(p, slow_after=2)
    result = ps.compute(meta("spike", p), samples, db, resources, {}, None, T0 + 400)
    assert result["recovery"]["degraded_during_spike"] is True
    assert result["recovery"]["recovery_seconds"] is not None


def test_should_abort():
    p = plan("stress", start_rate=1, peak_rate=3, step_rate=1, step_seconds=60)
    samples, db, _ = synth(p, error_after=1.5)
    outcomes = ps.request_outcomes(samples, db, None)
    win = ps.windows(samples, outcomes, p, T0, T0 + ps.plan_seconds(p))
    slo = dict(ps.SLO_DEFAULTS)
    assert ps.should_abort(win, slo, auto_abort=True, now_offset=ps.plan_seconds(p)) is not None
    # Tanpa auto_abort hanya batas keamanan (error > 50% dua jendela) yang menghentikan.
    early = [w for w in win if w["t"] < 60]
    assert ps.should_abort(early, slo, auto_abort=False, now_offset=60) is None
    assert ps.should_abort([], slo, auto_abort=True, now_offset=10, in_flight=900, current_rate=1) is None
    full = [
        {"t": 0, "sent": 30, "error_rate": 0, "entry_p95": 1000},
        {"t": 30, "sent": 30, "error_rate": 0, "entry_p95": 1000},
    ]
    assert "belum tuntas" in ps.should_abort(full, slo, auto_abort=True, now_offset=90, in_flight=700, current_rate=2)


@pytest.mark.parametrize("profile", ["smoke", "load", "stress", "spike", "soak"])
def test_report_renders_every_profile(profile):
    params = {"soak": {"duration_seconds": 240}, "stress": {"peak_rate": 1.5, "step_seconds": 60}}.get(profile, {})
    p = plan(profile, **params)
    samples, db, resources = synth(p, slow_after=1.2)
    m = meta(profile, p)
    result = ps.compute(m, samples, db, resources, {}, {"iterations": len(samples), "dropped_iterations": 0}, T0 + 9999)
    result["baseline"] = {"run_id": "pt000000", "created_at": T0 - 3600, "summary": result["summary"], "versions": {}}
    pdf = perf_report.render(m, result)
    assert pdf.startswith(b"%PDF") and len(pdf) > 5000


def test_report_renders_empty_and_running_run():
    p = plan("smoke")
    m = meta("smoke", p, status="running", load_finished_at=None)
    result = ps.compute(m, {}, None, [], {}, None, T0 + 5)
    assert result["verdict"]["title"] == "TIDAK ADA REQUEST"
    assert perf_report.render(m, result).startswith(b"%PDF")


def test_a_chosen_file_that_is_gone_is_named_instead_of_silently_dropped():
    import loadtest
    from fastapi import HTTPException

    assert loadtest.chosen_images(["npwp1.jpg", "npwp1.jpg"], ["npwp1.jpg", "npwp2.jpg"]) == ["npwp1.jpg"]
    assert loadtest.chosen_images(None, ["npwp1.jpg"]) == []
    with pytest.raises(HTTPException) as raised:
        loadtest.chosen_images(["npwp1.jpg", "PN2503876I.pdf"], ["npwp1.jpg"])
    assert raised.value.status_code == 422
    assert "file uji sudah tidak ada: PN2503876I.pdf" in raised.value.detail


def test_the_configmap_size_counts_the_files_as_base64(tmp_path):
    import perftest

    image = tmp_path / "doc.pdf"
    image.write_bytes(b"x" * 300)

    assert perftest.configmap_size([image]) == 400 + perftest.SCRIPT.stat().st_size


def test_the_test_files_configmap_is_created_not_applied(monkeypatch):
    import asyncio

    import perftest

    calls = []

    async def kubectl(*args, **kwargs):
        calls.append(args)
        return 0, "", ""

    monkeypatch.setattr(perftest, "kubectl", kubectl)
    asyncio.run(perftest.apply({"kind": "ConfigMap"}, verb="create"))
    asyncio.run(perftest.apply({"kind": "Job"}))

    assert [call[0] for call in calls] == ["create", "apply"]


def cpu_tracking_load(the_plan, millicores_per_rps: float = 400.0):
    """Resource samples where the extraction pod's CPU follows the planned rate (other services stay light)."""
    samples = []
    for t in range(0, ps.plan_seconds(the_plan), 15):
        rate = ps.target_rate(the_plan, t)
        samples.append(
            {
                "ts": T0 + t,
                "pods": {
                    f"{RELEASE}-extraction-b-1": [rate * millicores_per_rps, 200.0],
                    f"{RELEASE}-orchestrator-a-1": [100.0, 150.0],
                    f"{RELEASE}-guardrails-e-1": [300.0, 600.0],
                    "nilam-perf-pt000001-x": [900.0, 90.0],  # the generator never counts
                },
            }
        )
    return samples


def test_stress_capacity_keeps_cpu_under_the_target():
    """Queueing theory: latency still passes at 2.5 rps, but extraction runs above 75% from 2 rps, so the planned
    capacity is the last step under the target."""
    p = plan("stress", start_rate=0.5, peak_rate=2.5, step_rate=0.5, step_seconds=60)
    samples, db, _ = synth(p)
    result = ps.compute(meta("stress", p), samples, db, cpu_tracking_load(p), {}, None, T0 + 900)
    capacity = result["capacity"]
    assert capacity["latency_rate"] == 2.5
    assert capacity["sustainable_rate"] == 1.5
    assert capacity["break_rate"] == 2.0
    assert capacity["break_reasons"] == ["CPU extraction 80%"]
    steps = {row["label"]: row for row in result["phases"] if row["kind"] == "hold"}
    assert steps["Step 3: 1.5 rps"]["cpu_busiest"] == "extraction"
    assert steps["Step 3: 1.5 rps"]["cpu_util_max"] == pytest.approx(0.6)
    assert "generator" in steps["Step 3: 1.5 rps"]["cpu_util"]  # measured, but never the busiest
    texts = [text for _, text in result["findings"]]
    assert any("CPU extraction rata-rata 80.0% limit di Step 4" in text for text in texts)
    assert any("Menurut latensi/error saja" in text for text in texts)


def test_cpu_slo_fails_a_load_run_above_the_target():
    p = plan("load", base_rate=2, ramp_seconds=20, duration_seconds=120)
    samples, db, _ = synth(p)
    result = ps.compute(meta("load", p), samples, db, cpu_tracking_load(p, 450), {}, None, T0 + 400)
    row = next(r for r in result["slo"] if r["key"] == "max_cpu_utilization")
    assert row["ok"] is False
    assert row["actual"] == pytest.approx(0.9)
    assert row["note"].startswith("extraction")
    assert result["verdict"]["tone"] == "bad"


def test_cpu_slo_passes_under_the_target_and_is_unknown_without_data():
    p = plan("load", base_rate=1, ramp_seconds=20, duration_seconds=120)
    samples, db, resources = synth(p)
    passing = ps.compute(meta("load", p), samples, db, resources, {}, None, T0 + 400)
    assert next(r for r in passing["slo"] if r["key"] == "max_cpu_utilization")["ok"] is True
    unmeasured = ps.compute(meta("load", p), samples, db, [], {}, None, T0 + 400)
    assert next(r for r in unmeasured["slo"] if r["key"] == "max_cpu_utilization")["ok"] is None


def test_report_shows_the_cpu_target():
    p = plan("stress", start_rate=0.5, peak_rate=2.5, step_rate=0.5, step_seconds=60)
    samples, db, _ = synth(p)
    m = meta("stress", p)
    result = ps.compute(m, samples, db, cpu_tracking_load(p), {}, None, T0 + 900)
    pdf = perf_report.render(m, result)
    assert pdf.startswith(b"%PDF") and len(pdf) > 5000
