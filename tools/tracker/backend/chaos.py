"""
Skenario gangguan ("shit happens") untuk uji kesiapan, di stack lokal (docker compose). Tracker
mengendalikan dua hal yang di produksi tidak bisa kita atur: container service (stop = SIGTERM seperti
rolling restart / eviction, kill = SIGKILL seperti OOM atau node hilang) dan cara Orkestrasi pusat (tracker)
menjawab callback. Setiap skenario mengirim dokumen sungguhan lewat orchestrator, membuat gangguannya pada
saat yang tepat, lalu memeriksa database, outbox, dan callback: tiap cek PASS, WARN (perilaku yang benar tapi
harus diketahui tim / Orkestrasi pusat), atau FAIL.

    GET  /api/chaos                  keadaan container + gangguan yang sedang berjalan
    POST /api/chaos                  {"service", "action": "stop" | "kill", "seconds", "grace_seconds"}
    POST /api/chaos/{service}/start  nyalakan lagi sekarang
    GET  /api/scenarios              daftar skenario, file uji, pengaturan, run terakhir tiap skenario
    POST /api/scenarios/run          {"scenarios": [...] | null (semua), "image"}: jalankan berurutan
    POST /api/scenarios/stop         hentikan run yang berjalan (container dinyalakan lagi, callback normal)
    GET  /api/scenarios/runs         run terakhir (terbaru dulu)
    GET  /api/scenarios/runs/{id}    laporan satu run: langkah, cek, request_id

Satu run pada satu waktu: simulasi callback berlaku global. Semua gangguan dibatalkan di akhir run.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException

# Tabel jobs tahap OCR sejak migrasi 0014; tahap lain `nilam_<tahap>_jobs`.
JOB_TABLES = {"OCR": "nilam_ocr_extraction_jobs"}

log = logging.getLogger("tracker.chaos")
router = APIRouter()
ctx: dict[str, Any] = {}

PREFIX = os.environ.get("TRACKER_CONTAINER_PREFIX", "nilam-ocr-")
SERVICES = ["orchestrator", "guardrails", "extraction", "structuring", "scoring", "postgres"]
# Sama dengan chart Helm: terminationGracePeriodSeconds 45 = drain job 30 dtk + preStop 5 dtk + cadangan.
STOP_GRACE = float(os.environ.get("TRACKER_STOP_GRACE_SECONDS", "45"))
# Harus sama dengan env service lokal (default service: 30, 300, 30). Lease 300 dtk membuat skenario crash
# menunggu > 5 menit; untuk latihan isi PIPELINE_JOB_LEASE_SECONDS=30 dan PIPELINE_STALE_JOB_INTERVAL_SECONDS=5
# di services/*/.env, lalu TRACKER_JOB_LEASE_SECONDS=30 dan TRACKER_STALE_JOB_INTERVAL_SECONDS=5 di sini.
DRAIN = float(os.environ.get("TRACKER_DRAIN_SECONDS", "30"))
LEASE = float(os.environ.get("TRACKER_JOB_LEASE_SECONDS", "300"))
STALE_INTERVAL = float(os.environ.get("TRACKER_STALE_JOB_INTERVAL_SECONDS", "30"))
END_TIMEOUT = 120.0
PREFIX_OF_STAGE = {"OCR": "extraction", "STRUCTURING": "structuring", "SCORING": "scoring"}
STAGES = ["OCR", "STRUCTURING", "SCORING"]
RUNS_KEY = "ocr:scenario:runs"


def configure(**values: Any) -> None:
    """Dipanggil app.py: redis, get_pool, run_request, set_callback_mode, callback_stats, emit, live_requests,
    service_urls, http, images_dir, target, wait_seconds, callback_slow_seconds, flaky_failures."""
    ctx.update(values)


# --- docker -------------------------------------------------------------------------


async def docker(*args: str, timeout: float = 120.0) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        proc.kill()
        return 124, f"docker {' '.join(args)}: timeout {timeout:.0f}s"
    return proc.returncode or 0, out.decode(errors="replace").strip()


def container(service: str) -> str:
    return f"{PREFIX}{service}"


async def container_state(service: str) -> dict[str, Any]:
    code, out = await docker(
        "inspect",
        "--format",
        "{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}",
        container(service),
    )
    if code != 0:
        return {"service": service, "container": container(service), "status": "missing", "detail": out[-200:]}
    status, _, health = out.partition("|")
    return {"service": service, "container": container(service), "status": status, "health": health or None}


async def chaos_event(text: str, **fields: Any) -> None:
    """Gangguan ke timeline setiap request yang sedang dipantau."""
    for request_id in ctx["live_requests"]():
        await ctx["emit"](request_id, "CHAOS", fields.get("action", "INFO").upper(), type="chaos", text=text, **fields)


async def stop(service: str, *, kill: bool = False, grace: float = STOP_GRACE) -> float:
    """SIGTERM (docker stop, menunggu sampai `grace` seperti terminationGracePeriodSeconds) atau SIGKILL.
    Mengembalikan lamanya container berhenti."""
    started = time.monotonic()
    if kill:
        await chaos_event(f"{service}: SIGKILL (OOM / node hilang), tanpa drain", action="kill", service=service)
        code, out = await docker("kill", container(service))
    else:
        await chaos_event(
            f"{service}: SIGTERM (rolling restart / eviction), grace {grace:.0f} dtk", action="stop", service=service
        )
        code, out = await docker("stop", "-t", str(int(grace)), container(service), timeout=grace + 30)
    if code != 0:
        raise RuntimeError(f"docker {'kill' if kill else 'stop'} {container(service)} gagal: {out[-300:]}")
    took = time.monotonic() - started
    await chaos_event(f"{service}: berhenti setelah {took:.1f} dtk", action="stopped", service=service)
    return took


async def start(service: str, *, wait: float = 90.0) -> bool:
    """docker start, lalu tunggu sampai sehat (/health, atau healthcheck Postgres)."""
    code, out = await docker("start", container(service))
    if code != 0:
        raise RuntimeError(f"docker start {container(service)} gagal: {out[-300:]}")
    await chaos_event(f"{service}: dinyalakan lagi", action="start", service=service)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if await healthy(service):
            await chaos_event(f"{service}: sehat lagi", action="up", service=service)
            return True
        await asyncio.sleep(1.0)
    return False


async def healthy(service: str) -> bool:
    if service == "postgres":
        return (await container_state(service)).get("health") == "healthy"
    url = ctx["service_urls"].get(service)
    if not url:
        return (await container_state(service)).get("status") == "running"
    try:
        r = await ctx["http"].get(f"{url}/health", timeout=3.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


disruptions: dict[str, dict[str, Any]] = {}


async def _disrupt(service: str, action: str, seconds: float, grace: float) -> None:
    try:
        await stop(service, kill=action == "kill", grace=grace)
        disruptions[service]["until"] = time.time() + seconds
        await asyncio.sleep(seconds)
    finally:
        try:
            await start(service)
        except RuntimeError as exc:
            log.error("%s", exc)
        disruptions.pop(service, None)


@router.get("/api/chaos")
async def chaos_status():
    states = await asyncio.gather(*(container_state(service) for service in SERVICES))
    return {
        "available": ctx["target"] == "local",
        "containers": states,
        "disruptions": [{k: v for k, v in d.items() if k != "task"} for d in disruptions.values()],
        "stop_grace_seconds": STOP_GRACE,
    }


@router.post("/api/chaos")
async def chaos_start(body: dict[str, Any]):
    _require_local()
    service = body.get("service")
    action = body.get("action", "stop")
    if service not in SERVICES:
        raise HTTPException(status_code=422, detail=f"service harus salah satu dari {SERVICES}")
    if action not in ("stop", "kill"):
        raise HTTPException(status_code=422, detail="action harus stop (SIGTERM) atau kill (SIGKILL)")
    if service in disruptions:
        raise HTTPException(status_code=409, detail=f"{service} sedang diganggu")
    seconds = float(body.get("seconds", 20))
    grace = float(body.get("grace_seconds", STOP_GRACE))
    disruptions[service] = {"service": service, "action": action, "seconds": seconds, "started": time.time()}
    disruptions[service]["task"] = asyncio.create_task(_disrupt(service, action, seconds, grace))
    return {"started": True}


@router.post("/api/chaos/{service}/start")
async def chaos_restart(service: str):
    _require_local()
    if service not in SERVICES:
        raise HTTPException(status_code=422, detail=f"service harus salah satu dari {SERVICES}")
    task = disruptions.get(service, {}).get("task")
    if task is not None:
        task.cancel()  # finally di _disrupt menyalakannya
        return {"started": True}
    return {"started": await start(service)}


def _require_local() -> None:
    if ctx["target"] != "local":
        raise HTTPException(
            status_code=409, detail="gangguan container hanya di mode lokal (docker compose); tracker sedang ke GKE"
        )


# --- skenario ------------------------------------------------------------------------


class Aborted(Exception):
    pass


@dataclass
class Scenario:
    id: str
    title: str
    simulates: str  # apa yang terjadi di produksi
    expect: str  # apa artinya "siap"
    run: Callable[["Run"], Awaitable[None]]
    needs_lease: bool = False


class Run:
    """Satu skenario yang sedang berjalan: langkah, cek, dan alat bantunya."""

    def __init__(self, run_id: str, scenario: Scenario, image: tuple[bytes, str, str]):
        self.id = run_id
        self.scenario = scenario
        self.image = image
        self.started = time.time()
        self.steps: list[dict[str, Any]] = []
        self.checks: list[dict[str, Any]] = []
        self.request_ids: list[str] = []
        self.status = "running"
        self.ended: float | None = None
        self.stopped_services: set[str] = set()
        self.cancelled = False

    def view(self) -> dict[str, Any]:
        return {
            "run_id": self.id,
            "scenario": self.scenario.id,
            "title": self.scenario.title,
            "status": self.status,
            "started_at": self.started,
            "ended_at": self.ended,
            "steps": self.steps,
            "checks": self.checks,
            "request_ids": self.request_ids,
        }

    async def save(self) -> None:
        await ctx["redis"].hset(RUNS_KEY, self.id, json.dumps(self.view()))

    def _alive(self) -> None:
        if self.cancelled:
            raise Aborted("dihentikan")

    async def step(self, text: str) -> None:
        self._alive()
        self.steps.append({"t": round(time.time() - self.started, 1), "text": text})
        log.info("scenario %s: %s", self.scenario.id, text)
        await self.save()

    async def check(self, label: str, ok: bool, detail: str = "", *, warn: bool = False) -> bool:
        """ok -> PASS; tidak ok -> FAIL, atau WARN kalau `warn` (perilaku yang benar tapi perlu diketahui)."""
        level = "pass" if ok else "warn" if warn else "fail"
        self.checks.append(
            {"label": label, "level": level, "detail": detail, "t": round(time.time() - self.started, 1)}
        )
        await self.save()
        return ok

    async def check_contract(self) -> None:
        """Setelah setiap skenario: body setiap callback yang tiba untuk request skenario ini sesuai kontrak
        (contract.py). Request tanpa callback (ditolak guardrails, jawaban sinkron saja) dilewati."""
        for request_id in self.request_ids:
            reports = (await ctx["callback_stats"](request_id))["contract"]
            if not reports:
                continue
            failed = sorted({rule for report in reports for rule in report["failed"]})
            warned = sorted({rule for report in reports for rule in report["warned"]})
            detail = f"{len(reports)} callback ({reports[-1]['format']}, {reports[-1]['kind']})"
            if failed:
                detail += f"; melanggar: {'; '.join(failed)}"
            elif warned:
                detail += f"; perlu diketahui: {'; '.join(warned)}"
            await self.check(
                f"Body callback {request_id} sesuai kontrak", not failed and not warned, detail, warn=not failed
            )

    async def note(self, label: str, detail: str, *, level: str = "info") -> None:
        """Temuan yang bukan lulus / gagal: info, atau warn untuk hal yang harus diketahui Orkestrasi pusat."""
        self.checks.append(
            {"label": label, "level": level, "detail": detail, "t": round(time.time() - self.started, 1)}
        )
        await self.save()

    # -- aksi --

    async def callback(self, mode: str) -> None:
        await ctx["set_callback_mode"](mode)
        await self.step(f"Orkestrasi pusat (tracker) menjawab callback: {mode}")

    async def submit(self, **options: Any) -> dict[str, Any]:
        self._alive()
        content, filename, content_type = self.image
        out = await ctx["run_request"](
            content, filename, content_type, origin=f"scenario:{self.scenario.id}", **options
        )
        if out["request_id"] not in self.request_ids:
            self.request_ids.append(out["request_id"])
        await self.save()
        return out

    def submit_later(self, **options: Any) -> asyncio.Task[dict[str, Any]]:
        """Kirim di latar (jawaban orchestrator bisa 15 dtk); request_id-nya dibuat di sini supaya langsung
        bisa dipantau."""
        request_id = options.pop("request_id", None) or f"REQ_{uuid.uuid4().hex[:12]}"
        self.request_ids.append(request_id)
        return asyncio.create_task(self.submit(request_id=request_id, resend=False, **options))

    async def stop(self, service: str, *, kill: bool = False, grace: float = STOP_GRACE) -> float:
        self._alive()
        self.stopped_services.add(service)
        await self.step(
            f"{'SIGKILL' if kill else 'SIGTERM'} ke {service}" + ("" if kill else f" (grace {grace:.0f} dtk)")
        )
        took = await stop(service, kill=kill, grace=grace)
        await self.step(f"{service} berhenti setelah {took:.1f} dtk")
        return took

    async def start(self, service: str) -> None:
        await self.step(f"nyalakan {service}")
        up = await start(service)
        self.stopped_services.discard(service)
        if not up:
            raise RuntimeError(f"{service} tidak sehat lagi dalam 90 dtk")
        await self.step(f"{service} sehat lagi")

    async def sleep(self, seconds: float, why: str) -> None:
        await self.step(f"tunggu {seconds:.0f} dtk: {why}")
        end = time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0:
            self._alive()
            await asyncio.sleep(min(0.5, left))

    async def wait(self, what: str, predicate: Callable[[], Awaitable[Any]], timeout: float) -> Any:
        """Tunggu sampai `predicate()` truthy; nilainya dikembalikan, atau None kalau waktu habis."""
        await self.step(f"tunggu: {what} (maks. {timeout:.0f} dtk)")
        deadline = time.monotonic() + timeout
        while True:
            self._alive()
            value = await predicate()
            if value:
                return value
            if time.monotonic() >= deadline:
                await self.step(f"waktu habis menunggu: {what}")
                return None
            await asyncio.sleep(0.5)

    # -- membaca keadaan --

    async def job(self, stage: str, request_id: str) -> dict[str, Any] | None:
        table = JOB_TABLES.get(stage, f"nilam_{stage.lower()}_jobs")
        rows = await self._query(
            f"SELECT status, attempts, error_message, updated_at FROM {table} WHERE request_id = $1",
            request_id,
        )
        return dict(rows[0]) if rows else None

    async def outbox(self, request_id: str) -> list[dict[str, Any]] | None:
        rows = await self._query(
            "SELECT id, stage, kind, payload, attempts, failed_at, last_error FROM nilam_pipeline_outbox "
            "WHERE request_id = $1 ORDER BY id",
            request_id,
        )
        if rows is None:
            return None
        out = []
        for row in rows:
            payload = row["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            out.append({**dict(row), "payload": payload})
        return out

    async def _query(self, sql: str, *args: Any) -> list[Any] | None:
        pool = await ctx["get_pool"]()
        if pool is None:
            raise RuntimeError(
                "database tidak terbaca (TRACKER_DATABASE_URL); skenario butuh stack docker-compose.db.yml"
            )
        try:
            async with pool.acquire() as conn:
                return list(await conn.fetch(sql, *args))
        except Exception as exc:  # noqa: BLE001 - Postgres sedang dimatikan skenario
            log.info("query gagal (%s): %s", type(exc).__name__, exc)
            return None

    async def final(self, request_id: str) -> dict[str, Any]:
        return await ctx["callback_stats"](request_id)

    async def wait_final(self, request_id: str, timeout: float = END_TIMEOUT) -> dict[str, Any] | None:
        async def accepted():
            stats = await self.final(request_id)
            return stats if stats["accepted"] else None

        return await self.wait(f"callback akhir {request_id} diterima Orkestrasi", accepted, timeout)

    async def last_stage_done(self, request_id: str) -> bool:
        job = await self.job("SCORING", request_id)
        return bool(job and job["status"] == "DONE")

    async def expect_outbox_drained(self, request_id: str, timeout: float = 30.0) -> None:
        async def drained():
            rows = await self.outbox(request_id)
            return rows == []

        ok = await self.wait("outbox request ini kosong", drained, timeout)
        rows = await self.outbox(request_id) or []
        await self.check(
            "Outbox kosong: semua handoff dan callback terkirim, tidak ada dead letter",
            bool(ok),
            "" if ok else "; ".join(_row_text(row) for row in rows),
        )

    async def expect_no_stuck_jobs(self, request_id: str) -> None:
        stuck = []
        for stage in STAGES:
            job = await self.job(stage, request_id)
            if job and job["status"] == "PROCESSING":
                stuck.append(f"{stage} PROCESSING (attempt {job['attempts']})")
        await self.check("Tidak ada job yang tertinggal PROCESSING", not stuck, ", ".join(stuck))

    async def service_outbox(self, stage: str) -> dict[str, Any] | None:
        prefix = PREFIX_OF_STAGE[stage]
        try:
            r = await ctx["http"].get(f"{ctx['service_urls'][prefix]}/v1/{prefix}/outbox", timeout=5.0)
            return r.json().get("data") if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

    async def release(self, stage: str, request_id: str) -> int | None:
        """Operasional: POST /v1/<tahap>/outbox/release?request_id=... di service pemilik baris."""
        prefix = PREFIX_OF_STAGE[stage]
        try:
            r = await ctx["http"].post(
                f"{ctx['service_urls'][prefix]}/v1/{prefix}/outbox/release", params={"request_id": request_id}
            )
            data = r.json().get("data") if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None
        return int(data.get("released", 0)) if isinstance(data, dict) else None

    async def cleanup(self) -> None:
        await ctx["set_callback_mode"]("ok")
        for service in sorted(self.stopped_services):
            try:
                await start(service)
            except RuntimeError as exc:
                log.error("cleanup: %s", exc)
        self.stopped_services.clear()


def _row_text(row: dict[str, Any]) -> str:
    what = f"handoff → {row['payload'].get('next_stage')}" if row["kind"] == "handoff" else "callback"
    state = "DEAD" if row["failed_at"] else "pending"
    return f"#{row['id']} {row['stage']} {what} {state}, attempt {row['attempts']}: {row['last_error'] or '-'}"


def _answer(out: dict[str, Any]) -> str:
    if out.get("unreachable"):
        return f"orchestrator tidak terjangkau: {out.get('reason')}"
    parts = [f"HTTP {out.get('http_status')}", f"job_status={out.get('job_status')}"]
    if out.get("errors"):
        parts.append(f"errors={out['errors']}")
    parts.append(f"pipeline_last_stage={out.get('pipeline_last_stage')}")
    parts.append(f"{(out.get('elapsed_ms') or 0) / 1000:.1f} dtk")
    if out.get("message") and out.get("http_status") != 200:
        parts.append(f"message={out['message']}")
    return ", ".join(parts)


async def _completed_answer(run: Run, out: dict[str, Any]) -> None:
    """Syarat awal: dokumen uji lolos guardrails dan aturan structuring, supaya cek berikutnya bermakna."""
    if out.get("http_status") == 400:
        raise RuntimeError(
            f"dokumen uji ditolak ({out.get('message')}); pilih foto NPWP yang lolos guardrails dan aturan structuring"
        )


# -- skenario-skenario --


async def s_baseline(run: Run) -> None:
    await run.callback("ok")
    out = await run.submit()
    await _completed_answer(run, out)
    rid = out["request_id"]
    await run.check(
        "Orchestrator menjawab 200 completed dengan data",
        out.get("http_status") == 200 and out.get("job_status") == "completed",
        _answer(out),
        warn=out.get("http_status") == 202,
    )
    final = await run.wait_final(rid, 60)
    await run.check("Callback akhir diterima Orkestrasi", bool(final), json.dumps(final))
    stats = await run.final(rid)
    await run.check("Callback akhir tiba tepat satu kali", stats["arrived"] == 1, f"tiba {stats['arrived']}x")
    await run.expect_outbox_drained(rid)
    await run.expect_no_stuck_jobs(rid)


async def s_callback_down(run: Run) -> None:
    outage = 20.0
    await run.callback("down")
    down_at = time.monotonic()
    out = await run.submit()
    await _completed_answer(run, out)
    rid = out["request_id"]
    await run.check(
        "Jawaban POST tidak bergantung pada callback (tetap 200 / 202)",
        out.get("http_status") in (200, 202),
        _answer(out),
    )
    done = await run.wait("SCORING DONE di database", lambda: run.last_stage_done(rid), 60)
    await run.check("Pipeline tetap selesai walau Orkestrasi pusat mati", bool(done))

    async def retrying():
        rows = await run.outbox(rid) or []
        return [row for row in rows if row["kind"] == "callback" and row["attempts"] >= 2 and not row["failed_at"]]

    rows = await run.wait("baris callback diulang (attempt ≥ 2)", retrying, 30)
    dead = [row for row in await run.outbox(rid) or [] if row["failed_at"]]
    await run.check(
        "Callback yang gagal (503) tertahan di outbox dan diulang dengan backoff, bukan hilang atau dead letter",
        bool(rows) and not dead,
        "; ".join(_row_text(row) for row in (rows or []) + dead),
    )
    await run.sleep(max(outage - (time.monotonic() - down_at), 0), f"Orkestrasi pusat mati {outage:.0f} dtk")
    await run.callback("ok")
    up_at = time.monotonic()
    final = await run.wait_final(rid, 90)
    await run.check(
        "Setelah Orkestrasi hidup, callback akhir terkirim sendiri tanpa tindakan manual",
        bool(final),
        f"{time.monotonic() - up_at:.1f} dtk setelah pulih (backoff 0,5 dtk ×2, maks 5 menit)" if final else "",
    )
    stats = await run.final(rid)
    await run.check("Callback akhir diterima tepat satu kali", stats["accepted"] == 1, f"diterima {stats['accepted']}x")
    await run.expect_outbox_drained(rid)


async def s_callback_flaky(run: Run) -> None:
    failures = ctx["flaky_failures"]
    await run.callback("flaky")
    out = await run.submit()
    await _completed_answer(run, out)
    rid = out["request_id"]
    final = await run.wait_final(rid, 60)
    stats = await run.final(rid)
    await run.check(
        f"Orkestrasi yang {failures}x menjawab 503 lalu pulih: callback akhir tetap sampai",
        bool(final),
        f"tiba {stats['arrived']}x, diterima {stats['accepted']}x",
    )
    await run.check(
        "Diterima tepat satu kali (tidak ada duplikat)", stats["accepted"] == 1, f"diterima {stats['accepted']}x"
    )
    await run.expect_outbox_drained(rid)


async def s_callback_not_ready(run: Run) -> None:
    failures = ctx["not_ready_failures"]
    await run.callback("not_ready")
    out = await run.submit()
    await _completed_answer(run, out)
    rid = out["request_id"]
    final = await run.wait_final(rid, 60)
    stats = await run.final(rid)
    await run.check(
        f"Pusat yang {failures}x menjawab 409 RESULT_NOT_READY: callback akhir dikirim ulang dan sampai",
        bool(final),
        f"tiba {stats['arrived']}x, diterima {stats['accepted']}x",
    )
    await run.check(
        f"Dikirim ulang tepat {failures}x lalu diterima (bukan langsung dead letter seperti 4xx lain)",
        stats["arrived"] == failures + 1,
        f"tiba {stats['arrived']}x",
    )
    await run.check(
        "Diterima tepat satu kali (tidak ada duplikat)", stats["accepted"] == 1, f"diterima {stats['accepted']}x"
    )
    await run.expect_outbox_drained(rid)


async def s_callback_slow(run: Run) -> None:
    slow = ctx["callback_slow_seconds"]
    await run.callback("slow")
    out = await run.submit()
    await _completed_answer(run, out)
    rid = out["request_id"]

    async def twice():
        stats = await run.final(rid)
        return stats if stats["accepted"] >= 2 else None

    stats = await run.wait("callback akhir diterima ≥ 2 kali", twice, 60)
    await run.check(
        f"Orkestrasi menjawab setelah {slow:.0f} dtk (> ORCHESTRATION_TIMEOUT_SECONDS 10): relay menganggapnya gagal "
        "dan mengirim ulang",
        bool(stats),
        json.dumps(stats) if stats else "",
    )
    if stats:
        await run.note(
            "Orkestrasi pusat menerima callback akhir yang SAMA lebih dari sekali",
            f"{stats['accepted']}x untuk {rid}. Pengiriman outbox at-least-once: endpoint callback "
            "pusat wajib idempoten per request_id (upsert, bukan insert), dan harus menjawab < 10 dtk.",
            level="warn",
        )
    await run.callback("ok")
    await run.expect_outbox_drained(rid, 60)


async def s_callback_rejected(run: Run) -> None:
    await run.callback("unauthorized")
    out = await run.submit()
    await _completed_answer(run, out)
    rid = out["request_id"]
    await run.wait("SCORING DONE di database", lambda: run.last_stage_done(rid), 60)

    async def dead_rows():
        rows = await run.outbox(rid) or []
        return [row for row in rows if row["kind"] == "callback" and row["failed_at"]]

    dead = await run.wait("baris callback jadi dead letter", dead_rows, 30) or []
    await run.check(
        "401 (X-Callback-Key salah) dari Orkestrasi: relay berhenti setelah 1 attempt, baris jadi dead letter",
        bool(dead) and all(row["attempts"] == 1 for row in dead),
        "; ".join(_row_text(row) for row in dead),
    )
    owners = sorted({row["stage"] for row in dead})
    for stage in owners:
        stats = await run.service_outbox(stage)
        await run.check(
            f"Dead letter terlihat di GET /v1/{PREFIX_OF_STAGE[stage]}/outbox (bahan alert)",
            bool(stats and stats.get("dead_letters", 0) >= 1),
            json.dumps(stats),
        )
    await run.note(
        "Dead letter TIDAK terkirim sendiri walau Orkestrasi sudah diperbaiki",
        "Request ini selesai di pipeline, tapi Orkestrasi pusat tidak pernah tahu sampai seseorang melepasnya: "
        "POST /v1/<tahap>/outbox/release?request_id=... Pasang alert di log 'outbox ... dead letters' / metrik "
        "dead_letters, dan pastikan ORCHESTRATION_CALLBACK_KEY sama dengan milik pusat sebelum go-live.",
        level="warn",
    )
    await run.callback("ok")
    released = 0
    for stage in owners:
        released += await run.release(stage, rid) or 0
    await run.check("POST /v1/<tahap>/outbox/release melepas dead letter-nya", released >= 1, f"{released} baris")
    final = await run.wait_final(rid, 30)
    await run.check("Setelah dilepas, callback akhir diterima", bool(final))
    await run.expect_outbox_drained(rid)


async def s_stage_down(run: Run) -> None:
    outage = 25.0
    await run.callback("ok")
    task = run.submit_later(slow_seconds=8)
    rid = run.request_ids[-1]

    async def ocr_processing():
        job = await run.job("OCR", rid)
        return job if job and job["status"] == "PROCESSING" else None

    await run.wait("job OCR PROCESSING", ocr_processing, 30)
    await run.stop("structuring", grace=5)
    down_at = time.monotonic()

    async def handoff_retrying():
        rows = await run.outbox(rid) or []
        return [row for row in rows if row["kind"] == "handoff" and row["attempts"] >= 2]

    rows = await run.wait("handoff OCR → STRUCTURING diulang", handoff_retrying, 40)
    await run.check(
        "Handoff ke structuring yang mati tertahan di outbox dan diulang dengan backoff",
        bool(rows) and not any(row["failed_at"] for row in rows or []),
        "; ".join(_row_text(row) for row in rows or []),
    )
    await run.sleep(max(outage - (time.monotonic() - down_at), 0), f"structuring mati {outage:.0f} dtk")
    await run.start("structuring")
    final = await run.wait_final(rid, 90)
    await run.check("Setelah structuring hidup, handoff terkirim dan pipeline selesai", bool(final), json.dumps(final))
    out = await task
    await run.check(
        "Orkestrasi pusat mendapat 202 (bukan 5xx) selama tahap tengah mati, lalu hasil lewat callback",
        out.get("http_status") in (200, 202),
        _answer(out),
    )
    await run.expect_outbox_drained(rid)
    await run.expect_no_stuck_jobs(rid)


async def s_sigterm_drained(run: Run) -> None:
    await run.callback("ok")
    task = run.submit_later(slow_seconds=10)
    rid = run.request_ids[-1]

    async def ocr_processing():
        job = await run.job("OCR", rid)
        return job if job and job["status"] == "PROCESSING" else None

    await run.wait("job OCR PROCESSING", ocr_processing, 30)
    took = await run.stop("extraction")
    job = await run.job("OCR", rid)
    await run.check(
        "Job yang sedang jalan diselesaikan dulu (drain) sebelum proses berhenti",
        bool(job and job["status"] == "DONE"),
        f"OCR {job and job['status']}, berhenti setelah {took:.1f} dtk",
    )
    await run.start("extraction")
    final = await run.wait_final(rid, 90)
    await run.check("Pipeline selesai; callback akhir diterima", bool(final), json.dumps(final))
    await task
    await run.expect_outbox_drained(rid)
    await run.expect_no_stuck_jobs(rid)


async def s_sigterm_interrupted(run: Run) -> None:
    await run.callback("ok")
    slow = int(DRAIN + 15)
    task = run.submit_later(slow_seconds=slow)
    rid = run.request_ids[-1]

    async def ocr_processing():
        job = await run.job("OCR", rid)
        return job if job and job["status"] == "PROCESSING" else None

    await run.wait("job OCR PROCESSING", ocr_processing, 30)
    took = await run.stop("extraction")
    await run.check(
        f"Proses berhenti di dalam grace ({STOP_GRACE:.0f} dtk), tidak perlu SIGKILL",
        took < STOP_GRACE - 1,
        f"{took:.1f} dtk",
    )
    job = await run.job("OCR", rid)
    await run.check(
        f"Job yang lebih lama dari drain ({DRAIN:.0f} dtk) ditandai FAILED, tidak tertinggal PROCESSING",
        bool(job and job["status"] == "FAILED" and "shutdown" in (job["error_message"] or "")),
        f"OCR {job and job['status']}: {job and job['error_message']}",
    )
    await run.start("extraction")
    final = await run.wait_final(rid, 60)
    await run.check(
        "Orkestrasi pusat menerima FAILED untuk request ini",
        bool(final and final["statuses"][:1] == ["FAILED"]),
        json.dumps(final),
    )
    await task
    await run.step("Orkestrasi pusat mengirim ulang request_id yang sama (tanpa simulasi lambat)")
    again = await run.submit(request_id=rid)
    await run.check(
        "Kiriman ulang request_id yang sama menjalankan lagi pipeline yang FAILED dan selesai",
        again.get("http_status") in (200, 202),
        _answer(again),
    )
    stats = await run.wait(
        "callback akhir kedua (DONE)",
        lambda: _final_with(run, rid, "DONE"),
        90,
    )
    await run.check("Callback DONE menyusul setelah FAILED", bool(stats), json.dumps(stats))
    await run.note(
        "Rolling restart di tengah job panjang = FAILED + kirim ulang",
        "Job yang tidak selesai dalam PIPELINE_DRAIN_TIMEOUT_SECONDS (30) dihentikan dan dilaporkan FAILED "
        "'interrupted by a service shutdown'. Orkestrasi pusat harus menganggap FAILED itu boleh dicoba lagi "
        "dengan request_id yang sama; deploy sebaiknya di luar jam sibuk.",
        level="warn",
    )
    await run.expect_outbox_drained(rid)


async def _final_with(run: Run, request_id: str, status: str) -> dict[str, Any] | None:
    stats = await run.final(request_id)
    return stats if status in stats["statuses"] else None


async def _crash(run: Run, source: str) -> None:
    await run.callback("ok")
    task = run.submit_later(slow_seconds=20, source=source)
    rid = run.request_ids[-1]

    async def ocr_processing():
        job = await run.job("OCR", rid)
        return job if job and job["status"] == "PROCESSING" else None

    await run.wait("job OCR PROCESSING", ocr_processing, 30)
    await run.stop("extraction", kill=True)
    await run.start("extraction")
    job = await run.job("OCR", rid)
    await run.note(
        "Setelah SIGKILL, job tertinggal tanpa pemilik",
        f"OCR {job and job['status']} (attempt {job and job['attempts']})",
    )
    await task

    async def reclaimed():
        job = await run.job("OCR", rid)
        return job if job and job["attempts"] >= 2 else None

    budget = LEASE + STALE_INTERVAL + 30
    t0 = time.monotonic()
    job = await run.wait(f"reaper mengambil alih job (lease {LEASE:.0f} dtk)", reclaimed, budget)
    await run.check(
        "Reaper mengambil alih job yatim setelah lease habis (PIPELINE_JOB_LEASE_SECONDS)",
        bool(job),
        f"{time.monotonic() - t0:.0f} dtk setelah proses mati" if job else f"belum dalam {budget:.0f} dtk",
    )
    if source == "file_url":
        final = await run.wait("callback DONE", lambda: _final_with(run, rid, "DONE"), 90)
        await run.check(
            "Dokumen file_url diunduh ulang dan pipeline selesai tanpa campur tangan", bool(final), json.dumps(final)
        )
    else:
        final = await run.wait("callback FAILED", lambda: _final_with(run, rid, "FAILED"), 60)
        job = await run.job("OCR", rid)
        await run.check(
            "Upload inline tidak bisa dipulihkan: job FAILED dengan pesan minta kirim ulang, bukan PROCESSING terus",
            bool(final and job and job["status"] == "FAILED" and "inline" in (job["error_message"] or "")),
            f"OCR {job and job['status']}: {job and job['error_message']}",
        )
        await run.note(
            "Upload inline yang prosesnya mati = FAILED",
            "Isi file ikut hilang bersama proses. Orkestrasi pusat sebaiknya mengirim file_url (MinIO), yang bisa "
            "diunduh lagi saat job dijalankan ulang; kalau tetap multipart, FAILED ini harus dikirim ulang.",
            level="warn",
        )
    await run.expect_outbox_drained(rid)
    await run.expect_no_stuck_jobs(rid)


async def s_crash_inline(run: Run) -> None:
    await _crash(run, "upload")


async def s_crash_file_url(run: Run) -> None:
    await _crash(run, "file_url")


async def s_postgres_down(run: Run) -> None:
    outage = 20.0
    await run.callback("ok")
    task = run.submit_later(slow_seconds=8, source="file_url")
    rid = run.request_ids[-1]

    async def ocr_processing():
        job = await run.job("OCR", rid)
        return job if job and job["status"] == "PROCESSING" else None

    await run.wait("job OCR PROCESSING", ocr_processing, 30)
    await run.stop("postgres", grace=10)
    down_at = time.monotonic()
    await run.step("request baru masuk selama database mati")
    during = await run.submit()
    await run.check(
        "Request baru selama database mati ditolak jelas (5xx), bukan diterima lalu hilang",
        bool(during.get("http_status") and during["http_status"] >= 500),
        _answer(during),
    )
    await run.sleep(max(outage - (time.monotonic() - down_at), 0), f"database mati {outage:.0f} dtk")
    await run.start("postgres")
    out = await task
    await run.note("Jawaban POST request yang sedang jalan", _answer(out))

    async def settled():
        job = await run.job("OCR", rid)
        return job if job and job["status"] != "PROCESSING" else None

    budget = LEASE + STALE_INTERVAL + 60
    job = await run.wait("job OCR tidak lagi PROCESSING (reaper, lease)", settled, budget)
    await run.check(
        "Setelah database pulih, job yang terputus tidak tertinggal PROCESSING",
        bool(job),
        f"OCR {job['status']}, attempt {job['attempts']}" if job else f"masih PROCESSING setelah {budget:.0f} dtk",
    )
    final = await run.wait_final(rid, 90)
    stats = await run.final(rid)
    await run.check("Orkestrasi pusat menerima keadaan akhir", bool(final), json.dumps(stats))
    if len(set(stats["statuses"])) > 1:
        await run.note(
            "Orkestrasi pusat menerima dua keadaan akhir berbeda untuk request_id yang sama",
            f"berurutan: {', '.join(stats['statuses'])}. Saat database putus, tahap melaporkan FAILED langsung, lalu "
            "reaper menjalankan job yang sama lagi dan bisa melaporkan DONE. Pusat harus menerima DONE yang datang "
            "setelah FAILED (keadaan terakhir menang).",
            level="warn",
        )
    await run.expect_outbox_drained(rid, 60)
    await run.expect_no_stuck_jobs(rid)


async def s_resend(run: Run) -> None:
    await run.callback("ok")
    first = await run.submit()
    await _completed_answer(run, first)
    rid = first["request_id"]
    await run.wait_final(rid, 60)
    await run.expect_outbox_drained(rid)
    before = {stage: await run.job(stage, rid) for stage in STAGES}
    cb_before = await run.final(rid)
    await run.step("Orkestrasi pusat mengirim ulang request_id yang sama (mis. timeout di sisinya)")
    again = await run.submit(request_id=rid)
    await run.check(
        "Kiriman ulang dijawab dengan hasil yang tersimpan",
        again.get("http_status") == first.get("http_status") and again.get("data") == first.get("data"),
        f"pertama: {_answer(first)} · ulang: {_answer(again)}",
    )
    await run.sleep(3, "beri waktu kalau ada yang ikut jalan lagi")
    after = {stage: await run.job(stage, rid) for stage in STAGES}
    rerun = [
        stage
        for stage in STAGES
        if before[stage] and after[stage] and after[stage]["attempts"] != before[stage]["attempts"]
    ]
    await run.check("Pipeline tidak dijalankan dua kali (attempt job tidak berubah)", not rerun, ", ".join(rerun))
    cb_after = await run.final(rid)
    await run.check(
        "Tidak ada callback akhir baru", cb_after["arrived"] == cb_before["arrived"], f"{cb_before} → {cb_after}"
    )


async def s_concurrent(run: Run) -> None:
    await run.callback("ok")
    rid = f"REQ_{uuid.uuid4().hex[:12]}"
    await run.step("dua POST dengan request_id sama pada saat yang sama")
    first, second = await asyncio.gather(
        run.submit(request_id=rid, resend=False), run.submit(request_id=rid, resend=False)
    )
    await run.note("Jawaban", f"1: {_answer(first)} · 2: {_answer(second)}")
    await run.check(
        "Keduanya dijawab tanpa 5xx",
        all((out.get("http_status") or 500) < 500 for out in (first, second)),
    )
    await run.wait_final(rid, 60)
    await run.sleep(3, "beri waktu kalau ada callback ganda")
    job = await run.job("OCR", rid)
    await run.check(
        "Job OCR hanya diklaim sekali", bool(job and job["attempts"] == 1), f"attempt {job and job['attempts']}"
    )
    stats = await run.final(rid)
    await run.check("Callback akhir hanya satu", stats["arrived"] == 1, f"tiba {stats['arrived']}x")
    await run.expect_outbox_drained(rid)


async def s_entry_down(run: Run) -> None:
    await run.callback("ok")
    for service, stage in (("guardrails", "guardrails"), ("extraction", "extraction")):
        await run.stop(service, grace=5)
        out = await run.submit()
        await run.check(
            f"{service} mati: orchestrator menjawab 5xx dengan pipeline_last_stage={stage}, tidak ada yang dimulai",
            bool(out.get("http_status") and out["http_status"] >= 500 and out.get("pipeline_last_stage") == stage),
            _answer(out),
        )
        job = await run.job("OCR", out["request_id"])
        await run.check(f"{service} mati: tidak ada job OCR yang tertinggal", job is None, json.dumps(job, default=str))
        await run.start(service)
    await run.note(
        "5xx di pintu masuk = tidak ada yang berjalan",
        "Orkestrasi pusat boleh mengulang request itu (request_id sama) setelah jeda.",
    )


SCENARIOS = [
    Scenario(
        "baseline",
        "Jalur normal",
        "Tidak ada gangguan: patokan untuk skenario lain.",
        "200 completed, satu callback akhir, outbox kosong, tidak ada job PROCESSING.",
        s_baseline,
    ),
    Scenario(
        "callback-down",
        "Orkestrasi pusat mati 20 dtk (503)",
        "Endpoint callback pusat down / deploy / overload saat hasil kita siap.",
        "Pipeline tetap selesai, callback tertahan di outbox dengan backoff, lalu terkirim sendiri saat pusat hidup.",
        s_callback_down,
    ),
    Scenario(
        "callback-flaky",
        "Orkestrasi pusat tersendat (503 lalu 200)",
        "Pusat kadang gagal sesaat (restart pod, koneksi DB-nya penuh).",
        "Callback akhir sampai tanpa tindakan manual, tepat satu kali.",
        s_callback_flaky,
    ),
    Scenario(
        "callback-not-ready",
        "Callback tiba sebelum pusat mencatat 202-nya (409)",
        "Race: pipeline selesai sangat cepat setelah 202, pusat belum menyimpan request-nya.",
        "409 RESULT_NOT_READY dikenali dan dikirim ulang tiap 1,5 dtk; callback akhir diterima tepat sekali.",
        s_callback_not_ready,
    ),
    Scenario(
        "callback-slow",
        "Orkestrasi pusat lambat menjawab callback",
        "Pusat memproses callback lebih lama dari ORCHESTRATION_TIMEOUT_SECONDS (10 dtk).",
        "Relay mengirim ulang; pusat menerima duplikat dan harus idempoten.",
        s_callback_slow,
    ),
    Scenario(
        "callback-rejected",
        "Orkestrasi pusat menolak callback (401)",
        "X-Callback-Key salah / path salah / validasi pusat menolak body kita.",
        "Dead letter terlihat di /outbox, lalu terkirim setelah dilepas manual.",
        s_callback_rejected,
    ),
    Scenario(
        "stage-down",
        "Structuring mati 25 dtk di tengah request",
        "Pod tahap berikutnya crash-loop / di-deploy saat handoff dikirim.",
        "Handoff tertahan dan diulang, pipeline selesai setelah structuring hidup, pusat mendapat 202 lalu callback.",
        s_stage_down,
    ),
    Scenario(
        "sigterm-drained",
        "Rolling restart extraction di tengah job pendek",
        "kubectl rollout / eviction saat OCR sedang bekerja.",
        "Job diselesaikan dalam drain, handoff tetap terkirim, pipeline selesai.",
        s_sigterm_drained,
    ),
    Scenario(
        "sigterm-interrupted",
        "Rolling restart extraction di tengah job panjang",
        "Restart saat job butuh lebih lama dari PIPELINE_DRAIN_TIMEOUT_SECONDS.",
        "Job FAILED 'interrupted', pusat menerima FAILED, kiriman ulang request_id yang sama selesai.",
        s_sigterm_interrupted,
    ),
    Scenario(
        "crash-inline",
        "Extraction mati mendadak (SIGKILL), dokumen upload",
        "OOMKilled / node hilang saat OCR bekerja, dokumen dikirim multipart.",
        "Reaper mengambil alih setelah lease; job FAILED dengan pesan minta kirim ulang.",
        s_crash_inline,
        needs_lease=True,
    ),
    Scenario(
        "crash-file-url",
        "Extraction mati mendadak (SIGKILL), dokumen file_url",
        "Sama, dokumen dikirim sebagai file_url seperti contoh cURL pusat.",
        "Reaper mengambil alih setelah lease, mengunduh ulang, pipeline selesai sendiri.",
        s_crash_file_url,
        needs_lease=True,
    ),
    Scenario(
        "postgres-down",
        "Database mati 20 dtk",
        "Cloud SQL failover / maintenance / koneksi putus saat pipeline jalan.",
        "Request baru ditolak 5xx; request yang terputus tidak tertinggal PROCESSING dan pusat menerima keadaan akhir.",
        s_postgres_down,
        needs_lease=True,
    ),
    Scenario(
        "resend",
        "Pusat mengirim ulang request_id yang sudah selesai",
        "Pusat timeout di sisinya lalu retry dengan request_id yang sama.",
        "Dijawab dari hasil tersimpan, pipeline tidak jalan dua kali, tidak ada callback baru.",
        s_resend,
    ),
    Scenario(
        "concurrent",
        "Dua POST bersamaan dengan request_id sama",
        "Retry pusat yang tumpang tindih dengan request aslinya.",
        "Job hanya diklaim sekali, satu callback akhir.",
        s_concurrent,
    ),
    Scenario(
        "entry-down",
        "Guardrails / extraction mati saat request masuk",
        "Service hilir tidak tersedia ketika pusat memanggil kita.",
        "5xx yang jelas dengan pipeline_last_stage, tidak ada job setengah jalan.",
        s_entry_down,
    ),
]
BY_ID = {scenario.id: scenario for scenario in SCENARIOS}

current: dict[str, Any] = {"task": None, "run": None}


def _images() -> list[Path]:
    folders = [ctx["images_dir"], ctx["images_dir"].parent / "assets"]
    return sorted(
        path
        for folder in folders
        if folder.is_dir()
        for path in folder.iterdir()
        if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".pdf")
    )


def _load_image(name: str | None) -> tuple[bytes, str, str]:
    images = _images()
    chosen = next((path for path in images if path.name == name), None) if name else (images[0] if images else None)
    if chosen is None:
        raise HTTPException(status_code=422, detail=f"file uji tidak ada: {name or '(folder images kosong)'}")
    content_type = {".png": "image/png", ".pdf": "application/pdf"}.get(chosen.suffix.lower(), "image/jpeg")
    return chosen.read_bytes(), chosen.name, content_type


async def _run_all(ids: list[str], image: tuple[bytes, str, str]) -> None:
    for scenario_id in ids:
        scenario = BY_ID[scenario_id]
        run = Run(f"{int(time.time())}-{scenario_id}", scenario, image)
        current["run"] = run
        await run.save()
        try:
            await scenario.run(run)
            await run.check_contract()
            levels = {check["level"] for check in run.checks}
            run.status = "fail" if "fail" in levels else "warn" if "warn" in levels else "pass"
        except Aborted:
            run.status = "stopped"
        except Exception as exc:  # noqa: BLE001 - laporkan, lanjut ke skenario berikutnya
            log.exception("scenario %s crashed", scenario_id)
            run.status = "error"
            run.steps.append({"t": round(time.time() - run.started, 1), "text": f"ERROR: {type(exc).__name__}: {exc}"})
        finally:
            await run.cleanup()
            run.ended = time.time()
            await run.save()
        if run.status == "stopped":
            break
    current["run"] = None


@router.get("/api/scenarios")
async def list_scenarios():
    runs = [json.loads(value) for value in (await ctx["redis"].hgetall(RUNS_KEY)).values()]
    runs.sort(key=lambda run: run["started_at"], reverse=True)
    last = {}
    for run in runs:
        last.setdefault(run["scenario"], run)
    return {
        "available": ctx["target"] == "local",
        "running": current["run"].id if current["run"] else None,
        "images": [path.name for path in _images()],
        "settings": {
            "stop_grace_seconds": STOP_GRACE,
            "drain_seconds": DRAIN,
            "job_lease_seconds": LEASE,
            "stale_job_interval_seconds": STALE_INTERVAL,
            "callback_slow_seconds": ctx["callback_slow_seconds"],
            "wait_seconds": ctx["wait_seconds"],
        },
        "scenarios": [
            {
                "id": scenario.id,
                "title": scenario.title,
                "simulates": scenario.simulates,
                "expect": scenario.expect,
                "needs_lease": scenario.needs_lease,
                "last": last.get(scenario.id),
            }
            for scenario in SCENARIOS
        ],
    }


@router.post("/api/scenarios/run")
async def run_scenarios(body: dict[str, Any]):
    _require_local()
    task = current["task"]
    if task is not None and not task.done():
        raise HTTPException(status_code=409, detail="masih ada skenario yang berjalan")
    ids = body.get("scenarios") or [scenario.id for scenario in SCENARIOS]
    unknown = [scenario_id for scenario_id in ids if scenario_id not in BY_ID]
    if unknown:
        raise HTTPException(status_code=422, detail=f"skenario tidak dikenal: {unknown}")
    image = _load_image(body.get("image"))
    current["task"] = asyncio.create_task(_run_all(ids, image))
    return {"started": ids}


@router.post("/api/scenarios/stop")
async def stop_scenarios():
    run = current["run"]
    if run is None:
        return {"stopped": False}
    run.cancelled = True
    return {"stopped": True}


@router.get("/api/scenarios/runs")
async def list_runs():
    runs = [json.loads(value) for value in (await ctx["redis"].hgetall(RUNS_KEY)).values()]
    runs.sort(key=lambda run: run["started_at"], reverse=True)
    return runs[:100]


@router.get("/api/scenarios/runs/{run_id}")
async def get_run(run_id: str):
    value = await ctx["redis"].hget(RUNS_KEY, run_id)
    if value is None:
        raise HTTPException(status_code=404, detail="run tidak ada")
    return json.loads(value)


@router.delete("/api/scenarios/runs")
async def clear_runs():
    await ctx["redis"].delete(RUNS_KEY)
    return {"cleared": True}
