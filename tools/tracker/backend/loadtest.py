"""Load testing dari tracker: jalankan k6 di Docker, kumpulkan sampel yang dilaporkan k6 dan
callback tahap untuk request berprefiks LT_, lalu hitung campuran 200/202/4xx/5xx, latensi
pintu masuk, dan waktu end-to-end sampai callback SCORING DONE.

Kunci Redis:
  ocr:loadtests                 hash run_id -> meta (json)
  ocr:lt:<run>:samples          list sampel dari k6 (json), urut kedatangan
  ocr:lt:<run>:status           hash "200" | "202" | "4xx" | "5xx" | "timeout" -> jumlah
  ocr:lt:<run>:cb               hash "<STAGE>:<STATUS>" -> jumlah callback yang tiba
  ocr:lt:<run>:done             hash request_id -> ts callback SCORING DONE diterima
  ocr:lt:<run>:failed           hash request_id -> "<STAGE>: <pesan>"
"""

import asyncio
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

log = logging.getLogger("tracker.loadtest")
router = APIRouter(prefix="/api/loadtest")

HERE = Path(__file__).resolve().parent
LT_DIR = Path(os.environ.get("LOAD_TESTER_DIR") or HERE.parent.parent / "load-tester")
K6_IMAGE = os.environ.get("K6_IMAGE", "grafana/k6:latest")
K6_NETWORK = os.environ.get("K6_NETWORK", "ocr_default")
K6_TARGET = os.environ.get("K6_TARGET", "http://guardrails:8031")
K6_TRACKER = os.environ.get("K6_TRACKER") or f"http://host.docker.internal:{os.environ.get('PORT', '8090')}"
K6_API_KEY = os.environ.get("K6_API_KEY", "")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".pdf")
MAX_RATE = 50.0
MAX_DURATION = 1800

ctx: dict[str, Any] = {}
watchers: dict[str, asyncio.Task[None]] = {}


def configure(*, redis: Any, get_pool: Any, wait_seconds: float, tables: dict[str, str]) -> None:
    ctx.update(redis=redis, get_pool=get_pool, wait_seconds=wait_seconds, tables=tables)


def _key(run: str, suffix: str) -> str:
    return f"ocr:lt:{run}:{suffix}"


def _container(run: str) -> str:
    return f"nilam-lt-{run}"


def list_images() -> list[str]:
    folder = LT_DIR / "images"
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


async def _docker(*args: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "docker", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    out, _ = await proc.communicate()
    return proc.returncode or 0, out.decode("utf-8", "replace").strip()


async def _meta(run: str) -> dict[str, Any] | None:
    raw = await ctx["redis"].hget("ocr:loadtests", run)
    return json.loads(raw) if raw else None


async def _save_meta(meta: dict[str, Any]) -> None:
    await ctx["redis"].hset("ocr:loadtests", meta["run_id"], json.dumps(meta))


# --- lifecycle container k6 ------------------------------------------------------


async def _start_container(meta: dict[str, Any]) -> str:
    name = _container(meta["run_id"])
    await _docker("rm", "-f", name)
    cmd = [
        "run",
        "-d",
        "--name",
        name,
        "--network",
        K6_NETWORK,
        "--add-host",
        "host.docker.internal:host-gateway",
        "-v",
        f"{LT_DIR / 'k6'}:/scripts:ro",
        "-v",
        f"{LT_DIR / 'images'}:/images:ro",
        "-v",
        f"{LT_DIR / 'out'}:/out",
        "-e",
        f"RUN_ID={meta['run_id']}",
        "-e",
        f"TARGET={K6_TARGET}",
        "-e",
        f"TRACKER={K6_TRACKER}",
        "-e",
        f"RATE={meta['rate']}",
        "-e",
        f"DURATION={meta['duration_seconds']}s",
        "-e",
        f"MODE={meta['mode']}",
        "-e",
        f"IMAGES={','.join(meta['images'])}",
        "-e",
        f"WAIT_SECONDS={ctx['wait_seconds']}",
        "-e",
        f"API_KEY={K6_API_KEY}",
        K6_IMAGE,
        "run",
        "--quiet",
        "/scripts/extract-ocr.js",
    ]
    code, out = await _docker(*cmd)
    if code != 0:
        raise HTTPException(status_code=502, detail=f"docker run gagal: {out[-600:]}")
    return out.splitlines()[-1][:12]


async def _watch(run: str) -> None:
    """Tunggu container k6 selesai, simpan ekor log-nya, lalu buang container."""
    name = _container(run)
    code, out = await _docker("wait", name)
    exit_code = int(out.strip().splitlines()[-1]) if code == 0 and out.strip() else -1
    _, logs = await _docker("logs", "--tail", "80", name)
    await _docker("rm", "-f", name)
    meta = await _meta(run)
    if meta is None:
        return
    meta.update(
        status="finished" if exit_code == 0 else "failed",
        exit_code=exit_code,
        finished_at=time.time(),
        log_tail=logs[-6000:],
    )
    await _save_meta(meta)
    log.info("load test %s selesai (exit %s)", run, exit_code)


def _spawn_watcher(run: str) -> None:
    task = watchers.get(run)
    if task is not None and not task.done():
        return
    watchers[run] = asyncio.create_task(_watch(run))


async def reconcile() -> None:
    """Saat backend start: run yang masih 'running' di Redis dicocokkan dengan container yang ada."""
    for raw in (await ctx["redis"].hgetall("ocr:loadtests")).values():
        meta = json.loads(raw)
        if meta.get("status") != "running":
            continue
        code, _ = await _docker("inspect", "--format", "{{.State.Status}}", _container(meta["run_id"]))
        if code == 0:
            _spawn_watcher(meta["run_id"])
        else:
            meta.update(
                status="unknown", finished_at=time.time(), log_tail="container tidak ditemukan saat backend start"
            )
            await _save_meta(meta)


# --- callback dari pipeline untuk request LT_ --------------------------------------


async def record_callback(body: dict[str, Any], *, accepted: bool) -> None:
    request_id = str(body.get("request_id", ""))
    parts = request_id.split("_", 2)
    if len(parts) < 3:
        return
    run = parts[1]
    stage, status = str(body.get("stage", "")), str(body.get("status", ""))
    r = ctx["redis"]
    await r.hincrby(_key(run, "cb"), f"{stage}:{status}", 1)
    if not accepted:
        return
    if stage == "SCORING" and status == "DONE":
        await r.hset(_key(run, "done"), request_id, time.time())
    elif status == "FAILED":
        await r.hset(_key(run, "failed"), request_id, f"{stage}: {body.get('error_message') or '-'}")


# --- statistik ----------------------------------------------------------------------


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def _trend(values: list[float]) -> dict[str, float | None]:
    values = sorted(values)
    return {
        "count": len(values),
        "avg": sum(values) / len(values) if values else None,
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "max": values[-1] if values else None,
    }


def _k6_summary(run: str) -> dict[str, Any] | None:
    path = LT_DIR / "out" / f"{run}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    metrics = data.get("metrics", {})

    def value(name: str, field: str = "count") -> Any:
        return (metrics.get(name) or {}).get("values", {}).get(field)

    return {
        "iterations": value("iterations"),
        "dropped_iterations": value("dropped_iterations") or 0,
        "vus_max": value("vus_max", "value"),
        "http_req_failed_rate": value("http_req_failed{name:extract-ocr}", "rate"),
        "extract_p95_ms": value("http_req_duration{name:extract-ocr}", "p(95)"),
    }


async def stats(run: str) -> dict[str, Any]:
    r = ctx["redis"]
    meta = await _meta(run)
    if meta is None:
        raise HTTPException(status_code=404, detail="run tidak ditemukan")
    samples = [json.loads(s) for s in await r.lrange(_key(run, "samples"), 0, -1)]
    counts = {k: int(v) for k, v in (await r.hgetall(_key(run, "status"))).items()}
    done = await r.hgetall(_key(run, "done"))
    failed = await r.hgetall(_key(run, "failed"))
    callbacks = {k: int(v) for k, v in (await r.hgetall(_key(run, "cb"))).items()}

    submitted = {s["request_id"]: float(s["started_at"]) for s in samples}
    starts = sorted(submitted.values())
    e2e = [(float(ts) - submitted[rid]) * 1000 for rid, ts in done.items() if rid in submitted]
    done_ts = sorted(float(ts) for ts in done.values())

    achieved_rate = None
    if len(starts) > 1 and starts[-1] > starts[0]:
        achieved_rate = (len(starts) - 1) / (starts[-1] - starts[0])
    completed_per_minute = None
    if starts and done_ts and done_ts[-1] > starts[0]:
        completed_per_minute = len(done_ts) / ((done_ts[-1] - starts[0]) / 60)

    sent = len(samples)
    return {
        **meta,
        "sent": sent,
        "counts": {k: counts.get(k, 0) for k in ("200", "202", "4xx", "5xx", "timeout")},
        "extract": _trend([float(s["elapsed_ms"]) for s in samples]),
        "achieved_rate": achieved_rate,
        "completed": len(done),
        "failed": len(failed),
        "in_flight": max(sent - len(e2e) - len([rid for rid in failed if rid in submitted]), 0),
        "completed_per_minute": completed_per_minute,
        "e2e": _trend(e2e),
        "callbacks": callbacks,
        "failures": [{"request_id": k, "reason": v} for k, v in list(failed.items())[:20]],
        "last_errors": [
            {"request_id": s["request_id"], "status": s["status"], "message": s.get("message") or s.get("errors")}
            for s in samples[-200:]
            if s["status"] not in (200, 202)
        ][-10:],
        "k6": _k6_summary(run),
    }


# --- endpoint ----------------------------------------------------------------------


@router.get("/config")
async def config() -> dict[str, Any]:
    code, out = await _docker("image", "inspect", "--format", "{{.Id}}", K6_IMAGE)
    return {
        "images": list_images(),
        "k6_image": K6_IMAGE,
        "k6_image_ready": code == 0,
        "network": K6_NETWORK,
        "target": K6_TARGET,
        "tracker": K6_TRACKER,
        "wait_seconds": ctx["wait_seconds"],
        "max_rate": MAX_RATE,
        "max_duration": MAX_DURATION,
        "load_tester_dir": str(LT_DIR),
    }


@router.get("")
async def list_runs() -> list[dict[str, Any]]:
    r = ctx["redis"]
    rows = [json.loads(v) for v in (await r.hgetall("ocr:loadtests")).values()]
    rows.sort(key=lambda m: m.get("started_at", 0), reverse=True)
    for meta in rows:
        run = meta["run_id"]
        counts = {k: int(v) for k, v in (await r.hgetall(_key(run, "status"))).items()}
        meta["sent"] = sum(counts.values())
        meta["counts"] = {k: counts.get(k, 0) for k in ("200", "202", "4xx", "5xx", "timeout")}
        meta["completed"] = await r.hlen(_key(run, "done"))
    return rows


@router.post("")
async def start(request: Request) -> dict[str, Any]:
    body = await request.json()
    try:
        rate = float(body.get("rate", 1))
        duration = int(body.get("duration_seconds", 60))
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="rate dan duration_seconds harus angka")
    mode = body.get("mode", "constant")
    if not 0 < rate <= MAX_RATE:
        raise HTTPException(status_code=422, detail=f"rate harus di antara 0 dan {MAX_RATE:g} rps")
    if not 1 <= duration <= MAX_DURATION:
        raise HTTPException(status_code=422, detail=f"duration_seconds harus 1..{MAX_DURATION}")
    if mode not in ("constant", "ramp"):
        raise HTTPException(status_code=422, detail="mode harus constant atau ramp")
    available = list_images()
    images = [name for name in (body.get("images") or available) if name in available]
    if not images:
        raise HTTPException(status_code=422, detail=f"tidak ada gambar contoh di {LT_DIR / 'images'}")
    if any(m.get("status") == "running" for m in await list_runs()):
        raise HTTPException(status_code=409, detail="masih ada run yang berjalan; hentikan dulu")

    meta = {
        "run_id": secrets.token_hex(3),
        "rate": rate,
        "duration_seconds": duration,
        "mode": mode,
        "images": images,
        "target": K6_TARGET,
        "started_at": time.time(),
        "status": "running",
    }
    meta["container"] = await _start_container(meta)
    await _save_meta(meta)
    _spawn_watcher(meta["run_id"])
    log.info("load test %s mulai: %s rps selama %ss (%s)", meta["run_id"], rate, duration, mode)
    return meta


@router.post("/{run}/samples")
async def add_sample(run: str, request: Request) -> dict[str, Any]:
    sample = await request.json()
    r = ctx["redis"]
    status = int(sample.get("status") or 0)
    bucket = (
        "200"
        if status == 200
        else "202"
        if status == 202
        else "timeout"
        if status == 0
        else "5xx"
        if status >= 500
        else "4xx"
    )
    sample["status"] = status
    sample["received_at"] = time.time()
    async with r.pipeline(transaction=False) as pipe:
        pipe.rpush(_key(run, "samples"), json.dumps(sample))
        pipe.hincrby(_key(run, "status"), bucket, 1)
        await pipe.execute()
    return {"ok": True}


@router.get("/{run}")
async def detail(run: str) -> dict[str, Any]:
    return await stats(run)


@router.post("/{run}/stop")
async def stop(run: str) -> dict[str, Any]:
    meta = await _meta(run)
    if meta is None:
        raise HTTPException(status_code=404, detail="run tidak ditemukan")
    await _docker("stop", "-t", "5", _container(run))
    _spawn_watcher(run)
    return {"ok": True}


@router.delete("/{run}")
async def remove(run: str) -> dict[str, Any]:
    meta = await _meta(run)
    if meta is None:
        raise HTTPException(status_code=404, detail="run tidak ditemukan")
    await _docker("rm", "-f", _container(run))
    task = watchers.pop(run, None)
    if task is not None:
        task.cancel()
    r = ctx["redis"]
    await r.delete(*[_key(run, s) for s in ("samples", "status", "cb", "done", "failed")])
    await r.hdel("ocr:loadtests", run)
    try:
        (LT_DIR / "out" / f"{run}.json").unlink(missing_ok=True)
    except OSError:
        pass
    deleted = await _cleanup_db(run)
    return {"ok": True, "db_rows_deleted": deleted}


async def _cleanup_db(run: str) -> dict[str, int] | None:
    pool = await ctx["get_pool"]()
    if pool is None:
        return None
    pattern = f"LT_{run}\\_%"
    tables = [f"{t}_jobs" for t in ctx["tables"].values()] + [f"{t}_results" for t in ctx["tables"].values()]
    tables += ["pipeline_outbox", "orchestration_extract_ocr"]
    deleted: dict[str, int] = {}
    async with pool.acquire() as conn:
        for table in tables:
            try:
                result = await conn.execute(f"DELETE FROM {table} WHERE request_id LIKE $1", pattern)
                deleted[table] = int(result.split()[-1])
            except Exception as exc:  # noqa: BLE001 - tabel opsional boleh tidak ada
                log.warning("cleanup %s dilewati: %s", table, exc)
    return deleted
