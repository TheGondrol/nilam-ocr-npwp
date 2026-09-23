"""
Tracker pipeline untuk uji coba lokal. Memerankan ORKESTRASI di sequence
diagram, secukupnya untuk melihat pipeline dan pola outbox-nya hidup:

    POST /api/requests               upload dokumen -> guardrails /v1/extract-ocr (menunggu maks. PIPELINE_WAIT_SECONDS)
    POST /v1/callbacks/stage         dipanggil relay ekstraksi / structuring / scoring (ORCHESTRATION_URL)
    GET  /api/requests               daftar request terakhir
    GET  /api/requests/{id}/events   SSE: semua event request itu (replay dari awal, lalu live)
    POST /api/requests/{id}/outbox/release   lepaskan dead letter request itu (failed_at = NULL)
    GET/PUT /api/simulation          bagaimana tracker menjawab callback: ok | down (503) | reject (422)
    GET  /api/outbox                 backlog outbox tiap service (GET /v1/<tahap>/outbox)

Redis Streams sebagai bus event: tiap request punya stream `ocr:events:<request_id>`.
Setiap event punya `type`:
    client    upload diterima
    http      jawaban guardrails /v1/extract-ocr (200 / 202 / 400 / 422) dan lamanya
    stage     status tahap (PROCESSING / DONE / FAILED / REJECTED), `source`: db | callback
    outbox    baris pipeline_outbox request ini: QUEUED / CLAIMED / RETRY / DELIVERED / DEAD / RELEASED
    callback  tiap callback yang datang ke tracker, dengan attempt ke-n dan jawaban tracker
    pipeline  END: tidak ada lagi yang akan terjadi untuk request ini

Baris outbox dan status job dibaca langsung dari PostgreSQL (TRACKER_DATABASE_URL),
tiap 250 ms selama request masih hidup. Tanpa itu (mis. mode GKE) tracker hanya
melihat callback dan polling status, seperti dulu.

Bukan bagian dari deliverable: tanpa auth, tanpa retensi, satu proses.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any

import httpx
import loadtest
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from redis.asyncio import Redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tracker")


def _load_env_file() -> None:
    """Baca tools/tracker/.env supaya `python backend/app.py` sama dengan run.sh."""
    path = os.path.join(os.path.dirname(__file__), "..", ".env")
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
    except OSError:
        pass


_load_env_file()

SERVICES = {
    "GUARDRAILS": os.environ.get("GUARDRAILS_URL", "http://127.0.0.1:8031"),
    "OCR": os.environ.get("EKSTRAKSI_URL", "http://127.0.0.1:8030"),
    "STRUCTURING": os.environ.get("STRUCTURING_URL", "http://127.0.0.1:8032"),
    "SCORING": os.environ.get("SCORING_URL", "http://127.0.0.1:8033"),
}
PREFIXES = {"OCR": "ekstraksi", "STRUCTURING": "structuring", "SCORING": "scoring"}
TABLES = {"OCR": "ocr", "STRUCTURING": "structuring", "SCORING": "scoring"}
JOB_PATHS = {stage: f"/v1/{prefix}/jobs" for stage, prefix in PREFIXES.items()}
API_KEY = os.environ.get("API_KEY")  # kosong kalau service dijalankan dengan AUTH_DISABLED=true
REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
STAGES = ["GUARDRAILS", "OCR", "STRUCTURING", "SCORING"]
NEXT_STAGE = {"OCR": "STRUCTURING", "STRUCTURING": "SCORING"}

TARGET = os.environ.get("TRACKER_TARGET", "local")
POLL = os.environ.get("TRACKER_POLL") == "1"
POLL_INTERVAL = float(os.environ.get("TRACKER_POLL_INTERVAL", "2"))
POLL_TIMEOUT = float(os.environ.get("TRACKER_POLL_TIMEOUT", "300"))
WAIT_SECONDS = float(os.environ.get("TRACKER_WAIT_SECONDS", "15"))  # = PIPELINE_WAIT_SECONDS guardrails, untuk label
DB_URL = os.environ.get("TRACKER_DATABASE_URL") or (
    f"postgresql://postgres:changeme@127.0.0.1:{os.environ.get('POSTGRES_HOST_PORT', '5434')}/bribrain_ocr_nilam"
    if TARGET == "local"
    else ""
)
DB_INTERVAL = float(os.environ.get("TRACKER_DB_INTERVAL", "0.25"))
DB_WATCH_TIMEOUT = float(os.environ.get("TRACKER_DB_WATCH_TIMEOUT", "900"))

CALLBACK_MODES = {"ok": 200, "down": 503, "reject": 422}

app = FastAPI(title="nilam-ocr tracker (stand-in Orkestrasi)")
redis = Redis.from_url(REDIS_URL, decode_responses=True)
http = httpx.AsyncClient(timeout=120.0, headers={"X-API-Key": API_KEY} if API_KEY else {})
simulation: dict[str, Any] = {"callback": "ok"}
watchers: dict[str, asyncio.Task[None]] = {}
pool: Any = None
db_error: str | None = None


# --- event bus -----------------------------------------------------------------


def _stream(request_id: str) -> str:
    return f"ocr:events:{request_id}"


async def emit(request_id: str, stage: str, status: str, *, type: str = "stage", **fields: Any) -> None:
    """Satu event ke stream request itu + perbarui ringkasan."""
    event = {"type": type, "request_id": request_id, "stage": stage, "status": status, "ts": time.time(), **fields}
    await redis.xadd(_stream(request_id), {"json": json.dumps(event)}, maxlen=2000, approximate=True)
    summary = json.loads(await redis.hget("ocr:requests", request_id) or "{}")
    summary.setdefault("request_id", request_id)
    summary.setdefault("created_at", event["ts"])
    summary["updated_at"] = event["ts"]
    if type == "stage":
        summary.update(stage=stage, status=status)
    elif type == "http":
        # Body ikut disimpan supaya daftar "Request terakhir" bisa memperlihatkan bedanya jawaban 200 dan 202.
        summary.update(
            http_status=fields.get("http_status"),
            job_status=fields.get("job_status"),
            elapsed_ms=fields.get("elapsed_ms"),
            wait_seconds=fields.get("wait_seconds"),
            body=fields.get("body"),
        )
    elif type == "client":
        summary.update(filename=fields.get("filename"), slow=fields.get("slow"), stage="CLIENT", status=status)
    elif type == "pipeline":
        summary["ended"] = True
    await redis.hset("ocr:requests", request_id, json.dumps(summary))
    log.info("event %s %s %s %s", type, request_id, stage, status)


# --- database watcher: outbox rows and job rows of one request -----------------


async def get_pool():
    global pool, db_error
    if pool is not None or not DB_URL:
        return pool
    try:
        import asyncpg

        pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=3, timeout=5)
        db_error = None
    except Exception as exc:  # noqa: BLE001
        db_error = f"{type(exc).__name__}: {exc}"
        log.warning("database watcher off: %s", db_error)
    return pool


loadtest.configure(redis=redis, get_pool=get_pool, wait_seconds=WAIT_SECONDS, tables=TABLES)
app.include_router(loadtest.router)


async def fetch_result(stage: str, request_id: str) -> Any:
    try:
        r = await http.get(f"{SERVICES[stage]}{JOB_PATHS[stage]}/{request_id}")
        return r.json()["data"]["result"]
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        log.warning("cannot fetch %s result for %s: %s", stage, request_id, exc)
        return None


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def _outbox_view(row: Any) -> dict[str, Any]:
    payload = row["payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    target = payload.get("next_stage") if row["kind"] == "handoff" else "ORKESTRASI"
    return {
        "id": row["id"],
        "owner": row["stage"],
        "kind": row["kind"],
        "target": target,
        "message": (
            f"POST /v1/{PREFIXES.get(target, '?')}/jobs"
            if row["kind"] == "handoff"
            else f"callback {payload.get('stage')} {payload.get('status')}"
        ),
        "attempts": row["attempts"],
        "next_attempt_at": _iso(row["next_attempt_at"]),
        "failed_at": _iso(row["failed_at"]),
        "last_error": row["last_error"],
        "created_at": _iso(row["created_at"]),
    }


async def guardrails_answered(request_id: str) -> bool:
    """True kalau jawaban HTTP guardrails untuk request ini sudah dicatat di ringkasan."""
    summary = json.loads(await redis.hget("ocr:requests", request_id) or "{}")
    return summary.get("http_status") is not None


async def watch_db(request_id: str) -> None:
    """Baca pipeline_outbox dan <tahap>_jobs request ini tiap DB_INTERVAL, pancarkan perubahannya
    sebagai event, dan tutup request (END) ketika tidak ada lagi yang akan terjadi."""
    db = await get_pool()
    if db is None:
        await emit(request_id, "OUTBOX", "UNAVAILABLE", type="outbox", error_message=db_error)
        return
    seen_jobs: dict[str, tuple[str, int]] = {}
    seen_rows: dict[int, dict[str, Any]] = {}
    deadline = time.time() + DB_WATCH_TIMEOUT
    while time.time() < deadline:
        async with db.acquire() as conn:
            jobs = {}
            for stage, table in TABLES.items():
                jobs[stage] = await conn.fetchrow(
                    f"SELECT status, attempts, error_message FROM {table}_jobs WHERE request_id = $1", request_id
                )
            rows = await conn.fetch(
                "SELECT id, stage, kind, payload, attempts, next_attempt_at, failed_at, last_error, created_at "
                "FROM pipeline_outbox WHERE request_id = $1 ORDER BY id",
                request_id,
            )

        for stage, job in jobs.items():
            if job is None:
                continue
            key = (job["status"], job["attempts"])
            if seen_jobs.get(stage) == key:
                continue
            seen_jobs[stage] = key
            result = await fetch_result(stage, request_id) if job["status"] == "DONE" else None
            await emit(
                request_id,
                stage,
                job["status"],
                source="db",
                attempt=job["attempts"],
                result=result,
                error_message=job["error_message"],
            )

        current = {row["id"]: _outbox_view(row) for row in rows}
        for row_id, view in current.items():
            before = seen_rows.get(row_id)
            if before is None:
                await emit(request_id, "OUTBOX", "QUEUED", type="outbox", message=view)
            else:
                if view["attempts"] > before["attempts"]:
                    await emit(request_id, "OUTBOX", "CLAIMED", type="outbox", message=view)
                if view["failed_at"] and not before["failed_at"]:
                    await emit(request_id, "OUTBOX", "DEAD", type="outbox", message=view)
                elif view["last_error"] != before["last_error"] or (before["failed_at"] and not view["failed_at"]):
                    status = "RELEASED" if before["failed_at"] and not view["failed_at"] else "RETRY"
                    await emit(request_id, "OUTBOX", status, type="outbox", message=view)
            seen_rows[row_id] = view
        for row_id in list(seen_rows):
            if row_id not in current:
                await emit(request_id, "OUTBOX", "DELIVERED", type="outbox", message=seen_rows.pop(row_id))

        statuses = {stage: job["status"] for stage, job in jobs.items() if job is not None}
        finished = statuses.get("SCORING") == "DONE" or "FAILED" in statuses.values()
        pending = [v for v in current.values() if not v["failed_at"]]
        dead = [v for v in current.values() if v["failed_at"]]
        # Jalur 200: guardrails baru menjawab setelah pipeline selesai, jadi jawabannya bisa tiba
        # sepersekian detik setelah scoring DONE. END menutup SSE; tunda sampai jawaban itu tercatat
        # supaya event GUARDRAILS DONE tidak tertulis di belakang END dan kartu tidak tersangkut PENDING.
        if finished and not pending and await guardrails_answered(request_id):
            await emit(request_id, "PIPELINE", "END", type="pipeline", dead_letters=len(dead))
            return
        await asyncio.sleep(DB_INTERVAL)
    await emit(request_id, "PIPELINE", "END", type="pipeline", timeout=True)


def start_watcher(request_id: str) -> None:
    task = watchers.get(request_id)
    if task is not None and not task.done():
        return
    watchers[request_id] = asyncio.create_task(watch_db(request_id))


async def poll_stages(request_id: str) -> None:
    """Pengganti callback di mode GKE: tarik status tiap tahap sampai selesai."""
    deadline = time.time() + POLL_TIMEOUT
    for stage in ("OCR", "STRUCTURING", "SCORING"):
        while time.time() < deadline:
            await asyncio.sleep(POLL_INTERVAL)
            try:
                r = await http.get(f"{SERVICES[stage]}{JOB_PATHS[stage]}/{request_id}")
            except httpx.HTTPError as exc:
                log.warning("poll %s %s: %s", stage, request_id, exc)
                continue
            if r.status_code != 200:  # 404 = tahap itu belum membuat job
                continue
            data = r.json().get("data") or {}
            status = data.get("status")
            if status == "DONE":
                await emit(request_id, stage, "DONE", source="poll", result=data.get("result"))
                if stage != "SCORING":
                    await emit(request_id, NEXT_STAGE[stage], "PROCESSING", source="poll")
                break
            if status == "FAILED":
                await emit(request_id, stage, "FAILED", source="poll", error_message=data.get("error_message"))
                await emit(request_id, "PIPELINE", "END", type="pipeline")
                return
        else:
            await emit(request_id, stage, "FAILED", source="poll", error_message=f"timeout {POLL_TIMEOUT:.0f}s")
            await emit(request_id, "PIPELINE", "END", type="pipeline")
            return
    await emit(request_id, "PIPELINE", "END", type="pipeline")


# --- the orchestrator's one call -------------------------------------------------


@app.post("/api/requests")
async def submit(
    file: UploadFile = File(...),
    document_type: str = Form("npwp"),
    slow_seconds: int = Form(0),
):
    request_id = f"REQ_{uuid.uuid4().hex[:12]}"
    content = await file.read()
    filename = file.filename or "upload"
    if slow_seconds > 0:
        # Hook lokal di ekstraksi (ENVIRONMENT=local): tahap OCR ditunda sebelum bekerja,
        # supaya pipeline melewati PIPELINE_WAIT_SECONDS dan guardrails menjawab 202.
        filename = f"delay{slow_seconds}s-{filename}"
    content_type = file.content_type or "image/jpeg"
    await emit(
        request_id,
        "CLIENT",
        "SUBMITTED",
        type="client",
        filename=filename,
        size=len(content),
        document_type=document_type,
        slow=slow_seconds,
        callback_mode=simulation["callback"],
        wait_seconds=WAIT_SECONDS,
    )
    start_watcher(request_id)

    started = time.perf_counter()
    try:
        r = await http.post(
            f"{SERVICES['GUARDRAILS']}/v1/extract-ocr",
            data={"request_id": request_id, "document_type": document_type},
            files={"file": (filename, content, content_type)},
        )
        body = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        await emit(request_id, "GUARDRAILS", "FAILED", error_message=f"guardrails unreachable: {exc}")
        await emit(request_id, "PIPELINE", "END", type="pipeline")
        raise HTTPException(status_code=503, detail="guardrails service unavailable") from exc
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    await emit(
        request_id,
        "GUARDRAILS",
        "RESPONSE",
        type="http",
        http_status=r.status_code,
        job_status=body.get("job_status"),
        errors=body.get("errors"),
        message=body.get("message"),
        elapsed_ms=elapsed_ms,
        wait_seconds=WAIT_SECONDS,
        body=body,
    )
    if r.status_code == 400 and body.get("errors") == "DOWNSTREAM_VALIDATION_ERROR":
        await emit(request_id, "GUARDRAILS", "REJECTED", elapsed_ms=elapsed_ms, error_message=body.get("message"))
        await emit(request_id, "PIPELINE", "END", type="pipeline")
        return {"request_id": request_id, "accepted": False, "reason": body.get("message")}
    if r.status_code not in (200, 202, 422):
        await emit(request_id, "GUARDRAILS", "FAILED", http_status=r.status_code, error_message=body.get("message"))
        await emit(request_id, "PIPELINE", "END", type="pipeline")
        return {"request_id": request_id, "accepted": False, "reason": body.get("message")}
    await emit(request_id, "GUARDRAILS", "DONE", elapsed_ms=elapsed_ms)
    if pool is None:
        await emit(request_id, "OCR", "PROCESSING", source="guardrails")
    if POLL:
        asyncio.create_task(poll_stages(request_id))
    return {
        "request_id": request_id,
        "accepted": True,
        "http_status": r.status_code,
        "job_status": body.get("job_status"),
    }


# --- callbacks from the relays ---------------------------------------------------


@app.post("/v1/callbacks/stage")
async def callback(request: Request):
    """Kontrak callback yang diusulkan repo ini: {request_id, stage, status, result, error_message}.
    Jawaban tracker mengikuti simulasi: ok -> 200, down -> 503 (relay mengulang dengan backoff),
    reject -> 422 (relay berhenti: dead letter)."""
    body = await request.json()
    request_id, stage, status = body["request_id"], body["stage"], body["status"]
    mode = simulation["callback"]
    http_status = CALLBACK_MODES[mode]
    if request_id.startswith("LT_"):
        # Request dari load tester: dihitung terpisah, tidak masuk daftar request dan stream event.
        await loadtest.record_callback(body, accepted=http_status == 200)
        if http_status != 200:
            return JSONResponse(
                status_code=http_status, content={"detail": f"simulasi: orkestrasi menjawab {http_status}"}
            )
        return {"ok": True}
    attempt = await redis.hincrby(f"ocr:callbacks:{request_id}", f"{stage}:{status}", 1)
    await emit(
        request_id,
        stage,
        status,
        type="callback",
        attempt=attempt,
        accepted=http_status == 200,
        http_status=http_status,
        mode=mode,
        error_message=body.get("error_message"),
    )
    if http_status != 200:
        return JSONResponse(status_code=http_status, content={"detail": f"simulasi: orkestrasi menjawab {http_status}"})
    result = body.get("result")
    if status == "DONE" and result is None and stage in JOB_PATHS:
        result = await fetch_result(stage, request_id)
    await emit(request_id, stage, status, source="callback", result=result, error_message=body.get("error_message"))
    if pool is None and status == "DONE" and stage in NEXT_STAGE:
        await emit(request_id, NEXT_STAGE[stage], "PROCESSING", source="callback")
    if pool is None and not POLL and (status == "FAILED" or stage == "SCORING"):
        await emit(request_id, "PIPELINE", "END", type="pipeline")
    return {}


# --- simulation, outbox, listing ----------------------------------------------


@app.get("/api/simulation")
async def get_simulation():
    return {**simulation, "wait_seconds": WAIT_SECONDS, "database": pool is not None, "database_error": db_error}


@app.put("/api/simulation")
async def put_simulation(body: dict[str, Any]):
    mode = body.get("callback", simulation["callback"])
    if mode not in CALLBACK_MODES:
        raise HTTPException(status_code=422, detail=f"callback harus salah satu dari {sorted(CALLBACK_MODES)}")
    simulation["callback"] = mode
    log.info("simulation: callback=%s", mode)
    return await get_simulation()


@app.post("/api/requests/{request_id}/outbox/release")
async def release(request_id: str):
    db = await get_pool()
    if db is None:
        raise HTTPException(status_code=503, detail=f"database tidak tersedia: {db_error}")
    async with db.acquire() as conn:
        released = await conn.execute(
            "UPDATE pipeline_outbox SET failed_at = NULL, next_attempt_at = now(), updated_at = now() "
            "WHERE request_id = $1 AND failed_at IS NOT NULL",
            request_id,
        )
    start_watcher(request_id)
    return {"released": int(released.split()[-1])}


@app.get("/api/outbox")
async def outbox_overview():
    """Backlog tiap tahap dari service-nya sendiri (GET /v1/<tahap>/outbox)."""
    out = {}
    for stage, prefix in PREFIXES.items():
        try:
            r = await http.get(f"{SERVICES[stage]}/v1/{prefix}/outbox", timeout=3.0)
            out[stage] = r.json().get("data") if r.status_code == 200 else {"error": r.status_code}
        except (httpx.HTTPError, ValueError) as exc:
            out[stage] = {"error": type(exc).__name__}
    return out


@app.get("/api/requests")
async def list_requests():
    rows = [json.loads(v) for v in (await redis.hgetall("ocr:requests")).values()]
    rows.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return rows[:50]


@app.get("/api/requests/{request_id}/events")
async def events(request_id: str, request: Request):
    """SSE: replay seluruh stream lalu live. Ditutup pada event pipeline END atau saat client pergi."""

    async def generate():
        last_id = "0"
        idle = 0
        while not await request.is_disconnected():
            entries = await redis.xread({_stream(request_id): last_id}, block=5000, count=100)
            if not entries:
                idle += 1
                yield ": keep-alive\n\n"
                if idle > 240:  # ~20 menit tanpa event
                    break
                continue
            idle = 0
            for _, items in entries:
                for entry_id, fields in items:
                    last_id = entry_id
                    event = json.loads(fields["json"])
                    yield f"id: {entry_id}\nevent: stage\ndata: {json.dumps(event)}\n\n"
                    if event["type"] == "pipeline" and event["status"] == "END":
                        yield "event: end\ndata: {}\n\n"
                        return

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
async def health():
    services = {}
    for name, url in SERVICES.items():
        try:
            r = await http.get(f"{url}/health", timeout=3.0)
            services[name] = r.json().get("backends", r.status_code)
        except httpx.HTTPError:
            services[name] = "unreachable"
    await get_pool()
    return {
        "target": TARGET,
        "poll": POLL,
        "redis": await redis.ping(),
        "database": pool is not None,
        "database_error": db_error,
        "simulation": simulation,
        "services": services,
    }


@app.on_event("startup")
async def startup():
    await get_pool()
    await loadtest.reconcile()


@app.on_event("shutdown")
async def shutdown():
    for task in watchers.values():
        task.cancel()
    await http.aclose()
    await redis.aclose()
    if pool is not None:
        await pool.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8090")))
