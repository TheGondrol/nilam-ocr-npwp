"""
Bukti pengiriman outbox dari log relay (mode lokal). Baris nilam_pipeline_outbox di jalur normal hanya hidup
beberapa milidetik: ditulis bersama hasil tahap, lalu langsung diklaim, dikirim, dan dihapus relay. Watcher DB
tracker membaca tabel tiap TRACKER_DB_INTERVAL (0,25 dtk), jadi baris itu hampir tidak pernah terlihat.

Relay mencatat setiap pengiriman di log service-nya (ocr_common.pipeline.outbox, `_log_delivery`):

    ... INFO ocr_common.pipeline.outbox [REQ_x]: outbox handoff OCR -> STRUCTURING: delivered (attempt 1)
    ... INFO ocr_common.pipeline.outbox [REQ_x]: outbox callback OCR -> orchestration: skipped (attempt 1)

Modul ini mengikuti `docker logs -f` container extraction / structuring / scoring dan memancarkan hasil akhir
setiap pesan (`delivered` / `skipped`) sebagai event outbox `source: log`. RETRY dan DEAD tetap dari tabel:
baris yang gagal menetap cukup lama untuk terbaca, dan di tabel ada `next_attempt_at` serta id-nya.
Hanya format log teks (default lokal, LOG_FORMAT tidak diisi).
"""

import asyncio
import logging
import re
import subprocess
import threading
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, cast

log = logging.getLogger("tracker.outbox_logs")

SERVICES = ("extraction", "structuring", "scoring")
LINE = re.compile(
    r"\[(?P<rid>[^\]\s]+)\]: outbox (?P<kind>handoff|callback) (?P<owner>[A-Z]+) -> (?P<target>\S+): "
    r"(?P<result>delivered|skipped) \(attempt (?P<attempt>\d+)\)"
)
RESTART_SECONDS = 2.0

Emit = Callable[..., Awaitable[None]]
Known = Callable[[str], Awaitable[bool]]


def parse(line: str) -> dict[str, Any] | None:
    """Satu baris `docker logs -t`: `<RFC3339Nano> <baris log>`. None kalau bukan log pengiriman outbox."""
    stamp, _, text = line.partition(" ")
    match = LINE.search(text)
    if match is None:
        return None
    try:
        # Docker memberi nanodetik; datetime hanya sampai mikrodetik.
        head, _, frac = stamp.rstrip("Z").partition(".")
        ts = datetime.fromisoformat(f"{head}.{(frac + '000000')[:6]}+00:00").timestamp()
    except ValueError:
        return None
    target = match["target"]
    return {
        "request_id": match["rid"],
        "ts": ts,
        "status": match["result"].upper(),
        "message": {
            "id": None,
            "owner": match["owner"],
            "kind": match["kind"],
            "target": "ORKESTRASI" if target == "orchestration" else target,
            "attempts": int(match["attempt"]),
            "last_error": None,
        },
    }


class OutboxLogFollower:
    """Satu `docker logs -f` per service, dimulai ulang kalau container dibuat ulang atau prosesnya berhenti."""

    def __init__(self, emit: Emit, known: Known, container: Callable[[str], str]):
        self._emit = emit
        self._known = known
        self._container = container
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopping = threading.Event()
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        self._streaming: set[str] = set()

    @property
    def active(self) -> bool:
        """True selama log semua service pipeline sedang diikuti: watcher DB menyerahkan DELIVERED ke log."""
        return self._streaming.issuperset(SERVICES)

    def start(self) -> None:
        # Thread + subprocess biasa, bukan asyncio subprocess: uvicorn --reload di Windows memakai SelectorEventLoop,
        # yang tidak bisa menjalankan subprocess.
        self._loop = asyncio.get_running_loop()
        for service in SERVICES:
            threading.Thread(target=self._follow, args=(service,), name=f"outbox-log-{service}", daemon=True).start()

    async def stop(self) -> None:
        self._stopping.set()
        for proc in list(self._procs.values()):
            if proc.poll() is None:
                proc.kill()

    def _follow(self, service: str) -> None:
        since = datetime.now().astimezone().isoformat()
        while not self._stopping.is_set():
            try:
                proc = subprocess.Popen(
                    ["docker", "logs", "-f", "-t", "--since", since, self._container(service)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
            except OSError as exc:
                log.warning("log outbox %s tidak terbaca: %s", service, exc)
                time.sleep(RESTART_SECONDS)
                continue
            self._procs[service] = proc
            self._streaming.add(service)
            assert proc.stdout is not None
            for raw in proc.stdout:
                event = parse(raw.decode("utf-8", "replace").rstrip())
                if event is not None and not event["request_id"].startswith("LT_"):
                    asyncio.run_coroutine_threadsafe(self._handle(event), cast(asyncio.AbstractEventLoop, self._loop))
            proc.wait()
            self._streaming.discard(service)
            # Container dihentikan / dibuat ulang: ikuti lagi dari sekarang.
            since = datetime.now().astimezone().isoformat()
            time.sleep(RESTART_SECONDS)

    async def _handle(self, event: dict[str, Any]) -> None:
        try:
            if not await self._known(event["request_id"]):
                return
            await self._emit(
                event["request_id"],
                "OUTBOX",
                event["status"],
                type="outbox",
                source="log",
                ts=event["ts"],
                message=event["message"],
            )
        except Exception:  # noqa: BLE001 - satu baris log yang gagal dipancarkan tidak boleh menghentikan pembaca
            log.exception("event outbox dari log %s gagal dipancarkan", event["request_id"])
