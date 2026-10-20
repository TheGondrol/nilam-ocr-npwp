"""Performance test ke pipeline yang terdeploy di GKE (menu "Performance Test"): k6 dijalankan sebagai Job di namespace
pipeline dan menembak Service orchestrator langsung, bukan lewat port-forward (yang mengukur tunnel, bukan
service), di endpoint -test (tabel nilam_testing_*, tanpa callback ke Orkestrasi pusat).

Satu run:
  1. ConfigMap nilam-perf-<run> (perf.js + file uji) dan Job nilam-perf-<run> (grafana/k6) dibuat lewat kubectl.
  2. Satu baris log `PT|...` k6 per request dibaca dengan `kubectl logs -f` (pod tidak bisa menjangkau laptop).
  3. Selama run: `kubectl top pods` tiap 15 dtk (CPU/memori tiap service dan generator k6), restart/OOMKilled pod,
     dan snapshot tabel nilam_testing_* tiap 10 dtk lewat pod relay Cloud SQL (port-forward), untuk end-to-end,
     durasi per tahap, dan jeda antar tahap. Profil stress berhenti otomatis kalau SLO jelas dilanggar.
  4. Setelah k6 selesai, tunggu pipeline menghabiskan antrean (maks. DRAIN_SECONDS), hitung hasil
     (perf_stats.compute), tulis laporan PDF (perf_report) ke load-tester/out/perf-<run>.pdf, hapus Job/ConfigMap.

Kunci Redis:
  ocr:perftests              hash run_id -> meta (json)
  ocr:pt:<run>:samples       hash request_id -> sampel k6 (json)
  ocr:pt:<run>:res           list sampel resource (json)
  ocr:pt:<run>:pods          restart per pod (json)
  ocr:pt:<run>:db            snapshot database terakhir (json)
  ocr:pt:<run>:k6            ringkasan akhir k6 (json)
  ocr:pt:<run>:log           ekor log k6 selain baris sampel
  ocr:pt:<run>:final         hasil akhir (json), dipakai detail dan laporan run yang sudah selesai
"""

import asyncio
import base64
import contextlib
import json
import logging
import math
import os
import secrets
import socket
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse

import loadtest
import perf_report
import perf_stats as ps
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

log = logging.getLogger("tracker.perftest")
router = APIRouter(prefix="/api/perftest")

CONTEXT = os.environ.get("PERF_KUBE_CONTEXT", "gke_ddb-kubecluster-dev-01_asia-southeast2_gc-ddb-dev-gke-cluster-01")
NAMESPACE = os.environ.get("PERF_NAMESPACE", "nilam-ocr-npwp")
RELEASE = os.environ.get("PERF_RELEASE", "nilam-ocr-npwp")
SECRET = os.environ.get("PERF_SECRET", f"{RELEASE}-secrets")
TARGET = os.environ.get("PERF_TARGET", f"http://{RELEASE}-orchestrator:8034")
ENDPOINT = "/v1/extract-ocr-test"
K6_IMAGE = os.environ.get("PERF_K6_IMAGE", "grafana/k6:2.3.0")
# Pod relay TCP ke Cloud SQL yang sudah ada di namespace; kalau tidak ada, run membuat relay sendiri (RELAY_IMAGE).
DB_RELAY_POD = os.environ.get("PERF_DB_RELAY_POD", "cloudsql-relay-ocr-nilam")
RELAY_IMAGE = os.environ.get("PERF_RELAY_IMAGE", "alpine/socat:1.8.0.3")
MAX_RATE = float(os.environ.get("PERF_MAX_RATE", "10"))
MAX_SECONDS = int(os.environ.get("PERF_MAX_SECONDS", str(2 * 3600)))
DRAIN_SECONDS = int(os.environ.get("PERF_DRAIN_SECONDS", "180"))
POD_START_TIMEOUT = 240
RESOURCE_INTERVAL = 15
DB_INTERVAL = 10
ABORT_INTERVAL = 15
# ConfigMap maksimal 1 MiB, diukur dari isinya: file uji di binaryData sebagai base64 (4/3 ukuran file), skrip
# k6 apa adanya. Sisanya ruang untuk metadata. Jadi total file uji mentah paling banyak sekitar 720 KB.
CONFIGMAP_LIMIT = 960 * 1024
GENERATOR_LIMITS = {"cpu": "1", "memory": "1Gi"}
OUT_DIR = loadtest.LT_DIR / "out"
SCRIPT = loadtest.LT_DIR / "k6" / "perf.js"
ACTIVE = ("starting", "running", "draining")
# Tabel lajur -test (ocr_common.pipeline.tables.stage_table_names dengan prefiks testing_), schema nilam_ocr_npwp.
TEST_TABLES = {
    "extraction": "nilam_testing_ocr_extraction",
    "structuring": "nilam_testing_structuring",
    "scoring": "nilam_testing_scoring",
}
TEST_GUARDRAILS = "nilam_testing_guardrails_results"
TEST_OUTBOX = "nilam_testing_pipeline_outbox"

ctx: dict[str, Any] = {}
tasks: dict[str, asyncio.Task[None]] = {}


def configure(*, redis: Any, wait_seconds: float) -> None:
    ctx.update(redis=redis, wait_seconds=wait_seconds)


def _key(run: str, suffix: str) -> str:
    return f"ocr:pt:{run}:{suffix}"


def _job(run: str) -> str:
    return f"nilam-perf-{run}"


def _labels(run: str) -> dict[str, str]:
    return {
        "app.kubernetes.io/name": "nilam-perftest",
        "app.kubernetes.io/managed-by": "nilam-tracker",
        "nilam/perf-run": run,
    }


# --- kubectl -------------------------------------------------------------------------


def _kube_cmd(*args: str, namespaced: bool = True) -> list[str]:
    cmd = ["kubectl", "--context", CONTEXT]
    if namespaced:
        cmd += ["-n", NAMESPACE]
    return [*cmd, *args]


def _kubectl_sync(*args: str, input_text: str | None = None, timeout: float = 60) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            _kube_cmd(*args),
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return proc.returncode, proc.stdout or "", (proc.stderr or "").strip()
    except Exception as exc:  # noqa: BLE001 - kubectl yang tidak ada / macet dilaporkan sebagai teks
        return 1, "", str(exc)


async def kubectl(*args: str, input_text: str | None = None, timeout: float = 60) -> tuple[int, str, str]:
    return await asyncio.to_thread(_kubectl_sync, *args, input_text=input_text, timeout=timeout)


async def kube_json(*args: str) -> Any:
    code, out, err = await kubectl(*args, "-o", "json")
    if code != 0:
        raise RuntimeError(err or out)
    return json.loads(out)


async def apply(manifest: dict[str, Any], *, verb: str = "apply") -> None:
    """`kubectl apply`, or `create` for an object that is new by construction (one per run). The ConfigMap with the
    test files must be created: apply copies the whole object into its last-applied-configuration annotation, and
    annotations may not exceed 256 KiB, far below the 1 MiB of the ConfigMap itself."""
    code, out, err = await kubectl(verb, "-f", "-", input_text=json.dumps(manifest))
    if code != 0:
        raise RuntimeError(f"kubectl {verb} {manifest['kind']} gagal: {(err or out)[-600:]}")


def _cpu_m(value: str | None) -> float | None:
    if not value:
        return None
    value = str(value)
    if value.endswith("m"):
        return float(value[:-1])
    if value.endswith("n"):
        return float(value[:-1]) / 1e6
    return float(value) * 1000


def _mem_mi(value: str | None) -> float | None:
    if not value:
        return None
    value = str(value)
    for suffix, factor in (("Ki", 1 / 1024), ("Mi", 1), ("Gi", 1024), ("K", 1 / 1024), ("M", 1), ("G", 1024)):
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * factor
    return float(value) / (1024 * 1024)


def _rfc3339(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


# --- Redis ------------------------------------------------------------------------------


async def _meta(run: str) -> dict[str, Any] | None:
    raw = await ctx["redis"].hget("ocr:perftests", run)
    return json.loads(raw) if raw else None


async def _save(meta: dict[str, Any]) -> None:
    await ctx["redis"].hset("ocr:perftests", meta["run_id"], json.dumps(meta))


async def _update(run: str, **fields: Any) -> dict[str, Any]:
    meta = await _meta(run) or {"run_id": run}
    meta.update(fields)
    await _save(meta)
    return meta


async def _json_key(run: str, suffix: str, default: Any = None) -> Any:
    raw = await ctx["redis"].get(_key(run, suffix))
    return json.loads(raw) if raw else default


async def _samples(run: str) -> dict[str, dict[str, Any]]:
    return {rid: json.loads(v) for rid, v in (await ctx["redis"].hgetall(_key(run, "samples"))).items()}


async def _resources(run: str) -> list[dict[str, Any]]:
    return [json.loads(v) for v in await ctx["redis"].lrange(_key(run, "res"), 0, -1)]


# --- cluster: batas resource, versi image ------------------------------------------------


async def _cluster_info() -> dict[str, Any]:
    """Replika, limit per pod, image, dan PIPELINE_WAIT_SECONDS orchestrator, dibaca saat run dimulai."""
    data = await kube_json("get", "deploy")
    limits: dict[str, Any] = {}
    versions: dict[str, str] = {}
    wait_seconds = ctx.get("wait_seconds", 15)
    affinity = None
    for item in data.get("items", []):
        name = item["metadata"]["name"]
        service = next((s for s in ps.SERVICES if name == f"{RELEASE}-{s}"), None)
        if service is None:
            continue
        spec = item["spec"]["template"]["spec"]
        container = spec["containers"][0]
        res = container.get("resources", {})
        limits[service] = {
            "replicas": item["spec"].get("replicas", 1),
            "cpu_m": _cpu_m(res.get("limits", {}).get("cpu")),
            "mem_mi": _mem_mi(res.get("limits", {}).get("memory")),
            "cpu_request_m": _cpu_m(res.get("requests", {}).get("cpu")),
        }
        versions[service] = container.get("image", "").rsplit("/", 1)[-1]
        if service == "orchestrator":
            affinity = spec.get("affinity")
            for env in container.get("env", []):
                if env.get("name") == "PIPELINE_WAIT_SECONDS" and env.get("value"):
                    wait_seconds = float(env["value"])
    limits["generator"] = {
        "replicas": 1,
        "cpu_m": _cpu_m(GENERATOR_LIMITS["cpu"]),
        "mem_mi": _mem_mi(GENERATOR_LIMITS["memory"]),
    }
    return {"limits": limits, "versions": versions, "wait_seconds": wait_seconds, "affinity": affinity}


# --- manifest ---------------------------------------------------------------------------


def configmap_size(images: list[Path]) -> int:
    """Bytes the ConfigMap of `_configmap` holds: each file base64-encoded, plus the k6 script."""
    return sum(4 * math.ceil(p.stat().st_size / 3) for p in images) + SCRIPT.stat().st_size


def _configmap(run: str, images: list[Path]) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": _job(run), "labels": _labels(run)},
        "data": {"perf.js": SCRIPT.read_text(encoding="utf-8")},
        "binaryData": {p.name: base64.b64encode(p.read_bytes()).decode() for p in images},
    }


def _job_manifest(meta: dict[str, Any], affinity: Any) -> dict[str, Any]:
    run = meta["run_id"]
    start_rate, stages = ps.k6_stages(meta["plan"])
    slo = meta["slo"]
    env = {
        "RUN_ID": run,
        "TARGET": TARGET,
        "ENDPOINT": ENDPOINT,
        "STAGES": json.dumps(stages),
        "START_RATE": str(start_rate),
        "PRE_VUS": str(meta["vus"][0]),
        "MAX_VUS": str(meta["vus"][1]),
        "IMAGES": ",".join(meta["images"]),
        "WAIT_SECONDS": str(meta["wait_seconds"]),
        "SLO_ERROR_RATE": str(slo["error_rate"]),
        "SLO_ENTRY_P95_MS": str(slo["entry_p95_ms"]),
        "HOME": "/tmp",
    }
    if meta.get("pipeline_name_sequence"):
        env["PIPELINE_NAME_SEQUENCE"] = json.dumps(meta["pipeline_name_sequence"])
    env_list: list[dict[str, Any]] = [{"name": k, "value": v} for k, v in env.items()]
    env_list.append({"name": "API_KEY", "valueFrom": {"secretKeyRef": {"name": SECRET, "key": "API_KEY"}}})
    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "automountServiceAccountToken": False,
        "securityContext": {"runAsNonRoot": True, "runAsUser": 12345, "seccompProfile": {"type": "RuntimeDefault"}},
        "containers": [
            {
                "name": "k6",
                "image": K6_IMAGE,
                "imagePullPolicy": "IfNotPresent",
                "args": ["run", "--log-format=raw", "--quiet", "--no-color", "/perf/perf.js"],
                "env": env_list,
                "resources": {"requests": {"cpu": "100m", "memory": "128Mi"}, "limits": GENERATOR_LIMITS},
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "readOnlyRootFilesystem": True,
                    "capabilities": {"drop": ["ALL"]},
                },
                "volumeMounts": [
                    {"name": "perf", "mountPath": "/perf", "readOnly": True},
                    {"name": "tmp", "mountPath": "/tmp"},
                ],
            }
        ],
        "volumes": [{"name": "perf", "configMap": {"name": _job(run)}}, {"name": "tmp", "emptyDir": {}}],
    }
    # Afinitas orchestrator ikut dipakai: node yang sengaja dihindari pipeline juga dihindari generator.
    if affinity:
        pod_spec["affinity"] = affinity
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": _job(run), "labels": _labels(run)},
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 3600,
            "activeDeadlineSeconds": ps.plan_seconds(meta["plan"]) + 600,
            "template": {"metadata": {"labels": _labels(run)}, "spec": pod_spec},
        },
    }


# --- log k6 --------------------------------------------------------------------------------


async def _handle_line(run: str, line: str, state: dict[str, Any]) -> None:
    """Satu baris `kubectl logs --timestamps`: `<RFC3339> <pesan>`."""
    stamp, _, text = line.partition(" ")
    if not text:
        stamp, text = "", line
    state["last_ts"] = stamp or state.get("last_ts")
    r = ctx["redis"]
    if text.startswith("PT|"):
        try:
            sample = json.loads(text[3:])
        except ValueError:
            return
        await r.hset(_key(run, "samples"), sample["r"], json.dumps(sample))
    elif text.startswith("PTSUMMARY|"):
        with contextlib.suppress(ValueError):
            json.loads(text[10:])
            await r.set(_key(run, "k6"), text[10:])
    elif text.strip():
        await r.rpush(_key(run, "log"), text[:500])
        await r.ltrim(_key(run, "log"), -200, -1)


def _stream_thread(cmd: list[str], loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, holder: dict) -> None:
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
        )
    except Exception as exc:  # noqa: BLE001
        loop.call_soon_threadsafe(queue.put_nowait, f"__ERROR__ {exc}")
        loop.call_soon_threadsafe(queue.put_nowait, None)
        return
    holder["proc"] = proc
    assert proc.stdout is not None
    for line in proc.stdout:
        loop.call_soon_threadsafe(queue.put_nowait, line.rstrip("\n"))
    proc.wait()
    loop.call_soon_threadsafe(queue.put_nowait, None)


async def _job_state(run: str) -> tuple[str, int | None]:
    """(running | finished | missing, exit code container k6)."""
    code, out, _ = await kubectl("get", "pods", "-l", f"job-name={_job(run)}", "-o", "json")
    if code != 0:
        return "missing", None
    pods = json.loads(out).get("items", [])
    if not pods:
        code, _, _ = await kubectl("get", "job", _job(run))
        return ("running" if code == 0 else "missing"), None
    status = pods[-1].get("status", {})
    phase = status.get("phase")
    exit_code = None
    for cs in status.get("containerStatuses", []):
        terminated = (cs.get("state") or {}).get("terminated")
        if terminated:
            exit_code = terminated.get("exitCode")
    return ("finished" if phase in ("Succeeded", "Failed") else "running"), exit_code


async def _follow_logs(run: str) -> int | None:
    """Ikuti log Job sampai k6 selesai; sambung ulang (--since-time) kalau stream putus di tengah run."""
    state: dict[str, Any] = {}
    loop = asyncio.get_running_loop()
    while True:
        cmd = _kube_cmd("logs", "-f", f"job/{_job(run)}", "--timestamps", "--pod-running-timeout=2m")
        if state.get("last_ts"):
            cmd.append(f"--since-time={state['last_ts']}")
        queue: asyncio.Queue = asyncio.Queue()
        holder: dict[str, Any] = {}
        thread = threading.Thread(target=_stream_thread, args=(cmd, loop, queue, holder), daemon=True)
        thread.start()
        try:
            while (line := await queue.get()) is not None:
                await _handle_line(run, line, state)
        finally:
            proc = holder.get("proc")
            if proc is not None and proc.poll() is None:
                proc.terminate()
        job_state, exit_code = await _job_state(run)
        if job_state != "running":
            return exit_code
        await asyncio.sleep(2)


# --- sampler resource ------------------------------------------------------------------------


async def _sample_resources(run: str) -> None:
    r = ctx["redis"]
    baseline: dict[str, int] = {}
    pods: dict[str, dict[str, Any]] = {}
    tick = 0
    while True:
        code, out, _ = await kubectl("top", "pods", "--no-headers", timeout=30)
        if code == 0:
            values = {}
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    with contextlib.suppress(ValueError, TypeError):
                        values[parts[0]] = [_cpu_m(parts[1]), _mem_mi(parts[2])]
            if values:
                await r.rpush(_key(run, "res"), json.dumps({"ts": time.time(), "pods": values}))
        if tick % 2 == 0:
            with contextlib.suppress(Exception):
                data = await kube_json("get", "pods")
                for item in data.get("items", []):
                    name = item["metadata"]["name"]
                    service = ps.service_of_pod(name, RELEASE)
                    if service in (None, "generator"):
                        continue
                    statuses = item.get("status", {}).get("containerStatuses", [])
                    count = sum(int(cs.get("restartCount") or 0) for cs in statuses)
                    reasons = [
                        ((cs.get("lastState") or {}).get("terminated") or {}).get("reason")
                        for cs in statuses
                        if (cs.get("lastState") or {}).get("terminated")
                    ]
                    if tick == 0:
                        baseline[name] = count
                    entry = pods.setdefault(name, {"service": service, "restarts": 0, "reasons": []})
                    delta = count - baseline.get(name, 0)
                    if delta > entry["restarts"]:
                        entry["restarts"] = delta
                        entry["reasons"] = [x for x in reasons if x]
                await r.set(_key(run, "pods"), json.dumps(pods))
        tick += 1
        await asyncio.sleep(RESOURCE_INTERVAL)


# --- database (lewat relay Cloud SQL) --------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


class DbLink:
    """Port-forward ke pod relay Cloud SQL + satu koneksi asyncpg; dibuka ulang kalau putus."""

    def __init__(self, run: str):
        self.run = run
        self.relay: str | None = None
        self.own_relay = False
        self.proc: subprocess.Popen | None = None
        self.conn: Any = None
        self.port = 0
        self.dsn: dict[str, Any] | None = None

    async def _dsn(self) -> dict[str, Any]:
        code, out, err = await kubectl("get", "secret", SECRET, "-o", "jsonpath={.data.DATABASE_URL}")
        if code != 0 or not out:
            raise RuntimeError(f"DATABASE_URL tidak terbaca dari Secret {SECRET}: {err}")
        url = urlparse(base64.b64decode(out).decode())
        query = dict(parse_qsl(url.query))
        ssl = query.pop("ssl", None) or query.pop("sslmode", None)
        return {
            # urlparse tidak men-decode %XX; password di DATABASE_URL di-percent-encode.
            "user": unquote(url.username or ""),
            "password": unquote(url.password or ""),
            "host": url.hostname,
            "port": url.port or 5432,
            "database": url.path.lstrip("/"),
            "ssl": ssl,
            "query": query,
        }

    async def _ensure_relay(self) -> str:
        code, out, _ = await kubectl("get", "pod", DB_RELAY_POD, "-o", "jsonpath={.status.phase}")
        if code == 0 and out.strip() == "Running":
            return DB_RELAY_POD
        name = f"nilam-perf-dbrelay-{self.run}"
        assert self.dsn is not None
        await apply(
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": {"name": name, "labels": _labels(self.run)},
                "spec": {
                    "automountServiceAccountToken": False,
                    "containers": [
                        {
                            "name": "socat",
                            "image": RELAY_IMAGE,
                            "args": [
                                "tcp-listen:5432,fork,reuseaddr",
                                f"tcp-connect:{self.dsn['host']}:{self.dsn['port']}",
                            ],
                            "resources": {"requests": {"cpu": "10m", "memory": "16Mi"}, "limits": {"memory": "64Mi"}},
                        }
                    ],
                },
            }
        )
        self.own_relay = True
        for _ in range(30):
            code, out, _ = await kubectl("get", "pod", name, "-o", "jsonpath={.status.phase}")
            if out.strip() == "Running":
                return name
            await asyncio.sleep(2)
        raise RuntimeError(f"pod relay {name} tidak Running")

    async def connect(self) -> Any:
        if self.conn is not None and not self.conn.is_closed() and self.proc and self.proc.poll() is None:
            return self.conn
        await self.close(keep_relay=True)
        import asyncpg

        self.dsn = self.dsn or await self._dsn()
        self.relay = self.relay or await self._ensure_relay()
        self.port = _free_port()
        self.proc = subprocess.Popen(
            _kube_cmd("port-forward", f"pod/{self.relay}", f"{self.port}:5432"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(40):
            if _port_open(self.port):
                break
            await asyncio.sleep(0.5)
        else:
            raise RuntimeError("port-forward ke relay database tidak siap")
        dsn = self.dsn
        query = {**dsn["query"], **({"sslmode": dsn["ssl"]} if dsn["ssl"] else {})}
        url = urlunparse(("postgresql", f"127.0.0.1:{self.port}", f"/{dsn['database']}", "", urlencode(query), ""))
        self.conn = await asyncpg.connect(
            url,
            user=dsn["user"],
            password=dsn["password"],
            timeout=10,
            server_settings={"search_path": "nilam_ocr_npwp, public"},
        )
        return self.conn

    async def close(self, *, keep_relay: bool = False) -> None:
        if self.conn is not None:
            with contextlib.suppress(Exception):
                await self.conn.close()
            self.conn = None
        if self.proc is not None:
            with contextlib.suppress(Exception):
                self.proc.terminate()
            self.proc = None
        if not keep_relay and self.own_relay and self.relay:
            await kubectl("delete", "pod", self.relay, "--wait=false", "--ignore-not-found")
            self.own_relay = False
            self.relay = None


async def _fetch_db(conn: Any, run: str, sequence: list[str] | None) -> dict[str, Any]:
    pattern = f"TEST\\_{run}\\_%"
    names = [s for s in (sequence or ps.PIPELINE_NAMES) if s in TEST_TABLES]
    stages: dict[str, dict[str, list[Any]]] = {}
    for service in names:
        rows = await conn.fetch(
            f"SELECT request_id, status, error_message, extract(epoch FROM created_at) AS c, "  # noqa: S608
            f"extract(epoch FROM updated_at) AS u FROM {TEST_TABLES[service]}_jobs WHERE request_id LIKE $1",
            pattern,
        )
        stages[service] = {
            row["request_id"]: [row["status"], float(row["c"]), float(row["u"]), row["error_message"]] for row in rows
        }
    guard = {
        row["request_id"]: [row["passed"], float(row["c"])]
        for row in await conn.fetch(
            f"SELECT request_id, passed, extract(epoch FROM created_at) AS c FROM {TEST_GUARDRAILS} "  # noqa: S608
            "WHERE request_id LIKE $1",
            pattern,
        )
    }
    dead = await conn.fetchval(
        f"SELECT count(*) FROM {TEST_OUTBOX} WHERE request_id LIKE $1 AND failed_at IS NOT NULL",  # noqa: S608
        pattern,
    )
    return {"stages": stages, "guard": guard, "dead": int(dead or 0), "fetched_at": time.time()}


async def _poll_db(run: str, link: DbLink, sequence: list[str] | None) -> None:
    while True:
        try:
            conn = await link.connect()
            snapshot = await _fetch_db(conn, run, sequence)
            await ctx["redis"].set(_key(run, "db"), json.dumps(snapshot))
            await _update(run, db_note=None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("perftest %s: database tidak terbaca: %s", run, exc)
            await _update(run, db_note=f"{type(exc).__name__}: {str(exc)[:200]}")
            await link.close(keep_relay=True)
        await asyncio.sleep(DB_INTERVAL)


# --- hasil -------------------------------------------------------------------------------------


async def _result(run: str, meta: dict[str, Any], now: float | None = None) -> dict[str, Any]:
    pods = await _json_key(run, "pods", {})
    return ps.compute(
        meta,
        await _samples(run),
        await _json_key(run, "db"),
        await _resources(run),
        pods,
        await _json_key(run, "k6"),
        now or time.time(),
    )


async def _watch_abort(run: str) -> None:
    profile = ps.PROFILES[(await _meta(run) or {}).get("profile", "smoke")]
    while True:
        await asyncio.sleep(ABORT_INTERVAL)
        meta = await _meta(run)
        if meta is None or meta.get("status") != "running" or not meta.get("load_started_at"):
            continue
        now = time.time()
        offset = now - meta["load_started_at"]
        samples = await _samples(run)
        db = await _json_key(run, "db")
        outcomes = ps.request_outcomes(samples, db, meta.get("pipeline_name_sequence"))
        win = ps.windows(samples, outcomes, meta["plan"], meta["load_started_at"], now)
        in_flight = sum(1 for o in outcomes.values() if o["state"] == "in_flight") if db else 0
        reason = ps.should_abort(
            win,
            {**ps.SLO_DEFAULTS, **meta["slo"]},
            auto_abort=bool(meta.get("auto_abort", profile["auto_abort"])),
            now_offset=offset,
            in_flight=in_flight,
            current_rate=ps.target_rate(meta["plan"], offset),
        )
        if reason:
            log.warning("perftest %s dihentikan otomatis: %s", run, reason)
            await _update(run, aborted_reason=reason, status_detail=f"dihentikan otomatis: {reason}")
            await kubectl("delete", "job", _job(run), "--wait=false", "--ignore-not-found")
            return


async def _wait_pod(run: str) -> tuple[str, float | None]:
    """(running | failed: <alasan> | missing, waktu container mulai)."""
    deadline = time.time() + POD_START_TIMEOUT
    while time.time() < deadline:
        code, out, _ = await kubectl("get", "pods", "-l", f"job-name={_job(run)}", "-o", "json")
        pods = json.loads(out).get("items", []) if code == 0 else []
        if not pods:
            code, _, _ = await kubectl("get", "job", _job(run))
            if code != 0:
                return "missing", None
        for pod in pods:
            status = pod.get("status", {})
            for cs in status.get("containerStatuses", []):
                waiting = (cs.get("state") or {}).get("waiting") or {}
                if waiting.get("reason") in ("ErrImagePull", "ImagePullBackOff", "CreateContainerConfigError"):
                    return f"failed: {waiting['reason']}: {waiting.get('message', '')[:300]}", None
                running = (cs.get("state") or {}).get("running") or (cs.get("state") or {}).get("terminated")
                if running:
                    return "running", _rfc3339(running.get("startedAt"))
            for cond in status.get("conditions", []):
                if cond.get("type") == "PodScheduled" and cond.get("status") == "False":
                    await _update(run, status_detail=f"pod belum terjadwal: {cond.get('message', '')[:200]}")
        await asyncio.sleep(3)
    return "failed: pod k6 tidak jalan dalam 4 menit (cluster penuh?)", None


async def _run(run: str) -> None:
    """Satu run dari pod k6 jalan sampai laporan tertulis; juga dipakai untuk menyambung run setelah tracker restart."""
    link = DbLink(run)
    side: list[asyncio.Task[None]] = []
    meta = await _meta(run) or {}
    final_status = "failed"
    try:
        state, started = await _wait_pod(run)
        if state != "running":
            final_status = "failed" if state.startswith("failed") else "interrupted"
            await _update(run, error=state if state != "missing" else "Job tidak ditemukan di cluster")
            return
        meta = await _update(
            run,
            status="running",
            status_detail=None,
            load_started_at=meta.get("load_started_at") or started or time.time(),
        )
        side.append(asyncio.create_task(_sample_resources(run)))
        if any(s in TEST_TABLES for s in (meta.get("pipeline_name_sequence") or ps.PIPELINE_NAMES)):
            side.append(asyncio.create_task(_poll_db(run, link, meta.get("pipeline_name_sequence"))))
        side.append(asyncio.create_task(_watch_abort(run)))

        exit_code = await _follow_logs(run)
        meta = await _update(run, load_finished_at=time.time(), exit_code=exit_code, status="draining")
        # Tunggu pipeline menghabiskan antrean: request yang dijawab 202 masih berjalan setelah k6 berhenti.
        deadline = time.time() + DRAIN_SECONDS
        while time.time() < deadline and await _json_key(run, "db") is not None:
            outcomes = ps.request_outcomes(
                await _samples(run), await _json_key(run, "db"), meta.get("pipeline_name_sequence")
            )
            if not any(o["state"] == "in_flight" for o in outcomes.values()):
                break
            await asyncio.sleep(DB_INTERVAL)
        meta = await _meta(run) or meta
        if meta.get("aborted_reason"):
            final_status = "aborted"
        elif meta.get("stop_requested"):
            final_status = "stopped"
        else:
            # 99: threshold k6 terlewati, run tetap lengkap; penilaiannya ada di laporan.
            final_status = "finished" if exit_code in (0, 99) else "failed"
    except asyncio.CancelledError:
        final_status = "interrupted"
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("perftest %s gagal", run)
        await _update(run, error=f"{type(exc).__name__}: {exc}")
    finally:
        for task in side:
            task.cancel()
        for task in side:
            with contextlib.suppress(BaseException):
                await task
        with contextlib.suppress(Exception):
            # Snapshot terakhir setelah antrean habis, supaya request yang tuntas di detik terakhir ikut terhitung.
            if any(s in TEST_TABLES for s in (meta.get("pipeline_name_sequence") or ps.PIPELINE_NAMES)):
                conn = await link.connect()
                await ctx["redis"].set(
                    _key(run, "db"), json.dumps(await _fetch_db(conn, run, meta.get("pipeline_name_sequence")))
                )
        await link.close()
        await _cleanup_cluster(run)
        await _finalize(run, final_status)


async def _cleanup_cluster(run: str) -> None:
    await kubectl("delete", "job", _job(run), "--wait=false", "--ignore-not-found", "--cascade=background")
    await kubectl("delete", "configmap", _job(run), "--ignore-not-found")


async def _finalize(run: str, status: str) -> None:
    meta = await _meta(run)
    if meta is None:
        return
    meta.update(status=status, finished_at=time.time(), status_detail=None)
    meta["uploads_deleted"] = loadtest._delete_uploads(meta.get("uploads", []))
    meta["log_tail"] = "\n".join(await ctx["redis"].lrange(_key(run, "log"), -40, -1))
    try:
        result = await _result(run, meta, meta.get("load_finished_at") or time.time())
        result["baseline"] = await _baseline(meta)
        await ctx["redis"].set(_key(run, "final"), json.dumps(result, default=str))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"perf-{run}.json").write_text(json.dumps({"meta": meta, "result": result}, default=str), "utf-8")
        (OUT_DIR / f"perf-{run}.pdf").write_bytes(perf_report.render(meta, result))
        meta["report"] = f"perf-{run}.pdf"
        meta["verdict"] = result["verdict"]
    except Exception as exc:  # noqa: BLE001
        log.exception("laporan perftest %s gagal", run)
        meta["report_error"] = f"{type(exc).__name__}: {exc}"
    await _save(meta)
    log.info("perftest %s selesai: %s", run, status)


async def _baseline(meta: dict[str, Any]) -> dict[str, Any] | None:
    """Run selesai sebelumnya dengan profil dan sequence yang sama, untuk perbandingan di laporan."""
    rows = [json.loads(v) for v in (await ctx["redis"].hgetall("ocr:perftests")).values()]
    rows = [
        m
        for m in rows
        if m["run_id"] != meta["run_id"]
        and m.get("profile") == meta["profile"]
        and m.get("pipeline_name_sequence") == meta.get("pipeline_name_sequence")
        and m.get("status") == "finished"
        and m.get("created_at", 0) < meta.get("created_at", 0)
    ]
    if not rows:
        return None
    previous = max(rows, key=lambda m: m.get("created_at", 0))
    final = await _json_key(previous["run_id"], "final")
    if not final:
        return None
    return {
        "run_id": previous["run_id"],
        "created_at": previous.get("created_at"),
        "versions": previous.get("versions"),
        "summary": final["summary"],
        "capacity": final.get("capacity"),
    }


def _spawn(run: str) -> None:
    task = tasks.get(run)
    if task is None or task.done():
        tasks[run] = asyncio.create_task(_run(run))


async def reconcile() -> None:
    """Saat tracker start: run yang masih aktif disambung lagi (Job masih ada) atau diselesaikan dengan data yang
    ada."""
    for raw in (await ctx["redis"].hgetall("ocr:perftests")).values():
        meta = json.loads(raw)
        if meta.get("status") in ACTIVE:
            _spawn(meta["run_id"])


# --- endpoint -------------------------------------------------------------------------------------


_config_cache: dict[str, Any] = {}


@router.get("/config")
async def config() -> dict[str, Any]:
    if not _config_cache or time.time() - _config_cache.get("_at", 0) > 60:
        code, out, err = await kubectl("auth", "can-i", "create", "jobs", timeout=20)
        relay_code, relay, _ = await kubectl("get", "pod", DB_RELAY_POD, "-o", "jsonpath={.status.phase}", timeout=20)
        _config_cache.update(
            _at=time.time(),
            cluster_ok=code == 0 and out.strip() == "yes",
            cluster_error=None if code == 0 else (err or out)[:300],
            db_relay=relay.strip() if relay_code == 0 else None,
        )
    files = [
        {"name": p.name, "size": p.stat().st_size, "upload": p.parent == loadtest.ASSETS_DIR}
        for p in loadtest._image_paths()
    ]
    return {
        "profiles": {
            name: {**{k: v for k, v in p.items()}, "params": {k: ps.PARAM_LABELS[k] for k in p["defaults"]}}
            for name, p in ps.PROFILES.items()
        },
        "slo": ps.SLO_DEFAULTS,
        "slo_labels": ps.SLO_LABELS,
        "context": CONTEXT,
        "namespace": NAMESPACE,
        "release": RELEASE,
        "target": f"{TARGET}{ENDPOINT}",
        "k6_image": K6_IMAGE,
        "max_rate": MAX_RATE,
        "max_seconds": MAX_SECONDS,
        "configmap_limit": CONFIGMAP_LIMIT,
        "image_files": files,
        "cluster_ok": _config_cache["cluster_ok"],
        "cluster_error": _config_cache["cluster_error"],
        "db_relay": _config_cache["db_relay"],
        "db_relay_pod": DB_RELAY_POD,
    }


@router.get("")
async def list_runs() -> list[dict[str, Any]]:
    rows = [json.loads(v) for v in (await ctx["redis"].hgetall("ocr:perftests")).values()]
    rows.sort(key=lambda m: m.get("created_at", 0), reverse=True)
    for meta in rows:
        meta["sent"] = await ctx["redis"].hlen(_key(meta["run_id"], "samples"))
        meta.pop("log_tail", None)
    return rows


@router.post("")
async def start(request: Request) -> dict[str, Any]:
    body = await request.json()
    profile = body.get("profile", "smoke")
    try:
        plan = ps.build_plan(profile, body.get("params") or {}, max_rate=MAX_RATE, max_seconds=MAX_SECONDS)
    except ps.PlanError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    slo = dict(ps.SLO_DEFAULTS)
    for key, value in (body.get("slo") or {}).items():
        if key in slo and value not in (None, ""):
            try:
                slo[key] = float(value)
            except (TypeError, ValueError):
                raise HTTPException(status_code=422, detail=f"SLO {key} harus angka") from None
    sequence = body.get("pipeline_name_sequence") or None
    if sequence is not None and (not isinstance(sequence, list) or not all(n in ps.PIPELINE_NAMES for n in sequence)):
        raise HTTPException(status_code=422, detail=f"pipeline_name_sequence harus dari {', '.join(ps.PIPELINE_NAMES)}")
    available = {p.name: p for p in loadtest._image_paths()}
    names = loadtest.chosen_images(body.get("images"), list(available))
    if not names:
        raise HTTPException(status_code=422, detail="pilih minimal satu file uji")
    paths = [available[n] for n in names]
    raw = sum(p.stat().st_size for p in paths)
    size = configmap_size(paths)
    if size > CONFIGMAP_LIMIT:
        raise HTTPException(
            status_code=422,
            detail=(
                f"file uji {raw // 1024} KB menjadi {size // 1024} KB di ConfigMap (base64 + skrip k6), melebihi "
                f"batas {CONFIGMAP_LIMIT // 1024} KB; pilih lebih sedikit atau file yang lebih kecil"
            ),
        )
    runs = [json.loads(v) for v in (await ctx["redis"].hgetall("ocr:perftests")).values()]
    if any(m.get("status") in ACTIVE for m in runs):
        raise HTTPException(status_code=409, detail="masih ada performance test yang berjalan")
    try:
        info = await _cluster_info()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"cluster tidak terbaca ({CONTEXT}): {exc}") from None

    run = f"pt{secrets.token_hex(3)}"
    uploads = set(loadtest.list_uploads())
    meta = {
        "run_id": run,
        "profile": profile,
        "params": body.get("params") or {},
        "plan": plan,
        "slo": slo,
        "auto_abort": bool(body.get("auto_abort", ps.PROFILES[profile]["auto_abort"])),
        "images": names,
        "uploads": [n for n in names if n in uploads],
        "pipeline_name_sequence": sequence,
        "context": CONTEXT,
        "namespace": NAMESPACE,
        "release": RELEASE,
        "target": f"{TARGET}{ENDPOINT}",
        "k6_image": K6_IMAGE,
        "wait_seconds": info["wait_seconds"],
        "vus": ps.vus_for(ps.plan_peak(plan), info["wait_seconds"]),
        "limits": info["limits"],
        "versions": info["versions"],
        "note": (body.get("note") or "")[:300],
        "created_at": time.time(),
        "status": "starting",
        "status_detail": "membuat ConfigMap dan Job k6",
    }
    try:
        await apply(_configmap(run, paths), verb="create")
        await apply(_job_manifest(meta, info["affinity"]))
    except Exception as exc:  # noqa: BLE001
        await _cleanup_cluster(run)
        raise HTTPException(status_code=502, detail=str(exc)) from None
    await _save(meta)
    _spawn(run)
    log.info("perftest %s mulai: %s, %s dtk, puncak %s rps", run, profile, ps.plan_seconds(plan), ps.plan_peak(plan))
    return meta


@router.get("/{run}")
async def detail(run: str) -> dict[str, Any]:
    meta = await _meta(run)
    if meta is None:
        raise HTTPException(status_code=404, detail="run tidak ditemukan")
    final = await _json_key(run, "final") if meta.get("status") not in ACTIVE else None
    result = final or await _result(run, meta)
    if meta.get("status") in ACTIVE:
        meta["log_tail"] = "\n".join(await ctx["redis"].lrange(_key(run, "log"), -40, -1))
        if meta.get("load_started_at"):
            meta["elapsed_seconds"] = time.time() - meta["load_started_at"]
    return {"meta": meta, "result": result}


@router.post("/{run}/stop")
async def stop(run: str) -> dict[str, Any]:
    meta = await _meta(run)
    if meta is None:
        raise HTTPException(status_code=404, detail="run tidak ditemukan")
    await _update(run, stop_requested=True, status_detail="dihentikan pengguna")
    await kubectl("delete", "job", _job(run), "--wait=false", "--ignore-not-found")
    return {"ok": True}


@router.get("/{run}/report.pdf")
async def report_pdf(run: str) -> FileResponse:
    meta = await _meta(run)
    if meta is None:
        raise HTTPException(status_code=404, detail="run tidak ditemukan")
    path = OUT_DIR / f"perf-{run}.pdf"
    if meta.get("status") in ACTIVE:
        # Laporan sementara selama run berjalan; tidak menimpa laporan akhir.
        path = OUT_DIR / f"perf-{run}-sementara.pdf"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(perf_report.render(meta, await _result(run, meta)))
    elif not path.is_file():
        final = await _json_key(run, "final")
        if final is None:
            raise HTTPException(status_code=404, detail="hasil run belum ada")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(perf_report.render(meta, final))
    stamp = datetime.fromtimestamp(meta.get("created_at", time.time())).strftime("%Y%m%d-%H%M")
    return FileResponse(path, media_type="application/pdf", filename=f"perftest-{meta['profile']}-{stamp}-{run}.pdf")


@router.delete("/{run}")
async def remove(run: str, cleanup_db: bool = False) -> dict[str, Any]:
    meta = await _meta(run)
    if meta is None:
        raise HTTPException(status_code=404, detail="run tidak ditemukan")
    if meta.get("status") in ACTIVE:
        raise HTTPException(status_code=409, detail="run masih berjalan; hentikan dulu")
    deleted = await _cleanup_db(run) if cleanup_db else None
    r = ctx["redis"]
    await r.delete(*[_key(run, s) for s in ("samples", "res", "pods", "db", "k6", "log", "final")])
    await r.hdel("ocr:perftests", run)
    for name in (f"perf-{run}.pdf", f"perf-{run}.json", f"perf-{run}-sementara.pdf"):
        with contextlib.suppress(OSError):
            (OUT_DIR / name).unlink(missing_ok=True)
    return {"ok": True, "db_rows_deleted": deleted}


async def _cleanup_db(run: str) -> dict[str, int]:
    """Hapus baris TEST_<run>_% dari tabel nilam_testing_*. nilam_testing_ocr_results append-only, tidak disentuh."""
    link = DbLink(run)
    pattern = f"TEST\\_{run}\\_%"
    deleted: dict[str, int] = {}
    try:
        conn = await link.connect()
        tables = [f"{t}_results" for t in TEST_TABLES.values()] + [f"{t}_jobs" for t in TEST_TABLES.values()]
        for table in [*tables, TEST_GUARDRAILS, TEST_OUTBOX]:
            try:
                result = await conn.execute(f"DELETE FROM {table} WHERE request_id LIKE $1", pattern)  # noqa: S608
                deleted[table] = int(result.split()[-1])
            except Exception as exc:  # noqa: BLE001 - tabel opsional
                log.warning("cleanup %s dilewati: %s", table, exc)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"database tidak terbaca: {exc}") from None
    finally:
        await link.close()
    return deleted
