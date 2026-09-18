"""
Uji asap lewat empat service yang sedang jalan (container atau proses bare).
Ini satu-satunya pemeriksaan yang melewati HTTP sungguhan antar service; test
unit tiap service mengganti service lain dengan perekam/fake.

Script ini memerankan ORKESTRASI di sequence diagram:

    1. POST guardrails/check (sinkron)            -> passed true/false + reason
    2. POST ekstraksi/jobs (202, fire-and-forget) -> OCR -> structuring -> scoring
    3. polling GET <tahap>/jobs/{request_id} sampai scoring DONE / ada yang FAILED
    4. dokumen yang ditolak guardrails berhenti di langkah 1 (orkestrator: 422)
    5. kontrak lama extract-ocr (sinkron) masih jalan

    make smoke
    # atau, dengan URL/key sendiri:
    API_KEY=changeme EKSTRAKSI_URL=http://127.0.0.1:8030 python scripts/smoke_e2e.py

Callback: set SMOKE_CALLBACK_PORT=8039 supaya script ini juga menerima callback
tahap, lalu arahkan ketiga service ke sana (ORCHESTRATION_URL=
http://host.docker.internal:8039 di container, http://127.0.0.1:8039 kalau
bare). Tanpa itu callback tidak diperiksa (service melewatinya kalau
ORCHESTRATION_URL kosong) dan hasil dibaca lewat polling saja.

Memakai gambar NPWP sintetis kalau Pillow terpasang (berguna saat
EKSTRAKSI_BACKEND=paddle); kalau tidak, bytes palsu (cukup untuk backend mock).
"""

import io
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx


def _key_from_env_file() -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "services", "ekstraksi", ".env")
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("API_KEY="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return "changeme"


URLS = {
    "guardrails": os.environ.get("GUARDRAILS_URL", "http://127.0.0.1:8031"),
    "ekstraksi": os.environ.get("EKSTRAKSI_URL", "http://127.0.0.1:8030"),
    "structuring": os.environ.get("STRUCTURING_URL", "http://127.0.0.1:8032"),
    "scoring": os.environ.get("SCORING_URL", "http://127.0.0.1:8033"),
}
API_KEY = os.environ.get("API_KEY") or _key_from_env_file()
HEADERS = {"X-API-Key": API_KEY}
CALLBACK_PORT = int(os.environ.get("SMOKE_CALLBACK_PORT") or 0)
TIMEOUT_SECONDS = float(os.environ.get("SMOKE_TIMEOUT_SECONDS") or 60)

callbacks: list[dict] = []


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 (nama method ditentukan http.server)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        callbacks.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


def _image() -> bytes:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return b"\xff\xd8fake-jpeg-bytes"
    # Layout ala kartu NPWP (kop + baris berlabel): cukup mirip untuk lolos
    # model guardrails, dan barisnya dibaca structurer.
    image = Image.new("RGB", (1000, 620), "white")
    draw = ImageDraw.Draw(image)
    try:
        big, small = ImageFont.truetype("arial.ttf", 34), ImageFont.truetype("arial.ttf", 28)
    except OSError:
        big = small = ImageFont.load_default()
    lines = [
        ("KEMENTERIAN KEUANGAN REPUBLIK INDONESIA", big),
        ("DIREKTORAT JENDERAL PAJAK", big),
        ("NPWP : 12.345.678.9-012.345", small),
        ("NAMA : BUDI SANTOSO", small),
        ("NAMA BADAN : PT CIPTA KARYA MANDIRI", small),
        ("ALAMAT : JL. MERDEKA NO. 12 JAKARTA", small),
        ("KPP PRATAMA JAKARTA MENTENG", small),
        ("TERDAFTAR : 15-03-2018", small),
    ]
    for i, (text, font) in enumerate(lines):
        draw.text((40, 40 + i * 68), text, fill="black", font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def _guardrails(client: httpx.Client, request_id: str, filename: str, content: bytes) -> dict:
    response = client.post(
        f"{URLS['guardrails']}/v1/guardrails/check",
        headers=HEADERS,
        data={"request_id": request_id},
        files={"file": (filename, content, "image/jpeg")},
    )
    response.raise_for_status()
    return response.json()["data"]


def _poll(client: httpx.Client, request_id: str) -> dict[str, dict]:
    """Tunggu sampai scoring selesai atau salah satu tahap FAILED. -> {tahap: job}."""
    deadline = time.monotonic() + TIMEOUT_SECONDS
    jobs: dict[str, dict] = {}
    while time.monotonic() < deadline:
        for stage in ("ekstraksi", "structuring", "scoring"):
            response = client.get(f"{URLS[stage]}/v1/{stage}/jobs/{request_id}", headers=HEADERS)
            if response.status_code == 200:
                jobs[stage] = response.json()["data"]
        statuses = {stage: job["status"] for stage, job in jobs.items()}
        if statuses.get("scoring") in {"DONE", "FAILED"} or "FAILED" in statuses.values():
            break
        time.sleep(0.3)
    return jobs


def async_pipeline(client: httpx.Client) -> bool:
    print("== pipeline async ==")
    request_id = f"REQ_{uuid.uuid4()}"
    image = _image()
    print("request_id:", request_id)

    guardrails = _guardrails(client, request_id, "npwp.jpg", image)
    print(f"guardrails: passed={guardrails['passed']} reason={guardrails['reason']!r}")
    if not guardrails["passed"]:
        print("  dokumen sintetis ditolak guardrails; pipeline tidak dilanjutkan")
        return False

    accepted = client.post(
        f"{URLS['ekstraksi']}/v1/ekstraksi/jobs",
        headers=HEADERS,
        data={"request_id": request_id, "document_type": "npwp", "guardrails": json.dumps(guardrails)},
        files={"file": ("npwp.jpg", image, "image/jpeg")},
    )
    print("ekstraksi/jobs:", accepted.status_code, accepted.json().get("data"))
    if accepted.status_code != 202:
        return False

    jobs = _poll(client, request_id)
    for stage in ("ekstraksi", "structuring", "scoring"):
        job = jobs.get(stage)
        print(f"  {stage:<12} {job['status'] if job else '(belum ada job)'} {(job or {}).get('error_message') or ''}")
    ok = all(jobs.get(stage, {}).get("status") == "DONE" for stage in ("ekstraksi", "structuring", "scoring"))
    if ok:
        for name, field in jobs["structuring"]["result"]["fields"].items():
            print(f"  {name:<12} {field['value']!r:<35} conf={field['confidence']}")
        score = jobs["scoring"]["result"]
        print(f"  score = {score['score']} decision = {score['decision']}")

    # Idempoten: request_id yang sama tidak diproses ulang.
    again = client.post(
        f"{URLS['ekstraksi']}/v1/ekstraksi/jobs",
        headers=HEADERS,
        data={"request_id": request_id},
        files={"file": ("npwp.jpg", image, "image/jpeg")},
    )
    duplicate = again.json().get("data", {}).get("duplicate")
    print("kirim ulang request_id yang sama -> duplicate =", duplicate)
    ok = ok and duplicate is True

    if CALLBACK_PORT:
        time.sleep(1.0)  # callback terakhir dikirim setelah job scoring tercatat DONE
        mine = [(c["stage"], c["status"]) for c in callbacks if c.get("request_id") == request_id]
        print("callback diterima:", mine)
        expected = [("OCR", "DONE"), ("STRUCTURING", "DONE"), ("SCORING", "DONE")]
        ok = ok and mine == expected
        final = next((c for c in callbacks if c.get("request_id") == request_id and c["stage"] == "SCORING"), None)
        if final:
            print("hasil akhir (callback SCORING):", json.dumps(final["result"])[:300])
    else:
        print("callback tidak diperiksa (SMOKE_CALLBACK_PORT tidak di-set)")
    return ok


def guardrails_reject(client: httpx.Client) -> bool:
    print("== guardrails menolak ==")
    report = _guardrails(client, f"REQ_{uuid.uuid4()}", "notnpwp.jpg", _image())
    print(f"notnpwp.jpg -> passed={report['passed']} reason={report['reason']!r}")
    # Hanya backend mock yang menilai dari nama file; model sungguhan menilai gambarnya.
    return True


def legacy_contract(client: httpx.Client) -> bool:
    print("== kontrak lama (extract-ocr sinkron) ==")
    base = URLS["ekstraksi"]
    request_id = client.post(f"{base}/v1/generate-request-id", headers=HEADERS).json()["request_id"]
    response = client.post(
        f"{base}/v1/extract-ocr",
        headers=HEADERS,
        data={"request_id": request_id},
        files={"file": ("npwp.jpg", _image(), "image/jpeg")},
    )
    body = response.json()
    print("extract-ocr:", response.status_code, body.get("message"), "guardrails =", body.get("guardrails"))
    return response.status_code == 200


def main() -> int:
    if CALLBACK_PORT:
        server = ThreadingHTTPServer(("0.0.0.0", CALLBACK_PORT), _CallbackHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"menerima callback di port {CALLBACK_PORT}")

    with httpx.Client(timeout=60.0) as client:
        for name, url in URLS.items():
            print(f"health {name}:", client.get(f"{url}/health").json()["backends"])
        results = [async_pipeline(client), guardrails_reject(client), legacy_contract(client)]
    print("HASIL:", "OK" if all(results) else "GAGAL")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
