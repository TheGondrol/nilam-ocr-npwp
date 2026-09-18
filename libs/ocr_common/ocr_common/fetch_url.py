import asyncio
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional, Tuple

ALLOWED_SCHEMES = {"http", "https"}


class FetchUrlError(Exception):
    """Gagal mengambil dokumen dari URL; pesannya ikut ke klien apa adanya."""


def _read(url: str, limit: int, timeout: float) -> Tuple[bytes, Optional[str]]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        # limit + 1: kelebihan satu byte sudah cukup membuktikan file terlalu
        # besar, tanpa menarik sisanya ke memori.
        return response.read(limit + 1), response.headers.get("Content-Type")


async def fetch(url: str, *, limit: int, timeout: float = 10.0) -> Tuple[bytes, str, str]:
    """Kembalikan (content, filename, content_type). Raise FetchUrlError kalau gagal."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise FetchUrlError(f"Unsupported URL scheme: {parsed.scheme or '(none)'}")

    try:
        content, header_type = await asyncio.to_thread(_read, url, limit, timeout)
    except urllib.error.HTTPError as exc:
        raise FetchUrlError(f"Could not fetch file_url: {exc.code} {exc.reason}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FetchUrlError(f"Could not fetch file_url: {exc}")

    filename = parsed.path.rsplit("/", 1)[-1] or "download"
    # Presigned URL sering dilayani sebagai application/octet-stream; tebak
    # dari ekstensi supaya pemeriksaan content type di service tetap berguna.
    content_type = header_type or ""
    if content_type.split(";")[0].strip() in ("", "application/octet-stream", "binary/octet-stream"):
        lowered = filename.lower()
        if lowered.endswith(".pdf"):
            content_type = "application/pdf"  # tambahan dari salinan ocr-*: model OCR membaca PDF
        else:
            content_type = "image/png" if lowered.endswith(".png") else "image/jpeg"
    return content, filename, content_type.split(";")[0].strip()


if __name__ == "__main__":  # python -m src.core.fetch_url
    # Self-check: server lokal sekali pakai, tidak butuh pytest.
    import http.server
    import threading

    body = b"x" * 100

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(404 if self.path == "/missing" else 200)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"

    content, filename, content_type = asyncio.run(fetch(f"{base}/bench/npwp.jpg", limit=1000))
    assert content == body, len(content)
    assert filename == "npwp.jpg", filename
    # octet-stream ditebak dari ekstensi, kalau tidak service menolaknya 400.
    assert content_type == "image/jpeg", content_type
    assert asyncio.run(fetch(f"{base}/a.png", limit=1000))[2] == "image/png"

    # Batas: limit + 1 byte terbaca, cukup untuk membuktikan kelebihan ukuran.
    assert len(asyncio.run(fetch(f"{base}/a.jpg", limit=10))[0]) == 11

    for bad, expect in ((f"{base}/missing", "404"), ("file:///etc/passwd", "scheme")):
        try:
            asyncio.run(fetch(bad, limit=1000))
        except FetchUrlError as exc:
            assert expect in str(exc), exc
        else:
            raise AssertionError(f"{bad} should have failed")

    server.shutdown()
    print("ok")
