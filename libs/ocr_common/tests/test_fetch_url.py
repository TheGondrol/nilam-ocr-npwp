import http.server
import socket
import threading
import time

import pytest

from ocr_common.clients.fetch_url import STRICT_URL_POLICY, FetchUrlError, UrlPolicy, _resolve, fetch

LOCAL = UrlPolicy(allow_private=True)
BODY = b"x" * 100


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/missing":
            self.send_response(404)
            self.end_headers()
            return
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            self.end_headers()
            return
        if self.path == "/slow":
            time.sleep(2)
        body = self.headers["Host"].encode() if self.path == "/host" else BODY
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="module")
def port():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_port
    server.shutdown()


def _fake_dns(monkeypatch, mapping: dict[str, list[str]]) -> list[str]:
    real = socket.getaddrinfo
    looked_up: list[str] = []

    def getaddrinfo(host, port, *args, **kwargs):
        if host not in mapping:
            return real(host, port, *args, **kwargs)
        looked_up.append(host)
        family = {4: socket.AF_INET, 6: socket.AF_INET6}
        return [(family[6 if ":" in ip else 4], socket.SOCK_STREAM, 6, "", (ip, port)) for ip in mapping[host]]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return looked_up


async def test_downloads_and_guesses_content_type_from_the_extension(port):
    content, filename, content_type = await fetch(f"http://127.0.0.1:{port}/bench/npwp.jpg", limit=1000, policy=LOCAL)
    assert (content, filename, content_type) == (BODY, "npwp.jpg", "image/jpeg")
    assert (await fetch(f"http://127.0.0.1:{port}/a.png", limit=1000, policy=LOCAL))[2] == "image/png"
    assert (await fetch(f"http://127.0.0.1:{port}/a.pdf", limit=1000, policy=LOCAL))[2] == "application/pdf"


async def test_reads_at_most_one_byte_over_the_limit(port):
    content, _, _ = await fetch(f"http://127.0.0.1:{port}/a.jpg", limit=10, policy=LOCAL)
    assert len(content) == 11


@pytest.mark.parametrize(
    ("path", "message"),
    [("/missing", "404"), ("/redirect", r"redirects are not followed \(302\)")],
)
async def test_error_statuses_and_redirects_are_refused(port, path, message):
    with pytest.raises(FetchUrlError, match=message):
        await fetch(f"http://127.0.0.1:{port}{path}", limit=1000, policy=LOCAL)


async def test_unsupported_scheme_is_refused():
    with pytest.raises(FetchUrlError, match="Unsupported URL scheme: file"):
        await fetch("file:///etc/passwd", limit=1000)


async def test_connection_errors_do_not_leak_details():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    with pytest.raises(FetchUrlError) as exc:
        await fetch(f"http://127.0.0.1:{closed_port}/a.jpg", limit=1000, policy=LOCAL)
    assert str(exc.value) == "Could not fetch file_url: connection failed or timed out"


async def test_slow_download_is_bounded_by_the_timeout(port):
    started = time.monotonic()
    with pytest.raises(FetchUrlError, match="connection failed or timed out"):
        await fetch(f"http://127.0.0.1:{port}/slow", limit=1000, timeout=0.5, policy=LOCAL)
    assert time.monotonic() - started < 1.9


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:{port}/a.jpg",
        "http://localhost:{port}/a.jpg",
        "http://[::1]:{port}/a.jpg",
        "http://[::ffff:127.0.0.1]:{port}/a.jpg",
        "http://0.0.0.0:{port}/a.jpg",
        "http://169.254.169.254/computeMetadata/v1/",
        "http://10.0.0.5/a.jpg",
        "http://192.168.1.10/a.jpg",
    ],
)
async def test_internal_addresses_are_refused_without_being_contacted(port, url, monkeypatch):
    def no_connect(*args, **kwargs):
        raise AssertionError("must not connect")

    monkeypatch.setattr(socket, "create_connection", no_connect)
    with pytest.raises(FetchUrlError, match="file_url host is not allowed"):
        await fetch(url.format(port=port), limit=1000, policy=STRICT_URL_POLICY)


def test_host_allowlist_matches_exact_hosts_and_dot_suffixes():
    policy = UrlPolicy(allowed_hosts=("minio.internal", ".bri.co.id"))
    assert policy.host_allowed("minio.internal")
    assert policy.host_allowed("MINIO.internal.")
    assert policy.host_allowed("files.bri.co.id")
    assert not policy.host_allowed("evilbri.co.id")
    assert not policy.host_allowed("bri.co.id")
    assert not policy.host_allowed("minio.internal.evil.com")


async def test_host_outside_the_allowlist_is_refused_before_dns():
    policy = UrlPolicy(allowed_hosts=("minio.internal",))
    with pytest.raises(FetchUrlError, match="file_url host is not allowed: evil.example"):
        await fetch("http://evil.example/a.jpg", limit=1000, policy=policy)


@pytest.mark.parametrize(
    ("addresses", "allowed"),
    [(["10.1.2.3"], True), (["169.254.169.254"], False), (["127.0.0.1"], False), (["10.1.2.3", "::1"], False)],
)
def test_allowlisted_host_may_be_private_but_never_loopback_or_link_local(monkeypatch, addresses, allowed):
    _fake_dns(monkeypatch, {"minio.internal": addresses})
    policy = UrlPolicy(allowed_hosts=("minio.internal",))
    if allowed:
        assert _resolve("minio.internal", 80, policy) == addresses[0]
    else:
        with pytest.raises(FetchUrlError, match="not allowed"):
            _resolve("minio.internal", 80, policy)


@pytest.mark.parametrize(
    ("addresses", "allowed"),
    [(["93.184.216.34"], True), (["10.0.0.5"], False), (["93.184.216.34", "10.0.0.5"], False)],
)
def test_without_allowlist_only_public_addresses_are_allowed(monkeypatch, addresses, allowed):
    _fake_dns(monkeypatch, {"files.example": addresses})
    if allowed:
        assert _resolve("files.example", 80, STRICT_URL_POLICY) == addresses[0]
    else:
        with pytest.raises(FetchUrlError, match="not allowed"):
            _resolve("files.example", 80, STRICT_URL_POLICY)


async def test_connection_goes_to_the_checked_address_with_the_original_host(port, monkeypatch):
    looked_up = _fake_dns(monkeypatch, {"files.example": ["127.0.0.1"]})
    content, _, _ = await fetch(f"http://files.example:{port}/host", limit=1000, policy=LOCAL)
    assert content == f"files.example:{port}".encode()
    assert looked_up == ["files.example"]
