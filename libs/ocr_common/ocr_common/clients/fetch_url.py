import asyncio
import http.client
import ipaddress
import logging
import socket
import ssl
import time
import urllib.parse
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = {"http", "https"}
_CHUNK_BYTES = 64 * 1024
_TLS = ssl.create_default_context()


class FetchUrlError(Exception):
    pass


@dataclass(frozen=True)
class UrlPolicy:
    allowed_hosts: tuple[str, ...] = ()
    allow_private: bool = False

    def host_allowed(self, host: str) -> bool:
        if not self.allowed_hosts:
            return True
        host = host.lower().rstrip(".")
        return any(host == entry or (entry.startswith(".") and host.endswith(entry)) for entry in self.allowed_hosts)

    def address_allowed(self, address: str) -> bool:
        if self.allow_private:
            return True
        ip = ipaddress.ip_address(address.split("%", 1)[0])
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified or ip.is_reserved:
            return False
        return ip.is_global or bool(self.allowed_hosts)


STRICT_URL_POLICY = UrlPolicy()


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, port: int, *, address: str, timeout: float):
        super().__init__(host, port, timeout=timeout)
        self._address = address

    def connect(self) -> None:
        self.sock = socket.create_connection((self._address, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, port: int, *, address: str, timeout: float):
        super().__init__(host, port, timeout=timeout, context=_TLS)
        self._address = address

    def connect(self) -> None:
        sock = socket.create_connection((self._address, self.port), self.timeout)
        self.sock = _TLS.wrap_socket(sock, server_hostname=self.host)


def _resolve(host: str, port: int, policy: UrlPolicy) -> str:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchUrlError("Could not fetch file_url: host could not be resolved") from exc
    addresses = [str(info[4][0]) for info in infos]
    blocked = [address for address in addresses if not policy.address_allowed(address)]
    if blocked or not addresses:
        logger.warning("file_url host %s refused: resolves to %s", host, ", ".join(blocked) or "nothing")
        raise FetchUrlError(f"file_url host is not allowed: {host}")
    return addresses[0]


def _download(
    parsed: urllib.parse.SplitResult, port: int, policy: UrlPolicy, limit: int, timeout: float
) -> tuple[bytes, str | None]:
    host = parsed.hostname or ""
    address = _resolve(host, port, policy)
    connection_class = _PinnedHTTPSConnection if parsed.scheme == "https" else _PinnedHTTPConnection
    connection = connection_class(host, port, address=address, timeout=timeout)
    target = (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")
    deadline = time.monotonic() + timeout
    try:
        connection.request("GET", target, headers={"Accept": "*/*", "User-Agent": "nilam-ocr-npwp"})
        response = connection.getresponse()
        if 300 <= response.status < 400:
            raise FetchUrlError(f"Could not fetch file_url: redirects are not followed ({response.status})")
        if response.status >= 400:
            raise FetchUrlError(f"Could not fetch file_url: {response.status} {response.reason}")
        chunks: list[bytes] = []
        size = 0
        while size <= limit:
            if time.monotonic() > deadline:
                raise TimeoutError("file_url download exceeded its deadline")
            chunk = response.read(min(_CHUNK_BYTES, limit + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        return b"".join(chunks), response.getheader("Content-Type")
    except (OSError, http.client.HTTPException) as exc:
        logger.warning("file_url %s://%s fetch failed: %r", parsed.scheme, host, exc)
        raise FetchUrlError("Could not fetch file_url: connection failed or timed out") from exc
    finally:
        connection.close()


async def fetch(
    url: str, *, limit: int, timeout: float = 10.0, policy: UrlPolicy = STRICT_URL_POLICY
) -> tuple[bytes, str, str]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise FetchUrlError(f"Unsupported URL scheme: {parsed.scheme or '(none)'}")
    host = parsed.hostname
    if not host:
        raise FetchUrlError("file_url has no host")
    if not policy.host_allowed(host):
        raise FetchUrlError(f"file_url host is not allowed: {host}")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise FetchUrlError("file_url has an invalid port") from exc

    content, header_type = await asyncio.to_thread(_download, parsed, port, policy, limit, timeout)

    filename = parsed.path.rsplit("/", 1)[-1] or "download"
    content_type = header_type or ""
    if content_type.split(";")[0].strip() in ("", "application/octet-stream", "binary/octet-stream"):
        lowered = filename.lower()
        if lowered.endswith(".pdf"):
            content_type = "application/pdf"
        else:
            content_type = "image/png" if lowered.endswith(".png") else "image/jpeg"
    return content, filename, content_type.split(";")[0].strip()
