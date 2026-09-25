"""HTTP client to a model service or another stage, mapping transport failures and error statuses to
`ServiceError`s, and forwarding the current `X-Request-ID`.
"""

import json
from collections.abc import Collection
from typing import Any

import httpx

from ocr_common.errors import InternalError, ServiceError, UpstreamTimeout, UpstreamUnavailable
from ocr_common.web.request_id import REQUEST_ID_HEADER, current_request_id


class RemoteModelClient:
    """One httpx client per remote service. Timeouts become 504, connection errors 503, and an
    error status becomes 500 with the remote's message, or the same status with the remote's message
    when it is a 4xx and `passthrough_client_errors` is set (a stage relaying another stage's
    validation error), or when it is listed in `passthrough_statuses` (e.g. `(400, 413, 503, 504)`
    to relay a model service's refusals and outages as they are, but not its 401).
    """

    def __init__(
        self,
        base_url: str,
        timeout: float,
        *,
        name: str,
        headers: dict[str, str] | None = None,
        passthrough_client_errors: bool = False,
        passthrough_statuses: Collection[int] = (),
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.name = name
        self._passthrough = passthrough_client_errors
        self._passthrough_statuses = frozenset(passthrough_statuses)
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, headers=headers or {}, transport=transport
        )

    async def get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        """GET `path` and decode the JSON body."""
        return await self._request("GET", path, params=params)

    async def post_multipart(
        self,
        path: str,
        *,
        filename: str,
        content: bytes,
        content_type: str,
        field: str = "file",
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """POST one file as multipart form data, with optional form fields and query params."""
        files = {field: (filename, content, content_type)}
        return await self._request("POST", path, files=files, data=data, params=params)

    async def post_form(self, path: str, *, data: dict[str, Any], params: dict[str, Any] | None = None) -> Any:
        """A form POST without a file, e.g. a job submitted with `file_url` instead of `file`."""
        return await self._request("POST", path, data=data, params=params)

    async def post_json(self, path: str, payload: Any, *, headers: dict[str, str] | None = None) -> Any:
        """POST a JSON body and decode the JSON answer."""
        return await self._request("POST", path, json=payload, headers=headers)

    async def aclose(self) -> None:
        """Close the connection pool; call once at shutdown."""
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        request_id = current_request_id()
        if request_id:
            kwargs["headers"] = {REQUEST_ID_HEADER: request_id, **(kwargs.get("headers") or {})}
        try:
            response = await self._client.request(method, path, **kwargs)
        except (httpx.ReadTimeout, httpx.WriteTimeout) as exc:
            raise UpstreamTimeout(f"{self.name} timed out after {self.timeout}s") from exc
        except httpx.RequestError as exc:
            raise UpstreamUnavailable(f"{self.name} is unavailable") from exc

        if response.status_code >= 400:
            detail = _error_detail(response)
            if response.status_code in self._passthrough_statuses or (
                self._passthrough and 400 <= response.status_code < 500
            ):
                raise ServiceError(response.status_code, detail)
            raise InternalError(f"{self.name} error ({response.status_code}): {detail}")
        try:
            return response.json()
        except ValueError as exc:
            raise InternalError(f"{self.name} returned an invalid response") from exc


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200] or response.reason_phrase
    if not isinstance(body, dict):
        return json.dumps(body)[:200]
    detail = body.get("detail")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        parts = [
            f"{'.'.join(str(loc) for loc in item.get('loc', []))}: {item.get('msg')}"
            for item in detail
            if isinstance(item, dict)
        ]
        if parts:
            return "; ".join(parts)
    message = body.get("message")
    if isinstance(message, str) and message:
        return message
    return json.dumps(body)[:200]
