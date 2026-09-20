import json
from typing import Any

import httpx

from ocr_common.errors import ServiceError


class RemoteModelClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        *,
        name: str,
        headers: dict[str, str] | None = None,
        passthrough_client_errors: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.name = name
        self._passthrough = passthrough_client_errors
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout, headers=headers or {}, transport=transport
        )

    async def get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
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
        files = {field: (filename, content, content_type)}
        return await self._request("POST", path, files=files, data=data, params=params)

    async def post_json(self, path: str, payload: Any, *, headers: dict[str, str] | None = None) -> Any:
        return await self._request("POST", path, json=payload, headers=headers)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = await self._client.request(method, path, **kwargs)
        except (httpx.ReadTimeout, httpx.WriteTimeout) as exc:
            raise ServiceError(504, f"{self.name} timed out after {self.timeout}s") from exc
        except httpx.RequestError as exc:
            raise ServiceError(503, f"{self.name} is unavailable") from exc

        if response.status_code >= 400:
            detail = _error_detail(response)
            if self._passthrough and 400 <= response.status_code < 500:
                raise ServiceError(response.status_code, detail)
            raise ServiceError(500, f"{self.name} error ({response.status_code}): {detail}")
        try:
            return response.json()
        except ValueError as exc:
            raise ServiceError(500, f"{self.name} returned an invalid response") from exc


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
