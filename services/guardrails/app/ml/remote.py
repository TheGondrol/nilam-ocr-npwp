"""The `remote` backend: the ML team's guardrails model served over HTTP. The document is sent whole
and the response already carries the per-page and document verdicts."""

from typing import Any

from ocr_common.clients.remote import RemoteModelClient
from ocr_common.errors import ServiceError


class RemoteGuardrailsModel:
    name = "remote"
    PREDICT_PATH = "/v1/predict/json"

    def __init__(self, client: RemoteModelClient):
        self._client = client

    async def check_document(self, filename: str, content: bytes, content_type: str | None) -> dict[str, Any]:
        body = await self._client.post_multipart(
            self.PREDICT_PATH,
            filename=filename or "upload",
            content=content,
            content_type=content_type or "image/jpeg",
        )
        return parse_report(body, self._client.name)

    async def aclose(self) -> None:
        await self._client.aclose()


def _verdict(value: Any) -> str:
    if value not in ("accepted", "reject"):
        raise ValueError(f"unknown verdict: {value!r}")
    return value


def parse_report(body: Any, name: str) -> dict[str, Any]:
    """The model service's {data: {document, pages}} into the report shape GuardrailsService returns."""
    try:
        data = body["data"]
        document = data["document"]
        return {
            "document": {
                "verdict": _verdict(document["verdict"]),
                "confidence": float(document["confidence"]),
                "n_pages": int(document["n_pages"]),
                "n_approve": int(document["n_approve"]),
                "n_reject": int(document["n_reject"]),
            },
            "pages": [
                {
                    "page_index": int(page["page_index"]),
                    "proba_approve": float(page["proba_approve"]),
                    "proba_reject": float(page["proba_reject"]),
                    "verdict": _verdict(page["verdict"]),
                }
                for page in data["pages"]
            ],
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ServiceError(500, f"{name} returned an unexpected response") from exc
