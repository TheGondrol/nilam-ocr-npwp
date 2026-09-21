from typing import Any

from src.clients.ekstraksi import EkstraksiJobClient
from src.services.guardrails_service import GuardrailsService


class GuardrailsJobService:
    def __init__(self, guardrails: GuardrailsService, ekstraksi: EkstraksiJobClient):
        self._guardrails = guardrails
        self._ekstraksi = ekstraksi

    async def submit(
        self,
        request_id: str,
        document_type: str,
        filename: str,
        content_type: str | None,
        content: bytes,
        *,
        handoff: bool = True,
    ) -> dict[str, Any]:
        report = await self._guardrails.check(filename, content_type, content)
        job = None
        if report["passed"] and handoff:
            job = await self._ekstraksi.submit(request_id, document_type, report, filename, content_type, content)
        return {**report, "job": job}
