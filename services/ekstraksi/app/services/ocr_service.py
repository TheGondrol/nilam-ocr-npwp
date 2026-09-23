import uuid
from typing import Any

from ocr_common.errors import BadRequest, Conflict, NotFound, ServiceError
from ocr_common.npwp import DOCUMENT_TYPE, NPWP_FIELDS

from app.clients.stages import StageClients
from app.repositories.request_repository import InMemoryRequestRepository, SqlRequestRepository
from app.services.ekstraksi_service import EkstraksiService


class OcrService:
    def __init__(
        self,
        repository: InMemoryRequestRepository | SqlRequestRepository,
        ekstraksi: EkstraksiService,
        stages: StageClients,
    ):
        self._repository = repository
        self._ekstraksi = ekstraksi
        self._stages = stages

    async def generate_request_id(self) -> str:
        request_id = f"OCR_{uuid.uuid4()}"
        await self._repository.create(request_id)
        return request_id

    async def extract(
        self, request_id: str, filename: str, content_type: str | None, content: bytes
    ) -> tuple[dict[str, Any], float]:
        record = await self._repository.get(request_id)
        if record is None:
            raise BadRequest(f"Unknown request_id: {request_id}. Call /v1/generate-request-id first.")
        if record["status"] == "completed":
            raise Conflict(f"request_id {request_id} has already been processed")

        try:
            data, guardrails = await self._run_pipeline(request_id, filename, content_type, content)
        except ServiceError as exc:
            await self._repository.update(request_id, status="failed", error_message=exc.message)
            raise

        await self._repository.update(request_id, status="completed", result=data, guardrails=guardrails)
        return data, guardrails

    async def _run_pipeline(
        self, request_id: str, filename: str, content_type: str | None, content: bytes
    ) -> tuple[dict[str, Any], float]:
        report = await self._stages.guardrails.check(request_id, filename, content_type, content)
        document = report.get("document") or {}
        if document.get("verdict") != "accepted":
            raise BadRequest(_guardrail_message(document))

        ocr = await self._ekstraksi.extract(filename, content_type, content)
        lines = [{"text": block["text"], "confidence": block["confidence"]} for block in ocr["blocks"]]
        structured = await self._stages.structuring.structure(lines)
        score = await self._stages.scoring.score(DOCUMENT_TYPE, structured["fields"])

        data = {
            name: {"value": field.get("value"), "confidence": field.get("confidence", 0.0)}
            for name, field in structured["fields"].items()
            if name in NPWP_FIELDS
        }
        return data, float(score["score"])

    async def get_result(self, request_id: str) -> dict[str, Any]:
        record = await self._repository.get(request_id)
        if record is None:
            raise NotFound(f"No data found for request_id: {request_id}")

        return {
            "request_id": request_id,
            "status": record["status"],
            "result": record["result"],
            "guardrails": record.get("guardrails"),
            "error_message": record["error_message"],
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
        }


def _guardrail_message(document: dict[str, Any]) -> str:
    n_reject, n_pages = document.get("n_reject", "?"), document.get("n_pages", "?")
    confidence = document.get("confidence")
    suffix = f" (confidence {confidence:.2f})" if isinstance(confidence, int | float) else ""
    return f"Document rejected by guardrails: {n_reject}/{n_pages} page(s) rejected{suffix}"
