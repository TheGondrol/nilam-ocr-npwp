"""
Business logic kontrak OCR yang dipanggil ocr-orchestration:
generate-request-id -> extract-ocr -> get-ocr-result. Sama perannya dengan
OcrService di mock ocr-npwp, tapi di balik extract() ada pipeline sungguhan:

    guardrails (kualitas gambar, jenis dokumen)   gagal -> 400
    -> ekstraksi (OCR mentah)
    -> structuring (field bernama)
    -> scoring (skor dokumen)                      -> `guardrails` di envelope

Layer ini tidak tahu HTTP: melempar ServiceError, controller yang
menerjemahkannya. Skenario nama file mock (blur/notnpwp/servererror) datang
dari implementasi mock di src/models/, bukan dari sini.
"""

import uuid
from typing import Any

from src.core.errors import ServiceError
from src.models.structuring import DOCUMENT_TYPE, NPWP_FIELDS
from src.repositories.request_repository import InMemoryRequestRepository, SqlRequestRepository
from src.services.ekstraksi_service import EkstraksiService
from src.services.guardrails_service import GuardrailsService
from src.services.scoring_service import ScoringService
from src.services.structuring_service import StructuringService


class OcrService:
    def __init__(
        self,
        repository: InMemoryRequestRepository | SqlRequestRepository,
        guardrails: GuardrailsService,
        ekstraksi: EkstraksiService,
        structuring: StructuringService,
        scoring: ScoringService,
    ):
        self._repository = repository
        self._guardrails = guardrails
        self._ekstraksi = ekstraksi
        self._structuring = structuring
        self._scoring = scoring

    async def generate_request_id(self) -> str:
        request_id = f"OCR_{uuid.uuid4()}"
        await self._repository.create(request_id)
        return request_id

    async def extract(
        self, request_id: str, filename: str, content_type: str | None, content: bytes
    ) -> tuple[dict[str, Any], float]:
        record = await self._repository.get(request_id)
        if record is None:
            raise ServiceError(400, f"Unknown request_id: {request_id}. Call /v1/generate-request-id first.")
        if record["status"] == "completed":
            raise ServiceError(409, f"request_id {request_id} has already been processed")

        try:
            data, guardrails = self._run_pipeline(filename, content_type, content)
        except ServiceError as exc:
            await self._repository.update(request_id, status="failed", error_message=exc.message)
            raise

        await self._repository.update(request_id, status="completed", result=data, guardrails=guardrails)
        return data, guardrails

    def _run_pipeline(self, filename: str, content_type: str | None, content: bytes) -> tuple[dict[str, Any], float]:
        report = self._guardrails.check(filename, content_type, content)
        if not report["passed"]:
            failed = next(check for check in report["checks"] if not check["passed"])
            raise ServiceError(400, _guardrail_message(failed["name"]))

        ocr = self._ekstraksi.extract(filename, content_type, content)
        lines = [{"text": block["text"], "confidence": block["confidence"]} for block in ocr["blocks"]]
        structured = self._structuring.structure(lines)
        score = self._scoring.score(DOCUMENT_TYPE, structured["fields"])

        # Bentuk yang dijanjikan schema NpwpFields: tiap daun {value, confidence}.
        data = {
            name: {"value": field["value"], "confidence": field["confidence"]}
            for name, field in structured["fields"].items()
            if name in NPWP_FIELDS
        }
        return data, score["score"]

    async def get_result(self, request_id: str) -> dict[str, Any]:
        record = await self._repository.get(request_id)
        if record is None:
            raise ServiceError(404, f"No data found for request_id: {request_id}")

        return {
            "request_id": request_id,
            "status": record["status"],
            "result": record["result"],
            "guardrails": record.get("guardrails"),
            "error_message": record["error_message"],
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
        }


# Pesan 400 per check guardrails, disamakan dengan mock ocr-npwp supaya
# orchestrator dan Postman collection-nya tidak melihat kalimat baru.
_GUARDRAIL_MESSAGES = {
    "image_quality": "Image quality too low, NPWP could not be read",
    "document_type": "Uploaded image is not recognized as an NPWP",
}


def _guardrail_message(check_name: str) -> str:
    return _GUARDRAIL_MESSAGES.get(check_name, f"Guardrail check failed: {check_name}")
