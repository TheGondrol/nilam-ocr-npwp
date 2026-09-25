from collections.abc import Mapping, Sequence
from typing import Any

from starlette.concurrency import run_in_threadpool

from ocr_common.errors import UnprocessableEntity
from ocr_common.npwp import DOCUMENT_TYPE
from ocr_common.pipeline import STRUCTURING, HandoffPayload, StagePipeline, Work, chain, stored
from ocr_common.pipeline.results import StageResults, load_upstream
from ocr_common.types import OcrBlock, StructuringResult

from app.services.structuring_service import StructuringService

Handoff = HandoffPayload


def _rejection(structuring: Mapping[str, Any]) -> str | None:
    """A rejecting check of the ML team's rules stops the pipeline here: no scoring, a 400 for the client."""
    return structuring.get("reject_reason") or None


class StructuringJobService:
    def __init__(
        self,
        pipeline: StagePipeline,
        structuring: StructuringService,
        *,
        results: StageResults | None = None,
        handoff_by_reference: bool = False,
    ):
        self._pipeline = pipeline
        self._structuring = structuring
        self._results = results
        self._handoff_by_reference = handoff_by_reference

    async def submit(
        self,
        request_id: str,
        document_type: str,
        guardrails: dict[str, Any] | None,
        ocr: dict[str, Any] | None,
        sequence: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """`sequence` (pipeline_name_sequence, None = the full pipeline) decides whether the job is handed to
        scoring or ends here with the structuring result as the answer. A rejection stops it either way."""
        if ocr is None and self._results is None:
            raise UnprocessableEntity(
                "ocr is missing: the request refers to the OCR result by request_id, but this service has no "
                "DATABASE_URL to read ocr_results from",
            )
        work, handoff = self._spec(request_id, document_type, guardrails, ocr, sequence)
        return await self._pipeline.submit(
            request_id,
            work,
            **chain(sequence, STRUCTURING, handoff),
            rejection=_rejection,
            input={
                "document_type": document_type,
                "guardrails": guardrails,
                "pipeline_name_sequence": stored(sequence),
            },
        )

    async def resume(self, request_id: str, input: dict[str, Any] | None) -> None:
        """Run again a job a dead process left `PROCESSING`: the OCR result is read from the database,
        the rest comes from the `input` stored when the job was claimed."""
        input = input or {}
        sequence = input.get("pipeline_name_sequence")
        work, handoff = self._spec(
            request_id, input.get("document_type") or DOCUMENT_TYPE, input.get("guardrails"), None, sequence
        )
        await self._pipeline.resume(request_id, work, **chain(sequence, STRUCTURING, handoff), rejection=_rejection)

    async def get(self, request_id: str) -> dict[str, Any]:
        return await self._pipeline.get(request_id)

    def _spec(
        self,
        request_id: str,
        document_type: str,
        guardrails: dict[str, Any] | None,
        ocr: dict[str, Any] | None,
        sequence: Sequence[str] | None = None,
    ) -> tuple[Work, Handoff]:
        upstream: dict[str, Any] = {}

        async def work() -> StructuringResult:
            upstream["ocr"] = ocr if ocr is not None else await load_upstream(self._results, "ocr", request_id)
            lines: list[OcrBlock] = [
                {
                    "text": block.get("text") or "",
                    "confidence": block.get("confidence", 1.0),
                    "bbox": block.get("bbox"),
                    "page": block.get("page", 0),
                }
                for block in upstream["ocr"].get("blocks") or []
            ]
            return await run_in_threadpool(self._structuring.structure, lines)

        def handoff(structuring: Mapping[str, Any]) -> dict[str, Any]:
            body: dict[str, Any] = {"request_id": request_id, "document_type": document_type, "guardrails": guardrails}
            if sequence:
                body["pipeline_name_sequence"] = stored(sequence)
            if not self._handoff_by_reference:
                body.update(ocr=upstream["ocr"], structuring=dict(structuring))
            return body

        return work, handoff
