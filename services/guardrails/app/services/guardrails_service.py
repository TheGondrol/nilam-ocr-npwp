from typing import Any

from starlette.concurrency import run_in_threadpool

from ocr_common.image_validation import validate_image

from app.clients.reject_threshold import RejectThreshold, default_threshold
from app.config import Settings
from app.services.pages import check_page_count, render_pages

VERDICT_ACCEPTED = "accepted"
VERDICT_REJECT = "reject"


def _reject_reason(document: dict[str, Any]) -> str:
    return (
        f"Document rejected by guardrails: {document['n_reject']}/{document['n_pages']} page(s) rejected "
        f"(confidence {document['confidence']:.2f})"
    )


class GuardrailsService:
    """The entry checks of the pipeline, in order: type / empty / size (413 above `MAX_UPLOAD_BYTES`), page
    count (400 above `GUARDRAILS_MAX_DOCUMENT_PAGES`), then the guardrails model's verdict."""

    def __init__(self, classifier, settings: Settings, threshold: RejectThreshold | None = None):
        """Without `threshold`, the default one (GUARDRAILS_REJECT_THRESHOLD, else the checkpoint's)."""
        self._classifier = classifier
        self._settings = settings
        self._threshold = threshold or RejectThreshold(
            None, "", default_threshold(settings.guardrails_reject_threshold, classifier), cache_seconds=0
        )

    async def check(self, filename: str, content_type: str | None, content: bytes) -> dict[str, Any]:
        validate_image(content_type, content, self._settings)

        if hasattr(self._classifier, "check_document"):
            # The model service renders the PDF itself, so the page count is only known from its answer.
            report = await self._classifier.check_document(filename, content, content_type)
            check_page_count(int(report["document"]["n_pages"]), self._settings.guardrails_max_document_pages)
        else:
            threshold = await self._threshold.get()
            report = await run_in_threadpool(self._check_locally, filename, content_type, content, threshold)

        passed = report["document"]["verdict"] == VERDICT_ACCEPTED
        return {"passed": passed, "reason": None if passed else _reject_reason(report["document"]), **report}

    def _check_locally(
        self, filename: str, content_type: str | None, content: bytes, threshold: float
    ) -> dict[str, Any]:
        pages = render_pages(
            content_type,
            content,
            dpi=self._settings.guardrails_pdf_dpi,
            max_pages=self._settings.guardrails_max_pages,
            max_document_pages=self._settings.guardrails_max_document_pages,
        )
        predictions = self._classifier.classify(filename, pages)

        page_results = [
            {
                "page_index": index,
                "proba_approve": proba_approve,
                "proba_reject": proba_reject,
                "verdict": VERDICT_REJECT if proba_reject >= threshold else VERDICT_ACCEPTED,
            }
            for index, (proba_approve, proba_reject) in enumerate(predictions)
        ]
        # The threshold in the report: it can change at the orchestrator, so a verdict records the one it used.
        return {"document": {**self._aggregate(page_results), "reject_threshold": threshold}, "pages": page_results}

    def _aggregate(self, pages: list[dict[str, Any]]) -> dict[str, Any]:
        n_pages = len(pages)
        n_reject = sum(1 for page in pages if page["verdict"] == VERDICT_REJECT)
        n_approve = n_pages - n_reject

        if self._settings.guardrails_document_policy == "majority":
            accepted = n_approve > n_reject
        else:
            accepted = n_pages > 0 and n_reject == 0

        if accepted:
            confidence = min(page["proba_approve"] for page in pages)
        else:
            rejected = [page["proba_reject"] for page in pages if page["verdict"] == VERDICT_REJECT] or [0.0]
            confidence = max(rejected)

        return {
            "verdict": VERDICT_ACCEPTED if accepted else VERDICT_REJECT,
            "confidence": round(confidence, 4),
            "n_pages": n_pages,
            "n_approve": n_approve,
            "n_reject": n_reject,
        }
