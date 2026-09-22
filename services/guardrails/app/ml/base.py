from typing import Any, Protocol

from PIL import Image

PagePrediction = tuple[float, float]
"""(proba_approve, proba_reject) of one page."""


class PageClassifier(Protocol):
    """A model that runs in this process: every page is rendered to an image and classified here.
    GuardrailsService turns the per-page probabilities into the document verdict."""

    name: str
    reject_threshold: float
    metadata: dict[str, Any]

    def classify(self, filename: str, pages: list[Image.Image]) -> list[PagePrediction]: ...


class DocumentChecker(Protocol):
    """A model served elsewhere (the ML team's model service): the whole document is sent and the
    verdict comes back already aggregated."""

    name: str

    async def check_document(self, filename: str, content: bytes, content_type: str | None) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...


Classifier = PageClassifier | DocumentChecker
