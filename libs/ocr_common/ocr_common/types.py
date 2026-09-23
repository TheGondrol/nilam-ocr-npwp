"""The shapes of the data that travels between the stages, as TypedDicts.

They are the Python-side twin of the Pydantic payloads in `ocr_common.pipeline.schemas`: the schemas
validate what arrives over HTTP, these annotate what the engines, services and the pipeline pass
around in memory. At runtime they are plain dicts, so nothing changes in what is stored or sent."""

from typing import Any, NotRequired, TypedDict


class BoundingBox(TypedDict):
    """Upright box around a text line, in pixels of the image the OCR model worked on."""

    x1: float
    y1: float
    x2: float
    y2: float


class OcrBlock(TypedDict):
    """One recognised text line."""

    text: str
    confidence: float
    bbox: NotRequired[BoundingBox | None]
    page: NotRequired[int]


class OcrEngineResult(TypedDict):
    """What an OCR engine (`app/ml/*` of ekstraksi) returns."""

    blocks: list[OcrBlock]
    model: str | None


class OcrResult(OcrEngineResult):
    """The stored result of the OCR stage (`ocr_results.result`), forwarded to structuring and scoring."""

    engine: str
    elapsed_ms: float
    full_text: str


class StructuredField(TypedDict):
    """One named field read from the OCR lines; `value` is None when it was not found."""

    value: str | None
    confidence: float
    source: NotRequired[str | None]
    signals: NotRequired[dict[str, Any] | None]


class StructuringResult(TypedDict):
    """The stored result of the structuring stage: one `StructuredField` per name in `npwp.NPWP_FIELDS`."""

    document_type: str
    fields: dict[str, StructuredField]


class FieldConfidences(TypedDict):
    """Output of the trust model: probability that each extracted field is correct (None = no field)."""

    npwp_confidence: float | None
    name_confidence: float | None


class ScoringResult(FieldConfidences):
    """The stored result of the scoring stage: the confidences plus the exact payload that was scored."""

    payload: NotRequired[dict[str, Any]]


class FinalField(TypedDict):
    """A field of the final result: the value and the OCR score of the line it came from."""

    value: str | None
    confidence: float


class FinalResult(TypedDict):
    """What the pipeline produced for one request: carried by the SCORING callback and used to build
    the orchestrator's `extract-ocr` data."""

    document_type: str
    fields: dict[str, FinalField]
    scoring: FieldConfidences
    guardrails: dict[str, Any] | None


class ContractField(TypedDict):
    """A field in the orchestrator's `extract-ocr` contract: the value and a 0/1 confidence flag."""

    value: str | None
    confidence: int
