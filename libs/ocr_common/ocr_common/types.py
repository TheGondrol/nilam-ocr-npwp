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


class StructuredDocument(TypedDict):
    """What a structurer (`app/ml/*` of structuring) returns: one `StructuredField` per name in
    `npwp.NPWP_FIELDS`, the document-level flag of the ML team's rules, and the rejection it implies.

    `flag` / `flag_reason` is raised by any of the rules' checks and is an input of the trust model.
    `reject_reason` is set when one of the checks that reject the document fired (everything but a
    single-word name and a letter in the number): the pipeline stops at structuring and the client
    gets a 400 with that message. It is the first rejecting check in the rules' priority order, which
    is not always `flag_reason` (a single-word name outranks the invalid-code checks there)."""

    fields: dict[str, StructuredField]
    flag: bool
    flag_reason: str | None
    reject_reason: NotRequired[str | None]


class StructuringResult(StructuredDocument):
    """The stored result of the structuring stage: the structured document plus its document type."""

    document_type: str


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
    flag: bool
    flag_reason: str | None


class ContractField(TypedDict):
    """A field in the orchestrator's `extract-ocr` contract: the value and a 0/1 confidence flag."""

    value: str | None
    confidence: int


class ContractData(TypedDict):
    """`data` of the orchestrator's `extract-ocr` contract. The rules' flag is internal (an input of the
    trust model) and is not part of it."""

    nomor_npwp: ContractField
    nama: ContractField
