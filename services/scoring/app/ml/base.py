from typing import Any, Protocol


class Scorer(Protocol):
    """Legacy document scorer behind POST /v1/scoring/score: one score for the whole document from the
    structured fields, plus per-field scores and the reasons a field was penalised.

    The pipeline callback uses the trust model (app/ml/trust_model.py) instead, which is not a backend.
    """

    name: str

    @property
    def supported_document_types(self) -> tuple[str, ...]: ...

    def score(self, fields: dict[str, dict[str, Any]]) -> dict[str, Any]: ...
