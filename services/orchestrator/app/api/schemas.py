from typing import Any, Literal

from pydantic import BaseModel, Field


class ContractField(BaseModel):
    value: str | None = Field(
        None, description="The value read; null when the field was not found", examples=["12.345.678.9-012.345"]
    )
    confidence: int | float = Field(
        ...,
        description=(
            "With a threshold for this field in `column_confidence_threshold` (its own key, or `all_field`): 1 when "
            "the ML team's trust model gives this value a probability of being correct of at least that threshold, "
            "else 0. Without one: that probability itself, a float from 0 to 1 (4 decimals). 0 when there is no "
            "value"
        ),
        examples=[1, 0.9731],
    )


class NpwpData(BaseModel):
    nomor_npwp: ContractField = Field(
        ...,
        description="15-digit number as `XX.XXX.XXX.X-XXX.XXX`, or a 16-digit NIK-based number as 16 plain digits",
    )
    nama: ContractField = Field(
        ..., description="The name on the card: the taxpayer's name, or the registered name on a company's card"
    )


class ExtractOcrResponse(BaseModel):
    status_code: int = Field(..., description="Same as the HTTP status code", examples=[200])
    status_desc: str = Field(..., description="Reason phrase of `status_code`", examples=["OK"])
    message: str = Field(
        ...,
        description="Human-readable; its wording may change, so branch on `errors` instead",
        examples=["OCR extraction completed successfully"],
    )
    data: NpwpData | dict[str, Any] | None = Field(
        None,
        description=(
            "The result on 200 (finished); null otherwise. The fields (`nomor_npwp`, `nama`) when "
            "scoring ended the request; the result of the last service of `pipeline_name_sequence`, as it is, "
            "when the sequence ends earlier (the guardrails report, the OCR result, or the structuring result)"
        ),
    )
    errors: str | None = Field(
        None, description="Failure code when the request failed or was refused; null otherwise", examples=[None]
    )
    request_id: str | None = Field(None, description="The request_id this response belongs to")
    pipeline_last_stage: Literal["orchestrator", "guardrails", "extraction", "structuring", "scoring"] | None = Field(
        None,
        description=(
            "Null on a success answer (200 `completed`, 202 `processing`). On an error, the service it comes "
            "from. A pipeline service named as in `pipeline_name_sequence`: the one that rejected (`guardrails`, "
            "`structuring`) or failed; the one that could not be reached or answered an error. `orchestrator` when "
            "this service refused the request itself before any pipeline service was called (API key, file "
            "checks, `pipeline_name_sequence`, thresholds, `document_type`, an unknown request_id)"
        ),
        examples=["structuring"],
    )
    guardrails: int | float | None = Field(
        None,
        description=(
            "With `guardrails_confidence_threshold`: 0 the document passed the guardrails model (and the pipeline "
            "ran). Without it the document is accepted whatever the model says, and this is the model's accepted "
            "probability (the lowest of its pages), a float from 0 to 1 (4 decimals). 1: rejected, by the "
            "guardrails model (only with a threshold) or by the structuring rules. Null on 202, and when the "
            "request was refused before the check"
        ),
        examples=[0, 0.9821],
    )
