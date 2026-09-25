"""`pipeline_name_sequence`: which services one request runs, chosen by the central orchestrator.

The order is always guardrails -> extraction -> structuring -> scoring. A request may leave guardrails
out at the front and cut the end off, never skip a service in the middle: every service after
guardrails needs the result of the one before it. Valid: [guardrails, extraction, structuring,
scoring] (the default), [guardrails, extraction], [extraction, structuring], [guardrails]. Not valid:
[extraction, scoring], [structuring, scoring], a wrong order, a name twice.

The last service in the sequence ends the request: its result is the answer, as it is.
"""

from collections.abc import Sequence
from typing import Any

from ocr_common.pipeline.stage import STAGE_OCR, STAGE_SCORING, STAGE_STRUCTURING, HandoffPayload

GUARDRAILS = "guardrails"
EXTRACTION = "extraction"
STRUCTURING = "structuring"
SCORING = "scoring"

PIPELINE_NAMES: tuple[str, ...] = (GUARDRAILS, EXTRACTION, STRUCTURING, SCORING)
DEFAULT_SEQUENCE = PIPELINE_NAMES

# The stage name of each pipeline service in jobs, callbacks and the orchestrator's tables.
STAGE_OF = {GUARDRAILS: "GUARDRAILS", EXTRACTION: STAGE_OCR, STRUCTURING: STAGE_STRUCTURING, SCORING: STAGE_SCORING}
# And back: the pipeline service of a stage name, as pipeline_name_sequence and pipeline_last_stage name it.
SERVICE_OF_STAGE = {stage: name for name, stage in STAGE_OF.items()}


class InvalidSequence(ValueError):
    """The sequence breaks the rules above; the message says how."""


def validate_sequence(names: Sequence[str] | None) -> tuple[str, ...]:
    """The sequence to run: `names` when valid, the full pipeline when None or empty."""
    if not names:
        return DEFAULT_SEQUENCE
    names = tuple(names)
    unknown = [name for name in names if name not in PIPELINE_NAMES]
    if unknown:
        raise InvalidSequence(f"unknown service {unknown[0]!r}; allowed: {', '.join(PIPELINE_NAMES)}")
    if len(set(names)) != len(names):
        raise InvalidSequence("a service is listed twice")
    start = PIPELINE_NAMES.index(names[0])
    if start > PIPELINE_NAMES.index(EXTRACTION):
        raise InvalidSequence(f"{names[0]} cannot come first: it needs the result of {PIPELINE_NAMES[start - 1]}")
    if names != PIPELINE_NAMES[start : start + len(names)]:
        raise InvalidSequence(
            f"services must keep the order {' -> '.join(PIPELINE_NAMES)} without skipping one in the middle"
        )
    return names


def next_service(sequence: Sequence[str] | None, name: str) -> str | None:
    """The service after `name` in `sequence` (validated; None = the full pipeline), or None when `name`
    is the last one. Raises `InvalidSequence` when `name` is not in it."""
    sequence = validate_sequence(sequence)
    if name not in sequence:
        raise InvalidSequence(f"pipeline_name_sequence does not include {name}: {list(sequence)}")
    index = sequence.index(name)
    return sequence[index + 1] if index + 1 < len(sequence) else None


def checked_sequence(names: Sequence[str] | None, name: str) -> list[str] | None:
    """A stage's check of the sequence it was sent: valid and including `name`. None stays None (the full
    pipeline). Raises `InvalidSequence`, a ValueError, so a Pydantic validator turns it into a 422."""
    if not names:
        return None
    sequence = validate_sequence(names)
    if name not in sequence:
        raise InvalidSequence(f"pipeline_name_sequence does not include {name}: {list(sequence)}")
    return list(sequence)


def chain(sequence: Sequence[str] | None, name: str, handoff_payload: HandoffPayload | None) -> dict[str, Any]:
    """The `StagePipeline.submit` / `resume` arguments that continue the request after `name`: a
    hand-off to the next service of `sequence`, or, when `name` is the last one, its result as it is
    for the DONE callback and the orchestrator's outcome row (the request ends here)."""
    following = next_service(sequence, name)
    if following is None:
        return {"callback_result": dict, "outcome_data": dict}
    return {"handoff_payload": handoff_payload, "next_stage": STAGE_OF[following]}


def stored(sequence: Sequence[str] | None) -> list[str] | None:
    """The sequence as a job's `input` and a hand-off body carry it (JSON); None for the full pipeline."""
    return list(sequence) if sequence else None
