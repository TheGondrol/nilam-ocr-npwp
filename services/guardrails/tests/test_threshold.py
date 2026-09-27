import io

import pytest
from PIL import Image

from ocr_common.errors import UpstreamUnavailable

from app.clients.threshold import GuardrailsThreshold, Threshold, default_threshold, parse_threshold
from app.config import Settings
from app.services.guardrails_service import GuardrailsService

REJECT_05 = Threshold(0.5, "reject")


def _accept(value: float) -> dict:
    return {"threshold": value, "target": "accept"}


def _reject(value: float) -> dict:
    return {"threshold": value, "target": "reject"}


class StubOrchestrator:
    """The orchestrator's threshold endpoint: answers each GET with the next of `answers` (an exception is
    raised), repeating the last one."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.paths: list[str] = []
        self.closed = False

    async def get_json(self, path, *, params=None):
        self.paths.append(path)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def aclose(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _threshold(stub, clock=None, cache_seconds=60.0, default=REJECT_05) -> GuardrailsThreshold:
    return GuardrailsThreshold(
        stub, "/v1/thresholds/guardrails", default, cache_seconds=cache_seconds, clock=clock or Clock()
    )


class StubClassifier:
    """The model's answer for every page: proba_approve 0.7, proba_reject 0.3."""

    reject_threshold = 0.5

    def __init__(self, proba_approve: float = 0.7):
        self._prediction = (proba_approve, round(1 - proba_approve, 4))

    def classify(self, filename, pages):
        return [self._prediction for _ in pages]


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (200, 100), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


# --- how each side decides ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("threshold", "rejected"),
    [
        # The model's answer: proba_approve 0.7, proba_reject 0.3.
        (Threshold(0.6, "accept"), False),  # 0.7 passed the minimum of 0.6
        (Threshold(0.6, "reject"), False),  # 0.3 is below the tolerance of 0.6
        (Threshold(0.8, "accept"), True),  # 0.7 did not reach the minimum of 0.8
        (Threshold(0.2, "reject"), True),  # 0.3 is above the tolerance of 0.2
        # Exactly at the limit, as stated: accept at >= minimum, reject at >= tolerance.
        (Threshold(0.7, "accept"), False),
        (Threshold(0.3, "reject"), True),
    ],
)
def test_each_side_is_compared_with_its_own_probability(threshold, rejected):
    assert threshold.rejects(proba_approve=0.7, proba_reject=0.3) is rejected


def test_default_is_the_configured_threshold_else_the_checkpoints_on_the_reject_side():
    assert default_threshold(0.3, StubClassifier()) == Threshold(0.3, "reject")
    assert default_threshold(None, StubClassifier()) == REJECT_05
    assert default_threshold(None, object()) == REJECT_05


def test_parse_reads_the_threshold_and_its_side():
    assert parse_threshold(_accept(0.6)) == Threshold(0.6, "accept")
    assert parse_threshold(_reject(0.4)) == Threshold(0.4, "reject")


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"threshold": 0.6},
        {"threshold": 0.6, "target": "approve"},
        {"threshold": 0.6, "target": None},
        {"threshold": None, "target": "accept"},
        {"threshold": "0.5", "target": "accept"},
        {"threshold": True, "target": "reject"},
        {"threshold": 0, "target": "reject"},
        {"threshold": 1, "target": "accept"},
        {"threshold": float("nan"), "target": "reject"},
        {"reject_threshold": 0.5},
        [0.5],
    ],
)
def test_parse_refuses_anything_but_a_threshold_between_0_and_1_with_a_side(body):
    with pytest.raises(ValueError):
        parse_threshold(body)


# --- reading it from the orchestrator ---------------------------------------------------------


async def test_without_url_the_default_is_used_and_nothing_is_called():
    assert await GuardrailsThreshold(None, "", Threshold(0.4), cache_seconds=60).get() == Threshold(0.4, "reject")


async def test_the_orchestrators_threshold_is_used_and_cached():
    stub, clock = StubOrchestrator(_accept(0.6)), Clock()
    threshold = _threshold(stub, clock)

    assert await threshold.get() == Threshold(0.6, "accept")
    clock.now += 59
    assert await threshold.get() == Threshold(0.6, "accept")
    assert stub.paths == ["/v1/thresholds/guardrails"]


async def test_a_change_at_the_orchestrator_applies_after_the_cache_expires():
    stub, clock = StubOrchestrator(_accept(0.6), _reject(0.3)), Clock()
    threshold = _threshold(stub, clock)

    assert await threshold.get() == Threshold(0.6, "accept")
    clock.now += 60
    assert await threshold.get() == Threshold(0.3, "reject")
    assert len(stub.paths) == 2


async def test_unreachable_orchestrator_gives_the_default():
    stub = StubOrchestrator(UpstreamUnavailable("orchestrator threshold is unavailable"))
    assert await _threshold(stub).get() == REJECT_05


async def test_failure_after_a_success_keeps_the_last_value_and_retries_after_the_cache():
    stub = StubOrchestrator(_accept(0.6), {"threshold": 0.6, "target": "x"}, _reject(0.4))
    clock = Clock()
    threshold = _threshold(stub, clock)

    assert await threshold.get() == Threshold(0.6, "accept")
    clock.now += 60
    assert await threshold.get() == Threshold(0.6, "accept")  # invalid answer: the last one stays
    clock.now += 30
    assert await threshold.get() == Threshold(0.6, "accept")  # not asked again before the cache expires
    clock.now += 30
    assert await threshold.get() == Threshold(0.4, "reject")
    assert len(stub.paths) == 3


async def test_aclose_closes_the_client():
    stub = StubOrchestrator(_accept(0.6))
    await _threshold(stub).aclose()
    assert stub.closed


# --- the service ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "passed"),
    [(_accept(0.6), True), (_reject(0.6), True), (_accept(0.8), False), (_reject(0.2), False)],
)
async def test_the_service_judges_with_the_orchestrators_threshold_and_reports_it(answer, passed):
    settings = Settings(api_key="x", _env_file=None)
    threshold = _threshold(StubOrchestrator(answer))

    report = await GuardrailsService(StubClassifier(0.7), settings, threshold).check("a.jpg", "image/jpeg", _jpeg())

    assert report["passed"] is passed
    assert (report["document"]["threshold"], report["document"]["threshold_target"]) == (
        answer["threshold"],
        answer["target"],
    )


async def test_without_the_orchestrator_the_default_reject_threshold_applies():
    settings = Settings(api_key="x", _env_file=None)

    report = await GuardrailsService(StubClassifier(0.35), settings).check("a.jpg", "image/jpeg", _jpeg())

    assert report["passed"] is False  # proba_reject 0.65 >= 0.5
    assert (report["document"]["threshold"], report["document"]["threshold_target"]) == (0.5, "reject")


def test_threshold_url_on_localhost_is_refused_outside_local():
    with pytest.raises(ValueError, match="GUARDRAILS_THRESHOLD_URL"):
        Settings(
            api_key="x",
            _env_file=None,
            environment="production",
            guardrails_backend="efficientnet",
            guardrails_threshold_url="http://localhost:8090",
        )
