import io

import pytest
from PIL import Image

from ocr_common.errors import UpstreamUnavailable

from app.clients.reject_threshold import RejectThreshold, default_threshold, parse_threshold
from app.config import Settings
from app.services.guardrails_service import GuardrailsService


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


def _threshold(stub, clock=None, cache_seconds=60.0, default=0.5) -> RejectThreshold:
    return RejectThreshold(
        stub, "/v1/thresholds/guardrails", default, cache_seconds=cache_seconds, clock=clock or Clock()
    )


class StubClassifier:
    reject_threshold = 0.5

    def classify(self, filename, pages):
        return [(0.35, 0.65) for _ in pages]


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (200, 100), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_default_is_the_configured_threshold_else_the_checkpoints():
    assert default_threshold(0.3, StubClassifier()) == 0.3
    assert default_threshold(None, StubClassifier()) == 0.5
    assert default_threshold(None, object()) == 0.5


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"reject_threshold": None},
        {"reject_threshold": "0.5"},
        {"reject_threshold": True},
        {"reject_threshold": 0},
        {"reject_threshold": 1},
        {"reject_threshold": 1.5},
        {"reject_threshold": float("nan")},
        [0.5],
    ],
)
def test_parse_refuses_anything_but_a_threshold_between_0_and_1(body):
    with pytest.raises(ValueError):
        parse_threshold(body)


async def test_without_url_the_default_is_used_and_nothing_is_called():
    assert await RejectThreshold(None, "", 0.4, cache_seconds=60).get() == 0.4


async def test_the_orchestrators_threshold_is_used_and_cached():
    stub, clock = StubOrchestrator({"reject_threshold": 0.7}), Clock()
    threshold = _threshold(stub, clock)

    assert await threshold.get() == 0.7
    clock.now += 59
    assert await threshold.get() == 0.7
    assert stub.paths == ["/v1/thresholds/guardrails"]


async def test_a_change_at_the_orchestrator_applies_after_the_cache_expires():
    stub, clock = StubOrchestrator({"reject_threshold": 0.7}, {"reject_threshold": 0.3}), Clock()
    threshold = _threshold(stub, clock)

    assert await threshold.get() == 0.7
    clock.now += 60
    assert await threshold.get() == 0.3
    assert len(stub.paths) == 2


async def test_unreachable_orchestrator_gives_the_default():
    stub = StubOrchestrator(UpstreamUnavailable("orchestrator reject threshold is unavailable"))
    assert await _threshold(stub).get() == 0.5


async def test_failure_after_a_success_keeps_the_last_value_and_retries_after_the_cache():
    stub = StubOrchestrator({"reject_threshold": 0.7}, {"reject_threshold": "x"}, {"reject_threshold": 0.6})
    clock = Clock()
    threshold = _threshold(stub, clock)

    assert await threshold.get() == 0.7
    clock.now += 60
    assert await threshold.get() == 0.7  # invalid answer: the last value stays
    clock.now += 30
    assert await threshold.get() == 0.7  # not asked again before the cache expires
    clock.now += 30
    assert await threshold.get() == 0.6
    assert len(stub.paths) == 3


async def test_aclose_closes_the_client():
    stub = StubOrchestrator({"reject_threshold": 0.7})
    await _threshold(stub).aclose()
    assert stub.closed


async def test_service_judges_with_the_orchestrators_threshold_and_reports_it():
    settings = Settings(api_key="x", _env_file=None)
    lenient = _threshold(StubOrchestrator({"reject_threshold": 0.7}))
    report = await GuardrailsService(StubClassifier(), settings, lenient).check("a.jpg", "image/jpeg", _jpeg())
    assert (report["passed"], report["document"]["reject_threshold"]) == (True, 0.7)

    default = await GuardrailsService(StubClassifier(), settings).check("a.jpg", "image/jpeg", _jpeg())
    assert (default["passed"], default["document"]["reject_threshold"]) == (False, 0.5)


def test_threshold_url_on_localhost_is_refused_outside_local():
    with pytest.raises(ValueError, match="GUARDRAILS_THRESHOLD_URL"):
        Settings(
            api_key="x",
            _env_file=None,
            environment="production",
            guardrails_backend="efficientnet",
            guardrails_threshold_url="http://localhost:8090",
        )
