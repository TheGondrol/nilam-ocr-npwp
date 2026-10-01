import io

import pytest
from PIL import Image

from ocr_common.errors import UpstreamUnavailable

from app.clients.threshold import GuardrailsThreshold, default_threshold, parse_threshold, rejects
from app.config import Settings
from app.services.guardrails_service import GuardrailsService

DEFAULT = 0.5


def _answer(value: float) -> dict:
    return {"threshold": value}


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


def _threshold(stub, clock=None, cache_seconds=60.0, default=DEFAULT) -> GuardrailsThreshold:
    return GuardrailsThreshold(
        stub, "/v1/thresholds/guardrails", default, cache_seconds=cache_seconds, clock=clock or Clock()
    )


class StubClassifier:
    """The model's answer for every page: proba_approve 0.7 unless told otherwise."""

    accept_threshold = 0.5

    def __init__(self, proba_approve: float = 0.7):
        self._prediction = (proba_approve, round(1 - proba_approve, 4))

    def classify(self, filename, pages):
        return [self._prediction for _ in pages]


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (200, 100), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


# --- how the threshold decides ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("threshold", "rejected"),
    [
        # The model's answer: proba_approve 0.7.
        (0.6, False),  # 0.7 reached the minimum of 0.6
        (0.8, True),  # 0.7 did not reach the minimum of 0.8
        (0.7, False),  # exactly at the limit: accepted
    ],
)
def test_a_page_below_the_threshold_is_rejected(threshold, rejected):
    assert rejects(0.7, threshold) is rejected


def test_default_is_the_configured_threshold_else_the_checkpoints():
    assert default_threshold(0.3, StubClassifier()) == 0.3
    assert default_threshold(None, StubClassifier()) == DEFAULT
    assert default_threshold(None, object()) == DEFAULT


def test_parse_reads_the_threshold():
    assert parse_threshold(_answer(0.6)) == 0.6


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"threshold": None},
        {"threshold": "0.5"},
        {"threshold": True},
        {"threshold": 0},
        {"threshold": 1},
        {"threshold": float("nan")},
        {"reject_threshold": 0.5},
        [0.5],
    ],
)
def test_parse_refuses_anything_but_a_threshold_between_0_and_1(body):
    with pytest.raises(ValueError):
        parse_threshold(body)


# --- reading it from the orchestrator ---------------------------------------------------------


async def test_without_url_the_default_is_used_and_nothing_is_called():
    assert await GuardrailsThreshold(None, "", 0.4, cache_seconds=60).get() == 0.4


async def test_the_orchestrators_threshold_is_used_and_cached():
    stub, clock = StubOrchestrator(_answer(0.6)), Clock()
    threshold = _threshold(stub, clock)

    assert await threshold.get() == 0.6
    clock.now += 59
    assert await threshold.get() == 0.6
    assert stub.paths == ["/v1/thresholds/guardrails"]


async def test_a_change_at_the_orchestrator_applies_after_the_cache_expires():
    stub, clock = StubOrchestrator(_answer(0.6), _answer(0.3)), Clock()
    threshold = _threshold(stub, clock)

    assert await threshold.get() == 0.6
    clock.now += 60
    assert await threshold.get() == 0.3
    assert len(stub.paths) == 2


async def test_unreachable_orchestrator_gives_the_default():
    stub = StubOrchestrator(UpstreamUnavailable("orchestrator threshold is unavailable"))
    assert await _threshold(stub).get() == DEFAULT


async def test_failure_after_a_success_keeps_the_last_value_and_retries_after_the_cache():
    stub = StubOrchestrator(_answer(0.6), {"threshold": "x"}, _answer(0.4))
    clock = Clock()
    threshold = _threshold(stub, clock)

    assert await threshold.get() == 0.6
    clock.now += 60
    assert await threshold.get() == 0.6  # invalid answer: the last one stays
    clock.now += 30
    assert await threshold.get() == 0.6  # not asked again before the cache expires
    clock.now += 30
    assert await threshold.get() == 0.4
    assert len(stub.paths) == 3


async def test_aclose_closes_the_client():
    stub = StubOrchestrator(_answer(0.6))
    await _threshold(stub).aclose()
    assert stub.closed


# --- the service ------------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "passed"), [(0.6, True), (0.8, False)])
async def test_the_service_judges_with_the_orchestrators_threshold_and_reports_it(value, passed):
    settings = Settings(api_key="x", _env_file=None)
    threshold = _threshold(StubOrchestrator(_answer(value)))

    report = await GuardrailsService(StubClassifier(0.7), settings, threshold).check("a.jpg", "image/jpeg", _jpeg())

    assert report["passed"] is passed
    assert report["document"]["threshold"] == value
    assert "threshold_target" not in report["document"]


async def test_without_the_orchestrator_the_default_threshold_applies():
    settings = Settings(api_key="x", _env_file=None)

    report = await GuardrailsService(StubClassifier(0.35), settings).check("a.jpg", "image/jpeg", _jpeg())

    assert report["passed"] is False  # proba_approve 0.35 < 0.5
    assert report["document"]["threshold"] == 0.5


def test_threshold_url_on_localhost_is_refused_outside_local():
    with pytest.raises(ValueError, match="GUARDRAILS_THRESHOLD_URL"):
        Settings(
            api_key="x",
            _env_file=None,
            environment="production",
            guardrails_backend="efficientnet",
            guardrails_threshold_url="http://localhost:8090",
        )


# --- a threshold sent with the request (the central orchestrator, through the orchestrator NPWP) ----


async def test_a_threshold_given_with_the_request_wins_over_the_one_in_force():
    settings = Settings(api_key="x", _env_file=None)
    in_force = _threshold(StubOrchestrator(_answer(0.6)))  # would accept proba_approve 0.7
    service = GuardrailsService(StubClassifier(0.7), settings, in_force)

    report = await service.check("a.jpg", "image/jpeg", _jpeg(), 0.8)

    assert report["passed"] is False  # proba_approve 0.7 < 0.8
    assert report["document"]["threshold"] == 0.8


def _check(client, auth, **form):
    return client.post(
        "/v1/guardrails/check",
        data={"request_id": "OCR_t", **form},
        files={"file": ("npwp.jpg", _jpeg(), "image/jpeg")},
        headers=auth,
    )


def test_the_endpoint_judges_with_the_threshold_it_is_sent(client, auth, use_classifier):
    use_classifier(StubClassifier(0.7))

    strict = _check(client, auth, threshold="0.8").json()["data"]
    lenient = _check(client, auth, threshold="0.3").json()["data"]

    assert (strict["passed"], strict["document"]["threshold"]) == (False, 0.8)
    assert (lenient["passed"], lenient["document"]["threshold"]) == (True, 0.3)


def test_without_a_threshold_the_endpoint_falls_back_to_the_default(client, auth, use_classifier):
    use_classifier(StubClassifier(0.7))

    document = _check(client, auth).json()["data"]["document"]

    assert document["threshold"] == 0.5


@pytest.mark.parametrize("threshold", ["0", "1", "1.5", "tinggi"])
def test_a_threshold_out_of_range_is_422(client, auth, use_classifier, threshold):
    use_classifier(StubClassifier(0.7))

    assert _check(client, auth, threshold=threshold).status_code == 422
