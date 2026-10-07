import elasticapm
import pytest

from ocr_common.web import apm


@pytest.fixture(scope="module")
def apm_agent():
    """A real agent client that never sends anything and instruments nothing. One for the module: closing it
    waits for its sender thread."""
    agent = elasticapm.Client(
        service_name="test",
        server_url="http://127.0.0.1:9",
        disable_send=True,
        instrument=False,
        central_config=False,
        cloud_provider="none",
        metrics_interval="0ms",
    )
    yield agent
    agent.close()


@pytest.fixture
def apm_client(apm_agent, monkeypatch):
    """The test agent installed as the APM client."""
    monkeypatch.setattr(apm, "_client", apm_agent)
    return apm_agent
