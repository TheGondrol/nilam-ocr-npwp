import secrets
import sys
import types

import pytest

from ocr_common.pipeline import database

URL = (
    "postgresql+asyncpg://gc-bribrain-dev-sac-sql-01%40common-sec-dev-01.iam@/bribrain_ocr"
    "?cloudsql_instance=edm-bribrain-dev-01:asia-southeast2:gc-bribrain-dev-sql-psql-01"
)


class FakeConnector:
    made: list["FakeConnector"] = []

    def __init__(self, **options):
        self.options = options
        self.calls: list[tuple] = []
        self.closed = False
        FakeConnector.made.append(self)

    async def connect_async(self, instance, driver, **kwargs):
        self.calls.append((instance, driver, kwargs))
        return object()

    async def close_async(self):
        self.closed = True


@pytest.fixture
def connector(monkeypatch):
    """The Cloud SQL Python Connector replaced by `FakeConnector` (no Google credentials in the tests)."""
    FakeConnector.made = []
    module = types.ModuleType("google.cloud.sql.connector")
    module.__dict__.update(Connector=FakeConnector, IPTypes={"PRIVATE": "PRIVATE", "PUBLIC": "PRIMARY", "PSC": "PSC"})
    monkeypatch.setitem(sys.modules, "google.cloud.sql.connector", module)
    yield FakeConnector
    database._connectors.clear()


async def test_a_cloudsql_url_logs_in_with_iam_as_its_user_over_the_private_ip(connector):
    connect = database.cloudsql_connect(URL)

    await connect()
    await connect()

    [made] = connector.made
    assert made.options["enable_iam_auth"] is True
    assert made.options["ip_type"] == "PRIVATE"
    assert (
        made.calls
        == [
            (
                "edm-bribrain-dev-01:asia-southeast2:gc-bribrain-dev-sql-psql-01",
                "asyncpg",
                {"user": "gc-bribrain-dev-sac-sql-01@common-sec-dev-01.iam", "db": "bribrain_ocr"},
            )
        ]
        * 2
    ), "one connector per URL, reused"


async def test_the_ip_type_comes_from_the_url(connector):
    await database.cloudsql_connect(URL + "&cloudsql_ip_type=psc")()

    assert connector.made[0].options["ip_type"] == "PSC"


async def test_dispose_closes_the_connector(connector):
    await database.cloudsql_connect(URL)()

    await database.dispose_engines()

    assert connector.made[0].closed
    assert database._connectors == {}


async def test_with_a_password_it_logs_in_as_a_built_in_user(connector):
    password = secrets.token_hex(8)
    built_in = f"postgresql+asyncpg://nilam_ocr:{password}@/bribrain_ocr?cloudsql_instance=p:r:i"

    await database.cloudsql_connect(built_in)()

    [made] = connector.made
    assert made.options["enable_iam_auth"] is False
    assert made.calls == [("p:r:i", "asyncpg", {"user": "nilam_ocr", "db": "bribrain_ocr", "password": password})]


def test_a_cloudsql_url_needs_the_user_and_the_database():
    with pytest.raises(ValueError, match="IAM user and the database"):
        database.cloudsql_connect("postgresql+asyncpg://@/?cloudsql_instance=p:r:i")


def test_get_engine_does_not_connect_and_hands_the_connector_to_sqlalchemy():
    engine = database.get_engine(URL)
    try:
        assert engine.url.drivername == "postgresql+asyncpg"
        assert engine.url.host is None, "no host: the connector finds the instance"
    finally:
        database._engines.pop(URL)
