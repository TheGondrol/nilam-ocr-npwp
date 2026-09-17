"""Unit tests for src.services.threshold_provider."""

import pytest

import src.services.threshold_provider as tp_mod
from src.services.threshold_provider import (
    ThresholdProvider,
    get_provider,
    init_provider,
)


@pytest.fixture(autouse=True)
def _default_config(monkeypatch, tmp_path):
    """Point CONFIG_PATH at a non-existent file so __init__ uses defaults,
    unless a test overrides CONFIG_PATH itself."""
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "no_such_config.yaml"))
    yield


# --- Fake async engine for exercising _refresh_once without a real DB ---

class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, *args, **kwargs):
        return _FakeResult(self._rows)


class _FakeEngine:
    def __init__(self, rows=None):
        self._rows = rows or []
        self.disposed = False

    def connect(self):
        return _FakeConn(self._rows)

    async def dispose(self):
        self.disposed = True


# --- _load_settings ---

class TestLoadSettings:
    def test_missing_file_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "nope.yaml"))
        assert tp_mod._load_settings() == {}

    def test_reads_block(self, monkeypatch, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("threshold_provider:\n  refresh_interval_seconds: 5\n  table_name: t\n")
        monkeypatch.setenv("CONFIG_PATH", str(p))
        settings = tp_mod._load_settings()
        assert settings["refresh_interval_seconds"] == 5
        assert settings["table_name"] == "t"

    def test_no_block_returns_empty(self, monkeypatch, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("other_section:\n  x: 1\n")
        monkeypatch.setenv("CONFIG_PATH", str(p))
        assert tp_mod._load_settings() == {}

    def test_non_dict_block_returns_empty(self, monkeypatch, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("threshold_provider: 5\n")
        monkeypatch.setenv("CONFIG_PATH", str(p))
        assert tp_mod._load_settings() == {}

    def test_invalid_yaml_returns_empty(self, monkeypatch, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("threshold_provider: [unclosed\n")
        monkeypatch.setenv("CONFIG_PATH", str(p))
        assert tp_mod._load_settings() == {}


# --- construction / accessors ---

class TestConstructionAndAccessors:
    def test_init_reads_settings(self, monkeypatch, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text(
            "threshold_provider:\n"
            "  refresh_interval_seconds: 7\n"
            "  schema: myschema\n"
            "  table_name: tbl\n"
        )
        monkeypatch.setenv("CONFIG_PATH", str(p))
        tp = ThresholdProvider("svc", {"a": 1.0})
        assert tp._refresh_interval == 7
        assert tp._qualified_table == '"myschema"."tbl"'

    def test_qualified_table_without_schema(self, monkeypatch, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("threshold_provider:\n  schema: ''\n  table_name: tbl\n")
        monkeypatch.setenv("CONFIG_PATH", str(p))
        tp = ThresholdProvider("svc", {})
        assert tp._qualified_table == '"tbl"'

    def test_get_prefers_values_over_defaults(self):
        tp = ThresholdProvider("svc", {"a": 1.0})
        tp._values = {"a": 2.0}
        assert tp.get("a") == 2.0

    def test_get_falls_back_to_defaults(self):
        tp = ThresholdProvider("svc", {"a": 1.0})
        tp._values = {}
        assert tp.get("a") == 1.0

    def test_get_unknown_raises(self):
        tp = ThresholdProvider("svc", {"a": 1.0})
        with pytest.raises(KeyError):
            tp.get("missing")

    def test_get_all_returns_copy(self):
        tp = ThresholdProvider("svc", {"a": 1.0})
        snap = tp.get_all()
        snap["a"] = 99.0
        assert tp.get("a") == 1.0

    def test_service_name(self):
        assert ThresholdProvider("svc", {}).service_name == "svc"


# --- singleton helpers ---

class TestProviderSingleton:
    def test_init_provider_is_idempotent(self, monkeypatch):
        monkeypatch.setattr(tp_mod, "_provider", None)
        p1 = init_provider("svc", {"a": 1.0})
        p2 = init_provider("other", {"b": 2.0})
        assert p1 is p2
        assert get_provider() is p1

    def test_get_provider_uninitialized_raises(self, monkeypatch):
        monkeypatch.setattr(tp_mod, "_provider", None)
        with pytest.raises(RuntimeError, match="not initialized"):
            get_provider()


# --- initialize / shutdown / refresh (no real DB) ---

class TestLifecycle:
    async def test_initialize_no_url_uses_defaults(self, monkeypatch):
        monkeypatch.delenv("THRESHOLD_DB_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        tp = ThresholdProvider("svc", {"a": 1.0})
        await tp.initialize()
        assert tp._engine is None

    async def test_initialize_bad_url_falls_back(self, monkeypatch):
        monkeypatch.delenv("THRESHOLD_DB_URL", raising=False)
        monkeypatch.setenv("DATABASE_URL", "notadialect://localhost/db")
        tp = ThresholdProvider("svc", {"a": 1.0})
        await tp.initialize()
        assert tp._engine is None

    async def test_initialize_success_runs_refresh_and_starts_task(self, monkeypatch):
        monkeypatch.delenv("THRESHOLD_DB_URL", raising=False)
        monkeypatch.setenv("DATABASE_URL", "fake://db")
        fake = _FakeEngine([("a", "2.0")])
        monkeypatch.setattr(tp_mod, "create_async_engine", lambda *a, **k: fake)

        tp = ThresholdProvider("svc", {"a": 1.0})
        await tp.initialize()
        assert tp._engine is fake
        assert tp.get("a") == 2.0          # _refresh_once ran during initialize
        assert tp._task is not None

        await tp.shutdown()                # stop event -> loop returns -> dispose
        assert tp._engine is None
        assert fake.disposed is True

    async def test_shutdown_noop_when_uninitialized(self):
        tp = ThresholdProvider("svc", {"a": 1.0})
        await tp.shutdown()  # no task/engine -> must not raise

    async def test_shutdown_disposes_engine(self):
        tp = ThresholdProvider("svc", {"a": 1.0})
        engine = _FakeEngine()
        tp._engine = engine
        await tp.shutdown()
        assert engine.disposed is True
        assert tp._engine is None

    async def test_refresh_once_noop_without_engine(self):
        tp = ThresholdProvider("svc", {"a": 1.0})
        await tp._refresh_once()  # engine None -> early return
        assert tp.get("a") == 1.0

    async def test_refresh_once_updates_changes_and_skips_non_numeric(self):
        tp = ThresholdProvider("svc", {"a": 1.0, "b": 2.0})
        tp._engine = _FakeEngine([("a", "1.5"), ("c", "3.0"), ("bad", "xx")])
        await tp._refresh_once()
        assert tp.get("a") == 1.5      # changed from default
        assert tp.get("c") == 3.0      # new key from DB
        assert tp.get("b") == 2.0      # untouched default
        with pytest.raises(KeyError):
            tp.get("bad")              # non-numeric skipped

    async def test_refresh_once_no_changes_branch(self):
        tp = ThresholdProvider("svc", {"a": 1.0})
        tp._engine = _FakeEngine([("a", "1.0")])
        await tp._refresh_once()       # first load
        await tp._refresh_once()       # second identical load -> "no values changed"
        assert tp.get("a") == 1.0

    async def test_refresh_once_swallows_db_error(self):
        class _BoomEngine:
            def connect(self):
                raise RuntimeError("db down")

        tp = ThresholdProvider("svc", {"a": 1.0})
        tp._engine = _BoomEngine()
        await tp._refresh_once()       # exception caught -> defaults retained
        assert tp.get("a") == 1.0
