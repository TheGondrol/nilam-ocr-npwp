"""Unit tests for src.core.config module."""

from src.core.config import Config


class TestConfigSingleton:
    def test_singleton_returns_same_instance(self):
        c1 = Config()
        c2 = Config()
        assert c1 is c2

    def test_config_loaded(self):
        c = Config()
        assert c._config is not None
        assert isinstance(c._config, dict)


class TestConfigGet:
    def test_get_existing_key(self):
        c = Config()
        assert c.get("model.path") is not None

    def test_get_nested_key(self):
        c = Config()
        assert c.get("model.image_size") == 224

    def test_get_missing_returns_default(self):
        c = Config()
        assert c.get("nonexistent.key", "fallback") == "fallback"

    def test_get_missing_returns_none(self):
        c = Config()
        assert c.get("nonexistent") is None

    def test_get_all(self):
        c = Config()
        all_cfg = c.get_all()
        assert isinstance(all_cfg, dict)
        assert "model" in all_cfg


class TestConfigProperties:
    def test_model_path(self):
        c = Config()
        assert isinstance(c.model_path, str)

    def test_image_size(self):
        c = Config()
        assert c.image_size == 224

    def test_threshold(self):
        c = Config()
        assert 0 <= c.threshold <= 1

    def test_server_host(self):
        c = Config()
        assert c.server_host == "0.0.0.0"

    def test_server_port(self):
        c = Config()
        assert c.server_port > 0

    def test_log_to_database(self):
        c = Config()
        assert isinstance(c.log_to_database, bool)

    def test_torch_compile(self):
        c = Config()
        assert isinstance(c.torch_compile, bool)

    def test_thread_pool_workers(self):
        c = Config()
        assert c.thread_pool_workers > 0

    def test_max_file_size_mb(self):
        c = Config()
        assert c.max_file_size_mb > 0

    def test_max_batch_size(self):
        c = Config()
        assert c.max_batch_size > 0

    def test_db_pool_size(self):
        c = Config()
        assert c.db_pool_size >= 1

    def test_db_max_overflow(self):
        c = Config()
        assert c.db_max_overflow >= 0

    def test_db_pool_pre_ping(self):
        c = Config()
        assert isinstance(c.db_pool_pre_ping, bool)

    def test_db_pool_recycle(self):
        c = Config()
        assert c.db_pool_recycle > 0

    def test_db_pool_timeout(self):
        c = Config()
        assert c.db_pool_timeout > 0

    def test_db_connect_timeout(self):
        c = Config()
        assert c.db_connect_timeout > 0

    def test_gcs_bucket_name(self):
        c = Config()
        assert isinstance(c.gcs_bucket_name, str)

    def test_gcs_prefix(self):
        c = Config()
        assert isinstance(c.gcs_prefix, str)

    def test_gcs_model_prefix(self):
        c = Config()
        assert isinstance(c.gcs_model_prefix, str)

    def test_cors_config(self):
        c = Config()
        cors = c.cors_config
        assert isinstance(cors, dict)
        assert "allow_origins" in cors
        assert "allow_credentials" in cors
        assert "allow_methods" in cors
        assert "allow_headers" in cors


class TestConfigDefaults:
    def test_default_config_when_file_missing(self):
        from unittest.mock import patch
        Config._instance = None
        with patch.dict("os.environ", {"CONFIG_PATH": "/nonexistent/path.yaml"}):
            Config._instance = None
            c = Config.__new__(Config)
            c._load_config()
            assert c._config is not None
            assert "model" in c._config
