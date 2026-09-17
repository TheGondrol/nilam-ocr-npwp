"""Unit tests for src.core.config module."""

from src.core.config import Config, config


class TestConfigSingleton:
    def test_singleton_returns_same_instance(self):
        c1 = Config()
        c2 = Config()
        assert c1 is c2

    def test_global_config_is_instance(self):
        assert isinstance(config, Config)


class TestConfigGet:
    def test_get_existing_key(self):
        assert config.get("model.num_classes") == 2

    def test_get_nested_key(self):
        assert config.get("model.img_height") == 64

    def test_get_missing_key_returns_default(self):
        assert config.get("nonexistent.key", "fallback") == "fallback"

    def test_get_missing_key_returns_none(self):
        assert config.get("nonexistent.key") is None

    def test_get_partial_key_returns_default(self):
        # "model" exists but "model.nonexistent" doesn't
        assert config.get("model.nonexistent", 42) == 42

    def test_get_deeply_nested(self):
        assert config.get("logging.file.enabled") is True

    def test_get_top_level(self):
        assert config.get("debug_mode") is False


class TestConfigProperties:
    def test_model_path(self):
        assert "model" in config.model_path or "pth" in config.model_path

    def test_num_classes(self):
        assert config.num_classes == 2

    def test_img_height(self):
        assert config.img_height == 64

    def test_img_width(self):
        assert config.img_width == 320

    def test_normalize_mean(self):
        mean = config.normalize_mean
        assert isinstance(mean, list)
        assert len(mean) == 3

    def test_normalize_std(self):
        std = config.normalize_std
        assert isinstance(std, list)
        assert len(std) == 3

    def test_use_compile(self):
        assert isinstance(config.use_compile, bool)

    def test_compile_mode(self):
        assert config.compile_mode in ["default", "reduce-overhead", "max-autotune"]

    def test_bad_crop_threshold(self):
        assert isinstance(config.bad_crop_threshold, int)
        assert config.bad_crop_threshold > 0

    def test_min_width_ratio(self):
        assert isinstance(config.min_width_ratio, float)

    def test_min_width(self):
        assert isinstance(config.min_width, int)

    def test_server_host(self):
        assert config.server_host == "0.0.0.0"

    def test_server_port(self):
        assert config.server_port == 8100

    def test_thread_pool_workers(self):
        assert config.thread_pool_workers >= 1

    def test_debug_mode(self):
        assert isinstance(config.debug_mode, bool)

    def test_log_to_database(self):
        assert isinstance(config.log_to_database, bool)

    def test_db_pool_size(self):
        assert config.db_pool_size >= 1

    def test_db_max_overflow(self):
        assert isinstance(config.db_max_overflow, int)

    def test_db_pool_pre_ping(self):
        assert isinstance(config.db_pool_pre_ping, bool)

    def test_db_pool_recycle(self):
        assert config.db_pool_recycle > 0

    def test_db_pool_timeout(self):
        assert config.db_pool_timeout > 0

    def test_db_connect_timeout(self):
        assert config.db_connect_timeout > 0

    def test_gcs_bucket_name(self):
        assert isinstance(config.gcs_bucket_name, str)

    def test_gcs_prefix(self):
        assert isinstance(config.gcs_prefix, str)

    def test_gcs_model_prefix(self):
        assert isinstance(config.gcs_model_prefix, str)
