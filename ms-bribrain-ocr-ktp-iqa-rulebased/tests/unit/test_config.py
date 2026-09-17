"""Unit tests for src.core.config module."""

import pytest

from src.core.config import Config, get_config, load_config, reload_config


class TestConfig:
    def test_get_config_returns_instance(self):
        config = get_config()
        assert isinstance(config, Config)

    def test_singleton(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_app_config(self):
        config = get_config()
        assert config.app.name is not None
        assert config.app.version is not None

    def test_server_config(self):
        config = get_config()
        assert config.server.host == "0.0.0.0"
        assert config.server.port > 0

    def test_quality_blur(self):
        config = get_config()
        assert isinstance(config.quality.blur.enabled, bool)
        assert config.quality.blur.threshold > 0

    def test_quality_confidence(self):
        config = get_config()
        assert config.quality.confidence.enabled is True
        assert 0 <= config.quality.confidence.threshold_median <= 1

    def test_quality_glare(self):
        config = get_config()
        assert isinstance(config.quality.glare.enabled, bool)
        assert config.quality.glare.min_area > 0
        assert config.quality.glare.kernel_size > 0

    def test_quality_rotation(self):
        config = get_config()
        assert config.quality.rotation.enabled is True
        assert config.quality.rotation.min_face_proportion > 0
        assert 0 < config.quality.rotation.face_detection_confidence <= 1

    def test_quality_check_can_be_disabled(self):
        config = Config.model_validate({
            "quality": {
                "blur": {"enabled": False, "threshold": 100.0},
            }
        })
        assert config.quality.blur.enabled is False
        assert config.quality.confidence.enabled is True

    def test_database_config(self):
        config = get_config()
        assert config.database.table_name is not None
        assert config.database.pool_size >= 1

    def test_logging_config(self):
        config = get_config()
        assert config.logging.level in ("DEBUG", "INFO", "WARNING", "ERROR")

    def test_cors_config(self):
        config = get_config()
        assert isinstance(config.cors.allow_origins, list)

    def test_device_config(self):
        config = get_config()
        assert isinstance(config.device.prefer_gpu, bool)
        assert isinstance(config.device.fallback_to_cpu, bool)


class TestConfigGet:
    def test_get_existing(self):
        config = get_config()
        assert config.get("app.name") is not None

    def test_get_nested(self):
        config = get_config()
        assert config.get("quality.blur.threshold") > 0

    def test_get_missing_returns_default(self):
        config = get_config()
        assert config.get("nonexistent.key", "fallback") == "fallback"

    def test_get_missing_returns_none(self):
        config = get_config()
        assert config.get("nonexistent") is None


class TestLoadConfig:
    def test_load_default(self):
        cfg = load_config()
        assert isinstance(cfg, Config)

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("does_not_exist_12345.yaml")

    def test_reload_returns_fresh_instance(self):
        cfg1 = get_config()
        cfg2 = reload_config()
        assert isinstance(cfg2, Config)
        assert get_config() is cfg2
        assert cfg2 is not cfg1 or cfg2 == cfg1
