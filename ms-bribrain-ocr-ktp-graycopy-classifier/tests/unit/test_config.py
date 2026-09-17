from pathlib import Path

import yaml

from src.core.config import Config, DeviceConfig, load_config


def test_load_config_missing_file_returns_defaults(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"
    config = load_config(str(missing_path))

    assert isinstance(config, Config)
    assert config.model.image_size == 224
    assert config.prediction.threshold == 0.5
    assert config.server.port == 8020


def test_load_config_overrides_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    data = {
        "model": {"image_size": 256, "dropout_rate": 0.3},
        "prediction": {"threshold": 0.7},
        "server": {"port": 9000},
        "device": {"compile_mode": "max-autotune"},
    }
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    config = load_config(str(config_path))

    assert config.model.image_size == 256
    assert config.model.dropout_rate == 0.3
    assert config.prediction.threshold == 0.7
    assert config.server.port == 9000
    assert config.device.compile_mode == "max-autotune"


def test_device_compile_mode_validation() -> None:
    device = DeviceConfig(compile_mode="reduce-overhead")
    assert device.is_valid_compile_mode is True

    device_invalid = DeviceConfig(compile_mode="not-a-mode")
    assert device_invalid.is_valid_compile_mode is False


def test_load_config_empty_yaml_file_returns_defaults(tmp_path: Path) -> None:
    """An existing but empty YAML file should fall back to defaults (covers config.py:152)."""
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    config = load_config(str(config_path))

    assert isinstance(config, Config)
    assert config.model.image_size == 224
    assert config.server.port == 8020
