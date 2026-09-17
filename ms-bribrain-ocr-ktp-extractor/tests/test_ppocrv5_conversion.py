"""Tests for PP-OCRv5 Paddle-to-PyTorch conversion helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.prepare_ppocrv5_server_weights import build_config
from src.services.ppocrv5_conversion import (
    PPOCRv5ServerConversionConfig,
    ensure_ppocrv5_server_weights,
    infer_paddle_source_from_model_dir,
    load_paddlex_config_model_dir,
    resolve_optional_path,
)


def test_resolve_optional_path_handles_empty_and_relative(tmp_path):
    assert resolve_optional_path("", tmp_path) is None
    assert resolve_optional_path(None, tmp_path) is None
    assert resolve_optional_path("model/inference.pdiparams", tmp_path) == (
        tmp_path / "model/inference.pdiparams"
    ).resolve()


def test_load_paddlex_config_model_dir(tmp_path):
    config = tmp_path / "PaddleOCR_server.yaml"
    config.write_text(
        """
SubModules:
  TextDetection:
    model_dir: ./det_model
  TextRecognition:
    model_dir: ./rec_model
""",
        encoding="utf-8",
    )

    assert load_paddlex_config_model_dir(config, "TextDetection") == "./det_model"
    assert load_paddlex_config_model_dir(config, "TextRecognition") == "./rec_model"
    assert load_paddlex_config_model_dir(config, "Missing") is None


def test_infer_paddle_source_from_model_dir_uses_inference_pdiparams(tmp_path):
    source = infer_paddle_source_from_model_dir("./rec_model", tmp_path)

    assert source == (tmp_path / "rec_model/inference.pdiparams").resolve()


def test_ensure_ppocrv5_server_weights_noops_when_outputs_exist(tmp_path):
    ppocr_root = tmp_path / "PaddleOCR2Pytorch"
    ppocr_root.mkdir()
    det_output = tmp_path / "server_det.pth"
    rec_output = tmp_path / "server_rec.pth"
    det_output.write_bytes(b"det")
    rec_output.write_bytes(b"rec")
    config = PPOCRv5ServerConversionConfig(
        ppocr_root=ppocr_root,
        det_source_path=None,
        rec_source_path=None,
        det_output_path=det_output,
        rec_output_path=rec_output,
    )

    outputs = ensure_ppocrv5_server_weights(config)

    assert outputs == {"det": det_output, "rec": rec_output}


def test_ensure_ppocrv5_server_weights_requires_missing_source(tmp_path):
    ppocr_root = tmp_path / "PaddleOCR2Pytorch"
    ppocr_root.mkdir()
    config = PPOCRv5ServerConversionConfig(
        ppocr_root=ppocr_root,
        det_source_path=None,
        rec_source_path=None,
        det_output_path=tmp_path / "server_det.pth",
        rec_output_path=tmp_path / "server_rec.pth",
    )

    with pytest.raises(FileNotFoundError, match="Missing PP-OCRv5 det Paddle checkpoint"):
        ensure_ppocrv5_server_weights(config)


def test_prepare_script_build_config_supports_rec_only_source(tmp_path, monkeypatch):
    ppocr_root = tmp_path / "PaddleOCR2Pytorch"
    rec_source = tmp_path / "inference.pdiparams"
    ppocr_root.mkdir()
    rec_source.write_bytes(b"weights")

    args = SimpleNamespace(
        component="rec",
        ppocr_root=str(ppocr_root),
        det_src=None,
        rec_src=str(rec_source),
        det_dst=str(tmp_path / "server_det.pth"),
        rec_dst=str(tmp_path / "server_rec.pth"),
        paddlex_config=None,
        force=False,
        json_output=None,
    )

    config = build_config(args)

    assert config.components == ("rec",)
    assert config.ppocr_root == ppocr_root.resolve()
    assert config.rec_source_path == rec_source.resolve()
