# Copyright 2026 FireRedTeam

"""Tests for multi-mode VAD: config, CLI, dispatch, logging, and path validation.

Covers:
- VadConfig defaults and resolve_vad_model_dirs()
- CLI --vad-type argument acceptance / rejection
- _SessionVadState multi-mode model dispatch
- Logging on model load
- Path validation for missing model files
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

import numpy as np
import pytest

from asr2s_grpc.config import (
    ApiConfig,
    VadConfig,
    _VAD_TYPE_MAP,
    resolve_vad_model_dirs,
)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestVadConfigDefaults:
    """Tests for VadConfig default vad_type and resolve_vad_model_dirs()."""

    def test_default_vad_type_is_all(self):
        """VadConfig() default vad_type should be 'all'."""
        cfg = VadConfig()
        assert cfg.vad_type == "all"

    def test_resolve_vad_model_dirs_all_returns_three(self):
        """resolve_vad_model_dirs() with vad_type='all' returns 3 entries."""
        cfg = VadConfig(vad_type="all")
        dirs = resolve_vad_model_dirs(cfg, base_dir="/models/FireRedVAD")
        assert len(dirs) == 3
        assert set(dirs.keys()) == {"vad", "stream-vad", "aed"}

    def test_resolve_vad_model_dirs_all_paths(self):
        """resolve_vad_model_dirs() with 'all' returns correct paths."""
        cfg = VadConfig(vad_type="all")
        dirs = resolve_vad_model_dirs(cfg, base_dir="/models")
        assert dirs["vad"] == "/models/FireRedVAD/VAD"
        assert dirs["stream-vad"] == "/models/FireRedVAD/Stream-VAD"
        assert dirs["aed"] == "/models/FireRedVAD/AED"

    def test_resolve_vad_model_dirs_single_stream_vad(self):
        """resolve_vad_model_dirs() with 'stream-vad' returns 1 entry."""
        cfg = VadConfig(vad_type="stream-vad")
        dirs = resolve_vad_model_dirs(cfg, base_dir="/models")
        assert len(dirs) == 1
        assert dirs["stream-vad"] == "/models/FireRedVAD/Stream-VAD"

    def test_resolve_vad_model_dirs_single_vad(self):
        """resolve_vad_model_dirs() with 'vad' returns correct path."""
        cfg = VadConfig(vad_type="vad")
        dirs = resolve_vad_model_dirs(cfg, base_dir="/models")
        assert dirs == {"vad": "/models/FireRedVAD/VAD"}

    def test_resolve_vad_model_dirs_single_aed(self):
        """resolve_vad_model_dirs() with 'aed' returns correct path."""
        cfg = VadConfig(vad_type="aed")
        dirs = resolve_vad_model_dirs(cfg, base_dir="/models")
        assert dirs == {"aed": "/models/FireRedVAD/AED"}

    def test_resolve_vad_model_dirs_invalid_raises(self):
        """resolve_vad_model_dirs() with unknown vad_type raises ValueError."""
        cfg = VadConfig()
        cfg.vad_type = "invalid"
        with pytest.raises(ValueError, match="Unknown vad_type"):
            resolve_vad_model_dirs(cfg, base_dir="/models")

    def test_resolve_vad_model_dirs_derives_base_from_model_dir(self):
        """When base_dir=None, derive from vad_config.model_dir parent."""
        cfg = VadConfig(
            vad_type="stream-vad", model_dir="/my/models/FireRedVAD/Stream-VAD"
        )
        dirs = resolve_vad_model_dirs(cfg, base_dir=None)
        # base_dir should be /my/models/FireRedVAD
        assert dirs["stream-vad"] == "/my/models/FireRedVAD/FireRedVAD/Stream-VAD"


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestVadTypeCli:
    """Tests for --vad-type CLI argument parsing."""

    @staticmethod
    def _make_parser() -> "argparse.ArgumentParser":
        """Replicate the --vad-type argument from asr2s_grpc.cli."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--vad-type",
            type=str,
            default=None,
            choices=["vad", "stream-vad", "aed", "all"],
        )
        return parser

    def test_cli_vad_type_vad_accepted(self):
        """--vad-type vad should be accepted by argparse."""
        args = self._make_parser().parse_args(["--vad-type", "vad"])
        assert args.vad_type == "vad"

    def test_cli_vad_type_stream_vad_accepted(self):
        """--vad-type stream-vad should be accepted by argparse."""
        args = self._make_parser().parse_args(["--vad-type", "stream-vad"])
        assert args.vad_type == "stream-vad"

    def test_cli_vad_type_aed_accepted(self):
        """--vad-type aed should be accepted by argparse."""
        args = self._make_parser().parse_args(["--vad-type", "aed"])
        assert args.vad_type == "aed"

    def test_cli_vad_type_all_accepted(self):
        """--vad-type all should be accepted by argparse."""
        args = self._make_parser().parse_args(["--vad-type", "all"])
        assert args.vad_type == "all"

    def test_cli_vad_type_invalid_rejected(self):
        """--vad-type invalid should be rejected by argparse."""
        with pytest.raises(SystemExit) as exc_info:
            self._make_parser().parse_args(["--vad-type", "invalid"])
        # argparse exits with code 2 for invalid choices
        assert exc_info.value.code == 2

    def test_cli_vad_type_default_is_none(self):
        """Default --vad-type is None (falls back to config default 'all')."""
        args = self._make_parser().parse_args([])
        assert args.vad_type is None


# ---------------------------------------------------------------------------
# Mock vendor classes (following MockFireRedStreamVad pattern from test_slice_vad.py)
# ---------------------------------------------------------------------------


class MockFireRedStreamVad:
    """Mock FireRedStreamVad for testing without loading model weights."""

    def __init__(self, frame_results: Optional[list] = None) -> None:
        self.frame_results = frame_results or []
        self._initialized = True

    def detect_chunk(self, audio_chunk: Any) -> list:
        return self.frame_results

    @classmethod
    def from_pretrained(cls, model_dir: str, config: Any) -> "MockFireRedStreamVad":
        return cls()


class MockFireRedVad:
    """Mock FireRedVad for testing without loading model weights."""

    def detect(self, audio: Any) -> Tuple[Dict[str, Any], Any]:
        return {"dur": 3.0, "timestamps": [(0.5, 2.5)]}, None

    @classmethod
    def from_pretrained(cls, model_dir: str, config: Any) -> "MockFireRedVad":
        return cls()


class MockFireRedAed:
    """Mock FireRedAed for testing without loading model weights."""

    def detect(self, audio: Any) -> Tuple[Dict[str, Any], Any]:
        return {"dur": 3.0, "event2timestamps": {"speech": [(0.5, 2.5)]}}, None

    @classmethod
    def from_pretrained(cls, model_dir: str, config: Any) -> "MockFireRedAed":
        return cls()


class MockConfig:
    """Mock config object for any VAD config type."""

    def __init__(self, use_gpu: bool = False) -> None:
        self.use_gpu = use_gpu


# ---------------------------------------------------------------------------
# Multi-mode dispatch tests
# ---------------------------------------------------------------------------


class TestSessionVadStateMultiMode:
    """Tests for _SessionVadState multi-mode model dispatch."""

    @pytest.fixture(autouse=True)
    def _patch_firered_imports(self):
        """Patch fireredasr2s imports so _SessionVadState can be instantiated."""
        mock_fireredvad = mock.MagicMock()
        mock_fireredvad.FireRedStreamVad = MockFireRedStreamVad
        mock_fireredvad.FireRedStreamVadConfig = MockConfig
        mock_fireredvad.FireRedVad = MockFireRedVad
        mock_fireredvad.FireRedVadConfig = MockConfig
        mock_fireredvad.FireRedAed = MockFireRedAed
        mock_fireredvad.FireRedAedConfig = MockConfig

        mock_constants = mock.MagicMock()
        mock_constants.FRAME_LENGTH_SAMPLE = 400

        with mock.patch.dict(
            sys.modules,
            {
                "fireredasr2s": mock.MagicMock(),
                "fireredasr2s.fireredvad": mock_fireredvad,
                "fireredasr2s.fireredvad.core": mock.MagicMock(),
                "fireredasr2s.fireredvad.core.constants": mock_constants,
                "fireredasr2s.fireredvad.core.stream_vad_postprocessor": mock.MagicMock(),
            },
        ):
            yield

    def test_stream_vad_only_loads_stream_vad(self):
        """_SessionVadState(vad_type='stream-vad') loads only stream-vad model."""
        from asr2s_grpc.vad_utils import _SessionVadState

        preloaded = {
            "stream-vad": MockFireRedStreamVad(),
            "vad": MockFireRedVad(),
            "aed": MockFireRedAed(),
        }
        model_dirs = {
            "stream-vad": "/models/FireRedVAD/Stream-VAD",
            "vad": "/models/FireRedVAD/VAD",
            "aed": "/models/FireRedVAD/AED",
        }
        state = _SessionVadState(
            vad_type="stream-vad",
            preloaded_models=preloaded,
            model_dirs=model_dirs,
        )
        assert state._stream_vad is not None
        assert state._vad is None
        assert state._aed is None

    def test_vad_only_loads_vad(self):
        """_SessionVadState(vad_type='vad') loads only vad model."""
        from asr2s_grpc.vad_utils import _SessionVadState

        preloaded = {
            "stream-vad": MockFireRedStreamVad(),
            "vad": MockFireRedVad(),
            "aed": MockFireRedAed(),
        }
        state = _SessionVadState(
            vad_type="vad",
            preloaded_models=preloaded,
        )
        assert state._stream_vad is None
        assert state._vad is not None
        assert state._aed is None

    def test_aed_only_loads_aed(self):
        """_SessionVadState(vad_type='aed') loads only aed model."""
        from asr2s_grpc.vad_utils import _SessionVadState

        preloaded = {
            "stream-vad": MockFireRedStreamVad(),
            "vad": MockFireRedVad(),
            "aed": MockFireRedAed(),
        }
        state = _SessionVadState(
            vad_type="aed",
            preloaded_models=preloaded,
        )
        assert state._stream_vad is None
        assert state._vad is None
        assert state._aed is not None

    def test_all_loads_all_three(self):
        """_SessionVadState(vad_type='all') loads all 3 models."""
        from asr2s_grpc.vad_utils import _SessionVadState

        preloaded = {
            "stream-vad": MockFireRedStreamVad(),
            "vad": MockFireRedVad(),
            "aed": MockFireRedAed(),
        }
        model_dirs = {
            "stream-vad": "/models/FireRedVAD/Stream-VAD",
            "vad": "/models/FireRedVAD/VAD",
            "aed": "/models/FireRedVAD/AED",
        }
        state = _SessionVadState(
            vad_type="all",
            preloaded_models=preloaded,
            model_dirs=model_dirs,
        )
        assert state._stream_vad is not None
        assert state._vad is not None
        assert state._aed is not None

    def test_process_slice_audio_vad_calls_detect(self):
        """process_slice_audio() with vad_type='vad' calls detect(), not detect_chunk()."""
        from asr2s_grpc.vad_utils import _SessionVadState

        mock_vad = MockFireRedVad()
        mock_vad.detect = mock.MagicMock(
            return_value=({"dur": 1.0, "timestamps": [(0.1, 0.9)]}, None)
        )

        preloaded = {"vad": mock_vad}
        state = _SessionVadState(
            vad_type="vad",
            preloaded_models=preloaded,
        )

        audio = np.zeros(16000, dtype=np.int16)  # 1 second at 16kHz
        result = state.process_slice_audio(audio, sample_rate=16000)

        mock_vad.detect.assert_called_once()
        assert result is not None

    def test_process_slice_audio_aed_calls_detect(self):
        """process_slice_audio() with vad_type='aed' calls detect() on AED model."""
        from asr2s_grpc.vad_utils import _SessionVadState

        mock_aed = MockFireRedAed()
        mock_aed.detect = mock.MagicMock(
            return_value=(
                {"dur": 1.0, "event2timestamps": {"speech": [(0.1, 0.9)]}},
                None,
            )
        )

        preloaded = {"aed": mock_aed}
        state = _SessionVadState(
            vad_type="aed",
            preloaded_models=preloaded,
        )

        audio = np.zeros(16000, dtype=np.int16)
        result = state.process_slice_audio(audio, sample_rate=16000)

        mock_aed.detect.assert_called_once()
        assert result is not None

    def test_process_slice_audio_stream_vad_calls_detect_chunk(self):
        """process_slice_audio() with vad_type='stream-vad' calls detect_chunk()."""
        from asr2s_grpc.vad_utils import _SessionVadState

        mock_stream = MockFireRedStreamVad()
        mock_stream.detect_chunk = mock.MagicMock(return_value=[])

        preloaded = {"stream-vad": mock_stream}
        model_dirs = {"stream-vad": "/models/FireRedVAD/Stream-VAD"}
        state = _SessionVadState(
            vad_type="stream-vad",
            preloaded_models=preloaded,
            model_dirs=model_dirs,
        )
        # Override to use the mock directly (stream-vad creates a fresh instance)
        state._stream_vad = mock_stream

        audio = np.zeros(16000, dtype=np.int16)
        result = state.process_slice_audio(audio, sample_rate=16000)

        mock_stream.detect_chunk.assert_called_once()
        assert result is not None


# ---------------------------------------------------------------------------
# Logging tests
# ---------------------------------------------------------------------------


class TestVadLogging:
    """Verify logger.info called with 'Loaded' message on successful model load."""

    @pytest.fixture(autouse=True)
    def _patch_firered_imports(self):
        """Patch fireredasr2s imports so _SessionVadState can be instantiated."""
        mock_fireredvad = mock.MagicMock()
        mock_fireredvad.FireRedStreamVad = MockFireRedStreamVad
        mock_fireredvad.FireRedStreamVadConfig = MockConfig
        mock_fireredvad.FireRedVad = MockFireRedVad
        mock_fireredvad.FireRedVadConfig = MockConfig
        mock_fireredvad.FireRedAed = MockFireRedAed
        mock_fireredvad.FireRedAedConfig = MockConfig

        mock_constants = mock.MagicMock()
        mock_constants.FRAME_LENGTH_SAMPLE = 400

        with mock.patch.dict(
            sys.modules,
            {
                "fireredasr2s": mock.MagicMock(),
                "fireredasr2s.fireredvad": mock_fireredvad,
                "fireredasr2s.fireredvad.core": mock.MagicMock(),
                "fireredasr2s.fireredvad.core.constants": mock_constants,
                "fireredasr2s.fireredvad.core.stream_vad_postprocessor": mock.MagicMock(),
            },
        ):
            yield

    def test_initialize_logs_loaded_message(
        self, caplog: pytest.LogCaptureFixture, tmp_path
    ):
        """initialize() should log 'Loaded' on successful model load."""
        from asr2s_grpc.vad_utils import _SessionVadState

        # Create fake model files
        model_dir = tmp_path / "stream-vad"
        model_dir.mkdir()
        (model_dir / "model.pth.tar").write_bytes(b"fake")
        (model_dir / "cmvn.ark").write_bytes(b"fake")

        state = _SessionVadState(vad_type="stream-vad")

        with caplog.at_level(logging.INFO, logger="asr2s_grpc.vad_utils"):
            state.initialize(vad_model_dir={"stream-vad": str(model_dir)})

        loaded_msgs = [r for r in caplog.records if "Loaded" in r.message]
        assert len(loaded_msgs) >= 1, (
            f"Expected at least one 'Loaded' log message, got: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_preload_vad_models_logs_preloaded_message(
        self, caplog: pytest.LogCaptureFixture, tmp_path
    ):
        """preload_vad_models() should log 'Preloaded' on successful model load."""
        from asr2s_grpc.vad_utils import preload_vad_models

        # Create fake model files for stream-vad
        model_dir = tmp_path / "stream-vad"
        model_dir.mkdir()
        (model_dir / "model.pth.tar").write_bytes(b"fake")
        (model_dir / "cmvn.ark").write_bytes(b"fake")

        with caplog.at_level(logging.INFO, logger="asr2s_grpc.vad_utils"):
            models = preload_vad_models(
                model_dirs={"stream-vad": str(model_dir)},
                use_gpu=False,
            )

        preloaded_msgs = [r for r in caplog.records if "Preloaded" in r.message]
        assert len(preloaded_msgs) >= 1, (
            f"Expected 'Preloaded' log message, got: "
            f"{[r.message for r in caplog.records]}"
        )
        assert len(models) == 1
        assert "stream-vad" in models


# ---------------------------------------------------------------------------
# Path validation tests
# ---------------------------------------------------------------------------


class TestPathValidation:
    """Test that missing model files produce errors."""

    @pytest.fixture(autouse=True)
    def _patch_firered_imports(self):
        """Patch fireredasr2s imports."""
        mock_fireredvad = mock.MagicMock()
        mock_fireredvad.FireRedStreamVad = MockFireRedStreamVad
        mock_fireredvad.FireRedStreamVadConfig = MockConfig
        mock_fireredvad.FireRedVad = MockFireRedVad
        mock_fireredvad.FireRedVadConfig = MockConfig
        mock_fireredvad.FireRedAed = MockFireRedAed
        mock_fireredvad.FireRedAedConfig = MockConfig

        mock_constants = mock.MagicMock()
        mock_constants.FRAME_LENGTH_SAMPLE = 400

        with mock.patch.dict(
            sys.modules,
            {
                "fireredasr2s": mock.MagicMock(),
                "fireredasr2s.fireredvad": mock_fireredvad,
                "fireredasr2s.fireredvad.core": mock.MagicMock(),
                "fireredasr2s.fireredvad.core.constants": mock_constants,
                "fireredasr2s.fireredvad.core.stream_vad_postprocessor": mock.MagicMock(),
            },
        ):
            yield

    def test_missing_model_pth_raises_runtime_error(self, tmp_path):
        """Missing model.pth.tar should raise RuntimeError."""
        from asr2s_grpc.vad_utils import _SessionVadState

        model_dir = tmp_path / "stream-vad"
        model_dir.mkdir()
        # Only create cmvn.ark, skip model.pth.tar
        (model_dir / "cmvn.ark").write_bytes(b"fake")

        state = _SessionVadState(vad_type="stream-vad")

        with pytest.raises(RuntimeError, match="model file not found"):
            state.initialize(vad_model_dir={"stream-vad": str(model_dir)})

    def test_missing_cmvn_ark_raises_runtime_error(self, tmp_path):
        """Missing cmvn.ark should raise RuntimeError."""
        from asr2s_grpc.vad_utils import _SessionVadState

        model_dir = tmp_path / "stream-vad"
        model_dir.mkdir()
        # Only create model.pth.tar, skip cmvn.ark
        (model_dir / "model.pth.tar").write_bytes(b"fake")

        state = _SessionVadState(vad_type="stream-vad")

        with pytest.raises(RuntimeError, match="model file not found"):
            state.initialize(vad_model_dir={"stream-vad": str(model_dir)})

    def test_preload_missing_model_raises_runtime_error(self, tmp_path):
        """preload_vad_models() with missing model.pth.tar raises RuntimeError."""
        from asr2s_grpc.vad_utils import preload_vad_models

        model_dir = tmp_path / "empty"
        model_dir.mkdir()

        with pytest.raises(RuntimeError, match="model file not found"):
            preload_vad_models(
                model_dirs={"stream-vad": str(model_dir)},
                use_gpu=False,
            )

    def test_nonexistent_dir_raises_runtime_error(self, tmp_path):
        """preload_vad_models() with nonexistent directory raises RuntimeError."""
        from asr2s_grpc.vad_utils import preload_vad_models

        with pytest.raises(RuntimeError, match="model file not found"):
            preload_vad_models(
                model_dirs={"stream-vad": str(tmp_path / "nonexistent")},
                use_gpu=False,
            )


# ---------------------------------------------------------------------------
# Integration: VadConfig vad_type flows through ApiConfig
# ---------------------------------------------------------------------------


class TestVadTypeInApiConfig:
    """Test that vad_type flows correctly through ApiConfig."""

    def test_api_config_default_vad_type(self):
        """ApiConfig default vad_type should be 'all'."""
        config = ApiConfig()
        assert config.vad.vad_type == "all"

    def test_api_config_explicit_vad_type(self, monkeypatch: pytest.MonkeyPatch):
        """ApiConfig with explicit vad_type should preserve it."""
        monkeypatch.setenv("MODEL_DIR", "/tmp/models")
        config = ApiConfig(vad=VadConfig(vad_type="stream-vad"))
        assert config.vad.vad_type == "stream-vad"

    def test_api_config_vad_type_vad(self, monkeypatch: pytest.MonkeyPatch):
        """ApiConfig with vad_type='vad' should preserve it."""
        monkeypatch.setenv("MODEL_DIR", "/tmp/models")
        config = ApiConfig(vad=VadConfig(vad_type="vad"))
        assert config.vad.vad_type == "vad"

    def test_api_config_vad_type_aed(self, monkeypatch: pytest.MonkeyPatch):
        """ApiConfig with vad_type='aed' should preserve it."""
        monkeypatch.setenv("MODEL_DIR", "/tmp/models")
        config = ApiConfig(vad=VadConfig(vad_type="aed"))
        assert config.vad.vad_type == "aed"


# ---------------------------------------------------------------------------
# Per-request vad_type override tests
# ---------------------------------------------------------------------------


_VALID_PER_REQUEST_TYPES = ("vad", "stream-vad", "aed")


def _validate_client_vad_type(
    raw_vad_type: str,
    server_default: str,
    loaded_models: dict,
) -> tuple:
    """Replicate per-request vad_type validation from grpc_server.StreamingRecognize.

    Returns (effective_vad_type, error_code, error_message).
    If error_code is not None, the request should be rejected.
    """
    client_vad_type = raw_vad_type.strip().lower() if raw_vad_type else ""
    if client_vad_type == "":
        return server_default, None, None
    elif client_vad_type == "all":
        return None, "INVALID_VAD_TYPE", "vad_type 'all' is not allowed for per-session use"
    elif client_vad_type not in _VALID_PER_REQUEST_TYPES:
        return None, "INVALID_VAD_TYPE", "vad_type must be one of: vad, stream-vad, aed"
    elif client_vad_type not in loaded_models:
        return (
            None,
            "VAD_MODEL_UNAVAILABLE",
            "VAD model for '%s' is not loaded on this server" % client_vad_type,
        )
    else:
        return client_vad_type, None, None


class TestPerRequestVadType:
    """Tests for per-request vad_type override logic.

    Tests the validation logic used in StreamingRecognize for client-specified
    vad_type, and verifies _SessionVadState creation with the effective type.
    """

    # Reuse the firered import patching from TestSessionVadStateMultiMode
    @pytest.fixture(autouse=True)
    def _patch_firered_imports(self):
        """Patch fireredasr2s imports so _SessionVadState can be instantiated."""
        mock_fireredvad = mock.MagicMock()
        mock_fireredvad.FireRedStreamVad = MockFireRedStreamVad
        mock_fireredvad.FireRedStreamVadConfig = MockConfig
        mock_fireredvad.FireRedVad = MockFireRedVad
        mock_fireredvad.FireRedVadConfig = MockConfig
        mock_fireredvad.FireRedAed = MockFireRedAed
        mock_fireredvad.FireRedAedConfig = MockConfig

        mock_constants = mock.MagicMock()
        mock_constants.FRAME_LENGTH_SAMPLE = 400

        with mock.patch.dict(
            sys.modules,
            {
                "fireredasr2s": mock.MagicMock(),
                "fireredasr2s.fireredvad": mock_fireredvad,
                "fireredasr2s.fireredvad.core": mock.MagicMock(),
                "fireredasr2s.fireredvad.core.constants": mock_constants,
                "fireredasr2s.fireredvad.core.stream_vad_postprocessor": mock.MagicMock(),
            },
        ):
            yield

    def test_per_request_vad_type_override(self):
        """Mock config with vad_type='aed', verify _SessionVadState created with vad_type='aed'."""
        from asr2s_grpc.vad_utils import _SessionVadState

        # Simulate per-request validation: client sends vad_type='aed'
        effective, err_code, _ = _validate_client_vad_type(
            raw_vad_type="aed",
            server_default="stream-vad",
            loaded_models={"vad": MockFireRedVad(), "stream-vad": MockFireRedStreamVad(), "aed": MockFireRedAed()},
        )
        assert err_code is None
        assert effective == "aed"

        # Create _SessionVadState with the effective vad_type
        preloaded = {
            "stream-vad": MockFireRedStreamVad(),
            "vad": MockFireRedVad(),
            "aed": MockFireRedAed(),
        }
        state = _SessionVadState(
            vad_type=effective,
            preloaded_models=preloaded,
        )
        assert state._aed is not None
        assert state._vad is None
        assert state._stream_vad is None

    def test_per_request_vad_type_empty_uses_server_default(self):
        """Mock config with vad_type='', verify falls back to server default."""
        from asr2s_grpc.vad_utils import _SessionVadState

        # Simulate per-request validation: client sends empty vad_type
        effective, err_code, _ = _validate_client_vad_type(
            raw_vad_type="",
            server_default="stream-vad",
            loaded_models={"vad": MockFireRedVad(), "stream-vad": MockFireRedStreamVad(), "aed": MockFireRedAed()},
        )
        assert err_code is None
        assert effective == "stream-vad"

        # Create _SessionVadState with the server default
        preloaded = {
            "stream-vad": MockFireRedStreamVad(),
            "vad": MockFireRedVad(),
            "aed": MockFireRedAed(),
        }
        model_dirs = {"stream-vad": "/models/FireRedVAD/Stream-VAD"}
        state = _SessionVadState(
            vad_type=effective,
            preloaded_models=preloaded,
            model_dirs=model_dirs,
        )
        assert state._stream_vad is not None
        assert state._vad is None
        assert state._aed is None

    def test_per_request_vad_type_all_rejected(self):
        """Mock config with vad_type='all', verify ErrorResult with code INVALID_VAD_TYPE."""
        from asr2s_grpc import asr_pb2

        effective, err_code, err_msg = _validate_client_vad_type(
            raw_vad_type="all",
            server_default="stream-vad",
            loaded_models={"vad": MockFireRedVad(), "stream-vad": MockFireRedStreamVad(), "aed": MockFireRedAed()},
        )
        assert effective is None
        assert err_code == "INVALID_VAD_TYPE"

        # Verify ErrorResult can be constructed with the error code
        error = asr_pb2.ErrorResult(code=err_code, message=err_msg)
        assert error.code == "INVALID_VAD_TYPE"
        assert "all" in error.message

    def test_per_request_vad_type_invalid_rejected(self):
        """Mock config with vad_type='unknown', verify ErrorResult with code INVALID_VAD_TYPE."""
        from asr2s_grpc import asr_pb2

        effective, err_code, err_msg = _validate_client_vad_type(
            raw_vad_type="unknown",
            server_default="stream-vad",
            loaded_models={"vad": MockFireRedVad(), "stream-vad": MockFireRedStreamVad(), "aed": MockFireRedAed()},
        )
        assert effective is None
        assert err_code == "INVALID_VAD_TYPE"

        # Verify ErrorResult can be constructed with the error code
        error = asr_pb2.ErrorResult(code=err_code, message=err_msg)
        assert error.code == "INVALID_VAD_TYPE"

    def test_per_request_vad_type_unavailable_model(self):
        """Mock config with vad_type='aed' but server only has 'vad' loaded.

        Verify ErrorResult with code VAD_MODEL_UNAVAILABLE."""
        from asr2s_grpc import asr_pb2

        # Server only has 'vad' loaded, not 'aed'
        effective, err_code, err_msg = _validate_client_vad_type(
            raw_vad_type="aed",
            server_default="vad",
            loaded_models={"vad": MockFireRedVad()},
        )
        assert effective is None
        assert err_code == "VAD_MODEL_UNAVAILABLE"

        # Verify ErrorResult can be constructed with the error code
        error = asr_pb2.ErrorResult(code=err_code, message=err_msg)
        assert error.code == "VAD_MODEL_UNAVAILABLE"
        assert "aed" in error.message

    def test_per_request_vad_type_case_insensitive(self):
        """Mock config with vad_type='VAD', verify lowercased and accepted."""
        from asr2s_grpc.vad_utils import _SessionVadState

        # Simulate per-request validation: client sends 'VAD' (uppercase)
        effective, err_code, _ = _validate_client_vad_type(
            raw_vad_type="VAD",
            server_default="stream-vad",
            loaded_models={"vad": MockFireRedVad(), "stream-vad": MockFireRedStreamVad(), "aed": MockFireRedAed()},
        )
        assert err_code is None
        assert effective == "vad"  # lowercased

        # Create _SessionVadState with the lowercased effective vad_type
        preloaded = {
            "stream-vad": MockFireRedStreamVad(),
            "vad": MockFireRedVad(),
            "aed": MockFireRedAed(),
        }
        state = _SessionVadState(
            vad_type=effective,
            preloaded_models=preloaded,
        )
        assert state._vad is not None
        assert state._stream_vad is None
        assert state._aed is None
