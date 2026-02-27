# Copyright 2026 FireRedTeam

"""Tests for LLMBackend implementation.

Covers:
- __init__ loads model via FireRedAsr2.from_pretrained("llm", ...)
- get_max_audio_length returns 40.0
- transcribe returns dict with text and empty words list (no timestamps)
- Per-request llm_params create a new config without mutating shared config
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call


# ---------- Initialization tests ----------


class TestLLMBackendInit:
    """Test LLMBackend initialization."""

    @patch("fireredasr2s.fireredasr2.FireRedAsr2", autospec=False)
    @patch("fireredasr2s.fireredasr2.FireRedAsr2Config", autospec=False)
    def test_init_calls_from_pretrained_with_llm(self, MockConfig, MockAsr2):
        """__init__ calls FireRedAsr2.from_pretrained('llm', model_dir, config)."""
        # Need to patch at import time inside _load_model
        with patch("fireredasr2s.fireredasr2.FireRedAsr2.from_pretrained") as mock_from:
            mock_from.return_value = MagicMock()

            from fireredasr2s_api.backend import LLMBackend

            with patch.object(LLMBackend, "_load_model") as mock_load:
                backend = LLMBackend("/path/to/llm", use_gpu=False)
                mock_load.assert_called_once()

    def test_init_stores_attributes(self):
        """__init__ stores model_dir, use_gpu, default_config."""
        from fireredasr2s_api.backend import LLMBackend

        with patch.object(LLMBackend, "_load_model"):
            backend = LLMBackend(
                "/path/to/llm",
                use_gpu=False,
                config={"temperature": 0.5},
            )
            assert backend.model_dir == "/path/to/llm"
            assert backend.use_gpu is False
            assert backend.default_config == {"temperature": 0.5}

    def test_init_default_config_is_empty_dict(self):
        """config=None → default_config is empty dict."""
        from fireredasr2s_api.backend import LLMBackend

        with patch.object(LLMBackend, "_load_model"):
            backend = LLMBackend("/path/to/llm")
            assert backend.default_config == {}

    def test_load_model_calls_from_pretrained(self):
        """_load_model calls FireRedAsr2.from_pretrained('llm', ...)."""
        with patch("fireredasr2s.fireredasr2.FireRedAsr2.from_pretrained") as mock_from:
            with patch("fireredasr2s.fireredasr2.FireRedAsr2Config") as MockConfig:
                mock_model = MagicMock()
                mock_from.return_value = mock_model
                mock_config_instance = MagicMock()
                MockConfig.return_value = mock_config_instance

                from fireredasr2s_api.backend import LLMBackend

                backend = LLMBackend.__new__(LLMBackend)
                backend.model_dir = "/path/to/llm"
                backend.use_gpu = True
                backend.default_config = {}
                backend._load_model()

                mock_from.assert_called_once_with(
                    "llm", "/path/to/llm", mock_config_instance
                )
                assert backend.model is mock_model

    def test_load_model_propagation_on_failure(self):
        """_load_model raises when from_pretrained fails."""
        with patch(
            "fireredasr2s.fireredasr2.FireRedAsr2.from_pretrained",
            side_effect=RuntimeError("Model not found"),
        ):
            with patch("fireredasr2s.fireredasr2.FireRedAsr2Config"):
                from fireredasr2s_api.backend import LLMBackend

                backend = LLMBackend.__new__(LLMBackend)
                backend.model_dir = "/nonexistent"
                backend.use_gpu = False
                backend.default_config = {}

                with pytest.raises(RuntimeError, match="Model not found"):
                    backend._load_model()


# ---------- Max audio length tests ----------


class TestLLMBackendMaxAudioLength:
    """Test get_max_audio_length returns 40s."""

    def test_max_audio_length_is_40(self):
        """LLM backend max audio length is 40 seconds."""
        from fireredasr2s_api.backend import LLMBackend

        with patch.object(LLMBackend, "_load_model"):
            backend = LLMBackend("/path/to/llm")
            assert backend.get_max_audio_length() == 40.0

    def test_max_audio_length_type_is_float(self):
        """get_max_audio_length returns a float."""
        from fireredasr2s_api.backend import LLMBackend

        with patch.object(LLMBackend, "_load_model"):
            backend = LLMBackend("/path/to/llm")
            result = backend.get_max_audio_length()
            assert isinstance(result, float)


# ---------- Transcribe output format tests ----------


class TestLLMBackendTranscribeOutput:
    """Test transcribe returns correct format: text + empty words list."""

    def _make_backend(self):
        """Create an LLMBackend with mocked model."""
        from fireredasr2s_api.backend import LLMBackend

        with patch.object(LLMBackend, "_load_model"):
            backend = LLMBackend("/path/to/llm")
        backend.model = MagicMock()
        return backend

    def test_transcribe_returns_text_and_words(self):
        """transcribe returns dict with 'text' and 'words' keys."""
        backend = self._make_backend()
        backend.model.transcribe.return_value = [
            {"uttid": "stream_0001", "text": "hello world", "rtf": "0.05"}
        ]

        audio = np.zeros(16000, dtype=np.int16)
        result = backend.transcribe(audio, 16000)

        assert "text" in result
        assert "words" in result
        assert result["text"] == "hello world"
        assert result["words"] == []

    def test_transcribe_no_confidence_key(self):
        """LLM transcribe result does NOT contain 'confidence' key."""
        backend = self._make_backend()
        backend.model.transcribe.return_value = [
            {"uttid": "stream_0001", "text": "test", "rtf": "0.05"}
        ]

        audio = np.zeros(16000, dtype=np.int16)
        result = backend.transcribe(audio, 16000)

        assert "confidence" not in result

    def test_transcribe_no_timestamp_key(self):
        """LLM transcribe result does NOT contain 'timestamp' key."""
        backend = self._make_backend()
        backend.model.transcribe.return_value = [
            {"uttid": "stream_0001", "text": "test", "rtf": "0.05"}
        ]

        audio = np.zeros(16000, dtype=np.int16)
        result = backend.transcribe(audio, 16000)

        assert "timestamp" not in result

    def test_transcribe_empty_results(self):
        """transcribe returns empty text and empty words when model returns nothing."""
        backend = self._make_backend()
        backend.model.transcribe.return_value = []

        audio = np.zeros(16000, dtype=np.int16)
        result = backend.transcribe(audio, 16000)

        assert result["text"] == ""
        assert result["words"] == []

    def test_transcribe_words_always_empty_list(self):
        """words is always an empty list regardless of return_timestamps flag."""
        backend = self._make_backend()
        backend.model.transcribe.return_value = [
            {"uttid": "stream_0001", "text": "你好世界", "rtf": "0.05"}
        ]

        audio = np.zeros(16000, dtype=np.int16)
        result = backend.transcribe(audio, 16000, return_timestamps=True)

        assert result["words"] == []

    def test_transcribe_passes_audio_as_tuple(self):
        """transcribe passes (sample_rate, audio) tuple to model."""
        backend = self._make_backend()
        backend.model.transcribe.return_value = [
            {"uttid": "stream_0001", "text": "test", "rtf": "0.05"}
        ]

        audio = np.zeros(8000, dtype=np.int16)
        backend.transcribe(audio, 16000)

        args = backend.model.transcribe.call_args
        uttids, wav_paths = args[0]
        assert uttids == ["stream_0001"]
        assert len(wav_paths) == 1
        sr, wav_data = wav_paths[0]
        assert sr == 16000
        assert np.array_equal(wav_data, audio)

    def test_transcribe_propagates_exception(self):
        """transcribe raises when model.transcribe fails."""
        backend = self._make_backend()
        backend.model.transcribe.side_effect = RuntimeError("CUDA OOM")

        audio = np.zeros(16000, dtype=np.int16)
        with pytest.raises(RuntimeError, match="CUDA OOM"):
            backend.transcribe(audio, 16000)


# ---------- Per-request param override tests ----------


class TestLLMBackendParamOverride:
    """Test per-request llm_params create new config without mutating shared config."""

    def _make_backend(self, default_config=None):
        """Create LLMBackend with mocked model and optional default config."""
        from fireredasr2s_api.backend import LLMBackend

        with patch.object(LLMBackend, "_load_model"):
            backend = LLMBackend("/path/to/llm", config=default_config)
        backend.model = MagicMock()
        backend.model.transcribe.return_value = [
            {"uttid": "stream_0001", "text": "test", "rtf": "0.05"}
        ]
        # Set an initial config object on the model
        backend.model.config = MagicMock(name="original_config")
        return backend

    def test_no_params_uses_default_config(self):
        """No llm_params → model.config is not swapped."""
        backend = self._make_backend()
        original_config = backend.model.config

        audio = np.zeros(16000, dtype=np.int16)
        backend.transcribe(audio, 16000)

        # Config should remain the same reference
        assert backend.model.config is original_config

    def test_params_temporarily_swap_config(self):
        """llm_params → config is swapped for call then restored."""
        backend = self._make_backend()
        original_config = backend.model.config

        audio = np.zeros(16000, dtype=np.int16)

        with patch("fireredasr2s.fireredasr2.FireRedAsr2Config") as MockConfig:
            new_config = MagicMock(name="request_config")
            MockConfig.return_value = new_config

            backend.transcribe(audio, 16000, llm_params={"temperature": 0.5})

        # After call, original config is restored
        assert backend.model.config is original_config

    def test_params_restored_on_exception(self):
        """Config is restored even if transcribe raises."""
        backend = self._make_backend()
        original_config = backend.model.config
        backend.model.transcribe.side_effect = RuntimeError("fail")

        audio = np.zeros(16000, dtype=np.int16)

        with patch("fireredasr2s.fireredasr2.FireRedAsr2Config"):
            with pytest.raises(RuntimeError):
                backend.transcribe(audio, 16000, llm_params={"temperature": 0.5})

        # Config is restored despite exception
        assert backend.model.config is original_config

    def test_param_override_merges_with_defaults(self):
        """Per-request params merge with default_config (request wins)."""
        backend = self._make_backend(
            default_config={"temperature": 0.8, "decode_min_len": 5}
        )

        audio = np.zeros(16000, dtype=np.int16)

        with patch("fireredasr2s.fireredasr2.FireRedAsr2Config") as MockConfig:
            backend.transcribe(audio, 16000, llm_params={"temperature": 0.3})

            # Check the config was created with merged values
            config_kwargs = MockConfig.call_args[1]
            assert config_kwargs["temperature"] == 0.3  # overridden
            assert config_kwargs["decode_min_len"] == 5  # from default

    def test_default_config_not_mutated(self):
        """Per-request params do not mutate default_config."""
        default = {"temperature": 0.8}
        backend = self._make_backend(default_config=default)

        audio = np.zeros(16000, dtype=np.int16)

        with patch("fireredasr2s.fireredasr2.FireRedAsr2Config"):
            backend.transcribe(audio, 16000, llm_params={"temperature": 0.3})

        assert default == {"temperature": 0.8}


# ---------- Factory function tests ----------


class TestCreateBackendLLM:
    """Test create_backend with llm type."""

    def test_create_backend_returns_llm(self):
        """create_backend('llm', ...) returns LLMBackend."""
        from fireredasr2s_api.backend import create_backend, LLMBackend

        with patch.object(LLMBackend, "_load_model"):
            backend = create_backend("llm", "/path/to/llm")
            assert isinstance(backend, LLMBackend)

    def test_create_backend_passes_kwargs(self):
        """create_backend passes kwargs like use_gpu and config to LLMBackend."""
        from fireredasr2s_api.backend import create_backend, LLMBackend

        with patch.object(LLMBackend, "_load_model"):
            backend = create_backend(
                "llm",
                "/path/to/llm",
                use_gpu=False,
                config={"temperature": 0.5},
            )
            assert backend.use_gpu is False
            assert backend.default_config == {"temperature": 0.5}
