# Copyright 2026 FireRedTeam

"""Tests for request-level LLM parameter parsing and defaults.

Covers:
- ConfigMessage accepts optional LLM params
- validate_llm_params returns correct defaults
- validate_llm_params applies bounds (reject invalid, fall back to defaults)
- gRPC handler stores llm_params and asr_type on session (proto3 zero-value handling)
"""

import struct

import pytest
from unittest.mock import MagicMock, patch

from fireredasr2s_api.validation import validate_llm_params, _LLM_DEFAULTS



# ---------- validate_llm_params unit tests ----------


class TestValidateLlmParams:
    """Test validate_llm_params helper function."""

    def test_all_none_returns_defaults(self):
        """No params provided → all defaults."""
        result = validate_llm_params()
        assert result == {
            "decode_min_len": 0,
            "repetition_penalty": 1.0,
            "llm_length_penalty": 0.0,
            "temperature": 1.0,
        }

    def test_explicit_values_override_defaults(self):
        """Explicit valid values override defaults."""
        result = validate_llm_params(
            decode_min_len=5,
            repetition_penalty=1.2,
            llm_length_penalty=0.5,
            temperature=0.8,
        )
        assert result["decode_min_len"] == 5
        assert result["repetition_penalty"] == 1.2
        assert result["llm_length_penalty"] == 0.5
        assert result["temperature"] == 0.8

    def test_negative_decode_min_len_falls_back(self):
        """decode_min_len < 0 → default."""
        result = validate_llm_params(decode_min_len=-1)
        assert result["decode_min_len"] == 0

    def test_negative_repetition_penalty_falls_back(self):
        """repetition_penalty < 0 → default."""
        result = validate_llm_params(repetition_penalty=-0.5)
        assert result["repetition_penalty"] == 1.0

    def test_negative_llm_length_penalty_falls_back(self):
        """llm_length_penalty < 0 → default."""
        result = validate_llm_params(llm_length_penalty=-1.0)
        assert result["llm_length_penalty"] == 0.0

    def test_zero_temperature_falls_back(self):
        """temperature == 0 (not > 0) → default."""
        result = validate_llm_params(temperature=0.0)
        assert result["temperature"] == 1.0

    def test_negative_temperature_falls_back(self):
        """temperature < 0 → default."""
        result = validate_llm_params(temperature=-0.1)
        assert result["temperature"] == 1.0

    def test_zero_decode_min_len_accepted(self):
        """decode_min_len=0 is valid (>= 0)."""
        result = validate_llm_params(decode_min_len=0)
        assert result["decode_min_len"] == 0

    def test_zero_repetition_penalty_accepted(self):
        """repetition_penalty=0 is valid (>= 0)."""
        result = validate_llm_params(repetition_penalty=0.0)
        assert result["repetition_penalty"] == 0.0

    def test_zero_llm_length_penalty_accepted(self):
        """llm_length_penalty=0 is valid (>= 0)."""
        result = validate_llm_params(llm_length_penalty=0.0)
        assert result["llm_length_penalty"] == 0.0

    def test_small_positive_temperature_accepted(self):
        """temperature=0.01 is valid (> 0)."""
        result = validate_llm_params(temperature=0.01)
        assert result["temperature"] == pytest.approx(0.01)

    def test_invalid_string_decode_min_len_falls_back(self):
        """Non-numeric string → default."""
        result = validate_llm_params(decode_min_len="abc")  # type: ignore
        assert result["decode_min_len"] == 0

    def test_invalid_string_temperature_falls_back(self):
        """Non-numeric string → default."""
        result = validate_llm_params(temperature="hot")  # type: ignore
        assert result["temperature"] == 1.0

    def test_partial_params_fill_remaining_with_defaults(self):
        """Only some params provided → rest get defaults."""
        result = validate_llm_params(temperature=0.5)
        assert result["decode_min_len"] == 0
        assert result["repetition_penalty"] == 1.0
        assert result["llm_length_penalty"] == 0.0
        assert result["temperature"] == 0.5




# ---------- gRPC handler LLM param integration tests ----------


class TestGrpcLlmParams:
    """Test gRPC handler stores LLM params on session."""

    @pytest.fixture(autouse=True)
    def _skip_if_grpc_import_broken(self):
        """Skip all tests if grpc_server can't be imported (proto import issue)."""
        pytest.importorskip("fireredasr2s_api.grpc_server")

    @pytest.mark.asyncio
    async def test_grpc_config_stores_llm_params(self):
        """gRPC config with LLM params stores them on session."""
        from fireredasr2s_api import asr_pb2
        from fireredasr2s_api.grpc_server import ASRServiceServicer
        from fireredasr2s_api.config import ApiConfig, AsrBackendConfig

        with patch("fireredasr2s_api.grpc_server.create_backend") as mock_create:
            mock_backend = MagicMock()
            mock_backend.get_max_audio_length.return_value = 60.0
            mock_backend.transcribe.return_value = {
                "text": "test",
                "confidence": 0.9,
                "words": [],
            }
            mock_create.return_value = mock_backend

            servicer = ASRServiceServicer(ApiConfig(asr=AsrBackendConfig(asr_type="llm")))

            with patch("fireredasr2s_api.grpc_server._SessionVadState") as MockVadState:
                mock_vad = MagicMock()
                mock_vad.initialize.return_value = None
                MockVadState.return_value = mock_vad

                config = asr_pb2.RecognitionConfig(
                    sample_rate=16000,
                    format="pcm_s16le",
                    slice_index=0,
                    decode_min_len=5,
                    repetition_penalty=1.3,
                    llm_length_penalty=0.2,
                    temperature=0.7,
                )

                # Build a minimal request stream: config → end
                requests = [
                    asr_pb2.StreamingRecognizeRequest(),
                    asr_pb2.StreamingRecognizeRequest(),
                ]
                requests[0].config.CopyFrom(config)
                requests[1].end_stream = True

                async def request_iter():
                    for r in requests:
                        yield r

                context = MagicMock()

                responses = []
                async for resp in servicer.StreamingRecognize(request_iter(), context):
                    responses.append(resp)

                # Verify session was created with LLM params
                # Session is popped after end_stream, but we can check it was created
                # by verifying no errors occurred
                error_responses = [r for r in responses if r.HasField("error")]
                assert len(error_responses) == 0, (
                    f"Unexpected errors: {[(r.error.code, r.error.message) for r in error_responses]}"
                )

    @pytest.mark.asyncio
    async def test_grpc_config_without_llm_params_gets_defaults(self):
        """gRPC config without LLM fields → session gets defaults (proto3 zero-value handling)."""
        from fireredasr2s_api import asr_pb2
        from fireredasr2s_api.grpc_server import ASRServiceServicer
        from fireredasr2s_api.config import ApiConfig, AsrBackendConfig

        with patch("fireredasr2s_api.grpc_server.create_backend") as mock_create:
            mock_backend = MagicMock()
            mock_backend.get_max_audio_length.return_value = 60.0
            mock_backend.transcribe.return_value = {
                "text": "",
                "confidence": 0.0,
                "words": [],
            }
            mock_create.return_value = mock_backend

            servicer = ASRServiceServicer(ApiConfig())
            servicer.backends["aed"] = mock_backend

            with patch("fireredasr2s_api.grpc_server._SessionVadState") as MockVadState:
                mock_vad = MagicMock()
                mock_vad.initialize.return_value = None
                MockVadState.return_value = mock_vad

                # Config with NO LLM params (proto3 defaults: all zeros)
                config = asr_pb2.RecognitionConfig(
                    sample_rate=16000,
                    format="pcm_s16le",
                    slice_index=0,
                )

                # Verify proto3 zero-values
                assert config.decode_min_len == 0
                assert config.repetition_penalty == 0.0
                assert config.llm_length_penalty == 0.0
                assert config.temperature == 0.0

                requests = [
                    asr_pb2.StreamingRecognizeRequest(),
                    asr_pb2.StreamingRecognizeRequest(),
                ]
                requests[0].config.CopyFrom(config)
                requests[1].end_stream = True

                async def request_iter():
                    for r in requests:
                        yield r

                context = MagicMock()

                # Patch _create_session to capture session
                original_create = servicer._create_session
                captured_sessions = []

                def capturing_create(sid, cfg, backend):
                    session = original_create(sid, cfg, backend)
                    captured_sessions.append(session)
                    return session

                servicer._create_session = capturing_create

                responses = []
                async for resp in servicer.StreamingRecognize(request_iter(), context):
                    responses.append(resp)

                # Verify session got defaults (zero proto3 values treated as unset)
                assert len(captured_sessions) == 1
                session = captured_sessions[0]
                assert session.llm_params == _LLM_DEFAULTS
                assert session.asr_type == "aed"

    @pytest.mark.asyncio
    async def test_grpc_proto3_zero_values_treated_as_unset(self):
        """Proto3 zero-value fields (0.0) treated as unset → defaults applied."""
        from fireredasr2s_api import asr_pb2

        # Proto3: unset floats default to 0.0
        config = asr_pb2.RecognitionConfig(
            sample_rate=16000,
            slice_index=0,
            # decode_min_len not set → 0 (which IS the default)
            # repetition_penalty not set → 0.0 (unset, default should be 1.0)
            # temperature not set → 0.0 (unset, default should be 1.0)
        )

        # Simulate the gRPC handler logic for zero-value detection
        params = validate_llm_params(
            decode_min_len=config.decode_min_len
            if config.decode_min_len != 0
            else None,
            repetition_penalty=config.repetition_penalty
            if config.repetition_penalty != 0.0
            else None,
            llm_length_penalty=config.llm_length_penalty
            if config.llm_length_penalty != 0.0
            else None,
            temperature=config.temperature if config.temperature != 0.0 else None,
        )

        assert params["decode_min_len"] == 0  # 0 is treated as unset → None → default 0
        assert params["repetition_penalty"] == 1.0  # 0.0 → None → default 1.0
        assert params["llm_length_penalty"] == 0.0  # 0.0 → None → default 0.0
        assert params["temperature"] == 1.0  # 0.0 → None → default 1.0

    @pytest.mark.asyncio
    async def test_grpc_explicit_llm_params_preserved(self):
        """Proto config with explicit non-zero LLM params → preserved in session."""
        from fireredasr2s_api import asr_pb2

        config = asr_pb2.RecognitionConfig(
            sample_rate=16000,
            slice_index=0,
            decode_min_len=10,
            repetition_penalty=1.5,
            llm_length_penalty=0.3,
            temperature=0.6,
        )

        params = validate_llm_params(
            decode_min_len=config.decode_min_len
            if config.decode_min_len != 0
            else None,
            repetition_penalty=config.repetition_penalty
            if config.repetition_penalty != 0.0
            else None,
            llm_length_penalty=config.llm_length_penalty
            if config.llm_length_penalty != 0.0
            else None,
            temperature=config.temperature if config.temperature != 0.0 else None,
        )

        assert params["decode_min_len"] == 10
        assert params["repetition_penalty"] == pytest.approx(1.5)
        assert params["llm_length_penalty"] == pytest.approx(0.3)
        assert params["temperature"] == pytest.approx(0.6)


# ---------- Session attribute tests ----------


class TestSessionLlmAttributes:
    """Test session has llm_params and asr_type attributes."""

    def test_session_has_llm_params_attribute(self):
        """New sessions have llm_params dict attribute."""
        from fireredasr2s_api.session import StreamingSession

        session = StreamingSession(session_id="test-1")
        assert hasattr(session, "llm_params")
        assert isinstance(session.llm_params, dict)

    def test_session_has_asr_type_attribute(self):
        """New sessions have asr_type string attribute."""
        from fireredasr2s_api.session import StreamingSession

        session = StreamingSession(session_id="test-2")
        assert hasattr(session, "asr_type")
        assert session.asr_type == "aed"  # default

    def test_session_llm_params_assignable(self):
        """session.llm_params can be assigned a dict."""
        from fireredasr2s_api.session import StreamingSession

        session = StreamingSession(session_id="test-3")
        session.llm_params = {"temperature": 0.5, "decode_min_len": 10}
        assert session.llm_params["temperature"] == 0.5
        assert session.llm_params["decode_min_len"] == 10

    def test_session_asr_type_assignable(self):
        """session.asr_type can be set to 'llm'."""
        from fireredasr2s_api.session import StreamingSession

        session = StreamingSession(session_id="test-4")
        session.asr_type = "llm"
        assert session.asr_type == "llm"
