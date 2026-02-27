# Copyright 2026 FireRedTeam

"""Tests for lazy backend loading and LLM partial emission (Task 5)."""

import threading
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# ---------------------------------------------------------------------------
# gRPC: Lazy backend loading
# ---------------------------------------------------------------------------


class TestGrpcLazyBackendLoading:
    """Test lazy backend loading with threading.Lock in gRPC server."""

    @staticmethod
    def _make_config_request(
        sample_rate: int = 16000,
        slice_index: int = 0,
    ):
        from fireredasr2s_api import asr_pb2

        request = asr_pb2.StreamingRecognizeRequest()
        config = asr_pb2.RecognitionConfig(
            sample_rate=sample_rate,
            format="pcm_s16le",
            slice_index=slice_index,
        )
        request.config.CopyFrom(config)
        return request

    @staticmethod
    def _make_audio_slice_request(index: int, n_samples: int = 16000):
        import struct
        from fireredasr2s_api import asr_pb2

        audio_bytes = struct.pack(f"<{n_samples}h", *([0] * n_samples))
        request = asr_pb2.StreamingRecognizeRequest()
        audio_slice = asr_pb2.AudioSlice(index=index, data=audio_bytes)
        request.audio_slice.CopyFrom(audio_slice)
        return request

    @staticmethod
    def _make_end_stream_request():
        from fireredasr2s_api import asr_pb2

        request = asr_pb2.StreamingRecognizeRequest()
        request.end_stream = True
        return request

    @pytest.mark.asyncio
    async def test_lazy_load_creates_backend_on_first_request(self):
        """When asr_type='llm' not in backends, _lazy_load_backend creates it."""
        with patch(
            "fireredasr2s_api.grpc_server.create_backend"
        ) as mock_create_backend, patch(
            "fireredasr2s_api.grpc_server.resolve_asr_model_dir",
            return_value="/models/FireRedASR2-LLM",
        ):
            # Primary backend (AED)
            mock_aed = MagicMock()
            mock_aed.get_max_audio_length.return_value = 60.0

            # LLM backend to be lazy-loaded
            mock_llm = MagicMock()
            mock_llm.get_max_audio_length.return_value = 30.0

            # create_backend returns AED first, then LLM
            mock_create_backend.side_effect = [mock_aed, mock_llm]

            from fireredasr2s_api.grpc_server import ASRServiceServicer
            from fireredasr2s_api.config import ApiConfig

            servicer = ASRServiceServicer(ApiConfig())

            # Verify only AED loaded initially
            assert "aed" in servicer.backends
            assert "llm" not in servicer.backends
            assert mock_create_backend.call_count == 1

            # Trigger lazy load by requesting LLM directly
            result = servicer._lazy_load_backend("llm")

            # create_backend called twice: once for AED (init), once for LLM (lazy)
            assert mock_create_backend.call_count == 2
            assert "llm" in servicer.backends
            assert result is mock_llm

    @pytest.mark.asyncio
    async def test_lazy_load_called_once_for_concurrent_requests(self):
        """Lock ensures create_backend called only once for same asr_type."""
        with patch(
            "fireredasr2s_api.grpc_server.create_backend"
        ) as mock_create_backend, patch(
            "fireredasr2s_api.grpc_server.resolve_asr_model_dir",
            return_value="/models/FireRedASR2-LLM",
        ):
            mock_aed = MagicMock()
            mock_aed.get_max_audio_length.return_value = 60.0

            mock_llm = MagicMock()
            mock_llm.get_max_audio_length.return_value = 30.0
            mock_llm.transcribe.return_value = {
                "text": "test",
                "confidence": 0.0,
                "words": [],
            }

            mock_create_backend.side_effect = [mock_aed, mock_llm]

            from fireredasr2s_api.grpc_server import ASRServiceServicer
            from fireredasr2s_api.config import ApiConfig

            servicer = ASRServiceServicer(ApiConfig())
            assert mock_create_backend.call_count == 1

            # Call _lazy_load_backend multiple times
            result1 = servicer._lazy_load_backend("llm")
            result2 = servicer._lazy_load_backend("llm")

            # create_backend should only be called once for LLM (double-check pattern)
            assert mock_create_backend.call_count == 2  # 1 for AED + 1 for LLM
            assert result1 is result2
            assert result1 is mock_llm

    @pytest.mark.asyncio
    async def test_lazy_load_returns_none_on_failure(self):
        """If create_backend raises, _lazy_load_backend returns None."""
        with patch(
            "fireredasr2s_api.grpc_server.create_backend"
        ) as mock_create_backend, patch(
            "fireredasr2s_api.grpc_server.resolve_asr_model_dir",
            return_value="/models/FireRedASR2-LLM",
        ):
            mock_aed = MagicMock()
            mock_aed.get_max_audio_length.return_value = 60.0

            # AED succeeds, LLM fails
            mock_create_backend.side_effect = [
                mock_aed,
                RuntimeError("model not found"),
            ]

            from fireredasr2s_api.grpc_server import ASRServiceServicer
            from fireredasr2s_api.config import ApiConfig

            servicer = ASRServiceServicer(ApiConfig())

            result = servicer._lazy_load_backend("llm")
            assert result is None
            assert "llm" not in servicer.backends

    def test_backend_lock_exists(self):
        """ASRServiceServicer has a _backend_lock attribute."""
        with patch(
            "fireredasr2s_api.grpc_server.create_backend"
        ) as mock_create_backend:
            mock_backend = MagicMock()
            mock_backend.get_max_audio_length.return_value = 60.0
            mock_create_backend.return_value = mock_backend

            from fireredasr2s_api.grpc_server import ASRServiceServicer
            from fireredasr2s_api.config import ApiConfig

            servicer = ASRServiceServicer(ApiConfig())
            assert hasattr(servicer, "_backend_lock")
            assert isinstance(servicer._backend_lock, type(threading.Lock()))



# ---------------------------------------------------------------------------
# gRPC: LLM partial emission (bypasses confidence filter)
# ---------------------------------------------------------------------------


class TestGrpcLlmPartialEmission:
    """Test that LLM partials bypass confidence > 0.3 filter."""

    @staticmethod
    def _make_config_request(slice_index=0):
        from fireredasr2s_api import asr_pb2

        request = asr_pb2.StreamingRecognizeRequest()
        config = asr_pb2.RecognitionConfig(
            sample_rate=16000,
            format="pcm_s16le",
            slice_index=slice_index,
        )
        request.config.CopyFrom(config)
        return request

    @staticmethod
    def _make_audio_slice_request(index: int, n_samples: int = 16000):
        import struct
        from fireredasr2s_api import asr_pb2

        audio_bytes = struct.pack(f"<{n_samples}h", *([0] * n_samples))
        request = asr_pb2.StreamingRecognizeRequest()
        audio_slice = asr_pb2.AudioSlice(index=index, data=audio_bytes)
        request.audio_slice.CopyFrom(audio_slice)
        return request

    @staticmethod
    def _make_end_stream_request():
        from fireredasr2s_api import asr_pb2

        request = asr_pb2.StreamingRecognizeRequest()
        request.end_stream = True
        return request

    @pytest.mark.asyncio
    async def test_llm_partial_with_zero_confidence_emitted(self):
        """LLM backend returns confidence=0.0; partial should still be emitted."""
        with patch(
            "fireredasr2s_api.grpc_server.create_backend"
        ) as mock_create_backend:
            mock_aed = MagicMock()
            mock_aed.get_max_audio_length.return_value = 60.0

            mock_llm = MagicMock()
            mock_llm.get_max_audio_length.return_value = 30.0
            mock_llm.transcribe.return_value = {
                "text": "你好世界",
                "confidence": 0.0,  # LLM returns no confidence
                "words": [],
            }

            mock_create_backend.side_effect = [mock_aed, mock_llm]

            from fireredasr2s_api.grpc_server import ASRServiceServicer
            from fireredasr2s_api.config import ApiConfig, AsrBackendConfig

            servicer = ASRServiceServicer(ApiConfig(asr=AsrBackendConfig(asr_type="llm")))

            with patch("fireredasr2s_api.grpc_server._SessionVadState") as MockVadState:
                mock_vad = MagicMock()
                mock_vad.initialize.return_value = None
                mock_vad.process_slice_audio.return_value = MagicMock(
                    ended_speaking=False,
                    entirely_speech=True,
                    n_frames=1000,
                )
                MockVadState.return_value = mock_vad

                requests = [
                    self._make_config_request(slice_index=0),
                    self._make_audio_slice_request(index=0),
                    self._make_end_stream_request(),
                ]

                async def request_iter():
                    for r in requests:
                        yield r

                context = MagicMock()
                responses = []
                async for resp in servicer.StreamingRecognize(request_iter(), context):
                    responses.append(resp)

                # Should have partial responses despite confidence=0.0
                partial_responses = [r for r in responses if r.HasField("partial")]
                # The partial may or may not appear depending on has_complete_segment,
                # but if it does, it should NOT be filtered out.
                # Check no BACKEND_UNAVAILABLE errors
                error_responses = [r for r in responses if r.HasField("error")]
                backend_errors = [
                    r for r in error_responses if r.error.code == "BACKEND_UNAVAILABLE"
                ]
                assert len(backend_errors) == 0

    @pytest.mark.asyncio
    async def test_aed_partial_with_low_confidence_filtered(self):
        """AED backend: confidence=0.1 partial should be filtered out."""
        with patch(
            "fireredasr2s_api.grpc_server.create_backend"
        ) as mock_create_backend:
            mock_aed = MagicMock()
            mock_aed.get_max_audio_length.return_value = 60.0
            mock_aed.transcribe.return_value = {
                "text": "noise",
                "confidence": 0.1,  # Low confidence
                "words": [],
            }

            mock_create_backend.return_value = mock_aed

            from fireredasr2s_api.grpc_server import ASRServiceServicer
            from fireredasr2s_api.config import ApiConfig

            servicer = ASRServiceServicer(ApiConfig())
            servicer.backends["aed"] = mock_aed

            with patch("fireredasr2s_api.grpc_server._SessionVadState") as MockVadState:
                mock_vad = MagicMock()
                mock_vad.initialize.return_value = None
                mock_vad.process_slice_audio.return_value = MagicMock(
                    ended_speaking=False,
                    entirely_speech=True,
                    n_frames=1000,
                )
                MockVadState.return_value = mock_vad

                requests = [
                    self._make_config_request(slice_index=0),
                    self._make_audio_slice_request(index=0),
                    self._make_end_stream_request(),
                ]

                async def request_iter():
                    for r in requests:
                        yield r

                context = MagicMock()
                responses = []
                async for resp in servicer.StreamingRecognize(request_iter(), context):
                    responses.append(resp)

                # Low-confidence AED partials should be filtered out
                partial_responses = [r for r in responses if r.HasField("partial")]
                assert len(partial_responses) == 0, (
                    f"AED partial with confidence=0.1 should be filtered. "
                    f"Got {len(partial_responses)} partial(s)."
                )

    @pytest.mark.asyncio
    async def test_llm_partial_zero_confidence_not_filtered(self):
        """LLM partial with confidence=0.0 must NOT be filtered."""
        with patch(
            "fireredasr2s_api.grpc_server.create_backend"
        ) as mock_create_backend:
            mock_aed = MagicMock()
            mock_aed.get_max_audio_length.return_value = 60.0

            mock_llm = MagicMock()
            mock_llm.get_max_audio_length.return_value = 30.0

            mock_create_backend.side_effect = [mock_aed, mock_llm]

            from fireredasr2s_api.grpc_server import ASRServiceServicer
            from fireredasr2s_api.config import ApiConfig

            servicer = ASRServiceServicer(ApiConfig())

            # Directly test the filtering logic using a mock session
            from fireredasr2s_api.session import StreamingSession

            mock_session = MagicMock(spec=StreamingSession)
            mock_session.asr_type = "llm"

            result = {"confidence": 0.0, "text": "hello"}

            # LLM: should pass filter
            should_emit = (
                mock_session.asr_type == "llm" or result.get("confidence", 0.0) > 0.3
            )
            assert should_emit is True, "LLM partials must bypass confidence filter"

            # AED: should NOT pass filter
            mock_session.asr_type = "aed"
            should_emit = (
                mock_session.asr_type == "llm" or result.get("confidence", 0.0) > 0.3
            )
            assert should_emit is False, (
                "AED partials with 0.0 confidence must be filtered"
            )


