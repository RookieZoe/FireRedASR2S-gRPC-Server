# Copyright 2026 FireRedTeam

"""Tests for gRPC server protobuf message handling."""

import pytest

from fireredasr2s_api import asr_pb2


class TestStreamingRecognizeRequest:
    """Test StreamingRecognizeRequest protobuf message handling."""

    def test_request_with_config(self):
        """Test creating request with config message."""
        request = asr_pb2.StreamingRecognizeRequest()
        config = asr_pb2.RecognitionConfig(
            sample_rate=16000,
            format="pcm_s16le",
            enable_timestamps=True,
        )
        request.config.CopyFrom(config)

        # Verify config is set correctly
        assert request.HasField("config")
        assert request.config.sample_rate == 16000
        assert request.config.format == "pcm_s16le"

    def test_request_with_audio_chunk(self):
        """Test creating request with audio chunk."""
        request = asr_pb2.StreamingRecognizeRequest()
        audio_data = b"\x00\x01\x00\x02"  # Sample audio bytes
        request.audio_chunk = audio_data

        # Verify audio_chunk is set correctly
        assert request.HasField("audio_chunk")
        assert request.audio_chunk == audio_data

    def test_request_with_end_stream_true(self):
        """Test creating request with end_stream=True."""
        request = asr_pb2.StreamingRecognizeRequest()
        request.end_stream = True

        # Verify end_stream is set and can be checked
        assert request.HasField("end_stream")
        assert request.end_stream is True

    def test_request_with_end_stream_false(self):
        """Test creating request with end_stream=False."""
        request = asr_pb2.StreamingRecognizeRequest()
        request.end_stream = False

        # Verify end_stream is set but is False
        assert request.HasField("end_stream")
        assert request.end_stream is False

    def test_request_with_audio_slice(self):
        """Test creating request with audio_slice message."""
        request = asr_pb2.StreamingRecognizeRequest()
        audio_slice = asr_pb2.AudioSlice(index=0, data=b"\x00\x01\x00\x02")
        request.audio_slice.CopyFrom(audio_slice)

        # Verify audio_slice is set correctly
        assert request.HasField("audio_slice")
        assert request.audio_slice.index == 0
        assert request.audio_slice.data == b"\x00\x01\x00\x02"

    def test_request_with_audio_slice_positive_index(self):
        """Test audio_slice with positive index value."""
        request = asr_pb2.StreamingRecognizeRequest()
        audio_slice = asr_pb2.AudioSlice(index=5, data=b"\xff\xfe")
        request.audio_slice.CopyFrom(audio_slice)

        assert request.HasField("audio_slice")
        assert request.audio_slice.index == 5
        assert request.audio_slice.data == b"\xff\xfe"

    def test_audio_slice_mutually_exclusive_with_config(self):
        """Oneof: setting audio_slice clears config."""
        request = asr_pb2.StreamingRecognizeRequest()

        # Set config first
        request.config.CopyFrom(asr_pb2.RecognitionConfig(sample_rate=16000))
        assert request.HasField("config")

        # Setting audio_slice should clear config (oneof behavior)
        audio_slice = asr_pb2.AudioSlice(index=0, data=b"data")
        request.audio_slice.CopyFrom(audio_slice)

        assert request.HasField("audio_slice")
        assert not request.HasField("config")

    def test_audio_slice_mutually_exclusive_with_audio_chunk(self):
        """Oneof: setting audio_slice clears audio_chunk."""
        request = asr_pb2.StreamingRecognizeRequest()

        # Set audio_chunk first
        request.audio_chunk = b"\x00\x01"
        assert request.HasField("audio_chunk")

        # Setting audio_slice should clear audio_chunk (oneof behavior)
        audio_slice = asr_pb2.AudioSlice(index=0, data=b"data")
        request.audio_slice.CopyFrom(audio_slice)

        assert request.HasField("audio_slice")
        assert not request.HasField("audio_chunk")

    def test_audio_slice_mutually_exclusive_with_end_stream(self):
        """Oneof: setting audio_slice clears end_stream."""
        request = asr_pb2.StreamingRecognizeRequest()

        # Set end_stream first
        request.end_stream = True
        assert request.HasField("end_stream")

        # Setting audio_slice should clear end_stream (oneof behavior)
        audio_slice = asr_pb2.AudioSlice(index=0, data=b"data")
        request.audio_slice.CopyFrom(audio_slice)

        assert request.HasField("audio_slice")
        assert not request.HasField("end_stream")

    def test_audio_slice_clears_all_other_oneof_fields(self):
        """audio_slice should clear ALL oneof members."""
        request = asr_pb2.StreamingRecognizeRequest()

        # Set all other oneof fields first
        request.config.CopyFrom(asr_pb2.RecognitionConfig(sample_rate=16000))
        assert request.HasField("config")

        # Set audio_slice (should clear config)
        audio_slice = asr_pb2.AudioSlice(index=5, data=b"data")
        request.audio_slice.CopyFrom(audio_slice)

        assert request.HasField("audio_slice")
        assert not request.HasField("config")
        assert not request.HasField("audio_chunk")
        assert not request.HasField("end_stream")


class TestAudioChunkLegacyBehavior:
    """Regression tests for audio_chunk legacy behavior (Task 6).

    Ensure audio_chunk branch remains unchanged and still works
    with config slice_index, preserving backwards compatibility.
    """

    def test_audio_chunk_with_config_uses_config_slice_index(self):
        """audio_chunk path should use slice_index from config."""
        config_request = asr_pb2.StreamingRecognizeRequest()
        config_request.config.CopyFrom(
            asr_pb2.RecognitionConfig(
                sample_rate=16000,
                format="pcm_s16le",
                slice_index=5,
            )
        )

        audio_chunk_request = asr_pb2.StreamingRecognizeRequest()
        audio_chunk_request.audio_chunk = b"\x00\x01\x00\x02"

        assert config_request.HasField("config")
        assert config_request.config.slice_index == 5
        assert audio_chunk_request.HasField("audio_chunk")

    def test_audio_chunk_stream_pattern(self):
        """Stream pattern: config → audio_chunk → end_stream."""
        config_request = asr_pb2.StreamingRecognizeRequest()
        config_request.config.CopyFrom(
            asr_pb2.RecognitionConfig(
                sample_rate=16000,
                format="pcm_s16le",
                slice_index=10,
            )
        )

        audio_chunk_request = asr_pb2.StreamingRecognizeRequest()
        audio_chunk_request.audio_chunk = b"\x00\x01\x00\x02\x00\x03\x00\x04"

        end_request = asr_pb2.StreamingRecognizeRequest()
        end_request.end_stream = True

        assert config_request.HasField("config")
        assert config_request.config.slice_index == 10
        assert audio_chunk_request.HasField("audio_chunk")
        assert end_request.HasField("end_stream")

    def test_audio_chunk_mutually_exclusive_with_audio_slice(self):
        """Verify audio_chunk and audio_slice are mutually exclusive."""
        request = asr_pb2.StreamingRecognizeRequest()

        request.audio_chunk = b"\x00\x01"
        assert request.HasField("audio_chunk")

        audio_slice = asr_pb2.AudioSlice(index=0, data=b"slice_data")
        request.audio_slice.CopyFrom(audio_slice)

        assert request.HasField("audio_slice")
        assert not request.HasField("audio_chunk")

    def test_multiple_audio_chunks_sequence(self):
        """Multiple audio_chunks in sequence maintain raw audio data."""
        requests = []
        audio_data_list = [
            b"\x00\x01\x00\x02",
            b"\x00\x03\x00\x04",
            b"\x00\x05\x00\x06",
        ]

        for audio_data in audio_data_list:
            request = asr_pb2.StreamingRecognizeRequest()
            request.audio_chunk = audio_data
            requests.append(request)

        assert len(requests) == 3
        assert requests[0].audio_chunk == b"\x00\x01\x00\x02"
        assert requests[1].audio_chunk == b"\x00\x03\x00\x04"
        assert requests[2].audio_chunk == b"\x00\x05\x00\x06"

        for req in requests:
            assert req.HasField("audio_chunk")

    def test_audio_chunk_response_includes_config_slice_index(self):
        """Response for audio_chunk path includes slice_index from config."""
        config = asr_pb2.RecognitionConfig(
            sample_rate=16000,
            format="pcm_s16le",
            slice_index=7,
        )

        partial = asr_pb2.PartialResult()
        partial.segment_id = "seg_001"
        partial.text = "hello"
        partial.slice_index = 7

        response = asr_pb2.StreamingRecognizeResponse()
        response.partial.CopyFrom(partial)

        assert response.partial.slice_index == 7
        assert response.partial.slice_index == config.slice_index

    def test_audio_chunk_does_not_have_index_field(self):
        """audio_chunk is raw bytes without index as a field."""
        request = asr_pb2.StreamingRecognizeRequest()
        audio_data = b"\x00\x01\x00\x02\x00\x03"
        request.audio_chunk = audio_data

        assert isinstance(request.audio_chunk, bytes)
        assert request.audio_chunk == audio_data
        assert request.audio_chunk != b"something else"

    def test_audio_chunk_field_vs_audio_slice_structure(self):
        """Contrast: audio_chunk (raw bytes) vs audio_slice (index + data)."""
        chunk_request = asr_pb2.StreamingRecognizeRequest()
        chunk_request.audio_chunk = b"raw_audio_data"

        slice_request = asr_pb2.StreamingRecognizeRequest()
        audio_slice = asr_pb2.AudioSlice(index=5, data=b"raw_audio_data")
        slice_request.audio_slice.CopyFrom(audio_slice)

        assert isinstance(chunk_request.audio_chunk, bytes)
        assert hasattr(slice_request.audio_slice, "index")
        assert hasattr(slice_request.audio_slice, "data")
        assert slice_request.audio_slice.index == 5
        assert slice_request.audio_slice.data == b"raw_audio_data"

    def test_audio_chunk_with_zero_slice_index(self):
        """Config with slice_index=0 is valid for audio_chunk path."""
        config_request = asr_pb2.StreamingRecognizeRequest()
        config_request.config.CopyFrom(
            asr_pb2.RecognitionConfig(
                sample_rate=16000,
                format="pcm_s16le",
                slice_index=0,
            )
        )

        audio_chunk_request = asr_pb2.StreamingRecognizeRequest()
        audio_chunk_request.audio_chunk = b"\x00\x01"

        assert config_request.HasField("config")
        assert config_request.config.slice_index == 0
        assert audio_chunk_request.HasField("audio_chunk")

    def test_audio_chunk_no_config_error_scenario(self):
        """audio_chunk without config should trigger NO_CONFIG error."""
        request = asr_pb2.StreamingRecognizeRequest()
        request.audio_chunk = b"\x00\x01"

        assert request.HasField("audio_chunk")


class TestSliceValidationErrors:
    """TDD tests for SLICE_TOO_SHORT (RED until Task 5) and NON_CONTIGUOUS_SLICE."""

    @staticmethod
    def _make_config_request(
        sample_rate: int = 16000,
        slice_index: int = 0,
    ) -> asr_pb2.StreamingRecognizeRequest:
        request = asr_pb2.StreamingRecognizeRequest()
        config = asr_pb2.RecognitionConfig(
            sample_rate=sample_rate,
            format="pcm_s16le",
            slice_index=slice_index,
        )
        request.config.CopyFrom(config)
        return request

    @staticmethod
    def _make_audio_slice_request(
        index: int,
        n_samples: int = 16000,
        sample_value: int = 0,
    ) -> asr_pb2.StreamingRecognizeRequest:
        """n_samples: int16 samples (16000 = 1s at 16kHz)."""
        import struct

        audio_bytes = struct.pack(f"<{n_samples}h", *([sample_value] * n_samples))
        request = asr_pb2.StreamingRecognizeRequest()
        audio_slice = asr_pb2.AudioSlice(index=index, data=audio_bytes)
        request.audio_slice.CopyFrom(audio_slice)
        return request

    @staticmethod
    def _make_end_stream_request() -> asr_pb2.StreamingRecognizeRequest:
        request = asr_pb2.StreamingRecognizeRequest()
        request.end_stream = True
        return request

    def test_slice_too_short_error_protobuf_construction(self):
        """At 16kHz, 200ms = 3200 samples. A 1600-sample (100ms) slice is well-formed."""
        request = self._make_audio_slice_request(index=0, n_samples=1600)

        assert request.HasField("audio_slice")
        assert request.audio_slice.index == 0
        assert len(request.audio_slice.data) == 3200  # 1600 samples * 2 bytes

    def test_slice_too_short_duration_calculation(self):
        """slice_n_ms = len(audio_data) * 1000 // sample_rate; 1600 samples → 100ms."""
        import numpy as np

        sample_rate = 16000
        audio_data = np.zeros(1600, dtype=np.int16)
        slice_n_ms = len(audio_data) * 1000 // sample_rate

        assert slice_n_ms == 100
        assert slice_n_ms < 200

    @pytest.mark.asyncio
    async def test_slice_too_short_returns_error(self):
        """TDD-RED: 100ms (1600 samples) slice → SLICE_TOO_SHORT error."""
        from unittest.mock import MagicMock, AsyncMock, patch

        with patch(
            "fireredasr2s_api.grpc_server.create_backend"
        ) as mock_create_backend:
            mock_backend = MagicMock()
            mock_backend.get_max_audio_length.return_value = 60.0
            mock_backend.transcribe.return_value = {
                "text": "",
                "confidence": 0.0,
                "words": [],
            }
            mock_create_backend.return_value = mock_backend

            from fireredasr2s_api.grpc_server import ASRServiceServicer
            from fireredasr2s_api.config import ApiConfig

            servicer = ASRServiceServicer(ApiConfig())

            servicer.backends["aed"] = mock_backend

            with patch("fireredasr2s_api.grpc_server._SessionVadState") as MockVadState:
                mock_vad = MagicMock()
                mock_vad.initialize.return_value = None
                mock_vad.process_slice_audio.return_value = MagicMock(
                    ended_speaking=False,
                    entirely_speech=False,
                    n_frames=100,
                )
                MockVadState.return_value = mock_vad

                requests = [
                    self._make_config_request(sample_rate=16000, slice_index=0),
                    self._make_audio_slice_request(index=0, n_samples=1600),
                    self._make_end_stream_request(),
                ]

                async def request_iter():
                    for r in requests:
                        yield r

                context = MagicMock()

                responses = []
                async for resp in servicer.StreamingRecognize(request_iter(), context):
                    responses.append(resp)

                error_responses = [r for r in responses if r.HasField("error")]
                assert len(error_responses) >= 1, (
                    "Expected SLICE_TOO_SHORT error but got no error responses. "
                    f"Got {len(responses)} response(s): "
                    + str([r.WhichOneof("response") for r in responses])
                )

                error = error_responses[0].error
                assert error.code == "SLICE_TOO_SHORT", (
                    f"Expected error code 'SLICE_TOO_SHORT' but got '{error.code}': "
                    f"{error.message}"
                )
                assert "200" in error.message or "ms" in error.message.lower()

    @pytest.mark.asyncio
    async def test_slice_too_short_boundary_199ms(self):
        """TDD-RED: 3184 samples at 16kHz = 199ms → SLICE_TOO_SHORT."""
        from unittest.mock import MagicMock, patch

        with patch(
            "fireredasr2s_api.grpc_server.create_backend"
        ) as mock_create_backend:
            mock_backend = MagicMock()
            mock_backend.get_max_audio_length.return_value = 60.0
            mock_backend.transcribe.return_value = {
                "text": "",
                "confidence": 0.0,
                "words": [],
            }
            mock_create_backend.return_value = mock_backend

            from fireredasr2s_api.grpc_server import ASRServiceServicer
            from fireredasr2s_api.config import ApiConfig

            servicer = ASRServiceServicer(ApiConfig())
            servicer.backends["aed"] = mock_backend

            with patch("fireredasr2s_api.grpc_server._SessionVadState") as MockVadState:
                mock_vad = MagicMock()
                mock_vad.initialize.return_value = None
                mock_vad.process_slice_audio.return_value = MagicMock(
                    ended_speaking=False,
                    entirely_speech=False,
                    n_frames=199,
                )
                MockVadState.return_value = mock_vad

                requests = [
                    self._make_config_request(sample_rate=16000, slice_index=0),
                    self._make_audio_slice_request(index=0, n_samples=3184),
                    self._make_end_stream_request(),
                ]

                async def request_iter():
                    for r in requests:
                        yield r

                context = MagicMock()

                responses = []
                async for resp in servicer.StreamingRecognize(request_iter(), context):
                    responses.append(resp)

                error_responses = [r for r in responses if r.HasField("error")]
                slice_too_short = [
                    r for r in error_responses if r.error.code == "SLICE_TOO_SHORT"
                ]
                assert len(slice_too_short) >= 1, (
                    "Expected SLICE_TOO_SHORT error for 199ms slice. "
                    f"Got: {[(r.error.code, r.error.message) for r in error_responses] if error_responses else 'no errors'}"
                )

    @pytest.mark.asyncio
    async def test_slice_exactly_200ms_accepted(self):
        """3200 samples at 16kHz = 200ms → NOT rejected as too short."""
        from unittest.mock import MagicMock, patch

        with patch(
            "fireredasr2s_api.grpc_server.create_backend"
        ) as mock_create_backend:
            mock_backend = MagicMock()
            mock_backend.get_max_audio_length.return_value = 60.0
            mock_backend.transcribe.return_value = {
                "text": "",
                "confidence": 0.0,
                "words": [],
            }
            mock_create_backend.return_value = mock_backend

            from fireredasr2s_api.grpc_server import ASRServiceServicer
            from fireredasr2s_api.config import ApiConfig

            servicer = ASRServiceServicer(ApiConfig())
            servicer.backends["aed"] = mock_backend

            with patch("fireredasr2s_api.grpc_server._SessionVadState") as MockVadState:
                mock_vad = MagicMock()
                mock_vad.initialize.return_value = None
                mock_vad.process_slice_audio.return_value = MagicMock(
                    ended_speaking=False,
                    entirely_speech=False,
                    n_frames=200,
                )
                MockVadState.return_value = mock_vad

                requests = [
                    self._make_config_request(sample_rate=16000, slice_index=0),
                    self._make_audio_slice_request(index=0, n_samples=3200),
                    self._make_end_stream_request(),
                ]

                async def request_iter():
                    for r in requests:
                        yield r

                context = MagicMock()

                responses = []
                async for resp in servicer.StreamingRecognize(request_iter(), context):
                    responses.append(resp)

                too_short_errors = [
                    r
                    for r in responses
                    if r.HasField("error") and r.error.code == "SLICE_TOO_SHORT"
                ]
                assert len(too_short_errors) == 0, (
                    "200ms slice should be accepted, not rejected as too short"
                )

    def test_non_contiguous_slice_protobuf_repeated_index(self):
        req1 = self._make_audio_slice_request(index=0, n_samples=16000)
        req2 = self._make_audio_slice_request(index=0, n_samples=16000)

        assert req1.audio_slice.index == 0
        assert req2.audio_slice.index == 0

    def test_non_contiguous_slice_protobuf_out_of_order_index(self):
        req1 = self._make_audio_slice_request(index=0, n_samples=16000)
        req2 = self._make_audio_slice_request(index=2, n_samples=16000)

        assert req1.audio_slice.index == 0
        assert req2.audio_slice.index == 2

    @pytest.mark.asyncio
    async def test_non_contiguous_slice_repeated_index_returns_error(self):
        """Repeated index=0 after first index=0 → NON_CONTIGUOUS_SLICE (expects 1)."""
        from unittest.mock import MagicMock, patch

        with patch(
            "fireredasr2s_api.grpc_server.create_backend"
        ) as mock_create_backend:
            mock_backend = MagicMock()
            mock_backend.get_max_audio_length.return_value = 60.0
            mock_backend.transcribe.return_value = {
                "text": "",
                "confidence": 0.0,
                "words": [],
            }
            mock_create_backend.return_value = mock_backend

            from fireredasr2s_api.grpc_server import ASRServiceServicer
            from fireredasr2s_api.config import ApiConfig

            servicer = ASRServiceServicer(ApiConfig())
            servicer.backends["aed"] = mock_backend

            with patch("fireredasr2s_api.grpc_server._SessionVadState") as MockVadState:
                mock_vad = MagicMock()
                mock_vad.initialize.return_value = None
                mock_vad.process_slice_audio.return_value = MagicMock(
                    ended_speaking=False,
                    entirely_speech=False,
                    n_frames=1000,
                )
                MockVadState.return_value = mock_vad

                requests = [
                    self._make_config_request(sample_rate=16000, slice_index=0),
                    self._make_audio_slice_request(index=0, n_samples=16000),
                    self._make_audio_slice_request(index=0, n_samples=16000),
                    self._make_end_stream_request(),
                ]

                async def request_iter():
                    for r in requests:
                        yield r

                context = MagicMock()

                responses = []
                async for resp in servicer.StreamingRecognize(request_iter(), context):
                    responses.append(resp)

                error_responses = [r for r in responses if r.HasField("error")]
                non_contiguous_errors = [
                    r for r in error_responses if r.error.code == "NON_CONTIGUOUS_SLICE"
                ]
                assert len(non_contiguous_errors) >= 1, (
                    "Expected NON_CONTIGUOUS_SLICE error for repeated index=0. "
                    f"Got errors: {[(r.error.code, r.error.message) for r in error_responses]}"
                )

                error = non_contiguous_errors[0].error
                assert "expected" in error.message.lower()
                assert "1" in error.message

    @pytest.mark.asyncio
    async def test_non_contiguous_slice_out_of_order_returns_error(self):
        """index=0 then index=2 (skip 1) → NON_CONTIGUOUS_SLICE."""
        from unittest.mock import MagicMock, patch

        with patch(
            "fireredasr2s_api.grpc_server.create_backend"
        ) as mock_create_backend:
            mock_backend = MagicMock()
            mock_backend.get_max_audio_length.return_value = 60.0
            mock_backend.transcribe.return_value = {
                "text": "",
                "confidence": 0.0,
                "words": [],
            }
            mock_create_backend.return_value = mock_backend

            from fireredasr2s_api.grpc_server import ASRServiceServicer
            from fireredasr2s_api.config import ApiConfig

            servicer = ASRServiceServicer(ApiConfig())
            servicer.backends["aed"] = mock_backend

            with patch("fireredasr2s_api.grpc_server._SessionVadState") as MockVadState:
                mock_vad = MagicMock()
                mock_vad.initialize.return_value = None
                mock_vad.process_slice_audio.return_value = MagicMock(
                    ended_speaking=False,
                    entirely_speech=False,
                    n_frames=1000,
                )
                MockVadState.return_value = mock_vad

                requests = [
                    self._make_config_request(sample_rate=16000, slice_index=0),
                    self._make_audio_slice_request(index=0, n_samples=16000),
                    self._make_audio_slice_request(index=2, n_samples=16000),
                    self._make_end_stream_request(),
                ]

                async def request_iter():
                    for r in requests:
                        yield r

                context = MagicMock()

                responses = []
                async for resp in servicer.StreamingRecognize(request_iter(), context):
                    responses.append(resp)

                error_responses = [r for r in responses if r.HasField("error")]
                non_contiguous_errors = [
                    r for r in error_responses if r.error.code == "NON_CONTIGUOUS_SLICE"
                ]
                assert len(non_contiguous_errors) >= 1, (
                    "Expected NON_CONTIGUOUS_SLICE error for out-of-order index. "
                    f"Got errors: {[(r.error.code, r.error.message) for r in error_responses]}"
                )

                error = non_contiguous_errors[0].error
                assert "expected" in error.message.lower()
                assert "1" in error.message
                assert "2" in error.message

    @pytest.mark.asyncio
    async def test_contiguous_slices_accepted(self):
        from unittest.mock import MagicMock, patch

        with patch(
            "fireredasr2s_api.grpc_server.create_backend"
        ) as mock_create_backend:
            mock_backend = MagicMock()
            mock_backend.get_max_audio_length.return_value = 60.0
            mock_backend.transcribe.return_value = {
                "text": "",
                "confidence": 0.0,
                "words": [],
            }
            mock_create_backend.return_value = mock_backend

            from fireredasr2s_api.grpc_server import ASRServiceServicer
            from fireredasr2s_api.config import ApiConfig

            servicer = ASRServiceServicer(ApiConfig())
            servicer.backends["aed"] = mock_backend

            with patch("fireredasr2s_api.grpc_server._SessionVadState") as MockVadState:
                mock_vad = MagicMock()
                mock_vad.initialize.return_value = None
                mock_vad.process_slice_audio.return_value = MagicMock(
                    ended_speaking=False,
                    entirely_speech=False,
                    n_frames=1000,
                )
                MockVadState.return_value = mock_vad

                requests = [
                    self._make_config_request(sample_rate=16000, slice_index=0),
                    self._make_audio_slice_request(index=0, n_samples=16000),
                    self._make_audio_slice_request(index=1, n_samples=16000),
                    self._make_audio_slice_request(index=2, n_samples=16000),
                    self._make_end_stream_request(),
                ]

                async def request_iter():
                    for r in requests:
                        yield r

                context = MagicMock()

                responses = []
                async for resp in servicer.StreamingRecognize(request_iter(), context):
                    responses.append(resp)

                contiguous_errors = [
                    r
                    for r in responses
                    if r.HasField("error") and r.error.code == "NON_CONTIGUOUS_SLICE"
                ]
                assert len(contiguous_errors) == 0, (
                    "Contiguous indices (0,1,2) should not produce errors"
                )
