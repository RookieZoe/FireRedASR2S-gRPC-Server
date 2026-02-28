# Copyright 2026 FireRedTeam

"""Tests for streaming session management."""

import time

import numpy as np
import pytest

from asr2s_grpc.config import SessionConfig
from asr2s_grpc.session import (
    FinalResult,
    PartialResult,
    SessionState,
    StreamingSession,
)


class TestSessionConfig:
    """Test session configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SessionConfig()
        assert config.sample_rate == 16000
        assert config.chunk_duration_ms == 100
        assert config.silence_timeout_ms == 500
        assert config.max_segment_duration_ms == 60000

    def test_invalid_sample_rate(self):
        """Test that non-16kHz sample rate raises error."""
        with pytest.raises(ValueError, match="Only 16kHz sample rate is supported"):
            SessionConfig(sample_rate=8000)

    def test_invalid_chunk_duration(self):
        """Test that invalid chunk duration raises error."""
        with pytest.raises(ValueError, match="Chunk duration must be between"):
            SessionConfig(chunk_duration_ms=10)
        with pytest.raises(ValueError, match="Chunk duration must be between"):
            SessionConfig(chunk_duration_ms=600)


class TestStreamingSession:
    """Test streaming session functionality."""

    @pytest.fixture
    def session(self):
        """Create a test session."""
        return StreamingSession(
            session_id="test-session-001",
            sample_rate=16000,
            chunk_duration_ms=100,
            silence_timeout_ms=500,
            max_segment_duration_ms=60000,
        )

    @pytest.fixture
    def audio_chunk(self):
        """Create a test audio chunk (100ms at 16kHz)."""
        samples = int(16000 * 0.1)  # 100ms
        return np.zeros(samples, dtype=np.int16)

    def test_session_creation(self, session):
        """Test session initialization."""
        assert session.session_id == "test-session-001"
        assert session.state == SessionState.CONNECTING
        assert session.sample_rate == 16000
        assert session.is_active

    def test_add_audio(self, session, audio_chunk):
        """Test adding audio to session."""
        session.add_audio(audio_chunk)

        assert session.state == SessionState.STREAMING
        assert len(session._audio_buffer) == 1
        assert session._buffer_duration_ms == 100

    def test_add_invalid_audio_format(self, session):
        """Test that invalid audio format raises error."""
        # Wrong dtype
        invalid_audio = np.zeros(1600, dtype=np.float32)
        with pytest.raises(ValueError, match="Audio must be int16"):
            session.add_audio(invalid_audio)

    def test_add_audio_wrong_state(self, session, audio_chunk):
        """Test that adding audio in wrong state raises error."""
        session.transition_to(SessionState.DONE)
        with pytest.raises(RuntimeError, match="Cannot add audio in state"):
            session.add_audio(audio_chunk)

    def test_end_stream(self, session, audio_chunk):
        """Test ending stream."""
        session.add_audio(audio_chunk)
        session.end_stream()

        assert session.state == SessionState.DONE

    def test_session_timeout(self, session):
        """Test that session becomes inactive after timeout."""
        session.session_timeout_seconds = 0.001  # 1ms for testing
        time.sleep(0.01)  # Wait for timeout
        assert not session.is_active

    def test_state_transitions(self, session):
        """Test state transitions."""
        assert session.state == SessionState.CONNECTING

        session.transition_to(SessionState.STREAMING)
        assert session.state == SessionState.STREAMING

        session.transition_to(SessionState.FINALIZING)
        assert session.state == SessionState.FINALIZING

        session.transition_to(SessionState.DONE)
        assert session.state == SessionState.DONE

    def test_duration_tracking(self, session):
        """Test session duration tracking."""
        start_time = session.created_at
        time.sleep(0.01)
        duration = session.duration_ms

        assert duration >= 10  # At least 10ms

    def test_error_handling(self, session):
        """Test error handling."""
        error_message = "Test error"

        # Mock callback
        errors = []
        session.on_error = lambda msg: errors.append(msg)

        session.error(error_message)

        assert session.state == SessionState.ERROR
        assert len(errors) == 1
        assert errors[0] == error_message

    def test_add_audio_in_finalizing_state(self, session, audio_chunk):
        """Test that add_audio raises RuntimeError when session is FINALIZING."""
        # Transition to FINALIZING state
        session.transition_to(SessionState.FINALIZING)

        # Attempting to add audio should raise RuntimeError
        with pytest.raises(RuntimeError, match="Cannot add audio in state FINALIZING"):
            session.add_audio(audio_chunk)

    def test_add_audio_in_error_state(self, session, audio_chunk):
        """Test that add_audio raises RuntimeError when session is in ERROR state."""
        # Transition to ERROR state
        session.transition_to(SessionState.ERROR)

        # Attempting to add audio should raise RuntimeError
        with pytest.raises(RuntimeError, match="Cannot add audio in state ERROR"):
            session.add_audio(audio_chunk)

    def test_end_stream_idempotent(self, session, audio_chunk):
        """Test that end_stream can be called multiple times safely (idempotent)."""
        # Add audio and call end_stream once
        session.add_audio(audio_chunk)
        session.end_stream()

        assert session.state == SessionState.DONE

        # Calling end_stream again should not raise an exception
        session.end_stream()

        # State should still be DONE
        assert session.state == SessionState.DONE

    def test_end_stream_idempotent_from_connecting(self, session):
        """Test that end_stream can be called multiple times from CONNECTING state."""
        # Session starts in CONNECTING state
        assert session.state == SessionState.CONNECTING

        # First call to end_stream
        session.end_stream()
        assert session.state == SessionState.DONE

        # Second call to end_stream should be safe
        session.end_stream()
        assert session.state == SessionState.DONE
