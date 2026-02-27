# Copyright 2026 FireRedTeam

"""Streaming session management for real-time ASR."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Session state machine states."""

    CONNECTING = auto()
    STREAMING = auto()
    FINALIZING = auto()
    DONE = auto()
    ERROR = auto()


@dataclass
class AudioSegment:
    """Represents an audio segment with metadata."""

    audio: NDArray[np.int16]
    start_time_ms: int
    end_time_ms: int
    is_speech: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.audio, np.ndarray):
            raise ValueError("Audio must be numpy array")
        if self.audio.dtype != np.int16:
            raise ValueError("Audio must be int16")


@dataclass
class PartialResult:
    """Partial transcription result."""

    segment_id: str
    revision: int
    text: str
    start_ms: int
    end_ms: int
    confidence: float
    is_final: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class FinalResult:
    """Final transcription result with all post-processing."""

    uttid: str
    text: str
    sentences: List[Dict[str, Any]]
    vad_segments_ms: List[Tuple[int, int]]
    dur_s: float
    words: List[Dict[str, Any]]


class StreamingSession:
    """Manages a single real-time streaming ASR session.

    This class handles:
    - Audio chunk buffering and VAD segmentation
    - State machine transitions
    - Partial and final result delivery
    - Session lifecycle management
    """

    def __init__(
        self,
        session_id: str,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,
        silence_timeout_ms: int = 500,
        max_segment_duration_ms: int = 60000,
        enable_partial_results: bool = True,
        partial_result_interval_ms: int = 200,
        session_timeout_seconds: int = 300,
        on_partial: Optional[Callable[[PartialResult], None]] = None,
        on_final: Optional[Callable[[FinalResult], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        enable_lid: bool = True,
        enable_punc: bool = True,
    ):
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.silence_timeout_ms = silence_timeout_ms
        self.max_segment_duration_ms = max_segment_duration_ms
        self.enable_partial_results = enable_partial_results
        self.partial_result_interval_ms = partial_result_interval_ms
        self.session_timeout_seconds = session_timeout_seconds
        self.enable_lid = enable_lid
        self.enable_punc = enable_punc

        # Postprocessing model references (set externally by servicer)
        self.lid_model: Optional[Any] = None
        self.punc_model: Optional[Any] = None

        # Callbacks
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_error = on_error

        # State
        self.state = SessionState.CONNECTING
        self.created_at = time.time()
        self.last_activity = time.time()

        # Audio buffer
        self._audio_buffer: List[NDArray[np.int16]] = []
        self._buffer_duration_ms = 0
        self._full_audio: List[NDArray[np.int16]] = []

        # Segment tracking
        self._current_segment_start_ms = 0
        self._current_segment_duration_ms = 0
        self._segment_counter = 0
        self._partial_counter = 0

        # VAD state
        self._is_speaking = False
        self._last_speech_end_ms = 0

        # Slice tracking
        self._slice_index: int = -1  # -1 means not set
        self._slice_m_ms: int = 0  # computed m for current slice (ms)
        self._global_frame_count: int = 0  # total frames across all slices

        # LLM parameters and ASR type (set by handler from config message)
        self.llm_params: Dict[str, Any] = {}
        self.asr_type: str = "aed"

        # Results
        self._partial_results: List[PartialResult] = []
        self._final_result: Optional[FinalResult] = None

        logger.info(f"Session {session_id} created")

    @property
    def is_active(self) -> bool:
        """Check if session is still active."""
        if self.state in (SessionState.DONE, SessionState.ERROR):
            return False
        if time.time() - self.last_activity > self.session_timeout_seconds:
            return False
        return True

    @property
    def duration_ms(self) -> int:
        """Get session duration in milliseconds."""
        return int((time.time() - self.created_at) * 1000)

    def transition_to(self, new_state: SessionState) -> None:
        """Transition to a new state."""
        logger.info(
            f"Session {self.session_id} transitioning from {self.state.name} to {new_state.name}"
        )
        self.state = new_state
        self.last_activity = time.time()

    def add_audio(self, audio_chunk: NDArray[np.int16]) -> None:
        """Add audio chunk to buffer.

        Args:
            audio_chunk: Audio data as int16 numpy array

        Raises:
            ValueError: If audio format is invalid
            RuntimeError: If session is not in valid state
        """
        if self.state not in (SessionState.CONNECTING, SessionState.STREAMING):
            raise RuntimeError(f"Cannot add audio in state {self.state.name}")

        # Validate audio
        if not isinstance(audio_chunk, np.ndarray):
            raise ValueError("Audio must be numpy array")
        if audio_chunk.dtype != np.int16:
            raise ValueError(f"Audio must be int16, got {audio_chunk.dtype}")

        # Add to buffer
        self._audio_buffer.append(audio_chunk)
        self._full_audio.append(audio_chunk)
        chunk_duration_ms = len(audio_chunk) * 1000 // self.sample_rate
        self._buffer_duration_ms += chunk_duration_ms
        self.last_activity = time.time()

        if self.state == SessionState.CONNECTING:
            self.transition_to(SessionState.STREAMING)

        logger.debug(
            f"Added {chunk_duration_ms}ms audio, buffer now {self._buffer_duration_ms}ms"
        )

    def end_stream(self) -> None:
        """Signal end of audio stream."""
        if self.state not in (SessionState.CONNECTING, SessionState.STREAMING):
            return

        logger.info(f"Session {self.session_id} received end signal")
        self.transition_to(SessionState.FINALIZING)

        # Process any remaining audio
        if self._buffer_duration_ms > 0:
            self._process_buffer(final=True)

        # Generate final result
        self._generate_final_result()
        self.transition_to(SessionState.DONE)

    def _process_buffer(self, final: bool = False) -> None:
        """Process buffered audio."""
        if not self._audio_buffer:
            return

        # Concatenate audio
        audio = np.concatenate(self._audio_buffer)
        duration_ms = len(audio) * 1000 // self.sample_rate

        # Clear buffer
        self._audio_buffer = []
        self._buffer_duration_ms = 0

        # Process segment
        self._process_segment(audio, duration_ms, final)

    def _process_segment(
        self, audio: NDArray[np.int16], duration_ms: int, final: bool
    ) -> None:
        """Process a single audio segment."""
        segment_id = f"{self.session_id}_seg{self._segment_counter}"
        self._segment_counter += 1

        # Update segment tracking
        self._current_segment_duration_ms += duration_ms

        logger.info(f"Processing segment {segment_id}: {duration_ms}ms")

        # TODO: Integrate with actual ASR backend
        # For now, create placeholder partial result
        if self.enable_partial_results and self.on_partial:
            partial = PartialResult(
                segment_id=segment_id,
                revision=self._partial_counter,
                text="",  # Would come from ASR
                start_ms=self._current_segment_start_ms,
                end_ms=self._current_segment_start_ms + duration_ms,
                confidence=0.0,
                is_final=final,
            )
            self._partial_results.append(partial)
            self.on_partial(partial)
            self._partial_counter += 1

        # Check if segment should be finalized
        if final or self._current_segment_duration_ms >= self.max_segment_duration_ms:
            self._finalize_segment()

    def _finalize_segment(self) -> None:
        """Finalize current segment and prepare for next."""
        self._current_segment_start_ms += self._current_segment_duration_ms
        self._current_segment_duration_ms = 0

    def _generate_final_result(self) -> None:
        """Generate final transcription result."""
        # TODO: Integrate with actual LID and Punc modules

        sentences = []
        for partial in self._partial_results:
            if partial.is_final or True:  # Include all partials
                sentences.append(
                    {
                        "start_ms": partial.start_ms,
                        "end_ms": partial.end_ms,
                        "text": partial.text,
                        "asr_confidence": partial.confidence,
                        "lang": None,  # Would come from LID
                        "lang_confidence": 0.0,
                    }
                )

        text = " ".join(str(s["text"]) for s in sentences)

        self._final_result = FinalResult(
            uttid=self.session_id,
            text=text,
            sentences=sentences,
            vad_segments_ms=[],  # Would track VAD segments
            dur_s=self.duration_ms / 1000.0,
            words=[],  # Would include word-level timestamps if enabled
        )

        if self.on_final:
            self.on_final(self._final_result)

    def error(self, message: str) -> None:
        """Handle error in session."""
        logger.error(f"Session {self.session_id} error: {message}")
        self.transition_to(SessionState.ERROR)
        if self.on_error:
            self.on_error(message)

    @property
    def slice_index(self) -> int:
        return self._slice_index

    @slice_index.setter
    def slice_index(self, value: int) -> None:
        self._slice_index = value

    @property
    def slice_m_ms(self) -> int:
        return self._slice_m_ms

    @slice_m_ms.setter
    def slice_m_ms(self, value: int) -> None:
        self._slice_m_ms = value

    @property
    def global_frame_count(self) -> int:
        return self._global_frame_count

    def advance_global_frames(self, num_frames: int) -> None:
        self._global_frame_count += num_frames

    def has_complete_segment(self) -> bool:
        """Check if buffer has enough audio for a complete segment.

        Returns:
            True if buffer duration >= chunk_duration_ms, False otherwise.
        """
        return self._buffer_duration_ms >= self.chunk_duration_ms

    def get_segment_audio(self) -> NDArray[np.int16]:
        """Get audio data from current buffer.

        Returns:
            Concatenated audio as int16 numpy array. Empty array if buffer is empty.
        """
        if not self._audio_buffer:
            return np.array([], dtype=np.int16)
        return np.concatenate(self._audio_buffer)

    @property
    def current_segment_id(self) -> str:
        """Get the current segment ID.

        Returns:
            Segment ID in format: {session_id}_seg{segment_counter}
        """
        return f"{self.session_id}_seg{self._segment_counter}"

    def get_next_revision(self) -> int:
        """Get next revision number and increment counter.

        Returns:
            Current revision number before incrementing.
        """
        revision = self._partial_counter
        self._partial_counter += 1
        return revision

    def mark_segment_processed(self) -> None:
        """Mark current segment as processed.

        Clears the audio buffer, resets buffer duration, and increments segment counter.
        """
        self._audio_buffer = []
        self._buffer_duration_ms = 0
        self._segment_counter += 1

    def has_pending_audio(self) -> bool:
        """Check if there is unprocessed audio in buffer.

        Returns:
            True if audio buffer is not empty, False otherwise.
        """
        return len(self._audio_buffer) > 0

    def get_remaining_audio(self) -> NDArray[np.int16]:
        """Get all remaining audio from buffer without clearing it.

        Returns:
            Concatenated audio as int16 numpy array. Empty array if buffer is empty.
        """
        if not self._audio_buffer:
            return np.array([], dtype=np.int16)
        return np.concatenate(self._audio_buffer)

    def get_full_audio(self) -> NDArray[np.int16]:
        if not self._full_audio:
            return np.array([], dtype=np.int16)
        return np.concatenate(self._full_audio)
