# Copyright 2026 FireRedTeam
"""Shared VAD utilities for gRPC server."""

import logging
import os
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# VAD frame = 10ms
_VAD_FRAMES_PER_MS = 0.1
_MS_PER_VAD_FRAME = 10


class _SessionVadState:
    """Per-session VAD state maintained across slices."""

    def __init__(self) -> None:
        self.stream_vad: Optional[Any] = None
        self._available = False
        self._frame_offset = 0  # Track first frame idx of current slice
        try:
            from fireredasr2s.fireredvad import (
                FireRedStreamVad,
                FireRedStreamVadConfig,
            )
            from fireredasr2s.fireredvad.core.constants import FRAME_LENGTH_SAMPLE

            self._frame_length_sample = FRAME_LENGTH_SAMPLE
            self._available = True
        except ImportError:
            self._frame_length_sample = 400
            self._available = False

    def initialize(self, vad_model_dir: Optional[str] = None) -> None:
        if not self._available:
            raise RuntimeError("FireRedVAD not available")
        try:
            from fireredasr2s.fireredvad import (
                FireRedStreamVad,
                FireRedStreamVadConfig,
            )

            config = FireRedStreamVadConfig(use_gpu=False)
            if vad_model_dir:
                # Validate model files exist before loading
                model_pth = os.path.join(vad_model_dir, "model.pth.tar")
                cmvn_ark = os.path.join(vad_model_dir, "cmvn.ark")
                if not os.path.isfile(model_pth):
                    raise RuntimeError(f"VAD model file not found: {model_pth}")
                if not os.path.isfile(cmvn_ark):
                    raise RuntimeError(f"VAD model file not found: {cmvn_ark}")
                self.stream_vad = FireRedStreamVad.from_pretrained(
                    vad_model_dir, config
                )
                logger.info("Loaded VAD model from %s", vad_model_dir)
            else:
                raise RuntimeError("VAD model directory not provided")
        except Exception as e:
            logger.error("Failed to initialize StreamVAD: %s", e)
            raise RuntimeError(f"FireRedVAD initialization failed: {e}")

    def process_slice_audio(
        self,
        audio_data: NDArray[np.int16],
        sample_rate: int,
    ) -> "SliceVadResult":
        """Run VAD on a slice and return VAD analysis."""
        n_samples = len(audio_data)
        n_frames = n_samples * 1000 // (sample_rate * _MS_PER_VAD_FRAME)

        if not self._available or self.stream_vad is None:
            raise RuntimeError("VAD not initialized")

        frame_results = self.stream_vad.detect_chunk(audio_data)

        # Track first frame index for offset calculation
        first_frame_idx = frame_results[0].frame_idx if frame_results else 0
        self._frame_offset = first_frame_idx

        ended_speaking = False
        speech_start_frame: Optional[int] = None
        entirely_speech = True

        for result in frame_results:
            if not result.is_speech:
                entirely_speech = False
            if result.is_speech_start and result.speech_start_frame > 0:
                speech_start_frame = result.speech_start_frame
            ended_speaking = result.is_speech

        return SliceVadResult(
            n_frames=n_frames,
            ended_speaking=ended_speaking,
            speech_start_frame=speech_start_frame,
            entirely_speech=entirely_speech and len(frame_results) > 0,
            frame_offset=self._frame_offset,
        )


class SliceVadResult:
    __slots__ = (
        "n_frames",
        "ended_speaking",
        "speech_start_frame",
        "entirely_speech",
        "frame_offset",
    )

    def __init__(
        self,
        n_frames: int,
        ended_speaking: bool,
        speech_start_frame: Optional[int],
        entirely_speech: bool,
        frame_offset: int = 0,
    ) -> None:
        self.n_frames = n_frames
        self.ended_speaking = ended_speaking
        self.speech_start_frame = speech_start_frame
        self.entirely_speech = entirely_speech
        self.frame_offset = frame_offset  # First frame idx of this slice


def compute_slice_m_ms(vad_result: SliceVadResult) -> int:
    """Compute m (in ms) for a slice based on VAD analysis.

    FIXED: Now computes slice-relative m using frame_offset.
    """
    n_ms = vad_result.n_frames * _MS_PER_VAD_FRAME

    if vad_result.entirely_speech:
        return n_ms

    if not vad_result.ended_speaking:
        return n_ms

    # FIXED: Subtract frame_offset to get slice-relative position
    if vad_result.speech_start_frame is not None and vad_result.speech_start_frame > 0:
        relative_frame = vad_result.speech_start_frame - vad_result.frame_offset
        return max(0, relative_frame * _MS_PER_VAD_FRAME)

    return n_ms
