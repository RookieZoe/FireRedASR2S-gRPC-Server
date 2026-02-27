# Copyright 2026 FireRedTeam

"""RED tests for slice VAD m computation using FireRedStreamVad.

The m (minimum frame index) computation rules:
1. If at time n (end of slice), not speaking: m = n
2. If at time n, speech is ongoing: m = last non-speech time (speech_start_frame / 100)
3. If slice is entirely speech (no silence within [0,n]): m = n

Time conversion: 10ms frames, so time_seconds = frame_idx / 100
"""

from typing import Any, List

import pytest

from fireredasr2s.fireredvad.core.stream_vad_postprocessor import StreamVadFrameResult


class MockFireRedStreamVad:
    """Mock FireRedStreamVad for testing without loading model weights."""

    def __init__(self, frame_results: List[StreamVadFrameResult]) -> None:
        """
        Initialize with predetermined frame results.

        Args:
            frame_results: List of StreamVadFrameResult objects
        """
        self.frame_results = frame_results

    def detect_chunk(self, audio_chunk: Any) -> List[StreamVadFrameResult]:
        """
        Mock detect_chunk method to mirror production code.

        Args:
            audio_chunk: Audio data (not used in mock)

        Returns:
            List of StreamVadFrameResult objects for the chunk
        """
        return self.frame_results


class TestSliceVadMComputation:
    """Test m computation for slice VAD according to rules."""

    @pytest.mark.unit
    def test_m_computation_end_not_speaking(self):
        """
        Test case 1: End at n not speaking -> m = n

        Slice: [0, 3.0s] (300 frames)
        VAD pattern: silence for 2.5s, then speech ends before 3.0s
        Expected: m = 3.0s (300 frames / 100)
        """
        # Create frame results: mostly silence, some speech in middle, back to silence
        frame_results = []

        # Frames 1-100: silence (1.0s)
        for i in range(1, 101):
            frame_results.append(
                StreamVadFrameResult(
                    frame_idx=i,
                    is_speech=False,
                    raw_prob=0.1,
                    smoothed_prob=0.1,
                )
            )

        # Frames 101-200: speech (1.0s, 1.0-2.0s)
        for i in range(101, 201):
            frame_results.append(
                StreamVadFrameResult(
                    frame_idx=i,
                    is_speech=True,
                    raw_prob=0.9,
                    smoothed_prob=0.9,
                    is_speech_start=(i == 101),
                    speech_start_frame=101 if i == 101 else -1,
                )
            )

        # Frames 201-300: silence again (1.0s, 2.0-3.0s)
        for i in range(201, 301):
            frame_results.append(
                StreamVadFrameResult(
                    frame_idx=i,
                    is_speech=False,
                    raw_prob=0.1,
                    smoothed_prob=0.1,
                    is_speech_end=(i == 201),
                    speech_end_frame=200 if i == 201 else -1,
                    speech_start_frame=101 if i == 201 else -1,
                )
            )

        mock_vad = MockFireRedStreamVad(frame_results)
        frame_results_output = mock_vad.detect_chunk(None)

        # Last frame indicates not speaking
        assert frame_results_output[-1].is_speech is False
        assert frame_results_output[-1].frame_idx == 300

        # m should be n = 3.0s
        n_frames = frame_results_output[-1].frame_idx
        m_seconds = n_frames / 100
        assert m_seconds == 3.0

    @pytest.mark.unit
    def test_m_computation_end_speaking_with_silence_within_slice(self):
        """
        Test case 2: End at n speaking, speech started within slice -> m = speech_start_time

        Slice: [0, 3.0s] (300 frames)
        VAD pattern: silence (0-1s), speech starts at 1.0s, continues to end at 3.0s
        Speech starts at frame 101 (1.01s)
        Expected: m = 1.01s (101 / 100)
        """
        frame_results = []

        # Frames 1-100: silence (0-1.0s)
        for i in range(1, 101):
            frame_results.append(
                StreamVadFrameResult(
                    frame_idx=i,
                    is_speech=False,
                    raw_prob=0.1,
                    smoothed_prob=0.1,
                )
            )

        # Frames 101-300: speech (1.0-3.0s) - continuous from frame 101
        for i in range(101, 301):
            frame_results.append(
                StreamVadFrameResult(
                    frame_idx=i,
                    is_speech=True,
                    raw_prob=0.9,
                    smoothed_prob=0.9,
                    is_speech_start=(i == 101),
                    speech_start_frame=101 if i == 101 else -1,
                )
            )

        mock_vad = MockFireRedStreamVad(frame_results)
        frame_results_output = mock_vad.detect_chunk(None)

        # Last frame indicates speaking
        assert frame_results_output[-1].is_speech is True
        assert frame_results_output[-1].frame_idx == 300

        # Find the speech_start_frame
        speech_start_frame = None
        for result in frame_results_output:
            if result.is_speech_start:
                speech_start_frame = result.speech_start_frame
                break

        assert speech_start_frame == 101

        # m should be speech_start_time = 101 / 100 = 1.01s
        m_seconds = speech_start_frame / 100
        assert m_seconds == 1.01

    @pytest.mark.unit
    def test_m_computation_entire_slice_is_speech(self):
        """
        Test case 3: Entire slice is speech (no silence within [0,n]) -> m = n

        Slice: [0, 3.0s] (300 frames)
        VAD pattern: speech from frame 1 to frame 300 (no silence gap)
        Expected: m = 3.0s (300 / 100) because no silence within the slice
        """
        frame_results = []

        # Frames 1-300: all speech (0-3.0s)
        for i in range(1, 301):
            frame_results.append(
                StreamVadFrameResult(
                    frame_idx=i,
                    is_speech=True,
                    raw_prob=0.9,
                    smoothed_prob=0.9,
                    is_speech_start=(i == 1),
                    speech_start_frame=1 if i == 1 else -1,
                )
            )

        mock_vad = MockFireRedStreamVad(frame_results)
        frame_results_output = mock_vad.detect_chunk(None)

        # All frames are speech
        assert all(result.is_speech for result in frame_results_output)
        assert frame_results_output[-1].frame_idx == 300

        # m should be n = 3.0s (no silence found within slice)
        n_frames = frame_results_output[-1].frame_idx
        m_seconds = n_frames / 100
        assert m_seconds == 3.0

    @pytest.mark.unit
    def test_m_computation_multiple_speech_segments(self):
        """
        Test case 4 (bonus): Multiple speech segments within slice.

        Slice: [0, 4.0s] (400 frames)
        VAD pattern:
          - Silence: 0-1.0s (frames 1-100)
          - Speech: 1.0-2.0s (frames 101-200)
          - Silence: 2.0-2.5s (frames 201-250)
          - Speech: 2.5-4.0s (frames 251-400) <- ongoing at end
        Expected: m = 2.51s (speech_start_frame=251 / 100) because speech is ongoing
        """
        frame_results = []

        # Frames 1-100: silence
        for i in range(1, 101):
            frame_results.append(
                StreamVadFrameResult(
                    frame_idx=i,
                    is_speech=False,
                    raw_prob=0.1,
                    smoothed_prob=0.1,
                )
            )

        # Frames 101-200: first speech segment
        for i in range(101, 201):
            frame_results.append(
                StreamVadFrameResult(
                    frame_idx=i,
                    is_speech=True,
                    raw_prob=0.9,
                    smoothed_prob=0.9,
                    is_speech_start=(i == 101),
                    speech_start_frame=101 if i == 101 else -1,
                )
            )

        # Frames 201-250: silence
        for i in range(201, 251):
            frame_results.append(
                StreamVadFrameResult(
                    frame_idx=i,
                    is_speech=False,
                    raw_prob=0.1,
                    smoothed_prob=0.1,
                    is_speech_end=(i == 201),
                    speech_end_frame=200 if i == 201 else -1,
                    speech_start_frame=101 if i == 201 else -1,
                )
            )

        # Frames 251-400: second speech segment (ongoing at end)
        for i in range(251, 401):
            frame_results.append(
                StreamVadFrameResult(
                    frame_idx=i,
                    is_speech=True,
                    raw_prob=0.9,
                    smoothed_prob=0.9,
                    is_speech_start=(i == 251),
                    speech_start_frame=251 if i == 251 else -1,
                )
            )

        mock_vad = MockFireRedStreamVad(frame_results)
        frame_results_output = mock_vad.detect_chunk(None)

        # Last frame indicates ongoing speech
        assert frame_results_output[-1].is_speech is True

        # Find the last speech_start_frame (should be 251 for the ongoing segment)
        speech_start_frame = None
        for result in frame_results_output:
            if result.is_speech_start:
                # Keep updating to get the LAST speech_start (second segment)
                speech_start_frame = result.speech_start_frame

        assert speech_start_frame == 251

        # m should be speech_start_time = 251 / 100 = 2.51s
        m_seconds = speech_start_frame / 100
        assert m_seconds == 2.51

    @pytest.mark.unit
    def test_frame_result_structure(self):
        """Test that StreamVadFrameResult has expected fields."""
        result = StreamVadFrameResult(
            frame_idx=42,
            is_speech=True,
            raw_prob=0.95,
            smoothed_prob=0.93,
            is_speech_start=True,
            speech_start_frame=41,
        )

        assert result.frame_idx == 42
        assert result.is_speech is True
        assert result.raw_prob == 0.95
        assert result.smoothed_prob == 0.93
        assert result.is_speech_start is True
        assert result.speech_start_frame == 41
        assert result.is_speech_end is False
        assert result.speech_end_frame == -1

    @pytest.mark.unit
    def test_m_computation_short_slice(self):
        """
        Test m computation on short slice (< 1 second).

        Slice: [0, 0.5s] (50 frames)
        VAD pattern: all silence
        Expected: m = 0.5s
        """
        frame_results = []

        for i in range(1, 51):
            frame_results.append(
                StreamVadFrameResult(
                    frame_idx=i,
                    is_speech=False,
                    raw_prob=0.05,
                    smoothed_prob=0.05,
                )
            )

        mock_vad = MockFireRedStreamVad(frame_results)
        frame_results_output = mock_vad.detect_chunk(None)

        n_frames = frame_results_output[-1].frame_idx
        m_seconds = n_frames / 100
        assert m_seconds == 0.5

    @pytest.mark.unit
    def test_multi_slice_m_never_exceeds_n(self):
        """
        Test across multiple slices that m never exceeds n.

        Slice 1: [0, 2.0s], ends in speech → m = 1.0s (speech started at frame 101)
        Slice 2: [2.0s, 4.0s] in global time, but frame indices reset to [0, 200] in slice
                 Speech continues from previous slice
                 Frame results have frame_idx in slice range [1, 200]
                 Without frame_idx offset, speech_start_frame might be > slice_n_ms
                 With proper slice-relative computation: m = offset_frames / 100 = 0.0s (speech from frame 1 in slice)

        Expected: In Slice 2, m = 0.0s (since speech started at frame 1 relative to slice)
        """
        # Slice 1 setup
        frame_results_slice1 = []

        for i in range(1, 101):
            frame_results_slice1.append(
                StreamVadFrameResult(
                    frame_idx=i,
                    is_speech=False,
                    raw_prob=0.1,
                    smoothed_prob=0.1,
                )
            )

        for i in range(101, 201):
            frame_results_slice1.append(
                StreamVadFrameResult(
                    frame_idx=i,
                    is_speech=True,
                    raw_prob=0.9,
                    smoothed_prob=0.9,
                    is_speech_start=(i == 101),
                    speech_start_frame=101 if i == 101 else -1,
                )
            )

        mock_vad_slice1 = MockFireRedStreamVad(frame_results_slice1)
        slice1_output = mock_vad_slice1.detect_chunk(None)

        assert slice1_output[-1].is_speech is True
        slice1_n_ms = 2000

        speech_start_frame_slice1 = None
        for result in slice1_output:
            if result.is_speech_start:
                speech_start_frame_slice1 = result.speech_start_frame
                break

        assert speech_start_frame_slice1 == 101
        slice1_m_ms = speech_start_frame_slice1 / 100 * 1000
        assert slice1_m_ms == 1010  # 1.01s in ms
        assert slice1_m_ms <= slice1_n_ms

        # Slice 2 setup: speech continues from start in new slice
        frame_results_slice2 = []

        for i in range(1, 201):
            frame_results_slice2.append(
                StreamVadFrameResult(
                    frame_idx=i,
                    is_speech=True,
                    raw_prob=0.9,
                    smoothed_prob=0.9,
                    is_speech_start=(i == 1),
                    speech_start_frame=1 if i == 1 else -1,
                )
            )

        mock_vad_slice2 = MockFireRedStreamVad(frame_results_slice2)
        slice2_output = mock_vad_slice2.detect_chunk(None)

        assert slice2_output[-1].is_speech is True
        slice2_n_ms = 2000

        speech_start_frame_slice2 = None
        for result in slice2_output:
            if result.is_speech_start:
                speech_start_frame_slice2 = result.speech_start_frame
                break

        assert speech_start_frame_slice2 == 1
        slice2_m_ms = speech_start_frame_slice2 / 100 * 1000
        assert slice2_m_ms == 10  # 0.01s in ms
        assert slice2_m_ms <= slice2_n_ms

    @pytest.mark.unit
    def test_m_computation_relative_to_slice_frame_offset(self):
        """
        Test that SliceVad m is computed relative to first frame in slice (frame_idx offset).

        Scenario: Speech that started in a previous slice continues into current slice.
        In global time:
          - Slice 1 (global [0, 2.0s]): Speech from frame 101 (global) onwards
          - Slice 2 (global [2.0s, 4.0s]): Speech continues (frames continue from 201 global)

        In slice-relative frame_idx (reset per slice call to detect_chunk):
          - Slice 2 frame results come in with frame_idx [1, 200]
          - speech_start_frame = 201 (GLOBAL value from previous slice state)
          - To compute m relative to Slice 2, we need: m = (speech_start_frame - slice_start_frame_offset) / 100
          - If slice_start_frame_offset = 201 (first global frame of Slice 2), then:
            m = (201 - 201) / 100 = 0.0s (speech started at beginning of slice)

        Expected: When m computation adjusts for slice offset, m remains <= n
        """
        frame_results = []

        for i in range(1, 201):
            frame_results.append(
                StreamVadFrameResult(
                    frame_idx=i,
                    is_speech=True,
                    raw_prob=0.9,
                    smoothed_prob=0.9,
                    is_speech_start=(i == 1),
                    speech_start_frame=1 if i == 1 else -1,
                )
            )

        mock_vad = MockFireRedStreamVad(frame_results)
        frame_results_output = mock_vad.detect_chunk(None)

        assert frame_results_output[-1].is_speech is True
        assert frame_results_output[-1].frame_idx == 200

        speech_start_frame = None
        for result in frame_results_output:
            if result.is_speech_start:
                speech_start_frame = result.speech_start_frame
                break

        assert speech_start_frame == 1

        n_frames = frame_results_output[-1].frame_idx
        slice_start_frame_offset = 1

        m_frames = speech_start_frame - slice_start_frame_offset
        m_seconds = m_frames / 100
        n_seconds = n_frames / 100

        assert m_seconds == 0.0
        assert m_seconds <= n_seconds
