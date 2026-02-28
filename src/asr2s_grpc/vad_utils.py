# Copyright 2026 FireRedTeam
"""Shared VAD utilities for gRPC server."""

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedClass=false, reportUnusedImport=false, reportUnknownArgumentType=false, reportUnnecessaryCast=false

import logging
import os
from typing import cast

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# VAD frame = 10ms
_VAD_FRAMES_PER_MS = 0.1
_MS_PER_VAD_FRAME = 10


def preload_vad_models(
    model_dirs: dict[str, str],
    use_gpu: bool = False,
) -> dict[str, object]:
    """Preload VAD models at server startup.

    Args:
        model_dirs: Mapping of vad_type -> model directory path,
            as returned by resolve_vad_model_dirs().
        use_gpu: Whether to use GPU for VAD models.

    Returns:
        Dict mapping vad_type -> loaded model object.

    Raises:
        RuntimeError: If FireRedVAD is not available or model loading fails.
    """
    try:
        from fireredasr2s.fireredvad import (  # type: ignore[reportMissingImports]
            FireRedStreamVad,
            FireRedStreamVadConfig,
        )
    except ImportError as exc:
        raise RuntimeError("FireRedVAD not available") from exc

    models: dict[str, object] = {}

    for vad_type, model_dir in model_dirs.items():
        model_pth = os.path.join(model_dir, "model.pth.tar")
        cmvn_ark = os.path.join(model_dir, "cmvn.ark")
        if not os.path.isfile(model_pth):
            raise RuntimeError(f"VAD model file not found: {model_pth}")
        if not os.path.isfile(cmvn_ark):
            raise RuntimeError(f"VAD model file not found: {cmvn_ark}")

        if vad_type == "stream-vad":
            config = FireRedStreamVadConfig(use_gpu=use_gpu)
            model = FireRedStreamVad.from_pretrained(model_dir, config)
        elif vad_type == "vad":
            from fireredasr2s.fireredvad import (  # type: ignore[reportMissingImports]
                FireRedVad,
                FireRedVadConfig,
            )

            vad_config = FireRedVadConfig(use_gpu=use_gpu)
            model = FireRedVad.from_pretrained(model_dir, vad_config)
        elif vad_type == "aed":
            from fireredasr2s.fireredvad import (  # type: ignore[reportMissingImports]
                FireRedAed,
                FireRedAedConfig,
            )

            aed_config = FireRedAedConfig(use_gpu=use_gpu)
            model = FireRedAed.from_pretrained(model_dir, aed_config)
        else:
            raise ValueError(f"Unknown vad_type: {vad_type}")

        models[vad_type] = model
        logger.info("Preloaded %s VAD model from %s", vad_type, model_dir)

    logger.info("Preloaded %s VAD model(s)", len(models))
    return models


class _SessionVadState:
    """Per-session VAD state maintained across slices."""

    vad_type: str
    _stream_vad: object | None
    _vad: object | None
    _aed: object | None
    _vad_detect_result: dict[str, object] | None
    _aed_detect_result: dict[str, object] | None
    _available: bool
    _frame_offset: int
    _frame_length_sample: int

    def __init__(
        self,
        vad_type: str = "stream-vad",
        preloaded_models: dict[str, object] | None = None,
        model_dirs: dict[str, str] | None = None,
    ) -> None:
        self.vad_type = vad_type
        self._stream_vad = None
        self._vad = None
        self._aed = None
        self._vad_detect_result = None
        self._aed_detect_result = None
        self._available = False
        self._frame_offset = 0  # Track first frame idx of current slice
        try:
            from fireredasr2s.fireredvad import (  # type: ignore[reportMissingImports]
                FireRedStreamVad,
                FireRedStreamVadConfig,
            )
            from fireredasr2s.fireredvad.core.constants import (  # type: ignore[reportMissingImports]
                FRAME_LENGTH_SAMPLE,
            )

            self._frame_length_sample = FRAME_LENGTH_SAMPLE
            self._available = True
        except ImportError:
            self._frame_length_sample = 400
            self._available = False

        if preloaded_models and self._available:
            self._init_from_preloaded(preloaded_models, model_dirs)

    def _init_from_preloaded(
        self,
        preloaded_models: dict[str, object],
        model_dirs: dict[str, str] | None = None,
    ) -> None:
        """Initialize session VAD from preloaded server-level models.

        For stream-vad: creates a fresh instance per session (stateful).
        For vad/aed: shares the preloaded model reference (stateless per-call).
        """
        if self.vad_type in {"stream-vad", "all"}:
            if "stream-vad" in preloaded_models:
                if model_dirs and "stream-vad" in model_dirs:
                    # stream-vad is stateful — create fresh instance per session
                    from fireredasr2s.fireredvad import (  # type: ignore[reportMissingImports]
                        FireRedStreamVad,
                        FireRedStreamVadConfig,
                    )

                    config = FireRedStreamVadConfig(use_gpu=False)
                    self._stream_vad = FireRedStreamVad.from_pretrained(
                        model_dirs["stream-vad"], config
                    )
                else:
                    self._stream_vad = preloaded_models["stream-vad"]

        if self.vad_type in {"vad", "all"}:
            if "vad" in preloaded_models:
                # vad is stateless per-call — share directly
                self._vad = preloaded_models["vad"]

        if self.vad_type in {"aed", "all"}:
            if "aed" in preloaded_models:
                # aed is stateless per-call — share directly
                self._aed = preloaded_models["aed"]

    def initialize(self, vad_model_dir: object | None = None) -> None:
        """Legacy per-session initialization.

        Prefer passing preloaded_models to __init__ instead.
        """
        if not self._available:
            raise RuntimeError("FireRedVAD not available")
        try:
            from fireredasr2s.fireredvad import (  # type: ignore[reportMissingImports]
                FireRedStreamVad,
                FireRedStreamVadConfig,
            )

            config = FireRedStreamVadConfig(use_gpu=False)
            base_model_dir = vad_model_dir
            model_dirs = (
                cast(dict[str, str], vad_model_dir)
                if isinstance(vad_model_dir, dict)
                else {}
            )

            if self.vad_type in {"stream-vad", "all"}:
                stream_model_dir = model_dirs.get("stream-vad")
                if stream_model_dir is None and isinstance(base_model_dir, str):
                    stream_model_dir = base_model_dir
                if isinstance(stream_model_dir, str) and stream_model_dir:
                    # Validate model files exist before loading
                    model_pth = os.path.join(stream_model_dir, "model.pth.tar")
                    cmvn_ark = os.path.join(stream_model_dir, "cmvn.ark")
                    if not os.path.isfile(model_pth):
                        raise RuntimeError(f"VAD model file not found: {model_pth}")
                    if not os.path.isfile(cmvn_ark):
                        raise RuntimeError(f"VAD model file not found: {cmvn_ark}")
                    self._stream_vad = FireRedStreamVad.from_pretrained(
                        stream_model_dir, config
                    )
                    logger.info("Loaded stream-vad model from %s", stream_model_dir)
                else:
                    raise RuntimeError("VAD model directory not provided")

            if self.vad_type in {"vad", "all"}:
                try:
                    from fireredasr2s.fireredvad import (  # type: ignore[reportMissingImports]
                        FireRedVad,
                        FireRedVadConfig,
                    )

                    vad_dir = model_dirs.get("vad")
                    if vad_dir is None and isinstance(base_model_dir, str):
                        vad_dir = base_model_dir
                    if not isinstance(vad_dir, str) or not vad_dir:
                        raise RuntimeError("VAD model directory not provided")
                    vad_config = FireRedVadConfig(use_gpu=False)
                    self._vad = FireRedVad.from_pretrained(vad_dir, vad_config)
                    logger.info("Loaded vad model from %s", vad_dir)
                except Exception as e:
                    logger.error("Failed to initialize VAD: %s", e)
                    raise

            if self.vad_type in {"aed", "all"}:
                try:
                    from fireredasr2s.fireredvad import (  # type: ignore[reportMissingImports]
                        FireRedAed,
                        FireRedAedConfig,
                    )

                    aed_dir = model_dirs.get("aed")
                    if aed_dir is None and isinstance(base_model_dir, str):
                        aed_dir = base_model_dir
                    if not isinstance(aed_dir, str) or not aed_dir:
                        raise RuntimeError("AED model directory not provided")
                    aed_config = FireRedAedConfig(use_gpu=False)
                    self._aed = FireRedAed.from_pretrained(aed_dir, aed_config)
                    logger.info("Loaded aed model from %s", aed_dir)
                except Exception as e:
                    logger.error("Failed to initialize AED: %s", e)
                    raise

        except Exception as e:
            logger.error("Failed to initialize VAD: %s", e)
            raise RuntimeError(f"FireRedVAD initialization failed: {e}")

    def process_slice_audio(
        self,
        audio_data: NDArray[np.int16],
        sample_rate: int,
    ) -> "SliceVadResult":
        """Run VAD on a slice and return VAD analysis."""
        n_samples = len(audio_data)
        n_frames = n_samples * 1000 // (sample_rate * _MS_PER_VAD_FRAME)

        if not self._available:
            raise RuntimeError("VAD not initialized")

        audio_np = audio_data

        if self.vad_type in {"stream-vad", "all"}:
            if self._stream_vad is None:
                raise RuntimeError("StreamVAD not initialized")
            stream_vad = cast(object, self._stream_vad)
            frame_results = cast(
                list[object], getattr(stream_vad, "detect_chunk")(audio_data)
            )
        elif self.vad_type == "vad":
            if self._vad is None:
                raise RuntimeError("VAD not initialized")
            vad_model = cast(object, self._vad)
            vad_result, _ = cast(
                tuple[dict[str, object], object],
                getattr(vad_model, "detect")(audio_np),
            )
            self._vad_detect_result = vad_result
            return self._convert_vad_result(self._vad_detect_result, n_frames)
        elif self.vad_type == "aed":
            if self._aed is None:
                raise RuntimeError("AED not initialized")
            aed_model = cast(object, self._aed)
            aed_result, _ = cast(
                tuple[dict[str, object], object],
                getattr(aed_model, "detect")(audio_np),
            )
            self._aed_detect_result = aed_result
            return self._convert_aed_result(self._aed_detect_result, n_frames)
        else:
            raise RuntimeError(f"Unknown vad_type: {self.vad_type}")

        if not frame_results:
            self._frame_offset = 0
            return SliceVadResult(
                n_frames=n_frames,
                ended_speaking=False,
                speech_start_frame=None,
                entirely_speech=False,
                frame_offset=self._frame_offset,
            )

        # Track first frame index for offset calculation
        first_frame_idx = int(
            getattr(frame_results[0], "frame_idx", 0) if frame_results else 0
        )
        self._frame_offset = first_frame_idx

        ended_speaking = False
        speech_start_frame: int | None = None
        entirely_speech = True

        for result in frame_results:
            is_speech = bool(getattr(result, "is_speech", False))
            if not is_speech:
                entirely_speech = False
            is_speech_start = bool(getattr(result, "is_speech_start", False))
            start_frame = int(getattr(result, "speech_start_frame", -1))
            if is_speech_start and start_frame > 0:
                speech_start_frame = start_frame
            ended_speaking = is_speech

        slice_result = SliceVadResult(
            n_frames=n_frames,
            ended_speaking=ended_speaking,
            speech_start_frame=speech_start_frame,
            entirely_speech=entirely_speech and len(frame_results) > 0,
            frame_offset=self._frame_offset,
        )

        if self.vad_type == "all":
            if self._vad is not None:
                vad_model = cast(object, self._vad)
                vad_result, _ = cast(
                    tuple[dict[str, object], object],
                    getattr(vad_model, "detect")(audio_np),
                )
                self._vad_detect_result = vad_result
            if self._aed is not None:
                aed_model = cast(object, self._aed)
                aed_result, _ = cast(
                    tuple[dict[str, object], object],
                    getattr(aed_model, "detect")(audio_np),
                )
                self._aed_detect_result = aed_result

        return slice_result

    def get_vad_detect_result(self) -> dict[str, object] | None:
        return self._vad_detect_result

    def get_aed_detect_result(self) -> dict[str, object] | None:
        return self._aed_detect_result

    def _convert_vad_result(
        self, vad_result: dict[str, object] | None, n_frames: int
    ) -> "SliceVadResult":
        timestamps: list[tuple[float, float]] = []
        dur = n_frames / 100
        if vad_result is not None:
            dur_value = vad_result.get("dur", dur)
            if isinstance(dur_value, (int, float)):
                dur = float(dur_value)
            timestamps = cast(
                list[tuple[float, float]],
                vad_result.get("timestamps", []),
            )
        return self._timestamps_to_slice_result(timestamps, dur, n_frames)

    def _convert_aed_result(
        self, aed_result: dict[str, object] | None, n_frames: int
    ) -> "SliceVadResult":
        timestamps: list[tuple[float, float]] = []
        dur = n_frames / 100
        if aed_result is not None:
            dur_value = aed_result.get("dur", dur)
            if isinstance(dur_value, (int, float)):
                dur = float(dur_value)
            event2timestamps = aed_result.get("event2timestamps", {})
            if isinstance(event2timestamps, dict):
                timestamps = cast(
                    list[tuple[float, float]],
                    event2timestamps.get("speech", []),
                )
        return self._timestamps_to_slice_result(timestamps, dur, n_frames)

    def _timestamps_to_slice_result(
        self, timestamps: list[tuple[float, float]], dur: float, n_frames: int
    ) -> "SliceVadResult":
        if not timestamps:
            return SliceVadResult(
                n_frames=n_frames,
                ended_speaking=False,
                speech_start_frame=None,
                entirely_speech=False,
                frame_offset=0,
            )

        last_start, last_end = timestamps[-1]
        ended_speaking = last_end >= dur
        speech_start_frame = None
        if ended_speaking:
            speech_start_frame = int(round(last_start * 100))

        entirely_speech = (
            len(timestamps) == 1 and timestamps[0][0] <= 0 and timestamps[0][1] >= dur
        )

        return SliceVadResult(
            n_frames=n_frames,
            ended_speaking=ended_speaking,
            speech_start_frame=speech_start_frame,
            entirely_speech=entirely_speech,
            frame_offset=0,
        )


class SliceVadResult:
    n_frames: int
    ended_speaking: bool
    speech_start_frame: int | None
    entirely_speech: bool
    frame_offset: int
    __slots__: tuple[str, ...]
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
        speech_start_frame: int | None,
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
