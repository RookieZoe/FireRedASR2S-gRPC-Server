# Copyright 2026 FireRedTeam

"""ASR backend implementations for FireRedASR2S API."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ASRBackend(ABC):
    """Abstract base class for ASR backends."""

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        return_timestamps: bool = False,
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (int16)
            sample_rate: Sample rate of audio
            return_timestamps: Whether to return word timestamps

        Returns:
            Dictionary with transcription results
        """
        pass

    @abstractmethod
    def get_max_audio_length(self) -> float:
        """
        Get maximum audio length in seconds that this backend can process.

        Returns:
            Maximum audio length in seconds
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Release model resources (GPU memory, weights, etc.).

        Called when switch_offload is enabled and this backend is being
        evicted from the cache to free resources for another backend.
        """
        pass


class AEDBackend(ASRBackend):
    """AED (Attention-based Encoder-Decoder) backend."""

    def __init__(
        self,
        model_dir: str,
        use_gpu: bool = True,
        beam_size: int = 3,
        return_timestamps: bool = True,
    ):
        """
        Initialize AED backend.

        Args:
            model_dir: Path to model directory
            use_gpu: Whether to use GPU
            beam_size: Beam size for decoding
            return_timestamps: Whether to return word timestamps
        """
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        self.beam_size = beam_size
        self.return_timestamps = return_timestamps

        self._load_model()

    def _load_model(self) -> None:
        """Load the AED model."""
        try:
            from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

            config = FireRedAsr2Config(
                use_gpu=self.use_gpu,
                beam_size=self.beam_size,
                return_timestamp=self.return_timestamps,
            )

            self.model = FireRedAsr2.from_pretrained(
                "aed",
                self.model_dir,
                config,
            )
            logger.info(f"Loaded AED model from {self.model_dir}")

        except Exception as e:
            logger.error(f"Failed to load AED model: {e}")
            raise

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        return_timestamps: bool = False,
    ) -> Dict[str, Any]:
        """
        Transcribe audio using AED model.

        Args:
            audio: Audio data as numpy array (int16)
            sample_rate: Sample rate of audio
            return_timestamps: Whether to return word timestamps

        Returns:
            Dictionary with transcription results
        """
        try:
            # Prepare audio data
            uttid = "stream_0001"
            wav_path = (sample_rate, audio)

            # Run transcription
            results = self.model.transcribe([uttid], [wav_path])

            if not results:
                return {
                    "text": "",
                    "confidence": 0.0,
                    "words": [],
                }

            result = results[0]

            # Extract timestamps if available
            words = []
            if "timestamp" in result:
                for word, start, end in result["timestamp"]:
                    words.append(
                        {
                            "text": word,
                            "start_ms": int(start * 1000),
                            "end_ms": int(end * 1000),
                        }
                    )

            return {
                "text": result.get("text", ""),
                "confidence": result.get("confidence", 0.0),
                "words": words,
            }

        except Exception as e:
            logger.exception(f"Transcription failed: {e}")
            raise

    def get_max_audio_length(self) -> float:
        """
        Get maximum audio length for AED model.

        Returns:
            Maximum audio length in seconds (60s for AED)
        """
        return 60.0

    def cleanup(self) -> None:
        """Release AED model resources."""
        if hasattr(self, "model"):
            del self.model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("AED backend cleaned up")


class LLMBackend(ASRBackend):
    """LLM (Large Language Model) backend for FireRedASR2."""

    def __init__(
        self,
        model_dir: str,
        use_gpu: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LLM backend.

        Args:
            model_dir: Path to model directory
            use_gpu: Whether to use GPU
            config: Optional dict with LLM decode params
                (decode_min_len, repetition_penalty, llm_length_penalty, temperature)
        """
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        self.default_config = config or {}

        self._load_model()

    def _load_model(self) -> None:
        """Load the LLM model."""
        try:
            from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

            config = FireRedAsr2Config(
                use_gpu=self.use_gpu,
                decode_min_len=self.default_config.get("decode_min_len", 0),
                repetition_penalty=self.default_config.get("repetition_penalty", 1.0),
                llm_length_penalty=self.default_config.get("llm_length_penalty", 0.0),
                temperature=self.default_config.get("temperature", 1.0),
            )

            self.model = FireRedAsr2.from_pretrained(
                "llm",
                self.model_dir,
                config,
            )
            logger.info(f"Loaded LLM model from {self.model_dir}")

        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        return_timestamps: bool = False,
        llm_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio using LLM model.

        Args:
            audio: Audio data as numpy array (int16)
            sample_rate: Sample rate of audio
            return_timestamps: Ignored for LLM (no timestamp support)
            llm_params: Optional per-request LLM decode params
                (decode_min_len, repetition_penalty, llm_length_penalty, temperature)

        Returns:
            Dictionary with transcription results (text and words)
        """
        try:
            # If per-request params provided, create a new config for this call
            if llm_params:
                from fireredasr2s.fireredasr2 import FireRedAsr2Config

                merged = {**self.default_config, **llm_params}
                request_config = FireRedAsr2Config(
                    use_gpu=self.use_gpu,
                    decode_min_len=merged.get("decode_min_len", 0),
                    repetition_penalty=merged.get("repetition_penalty", 1.0),
                    llm_length_penalty=merged.get("llm_length_penalty", 0.0),
                    temperature=merged.get("temperature", 1.0),
                )
                # Temporarily swap config on the model
                original_config = self.model.config
                self.model.config = request_config
                try:
                    result = self._run_transcribe(audio, sample_rate)
                finally:
                    self.model.config = original_config
            else:
                result = self._run_transcribe(audio, sample_rate)

            return result

        except Exception as e:
            logger.exception(f"LLM transcription failed: {e}")
            raise

    def _run_transcribe(
        self, audio: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """Run transcription and format the result."""
        uttid = "stream_0001"
        wav_path = (sample_rate, audio)

        results = self.model.transcribe([uttid], [wav_path])

        if not results:
            return {
                "text": "",
                "words": [],
            }

        result = results[0]

        return {
            "text": result.get("text", ""),
            "words": [],  # LLM does not produce timestamps
        }

    def get_max_audio_length(self) -> float:
        """
        Get maximum audio length for LLM model.

        Returns:
            Maximum audio length in seconds (40s for LLM)
        """
        return 40.0

    def cleanup(self) -> None:
        """Release LLM model resources."""
        if hasattr(self, "model"):
            del self.model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("LLM backend cleaned up")


def create_backend(
    backend_type: str,
    model_dir: str,
    **kwargs: Any,
) -> ASRBackend:
    """
    Factory function to create ASR backend.

    Args:
        backend_type: Type of backend ('aed' or 'llm')
        model_dir: Path to model directory
        **kwargs: Additional arguments for backend

    Returns:
        ASRBackend instance

    Raises:
        ValueError: If backend_type is not supported
    """
    if backend_type == "aed":
        return AEDBackend(model_dir, **kwargs)
    elif backend_type == "llm":
        return LLMBackend(model_dir, **kwargs)
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")
