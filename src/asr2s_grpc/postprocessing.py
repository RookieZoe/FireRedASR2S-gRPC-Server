# Copyright 2026 FireRedTeam

"""Postprocessing module for LID (Language Identification) and Punc (Punctuation)."""

from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import ApiConfig, LidConfig, PuncConfig

logger = logging.getLogger(__name__)


class Postprocessor:
    """Conditional loader and runner for LID and Punc postprocessing models.

    Models are loaded lazily: only when the corresponding ``enable_*`` flag is
    ``True`` at construction time.  Once loaded, the model instances are cached
    for the lifetime of this object.

    Args:
        config: The top-level :class:`ApiConfig` instance.
    """

    def __init__(self, config: ApiConfig) -> None:
        self.enable_lid: bool = config.enable_lid
        self.enable_punc: bool = config.enable_punc

        self.lid_model: Any = None
        self.punc_model: Any = None

        self._lid_config: LidConfig = config.lid
        self._punc_config: PuncConfig = config.punc

        if self.enable_lid:
            self._load_lid()

        if self.enable_punc:
            self._load_punc()

    def _load_lid(self) -> None:
        try:
            from fireredasr2s.fireredlid import FireRedLid, FireRedLidConfig

            lid_cfg = FireRedLidConfig(
                use_gpu=self._lid_config.use_gpu,
                use_half=self._lid_config.use_half,
            )
            self.lid_model = FireRedLid.from_pretrained(
                self._lid_config.model_dir, lid_cfg
            )
            logger.info("Loaded LID model from %s", self._lid_config.model_dir)
        except Exception:
            logger.exception("Failed to load LID model")
            raise

    def _load_punc(self) -> None:
        try:
            from fireredasr2s.fireredpunc.punc import FireRedPunc, FireRedPuncConfig

            punc_cfg = FireRedPuncConfig(
                use_gpu=self._punc_config.use_gpu,
                sentence_max_length=self._punc_config.sentence_max_length,
            )
            self.punc_model = FireRedPunc.from_pretrained(
                self._punc_config.model_dir, punc_cfg
            )
            logger.info("Loaded Punc model from %s", self._punc_config.model_dir)
        except Exception:
            logger.exception("Failed to load Punc model")
            raise

    def process_lid(
        self,
        audio_data: NDArray[np.int16],
        sample_rate: int = 16000,
    ) -> Tuple[str, float]:
        """Run language identification on the full audio.

        LID should be called **once** on the final, complete audio segment
        rather than on every partial chunk.

        Args:
            audio_data: 1-D numpy array of audio samples (int16 or float).
            sample_rate: Sample rate in Hz (must be 16000).

        Returns:
            A ``(language, confidence)`` tuple.  *language* is a string such
            as ``"zh mandarin"`` or ``"en"``; *confidence* is a float in
            [0, 1].  Returns ``("", 0.0)`` when LID is disabled or the
            model is unavailable.
        """
        if not self.enable_lid or self.lid_model is None:
            logger.debug("LID is disabled or model not loaded; skipping")
            return ("", 0.0)

        try:
            # FireRedLid.process accepts (sample_rate, wav_np) tuples in
            # place of file paths (see fireredlid/data/feat.py:25-30).
            wav_data = (sample_rate, audio_data)
            results = self.lid_model.process(["api_lid"], [wav_data])

            if results and results[0].get("lang"):
                lang = results[0]["lang"]
                confidence = float(results[0].get("confidence", 0.0))
                return (lang, confidence)

            return ("", 0.0)

        except Exception:
            logger.exception("LID processing failed")
            return ("", 0.0)

    def process_punc(self, text: str) -> str:
        """Add punctuation to an unpunctuated text string.

        Uses the text-only ``process()`` path of FireRedPunc (no timestamp
        dependency).

        Args:
            text: Raw transcription text without punctuation.

        Returns:
            Punctuated text string.  Returns the original *text* unchanged
            when Punc is disabled or an error occurs.
        """
        if not self.enable_punc or self.punc_model is None:
            logger.debug("Punc is disabled or model not loaded; skipping")
            return text

        if not text or not text.strip():
            return text

        try:
            results = self.punc_model.process([text])

            if results and results[0].get("punc_text"):
                return results[0]["punc_text"]

            return text

        except Exception:
            logger.exception("Punc processing failed")
            return text
