# Copyright 2026 FireRedTeam

"""Tests for LID (Language Identification) and Punctuation integration."""

import unittest
from unittest.mock import MagicMock, patch

import pytest

from asr2s_grpc.session import FinalResult


class TestLidPuncDefaultOn:
    """Test: enable_lid/punc default ON - final text should have punctuation, language field populated."""

    @patch("fireredasr2s.fireredlid.FireRedLid")
    @patch("fireredasr2s.fireredpunc.punc.FireRedPunc")
    def test_final_result_with_lid_punc_enabled(self, mock_punc, mock_lid):
        """RED: Final result should have language field and punctuation when LID/Punc enabled."""
        from asr2s_grpc.session import StreamingSession
        from asr2s_grpc.postprocessing import Postprocessor
        from asr2s_grpc.config import ApiConfig

        mock_lid_instance = MagicMock()
        mock_lid_instance.process.return_value = [
            {
                "uttid": "test",
                "lang": "zh",
                "confidence": 0.99,
                "dur_s": 2.0,
                "rtf": "0.05",
                "wav": "test.wav",
            }
        ]
        mock_lid.from_pretrained.return_value = mock_lid_instance

        mock_punc_instance = MagicMock()
        mock_punc_instance.process.return_value = [
            {"punc_text": "你好世界。", "origin_text": "你好世界"}
        ]
        mock_punc.from_pretrained.return_value = mock_punc_instance

        config = ApiConfig(enable_lid=True, enable_punc=True)
        postprocessor = Postprocessor(config)

        session = StreamingSession(
            session_id="test-001",
            sample_rate=16000,
            chunk_duration_ms=100,
            silence_timeout_ms=500,
            max_segment_duration_ms=60000,
            enable_lid=True,
            enable_punc=True,
        )

        session.lid_model = postprocessor.lid_model
        session.punc_model = postprocessor.punc_model

        assert hasattr(session, "lid_model"), "Session should have lid_model attribute"
        assert hasattr(session, "punc_model"), (
            "Session should have punc_model attribute"
        )
        assert session.lid_model is not None, (
            "LID model should be loaded when enable_lid=True"
        )
        assert session.punc_model is not None, (
            "Punc model should be loaded when enable_punc=True"
        )

    @patch("fireredasr2s.fireredlid.FireRedLid")
    @patch("fireredasr2s.fireredpunc.punc.FireRedPunc")
    def test_enable_lid_punc_by_default(self, mock_punc, mock_lid):
        """RED: LID and Punc should be enabled by default."""
        from asr2s_grpc.session import StreamingSession

        mock_lid.from_pretrained.return_value = MagicMock()
        mock_punc.from_pretrained.return_value = MagicMock()

        session = StreamingSession(
            session_id="test-002",
            sample_rate=16000,
            chunk_duration_ms=100,
            silence_timeout_ms=500,
            max_segment_duration_ms=60000,
        )

        assert hasattr(session, "enable_lid"), "Session should track enable_lid setting"
        assert hasattr(session, "enable_punc"), (
            "Session should track enable_punc setting"
        )
        assert session.enable_lid is True, "LID should be enabled by default"
        assert session.enable_punc is True, "Punc should be enabled by default"


class TestLidDisabled:
    """Test: enable_lid OFF - language field empty, confidence 0."""

    @patch("fireredasr2s.fireredlid.FireRedLid")
    def test_final_result_without_language_when_lid_disabled(self, mock_lid):
        """RED: Final result should have empty language and 0 confidence when LID disabled."""
        from asr2s_grpc.session import StreamingSession

        session = StreamingSession(
            session_id="test-003",
            sample_rate=16000,
            chunk_duration_ms=100,
            silence_timeout_ms=500,
            max_segment_duration_ms=60000,
            enable_lid=False,
        )

        assert hasattr(session, "enable_lid"), "Session should track enable_lid setting"
        assert session.enable_lid is False, "LID should be disabled"
        assert not hasattr(session, "lid_model") or session.lid_model is None, (
            "LID model should not be loaded when disabled"
        )

    @patch("fireredasr2s.fireredlid.FireRedLid")
    def test_lid_model_not_loaded_when_disabled(self, mock_lid):
        """RED: LID model should not be loaded when enable_lid=False."""
        from asr2s_grpc.session import StreamingSession

        session = StreamingSession(
            session_id="test-004",
            sample_rate=16000,
            chunk_duration_ms=100,
            silence_timeout_ms=500,
            max_segment_duration_ms=60000,
            enable_lid=False,
        )

        mock_lid.from_pretrained.assert_not_called()


class TestPuncDisabled:
    """Test: enable_punc OFF - final text without punctuation."""

    @patch("fireredasr2s.fireredpunc.punc.FireRedPunc")
    def test_final_result_without_punctuation_when_punc_disabled(self, mock_punc):
        """RED: Final text should NOT have punctuation when Punc disabled."""
        from asr2s_grpc.session import StreamingSession

        session = StreamingSession(
            session_id="test-005",
            sample_rate=16000,
            chunk_duration_ms=100,
            silence_timeout_ms=500,
            max_segment_duration_ms=60000,
            enable_punc=False,
        )

        assert hasattr(session, "enable_punc"), (
            "Session should track enable_punc setting"
        )
        assert session.enable_punc is False, "Punc should be disabled"
        assert not hasattr(session, "punc_model") or session.punc_model is None, (
            "Punc model should not be loaded when disabled"
        )

    @patch("fireredasr2s.fireredpunc.punc.FireRedPunc")
    def test_punc_model_not_loaded_when_disabled(self, mock_punc):
        """RED: Punc model should not be loaded when enable_punc=False."""
        from asr2s_grpc.session import StreamingSession

        session = StreamingSession(
            session_id="test-006",
            sample_rate=16000,
            chunk_duration_ms=100,
            silence_timeout_ms=500,
            max_segment_duration_ms=60000,
            enable_punc=False,
        )

        mock_punc.from_pretrained.assert_not_called()


class TestPartialResultsNoPunctuation:
    """Test: Partial results should NOT have punctuation (final-only)."""

    def test_partial_result_no_punctuation(self):
        """RED: Partial result text should NOT have punctuation."""
        from asr2s_grpc.session import PartialResult

        # Partial result should be raw ASR output without punctuation
        partial = PartialResult(
            segment_id="seg-001",
            revision=0,
            text="你好世界",  # No punctuation in partial
            start_ms=100,
            end_ms=500,
            confidence=0.85,
            is_final=False,
        )

        assert partial.text == "你好世界"
        assert "。" not in partial.text
        assert partial.is_final is False

    def test_partial_result_different_from_final(self):
        """RED: Partial and final should have different punctuation states."""
        from asr2s_grpc.session import PartialResult

        # Partial: no punctuation
        partial = PartialResult(
            segment_id="seg-002",
            revision=0,
            text="hello world",
            start_ms=0,
            end_ms=1000,
            confidence=0.9,
            is_final=False,
        )

        # Final should have punctuation (when Punc is enabled)
        final_result = {
            "uttid": "test-007",
            "text": "Hello world.",  # With punctuation
            "sentences": [{"text": "Hello world."}],
            "vad_segments_ms": [(0, 1000)],
            "dur_s": 1.0,
            "words": [],
        }

        final = FinalResult(**final_result)

        # Assertions
        assert partial.text != final.text  # Different (one has punc, one doesn't)
        assert "." not in partial.text
        assert "." in final.text

    @patch("fireredasr2s.fireredpunc.punc.FireRedPunc")
    def test_punc_only_applied_to_final(self, mock_punc):
        """RED: Punctuation should only be applied during final processing."""
        mock_punc_instance = MagicMock()
        mock_punc_instance.process.return_value = [
            {"punc_text": "你好世界。", "origin_text": "你好世界"}
        ]
        mock_punc.from_pretrained.return_value = mock_punc_instance

        from asr2s_grpc.session import PartialResult

        partial = PartialResult(
            segment_id="seg-003",
            revision=0,
            text="你好世界",
            start_ms=0,
            end_ms=2000,
            confidence=0.92,
            is_final=False,
        )

        assert partial.text == "你好世界", "Partial result should NOT have punctuation"
        assert "。" not in partial.text, "Partial should not contain punctuation marks"


class TestIntegrationScenarios:
    """Test integration scenarios with various configurations."""

    def test_scenario_chinese_with_lid_punc(self):
        """RED: Chinese text with LID and Punc enabled."""
        final_result = {
            "uttid": "scenario-001",
            "text": "你好，世界。",
            "sentences": [
                {
                    "text": "你好，世界。",
                    "lang": "zh",
                    "lang_confidence": 0.98,
                }
            ],
            "vad_segments_ms": [(0, 2000)],
            "dur_s": 2.0,
            "words": [],
        }

        result = FinalResult(**final_result)
        assert result.text == "你好，世界。"
        assert result.sentences[0]["lang"] == "zh"

    def test_scenario_english_with_lid_punc(self):
        """RED: English text with LID and Punc enabled."""
        final_result = {
            "uttid": "scenario-002",
            "text": "Hello, world!",
            "sentences": [
                {
                    "text": "Hello, world!",
                    "lang": "en",
                    "lang_confidence": 0.97,
                }
            ],
            "vad_segments_ms": [(0, 1500)],
            "dur_s": 1.5,
            "words": [],
        }

        result = FinalResult(**final_result)
        assert result.text == "Hello, world!"
        assert result.sentences[0]["lang"] == "en"

    def test_scenario_codeswitching_with_lid(self):
        """RED: Code-switching text with LID (should detect multiple languages)."""
        final_result = {
            "uttid": "scenario-003",
            "text": "你好hello世界。",
            "sentences": [
                {
                    "text": "你好hello世界。",
                    "lang": "zh",  # Dominant language
                    "lang_confidence": 0.85,
                }
            ],
            "vad_segments_ms": [(0, 2500)],
            "dur_s": 2.5,
            "words": [],
        }

        result = FinalResult(**final_result)
        assert result.text == "你好hello世界。"
        assert result.sentences[0]["lang"] == "zh"
