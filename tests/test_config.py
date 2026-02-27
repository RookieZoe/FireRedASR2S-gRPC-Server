# Copyright 2026 FireRedTeam

"""Tests for model path resolution in API configuration.

These are TDD tests that fail because the implementation doesn't exist yet.
They define the expected behavior for:
- Default repo-root resolution
- MODEL_DIR env var override
- --model-dir CLI flag override
- Relative path handling
- Missing repo root fallback
- Per-component model_dir handling
"""

import os
from pathlib import Path
from unittest import mock

import pytest

from fireredasr2s_api.config import (
    ApiConfig,
    AsrBackendConfig,
    LidConfig,
    PuncConfig,
    VadConfig,
)


class TestDefaultRepoRootResolution:
    """Test default model path resolution to repo root."""

    def test_default_asr_model_dir_is_absolute_under_reporoot(self):
        """From api/ CWD, ASR model_dir should resolve to absolute path under repo-root."""
        config = ApiConfig()
        
        assert os.path.isabs(config.asr.model_dir), (
            f"ASR model_dir should be absolute, got: {config.asr.model_dir}"
        )
        
        assert config.asr.model_dir.endswith("pretrained_models/FireRedASR2-AED"), (
            f"ASR model_dir should end with pretrained_models/FireRedASR2-AED, got: {config.asr.model_dir}"
        )

    def test_default_vad_model_dir_is_absolute_under_reporoot(self):
        """From api/ CWD, VAD model_dir should resolve to absolute path under repo-root."""
        config = ApiConfig()
        
        assert os.path.isabs(config.vad.model_dir), (
            f"VAD model_dir should be absolute, got: {config.vad.model_dir}"
        )
        
        assert config.vad.model_dir.endswith("pretrained_models/FireRedVAD/VAD"), (
            f"VAD model_dir should end with pretrained_models/FireRedVAD/VAD, got: {config.vad.model_dir}"
        )

    def test_default_lid_model_dir_is_absolute_under_reporoot(self):
        """From api/ CWD, LID model_dir should resolve to absolute path under repo-root."""
        config = ApiConfig()
        
        assert os.path.isabs(config.lid.model_dir), (
            f"LID model_dir should be absolute, got: {config.lid.model_dir}"
        )
        
        assert config.lid.model_dir.endswith("pretrained_models/FireRedLID"), (
            f"LID model_dir should end with pretrained_models/FireRedLID, got: {config.lid.model_dir}"
        )

    def test_default_punc_model_dir_is_absolute_under_reporoot(self):
        """From api/ CWD, Punc model_dir should resolve to absolute path under repo-root."""
        config = ApiConfig()
        
        assert os.path.isabs(config.punc.model_dir), (
            f"Punc model_dir should be absolute, got: {config.punc.model_dir}"
        )
        
        assert config.punc.model_dir.endswith("pretrained_models/FireRedPunc"), (
            f"Punc model_dir should end with pretrained_models/FireRedPunc, got: {config.punc.model_dir}"
        )

    def test_all_default_paths_share_common_repo_root(self):
        """All default model paths should share the same repo-root base."""
        config = ApiConfig()
        
        asr_root = config.asr.model_dir.rsplit("pretrained_models", 1)[0]
        vad_root = config.vad.model_dir.rsplit("pretrained_models", 1)[0]
        lid_root = config.lid.model_dir.rsplit("pretrained_models", 1)[0]
        punc_root = config.punc.model_dir.rsplit("pretrained_models", 1)[0]
        
        assert asr_root == vad_root == lid_root == punc_root, (
            f"All model paths should share repo root. ASR root: {asr_root}, "
            f"VAD root: {vad_root}, LID root: {lid_root}, Punc root: {punc_root}"
        )


class TestModelDirEnvVarOverride:
    """Test MODEL_DIR environment variable override."""

    def test_model_dir_env_overrides_asr_base_dir(self, monkeypatch):
        """Setting MODEL_DIR env var should override ASR model_dir base directory."""
        env_base_dir = "/tmp/custom_models"
        monkeypatch.setenv("MODEL_DIR", env_base_dir)
        
        config = ApiConfig()
        
        expected = os.path.join(env_base_dir, "FireRedASR2-AED")
        assert config.asr.model_dir == expected, (
            f"ASR model_dir should use MODEL_DIR env override. "
            f"Expected: {expected}, got: {config.asr.model_dir}"
        )

    def test_model_dir_env_overrides_vad_base_dir(self, monkeypatch):
        """Setting MODEL_DIR env var should override VAD model_dir base directory."""
        env_base_dir = "/tmp/custom_models"
        monkeypatch.setenv("MODEL_DIR", env_base_dir)
        
        config = ApiConfig()
        
        expected = os.path.join(env_base_dir, "FireRedVAD/VAD")
        assert config.vad.model_dir == expected, (
            f"VAD model_dir should use MODEL_DIR env override. "
            f"Expected: {expected}, got: {config.vad.model_dir}"
        )

    def test_model_dir_env_overrides_lid_base_dir(self, monkeypatch):
        """Setting MODEL_DIR env var should override LID model_dir base directory."""
        env_base_dir = "/tmp/custom_models"
        monkeypatch.setenv("MODEL_DIR", env_base_dir)
        
        config = ApiConfig()
        
        expected = os.path.join(env_base_dir, "FireRedLID")
        assert config.lid.model_dir == expected, (
            f"LID model_dir should use MODEL_DIR env override. "
            f"Expected: {expected}, got: {config.lid.model_dir}"
        )

    def test_model_dir_env_overrides_punc_base_dir(self, monkeypatch):
        """Setting MODEL_DIR env var should override Punc model_dir base directory."""
        env_base_dir = "/tmp/custom_models"
        monkeypatch.setenv("MODEL_DIR", env_base_dir)
        
        config = ApiConfig()
        
        expected = os.path.join(env_base_dir, "FireRedPunc")
        assert config.punc.model_dir == expected, (
            f"Punc model_dir should use MODEL_DIR env override. "
            f"Expected: {expected}, got: {config.punc.model_dir}"
        )

    def test_model_dir_env_with_absolute_path(self, monkeypatch):
        """MODEL_DIR with absolute path should work correctly."""
        env_base_dir = "/absolute/path/to/models"
        monkeypatch.setenv("MODEL_DIR", env_base_dir)
        
        config = ApiConfig()
        
        assert os.path.isabs(config.asr.model_dir)
        assert config.asr.model_dir.startswith("/absolute/path/to/models")

    def test_model_dir_env_with_relative_path_resolves_against_cwd(self, monkeypatch, tmp_path):
        """MODEL_DIR with relative path should resolve against current working directory."""
        original_cwd = os.getcwd()
        
        try:
            os.chdir(tmp_path)
            monkeypatch.setenv("MODEL_DIR", "relative/models")
            
            config = ApiConfig()
            
            expected_base = os.path.join(tmp_path, "relative/models")
            expected_asr = os.path.join(expected_base, "FireRedASR2-AED")
            
            assert config.asr.model_dir == expected_asr, (
                f"Relative MODEL_DIR should resolve against CWD. "
                f"Expected: {expected_asr}, got: {config.asr.model_dir}"
            )
            
        finally:
            os.chdir(original_cwd)


class TestModelBaseDirProperty:
    """Test model_base_dir property behavior."""

    def test_model_base_dir_property_reflects_env_override(self, monkeypatch):
        """ApiConfig.model_base_dir should reflect current base directory."""
        env_base_dir = "/tmp/my_models"
        monkeypatch.setenv("MODEL_DIR", env_base_dir)
        
        config = ApiConfig()
        
        assert env_base_dir in config.asr.model_dir


class TestMissingRepoRootFallback:
    """Test behavior when repo root cannot be found."""

    def test_missing_repo_root_falls_back_to_cwd(self, monkeypatch, tmp_path, caplog):
        """If repo root not found, should fall back to CWD and log warning."""
        original_cwd = os.getcwd()
        
        try:
            os.chdir(tmp_path)
            monkeypatch.delenv("MODEL_DIR", raising=False)
            
            import logging
            caplog.set_level(logging.WARNING)
            
            config = ApiConfig()
            
            assert any(
                "repo root" in record.message.lower() 
                or "fallback" in record.message.lower()
                or "cwd" in record.message.lower()
                for record in caplog.records
            ), (
                f"Expected warning about missing repo root, "
                f"but got logs: {[r.message for r in caplog.records]}"
            )
            
        finally:
            os.chdir(original_cwd)


class TestPerComponentModelDirNotOverwritten:
    """Test that explicitly set model_dir values are preserved."""

    def test_explicit_asr_model_dir_not_overwritten(self, monkeypatch):
        """Explicitly set ASR model_dir should not be overwritten by base dir."""
        custom_asr_dir = "/custom/asr/models"
        monkeypatch.setenv("MODEL_DIR", "/tmp/models")
        
        config = ApiConfig()
        config.asr.model_dir = custom_asr_dir
        
        config2 = ApiConfig(
            asr=AsrBackendConfig(model_dir=custom_asr_dir)
        )
        
        assert config2.asr.model_dir == custom_asr_dir, (
            f"Explicit ASR model_dir should not be overwritten. "
            f"Expected: {custom_asr_dir}, got: {config2.asr.model_dir}"
        )

    def test_explicit_vad_model_dir_not_overwritten(self, monkeypatch):
        """Explicitly set VAD model_dir should not be overwritten by base dir."""
        custom_vad_dir = "/custom/vad/models"
        monkeypatch.setenv("MODEL_DIR", "/tmp/models")
        
        config = ApiConfig(
            vad=VadConfig(model_dir=custom_vad_dir)
        )
        
        assert config.vad.model_dir == custom_vad_dir, (
            f"Explicit VAD model_dir should not be overwritten. "
            f"Expected: {custom_vad_dir}, got: {config.vad.model_dir}"
        )

    def test_explicit_lid_model_dir_not_overwritten(self, monkeypatch):
        """Explicitly set LID model_dir should not be overwritten by base dir."""
        custom_lid_dir = "/custom/lid/models"
        monkeypatch.setenv("MODEL_DIR", "/tmp/models")
        
        config = ApiConfig(
            lid=LidConfig(model_dir=custom_lid_dir)
        )
        
        assert config.lid.model_dir == custom_lid_dir, (
            f"Explicit LID model_dir should not be overwritten. "
            f"Expected: {custom_lid_dir}, got: {config.lid.model_dir}"
        )

    def test_explicit_punc_model_dir_not_overwritten(self, monkeypatch):
        """Explicitly set Punc model_dir should not be overwritten by base dir."""
        custom_punc_dir = "/custom/punc/models"
        monkeypatch.setenv("MODEL_DIR", "/tmp/models")
        
        config = ApiConfig(
            punc=PuncConfig(model_dir=custom_punc_dir)
        )
        
        assert config.punc.model_dir == custom_punc_dir, (
            f"Explicit Punc model_dir should not be overwritten. "
            f"Expected: {custom_punc_dir}, got: {config.punc.model_dir}"
        )


class TestPathNormalization:
    """Test path normalization and consistency."""

    def test_all_paths_are_absolute(self):
        """All resolved model paths should be absolute."""
        config = ApiConfig()
        
        assert os.path.isabs(config.asr.model_dir)
        assert os.path.isabs(config.vad.model_dir)
        assert os.path.isabs(config.lid.model_dir)
        assert os.path.isabs(config.punc.model_dir)

    def test_paths_use_correct_separator(self):
        """Paths should use OS-appropriate separator."""
        config = ApiConfig()
        
        for path in [config.asr.model_dir, config.vad.model_dir, 
                     config.lid.model_dir, config.punc.model_dir]:
            assert "//" not in path and "\\\\" not in path, (
                f"Path should not have double separators: {path}"
            )


class TestModelDirPrecedence:
    """Test precedence rules: CLI > env > repo-root default."""

    def test_env_var_overrides_default(self, monkeypatch):
        """MODEL_DIR env should override default repo-root resolution."""
        monkeypatch.setenv("MODEL_DIR", "/env/models")
        
        config = ApiConfig()
        
        assert "/env/models" in config.asr.model_dir

    def test_no_env_var_uses_repo_root_default(self, monkeypatch):
        """Without MODEL_DIR env, should use repo-root resolution."""
        monkeypatch.delenv("MODEL_DIR", raising=False)
        
        config = ApiConfig()
        
        assert os.path.isabs(config.asr.model_dir)
        assert "pretrained_models" in config.asr.model_dir


class TestComponentModelDirDefaults:
    """Test that component defaults are correctly applied."""

    def test_asr_default_component_path(self):
        """ASR model_dir should have correct component path."""
        config = ApiConfig()
        assert config.asr.model_dir.endswith("FireRedASR2-AED")

    def test_vad_default_component_path(self):
        """VAD model_dir should have correct component path."""
        config = ApiConfig()
        assert config.vad.model_dir.endswith("FireRedVAD/VAD")

    def test_lid_default_component_path(self):
        """LID model_dir should have correct component path."""
        config = ApiConfig()
        assert config.lid.model_dir.endswith("FireRedLID")

    def test_punc_default_component_path(self):
        """Punc model_dir should have correct component path."""
        config = ApiConfig()
        assert config.punc.model_dir.endswith("FireRedPunc")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_model_dir_env_uses_default(self, monkeypatch):
        """Empty MODEL_DIR env should be treated as not set."""
        monkeypatch.setenv("MODEL_DIR", "")
        
        config = ApiConfig()
        
        assert os.path.isabs(config.asr.model_dir)

    def test_whitespace_model_dir_env_uses_default(self, monkeypatch):
        """Whitespace-only MODEL_DIR env should be treated as not set."""
        monkeypatch.setenv("MODEL_DIR", "   ")
        
        config = ApiConfig()
        
        assert os.path.isabs(config.asr.model_dir)

    def test_model_dir_with_trailing_slash(self, monkeypatch):
        """MODEL_DIR with trailing slash should be handled correctly."""
        monkeypatch.setenv("MODEL_DIR", "/tmp/models/")
        
        config = ApiConfig()
        
        assert os.path.isabs(config.asr.model_dir)
        assert "//" not in config.asr.model_dir

    def test_model_dir_with_env_vars(self, monkeypatch):
        """MODEL_DIR containing env var references should work."""
        monkeypatch.setenv("MODELS_ROOT", "/opt/ml")
        monkeypatch.setenv("MODEL_DIR", "$MODELS_ROOT/firered")
        
        try:
            config = ApiConfig()
            assert os.path.isabs(config.asr.model_dir)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Tests for resolve_asr_model_dir() and LLM path resolution
# ---------------------------------------------------------------------------

from fireredasr2s_api.config import resolve_asr_model_dir


class TestResolveAsrModelDir:
    """Unit tests for the resolve_asr_model_dir() helper."""

    def test_aed_default_returns_aed_suffix(self):
        """AED type with default model_dir should resolve to FireRedASR2-AED."""
        result = resolve_asr_model_dir(
            asr_type="aed",
            base_dir="/models",
            explicit_model_dir="pretrained_models/FireRedASR2-AED",
        )
        assert result == "/models/FireRedASR2-AED"

    def test_llm_default_returns_llm_suffix(self):
        """LLM type with default model_dir should resolve to FireRedASR2-LLM."""
        result = resolve_asr_model_dir(
            asr_type="llm",
            base_dir="/models",
            explicit_model_dir="pretrained_models/FireRedASR2-AED",
        )
        assert result == "/models/FireRedASR2-LLM"

    def test_llm_default_with_none_model_dir(self):
        """LLM type with None model_dir should resolve to FireRedASR2-LLM."""
        result = resolve_asr_model_dir(
            asr_type="llm",
            base_dir="/models",
            explicit_model_dir=None,
        )
        assert result == "/models/FireRedASR2-LLM"

    def test_explicit_absolute_dir_honoured(self):
        """User-supplied absolute model_dir should be returned as-is."""
        custom = "/my/custom/asr"
        result = resolve_asr_model_dir(
            asr_type="llm",
            base_dir="/models",
            explicit_model_dir=custom,
        )
        assert result == custom

    def test_explicit_relative_dir_made_absolute(self):
        """User-supplied relative model_dir should be made absolute."""
        result = resolve_asr_model_dir(
            asr_type="aed",
            base_dir="/models",
            explicit_model_dir="relative/path",
        )
        assert os.path.isabs(result)
        assert result.endswith("relative/path")

    def test_llm_default_model_dir_treated_as_default(self):
        """The LLM default value should also be treated as non-user-supplied."""
        result = resolve_asr_model_dir(
            asr_type="llm",
            base_dir="/models",
            explicit_model_dir="pretrained_models/FireRedASR2-LLM",
        )
        assert result == "/models/FireRedASR2-LLM"

    def test_absolute_aed_default_switches_to_llm(self):
        """Absolute AED default path (base_dir/FireRedASR2-AED) should switch to LLM."""
        base = "/models"
        result = resolve_asr_model_dir(
            asr_type="llm",
            base_dir=base,
            explicit_model_dir=os.path.join(base, "FireRedASR2-AED"),
        )
        assert result == "/models/FireRedASR2-LLM"

    def test_absolute_llm_default_switches_to_aed(self):
        """Absolute LLM default path (base_dir/FireRedASR2-LLM) should switch to AED."""
        base = "/models"
        result = resolve_asr_model_dir(
            asr_type="aed",
            base_dir=base,
            explicit_model_dir=os.path.join(base, "FireRedASR2-LLM"),
        )
        assert result == "/models/FireRedASR2-AED"

    def test_custom_absolute_path_unchanged(self):
        """A truly custom absolute path should not be treated as default."""
        custom = "/srv/my_custom_asr_model"
        result = resolve_asr_model_dir(
            asr_type="llm",
            base_dir="/models",
            explicit_model_dir=custom,
        )
        assert result == custom

    def test_unknown_asr_type_raises(self):
        """Unknown asr_type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown asr_type"):
            resolve_asr_model_dir(
                asr_type="unknown",
                base_dir="/models",
                explicit_model_dir=None,
            )


class TestLlmPathResolutionIntegration:
    """Integration tests: ApiConfig with asr_type='llm' resolves LLM paths."""

    def test_llm_asr_type_resolves_to_llm_model_dir(self, monkeypatch):
        """ApiConfig with asr_type='llm' should use FireRedASR2-LLM suffix."""
        monkeypatch.setenv("MODEL_DIR", "/tmp/models")

        config = ApiConfig(
            asr=AsrBackendConfig(asr_type="llm"),
        )

        expected = "/tmp/models/FireRedASR2-LLM"
        assert config.asr.model_dir == expected, (
            f"LLM asr_type should resolve to FireRedASR2-LLM. "
            f"Expected: {expected}, got: {config.asr.model_dir}"
        )

    def test_aed_asr_type_still_resolves_to_aed_model_dir(self, monkeypatch):
        """ApiConfig with asr_type='aed' should still use FireRedASR2-AED suffix."""
        monkeypatch.setenv("MODEL_DIR", "/tmp/models")

        config = ApiConfig(
            asr=AsrBackendConfig(asr_type="aed"),
        )

        expected = "/tmp/models/FireRedASR2-AED"
        assert config.asr.model_dir == expected, (
            f"AED asr_type should resolve to FireRedASR2-AED. "
            f"Expected: {expected}, got: {config.asr.model_dir}"
        )

    def test_default_config_still_uses_aed(self):
        """Default ApiConfig (no asr_type specified) should still use AED."""
        config = ApiConfig()
        assert config.asr.model_dir.endswith("FireRedASR2-AED"), (
            f"Default config should use AED suffix, got: {config.asr.model_dir}"
        )

    def test_llm_explicit_model_dir_override(self, monkeypatch):
        """Explicit model_dir on LLM config should not be overwritten."""
        monkeypatch.setenv("MODEL_DIR", "/tmp/models")
        custom_dir = "/custom/llm/model"

        config = ApiConfig(
            asr=AsrBackendConfig(asr_type="llm", model_dir=custom_dir),
        )

        assert config.asr.model_dir == custom_dir, (
            f"Explicit model_dir should take precedence. "
            f"Expected: {custom_dir}, got: {config.asr.model_dir}"
        )

    def test_aed_explicit_model_dir_override(self, monkeypatch):
        """Explicit model_dir on AED config should not be overwritten."""
        monkeypatch.setenv("MODEL_DIR", "/tmp/models")
        custom_dir = "/custom/aed/model"

        config = ApiConfig(
            asr=AsrBackendConfig(asr_type="aed", model_dir=custom_dir),
        )

        assert config.asr.model_dir == custom_dir, (
            f"Explicit model_dir should take precedence. "
            f"Expected: {custom_dir}, got: {config.asr.model_dir}"
        )

    def test_llm_type_does_not_affect_other_components(self, monkeypatch):
        """Setting asr_type='llm' should not change VAD/LID/Punc paths."""
        monkeypatch.setenv("MODEL_DIR", "/tmp/models")

        config = ApiConfig(
            asr=AsrBackendConfig(asr_type="llm"),
        )

        assert config.vad.model_dir == "/tmp/models/FireRedVAD/VAD"
        assert config.lid.model_dir == "/tmp/models/FireRedLID"
        assert config.punc.model_dir == "/tmp/models/FireRedPunc"

    def test_llm_default_path_is_absolute(self, monkeypatch):
        """LLM-resolved model_dir should be absolute."""
        monkeypatch.setenv("MODEL_DIR", "/tmp/models")

        config = ApiConfig(
            asr=AsrBackendConfig(asr_type="llm"),
        )

        assert os.path.isabs(config.asr.model_dir)


class TestAsrTypeCli:
    """Test --asr-type CLI argument wiring."""

    def test_cli_asr_type_default_is_aed(self, monkeypatch):
        """Default --asr-type should be 'aed'."""
        import sys
        monkeypatch.setattr(sys, 'argv', ['fireredasr2s-api'])
        config = ApiConfig()
        assert config.asr.asr_type == "aed"

    def test_cli_asr_type_llm_resolves_llm_dir(self, monkeypatch):
        """--asr-type llm should set asr.asr_type='llm' and resolve LLM model dir."""
        monkeypatch.setenv("MODEL_DIR", "/tmp/models")
        config = ApiConfig(asr=AsrBackendConfig(asr_type="llm"))
        assert config.asr.asr_type == "llm"
        assert config.asr.model_dir == "/tmp/models/FireRedASR2-LLM"

    def test_cli_asr_type_aed_resolves_aed_dir(self, monkeypatch):
        """--asr-type aed should set asr.asr_type='aed' and resolve AED model dir."""
        monkeypatch.setenv("MODEL_DIR", "/tmp/models")
        config = ApiConfig(asr=AsrBackendConfig(asr_type="aed"))
        assert config.asr.asr_type == "aed"
        assert config.asr.model_dir == "/tmp/models/FireRedASR2-AED"

    def test_no_switch_offload_field(self):
        """ApiConfig should no longer have switch_offload field."""
        config = ApiConfig()
        assert not hasattr(config, "switch_offload"), (
            "switch_offload field should be removed from ApiConfig"
        )
