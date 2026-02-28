# Copyright 2026 FireRedTeam

"""Configuration classes for FireRedASR2S API."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Map from component name to (default_model_dir_value, component_suffix)
_COMPONENT_DEFAULTS: Dict[str, Tuple[str, str]] = {
    "asr": ("models/FireRedASR2-AED", "FireRedASR2-AED"),
    "vad": ("models/FireRedVAD/Stream-VAD", "FireRedVAD/Stream-VAD"),
    "lid": ("models/FireRedLID", "FireRedLID"),
    "punc": ("models/FireRedPunc", "FireRedPunc"),
}

# Map from asr_type to the default model directory suffix
_ASR_TYPE_SUFFIXES: Dict[str, str] = {
    "aed": "FireRedASR2-AED",
    "llm": "FireRedASR2-LLM",
}

# Map from vad_type to the model directory suffix
_VAD_TYPE_MAP: dict[str, str] = {
    "vad": "FireRedVAD/VAD",
    "stream-vad": "FireRedVAD/Stream-VAD",
    "aed": "FireRedVAD/AED",
}

# Set of known ASR default model_dir values (any of these means "not user-supplied")
_ASR_DEFAULT_MODEL_DIRS = {
    f"models/{suffix}" for suffix in _ASR_TYPE_SUFFIXES.values()
}


def _is_default_asr_path(path: str, base_dir: str) -> bool:
    """Return True if *path* is a known default ASR model directory.

    A path is considered default if it matches one of the relative defaults
    in ``_ASR_DEFAULT_MODEL_DIRS`` **or** equals ``os.path.join(base_dir, suffix)``
    for any known ASR type suffix.
    """
    if path in _ASR_DEFAULT_MODEL_DIRS:
        return True
    for suffix in _ASR_TYPE_SUFFIXES.values():
        if path == os.path.join(base_dir, suffix):
            return True
    return False


def resolve_asr_model_dir(
    asr_type: str,
    base_dir: str,
    explicit_model_dir: Optional[str] = None,
) -> str:
    """Resolve the ASR model directory based on asr_type.

    If *explicit_model_dir* is user-supplied (i.e. not one of the dataclass
    defaults), return it as-is.  Otherwise, combine *base_dir* with the
    suffix that matches *asr_type* ('aed' → 'FireRedASR2-AED',
    'llm' → 'FireRedASR2-LLM').

    Args:
        asr_type: 'aed' or 'llm'.
        base_dir: Resolved model base directory (absolute).
        explicit_model_dir: The model_dir value from AsrBackendConfig.
            When it equals one of the built-in defaults (or is ``None``),
            it is treated as "not user-supplied".

    Returns:
        Absolute path to the ASR model directory.
    """
    # If user supplied a custom path, honour it
    if (
        explicit_model_dir is not None
        and not _is_default_asr_path(explicit_model_dir, base_dir)
    ):
        if not os.path.isabs(explicit_model_dir):
            return os.path.abspath(explicit_model_dir)
        return explicit_model_dir

    # Derive from asr_type
    suffix = _ASR_TYPE_SUFFIXES.get(asr_type)
    if suffix is None:
        raise ValueError(
            f"Unknown asr_type '{asr_type}'. Expected one of: {list(_ASR_TYPE_SUFFIXES)}"
        )
    return os.path.join(base_dir, suffix)

def resolve_vad_model_dirs(
    vad_config: "VadConfig",
    base_dir: Optional[str] = None,
) -> dict[str, str]:
    """Resolve VAD model directories based on vad_type.

    If vad_type='all', returns a dict with paths for all modes:
    {'vad': path, 'stream-vad': path, 'aed': path}

    If vad_type is a single mode (e.g., 'stream-vad'), returns a dict with
    just that mode: {'stream-vad': path}

    Args:
        vad_config: VadConfig instance with vad_type and model_dir.
        base_dir: Model base directory. If None, uses parent of vad_config.model_dir.

    Returns:
        Dict mapping vad_type(s) to absolute model directory paths.
    """
    if base_dir is None:
        # Derive base_dir from the configured model_dir
        base_dir = os.path.dirname(vad_config.model_dir)

    if vad_config.vad_type == "all":
        # Return all three VAD modes
        return {
            mode: os.path.join(base_dir, suffix)
            for mode, suffix in _VAD_TYPE_MAP.items()
        }
    else:
        # Return single mode
        suffix = _VAD_TYPE_MAP.get(vad_config.vad_type)
        if suffix is None:
            raise ValueError(
                f"Unknown vad_type '{vad_config.vad_type}'. "
                f"Expected 'all' or one of: {list(_VAD_TYPE_MAP.keys())}"
            )
        return {vad_config.vad_type: os.path.join(base_dir, suffix)}

def _resolve_repo_root() -> Optional[str]:
    """Walk up from CWD looking for a directory containing models/.

    Returns the repo root path as a string, or None if not found.
    """
    current = Path.cwd().resolve()
    while True:
        if (current / "models").is_dir():
            return str(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _get_model_base_dir(model_base_dir: Optional[str] = None) -> str:
    """Resolve model base directory.

    Precedence: explicit arg (CLI) > MODEL_DIR env > repo-root default > CWD fallback.
    Returns an absolute path to the model base directory.
    """
    # 1. Explicit argument (from CLI --model-dir or constructor)
    if model_base_dir is not None and model_base_dir.strip():
        resolved = model_base_dir.strip()
        if not os.path.isabs(resolved):
            resolved = os.path.join(os.getcwd(), resolved)
        return resolved

    # 2. MODEL_DIR environment variable
    env_dir = os.environ.get("MODEL_DIR", "").strip()
    if env_dir:
        env_dir = os.path.expandvars(env_dir)
        if not os.path.isabs(env_dir):
            env_dir = os.path.join(os.getcwd(), env_dir)
        return env_dir

    # 3. Repo root detection
    repo_root = _resolve_repo_root()
    if repo_root is not None:
        return os.path.join(repo_root, "models")

    # 4. Fallback to CWD
    logger.warning(
        "Could not find repo root with models/ directory. "
        "Falling back to CWD for model paths."
    )
    return os.path.join(os.getcwd(), "models")


@dataclass
class SessionConfig:
    """Configuration for streaming session behavior."""

    # Audio configuration
    sample_rate: int = 16000
    chunk_duration_ms: int = 100  # 100ms chunks

    # VAD configuration for streaming
    silence_timeout_ms: int = 500  # 500ms silence to trigger finalization
    max_segment_duration_ms: int = 60000  # 60s max for AED, 30s for LLM

    # Partial result configuration
    enable_partial_results: bool = True
    partial_result_interval_ms: int = 200  # Minimum interval between partials

    # Session timeout
    session_timeout_seconds: int = 300  # 5 minutes max session

    def __post_init__(self) -> None:
        if self.sample_rate != 16000:
            raise ValueError("Only 16kHz sample rate is supported")
        if self.chunk_duration_ms < 20 or self.chunk_duration_ms > 500:
            raise ValueError("Chunk duration must be between 20ms and 500ms")


@dataclass
class AsrBackendConfig:
    """Configuration for ASR backend (AED or LLM)."""

    # Model type and paths
    asr_type: str = "aed"  # "aed" or "llm"
    model_dir: str = "models/FireRedASR2-AED"

    # Inference configuration
    use_gpu: bool = True
    use_half: bool = False
    beam_size: int = 3
    nbest: int = 1

    # AED specific
    decode_max_len: int = 0
    softmax_smoothing: float = 1.25
    aed_length_penalty: float = 0.6
    eos_penalty: float = 1.0
    return_timestamp: bool = True

    # LLM specific
    decode_min_len: int = 0
    repetition_penalty: float = 1.0
    llm_length_penalty: float = 0.0
    temperature: float = 1.0

    # External LM
    elm_dir: str = ""
    elm_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.asr_type not in ["aed", "llm"]:
            raise ValueError("asr_type must be 'aed' or 'llm'")


@dataclass
class VadConfig:
    """Configuration for VAD module."""

    model_dir: str = "models/FireRedVAD/Stream-VAD"
    vad_type: str = "all"
    use_gpu: bool = False  # VAD typically runs on CPU

    # Non-streaming VAD parameters
    smooth_window_size: int = 5
    speech_threshold: float = 0.4
    min_speech_frame: int = 20
    max_speech_frame: int = 2000
    min_silence_frame: int = 20
    merge_silence_frame: int = 0
    extend_speech_frame: int = 0
    chunk_max_frame: int = 30000


@dataclass
class LidConfig:
    """Configuration for LID module."""

    model_dir: str = "models/FireRedLID"
    use_gpu: bool = True
    use_half: bool = False


@dataclass
class PuncConfig:
    """Configuration for Punctuation module."""

    model_dir: str = "models/FireRedPunc"
    use_gpu: bool = True
    batch_size: int = 1
    with_timestamp: bool = True
    sentence_max_length: int = -1


@dataclass
class ApiConfig:
    """Main API configuration."""

    # Server configuration
    host: str = "0.0.0.0"
    grpc_port: int = 50051

    # Session configuration
    session: SessionConfig = field(default_factory=SessionConfig)

    # Backend configurations
    asr: AsrBackendConfig = field(default_factory=AsrBackendConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    lid: LidConfig = field(default_factory=LidConfig)
    punc: PuncConfig = field(default_factory=PuncConfig)

    # Feature enablement
    enable_lid: bool = True
    enable_punc: bool = True

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"

    # Model paths (can override individual configs via CLI --model-dir)
    model_base_dir: Optional[str] = None

    def __post_init__(self) -> None:
        # Resolve the model base directory (CLI > env > repo-root > CWD fallback)
        base_dir = _get_model_base_dir(self.model_base_dir)
        self.model_base_dir = base_dir

        # Resolve ASR model_dir via asr_type-aware helper
        self.asr.model_dir = resolve_asr_model_dir(
            asr_type=self.asr.asr_type,
            base_dir=base_dir,
            explicit_model_dir=self.asr.model_dir,
        )

        # Resolve remaining components' model_dir (vad, lid, punc)
        for attr_name, (default_val, suffix) in _COMPONENT_DEFAULTS.items():
            if attr_name == "asr":
                continue  # already handled above
            component = getattr(self, attr_name)
            if component.model_dir == default_val:
                # Default value: resolve against base_dir
                component.model_dir = os.path.join(base_dir, suffix)
            elif not os.path.isabs(component.model_dir):
                # Relative custom path: make absolute against CWD
                component.model_dir = os.path.abspath(component.model_dir)
