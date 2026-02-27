"""Validation helpers for LLM parameters (gRPC API)."""

from typing import Any, Dict, Optional


# Default LLM parameter values (matching AsrBackendConfig)
_LLM_DEFAULTS = {
    "decode_min_len": 0,
    "repetition_penalty": 1.0,
    "llm_length_penalty": 0.0,
    "temperature": 1.0,
}


def validate_llm_params(
    decode_min_len: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    llm_length_penalty: Optional[float] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Validate and apply defaults for LLM parameters.

    Invalid values fall back to defaults.
    Bounds: decode_min_len >= 0, repetition_penalty >= 0,
           llm_length_penalty >= 0, temperature > 0.

    Returns:
        Dict with validated LLM params.
    """
    result: Dict[str, Any] = dict(_LLM_DEFAULTS)

    if decode_min_len is not None:
        try:
            v = int(decode_min_len)
            if v >= 0:
                result["decode_min_len"] = v
        except (TypeError, ValueError):
            pass  # fall back to default

    if repetition_penalty is not None:
        try:
            v = float(repetition_penalty)
            if v >= 0:
                result["repetition_penalty"] = v
        except (TypeError, ValueError):
            pass

    if llm_length_penalty is not None:
        try:
            v = float(llm_length_penalty)
            if v >= 0:
                result["llm_length_penalty"] = v
        except (TypeError, ValueError):
            pass

    if temperature is not None:
        try:
            v = float(temperature)
            if v > 0:
                result["temperature"] = v
        except (TypeError, ValueError):
            pass

    return result
