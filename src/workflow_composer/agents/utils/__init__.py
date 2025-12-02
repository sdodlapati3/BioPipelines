"""
Agent Utilities
===============

Utility modules for BioPipelines agents:
- json_repair: Fix malformed JSON from LLM outputs
- retry_strategy: Adaptive retry with parameter reduction
- response_validator: Validate LLM response structure
- degradation: Graceful fallback chains
- error_guidance: Generate actionable error recovery guidance

Inspired by patterns from DeepCode (HKUDS/DeepCode).
"""

from .json_repair import (
    repair_json,
    safe_json_loads,
    extract_json_from_text,
    JSONRepairError,
)

from .retry_strategy import (
    RetryConfig,
    RetryState,
    adjust_llm_params_for_retry,
    get_retry_delay,
    with_retry,
    AdaptiveLLMCaller,
    DIAGNOSTIC_RETRY_CONFIG,
    CODE_GENERATION_RETRY_CONFIG,
    VALIDATION_RETRY_CONFIG,
)

from .response_validator import (
    ValidationResult,
    ResponseValidator,
)

from .degradation import (
    FallbackResult,
    DegradationChain,
    create_llm_degradation_chain,
)

from .error_guidance import (
    ErrorCategory,
    ErrorGuidance,
    generate_error_guidance,
    generate_guidance_from_log,
    identify_error_category,
    BIOINFORMATICS_ERROR_PATTERNS,
)

__all__ = [
    # JSON repair
    "repair_json",
    "safe_json_loads",
    "extract_json_from_text",
    "JSONRepairError",
    # Retry strategy
    "RetryConfig",
    "RetryState",
    "adjust_llm_params_for_retry",
    "get_retry_delay",
    "with_retry",
    "AdaptiveLLMCaller",
    "DIAGNOSTIC_RETRY_CONFIG",
    "CODE_GENERATION_RETRY_CONFIG",
    "VALIDATION_RETRY_CONFIG",
    # Response validation
    "ValidationResult",
    "ResponseValidator",
    # Degradation
    "FallbackResult",
    "DegradationChain",
    "create_llm_degradation_chain",
    # Error guidance
    "ErrorCategory",
    "ErrorGuidance",
    "generate_error_guidance",
    "generate_guidance_from_log",
    "identify_error_category",
    "BIOINFORMATICS_ERROR_PATTERNS",
]

__all__ = [
    # JSON repair
    "repair_json",
    "safe_json_loads",
    "extract_json_from_text",
    "JSONRepairError",
    # Retry strategy
    "RetryConfig",
    "RetryState",
    "adjust_llm_params_for_retry",
    "get_retry_delay",
    "with_retry",
    "AdaptiveLLMCaller",
    # Response validation
    "ValidationResult",
    "ResponseValidator",
    # Degradation
    "FallbackResult",
    "DegradationChain",
]
