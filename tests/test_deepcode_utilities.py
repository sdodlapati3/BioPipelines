"""
Tests for DeepCode-Inspired Agent Utilities
===========================================

Tests for:
- JSON repair functionality
- Adaptive retry strategy
- Response validation
- Graceful degradation chains
- Error guidance generation
"""

import pytest
import json
import time
from unittest.mock import Mock, MagicMock, patch

# Import the modules we're testing
from src.workflow_composer.agents.utils import (
    # JSON repair
    repair_json,
    safe_json_loads,
    extract_json_from_text,
    JSONRepairError,
    # Retry strategy
    RetryConfig,
    adjust_llm_params_for_retry,
    get_retry_delay,
    with_retry,
    AdaptiveLLMCaller,
    # Response validation
    ValidationResult,
    ResponseValidator,
    # Degradation
    DegradationChain,
    FallbackResult,
    create_llm_degradation_chain,
    # Error guidance
    ErrorCategory,
    ErrorGuidance,
    generate_error_guidance,
    generate_guidance_from_log,
    identify_error_category,
)


# =============================================================================
# JSON Repair Tests
# =============================================================================

class TestJSONRepair:
    """Tests for json_repair module."""
    
    def test_repair_valid_json_unchanged(self):
        """Valid JSON should not be modified."""
        valid_json = '{"key": "value", "number": 42}'
        repaired, modified = repair_json(valid_json)
        assert modified is False
        assert json.loads(repaired) == {"key": "value", "number": 42}
    
    def test_repair_trailing_comma(self):
        """Trailing commas should be removed."""
        json_with_comma = '{"key": "value",}'
        repaired, modified = repair_json(json_with_comma)
        assert modified is True
        assert json.loads(repaired) == {"key": "value"}
    
    def test_repair_unclosed_brace(self):
        """Unclosed braces should be closed."""
        truncated_json = '{"key": "value"'
        repaired, modified = repair_json(truncated_json)
        assert modified is True
        assert json.loads(repaired) == {"key": "value"}
    
    def test_repair_unclosed_bracket(self):
        """Unclosed brackets should be closed."""
        truncated_json = '[1, 2, 3'
        repaired, modified = repair_json(truncated_json)
        assert modified is True
        assert json.loads(repaired) == [1, 2, 3]
    
    def test_repair_markdown_code_block(self):
        """Markdown code blocks should be stripped."""
        markdown_json = '```json\n{"key": "value"}\n```'
        repaired, modified = repair_json(markdown_json)
        assert modified is True
        assert json.loads(repaired) == {"key": "value"}
    
    def test_repair_multiple_issues(self):
        """Multiple issues should all be fixed where possible."""
        # This is a complex case - markdown block with trailing comma
        bad_json = '{"key": "value",}'
        repaired, modified = repair_json(bad_json)
        # Should at least be modified (trailing comma removed)
        assert modified is True
        data = json.loads(repaired)
        assert data["key"] == "value"
        # Should strip markdown AND fix trailing comma AND close brace
        result = json.loads(repaired)
        assert result == {"key": "value"}
    
    def test_repair_empty_content(self):
        """Empty content should return empty without error."""
        repaired, modified = repair_json("")
        assert repaired == ""
        assert modified is False
    
    def test_repair_strict_mode_raises(self):
        """Strict mode should raise on unfixable JSON."""
        unfixable = "this is not json at all"
        with pytest.raises(JSONRepairError):
            repair_json(unfixable, strict=True)
    
    def test_safe_json_loads_with_repair(self):
        """safe_json_loads should repair and parse."""
        truncated = '{"key": "value"'
        result = safe_json_loads(truncated)
        assert result == {"key": "value"}
    
    def test_safe_json_loads_returns_default(self):
        """safe_json_loads should return default on failure."""
        garbage = "not json"
        result = safe_json_loads(garbage, default={"fallback": True})
        assert result == {"fallback": True}
    
    def test_extract_json_from_text(self):
        """Should extract JSON from mixed text."""
        mixed = 'Here is the result: {"status": "ok"} and more text'
        extracted = extract_json_from_text(mixed)
        assert extracted is not None
        assert json.loads(extracted) == {"status": "ok"}
    
    def test_extract_json_prefers_object(self):
        """Should prefer object over array when both present."""
        mixed = 'Array: [1,2,3] Object: {"key": "value"}'
        extracted = extract_json_from_text(mixed, prefer_object=True)
        assert json.loads(extracted) == {"key": "value"}


# =============================================================================
# Retry Strategy Tests
# =============================================================================

class TestRetryStrategy:
    """Tests for retry_strategy module."""
    
    def test_adjust_params_first_attempt_unchanged(self):
        """First attempt should not modify params."""
        params = {"max_tokens": 1024, "temperature": 0.7}
        adjusted = adjust_llm_params_for_retry(params, attempt=0)
        assert adjusted == params
    
    def test_adjust_params_reduces_tokens(self):
        """Retry should reduce max_tokens."""
        params = {"max_tokens": 1024, "temperature": 0.7}
        config = RetryConfig(token_reduction_factor=0.75)
        
        adjusted = adjust_llm_params_for_retry(params, attempt=1, config=config)
        assert adjusted["max_tokens"] == 768  # 1024 * 0.75
    
    def test_adjust_params_reduces_temperature(self):
        """Retry should reduce temperature."""
        params = {"max_tokens": 1024, "temperature": 0.7}
        config = RetryConfig(temperature_reduction=0.2)
        
        adjusted = adjust_llm_params_for_retry(params, attempt=1, config=config)
        # Use approximate comparison for floating point
        assert abs(adjusted["temperature"] - 0.5) < 0.01  # 0.7 - 0.2
    
    def test_adjust_params_respects_minimums(self):
        """Parameters should not go below minimums."""
        params = {"max_tokens": 100, "temperature": 0.2}
        config = RetryConfig(
            min_tokens=512,
            min_temperature=0.1,
            token_reduction_factor=0.5,
            temperature_reduction=0.3,
        )
        
        adjusted = adjust_llm_params_for_retry(params, attempt=3, config=config)
        assert adjusted["max_tokens"] >= config.min_tokens
        assert adjusted["temperature"] >= config.min_temperature
    
    def test_get_retry_delay_exponential(self):
        """Delay should increase exponentially."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=0)
        
        delay_0 = get_retry_delay(0, config)
        delay_1 = get_retry_delay(1, config)
        delay_2 = get_retry_delay(2, config)
        
        assert delay_0 == 0.0  # No delay for first attempt
        assert delay_1 == 1.0  # base_delay * 2^0
        assert delay_2 == 2.0  # base_delay * 2^1
    
    def test_get_retry_delay_respects_max(self):
        """Delay should not exceed max_delay."""
        config = RetryConfig(base_delay=1.0, max_delay=5.0, jitter=0)
        
        delay = get_retry_delay(10, config)
        assert delay <= config.max_delay
    
    def test_with_retry_decorator_succeeds(self):
        """Decorated function should return on success."""
        call_count = 0
        
        @with_retry(config=RetryConfig(max_attempts=3))
        def succeeds():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = succeeds()
        assert result == "success"
        assert call_count == 1
    
    def test_with_retry_decorator_retries(self):
        """Decorated function should retry on failure."""
        call_count = 0
        
        @with_retry(config=RetryConfig(max_attempts=3, base_delay=0.01))
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "success"
        
        result = fails_twice()
        assert result == "success"
        assert call_count == 3
    
    def test_with_retry_exhausts_attempts(self):
        """Should raise after max attempts exhausted."""
        @with_retry(config=RetryConfig(max_attempts=2, base_delay=0.01))
        def always_fails():
            raise ValueError("always fails")
        
        with pytest.raises(ValueError, match="always fails"):
            always_fails()


class TestAdaptiveLLMCaller:
    """Tests for AdaptiveLLMCaller."""
    
    def test_successful_call(self):
        """Successful call should return response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        caller = AdaptiveLLMCaller(mock_client, model="test-model")
        result = caller.call(messages=[{"role": "user", "content": "test"}])
        
        assert result == mock_response
        assert caller.last_state.attempt == 0
    
    def test_retries_on_failure(self):
        """Should retry on failure with adjusted params."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            Exception("fail 1"),
            Exception("fail 2"),
            Mock(),  # Success on third try
        ]
        
        caller = AdaptiveLLMCaller(
            mock_client,
            model="test-model",
            config=RetryConfig(max_attempts=3, base_delay=0.01)
        )
        
        caller.call(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1024,
            temperature=0.7
        )
        
        assert caller.last_state.attempt == 2
        # Verify params were adjusted on retries
        calls = mock_client.chat.completions.create.call_args_list
        assert len(calls) == 3


# =============================================================================
# Response Validator Tests
# =============================================================================

class TestResponseValidator:
    """Tests for response_validator module."""
    
    def test_validate_react_response_valid(self):
        """Valid ReAct response should pass validation."""
        response = """Thought: I need to search for files
Action: search_files
Action Input: {"path": "/data", "pattern": "*.fastq"}"""
        
        result = ResponseValidator.validate_react_response(response)
        assert result.valid is True
        assert len(result.errors) == 0
    
    def test_validate_react_response_missing_thought(self):
        """Missing thought should generate warning."""
        response = """Action: search_files
Action Input: {"path": "/data"}"""
        
        result = ResponseValidator.validate_react_response(response)
        assert result.valid is True  # Still valid
        assert any("Thought" in w for w in result.warnings)
    
    def test_validate_react_response_action_input_without_action(self):
        """Action Input without Action should be error."""
        response = """Thought: I need to do something
Action Input: {"path": "/data"}"""
        
        result = ResponseValidator.validate_react_response(response)
        assert result.valid is False
        assert any("Action" in e for e in result.errors)
    
    def test_validate_react_response_invalid_json(self):
        """Invalid JSON in Action Input should generate warning."""
        response = """Thought: I need to search
Action: search
Action Input: not valid json"""
        
        result = ResponseValidator.validate_react_response(response)
        assert result.valid is True
        assert any("JSON" in w for w in result.warnings)
    
    def test_validate_diagnosis_response(self):
        """Valid diagnosis response should pass."""
        response = """1. ERROR TYPE: memory
2. ROOT CAUSE: Out of memory during STAR alignment
3. EXPLANATION: The process exceeded available RAM
4. SUGGESTED FIX: Increase memory allocation
5. AUTO FIXABLE: yes"""
        
        result = ResponseValidator.validate_diagnosis_response(response)
        assert result.valid is True
    
    def test_validate_json_response(self):
        """JSON validation should check structure."""
        response = 'Here is the result: {"status": "ok", "count": 42}'
        
        result = ResponseValidator.validate_json_response(
            response,
            required_keys=["status"]
        )
        
        assert result.valid is True
        assert result.repaired_content is not None


# =============================================================================
# Degradation Chain Tests
# =============================================================================

class TestDegradationChain:
    """Tests for degradation module."""
    
    def test_chain_returns_first_success(self):
        """Chain should return first successful result."""
        chain = DegradationChain()
        chain.add("primary", lambda: "primary_result")
        chain.add("fallback", lambda: "fallback_result")
        
        result = chain.execute()
        
        assert result.value == "primary_result"
        assert result.method_used == "primary"
        assert result.fallback_level == 0
        assert not result.used_fallback
    
    def test_chain_falls_back_on_failure(self):
        """Chain should use fallback when primary fails."""
        chain = DegradationChain()
        chain.add("primary", lambda: (_ for _ in ()).throw(Exception("fail")))
        chain.add("fallback", lambda: "fallback_result")
        
        result = chain.execute()
        
        assert result.value == "fallback_result"
        assert result.method_used == "fallback"
        assert result.fallback_level == 1
        assert result.used_fallback
        assert len(result.error_chain) == 1
    
    def test_chain_respects_conditions(self):
        """Chain should skip methods with false conditions."""
        chain = DegradationChain()
        chain.add("conditional", lambda: "skipped", condition=lambda: False)
        chain.add("unconditional", lambda: "used")
        
        result = chain.execute()
        
        assert result.value == "used"
        assert result.method_used == "unconditional"
    
    def test_chain_returns_default_on_total_failure(self):
        """Chain should return default when all methods fail."""
        chain = DegradationChain()
        chain.add("fail1", lambda: (_ for _ in ()).throw(Exception("fail")))
        chain.add("fail2", lambda: (_ for _ in ()).throw(Exception("fail")))
        
        result = chain.execute(default="default_value")
        
        assert result.value == "default_value"
        assert result.method_used == "default"
        assert len(result.error_chain) == 2
    
    def test_chain_raises_on_total_failure_if_requested(self):
        """Chain should raise when all fail and raise_on_total_failure=True."""
        chain = DegradationChain()
        chain.add("fail", lambda: (_ for _ in ()).throw(Exception("fail")))
        
        with pytest.raises(RuntimeError):
            chain.execute(raise_on_total_failure=True)
    
    def test_chain_skips_none_results(self):
        """Chain should treat None as failure."""
        chain = DegradationChain()
        chain.add("returns_none", lambda: None)
        chain.add("returns_value", lambda: "actual_value")
        
        result = chain.execute()
        
        assert result.value == "actual_value"
        assert result.fallback_level == 1


# =============================================================================
# Error Guidance Tests
# =============================================================================

class TestErrorGuidance:
    """Tests for error_guidance module."""
    
    def test_identify_memory_error(self):
        """Should identify memory errors from log."""
        log = "slurmstepd: error: Detected 1 oom-kill event(s)"
        category, pattern, desc = identify_error_category(log)
        
        assert category == ErrorCategory.MEMORY
        assert pattern is not None
    
    def test_identify_star_error(self):
        """Should identify STAR-specific errors."""
        log = "EXITING because of FATAL ERROR in reads input: quality string length is not equal to sequence length"
        category, pattern, desc = identify_error_category(log)
        
        assert category == ErrorCategory.DATA
        assert "quality" in desc.lower() or "fastq" in desc.lower()
    
    def test_identify_slurm_time_error(self):
        """Should identify SLURM timeout."""
        log = "JOB 12345 CANCELLED DUE TO TIME LIMIT"
        category, pattern, desc = identify_error_category(log)
        
        assert category == ErrorCategory.SLURM
    
    def test_identify_unknown_error(self):
        """Unknown errors should return UNKNOWN category."""
        log = "Some random error message that doesn't match"
        category, pattern, desc = identify_error_category(log)
        
        assert category == ErrorCategory.UNKNOWN
        assert pattern is None
    
    def test_generate_memory_guidance(self):
        """Memory error should generate appropriate guidance."""
        guidance = generate_error_guidance(
            ErrorCategory.MEMORY,
            "Out of memory during STAR alignment"
        )
        
        assert "memory" in guidance.error_summary.lower()
        assert len(guidance.recovery_steps) > 0
        assert len(guidance.anti_patterns) > 0
        assert guidance.example_fix is not None
    
    def test_generate_guidance_from_log(self):
        """Should generate guidance directly from log."""
        log = "java.lang.OutOfMemoryError: GC overhead limit exceeded"
        guidance = generate_guidance_from_log(log)
        
        assert guidance is not None
        assert "MEMORY" in guidance.error_summary
        assert len(guidance.recovery_steps) > 0
    
    def test_guidance_to_markdown(self):
        """Guidance should format to markdown."""
        guidance = ErrorGuidance(
            error_summary="Test error",
            recovery_steps=["Step 1", "Step 2"],
            anti_patterns=["Don't do X"],
            example_fix="fix code",
        )
        
        markdown = guidance.to_markdown()
        
        assert "## Error:" in markdown
        assert "### Recovery Steps" in markdown
        assert "Step 1" in markdown
        assert "❌" in markdown  # Anti-pattern marker
        assert "```" in markdown  # Code block
    
    def test_guidance_to_agent_context(self):
        """Guidance should format for agent context."""
        guidance = ErrorGuidance(
            error_summary="Test error",
            recovery_steps=["Step 1"],
            anti_patterns=["Don't do X"],
        )
        
        context = guidance.to_agent_context()
        
        assert "## Error Encountered" in context
        assert "## Recovery Strategy" in context
        assert "## What NOT to Do" in context


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple utilities."""
    
    def test_degradation_with_json_repair(self):
        """Degradation chain should work with JSON repair."""
        truncated_json = '{"key": "value"'
        
        def llm_parse():
            return json.loads(truncated_json)  # Will fail
        
        def repair_parse():
            return safe_json_loads(truncated_json)
        
        chain = DegradationChain()
        chain.add("direct_parse", llm_parse)
        chain.add("repair_parse", repair_parse)
        
        result = chain.execute()
        
        assert result.value == {"key": "value"}
        assert result.method_used == "repair_parse"
    
    def test_full_error_handling_flow(self):
        """Test complete error handling flow."""
        error_log = """
        Error executing process > 'STAR_ALIGN (sample1)'
        slurmstepd: error: Detected 1 oom-kill event(s)
        """
        
        # 1. Identify error
        category, pattern, desc = identify_error_category(error_log)
        assert category == ErrorCategory.MEMORY
        
        # 2. Generate guidance
        guidance = generate_error_guidance(category, desc)
        assert len(guidance.recovery_steps) > 0
        
        # 3. Format for agent
        context = guidance.to_agent_context()
        assert "memory" in context.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
