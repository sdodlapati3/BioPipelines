"""
LLM Response Validator
======================

Validates LLM responses before parsing to catch issues early.

Provides validators for:
- ReAct agent responses (Thought/Action/Action Input format)
- Diagnostic responses (ERROR TYPE/ROOT CAUSE/etc. format)
- Generic JSON responses with schema validation

Early validation helps:
1. Detect format issues before parsing fails
2. Provide better error messages
3. Enable repair attempts with context
"""

import re
import json
import logging
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Result of response validation.
    
    Attributes:
        valid: Whether the response meets minimum requirements
        errors: List of critical issues (response won't parse)
        warnings: List of non-critical issues (may affect quality)
        repaired_content: If repair was attempted, the repaired content
        metadata: Additional validation metadata
    """
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    repaired_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.valid
    
    def has_issues(self) -> bool:
        """Check if there are any errors or warnings."""
        return bool(self.errors or self.warnings)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "repaired_content": self.repaired_content,
            "metadata": self.metadata,
        }


class ResponseValidator:
    """
    Validates LLM responses for expected structure.
    
    Provides static methods for validating different response formats.
    Each validator returns a ValidationResult with details.
    """
    
    @staticmethod
    def validate_react_response(response: str) -> ValidationResult:
        """
        Validate ReAct agent response format.
        
        Expected format:
            Thought: <reasoning about what to do>
            Action: <tool name or "finish">
            Action Input: <JSON arguments>
        
        Args:
            response: LLM response text
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        metadata = {}
        
        if not response or not response.strip():
            return ValidationResult(
                valid=False,
                errors=["Empty response"],
            )
        
        # Check for Thought
        thought_match = re.search(r'^Thought:\s*(.+)', response, re.MULTILINE)
        has_thought = bool(thought_match)
        if thought_match:
            metadata["thought"] = thought_match.group(1).strip()[:200]
        
        # Check for Action
        action_match = re.search(r'^Action:\s*(\w+)', response, re.MULTILINE)
        has_action = bool(action_match)
        if action_match:
            metadata["action"] = action_match.group(1).strip()
        
        # Check for Action Input
        input_match = re.search(r'^Action Input:\s*(.+)', response, re.MULTILINE | re.DOTALL)
        has_input = bool(input_match)
        
        # Validation rules
        if not has_thought:
            warnings.append("Missing 'Thought:' prefix - will use full response as thought")
        
        if has_input and not has_action:
            errors.append("Has 'Action Input:' without 'Action:' - cannot execute")
        
        # Validate JSON in action input
        if has_input:
            input_content = input_match.group(1).strip()
            # Get just the first line or until next section
            input_content = input_content.split('\n')[0].strip()
            metadata["action_input_raw"] = input_content[:200]
            
            if input_content:
                if not (input_content.startswith('{') or input_content.startswith('[')):
                    warnings.append(
                        "Action Input doesn't appear to be JSON - "
                        "will attempt to parse as raw value"
                    )
                else:
                    try:
                        parsed = json.loads(input_content)
                        metadata["action_input_valid_json"] = True
                    except json.JSONDecodeError as e:
                        warnings.append(f"Action Input JSON parse error: {e}")
                        metadata["action_input_valid_json"] = False
        
        # Check for common issues
        if 'Observation:' in response:
            warnings.append(
                "Response contains 'Observation:' - "
                "LLM may be simulating tool output instead of waiting"
            )
        
        if response.count('Action:') > 1:
            warnings.append(
                "Multiple 'Action:' found - "
                "will use first one, others will be ignored"
            )
        
        # Determine overall validity
        # Valid if we can extract at least a thought OR action
        valid = has_thought or has_action
        if errors:
            valid = False
        
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )
    
    @staticmethod
    def validate_diagnosis_response(response: str) -> ValidationResult:
        """
        Validate coding agent diagnosis response format.
        
        Expected format:
            1. ERROR TYPE: <type>
            2. ROOT CAUSE: <cause>
            3. EXPLANATION: <explanation>
            4. SUGGESTED FIX: <fix>
            5. AUTO FIXABLE: <yes/no>
        
        Args:
            response: LLM response text
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        metadata = {}
        
        if not response or not response.strip():
            return ValidationResult(
                valid=False,
                errors=["Empty response"],
            )
        
        response_lower = response.lower()
        
        # Check for required fields
        required_fields = {
            "error type": False,
            "root cause": False,
            "explanation": False,
        }
        
        optional_fields = {
            "suggested fix": False,
            "auto fixable": False,
        }
        
        for field in required_fields:
            if f"{field}:" in response_lower:
                required_fields[field] = True
                # Extract value
                pattern = rf'{field}:\s*(.+?)(?=\n\d+\.|$)'
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    metadata[field.replace(" ", "_")] = match.group(1).strip()[:200]
        
        for field in optional_fields:
            if f"{field}:" in response_lower:
                optional_fields[field] = True
        
        # Report missing required fields
        missing_required = [f for f, found in required_fields.items() if not found]
        if missing_required:
            warnings.append(f"Missing expected fields: {', '.join(missing_required)}")
        
        # Check for valid error type
        valid_types = [
            "memory", "disk", "permission", "nextflow", "slurm",
            "tool", "network", "syntax", "unknown", "snakemake"
        ]
        if required_fields["error type"]:
            error_type_val = metadata.get("error_type", "").lower()
            if not any(t in error_type_val for t in valid_types):
                warnings.append(
                    f"Error type '{error_type_val}' not in known types: {valid_types}"
                )
        
        # Check for actionable content
        if not optional_fields["suggested fix"]:
            warnings.append("No suggested fix provided")
        
        # Diagnosis can work with partial info, so we're lenient
        valid = len(errors) == 0
        
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )
    
    @staticmethod
    def validate_json_response(
        response: str,
        required_keys: Optional[List[str]] = None,
        expected_types: Optional[Dict[str, type]] = None,
    ) -> ValidationResult:
        """
        Validate JSON response structure.
        
        Args:
            response: LLM response text (may contain JSON)
            required_keys: Keys that must be present in JSON
            expected_types: Expected types for specific keys
            
        Returns:
            ValidationResult with validation details
        """
        from .json_repair import extract_json_from_text, safe_json_loads
        
        errors = []
        warnings = []
        metadata = {}
        repaired = None
        
        if not response or not response.strip():
            return ValidationResult(
                valid=False,
                errors=["Empty response"],
            )
        
        # Try to find JSON
        json_str = extract_json_from_text(response)
        if not json_str:
            return ValidationResult(
                valid=False,
                errors=["No JSON found in response"],
                metadata={"response_preview": response[:200]},
            )
        
        # Try to parse
        data = safe_json_loads(json_str, default=None)
        if data is None:
            return ValidationResult(
                valid=False,
                errors=["Failed to parse JSON even after repair"],
                metadata={"json_preview": json_str[:200]},
            )
        
        repaired = json_str
        metadata["json_type"] = type(data).__name__
        
        # Check required keys
        if required_keys and isinstance(data, dict):
            missing = [k for k in required_keys if k not in data]
            if missing:
                warnings.append(f"Missing required keys: {missing}")
            metadata["found_keys"] = list(data.keys())
        
        # Check expected types
        if expected_types and isinstance(data, dict):
            for key, expected_type in expected_types.items():
                if key in data:
                    actual_type = type(data[key])
                    if not isinstance(data[key], expected_type):
                        warnings.append(
                            f"Key '{key}' has type {actual_type.__name__}, "
                            f"expected {expected_type.__name__}"
                        )
        
        return ValidationResult(
            valid=True,
            errors=errors,
            warnings=warnings,
            repaired_content=repaired,
            metadata=metadata,
        )
    
    @staticmethod
    def validate_workflow_config(response: str) -> ValidationResult:
        """
        Validate workflow configuration response.
        
        Checks for common Nextflow/Snakemake configuration issues.
        """
        errors = []
        warnings = []
        metadata = {}
        
        if not response or not response.strip():
            return ValidationResult(
                valid=False,
                errors=["Empty response"],
            )
        
        # Check for code blocks
        has_nextflow = bool(re.search(r'```(?:nextflow|groovy)', response, re.IGNORECASE))
        has_snakemake = bool(re.search(r'```(?:snakemake|python)', response, re.IGNORECASE))
        has_config = bool(re.search(r'```(?:config|yaml)', response, re.IGNORECASE))
        
        metadata["has_nextflow"] = has_nextflow
        metadata["has_snakemake"] = has_snakemake
        metadata["has_config"] = has_config
        
        if not any([has_nextflow, has_snakemake, has_config]):
            warnings.append("No code blocks found with workflow language markers")
        
        # Check for common issues in code blocks
        code_pattern = r'```\w*\n(.*?)```'
        code_blocks = re.findall(code_pattern, response, re.DOTALL)
        
        for i, code in enumerate(code_blocks):
            # Check brace balance
            if code.count('{') != code.count('}'):
                warnings.append(f"Code block {i+1}: Unbalanced braces")
            
            # Check for hardcoded paths
            if re.search(r'["\']\/home\/|["\']\/Users\/', code):
                warnings.append(f"Code block {i+1}: Contains hardcoded home directory path")
            
            # Check for placeholder values
            if re.search(r'TODO|FIXME|XXX|\<.*?\>', code):
                warnings.append(f"Code block {i+1}: Contains placeholder values")
        
        metadata["code_block_count"] = len(code_blocks)
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )


def validate_and_repair(
    response: str,
    validator_type: str,
    **kwargs,
) -> ValidationResult:
    """
    Convenience function to validate and optionally repair response.
    
    Args:
        response: LLM response text
        validator_type: One of "react", "diagnosis", "json", "workflow"
        **kwargs: Additional arguments for specific validators
        
    Returns:
        ValidationResult
    """
    validators = {
        "react": ResponseValidator.validate_react_response,
        "diagnosis": ResponseValidator.validate_diagnosis_response,
        "json": ResponseValidator.validate_json_response,
        "workflow": ResponseValidator.validate_workflow_config,
    }
    
    validator = validators.get(validator_type)
    if not validator:
        raise ValueError(f"Unknown validator type: {validator_type}")
    
    if validator_type == "json":
        return validator(response, **kwargs)
    else:
        return validator(response)
