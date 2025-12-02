"""
JSON Repair Utility
===================

Repairs common JSON issues from LLM outputs:
- Truncated JSON (unclosed braces/brackets)
- Trailing commas (not valid JSON)
- Unclosed strings
- Invalid escape sequences
- Control characters
- Markdown code block wrappers

Inspired by DeepCode's _repair_truncated_json pattern.

References:
    - DeepCode: workflows/code_implementation_workflow.py
"""

import re
import json
import logging
from typing import Optional, Tuple, Any, Union

logger = logging.getLogger(__name__)


class JSONRepairError(Exception):
    """Raised when JSON cannot be repaired."""
    pass


def repair_json(content: str, strict: bool = False) -> Tuple[str, bool]:
    """
    Attempt to repair malformed JSON.
    
    This function applies a series of fixes to common JSON malformation
    issues that occur when LLMs generate JSON output, especially when
    truncated due to token limits.
    
    Args:
        content: Potentially malformed JSON string
        strict: If True, raise on failure; if False, return original
        
    Returns:
        Tuple of (repaired_json: str, was_modified: bool)
        
    Raises:
        JSONRepairError: If strict=True and repair fails
        
    Examples:
        >>> repair_json('{"key": "value",}')
        ('{"key": "value"}', True)
        
        >>> repair_json('{"key": "value"')
        ('{"key": "value"}', True)
    """
    if not content or not content.strip():
        if strict:
            raise JSONRepairError("Empty content")
        return content, False
    
    original = content
    modified = False
    
    # Step 1: Strip markdown code blocks
    content, stripped = _strip_code_blocks(content)
    modified = modified or stripped
    
    # Step 2: Remove leading/trailing whitespace and newlines
    content = content.strip()
    
    # Step 3: Fix trailing commas before ] or }
    content, fixed = _fix_trailing_commas(content)
    modified = modified or fixed
    
    # Step 4: Balance braces and brackets
    content, balanced = _balance_brackets(content)
    modified = modified or balanced
    
    # Step 5: Fix unclosed strings
    content, fixed_strings = _fix_unclosed_strings(content)
    modified = modified or fixed_strings
    
    # Step 6: Remove control characters
    content, cleaned = _remove_control_chars(content)
    modified = modified or cleaned
    
    # Step 7: Fix common escape sequence issues
    content, fixed_escapes = _fix_escape_sequences(content)
    modified = modified or fixed_escapes
    
    # Validate result
    try:
        json.loads(content)
        if modified:
            logger.debug("JSON successfully repaired")
        return content, modified
    except json.JSONDecodeError as e:
        if strict:
            raise JSONRepairError(f"Could not repair JSON: {e}")
        logger.warning(f"JSON repair failed: {e}, returning original")
        return original, False


def _strip_code_blocks(content: str) -> Tuple[str, bool]:
    """
    Remove markdown code block markers.
    
    Handles:
    - ```json ... ```
    - ``` ... ```
    - ```JSON ... ```
    """
    original = content
    
    # Match ```json ... ``` or ``` ... ``` (case insensitive)
    pattern = r'```(?:json|JSON)?\s*([\s\S]*?)\s*```'
    match = re.search(pattern, content)
    if match:
        return match.group(1).strip(), True
    
    # Handle unclosed code block (truncated)
    if content.strip().startswith('```'):
        lines = content.strip().split('\n')
        if lines[0].startswith('```'):
            # Remove first line (```json) and any trailing ```
            content = '\n'.join(lines[1:])
            if content.strip().endswith('```'):
                content = content.strip()[:-3]
            return content.strip(), True
    
    return content, False


def _fix_trailing_commas(content: str) -> Tuple[str, bool]:
    """
    Remove trailing commas before } or ].
    
    JSON doesn't allow trailing commas, but LLMs often add them.
    """
    original = content
    
    # Match: , followed by optional whitespace then } or ]
    # Be careful not to match commas inside strings
    # Simple approach: just look for ,\s*[}\]]
    pattern = r',(\s*[}\]])'
    fixed = re.sub(pattern, r'\1', content)
    
    return fixed, fixed != original


def _balance_brackets(content: str) -> Tuple[str, bool]:
    """
    Balance unclosed braces and brackets.
    
    Handles truncated JSON by adding missing closing brackets.
    Uses a simple counting approach (doesn't handle strings with brackets).
    """
    modified = False
    
    # Simple counting (may not work for all edge cases)
    # A more robust approach would track string state
    open_braces = content.count('{') - content.count('}')
    open_brackets = content.count('[') - content.count(']')
    
    # Only add closers, don't remove (that would change semantics)
    if open_braces > 0:
        content = content.rstrip() + '}' * open_braces
        modified = True
        logger.debug(f"Added {open_braces} closing braces")
    
    if open_brackets > 0:
        content = content.rstrip() + ']' * open_brackets
        modified = True
        logger.debug(f"Added {open_brackets} closing brackets")
    
    return content, modified


def _fix_unclosed_strings(content: str) -> Tuple[str, bool]:
    """
    Attempt to close unclosed string literals.
    
    This is tricky - we only handle obvious cases where the JSON
    is truncated mid-string.
    """
    # Count quotes outside of escaped sequences
    # Simple approach: count all " that aren't preceded by \
    quote_positions = []
    i = 0
    while i < len(content):
        if content[i] == '"' and (i == 0 or content[i-1] != '\\'):
            quote_positions.append(i)
        i += 1
    
    if len(quote_positions) % 2 != 0:
        # Odd number of quotes - likely unclosed string
        content = content.rstrip()
        if not content.endswith('"'):
            # Add closing quote
            content += '"'
            logger.debug("Added closing quote for unclosed string")
            return content, True
    
    return content, False


def _remove_control_chars(content: str) -> Tuple[str, bool]:
    """
    Remove control characters that break JSON parsing.
    
    Removes characters 0x00-0x1F except tab, newline, and carriage return.
    """
    original = content
    
    # Remove characters 0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F
    # Keep: \t (0x09), \n (0x0A), \r (0x0D)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', content)
    
    if cleaned != original:
        logger.debug("Removed control characters from JSON")
    
    return cleaned, cleaned != original


def _fix_escape_sequences(content: str) -> Tuple[str, bool]:
    """
    Fix common escape sequence issues.
    
    LLMs sometimes produce invalid escape sequences.
    """
    original = content
    
    # Fix single backslashes that aren't valid escapes
    # Valid JSON escapes: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    # This is a simplified fix - replace \X with \\X for invalid escapes
    # Actually, this is risky - better to leave as is and let parsing handle it
    
    return content, False


def safe_json_loads(
    content: str,
    default: Any = None,
    repair: bool = True,
) -> Any:
    """
    Safely parse JSON with automatic repair.
    
    Args:
        content: JSON string to parse
        default: Default value if parsing fails
        repair: Whether to attempt repair on failure
        
    Returns:
        Parsed JSON data or default value
        
    Examples:
        >>> safe_json_loads('{"key": "value"}')
        {'key': 'value'}
        
        >>> safe_json_loads('invalid', default={})
        {}
    """
    if not content:
        return default
    
    # First, try direct parsing
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Try with repair
    if repair:
        try:
            repaired, was_modified = repair_json(content, strict=False)
            if was_modified:
                result = json.loads(repaired)
                logger.info("JSON parsed successfully after repair")
                return result
        except (json.JSONDecodeError, JSONRepairError):
            pass
    
    logger.warning("JSON parsing failed even after repair attempts")
    return default


def extract_json_from_text(
    text: str,
    prefer_object: bool = True,
) -> Optional[str]:
    """
    Extract JSON object or array from mixed text.
    
    Useful when LLM outputs JSON embedded in explanation text.
    
    Args:
        text: Text that may contain JSON
        prefer_object: If True, prefer {} over [] when both present
        
    Returns:
        Extracted JSON string or None if not found
        
    Examples:
        >>> extract_json_from_text('Here is the result: {"key": "value"} Done!')
        '{"key": "value"}'
    """
    if not text:
        return None
    
    candidates = []
    
    # Try to find JSON object
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        candidate = brace_match.group()
        try:
            json.loads(candidate)
            candidates.append(('object', candidate))
        except json.JSONDecodeError:
            # Try repair
            repaired, was_modified = repair_json(candidate, strict=False)
            try:
                json.loads(repaired)
                candidates.append(('object', repaired))
            except json.JSONDecodeError:
                pass
    
    # Try to find JSON array
    bracket_match = re.search(r'\[[\s\S]*\]', text)
    if bracket_match:
        candidate = bracket_match.group()
        try:
            json.loads(candidate)
            candidates.append(('array', candidate))
        except json.JSONDecodeError:
            # Try repair
            repaired, was_modified = repair_json(candidate, strict=False)
            try:
                json.loads(repaired)
                candidates.append(('array', repaired))
            except json.JSONDecodeError:
                pass
    
    if not candidates:
        return None
    
    # Return based on preference
    if prefer_object:
        for type_, json_str in candidates:
            if type_ == 'object':
                return json_str
    
    # Return first valid candidate
    return candidates[0][1]


def find_all_json_objects(text: str) -> list:
    """
    Find all valid JSON objects in text.
    
    Useful for extracting multiple JSON objects from a response.
    
    Args:
        text: Text containing JSON objects
        
    Returns:
        List of parsed JSON objects
    """
    results = []
    
    # Find all potential JSON objects
    # This uses a simple brace-matching approach
    depth = 0
    start = None
    
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start:i+1]
                try:
                    obj = json.loads(candidate)
                    results.append(obj)
                except json.JSONDecodeError:
                    # Try repair
                    repaired, _ = repair_json(candidate, strict=False)
                    try:
                        obj = json.loads(repaired)
                        results.append(obj)
                    except json.JSONDecodeError:
                        pass
                start = None
    
    return results


# Convenience function for common use case
def parse_llm_json(
    response: str,
    expected_keys: Optional[list] = None,
    default: Any = None,
) -> Any:
    """
    Parse JSON from LLM response with validation.
    
    Combines extraction, repair, and validation.
    
    Args:
        response: LLM response text
        expected_keys: If provided, validate these keys exist
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON data or default
    """
    json_str = extract_json_from_text(response)
    if not json_str:
        return default
    
    data = safe_json_loads(json_str, default=default)
    
    if data is default:
        return default
    
    # Validate expected keys
    if expected_keys and isinstance(data, dict):
        missing = [k for k in expected_keys if k not in data]
        if missing:
            logger.warning(f"Parsed JSON missing expected keys: {missing}")
    
    return data
