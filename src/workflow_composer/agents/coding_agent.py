"""
Coding Agent for BioPipelines
==============================

Specialized agent for code-related tasks:
- Error diagnosis from logs
- Code fix generation
- Workflow validation
- Performance optimization suggestions

Uses coding-optimized models (Qwen-Coder, DeepSeek-Coder) when available,
falls back to general models or cloud APIs.

Enhanced with DeepCode-inspired patterns:
- Adaptive retry with parameter reduction
- JSON repair for LLM outputs
- Error guidance generation
- Graceful degradation chains
"""

import os
import re
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

# DeepCode-inspired utilities
from .utils.json_repair import safe_json_loads, extract_json_from_text
from .utils.retry_strategy import (
    AdaptiveLLMCaller, 
    RetryConfig, 
    adjust_llm_params_for_retry,
    get_retry_delay,
)
from .utils.response_validator import ResponseValidator, ValidationResult
from .utils.degradation import DegradationChain, FallbackResult
from .utils.error_guidance import (
    ErrorGuidance,
    ErrorCategory,
    generate_error_guidance,
    generate_guidance_from_log,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Result Types
# =============================================================================

class ErrorType(Enum):
    """Types of errors the coding agent can diagnose."""
    SLURM = "slurm"              # SLURM job errors
    NEXTFLOW = "nextflow"        # Nextflow workflow errors
    SNAKEMAKE = "snakemake"      # Snakemake workflow errors
    TOOL = "tool"                # Bioinformatics tool errors
    MEMORY = "memory"            # Out of memory errors
    DISK = "disk"                # Disk space errors
    NETWORK = "network"          # Network/download errors
    PERMISSION = "permission"    # File permission errors
    SYNTAX = "syntax"            # Code syntax errors
    UNKNOWN = "unknown"          # Unknown error type


@dataclass
class DiagnosisResult:
    """Result of error diagnosis."""
    error_type: ErrorType
    root_cause: str
    explanation: str  # Human-friendly
    suggested_fix: Optional[str] = None  # Code or command
    confidence: float = 0.0
    auto_fixable: bool = False
    affected_file: Optional[str] = None
    line_number: Optional[int] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeFix:
    """A code fix suggestion."""
    original_code: str
    fixed_code: str
    diff: str
    explanation: str
    file_path: Optional[str] = None
    applied: bool = False


# =============================================================================
# Error Pattern Matching (Fast, No LLM)
# =============================================================================

ERROR_PATTERNS = {
    ErrorType.MEMORY: [
        (r"out of memory|oom|cannot allocate|memory exhausted|java\.lang\.OutOfMemoryError", 
         "Process ran out of memory. Try increasing memory allocation or processing smaller batches."),
        (r"exceeded.*memory.*limit|memory limit.*exceeded",
         "Job exceeded memory limit. Increase --mem in SLURM or process.memory in Nextflow."),
    ],
    ErrorType.DISK: [
        (r"no space left on device|disk quota exceeded|ENOSPC",
         "Disk space exhausted. Clean temp files or use a different work directory."),
    ],
    ErrorType.PERMISSION: [
        (r"permission denied|EACCES|cannot open.*for writing",
         "Permission denied. Check file ownership and permissions."),
    ],
    ErrorType.NEXTFLOW: [
        (r"Error executing process.*'([^']+)'",
         "Nextflow process '{}' failed. Check the process definition and inputs."),
        (r"Missing output file\(s\).*expected:.*'([^']+)'",
         "Process didn't create expected output '{}'. Check the script section."),
        (r"No such variable: (\w+)",
         "Undefined variable '{}' in Nextflow script. Define it in params or as an input."),
    ],
    ErrorType.SLURM: [
        (r"slurmstepd: error:.*Exceeded job memory limit",
         "SLURM job exceeded memory. Increase #SBATCH --mem."),
        (r"DUE TO TIME LIMIT",
         "Job exceeded time limit. Increase #SBATCH --time or optimize the workflow."),
        (r"srun: error: Unable to create step",
         "SLURM couldn't create job step. Check cluster resources or job parameters."),
    ],
    ErrorType.TOOL: [
        (r"(samtools|bwa|bowtie|star|hisat|salmon|kallisto|featureCounts).*error",
         "Bioinformatics tool error. Check input file formats and tool parameters."),
        (r"(gatk|picard|bcftools).*Exception",
         "GATK/Picard error. Verify reference genome and input file compatibility."),
    ],
    ErrorType.NETWORK: [
        (r"Connection refused|Connection timed out|Network is unreachable",
         "Network error. Check internet connection and firewall settings."),
        (r"Could not resolve host|Name or service not known",
         "DNS resolution failed. Check network configuration."),
    ],
}


def quick_diagnose(error_log: str) -> Optional[DiagnosisResult]:
    """
    Fast pattern-based diagnosis without LLM.
    
    Returns DiagnosisResult if a pattern matches, None otherwise.
    """
    for error_type, patterns in ERROR_PATTERNS.items():
        for pattern, explanation in patterns:
            match = re.search(pattern, error_log, re.IGNORECASE | re.MULTILINE)
            if match:
                # Fill in captured groups if any
                groups = match.groups()
                if groups:
                    explanation = explanation.format(*groups)
                
                return DiagnosisResult(
                    error_type=error_type,
                    root_cause=match.group(0)[:200],
                    explanation=explanation,
                    confidence=0.8,
                    auto_fixable=error_type in [ErrorType.MEMORY, ErrorType.DISK],
                )
    
    return None


# =============================================================================
# Coding Agent
# =============================================================================

# System prompt for the coding agent
CODING_SYSTEM_PROMPT = """You are an expert bioinformatics developer specializing in:
- Nextflow and Snakemake workflows
- SLURM cluster job management
- Bioinformatics tools (BWA, STAR, samtools, GATK, etc.)
- Error diagnosis and debugging

When diagnosing errors:
1. Identify the root cause precisely
2. Explain in simple terms what went wrong
3. Provide a specific fix (code or command)
4. Indicate if the fix can be applied automatically

When fixing code:
1. Preserve the original logic where possible
2. Add comments explaining the fix
3. Follow best practices for the language (Nextflow DSL2, Python 3.10+, etc.)

Be concise and actionable. Focus on solving the problem."""


class CodingAgent:
    """
    Specialized agent for code-related tasks.
    
    Uses a coding-optimized model when available:
    - Local: Qwen2.5-Coder-32B (via vLLM on GPU 2-3)
    - Cloud: DeepSeek-Coder via Lightning.ai
    - Fallback: GPT-4o or the main supervisor model
    
    Also uses fast pattern matching for common errors.
    
    Supports dependency injection: pass llm_client directly for better
    testability and decoupling (similar to ReactAgent pattern).
    
    Enhanced with DeepCode-inspired patterns:
    - Adaptive retry with parameter reduction on failure
    - JSON repair for malformed LLM outputs
    - Error guidance generation with anti-patterns
    - Graceful degradation (LLM -> pattern match -> default)
    """
    
    # Retry configuration - reduce complexity on failure
    DEFAULT_RETRY_CONFIG = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        token_reduction_factor=0.75,  # Reduce by 25% each retry
        temperature_reduction=0.2,    # Lower temp for stability
        min_temperature=0.1,
        min_tokens=512,
    )
    
    def __init__(
        self,
        llm_client: Any = None,
        model: str = None,
        coder_url: Optional[str] = None,
        coder_model: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        fallback_url: Optional[str] = None,
        fallback_model: str = "meta-llama/Llama-3.3-70B-Instruct",
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize the coding agent.
        
        Args:
            llm_client: Optional pre-configured OpenAI-compatible client (dependency injection)
            model: Model name to use with llm_client
            coder_url: URL of coding model vLLM server (e.g., http://localhost:8001/v1)
            coder_model: HuggingFace model ID for coding
            fallback_url: URL of fallback model server
            fallback_model: Model to use if coder unavailable
            retry_config: Configuration for adaptive retry behavior
            
        Note:
            If llm_client is provided, it takes precedence over URL-based discovery.
            This enables dependency injection for testing and decoupling.
        """
        # Dependency injection: if client provided, use it directly
        self._provided_client = llm_client
        self._provided_model = model
        
        # URL-based discovery (fallback if no client provided)
        self.coder_url = coder_url or os.environ.get("VLLM_CODER_URL")
        self.coder_model = coder_model
        self.fallback_url = fallback_url or os.environ.get("VLLM_URL", "http://localhost:8000/v1")
        self.fallback_model = fallback_model
        
        # Retry configuration
        self.retry_config = retry_config or self.DEFAULT_RETRY_CONFIG
        
        self._client = None
        self._model = None
        self._using_coder = False
        self._adaptive_caller: Optional[AdaptiveLLMCaller] = None
    
    def _get_client(self):
        """Get the OpenAI-compatible client, preferring coder model."""
        if self._client is not None:
            return self._client, self._model
        
        # Priority 1: Use injected client (dependency injection)
        if self._provided_client is not None:
            self._client = self._provided_client
            self._model = self._provided_model or "injected-model"
            logger.info("Using injected LLM client")
            return self._client, self._model
        
        try:
            from openai import OpenAI
            
            # Priority 2: Try coder model first
            if self.coder_url:
                try:
                    client = OpenAI(base_url=self.coder_url, api_key="not-needed")
                    client.models.list()  # Health check
                    self._client = client
                    self._model = self.coder_model
                    self._using_coder = True
                    logger.info(f"Using coder model at {self.coder_url}")
                    return client, self._model
                except Exception as e:
                    logger.debug(f"Coder model not available: {e}")
            
            # Try fallback vLLM
            if self.fallback_url:
                try:
                    client = OpenAI(base_url=self.fallback_url, api_key="not-needed")
                    client.models.list()
                    self._client = client
                    self._model = self.fallback_model
                    logger.info(f"Using fallback model at {self.fallback_url}")
                    return client, self._model
                except Exception as e:
                    logger.debug(f"Fallback vLLM not available: {e}")
            
            # Try Lightning.ai
            lightning_key = os.environ.get("LIGHTNING_API_KEY")
            if lightning_key:
                client = OpenAI(
                    base_url="https://api.lightning.ai/v1",
                    api_key=lightning_key
                )
                self._client = client
                self._model = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
                logger.info("Using Lightning.ai with DeepSeek-Coder")
                return client, self._model
            
            # Try OpenAI
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                client = OpenAI(api_key=openai_key)
                self._client = client
                self._model = "gpt-4o"
                logger.info("Using OpenAI GPT-4o")
                return client, self._model
            
        except ImportError:
            logger.error("openai package not installed")
        
        return None, None
    
    def diagnose_error(
        self,
        error_log: str,
        workflow_config: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        use_llm: bool = True,
        include_guidance: bool = True,
    ) -> DiagnosisResult:
        """
        Diagnose an error from logs.
        
        Uses a graceful degradation chain:
        1. LLM diagnosis (if available and use_llm=True)
        2. Pattern matching (fast, always available)
        3. Default unknown error
        
        Args:
            error_log: The error log content
            workflow_config: Optional workflow configuration
            context: Additional context
            use_llm: Whether to use LLM (False = pattern matching only)
            include_guidance: Include actionable guidance in result
            
        Returns:
            DiagnosisResult with diagnosis, suggested fix, and guidance
        """
        # Build degradation chain
        chain = DegradationChain()
        
        # Try LLM-based diagnosis first (most accurate)
        if use_llm:
            chain.add(
                name="llm_diagnosis",
                method=lambda: self._llm_diagnose(error_log, workflow_config, context),
                condition=lambda: self._get_client()[0] is not None,
            )
        
        # Fall back to pattern matching
        chain.add(
            name="pattern_diagnosis",
            method=lambda: quick_diagnose(error_log),
        )
        
        # Execute chain with default fallback
        default_result = DiagnosisResult(
            error_type=ErrorType.UNKNOWN,
            root_cause="Could not determine error cause",
            explanation="Unable to diagnose this error automatically.",
            confidence=0.0,
        )
        
        fallback = chain.execute(default=default_result)
        result = fallback.value
        
        if self.verbose:
            logger.info(f"Diagnosis method: {fallback.method_used}, level: {fallback.fallback_level}")
        
        # Add actionable guidance if requested
        if include_guidance and result:
            # Map ErrorType to ErrorCategory for guidance generation
            category_map = {
                ErrorType.MEMORY: ErrorCategory.MEMORY,
                ErrorType.DISK: ErrorCategory.DISK,
                ErrorType.PERMISSION: ErrorCategory.PERMISSION,
                ErrorType.NEXTFLOW: ErrorCategory.NEXTFLOW,
                ErrorType.SNAKEMAKE: ErrorCategory.SNAKEMAKE,
                ErrorType.SLURM: ErrorCategory.SLURM,
                ErrorType.TOOL: ErrorCategory.TOOL,
                ErrorType.NETWORK: ErrorCategory.NETWORK,
                ErrorType.SYNTAX: ErrorCategory.SYNTAX,
                ErrorType.UNKNOWN: ErrorCategory.UNKNOWN,
            }
            category = category_map.get(result.error_type, ErrorCategory.UNKNOWN)
            guidance = generate_error_guidance(category, result.root_cause)
            result.additional_context["guidance"] = guidance
            result.additional_context["guidance_markdown"] = guidance.to_markdown()
        
        return result
    
    def _llm_diagnose(
        self,
        error_log: str,
        workflow_config: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> DiagnosisResult:
        """
        LLM-based diagnosis with adaptive retry.
        
        On retry, reduces max_tokens and temperature for stability.
        """
        client, model = self._get_client()
        if not client:
            raise RuntimeError("No LLM client available")
        
        # Build prompt
        prompt = self._build_diagnosis_prompt(error_log, workflow_config, context)
        
        # Use adaptive retry
        last_error = None
        for attempt in range(self.retry_config.max_attempts):
            try:
                # Adjust params for retry (reduce complexity)
                params = adjust_llm_params_for_retry(
                    {"temperature": 0.1, "max_tokens": 1024},
                    attempt=attempt,
                    config=self.retry_config,
                )
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": CODING_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    **params,
                )
                
                response_text = response.choices[0].message.content
                
                # Validate response structure
                validation = ResponseValidator.validate_diagnosis_response(response_text)
                if validation.warnings:
                    logger.debug(f"Diagnosis validation warnings: {validation.warnings}")
                
                return self._parse_diagnosis(response_text, error_log)
                
            except Exception as e:
                last_error = e
                if attempt < self.retry_config.max_attempts - 1:
                    delay = get_retry_delay(attempt, self.retry_config)
                    logger.warning(f"LLM diagnosis attempt {attempt + 1} failed: {e}, retrying in {delay:.1f}s")
                    import time
                    time.sleep(delay)
        
        # All retries failed
        raise last_error or RuntimeError("LLM diagnosis failed")
    
    @property
    def verbose(self) -> bool:
        """Check if verbose logging is enabled."""
        return logger.isEnabledFor(logging.DEBUG)
    
    def generate_fix(
        self,
        diagnosis: DiagnosisResult,
        file_content: str,
        file_type: str = "nextflow",
    ) -> Optional[CodeFix]:
        """
        Generate a code fix based on diagnosis.
        
        Args:
            diagnosis: The error diagnosis
            file_content: Content of the file to fix
            file_type: Type of file (nextflow, snakemake, python, bash)
            
        Returns:
            CodeFix with the suggested fix, or None if unable
        """
        if not diagnosis.auto_fixable:
            logger.info("Error is not auto-fixable")
            return None
        
        client, model = self._get_client()
        if not client:
            return None
        
        prompt = f"""Based on this error diagnosis:
Error Type: {diagnosis.error_type.value}
Root Cause: {diagnosis.root_cause}
Explanation: {diagnosis.explanation}

Fix the following {file_type} code:

```{file_type}
{file_content}
```

Provide ONLY the corrected code, with comments explaining the changes.
Do not include explanations outside the code block."""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CODING_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            
            fixed_code = self._extract_code(response.choices[0].message.content, file_type)
            if not fixed_code:
                return None
            
            return CodeFix(
                original_code=file_content,
                fixed_code=fixed_code,
                diff=self._generate_diff(file_content, fixed_code),
                explanation=diagnosis.explanation,
            )
            
        except Exception as e:
            logger.error(f"Fix generation failed: {e}")
            return None
    
    def explain_code(
        self,
        code: str,
        language: str = "nextflow",
        detail_level: str = "detailed",
    ) -> str:
        """
        Explain what a piece of code does.
        
        Args:
            code: The code to explain
            language: Programming language
            detail_level: "brief", "detailed", or "beginner"
            
        Returns:
            Human-readable explanation
        """
        client, model = self._get_client()
        if not client:
            return "Unable to explain code - no AI model available."
        
        level_prompts = {
            "brief": "Explain this briefly in 2-3 sentences.",
            "detailed": "Explain this in detail, including what each part does.",
            "beginner": "Explain this as if to a beginner, avoiding jargon.",
        }
        
        prompt = f"""Explain this {language} code:

```{language}
{code}
```

{level_prompts.get(detail_level, level_prompts['detailed'])}"""

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CODING_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Unable to explain code: {e}"
    
    def validate_workflow(
        self,
        workflow_content: str,
        workflow_type: str = "nextflow",
    ) -> Dict[str, Any]:
        """
        Validate a workflow configuration.
        
        Args:
            workflow_content: The workflow code
            workflow_type: Type of workflow (nextflow, snakemake)
            
        Returns:
            Dict with 'valid', 'errors', and 'warnings'
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
        }
        
        # Basic syntax checks (no LLM needed)
        if workflow_type == "nextflow":
            result = self._validate_nextflow_syntax(workflow_content, result)
        elif workflow_type == "snakemake":
            result = self._validate_snakemake_syntax(workflow_content, result)
        
        # Use LLM for deeper analysis if available
        client, model = self._get_client()
        if client:
            try:
                prompt = f"""Review this {workflow_type} workflow for potential issues:

```{workflow_type}
{workflow_content}
```

List any:
1. Errors (will cause failure)
2. Warnings (potential issues)
3. Suggestions (improvements)

Format as JSON: {{"errors": [...], "warnings": [...], "suggestions": [...]}}"""

                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": CODING_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                )
                
                # Use JSON repair for LLM output (DeepCode pattern)
                content = response.choices[0].message.content
                
                # Extract and repair JSON from response
                json_str = extract_json_from_text(content)
                if json_str:
                    llm_result = safe_json_loads(json_str, default={})
                    if llm_result:
                        result["errors"].extend(llm_result.get("errors", []))
                        result["warnings"].extend(llm_result.get("warnings", []))
                        result["suggestions"].extend(llm_result.get("suggestions", []))
                else:
                    # Fallback: try parsing the whole content
                    llm_result = safe_json_loads(content, default=None)
                    if llm_result:
                        result["errors"].extend(llm_result.get("errors", []))
                        result["warnings"].extend(llm_result.get("warnings", []))
                        result["suggestions"].extend(llm_result.get("suggestions", []))
                    
            except Exception as e:
                logger.debug(f"LLM validation failed: {e}")
        
        result["valid"] = len(result["errors"]) == 0
        return result
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _build_diagnosis_prompt(
        self,
        error_log: str,
        workflow_config: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Build the diagnosis prompt."""
        prompt = f"""Diagnose this error:

ERROR LOG:
```
{error_log[:3000]}
```
"""
        if workflow_config:
            prompt += f"""
WORKFLOW CONFIG:
```
{workflow_config[:1500]}
```
"""
        if context:
            prompt += f"""
CONTEXT:
- Job ID: {context.get('job_id', 'unknown')}
- Workflow: {context.get('workflow_name', 'unknown')}
- Step: {context.get('step', 'unknown')}
"""
        
        prompt += """
Provide your diagnosis as:
1. ERROR TYPE: (memory/disk/permission/nextflow/slurm/tool/network/syntax/unknown)
2. ROOT CAUSE: (one line)
3. EXPLANATION: (2-3 sentences, beginner-friendly)
4. SUGGESTED FIX: (specific command or code change)
5. AUTO FIXABLE: (yes/no)"""
        
        return prompt
    
    def _parse_diagnosis(self, llm_response: str, error_log: str) -> DiagnosisResult:
        """Parse LLM diagnosis response."""
        # Default values
        error_type = ErrorType.UNKNOWN
        root_cause = "Unknown"
        explanation = llm_response[:500]
        suggested_fix = None
        auto_fixable = False
        
        # Try to parse structured response
        lines = llm_response.split('\n')
        for line in lines:
            line_lower = line.lower()
            if 'error type:' in line_lower:
                for et in ErrorType:
                    if et.value in line_lower:
                        error_type = et
                        break
            elif 'root cause:' in line_lower:
                root_cause = line.split(':', 1)[-1].strip()
            elif 'explanation:' in line_lower:
                explanation = line.split(':', 1)[-1].strip()
            elif 'suggested fix:' in line_lower or 'fix:' in line_lower:
                suggested_fix = line.split(':', 1)[-1].strip()
            elif 'auto fixable:' in line_lower:
                auto_fixable = 'yes' in line_lower
        
        return DiagnosisResult(
            error_type=error_type,
            root_cause=root_cause,
            explanation=explanation,
            suggested_fix=suggested_fix,
            confidence=0.85,
            auto_fixable=auto_fixable,
        )
    
    def _extract_code(self, response: str, language: str) -> Optional[str]:
        """Extract code block from LLM response."""
        # Try to find code block with language
        pattern = rf'```{language}\n(.*?)```'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try generic code block
        match = re.search(r'```\n?(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return None
    
    def _generate_diff(self, original: str, fixed: str) -> str:
        """Generate a simple diff between original and fixed code."""
        try:
            import difflib
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                fixed.splitlines(keepends=True),
                fromfile='original',
                tofile='fixed',
            )
            return ''.join(diff)
        except Exception:
            return ""
    
    def _validate_nextflow_syntax(self, content: str, result: Dict) -> Dict:
        """Basic Nextflow syntax validation."""
        # Check for common issues
        if 'process ' in content and 'output:' not in content:
            result["warnings"].append("Process without output block detected")
        
        if 'workflow {' not in content and 'workflow ' not in content:
            result["warnings"].append("No workflow block found - might be using DSL1")
        
        if 'nextflow.enable.dsl' not in content and 'DSL2' not in content:
            result["warnings"].append("DSL version not specified - add 'nextflow.enable.dsl=2'")
        
        # Check for unclosed braces
        if content.count('{') != content.count('}'):
            result["errors"].append("Mismatched braces detected")
        
        return result
    
    def _validate_snakemake_syntax(self, content: str, result: Dict) -> Dict:
        """Basic Snakemake syntax validation."""
        if 'rule ' not in content:
            result["errors"].append("No rules found in Snakemake file")
        
        if 'rule all:' not in content:
            result["warnings"].append("No 'rule all' found - workflow may not have a target")
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        client, model = self._get_client()
        return {
            "available": client is not None,
            "model": model,
            "using_coder_model": self._using_coder,
            "coder_url": self.coder_url,
            "fallback_url": self.fallback_url,
            "retry_config": {
                "max_attempts": self.retry_config.max_attempts,
                "token_reduction_factor": self.retry_config.token_reduction_factor,
                "temperature_reduction": self.retry_config.temperature_reduction,
            },
        }
    
    def get_guidance_for_error(
        self,
        error_type: ErrorType,
        root_cause: str = "",
    ) -> str:
        """
        Get formatted guidance for an error type.
        
        Useful for displaying to users or injecting into agent prompts.
        
        Args:
            error_type: Type of error
            root_cause: Optional root cause description
            
        Returns:
            Markdown-formatted guidance string
        """
        # Map ErrorType to ErrorCategory
        category_map = {
            ErrorType.MEMORY: ErrorCategory.MEMORY,
            ErrorType.DISK: ErrorCategory.DISK,
            ErrorType.PERMISSION: ErrorCategory.PERMISSION,
            ErrorType.NEXTFLOW: ErrorCategory.NEXTFLOW,
            ErrorType.SNAKEMAKE: ErrorCategory.SNAKEMAKE,
            ErrorType.SLURM: ErrorCategory.SLURM,
            ErrorType.TOOL: ErrorCategory.TOOL,
            ErrorType.NETWORK: ErrorCategory.NETWORK,
            ErrorType.SYNTAX: ErrorCategory.SYNTAX,
            ErrorType.UNKNOWN: ErrorCategory.UNKNOWN,
        }
        category = category_map.get(error_type, ErrorCategory.UNKNOWN)
        guidance = generate_error_guidance(category, root_cause)
        return guidance.to_markdown()
    
    def diagnose_with_full_guidance(
        self,
        error_log: str,
        workflow_config: Optional[str] = None,
    ) -> Tuple[DiagnosisResult, str]:
        """
        Diagnose error and return full formatted guidance.
        
        Convenience method that combines diagnosis + guidance formatting.
        
        Args:
            error_log: Error log content
            workflow_config: Optional workflow configuration
            
        Returns:
            Tuple of (DiagnosisResult, formatted_guidance_markdown)
        """
        result = self.diagnose_error(
            error_log, 
            workflow_config, 
            include_guidance=True
        )
        
        guidance_md = result.additional_context.get(
            "guidance_markdown",
            self.get_guidance_for_error(result.error_type, result.root_cause)
        )
        
        return result, guidance_md


# =============================================================================
# Convenience Functions
# =============================================================================

_coding_agent = None


def get_coding_agent() -> CodingAgent:
    """Get or create the global coding agent instance."""
    global _coding_agent
    if _coding_agent is None:
        _coding_agent = CodingAgent()
    return _coding_agent


def diagnose_job_error(job_id: str, logs: str, config: str = None) -> DiagnosisResult:
    """Convenience function to diagnose a job error."""
    agent = get_coding_agent()
    return agent.diagnose_error(logs, config, context={"job_id": job_id})


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample error
    sample_error = """
Error executing process > 'STAR_ALIGN'
Caused by:
    Process `STAR_ALIGN` terminated with an error exit status (137)
Command executed:
    STAR --runMode alignReads --genomeDir /ref/star_index ...
Command error:
    .command.sh: line 3: 12345 Killed
    slurmstepd: error: Detected 1 oom-kill event(s) in StepId=123.0
    """
    
    agent = CodingAgent()
    print("\n🔧 Coding Agent Status:")
    print(json.dumps(agent.get_status(), indent=2))
    
    print("\n📋 Diagnosing sample error...")
    result = agent.diagnose_error(sample_error, use_llm=False)
    print(f"  Error Type: {result.error_type.value}")
    print(f"  Root Cause: {result.root_cause}")
    print(f"  Explanation: {result.explanation}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Auto-fixable: {result.auto_fixable}")
