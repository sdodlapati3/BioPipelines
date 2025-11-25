"""
Main Error Diagnosis Agent.

Provides AI-powered error diagnosis for bioinformatics workflow failures
using a tiered approach:
1. Pattern matching (fast, offline)
2. LLM analysis (comprehensive, contextual)
"""

import re
import json
import logging
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from .categories import (
    ErrorCategory, 
    ErrorDiagnosis, 
    FixSuggestion,
    FixRiskLevel,
)
from .patterns import ERROR_PATTERNS, get_all_patterns
from .log_collector import LogCollector, CollectedLogs
from .prompts import (
    DIAGNOSIS_PROMPT,
    DIAGNOSIS_PROMPT_SIMPLE,
    SYSTEM_PROMPT_DIAGNOSIS,
    build_diagnosis_prompt,
)

logger = logging.getLogger(__name__)


# Provider configuration with priority
DIAGNOSIS_PROVIDERS = {
    "lightning": {
        "priority": 1,
        "cost": "free_tier",
        "use_for": ["general_diagnosis"],
        "env_var": "LIGHTNING_API_KEY",
    },
    "gemini": {
        "priority": 2,
        "cost": "free_tier", 
        "use_for": ["quick_analysis"],
        "env_var": "GOOGLE_API_KEY",
    },
    "ollama": {
        "priority": 3,
        "cost": "free",
        "use_for": ["offline_fallback"],
        "env_var": None,
    },
    "vllm": {
        "priority": 4,
        "cost": "free",
        "use_for": ["local_inference"],
        "env_var": "VLLM_API_BASE",
    },
    "openai": {
        "priority": 5,
        "cost": "paid",
        "use_for": ["complex_analysis"],
        "env_var": "OPENAI_API_KEY",
    },
    "anthropic": {
        "priority": 6,
        "cost": "paid",
        "use_for": ["complex_analysis"],
        "env_var": "ANTHROPIC_API_KEY",
    },
}


class ErrorDiagnosisAgent:
    """
    AI-powered error diagnosis agent for bioinformatics workflows.
    
    Uses a tiered approach:
    1. Pattern matching (fast, offline) - tries first
    2. LLM analysis (comprehensive) - for complex/unknown errors
    
    Example:
        agent = ErrorDiagnosisAgent()
        diagnosis = await agent.diagnose(failed_job)
        
        print(f"Error: {diagnosis.category}")
        for fix in diagnosis.suggested_fixes:
            print(f"  Fix: {fix.description}")
    """
    
    def __init__(
        self,
        llm=None,
        provider_priority: List[str] = None,
        pattern_confidence_threshold: float = 0.75,
    ):
        """
        Initialize the diagnosis agent.
        
        Args:
            llm: Pre-configured LLM adapter (optional)
            provider_priority: Ordered list of LLM providers to try
            pattern_confidence_threshold: Min confidence to use pattern match
        """
        self.llm = llm
        self.provider_priority = provider_priority or [
            "lightning",  # Free tier - priority
            "gemini",     # Free tier
            "ollama",     # Local/free
            "vllm",       # Local/free
            "openai",     # Paid backup
        ]
        self.pattern_confidence_threshold = pattern_confidence_threshold
        self.log_collector = LogCollector()
        self._llm_cache = {}
    
    async def diagnose(self, job) -> ErrorDiagnosis:
        """
        Perform full error diagnosis on a failed job.
        
        Args:
            job: PipelineJob object with failure information
            
        Returns:
            ErrorDiagnosis with root cause and fix suggestions
        """
        # Step 1: Collect all logs
        logger.info(f"Collecting logs for job diagnosis...")
        logs = self.log_collector.collect(job)
        
        if not logs.has_errors():
            return self._create_no_logs_diagnosis()
        
        # Step 2: Try pattern matching first (fast)
        pattern_result = self._match_patterns(logs)
        
        if pattern_result and pattern_result.confidence >= self.pattern_confidence_threshold:
            logger.info(
                f"Pattern match found: {pattern_result.category.value} "
                f"(confidence: {pattern_result.confidence:.0%})"
            )
            return pattern_result
        
        # Step 3: Use LLM for complex errors
        llm = self._get_available_llm()
        if llm:
            try:
                logger.info(f"Using LLM for deep analysis...")
                llm_result = await self._llm_diagnosis(logs, job, llm)
                if llm_result and llm_result.confidence > 0.5:
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM diagnosis failed: {e}")
        
        # Step 4: Return pattern result or unknown
        if pattern_result:
            return pattern_result
        
        return self._create_unknown_diagnosis(logs)
    
    def diagnose_sync(self, job) -> ErrorDiagnosis:
        """
        Synchronous wrapper for diagnose().
        
        Args:
            job: PipelineJob object
            
        Returns:
            ErrorDiagnosis
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.diagnose(job))
    
    def diagnose_from_logs(self, log_text: str) -> ErrorDiagnosis:
        """
        Diagnose from raw log text (pattern matching only).
        
        Args:
            log_text: Raw log content
            
        Returns:
            ErrorDiagnosis based on pattern matching
        """
        logs = CollectedLogs(nextflow_log=log_text)
        result = self._match_patterns(logs)
        
        if result:
            return result
        
        return self._create_unknown_diagnosis(logs)
    
    def _match_patterns(self, logs: CollectedLogs) -> Optional[ErrorDiagnosis]:
        """
        Match error patterns against log content.
        
        Args:
            logs: Collected log content
            
        Returns:
            ErrorDiagnosis if pattern matched, None otherwise
        """
        combined_logs = logs.get_full_log_text()
        
        if not combined_logs:
            return None
        
        best_match = None
        best_confidence = 0.0
        matched_text = ""
        match_count = 0
        
        for category, pattern_def in ERROR_PATTERNS.items():
            category_matches = 0
            category_matched_text = ""
            
            for pattern in pattern_def.patterns:
                try:
                    matches = re.findall(pattern, combined_logs, re.IGNORECASE | re.MULTILINE)
                    if matches:
                        category_matches += len(matches)
                        # Store first meaningful match
                        if not category_matched_text:
                            match_val = matches[0]
                            if isinstance(match_val, tuple):
                                match_val = match_val[0] if match_val else ""
                            category_matched_text = str(match_val)[:200]
                except re.error as e:
                    logger.warning(f"Invalid regex pattern: {pattern} - {e}")
                    continue
            
            if category_matches > 0:
                # Calculate confidence based on:
                # - Number of matches (more = more confident)
                # - Number of patterns matched (diversity)
                confidence = min(0.5 + (category_matches * 0.1), 0.95)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = pattern_def
                    matched_text = category_matched_text
                    match_count = category_matches
        
        if best_match:
            # Build diagnosis from pattern
            return ErrorDiagnosis(
                category=best_match.category,
                confidence=best_confidence,
                root_cause=f"{best_match.description}: {matched_text}".strip(": "),
                user_explanation=self._humanize_explanation(best_match, matched_text),
                log_excerpt=matched_text[:500] if matched_text else "",
                suggested_fixes=list(best_match.suggested_fixes),  # Copy list
                pattern_matched=True,
                failed_process=logs.failed_process,
                work_directory=logs.work_directory,
            )
        
        return None
    
    def _humanize_explanation(self, pattern, matched_text: str) -> str:
        """Convert technical error to user-friendly explanation."""
        explanations = {
            ErrorCategory.FILE_NOT_FOUND: (
                f"A required file could not be found. This often happens when "
                f"reference data hasn't been downloaded or the file path is incorrect."
            ),
            ErrorCategory.OUT_OF_MEMORY: (
                f"The analysis ran out of memory. Your input files may be too large "
                f"for the current memory allocation, or too many processes are running."
            ),
            ErrorCategory.CONTAINER_ERROR: (
                f"There's a problem with the software container. The container image "
                f"may not be built or Singularity may not be loaded."
            ),
            ErrorCategory.PERMISSION_DENIED: (
                f"Access to a file or directory was denied. Check file permissions "
                f"and ensure you have write access to the output directory."
            ),
            ErrorCategory.DEPENDENCY_MISSING: (
                f"A required software tool or package is not available. Make sure "
                f"all dependencies are installed in the container or environment."
            ),
            ErrorCategory.SLURM_ERROR: (
                f"The job scheduler (SLURM) reported an error. The job may have "
                f"exceeded time or memory limits, or been cancelled."
            ),
            ErrorCategory.NETWORK_ERROR: (
                f"A network connection failed. This could be due to server issues "
                f"or the compute node not having internet access."
            ),
            ErrorCategory.TOOL_ERROR: (
                f"A bioinformatics tool failed during execution. Check the input "
                f"data format and tool parameters."
            ),
            ErrorCategory.DATA_FORMAT_ERROR: (
                f"The input data format is incorrect or the file is corrupted. "
                f"Verify your input files are in the expected format."
            ),
        }
        
        return explanations.get(pattern.category, pattern.description)
    
    def _get_available_llm(self):
        """Get the first available LLM from priority list."""
        if self.llm:
            return self.llm
        
        # Try to import and check providers
        try:
            from ..llm import check_providers, get_llm
            
            available = check_providers()
            
            for provider in self.provider_priority:
                if available.get(provider):
                    # Check cache first
                    if provider in self._llm_cache:
                        return self._llm_cache[provider]
                    
                    try:
                        llm = get_llm(provider)
                        self._llm_cache[provider] = llm
                        logger.info(f"Using LLM provider: {provider}")
                        return llm
                    except Exception as e:
                        logger.debug(f"Failed to initialize {provider}: {e}")
                        continue
        except ImportError:
            logger.debug("LLM module not available")
        
        return None
    
    async def _llm_diagnosis(
        self, 
        logs: CollectedLogs, 
        job, 
        llm
    ) -> Optional[ErrorDiagnosis]:
        """
        Use LLM for deep error analysis.
        
        Args:
            logs: Collected logs
            job: Job object
            llm: LLM adapter
            
        Returns:
            ErrorDiagnosis or None
        """
        try:
            from ..llm import Message
        except ImportError:
            logger.warning("Message class not available")
            return None
        
        # Build context
        workflow_name = getattr(job, 'name', 'Unknown')
        analysis_type = getattr(job, 'analysis_type', 'Unknown')
        
        # Use simple prompt for free tier / faster response
        provider_name = llm.__class__.__name__.lower()
        use_simple = any(p in provider_name for p in ['lightning', 'gemini', 'ollama'])
        
        prompt = build_diagnosis_prompt(
            logs=logs,
            workflow_name=workflow_name,
            analysis_type=analysis_type,
            simple=use_simple,
            error_categories=", ".join([c.value for c in ErrorCategory]),
        )
        
        messages = [
            Message.system(SYSTEM_PROMPT_DIAGNOSIS),
            Message.user(prompt),
        ]
        
        # Get response
        response = llm.chat(messages)
        
        # Parse structured response
        return self._parse_llm_response(response.content, provider_name, logs)
    
    def _parse_llm_response(
        self, 
        content: str, 
        provider: str,
        logs: CollectedLogs
    ) -> Optional[ErrorDiagnosis]:
        """
        Parse LLM response into ErrorDiagnosis.
        
        Args:
            content: LLM response text
            provider: Name of LLM provider
            logs: Original logs for context
            
        Returns:
            ErrorDiagnosis or None
        """
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            logger.warning("No JSON found in LLM response")
            return None
        
        try:
            data = json.loads(json_match.group())
            
            # Map category string to enum
            category_str = data.get('error_category', 'unknown')
            category = ErrorCategory.from_string(category_str)
            
            # Parse fixes
            fixes = []
            for fix_data in data.get('suggested_fixes', []):
                risk_str = fix_data.get('risk_level', 'medium')
                risk = FixRiskLevel.from_string(risk_str)
                
                fixes.append(FixSuggestion(
                    description=fix_data.get('description', ''),
                    command=fix_data.get('command'),
                    risk_level=risk,
                    auto_executable=fix_data.get('auto_executable', False),
                ))
            
            # If no fixes from LLM, get from pattern database
            if not fixes and category in ERROR_PATTERNS:
                fixes = list(ERROR_PATTERNS[category].suggested_fixes)
            
            return ErrorDiagnosis(
                category=category,
                confidence=float(data.get('confidence', 0.7)),
                root_cause=data.get('root_cause', ''),
                user_explanation=data.get('user_explanation', ''),
                log_excerpt=data.get('log_excerpt', '')[:500],
                suggested_fixes=fixes,
                llm_provider_used=provider,
                pattern_matched=False,
                failed_process=logs.failed_process,
                work_directory=logs.work_directory,
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return None
    
    def _create_no_logs_diagnosis(self) -> ErrorDiagnosis:
        """Create diagnosis when no logs are available."""
        return ErrorDiagnosis(
            category=ErrorCategory.UNKNOWN,
            confidence=0.0,
            root_cause="No log files found",
            user_explanation=(
                "Unable to find log files for this job. The job may still be "
                "running, or the logs may have been deleted."
            ),
            suggested_fixes=[
                FixSuggestion(
                    description="Check if job is still running",
                    command="squeue -u $USER",
                    risk_level=FixRiskLevel.SAFE,
                    auto_executable=True,
                ),
                FixSuggestion(
                    description="Check job history",
                    command="sacct -j {job_id} --format=JobID,State,ExitCode",
                    risk_level=FixRiskLevel.SAFE,
                    auto_executable=True,
                ),
            ],
        )
    
    def _create_unknown_diagnosis(self, logs: CollectedLogs) -> ErrorDiagnosis:
        """Create diagnosis for unknown errors."""
        return ErrorDiagnosis(
            category=ErrorCategory.UNKNOWN,
            confidence=0.3,
            root_cause="Unable to automatically determine root cause",
            user_explanation=(
                "The error could not be automatically classified. Please review "
                "the log files manually for more details."
            ),
            log_excerpt=logs.get_combined_error_context(30),
            suggested_fixes=[
                FixSuggestion(
                    description="View full Nextflow log",
                    command="cat .nextflow.log | tail -200",
                    risk_level=FixRiskLevel.SAFE,
                    auto_executable=True,
                ),
                FixSuggestion(
                    description="Check work directory for process logs",
                    command=f"ls -la {logs.work_directory}" if logs.work_directory else "ls -la work/",
                    risk_level=FixRiskLevel.SAFE,
                    auto_executable=True,
                ),
            ],
            failed_process=logs.failed_process,
            work_directory=logs.work_directory,
        )


# Convenience function
def diagnose_job(job) -> ErrorDiagnosis:
    """
    Convenience function to diagnose a job.
    
    Args:
        job: PipelineJob or similar object
        
    Returns:
        ErrorDiagnosis
    """
    agent = ErrorDiagnosisAgent()
    return agent.diagnose_sync(job)


def diagnose_log(log_text: str) -> ErrorDiagnosis:
    """
    Convenience function to diagnose from log text.
    
    Args:
        log_text: Raw log content
        
    Returns:
        ErrorDiagnosis
    """
    agent = ErrorDiagnosisAgent()
    return agent.diagnose_from_logs(log_text)
