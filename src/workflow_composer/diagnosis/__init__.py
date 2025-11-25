"""
Error Diagnosis Agent for BioPipelines.

This package provides AI-powered error diagnosis capabilities for bioinformatics
workflow failures. It uses a tiered approach:

1. Pattern matching (fast, offline)
2. LLM analysis (comprehensive, contextual)
3. Auto-fix execution (safe remediation)

Usage:
    from workflow_composer.diagnosis import ErrorDiagnosisAgent
    
    agent = ErrorDiagnosisAgent()
    diagnosis = await agent.diagnose(failed_job)
    
    print(f"Error: {diagnosis.category}")
    print(f"Root Cause: {diagnosis.root_cause}")
    for fix in diagnosis.suggested_fixes:
        print(f"  - {fix.description}")
"""

from .categories import (
    ErrorCategory,
    FixRiskLevel,
    ErrorPattern,
    FixSuggestion,
    ErrorDiagnosis,
)
from .patterns import ERROR_PATTERNS, get_pattern, get_all_patterns
from .log_collector import LogCollector, CollectedLogs
from .agent import ErrorDiagnosisAgent
from .auto_fix import AutoFixEngine, FixResult
from .gemini_adapter import GeminiAdapter, get_gemini, check_gemini_available
from .lightning_adapter import LightningDiagnosisAdapter, get_lightning_adapter
from .github_agent import GitHubCopilotAgent, get_github_copilot_agent

__all__ = [
    # Categories
    "ErrorCategory",
    "FixRiskLevel",
    "ErrorPattern",
    "FixSuggestion",
    "ErrorDiagnosis",
    # Patterns
    "ERROR_PATTERNS",
    "get_pattern",
    "get_all_patterns",
    # Log Collection
    "LogCollector",
    "CollectedLogs",
    # Main Agent
    "ErrorDiagnosisAgent",
    # Auto Fix
    "AutoFixEngine",
    "FixResult",
    # LLM Adapters
    "GeminiAdapter",
    "get_gemini",
    "check_gemini_available",
    "LightningDiagnosisAdapter",
    "get_lightning_adapter",
    # GitHub Integration
    "GitHubCopilotAgent",
    "get_github_copilot_agent",
]
