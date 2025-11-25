"""
Error categories and data structures for diagnosis.

This module defines the taxonomy of pipeline errors and the data structures
used throughout the diagnosis system.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional


class ErrorCategory(Enum):
    """Classification of pipeline errors."""
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_DENIED = "permission_denied"
    OUT_OF_MEMORY = "out_of_memory"
    CONTAINER_ERROR = "container_error"
    DEPENDENCY_MISSING = "dependency_missing"
    TOOL_ERROR = "tool_error"
    SYNTAX_ERROR = "syntax_error"
    NETWORK_ERROR = "network_error"
    RESOURCE_LIMIT = "resource_limit"
    DATA_FORMAT_ERROR = "data_format_error"
    REFERENCE_MISSING = "reference_missing"
    INDEX_MISSING = "index_missing"
    CONFIGURATION_ERROR = "configuration_error"
    SLURM_ERROR = "slurm_error"
    NEXTFLOW_ERROR = "nextflow_error"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_string(cls, value: str) -> "ErrorCategory":
        """Convert string to ErrorCategory, with fallback."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.UNKNOWN


class FixRiskLevel(Enum):
    """Risk level for auto-fix actions."""
    SAFE = "safe"       # Can execute automatically (e.g., create dir)
    LOW = "low"         # Execute with notification (e.g., pull container)
    MEDIUM = "medium"   # Require user confirmation (e.g., modify config)
    HIGH = "high"       # Show instructions only (e.g., modify code)
    
    @classmethod
    def from_string(cls, value: str) -> "FixRiskLevel":
        """Convert string to FixRiskLevel, with fallback."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.MEDIUM


@dataclass
class FixSuggestion:
    """A suggested fix for an error."""
    description: str
    command: Optional[str] = None
    risk_level: FixRiskLevel = FixRiskLevel.MEDIUM
    auto_executable: bool = False
    requires_confirmation: bool = True
    
    def __post_init__(self):
        # Safe fixes don't require confirmation
        if self.risk_level == FixRiskLevel.SAFE:
            self.requires_confirmation = False
            self.auto_executable = True


@dataclass
class ErrorPattern:
    """Definition of an error pattern for matching."""
    category: ErrorCategory
    patterns: List[str]  # Regex patterns
    description: str
    common_causes: List[str] = field(default_factory=list)
    suggested_fixes: List[FixSuggestion] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)  # For fuzzy matching


@dataclass
class ErrorDiagnosis:
    """Result of error diagnosis."""
    category: ErrorCategory
    confidence: float  # 0.0 to 1.0
    root_cause: str
    user_explanation: str
    log_excerpt: str = ""
    suggested_fixes: List[FixSuggestion] = field(default_factory=list)
    llm_provider_used: Optional[str] = None
    pattern_matched: bool = False
    failed_process: Optional[str] = None
    work_directory: Optional[str] = None
    similar_issues: List[str] = field(default_factory=list)
    
    @property
    def is_confident(self) -> bool:
        """Check if diagnosis has high confidence."""
        return self.confidence >= 0.7
    
    @property
    def has_auto_fixes(self) -> bool:
        """Check if there are auto-executable fixes."""
        return any(fix.auto_executable for fix in self.suggested_fixes)
    
    def get_safe_fixes(self) -> List[FixSuggestion]:
        """Get fixes that can be auto-executed safely."""
        return [f for f in self.suggested_fixes if f.risk_level == FixRiskLevel.SAFE]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category.value,
            "confidence": self.confidence,
            "root_cause": self.root_cause,
            "user_explanation": self.user_explanation,
            "log_excerpt": self.log_excerpt,
            "suggested_fixes": [
                {
                    "description": f.description,
                    "command": f.command,
                    "risk_level": f.risk_level.value,
                    "auto_executable": f.auto_executable,
                }
                for f in self.suggested_fixes
            ],
            "llm_provider_used": self.llm_provider_used,
            "pattern_matched": self.pattern_matched,
            "failed_process": self.failed_process,
        }
