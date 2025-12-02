"""
Multi-Agent Specialists Package
===============================

Specialist agents for coordinated workflow generation:
- PlannerAgent: Designs workflow architecture
- CodeGenAgent: Generates Nextflow code
- ValidatorAgent: Reviews and validates code
- DocAgent: Generates documentation
- QCAgent: Quality control validation
- SupervisorAgent: Coordinates all specialists
- OrchestratedSupervisor: SupervisorAgent with Orchestrator-8B routing (ToolOrchestra)
- ReferenceDiscoveryAgent: Discovers relevant code references (DeepCode-inspired)
- CodebaseIndexer: Indexes existing codebases for intelligent reference (DeepCode-inspired)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class AgentRole(Enum):
    """Role identifiers for agents."""
    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    CODEGEN = "codegen"
    VALIDATOR = "validator"
    DOCS = "docs"
    QC = "qc"


@dataclass
class ValidationResult:
    """Result of code validation."""
    valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ValidationIssue:
    """A single validation issue."""
    severity: str  # 'error', 'warning', 'info'
    message: str
    line: Optional[int] = None
    rule: Optional[str] = None


# Import agents after defining shared types
from .planner import PlannerAgent, WorkflowPlan, WorkflowStep
from .codegen import CodeGenAgent
from .validator import ValidatorAgent
from .docs import DocAgent
from .qc import QCAgent, QCMetric, QCReport
from .supervisor import SupervisorAgent, WorkflowState
from .reference_discovery import (
    ReferenceDiscoveryAgent,
    ReferenceSource,
    CodeReference,
    ReferenceSearchResult,
)
from .codebase_indexer import (
    CodebaseIndexer,
    CodebaseIndex,
    CodeElement,
    CodeElementType,
)
from .orchestrated_supervisor import (
    OrchestratedSupervisor,
    OrchestratedResult,
    get_supervisor,
)


@dataclass
class WorkflowResult:
    """Complete result from workflow generation."""
    success: bool
    plan: Optional[WorkflowPlan] = None
    code: Optional[str] = None
    config: Optional[str] = None
    documentation: Optional[str] = None
    validation_passed: bool = False
    validation_issues: List[str] = field(default_factory=list)
    output_files: Dict[str, str] = field(default_factory=dict)


__all__ = [
    # Enums and data classes
    "AgentRole",
    "ValidationResult",
    "ValidationIssue",
    "WorkflowResult",
    "WorkflowPlan",
    "WorkflowStep",
    "WorkflowState",
    "QCMetric",
    "QCReport",
    # Core Agents
    "PlannerAgent",
    "CodeGenAgent",
    "ValidatorAgent",
    "DocAgent",
    "QCAgent",
    "SupervisorAgent",
    # DeepCode-inspired Agents
    "ReferenceDiscoveryAgent",
    "ReferenceSource",
    "CodeReference",
    "ReferenceSearchResult",
    "CodebaseIndexer",
    "CodebaseIndex",
    "CodeElement",
    "CodeElementType",
    # ToolOrchestra Integration
    "OrchestratedSupervisor",
    "OrchestratedResult",
    "get_supervisor",
]
