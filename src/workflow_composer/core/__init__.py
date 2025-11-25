"""
Core Components
===============

Core functionality for the Workflow Composer:
- Intent Parser: Extract structured intent from natural language
- Tool Selector: Query tool catalog and select appropriate tools
- Module Mapper: Map tools to Nextflow modules
- Workflow Generator: Generate Nextflow DSL2 workflows
"""

from .intent_parser import IntentParser, ParsedIntent, AnalysisType
from .tool_selector import ToolSelector, Tool, ToolMatch
from .module_mapper import ModuleMapper, Module
from .workflow_generator import WorkflowGenerator, Workflow

__all__ = [
    "IntentParser",
    "ParsedIntent",
    "AnalysisType",
    "ToolSelector",
    "Tool",
    "ToolMatch",
    "ModuleMapper",
    "Module",
    "WorkflowGenerator",
    "Workflow"
]
