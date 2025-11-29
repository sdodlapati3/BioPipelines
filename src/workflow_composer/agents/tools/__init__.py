"""
BioPipelines Agent Tools
========================

Modular tool system for the BioPipelines AI agent.
All tools are organized by category and unified through the AgentTools class.

Usage:
    from tools import get_agent_tools
    tools = get_agent_tools()
    result = tools.execute_tool("scan_data", path="/data")
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import base types
from .base import ToolResult, ToolName, TOOL_PATTERNS

# Import registry
from .registry import ToolRegistry, get_registry

# Import tool implementations and patterns
from .data_discovery import (
    SCAN_DATA_PATTERNS,
    SEARCH_DATABASES_PATTERNS,
    SEARCH_TCGA_PATTERNS,
    DESCRIBE_FILES_PATTERNS,
    VALIDATE_DATASET_PATTERNS,
    scan_data_impl,
    search_databases_impl,
    search_tcga_impl,
    describe_files_impl,
    validate_dataset_impl,
)

from .data_management import (
    DOWNLOAD_DATASET_PATTERNS,
    DOWNLOAD_REFERENCE_PATTERNS,
    BUILD_INDEX_PATTERNS,
    CLEANUP_DATA_PATTERNS,
    CONFIRM_CLEANUP_PATTERNS,
    download_dataset_impl,
    download_reference_impl,
    build_index_impl,
    cleanup_data_impl,
    confirm_cleanup_impl,
)

from .workflow import (
    GENERATE_WORKFLOW_PATTERNS,
    LIST_WORKFLOWS_PATTERNS,
    CHECK_REFERENCES_PATTERNS,
    VISUALIZE_WORKFLOW_PATTERNS,
    generate_workflow_impl,
    list_workflows_impl,
    check_references_impl,
    visualize_workflow_impl,
)

from .execution import (
    SUBMIT_JOB_PATTERNS,
    GET_JOB_STATUS_PATTERNS,
    GET_LOGS_PATTERNS,
    CANCEL_JOB_PATTERNS,
    submit_job_impl,
    get_job_status_impl,
    get_logs_impl,
    cancel_job_impl,
)

from .diagnostics import (
    DIAGNOSE_ERROR_PATTERNS,
    ANALYZE_RESULTS_PATTERNS,
    diagnose_error_impl,
    analyze_results_impl,
)

from .education import (
    EXPLAIN_CONCEPT_PATTERNS,
    COMPARE_SAMPLES_PATTERNS,
    GET_HELP_PATTERNS,
    explain_concept_impl,
    compare_samples_impl,
    get_help_impl,
    CONCEPT_KNOWLEDGE,
)


# =============================================================================
# UNIFIED TOOL PATTERNS
# =============================================================================

# Combined pattern list for regex-based tool detection
# Order matters - more specific patterns should come first
ALL_TOOL_PATTERNS = [
    # Data Discovery
    (ToolName.SCAN_DATA, SCAN_DATA_PATTERNS),
    (ToolName.SEARCH_DATABASES, SEARCH_DATABASES_PATTERNS),
    (ToolName.SEARCH_TCGA, SEARCH_TCGA_PATTERNS),
    (ToolName.DESCRIBE_FILES, DESCRIBE_FILES_PATTERNS),
    (ToolName.VALIDATE_DATASET, VALIDATE_DATASET_PATTERNS),
    
    # Data Management
    (ToolName.DOWNLOAD_DATASET, DOWNLOAD_DATASET_PATTERNS),
    (ToolName.DOWNLOAD_REFERENCE, DOWNLOAD_REFERENCE_PATTERNS),
    (ToolName.BUILD_INDEX, BUILD_INDEX_PATTERNS),
    (ToolName.CLEANUP_DATA, CLEANUP_DATA_PATTERNS),
    (ToolName.CONFIRM_CLEANUP, CONFIRM_CLEANUP_PATTERNS),
    
    # Workflow
    (ToolName.GENERATE_WORKFLOW, GENERATE_WORKFLOW_PATTERNS),
    (ToolName.LIST_WORKFLOWS, LIST_WORKFLOWS_PATTERNS),
    (ToolName.CHECK_REFERENCES, CHECK_REFERENCES_PATTERNS),
    (ToolName.VISUALIZE_WORKFLOW, VISUALIZE_WORKFLOW_PATTERNS),
    
    # Execution
    (ToolName.SUBMIT_JOB, SUBMIT_JOB_PATTERNS),
    (ToolName.GET_JOB_STATUS, GET_JOB_STATUS_PATTERNS),
    (ToolName.GET_LOGS, GET_LOGS_PATTERNS),
    (ToolName.CANCEL_JOB, CANCEL_JOB_PATTERNS),
    
    # Diagnostics
    (ToolName.DIAGNOSE_ERROR, DIAGNOSE_ERROR_PATTERNS),
    (ToolName.ANALYZE_RESULTS, ANALYZE_RESULTS_PATTERNS),
    
    # Education
    (ToolName.EXPLAIN_CONCEPT, EXPLAIN_CONCEPT_PATTERNS),
    (ToolName.COMPARE_SAMPLES, COMPARE_SAMPLES_PATTERNS),
    (ToolName.SHOW_HELP, GET_HELP_PATTERNS),
]


# =============================================================================
# UNIFIED AGENT TOOLS CLASS
# =============================================================================

class AgentTools:
    """
    Unified interface for all BioPipelines agent tools.
    
    This class provides:
    - Centralized tool execution
    - Pattern-based tool detection
    - OpenAI function definitions
    
    Usage:
        tools = AgentTools()
        result = tools.execute_tool("scan_data", path="/data")
    """
    
    def __init__(self, base_path: str = None):
        """
        Initialize agent tools.
        
        Args:
            base_path: Base directory for data operations
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        
        # Build tool dispatch table
        self._tool_dispatch = self._build_dispatch_table()
        
        # Cache for OpenAI function definitions
        self._function_definitions = None
        
        logger.info(f"AgentTools initialized with {len(self._tool_dispatch)} tools")
    
    def _build_dispatch_table(self) -> Dict[str, callable]:
        """Build mapping from tool name to implementation."""
        dispatch = {
            # Data Discovery
            "scan_data": lambda **kw: scan_data_impl(**kw),
            "search_databases": lambda **kw: search_databases_impl(**kw),
            "search_tcga": lambda **kw: search_tcga_impl(**kw),
            "describe_files": lambda **kw: describe_files_impl(**kw),
            "validate_dataset": lambda **kw: validate_dataset_impl(**kw),
            
            # Data Management
            "download_dataset": lambda **kw: download_dataset_impl(**kw),
            "download_reference": lambda **kw: download_reference_impl(**kw),
            "build_index": lambda **kw: build_index_impl(**kw),
            "cleanup_data": lambda **kw: cleanup_data_impl(**kw),
            "confirm_cleanup": lambda **kw: confirm_cleanup_impl(**kw),
            
            # Workflow
            "generate_workflow": lambda **kw: generate_workflow_impl(**kw),
            "list_workflows": lambda **kw: list_workflows_impl(**kw),
            "check_references": lambda **kw: check_references_impl(**kw),
            "visualize_workflow": lambda **kw: visualize_workflow_impl(**kw),
            
            # Execution
            "submit_job": lambda **kw: submit_job_impl(**kw),
            "get_job_status": lambda **kw: get_job_status_impl(**kw),
            "get_logs": lambda **kw: get_logs_impl(**kw),
            "cancel_job": lambda **kw: cancel_job_impl(**kw),
            
            # Diagnostics
            "diagnose_error": lambda **kw: diagnose_error_impl(**kw),
            "analyze_results": lambda **kw: analyze_results_impl(**kw),
            
            # Education
            "explain_concept": lambda **kw: explain_concept_impl(**kw),
            "compare_samples": lambda **kw: compare_samples_impl(**kw),
            "get_help": lambda **kw: get_help_impl(),
        }
        return dispatch
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool
            
        Returns:
            ToolResult with execution results
        """
        if tool_name not in self._tool_dispatch:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"Unknown tool: {tool_name}",
                message=f"❌ Unknown tool: `{tool_name}`. Use 'help' to see available tools."
            )
        
        try:
            tool_func = self._tool_dispatch[tool_name]
            result = tool_func(**kwargs)
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=str(e),
                message=f"❌ Error executing {tool_name}: {e}"
            )
    
    def detect_tool(self, message: str) -> Optional[str]:
        """
        Detect which tool a user message is requesting.
        
        Args:
            message: User message to analyze
            
        Returns:
            Tool name if detected, None otherwise
        """
        message_lower = message.lower().strip()
        
        for tool_name, patterns in ALL_TOOL_PATTERNS:
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    return tool_name.value
        
        return None
    
    def get_openai_functions(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI function definitions for all tools.
        
        Returns:
            List of function definitions in OpenAI format
        """
        if self._function_definitions:
            return self._function_definitions
        
        self._function_definitions = [
            {
                "name": "scan_data",
                "description": "Scan the workspace for data files (FASTQ, BAM, VCF, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory to scan"},
                    },
                    "required": []
                }
            },
            {
                "name": "search_databases",
                "description": "Comprehensive parallel search across ALL public databases (ENCODE, GEO, SRA, Ensembl) plus TCGA for cancer queries. Automatically deduplicates results. Use this for broad data discovery.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language search query (e.g., 'human brain methylation', 'mouse RNA-seq liver', 'cancer DNA methylation')"},
                        "include_tcga": {"type": "boolean", "description": "Include TCGA search for cancer-related queries (default: true)"},
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "search_tcga",
                "description": "Search TCGA/GDC for cancer data. Cancer types: GBM/glioblastoma (BRAIN), BRCA (BREAST), LUAD (LUNG), COAD (COLON), PRAD (PROSTATE), KIRC (KIDNEY), LIHC (LIVER), SKCM (MELANOMA).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cancer_type": {
                            "type": "string", 
                            "description": "Cancer type: GBM/brain (glioblastoma), BRCA (breast), LUAD (lung), COAD (colon), etc."
                        },
                        "data_type": {
                            "type": "string", 
                            "description": "Data type: methylation, RNA-seq, WXS, WGS, clinical"
                        },
                    },
                    "required": []
                }
            },
            {
                "name": "download_dataset",
                "description": "Download a dataset from GEO, ENCODE, or TCGA/GDC. For TCGA, use project codes like TCGA-GBM (brain cancer), TCGA-BRCA (breast), TCGA-LUAD (lung).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string", 
                            "description": "Dataset ID: GSE* (GEO), ENCSR* (ENCODE), or TCGA-* (GDC). For brain cancer use TCGA-GBM, for breast use TCGA-BRCA."
                        },
                        "data_type": {
                            "type": "string",
                            "description": "Type of data: methylation, rnaseq, wgs, wes, clinical"
                        },
                    },
                    "required": ["dataset_id"]
                }
            },
            {
                "name": "generate_workflow",
                "description": "Generate a Nextflow/Snakemake workflow",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_type": {"type": "string", "description": "Type (rnaseq, chipseq, etc.)"},
                        "input_files": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["workflow_type"]
                }
            },
            {
                "name": "list_workflows",
                "description": "List available workflow templates",
                "parameters": {"type": "object", "properties": {}, "required": []}
            },
            {
                "name": "check_references",
                "description": "Check availability of reference genomes, annotations, and aligner indexes. Uses ReferenceManager for comprehensive status.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "organism": {"type": "string", "description": "Organism (human, mouse, rat, zebrafish)"},
                        "assembly": {"type": "string", "description": "Genome assembly (GRCh38, GRCm39, etc.)"},
                    },
                    "required": []
                }
            },
            {
                "name": "download_reference",
                "description": "Download reference genome, GTF annotation, or transcriptome from Ensembl. Supports human, mouse, rat, zebrafish.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "organism": {"type": "string", "description": "Organism: human, mouse, rat, zebrafish"},
                        "assembly": {"type": "string", "description": "Assembly: GRCh38, GRCh37, GRCm39, GRCm38, mRatBN7.2, GRCz11"},
                        "resource": {"type": "string", "description": "Resource type: genome, gtf, transcriptome"},
                    },
                    "required": ["organism", "assembly", "resource"]
                }
            },
            {
                "name": "build_index",
                "description": "Build an aligner index (STAR, Salmon, BWA, HISAT2, Kallisto) for a genome.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "aligner": {"type": "string", "description": "Aligner: star, salmon, bwa, hisat2, kallisto"},
                        "organism": {"type": "string", "description": "Organism (used if genome_path not provided)"},
                        "assembly": {"type": "string", "description": "Assembly (used if genome_path not provided)"},
                        "genome_path": {"type": "string", "description": "Path to genome FASTA (optional)"},
                        "gtf_path": {"type": "string", "description": "Path to GTF annotation (optional, recommended for STAR)"},
                    },
                    "required": ["aligner"]
                }
            },
            {
                "name": "visualize_workflow",
                "description": "Generate a DAG diagram visualization of a Nextflow workflow.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_dir": {"type": "string", "description": "Path to workflow directory"},
                        "output_format": {"type": "string", "description": "Output format: png, svg, pdf, txt"},
                    },
                    "required": []
                }
            },
            {
                "name": "submit_job",
                "description": "Submit a workflow job to SLURM",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow_path": {"type": "string", "description": "Path to workflow"},
                        "profile": {"type": "string", "enum": ["slurm", "local", "docker"]},
                    },
                    "required": []
                }
            },
            {
                "name": "get_job_status",
                "description": "Get status of SLURM jobs and/or Nextflow workflow execution. Shows progress bars and process counts for Nextflow.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string", "description": "SLURM job ID to check"},
                        "workflow_dir": {"type": "string", "description": "Nextflow workflow directory to monitor"},
                    },
                    "required": []
                }
            },
            {
                "name": "diagnose_error",
                "description": "Diagnose pipeline errors using 50+ patterns with AI-powered analysis. Supports OutOfMemory, Permission, DiskSpace, Network, SLURM, and more.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "error_text": {"type": "string", "description": "Error text to diagnose"},
                        "log_file": {"type": "string", "description": "Path to log file to analyze"},
                        "job_id": {"type": "string", "description": "SLURM job ID to collect logs from"},
                        "work_dir": {"type": "string", "description": "Nextflow work directory to scan"},
                        "auto_fix": {"type": "boolean", "description": "Attempt automatic fixes for safe operations"},
                    },
                    "required": []
                }
            },
            {
                "name": "analyze_results",
                "description": "Analyze workflow results and provide interpretation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "results_path": {"type": "string", "description": "Path to results"},
                    },
                    "required": []
                }
            },
            {
                "name": "explain_concept",
                "description": "Explain a bioinformatics concept or tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "concept": {"type": "string", "description": "Concept to explain"},
                    },
                    "required": ["concept"]
                }
            },
            {
                "name": "get_help",
                "description": "Show help information about available commands",
                "parameters": {"type": "object", "properties": {}, "required": []}
            },
        ]
        
        return self._function_definitions
    
    def get_tool_names(self) -> List[str]:
        """Get list of all available tool names."""
        return list(self._tool_dispatch.keys())
    
    def get_tool_count(self) -> int:
        """Get count of available tools."""
        return len(self._tool_dispatch)
    
    @property
    def tools(self) -> Dict[str, callable]:
        """Get the tool dispatch table (for introspection)."""
        return self._tool_dispatch


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_agent_tools: Optional[AgentTools] = None


def get_agent_tools(base_path: str = None) -> AgentTools:
    """
    Get the singleton AgentTools instance.
    
    Args:
        base_path: Base path for data operations
        
    Returns:
        AgentTools instance
    """
    global _agent_tools
    if _agent_tools is None:
        _agent_tools = AgentTools(base_path=base_path)
    return _agent_tools


def process_tool_request(tool_name_or_message: str, args_or_state=None) -> Optional[ToolResult]:
    """
    Process a tool request.
    
    Supports two calling conventions:
    1. process_tool_request(tool_name, args_dict) - Execute specific tool
    2. process_tool_request(message, app_state) - Detect and execute from message
    
    Args:
        tool_name_or_message: Tool name or user's chat message
        args_or_state: Tool arguments dict or application state
        
    Returns:
        ToolResult if a tool was executed, None otherwise
    """
    tools = get_agent_tools()
    
    # Check if first arg is a known tool name (new convention)
    if tool_name_or_message in tools._tool_dispatch:
        args = args_or_state if isinstance(args_or_state, dict) else {}
        return tools.execute_tool(tool_name_or_message, **args)
    
    # Otherwise, try to detect tool from message (old convention)
    detected = tools.detect_tool(tool_name_or_message)
    
    if detected:
        return tools.execute_tool(detected)
    
    return None


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = [
    # Base types
    "ToolResult",
    "ToolName",
    "TOOL_PATTERNS",
    
    # Registry
    "ToolRegistry",
    "get_registry",
    
    # Main interface
    "AgentTools",
    "get_agent_tools",
    "process_tool_request",
    
    # Pattern lists
    "ALL_TOOL_PATTERNS",
    
    # Knowledge base
    "CONCEPT_KNOWLEDGE",
]
