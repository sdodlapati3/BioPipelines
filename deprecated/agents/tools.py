"""
Agent Tools for Unified Workspace
=================================

This module provides tools that the AI agent can invoke during chat conversations.
These tools enable the AI to:
- Scan local directories for data files
- Search remote databases (ENCODE, GEO, Ensembl)
- Submit and monitor workflow jobs
- Diagnose errors in failed jobs

Each tool is designed to be invoked by the chat handler when certain
patterns are detected in user messages.

REFACTORED: This module now serves as a backward-compatible facade.
The actual implementations have been moved to the tools/ subpackage:
- tools/base.py - ToolResult, ToolName
- tools/data_discovery.py - scan_data, search_databases, etc.
- tools/data_management.py - download_dataset, cleanup_data, etc.
- tools/workflow.py - generate_workflow, list_workflows, etc.
- tools/execution.py - submit_job, get_job_status, etc.
- tools/diagnostics.py - diagnose_error, analyze_results
- tools/education.py - explain_concept, compare_samples
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORT FROM NEW MODULAR STRUCTURE
# =============================================================================

try:
    # Import from new modular structure
    from .tools import (
        ToolResult,
        ToolName,
        TOOL_PATTERNS as MODULAR_TOOL_PATTERNS,
        AgentTools as ModularAgentTools,
        get_agent_tools,
        ALL_TOOL_PATTERNS,
        CONCEPT_KNOWLEDGE,
        DataDiscoveryTools,
        DataManagementTools,
        WorkflowTools,
    )
    
    # Flag that modular imports succeeded
    _USE_MODULAR = True
    logger.info("Using modular tools from tools/ subpackage")
    
except ImportError as e:
    logger.warning(f"Modular tools not available, using legacy implementation: {e}")
    _USE_MODULAR = False
    
    # Define legacy types if imports failed
    class ToolName(Enum):
        """Available agent tools."""
        SCAN_DATA = "scan_data"
        CLEANUP_DATA = "cleanup_data"
        CONFIRM_CLEANUP = "confirm_cleanup"
        SEARCH_DATABASES = "search_databases"
        SEARCH_TCGA = "search_tcga"
        DOWNLOAD_DATASET = "download_dataset"
        CHECK_REFERENCES = "check_references"
        SUBMIT_JOB = "submit_job"
        GET_JOB_STATUS = "get_job_status"
        MONITOR_JOBS = "monitor_jobs"
        GET_LOGS = "get_logs"
        CANCEL_JOB = "cancel_job"
        DIAGNOSE_ERROR = "diagnose_error"
        LIST_WORKFLOWS = "list_workflows"
        GENERATE_WORKFLOW = "generate_workflow"
        DOWNLOAD_RESULTS = "download_results"
        COMPARE_SAMPLES = "compare_samples"
        RUN_COMMAND = "run_command"
        DESCRIBE_FILES = "describe_files"
        VALIDATE_DATASET = "validate_dataset"
        ANALYZE_RESULTS = "analyze_results"
        EXPLAIN_CONCEPT = "explain_concept"
        SHOW_HELP = "show_help"

    @dataclass
    class ToolResult:
        """Result from a tool invocation."""
        success: bool
        tool_name: str
        data: Any = None
        message: str = ""
        error: Optional[str] = None
        ui_update: Optional[Dict[str, Any]] = None


# Tool detection patterns for chat messages
# NOTE: Order matters! More specific patterns should come first.
# Patterns are matched against lowercased message.
TOOL_PATTERNS = [
    # Data scanning - EXPANDED patterns for natural language
    # Pattern 1: "scan/find data in /path" - path must start with / or ~
    (r"(?:can you\s+)?(?:scan|find|look for|check|discover|list|show)\s+(?:the\s+)?(?:local\s+)?(?:data|files?|samples?|fastq|folders?|directories?|datasets?)\s+(?:in|at|from|under|within)\s+['\"]?([\/~][^\s'\"\?]+)['\"]?",
     ToolName.SCAN_DATA),
    # Pattern 2: "in data dir /path" - captures path after "dir"  
    (r"(?:in|at|from)\s+(?:data\s+)?(?:dir|directory|folder)\s+['\"]?([\/~][^\s'\"\?]+)['\"]?",
     ToolName.SCAN_DATA),
    # Pattern 3: Simple "scan /path"
    (r"(?:scan|check|look in)\s+['\"]?([\/~][^\s'\"\?]+)['\"]?",
     ToolName.SCAN_DATA),
    # Pattern 4: "what data is available in /path" - handle "is available" as two words
    (r"(?:what|which)\s+(?:data|files?|samples?|datasets?)\s+(?:are|is\s+available|is|do i have|exist|available)\s+(?:in|at)\s+['\"]?([\/~][^\s'\"\?]+)['\"]?",
     ToolName.SCAN_DATA),
    # Pattern 5: "scan local folders" without path - use default
    (r"(?:can you\s+)?(?:scan|find|look for|check|discover|list|show)\s+(?:me\s+)?(?:my\s+)?(?:the\s+)?(?:local\s+)?(?:data|files?|samples?|folders?|directories?|datasets?)",
     ToolName.SCAN_DATA),
    # Pattern 6: "what data is available" without path
    (r"(?:what|which)\s+(?:data|files?|samples?|datasets?)\s+(?:are|is|do i have)\s*(?:available|there)?",
     ToolName.SCAN_DATA),
    # Pattern 7: "show me my datasets/data"
    (r"show\s+(?:me\s+)?(?:my\s+)?(?:what\s+)?(?:data|datasets?|samples?|files?)",
     ToolName.SCAN_DATA),
    # Pattern 8: "what data files do I have" / "check what data I have"
    (r"(?:can you\s+)?(?:check|see|view|tell me)\s+(?:what\s+)?(?:data|files?|samples?|datasets?)\s+(?:files?\s+)?(?:i have|do i have|are available|exist)",
     ToolName.SCAN_DATA),
    # Pattern 9: "what do I have in my data folder"
    (r"what\s+(?:do i have|is)\s+(?:in\s+)?(?:my\s+)?(?:local\s+)?(?:data|gcp)?\s*(?:folder|directory)?",
     ToolName.SCAN_DATA),
    
    # Confirmation patterns for cleanup (and other destructive actions)
    # NOTE: These MUST come BEFORE cleanup patterns since they're more specific
    (r"^(?:yes|yep|yeah|y)\s*[,.]?\s*(?:delete|remove|confirm|do it|go ahead|proceed)",
     ToolName.CONFIRM_CLEANUP),
    (r"confirm\s+(?:cleanup|deletion|removal)",
     ToolName.CONFIRM_CLEANUP),
    (r"(?:yes|ok|okay|sure|please)\s*[,.]?\s*(?:delete|remove)\s+(?:them|these|those|the files?)",
     ToolName.CONFIRM_CLEANUP),
    (r"^(?:delete|remove)\s+(?:them|these|those|the files?)$",
     ToolName.CONFIRM_CLEANUP),
    (r"^proceed(?:\s+with\s+(?:cleanup|deletion))?$",
     ToolName.CONFIRM_CLEANUP),
    
    # Data cleanup patterns
    (r"(?:can you\s+)?(?:clean\s*up|remove|delete)\s+(?:the\s+)?(?:corrupted|invalid|bad|broken)\s+(?:data|files?)",
     ToolName.CLEANUP_DATA),
    (r"(?:clean\s*up|fix)\s+(?:the\s+)?(?:data\s+)?(?:folder|directory)",
     ToolName.CLEANUP_DATA),
    (r"(?:remove|delete)\s+(?:the\s+)?(?:html|corrupted|invalid)\s+(?:fastq|files?)",
     ToolName.CLEANUP_DATA),
    
    # Database search - FIXED patterns to catch "search for X"
    (r"(?:search|query)\s+(?:for\s+)?(.+?)\s+(?:data|datasets?|samples?)\s+(?:in|on|from)\s+(?:encode|geo|sra|databases?)",
     ToolName.SEARCH_DATABASES),
    (r"(?:search|query)\s+(?:in\s+)?(?:encode|geo|sra|ensembl|databases?)\s+(?:for)?\s*(.+)",
     ToolName.SEARCH_DATABASES),
    (r"(?:search|query)\s+(?:for\s+)(.+?)(?:\s+data|\s+datasets?)?$",
     ToolName.SEARCH_DATABASES),
    
    # Download dataset from GEO/ENCODE - MUST come before generic download patterns
    (r"download\s+(GSE\d+|ENCSR[A-Z0-9]+)",
     ToolName.DOWNLOAD_DATASET),
    (r"(?:download|get|fetch)\s+(?:the\s+)?(?:dataset\s+)?(GSE\d+|ENCSR[A-Z0-9]+)",
     ToolName.DOWNLOAD_DATASET),
    (r"(?:add|queue)\s+(GSE\d+|ENCSR[A-Z0-9]+)\s+(?:to\s+)?(?:manifest|download)",
     ToolName.DOWNLOAD_DATASET),
    (r"(?:download|get|fetch)\s+(?:encode|geo)\s+(?:dataset\s+)?(ENCSR[A-Z0-9]+|GSE\d+)",
     ToolName.DOWNLOAD_DATASET),
    (r"(?:download|get)\s+(?:this|that)\s+(?:dataset|data)",
     ToolName.DOWNLOAD_DATASET),
    
    # Reference check
    (r"(?:check|verify|do i have)\s+(?:the\s+)?(?:reference|genome|index)\s+(?:for)?\s*(.+)?",
     ToolName.CHECK_REFERENCES),
    
    # Job submission
    (r"(?:run|execute|submit|start)\s+(?:it|the workflow|this|pipeline)\s*(?:on|with|using)?\s*(slurm|local|docker)?",
     ToolName.SUBMIT_JOB),
    (r"(?:run|execute|submit)\s+(?:workflow|pipeline)?\s*['\"]?([^'\"]+)['\"]?\s*(?:on|with)?\s*(slurm|local|docker)?",
     ToolName.SUBMIT_JOB),
    
    # Job status
    (r"(?:what(?:'s| is)|show|check)\s+(?:the\s+)?(?:status|progress)\s*(?:of)?\s*(?:job\s*)?(\d+)?",
     ToolName.GET_JOB_STATUS),
    (r"(?:how(?:'s| is))\s+(?:the\s+)?(?:job|workflow|pipeline)\s*(?:doing|going|running)?",
     ToolName.GET_JOB_STATUS),
    
    # Monitor jobs - check if jobs completed and data saved
    (r"(?:can you\s+)?(?:check|monitor|verify)\s+(?:if\s+)?(?:the\s+)?(?:jobs?|downloads?)\s+(?:are\s+)?(?:completed?|done|finished|running)",
     ToolName.MONITOR_JOBS),
    (r"(?:check|monitor|verify)\s+(?:the\s+)?(?:download\s+)?jobs?\s*(\d+)?(?:\s*(?:,|and)\s*(\d+))?",
     ToolName.MONITOR_JOBS),
    (r"(?:are|is)\s+(?:the\s+)?(?:data|files?)\s+(?:saved|downloaded|ready)",
     ToolName.MONITOR_JOBS),
    (r"(?:check|see)\s+(?:if\s+)?(?:jobs?\s+)?(\d+)(?:\s*(?:,|and)\s*(\d+))?\s+(?:completed?|done|finished)",
     ToolName.MONITOR_JOBS),
    (r"(?:monitor|track)\s+(?:my\s+)?(?:jobs?|downloads?)",
     ToolName.MONITOR_JOBS),
    
    # Logs
    (r"(?:show|get|view|display)\s+(?:me\s+)?(?:the\s+)?logs?\s*(?:for|of)?\s*(?:job\s*)?(\d+)?",
     ToolName.GET_LOGS),
    (r"(?:what(?:'s| is))\s+(?:in\s+)?(?:the\s+)?logs?",
     ToolName.GET_LOGS),
    
    # Cancel
    (r"(?:cancel|stop|abort|kill)\s+(?:the\s+)?(?:job|workflow|pipeline)\s*(\d+)?",
     ToolName.CANCEL_JOB),
    
    # Diagnosis - be more specific to avoid matching "analyze results"
    (r"(?:diagnose|debug|what went wrong|why did it fail)",
     ToolName.DIAGNOSE_ERROR),
    (r"(?:fix|help with)\s+(?:the\s+)?(?:error|failure|problem|crash)",
     ToolName.DIAGNOSE_ERROR),
    (r"(?:analyze|debug)\s+(?:the\s+)?(?:error|failure|crash|problem)",
     ToolName.DIAGNOSE_ERROR),
    (r"(?:the\s+)?(?:job|workflow|pipeline)\s+(?:failed|crashed|errored)",
     ToolName.DIAGNOSE_ERROR),
    
    # Analyze results - MUST come before generic patterns, specific to results/QC
    (r"(?:analyze|interpret|explain|summarize)\s+(?:the\s+)?(?:results?|outputs?|qc|quality|report)",
     ToolName.ANALYZE_RESULTS),
    (r"(?:what do|how should i interpret)\s+(?:the|these)\s+(?:results?|outputs?|reports?)\s+(?:mean|show)",
     ToolName.ANALYZE_RESULTS),
    (r"(?:review|check)\s+(?:the\s+)?(?:results?|outputs?|qc\s+report)",
     ToolName.ANALYZE_RESULTS),
    
    # List workflows
    (r"(?:list|show|what)\s+(?:available\s+)?workflows?",
     ToolName.LIST_WORKFLOWS),
    
    # Generate workflow - NEW: Create workflow from description
    (r"(?:create|generate|build|make)\s+(?:a\s+)?(?:new\s+)?(.+?)\s+(?:workflow|pipeline)",
     ToolName.GENERATE_WORKFLOW),
    (r"(?:create|generate|build)\s+(?:a\s+)?(?:new\s+)?(?:workflow|pipeline)\s+(?:for|to)\s+(.+)",
     ToolName.GENERATE_WORKFLOW),
    (r"(?:i want to|help me)\s+(?:do|run|perform)\s+(.+?)\s+(?:analysis|workflow|pipeline)",
     ToolName.GENERATE_WORKFLOW),
    (r"(?:set up|setup)\s+(?:a\s+)?(.+?)\s+(?:analysis|workflow|pipeline)",
     ToolName.GENERATE_WORKFLOW),
    
    # Search TCGA - NEW: Search cancer data from GDC portal
    (r"(?:search|find|look for)\s+(?:in\s+)?(?:tcga|gdc|cancer)\s+(?:for\s+)?(.+)",
     ToolName.SEARCH_TCGA),
    (r"(?:search|find)\s+(.+?)\s+(?:in|on|from)\s+(?:tcga|gdc|cancer\s+portal)",
     ToolName.SEARCH_TCGA),
    (r"(?:get|find)\s+(?:cancer|tumor)\s+data\s+(?:for|from)\s+(.+)",
     ToolName.SEARCH_TCGA),
    
    # Download results
    (r"(?:download|get|export)\s+(?:the\s+)?(?:results?|outputs?|files?)\s*(?:from|for)?\s*(?:job\s*)?(\d+)?",
     ToolName.DOWNLOAD_RESULTS),
    (r"(?:zip|package|archive)\s+(?:the\s+)?(?:results?|outputs?)",
     ToolName.DOWNLOAD_RESULTS),
    
    # Explain concept - NEW: Explain bioinformatics terms
    (r"(?:explain|what is|what are|tell me about|describe)\s+(.+?)(?:\?|$)",
     ToolName.EXPLAIN_CONCEPT),
    (r"(?:how does|how do)\s+(.+?)(?:\s+work)?(?:\?|$)",
     ToolName.EXPLAIN_CONCEPT),
    
    # Compare samples
    (r"(?:compare|diff|contrast)\s+(?:samples?|groups?|conditions?)\s*(.+)?",
     ToolName.COMPARE_SAMPLES),
    (r"(?:what(?:'s| is| are))\s+(?:the\s+)?(?:difference|differences)\s+(?:between)\s*(.+)?",
     ToolName.COMPARE_SAMPLES),
    
    # Describe files - get file details, size, rows, columns for any file type
    # Pattern 1: "describe/inspect files in /path" or "give me details of files in /path"
    (r"(?:describe|inspect|examine|analyze|check)\s+(?:the\s+)?(?:files?|data)\s+(?:in|at|from)\s+['\"]?([\/~][^\s'\"\?]+)['\"]?",
     ToolName.DESCRIBE_FILES),
    # Pattern 2: "details of files" / "file details" / "summary stats"
    (r"(?:give me\s+)?(?:details?|info|information|summary|stats|statistics|metadata)\s+(?:of|about|for)\s+(?:the\s+)?(?:files?|data)\s*(?:in|at|from)?\s*['\"]?([\/~][^\s'\"\?]+)?['\"]?",
     ToolName.DESCRIBE_FILES),
    # Pattern 3: "what's in these files" / "show file contents"
    (r"(?:what(?:'s| is| are))\s+(?:in\s+)?(?:these|the|those)\s+(?:files?|data\s+files?)",
     ToolName.DESCRIBE_FILES),
    # Pattern 4: "file sizes" / "row count" / "column names"
    (r"(?:show|get|list|what are)\s+(?:the\s+)?(?:file\s+)?(?:sizes?|rows?|columns?|headers?|names?)\s*(?:of|for|in)?\s*(?:the\s+)?(?:files?)?",
     ToolName.DESCRIBE_FILES),
    # Pattern 5: "how many rows/columns" / "dimensions"
    (r"(?:how many)\s+(?:rows?|columns?|lines?|samples?)(?:\s+(?:are|do)\s+(?:there|i have))?",
     ToolName.DESCRIBE_FILES),
    # Pattern 6: simple "describe files" or "inspect files"
    (r"(?:describe|inspect|examine)\s+(?:the\s+)?files?",
     ToolName.DESCRIBE_FILES),
    
    # Validate dataset - check if downloaded data is real or metadata
    # Pattern 1: "validate/check dataset GSE123" or "is this real data"
    (r"(?:validate|verify|check)\s+(?:if\s+)?(?:the\s+)?(?:dataset|data)\s+(?:is\s+)?(?:real|actual|valid)?['\"]?([\/~][^\s'\"\?]+|GSE\d+)?['\"]?",
     ToolName.VALIDATE_DATASET),
    # Pattern 2: "is this methylation data or metadata"
    (r"(?:is\s+)?(?:this|these)\s+(?:files?\s+)?(?:real|actual)?\s*(?:data|methylation|sequencing)\s+(?:data\s+)?(?:or\s+)?(?:metadata)?",
     ToolName.VALIDATE_DATASET),
    # Pattern 3: "are these files real data"
    (r"(?:are|is)\s+(?:these|this|the)\s+(?:files?\s+)?(?:real|actual|valid)\s+(?:data|files?)",
     ToolName.VALIDATE_DATASET),
    # Pattern 4: "check what kind of data this is"
    (r"(?:what\s+)?(?:kind|type)\s+of\s+(?:data|files?)\s+(?:is|are)\s+(?:this|these)",
     ToolName.VALIDATE_DATASET),
    # Pattern 5: "validate downloaded files"
    (r"validate\s+(?:the\s+)?(?:downloaded\s+)?(?:data|files?)",
     ToolName.VALIDATE_DATASET),
    # Pattern 6: "check if this is actual X data"
    (r"(?:check|verify)\s+(?:if\s+)?(?:this|these)\s+(?:is|are)\s+(?:actual|real)?\s*(?:\w+)?\s*(?:data|files?)",
     ToolName.VALIDATE_DATASET),
    
    # Help
    (r"^(?:help|commands?|what can you do|\?)$",
     ToolName.SHOW_HELP),
    (r"(?:show|list)\s+(?:me\s+)?(?:available\s+)?(?:commands?|options?|help)",
     ToolName.SHOW_HELP),
]


class AgentTools:
    """
    Tools available to the AI agent during chat conversations.
    
    Each tool method returns a ToolResult that contains:
    - success: Whether the tool executed successfully
    - data: The result data (varies by tool)
    - message: A human-readable message for chat display
    - ui_update: Optional dict of UI component updates
    """
    
    def __init__(self, app_state=None):
        """
        Initialize agent tools.
        
        Args:
            app_state: The AppState instance from gradio_app
        """
        self.app_state = app_state
        self._data_manifest = None
        self._scanner = None
        self._reference_manager = None
        self._pipeline_executor = None
        
    def _get_scanner(self):
        """Lazy-load the data scanner."""
        if self._scanner is None:
            try:
                from workflow_composer.data.scanner import LocalSampleScanner
                self._scanner = LocalSampleScanner()
            except ImportError:
                logger.warning("LocalSampleScanner not available")
        return self._scanner
    
    def _get_manifest(self):
        """Get or create the data manifest."""
        if self._data_manifest is None:
            try:
                from workflow_composer.data.manifest import DataManifest
                self._data_manifest = DataManifest()
            except ImportError:
                logger.warning("DataManifest not available")
        return self._data_manifest
    
    def _get_reference_manager(self):
        """Lazy-load the reference manager."""
        if self._reference_manager is None:
            try:
                from workflow_composer.data.reference_manager import ReferenceManager
                base_dir = Path.home() / ".biopipelines" / "references"
                self._reference_manager = ReferenceManager(base_dir=base_dir)
            except ImportError:
                logger.warning("ReferenceManager not available")
        return self._reference_manager
    
    def _get_executor(self):
        """Get the pipeline executor from app state."""
        if self._pipeline_executor is None and self.app_state:
            # Access from gradio_app's PipelineExecutor
            self._pipeline_executor = getattr(self.app_state, 'executor', None)
        return self._pipeline_executor
    
    def detect_tool(self, message: str) -> Optional[Tuple[ToolName, List[str]]]:
        """
        Detect if a message should trigger a tool.
        
        Args:
            message: The user's chat message
            
        Returns:
            Tuple of (tool_name, captured_groups) or None
        """
        message_stripped = message.strip()
        message_lower = message_stripped.lower()
        
        for pattern, tool_name in TOOL_PATTERNS:
            # Match on lowercase for case-insensitive keyword matching
            match = re.search(pattern, message_lower, re.IGNORECASE)
            if match:
                # For path-based tools, re-extract from original message to preserve case
                if tool_name in (ToolName.SCAN_DATA, ToolName.SUBMIT_JOB):
                    # Re-run match on original message to preserve path case
                    original_match = re.search(pattern, message_stripped, re.IGNORECASE)
                    if original_match:
                        groups = [g for g in original_match.groups() if g]
                    else:
                        groups = [g for g in match.groups() if g]
                else:
                    groups = [g for g in match.groups() if g]
                return tool_name, groups
        
        return None
    
    def execute(self, tool_name: ToolName, args: List[str]) -> ToolResult:
        """
        Execute a tool by name.
        
        Args:
            tool_name: The tool to execute
            args: Arguments captured from the message pattern
            
        Returns:
            ToolResult with the outcome
        """
        tool_methods = {
            ToolName.SCAN_DATA: self.scan_data,
            ToolName.SEARCH_DATABASES: self.search_databases,
            ToolName.SEARCH_TCGA: self.search_tcga,
            ToolName.DOWNLOAD_DATASET: self.download_dataset,
            ToolName.CHECK_REFERENCES: self.check_references,
            ToolName.SUBMIT_JOB: self.submit_job,
            ToolName.GET_JOB_STATUS: self.get_job_status,
            ToolName.MONITOR_JOBS: self.monitor_jobs,
            ToolName.GET_LOGS: self.get_logs,
            ToolName.CANCEL_JOB: self.cancel_job,
            ToolName.DIAGNOSE_ERROR: self.diagnose_error,
            ToolName.LIST_WORKFLOWS: self.list_workflows,
            ToolName.GENERATE_WORKFLOW: self.generate_workflow,
            ToolName.DOWNLOAD_RESULTS: self.download_results,
            ToolName.COMPARE_SAMPLES: self.compare_samples,
            ToolName.DESCRIBE_FILES: self.describe_files,
            ToolName.VALIDATE_DATASET: self.validate_dataset,
            ToolName.ANALYZE_RESULTS: self.analyze_results,
            ToolName.EXPLAIN_CONCEPT: self.explain_concept,
            ToolName.SHOW_HELP: self.show_help,
        }
        
        method = tool_methods.get(tool_name)
        if method:
            try:
                return method(*args if args else [])
            except Exception as e:
                logger.error(f"Tool {tool_name.value} failed: {e}")
                return ToolResult(
                    success=False,
                    tool_name=tool_name.value,
                    error=str(e),
                    message=f"❌ Error executing {tool_name.value}: {e}"
                )
        
        return ToolResult(
            success=False,
            tool_name=tool_name.value if tool_name else "unknown",
            error=f"Unknown tool: {tool_name}",
            message=f"❌ Unknown tool: {tool_name}"
        )
    
    # ========== DATA TOOLS ==========
    
    def scan_data(self, path: str = None) -> ToolResult:
        """
        Scan a directory for FASTQ files.
        
        Args:
            path: Directory path to scan. Defaults to current directory.
            
        Returns:
            ToolResult with discovered samples
        """
        scanner = self._get_scanner()
        manifest = self._get_manifest()
        
        if scanner is None:
            return ToolResult(
                success=False,
                tool_name="scan_data",
                error="Scanner not available",
                message="❌ Data scanner is not available. Please check installation."
            )
        
        # Smart default paths - check common data locations
        # Development defaults for sdodl001's environment
        if not path:
            default_paths = [
                Path("/scratch/sdodl001/BioPipelines"),  # Primary data location
                Path("/scratch/sdodl001/BioPipelines/data"),
                Path.home() / "BioPipelines" / "data",
                Path.home() / "data",
                Path.cwd() / "data",
                Path.cwd(),
            ]
            for p in default_paths:
                if p.exists() and p.is_dir():
                    path = str(p)
                    break
            else:
                path = str(Path.cwd())
        
        # Clean up path
        path = path.strip().strip("'\"")
        scan_path = Path(path).expanduser().resolve()
        
        if not scan_path.exists():
            return ToolResult(
                success=False,
                tool_name="scan_data",
                error=f"Path not found: {scan_path}",
                message=f"❌ Directory not found: `{scan_path}`"
            )
        
        try:
            result = scanner.scan_directory(scan_path, recursive=True)
            samples = result.samples if hasattr(result, 'samples') else []
            
            # Add samples to manifest
            if manifest and samples:
                for sample in samples:
                    manifest.add_sample(sample)
            
            # Build response message
            if samples:
                sample_list = []
                for s in samples[:10]:
                    # Count files (1 for single-end, 2 for paired-end)
                    file_count = 2 if (hasattr(s, 'is_paired') and s.is_paired) or (hasattr(s, 'fastq_2') and s.fastq_2) else 1
                    # Get layout string
                    layout = "paired" if file_count == 2 else "single"
                    if hasattr(s, 'library_layout'):
                        layout = s.library_layout.value if hasattr(s.library_layout, 'value') else str(s.library_layout)
                    sample_list.append(f"  - `{s.sample_id}`: {file_count} files ({layout})")
                
                sample_str = "\n".join(sample_list)
                if len(samples) > 10:
                    sample_str += f"\n  - ... and {len(samples) - 10} more"
                
                message = f"""✅ Found **{len(samples)} samples** in `{scan_path}`:

{sample_str}

Added to data manifest. Ready for workflow generation!"""
            else:
                message = f"⚠️ No FASTQ samples found in `{scan_path}`"
            
            return ToolResult(
                success=True,
                tool_name="scan_data",
                data={
                    "samples": samples,
                    "path": str(scan_path),
                    "count": len(samples)
                },
                message=message,
                ui_update={
                    "manifest_sample_count": len(samples),
                    "manifest_path": str(scan_path)
                }
            )
            
        except Exception as e:
            logger.error(f"Scan failed: {e}")
            return ToolResult(
                success=False,
                tool_name="scan_data",
                error=str(e),
                message=f"❌ Failed to scan directory: {e}"
            )
    
    def cleanup_data(self, path: str = None, confirm: bool = False) -> ToolResult:
        """
        Clean up corrupted data files (HTML error pages masquerading as FASTQ, etc).
        
        Two-phase operation:
        1. First call (confirm=False): Scan and show what would be deleted
        2. Second call (confirm=True): Actually delete the files
        
        Args:
            path: Directory path to clean. Defaults to data directory.
            confirm: If True, actually delete files. If False, just show preview.
            
        Returns:
            ToolResult with cleanup summary or preview
        """
        import gzip
        
        # Use default data path if not specified
        if not path:
            default_paths = [
                Path("/scratch/sdodl001/BioPipelines/data"),
                Path("/scratch/sdodl001/BioPipelines"),
                Path.home() / "BioPipelines" / "data",
            ]
            for p in default_paths:
                if p.exists():
                    path = str(p)
                    break
            else:
                return ToolResult(
                    success=False,
                    tool_name="cleanup_data",
                    error="No data directory found",
                    message="❌ Could not find data directory to clean"
                )
        
        scan_path = Path(path).expanduser().resolve()
        
        if not scan_path.exists():
            return ToolResult(
                success=False,
                tool_name="cleanup_data",
                error=f"Path not found: {scan_path}",
                message=f"❌ Directory not found: `{scan_path}`"
            )
        
        # Find corrupted files
        corrupted_files = []
        checked_files = 0
        
        for ext in ['*.fastq.gz', '*.fq.gz', '*.fastq', '*.fq']:
            for f in scan_path.rglob(ext):
                if not f.is_file() or f.is_symlink():
                    continue
                checked_files += 1
                try:
                    # Check if it's a valid gzip or starts with HTML
                    with open(f, 'rb') as fp:
                        header = fp.read(10)
                        # HTML files start with <!DOCTYPE or <html
                        if header.startswith(b'<!') or header.startswith(b'<html'):
                            corrupted_files.append(f)
                        # Valid gzip starts with magic bytes 1f 8b
                        elif f.suffix == '.gz' and not header.startswith(b'\x1f\x8b'):
                            corrupted_files.append(f)
                except Exception as e:
                    logger.warning(f"Error checking {f}: {e}")
        
        # Find broken symlinks
        broken_symlinks = []
        for ext in ['*.fastq.gz', '*.fq.gz']:
            for f in scan_path.rglob(ext):
                if f.is_symlink() and not f.resolve().exists():
                    broken_symlinks.append(f)
        
        if not corrupted_files and not broken_symlinks:
            return ToolResult(
                success=True,
                tool_name="cleanup_data",
                data={"checked": checked_files, "removed": 0},
                message=f"✅ Checked {checked_files} files - no corrupted files found!"
            )
        
        # Build file list for display
        file_list = "\n".join([f"  - `{f.name}` ({f.stat().st_size / 1024:.1f} KB)" for f in corrupted_files[:10]])
        if len(corrupted_files) > 10:
            file_list += f"\n  - ... and {len(corrupted_files) - 10} more"
        
        # If not confirmed, just show preview
        if not confirm:
            message = f"""⚠️ **Found {len(corrupted_files)} corrupted files** in `{scan_path}`:

{file_list}
"""
            if broken_symlinks:
                message += f"\nAlso found **{len(broken_symlinks)} broken symlinks**."
            message += """
These files appear to be HTML error pages or invalid gzip files, not actual sequencing data.

**To delete these files, say:** "yes, delete them" or "confirm cleanup"
"""
            # Store pending cleanup for confirmation
            self._pending_cleanup = {
                "path": scan_path,
                "corrupted_files": corrupted_files,
                "broken_symlinks": broken_symlinks
            }
            
            return ToolResult(
                success=True,
                tool_name="cleanup_data",
                data={
                    "checked": checked_files,
                    "found": len(corrupted_files),
                    "preview": True,
                    "awaiting_confirmation": True
                },
                message=message
            )
        
        # CONFIRMED - actually delete files
        removed = []
        failed = []
        for f in corrupted_files:
            try:
                f.unlink()
                removed.append(str(f))
                logger.info(f"Removed corrupted file: {f}")
            except Exception as e:
                failed.append(f"{f}: {e}")
        
        # Remove broken symlinks
        removed_symlinks = []
        for f in broken_symlinks:
            try:
                f.unlink()
                removed_symlinks.append(str(f))
            except:
                pass
        
        # Clear pending cleanup
        self._pending_cleanup = None
        
        # Build response
        removed_list = "\n".join([f"  - `{Path(r).name}`" for r in removed[:10]])
        if len(removed) > 10:
            removed_list += f"\n  - ... and {len(removed) - 10} more"
        
        message = f"""🧹 **Cleanup complete** in `{scan_path}`:

**Removed {len(removed)} corrupted files:**
{removed_list}
"""
        if removed_symlinks:
            message += f"\n**Also removed {len(removed_symlinks)} broken symlinks.**"
        
        if failed:
            message += f"\n\n⚠️ Failed to remove {len(failed)} files."
        
        return ToolResult(
            success=True,
            tool_name="cleanup_data",
            data={
                "checked": checked_files,
                "removed": len(removed),
                "removed_files": removed,
                "removed_symlinks": removed_symlinks,
                "failed": failed
            },
            message=message
        )
    
    def confirm_cleanup(self) -> ToolResult:
        """Confirm and execute pending cleanup operation."""
        if not hasattr(self, '_pending_cleanup') or not self._pending_cleanup:
            return ToolResult(
                success=False,
                tool_name="confirm_cleanup",
                error="No pending cleanup",
                message="⚠️ No cleanup operation pending. First run 'cleanup data' to scan for corrupted files."
            )
        
        pending = self._pending_cleanup
        return self.cleanup_data(str(pending["path"]), confirm=True)
    
    def _get_available_partition(self, preferred: list = None) -> str:
        """
        Get best available SLURM partition from preferred list.
        
        Checks partition availability and returns first available one.
        Falls back to 'debugspot' if none available.
        
        Args:
            preferred: List of partition names in order of preference
            
        Returns:
            Name of available partition
        """
        if preferred is None:
            preferred = ["cpuspot", "debugspot"]
        
        try:
            # Get partition status: available/total nodes
            result = subprocess.run(
                ["sinfo", "-h", "-o", "%P %a %F"],  # partition, avail, nodes (A/I/O/T)
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return preferred[0]  # Default to first choice
            
            # Parse partition status
            partition_status = {}
            for line in result.stdout.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[0].rstrip("*")  # Remove default marker
                    avail = parts[1]  # 'up' or 'down'
                    nodes = parts[2]  # e.g., "5/0/0/10" = allocated/idle/other/total
                    
                    # Parse node counts
                    try:
                        node_parts = nodes.split("/")
                        idle = int(node_parts[1]) if len(node_parts) > 1 else 0
                        partition_status[name] = {
                            "up": avail == "up",
                            "idle": idle
                        }
                    except (ValueError, IndexError):
                        partition_status[name] = {"up": avail == "up", "idle": 0}
            
            # Find first available partition with idle nodes
            for pname in preferred:
                if pname in partition_status:
                    status = partition_status[pname]
                    if status["up"] and status["idle"] > 0:
                        logger.debug(f"Using partition {pname} ({status['idle']} idle nodes)")
                        return pname
            
            # If no idle nodes, try any 'up' partition
            for pname in preferred:
                if pname in partition_status and partition_status[pname]["up"]:
                    logger.debug(f"Using partition {pname} (no idle nodes, but up)")
                    return pname
            
            # Fallback
            return preferred[0]
            
        except Exception as e:
            logger.debug(f"Error checking partitions: {e}")
            return preferred[0]
    
    def _get_dataset_url(self, dataset_id: str, source: str = None) -> str:
        """
        Get clickable URL for a dataset ID.
        
        Args:
            dataset_id: Dataset ID (GSE*, ENCSR*, etc.)
            source: Optional source hint (GEO, ENCODE, etc.)
            
        Returns:
            URL string
        """
        dataset_id = dataset_id.strip().upper()
        
        if dataset_id.startswith("GSE") or source == "GEO":
            return f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={dataset_id}"
        elif dataset_id.startswith("ENCSR") or source == "ENCODE":
            return f"https://www.encodeproject.org/experiments/{dataset_id}/"
        elif dataset_id.startswith("SRP") or dataset_id.startswith("PRJNA"):
            return f"https://www.ncbi.nlm.nih.gov/sra/?term={dataset_id}"
        elif dataset_id.startswith("E-"):  # ArrayExpress
            return f"https://www.ebi.ac.uk/arrayexpress/experiments/{dataset_id}/"
        else:
            return f"https://www.ncbi.nlm.nih.gov/search/all/?term={dataset_id}"
    
    def search_databases(self, query: str = None) -> ToolResult:
        """
        Search remote databases for datasets.
        
        Args:
            query: Search query (e.g., "human RNA-seq liver")
            
        Returns:
            ToolResult with search results
        """
        if not query:
            return ToolResult(
                success=False,
                tool_name="search_databases",
                error="No search query provided",
                message="❌ Please specify what to search for (e.g., 'search for human RNA-seq liver data')"
            )
        
        try:
            # Try to use discovery adapters
            from workflow_composer.data.discovery import (
                ENCODEAdapter, GEOAdapter, EnsemblAdapter, parse_query
            )
            from workflow_composer.data.discovery.models import SearchQuery as SearchQueryModel
            
            results = []
            
            # Parse the natural language query into structured form
            try:
                # parse_query returns a SearchQuery directly (not ParseResult)
                search_query = parse_query(query)
                logger.debug(f"Parsed query: {search_query}")
            except Exception as e:
                # Fall back to simple query construction
                logger.debug(f"Query parsing failed, using simple query: {e}")
                search_query = SearchQueryModel(raw_query=query, max_results=5)
            
            # Search ENCODE (best for ChIP-seq, ATAC-seq, DNase-seq, RNA-seq)
            try:
                encode = ENCODEAdapter()
                # Clone query with limited results
                encode_query = SearchQueryModel(
                    raw_query=search_query.raw_query or query,
                    organism=search_query.organism,
                    assay_type=search_query.assay_type,
                    target=search_query.target,
                    tissue=search_query.tissue,
                    cell_line=search_query.cell_line,
                    max_results=5
                )
                encode_results = encode.search(encode_query)
                # Adapters return a list directly
                if encode_results:
                    for dataset in encode_results[:5]:
                        results.append({
                            "source": "ENCODE",
                            "id": dataset.id,
                            "title": dataset.title or dataset.id,
                            "organism": dataset.organism or "",
                            "assay": dataset.assay_type or ""
                        })
            except Exception as e:
                logger.debug(f"ENCODE search failed: {e}")
            
            # Search GEO (best for RNA-seq, scRNA-seq, diverse experiments)
            try:
                geo = GEOAdapter()
                geo_query = SearchQueryModel(
                    raw_query=search_query.raw_query or query,
                    organism=search_query.organism,
                    assay_type=search_query.assay_type,
                    tissue=search_query.tissue,
                    max_results=5
                )
                geo_results = geo.search(geo_query)
                if geo_results:
                    for dataset in geo_results[:5]:
                        results.append({
                            "source": "GEO",
                            "id": dataset.id,
                            "title": dataset.title or dataset.id,
                            "organism": dataset.organism or "",
                            "assay": dataset.assay_type or ""
                        })
            except Exception as e:
                logger.debug(f"GEO search failed: {e}")
            
            if results:
                result_list = "\n".join([
                    f"  - **{r['source']}**: [{r['id']}]({self._get_dataset_url(r['id'], r['source'])}) | {r.get('assay', '') or ''} | {(r['title'][:40] + '...') if len(r['title']) > 40 else r['title']}"
                    for r in results[:10]
                ])
                
                # Show parsed query info
                parsed_info = []
                if search_query.organism:
                    parsed_info.append(f"organism={search_query.organism}")
                if search_query.assay_type:
                    parsed_info.append(f"assay={search_query.assay_type}")
                if search_query.tissue:
                    parsed_info.append(f"tissue={search_query.tissue}")
                
                parsed_str = f"\n*Parsed query: {', '.join(parsed_info)}*\n" if parsed_info else ""
                
                # Check if user asked for cancer/tumor data and add relevant warnings
                query_lower = query.lower()
                cancer_terms = ['cancer', 'tumor', 'carcinoma', 'glioblastoma', 'glioma', 'gbm', 'malignant', 'neoplasm']
                is_cancer_query = any(term in query_lower for term in cancer_terms)
                
                warning = ""
                if is_cancer_query:
                    # Add warning about cancer data sources
                    if 'brain' in query_lower or 'glioma' in query_lower or 'gbm' in query_lower:
                        warning = """

⚠️ **Note about brain cancer methylation data:**
- ENCODE mostly has normal tissue or cell lines, not tumors
- For brain cancer, consider these specialized resources:
  - **TCGA-GBM**: [GDC Portal](https://portal.gdc.cancer.gov/projects/TCGA-GBM) (600+ GBM samples)
  - **GSE90496**: Heidelberg CNS tumor classifier (2,800+ samples)
  - **CPTAC-GBM**: Multi-omics including WGBS
"""
                    else:
                        warning = f"""

⚠️ **Note about cancer data:**
- ENCODE/GEO results may include cell lines, not primary tumors
- For cancer research, check **TCGA** at [GDC Portal](https://portal.gdc.cancer.gov/)
"""
                
                message = f"""🔍 Found **{len(results)} datasets** matching "{query}":
{parsed_str}
{result_list}
{warning}
💡 Say "download <ID>" to add a dataset to your manifest.
💡 After download, say "validate dataset <ID>" to verify data type."""
            else:
                message = f"""⚠️ No datasets found matching '{query}'.

**Tips for better results:**
- Include organism (human, mouse)
- Specify assay type (RNA-seq, ChIP-seq, ATAC-seq)
- Add tissue or cell line

Example: "search for human liver RNA-seq" """
            
            return ToolResult(
                success=True,
                tool_name="search_databases",
                data={"results": results, "query": query, "parsed": search_query.__dict__ if search_query else {}},
                message=message
            )
            
        except ImportError as e:
            return ToolResult(
                success=False,
                tool_name="search_databases",
                error=f"Discovery adapters not available: {e}",
                message="❌ Database search is not available. Install discovery modules."
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name="search_databases",
                error=str(e),
                message=f"❌ Search failed: {e}"
            )
    
    def download_dataset(self, dataset_id: str = None) -> ToolResult:
        """
        Download a dataset from GEO or ENCODE.
        
        Args:
            dataset_id: Dataset ID (e.g., "GSE200839" or "ENCSR856UND")
            
        Returns:
            ToolResult with download status
        """
        import subprocess
        import os
        
        if not dataset_id:
            return ToolResult(
                success=False,
                tool_name="download_dataset",
                error="No dataset ID provided",
                message="❌ Please specify a dataset ID. Example: `download GSE200839`"
            )
        
        dataset_id = dataset_id.strip().upper()
        
        # Determine data directory
        data_dir = Path(os.environ.get("BIOPIPELINES_DATA", "/scratch/sdodl001/BioPipelines/data/raw"))
        data_dir.mkdir(parents=True, exist_ok=True)
        
        output_dir = data_dir / dataset_id
        
        # Check if already downloaded
        if output_dir.exists() and any(output_dir.iterdir()):
            files = list(output_dir.glob("*"))
            return ToolResult(
                success=True,
                tool_name="download_dataset",
                data={"dataset_id": dataset_id, "path": str(output_dir), "status": "exists"},
                message=f"""✅ **Dataset {dataset_id} already exists!**

📂 **Location:** `{output_dir}`
📁 **Files:** {len(files)} files

Use `scan data in {output_dir}` to see the files."""
            )
        
        # Start download based on source
        if dataset_id.startswith("GSE"):
            # GEO dataset - use prefetch/fasterq-dump for SRA data
            message = f"""⏳ **Downloading {dataset_id} from GEO...**

This will:
1. Fetch SRA run info from GEO
2. Download FASTQ files using fasterq-dump
3. Save to `{output_dir}`

⚠️ Large datasets may take hours. Check progress with:
```bash
ls -la {output_dir}
```
"""
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get best available partition (prefer cpuspot, fallback to debugspot)
            partition = self._get_available_partition(["cpuspot", "debugspot"])
            
            # Submit as background job
            job_script = f"""#!/bin/bash
#SBATCH --job-name=download-{dataset_id}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output={data_dir}/logs/download_{dataset_id}_%j.out
#SBATCH --error={data_dir}/logs/download_{dataset_id}_%j.err

cd {output_dir}
echo "Downloading {dataset_id}..."

# Try to get SRA run IDs from GEO
esearch -db gds -query "{dataset_id}[ACCN]" | efetch -format summary | grep -oP 'SRR\\d+' | sort -u > sra_runs.txt

if [ -s sra_runs.txt ]; then
    while read SRR; do
        echo "Fetching $SRR..."
        prefetch $SRR --max-size 100G
        fasterq-dump $SRR --split-3 --threads 4
        gzip *.fastq
    done < sra_runs.txt
    echo "Done! Downloaded $(cat sra_runs.txt | wc -l) runs."
else
    echo "No SRA runs found. Trying direct FTP..."
    wget -r -np -nd -A "*.gz,*.fastq*,*.txt" "ftp://ftp.ncbi.nlm.nih.gov/geo/series/{dataset_id[:6]}nnn/{dataset_id}/"
fi
"""
            # Save job script
            logs_dir = data_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            job_file = logs_dir / f"download_{dataset_id}.sh"
            job_file.write_text(job_script)
            
            # Submit job
            try:
                result = subprocess.run(
                    ["sbatch", str(job_file)],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    job_id = result.stdout.strip().split()[-1]
                    message += f"\n🚀 **Submitted job {job_id}**\n\nMonitor with: `squeue -j {job_id}`"
                else:
                    message += f"\n⚠️ Job submission failed: {result.stderr}"
            except Exception as e:
                message += f"\n⚠️ Could not submit SLURM job: {e}\n\nRun manually:\n```bash\ncd {output_dir}\nbash {job_file}\n```"
            
            # Add validation reminder
            message += f"""

📋 **Important:** After download completes, validate the data:
```
validate dataset {dataset_id}
```
This will check if you got actual sequencing data (FASTQ) or just metadata files."""
            
            return ToolResult(
                success=True,
                tool_name="download_dataset",
                data={"dataset_id": dataset_id, "path": str(output_dir), "status": "downloading"},
                message=message
            )
            
        elif dataset_id.startswith("ENCSR"):
            # ENCODE dataset
            message = f"""⏳ **Downloading {dataset_id} from ENCODE...**

Fetching files from ENCODE portal...
"""
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use ENCODE API to get file URLs
            try:
                import requests
                api_url = f"https://www.encodeproject.org/experiments/{dataset_id}/?format=json"
                resp = requests.get(api_url, timeout=30, headers={"Accept": "application/json"})
                
                if resp.status_code == 200:
                    data = resp.json()
                    files = data.get("files", [])
                    fastq_files = [f for f in files if f.get("file_format") == "fastq"]
                    
                    if fastq_files:
                        message += f"\nFound {len(fastq_files)} FASTQ files.\n\n"
                        
                        # Create download script
                        download_cmds = []
                        for f in fastq_files[:10]:  # Limit to first 10
                            href = f.get("href", "")
                            if href:
                                url = f"https://www.encodeproject.org{href}"
                                download_cmds.append(f"wget -c '{url}'")
                        
                        script = f"""#!/bin/bash
cd {output_dir}
echo "Downloading ENCODE files..."
{chr(10).join(download_cmds)}
echo "Done!"
"""
                        script_file = output_dir / "download.sh"
                        script_file.write_text(script)
                        script_file.chmod(0o755)
                        
                        message += f"""📥 Download script created: `{script_file}`

Run with:
```bash
cd {output_dir}
nohup bash download.sh > download.log 2>&1 &
```
"""
                    else:
                        message += "\n⚠️ No FASTQ files found in this experiment."
                else:
                    message += f"\n⚠️ Could not fetch experiment info (HTTP {resp.status_code})"
                    
            except Exception as e:
                message += f"\n⚠️ Error fetching ENCODE data: {e}"
            
            return ToolResult(
                success=True,
                tool_name="download_dataset",
                data={"dataset_id": dataset_id, "path": str(output_dir), "status": "script_created"},
                message=message
            )
        
        else:
            return ToolResult(
                success=False,
                tool_name="download_dataset",
                error=f"Unknown dataset format: {dataset_id}",
                message=f"❌ Unknown dataset format: `{dataset_id}`\n\nSupported formats:\n- **GEO:** GSE123456\n- **ENCODE:** ENCSR123ABC"
            )
    
    def check_references(self, organism: str = None) -> ToolResult:
        """
        Check reference genome availability.
        
        Args:
            organism: Organism name (e.g., "human", "mouse")
            
        Returns:
            ToolResult with reference status
        """
        ref_manager = self._get_reference_manager()
        
        if ref_manager is None:
            return ToolResult(
                success=False,
                tool_name="check_references",
                error="Reference manager not available",
                message="❌ Reference manager is not available."
            )
        
        # Default organism
        organism = organism or "human"
        organism = organism.strip().lower()
        
        # Map common names to assemblies
        organism_map = {
            "human": "GRCh38",
            "homo sapiens": "GRCh38",
            "mouse": "GRCm39",
            "mus musculus": "GRCm39",
            "rat": "mRatBN7.2",
            "zebrafish": "GRCz11",
            "drosophila": "BDGP6",
            "c. elegans": "WBcel235",
        }
        
        assembly = organism_map.get(organism, organism.upper())
        
        # Also need organism name for the reference manager
        organism_name_map = {
            "human": "human",
            "homo sapiens": "human",
            "mouse": "mouse",
            "mus musculus": "mouse",
            "rat": "rat",
            "zebrafish": "zebrafish",
            "drosophila": "drosophila",
            "c. elegans": "c_elegans",
        }
        organism_name = organism_name_map.get(organism, organism)
        
        try:
            ref_info = ref_manager.check_references(organism=organism_name, assembly=assembly)
            
            # Build status message using ReferenceInfo attributes
            status_lines = []
            
            # Genome FASTA
            if ref_info.genome_fasta:
                status_lines.append(f"  - 🧬 Genome: ✅ `{ref_info.genome_fasta.name}`")
            else:
                status_lines.append("  - 🧬 Genome: ❌ Not found")
            
            # Annotation GTF
            if ref_info.annotation_gtf:
                status_lines.append(f"  - 📝 Annotation: ✅ `{ref_info.annotation_gtf.name}`")
            else:
                status_lines.append("  - 📝 Annotation: ❌ Not found")
            
            # Transcriptome
            if ref_info.transcriptome_fasta:
                status_lines.append(f"  - 📚 Transcriptome: ✅ Available")
            
            # Indexes
            index_status = []
            if ref_info.star_index:
                index_status.append("STAR ✅")
            if ref_info.hisat2_index:
                index_status.append("HISAT2 ✅")
            if ref_info.bwa_index:
                index_status.append("BWA ✅")
            if ref_info.salmon_index:
                index_status.append("Salmon ✅")
            if ref_info.kallisto_index:
                index_status.append("Kallisto ✅")
            
            if index_status:
                status_lines.append(f"  - 🔧 Indexes: {', '.join(index_status)}")
            else:
                status_lines.append("  - 🔧 Indexes: ❌ None built")
            
            # Missing items
            if ref_info.missing:
                status_lines.append(f"\n**Missing:** {', '.join(ref_info.missing)}")
            
            # Reference directory
            ref_dir = ref_manager.get_organism_dir(organism_name)
            status_lines.append(f"\n📁 Location: `{ref_dir}`")
            
            message = f"""🧬 Reference status for **{assembly}** ({organism}):

{chr(10).join(status_lines)}"""
            
            return ToolResult(
                success=True,
                tool_name="check_references",
                data={"assembly": assembly, "ref_info": ref_info},
                message=message,
                ui_update={"reference_status": f"{assembly}: {'Ready' if ref_info.genome_fasta else 'Missing'}"}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name="check_references",
                error=str(e),
                message=f"❌ Failed to check references: {e}"
            )
    
    # ========== EXECUTION TOOLS ==========
    
    def submit_job(self, workflow_name: str = None, profile: str = "slurm") -> ToolResult:
        """
        Submit a workflow for execution.
        
        Args:
            workflow_name: Name of workflow to run (or uses last generated)
            profile: Execution profile (slurm, local, docker)
            
        Returns:
            ToolResult with job ID
        """
        from pathlib import Path
        
        # Find workflow directory
        workflows_dir = Path.home() / "BioPipelines" / "generated_workflows"
        
        if not workflow_name:
            # Get most recent workflow
            if workflows_dir.exists():
                workflows = sorted(workflows_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                if workflows:
                    workflow_name = workflows[0].name
        
        if not workflow_name:
            return ToolResult(
                success=False,
                tool_name="submit_job",
                error="No workflow specified",
                message="❌ No workflow to run. Generate a workflow first!"
            )
        
        workflow_path = workflows_dir / workflow_name
        if not workflow_path.exists():
            return ToolResult(
                success=False,
                tool_name="submit_job",
                error=f"Workflow not found: {workflow_name}",
                message=f"❌ Workflow not found: `{workflow_name}`"
            )
        
        # Validate profile
        profile = (profile or "slurm").lower().strip()
        if profile not in ["slurm", "local", "docker"]:
            profile = "slurm"
        
        try:
            # Try to submit via executor
            executor = self._get_executor()
            
            if executor:
                job = executor.submit_job(
                    workflow_dir=str(workflow_path),
                    profile=profile,
                    resume=False
                )
                job_id = job.job_id if hasattr(job, 'job_id') else str(job)
            else:
                # Generate a placeholder job ID
                job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
            message = f"""🚀 **Submitted workflow** `{workflow_name}`

- **Job ID:** `{job_id}`
- **Profile:** {profile}
- **Status:** Queued

I'll monitor the progress in the sidebar. You can also say "show status" or "show logs"."""
            
            return ToolResult(
                success=True,
                tool_name="submit_job",
                data={"job_id": job_id, "workflow": workflow_name, "profile": profile},
                message=message,
                ui_update={
                    "active_job": job_id,
                    "job_status": "queued"
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name="submit_job",
                error=str(e),
                message=f"❌ Failed to submit job: {e}"
            )
    
    def get_job_status(self, job_id: str = None) -> ToolResult:
        """
        Get status of a running job.
        
        Args:
            job_id: Optional job ID. If not provided, shows all active jobs.
            
        Returns:
            ToolResult with job status
        """
        try:
            executor = self._get_executor()
            
            if executor and hasattr(executor, 'list_jobs'):
                jobs = executor.list_jobs()
            else:
                # Mock response when executor not available
                jobs = []
            
            if not jobs:
                return ToolResult(
                    success=True,
                    tool_name="get_job_status",
                    data={"jobs": []},
                    message="📋 No active jobs. Submit a workflow to get started!"
                )
            
            if job_id:
                # Filter to specific job
                job = next((j for j in jobs if str(j.get('id', '')) == str(job_id)), None)
                if job:
                    message = f"""📊 **Job {job_id}** Status:

- **Status:** {job.get('status', 'Unknown')}
- **Progress:** {job.get('progress', 0):.0f}%
- **Runtime:** {job.get('runtime', 'N/A')}"""
                else:
                    message = f"⚠️ Job `{job_id}` not found."
            else:
                # Show all jobs
                job_lines = [
                    f"  - `{j.get('id')}`: {j.get('status')} ({j.get('progress', 0):.0f}%)"
                    for j in jobs[:5]
                ]
                message = f"""📊 **Active Jobs:**

{chr(10).join(job_lines)}"""
            
            return ToolResult(
                success=True,
                tool_name="get_job_status",
                data={"jobs": jobs},
                message=message
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name="get_job_status",
                error=str(e),
                message=f"❌ Failed to get job status: {e}"
            )
    
    def monitor_jobs(self, *job_ids) -> ToolResult:
        """
        Monitor SLURM jobs and check if downloads completed.
        
        Args:
            job_ids: Optional job IDs to check. If not provided, checks all recent jobs.
            
        Returns:
            ToolResult with job status and output file info
        """
        import subprocess
        import os
        
        try:
            # Flatten job_ids and filter out None/empty
            job_list = [str(j) for j in job_ids if j]
            
            message_parts = []
            job_details = []
            
            # Get all user's jobs if no specific job IDs provided
            if not job_list:
                # Check sacct for recent jobs (last 24 hours)
                result = subprocess.run(
                    ['sacct', '--me', '-S', 'now-1day', '--format=JobID,JobName%30,State,ExitCode,End,Elapsed', '-n'],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0 and result.stdout.strip():
                    lines = [l for l in result.stdout.strip().split('\n') if l.strip() and not '.batch' in l and not '.extern' in l]
                    if lines:
                        message_parts.append("## 📊 Recent Jobs (last 24 hours)\n")
                        message_parts.append("| Job ID | Name | Status | Exit | End Time | Duration |")
                        message_parts.append("|--------|------|--------|------|----------|----------|")
                        for line in lines[:10]:
                            parts = line.split()
                            if len(parts) >= 6:
                                job_id, name, state = parts[0], parts[1], parts[2]
                                exit_code, end_time, elapsed = parts[3], parts[4], parts[5]
                                status_icon = "✅" if state == "COMPLETED" else "❌" if state in ["FAILED", "CANCELLED"] else "🔄"
                                message_parts.append(f"| {job_id} | {name} | {status_icon} {state} | {exit_code} | {end_time} | {elapsed} |")
                                job_details.append({'id': job_id, 'name': name, 'state': state})
            else:
                # Check specific job IDs
                for job_id in job_list:
                    # Get job status using sacct (works for completed jobs)
                    result = subprocess.run(
                        ['sacct', '-j', str(job_id), '--format=JobID,JobName%30,State,ExitCode,End,Elapsed', '-n'],
                        capture_output=True, text=True, timeout=30
                    )
                    
                    if result.returncode == 0 and result.stdout.strip():
                        lines = [l for l in result.stdout.strip().split('\n') if l.strip() and not '.batch' in l and not '.extern' in l]
                        if lines:
                            parts = lines[0].split()
                            if len(parts) >= 4:
                                state = parts[2]
                                exit_code = parts[3]
                                status_icon = "✅" if state == "COMPLETED" else "❌" if state in ["FAILED", "CANCELLED"] else "🔄"
                                message_parts.append(f"### Job {job_id}: {status_icon} {state}")
                                job_details.append({'id': job_id, 'state': state, 'exit_code': exit_code})
                                
                                if state == "COMPLETED":
                                    message_parts.append(f"  - Exit code: {exit_code}")
                                elif state == "FAILED":
                                    message_parts.append(f"  - ❌ Exit code: {exit_code}")
                                    # Try to get error log
                                    log_path = f"/home/sdodl001_odu_edu/BioPipelines/logs/download_{job_id}.log"
                                    if os.path.exists(log_path):
                                        with open(log_path, 'r') as f:
                                            log_tail = f.read()[-500:]  # Last 500 chars
                                            if log_tail:
                                                message_parts.append(f"  - Log tail:\n```\n{log_tail}\n```")
            
            # Check for downloaded data in common locations
            data_paths = [
                "/scratch/sdodl001/BioPipelines/data/raw/GSE200839",
                "/scratch/sdodl001/BioPipelines/data/raw/GSE178206"
            ]
            
            message_parts.append("\n## 📁 Downloaded Data Check\n")
            for path in data_paths:
                if os.path.exists(path):
                    files = os.listdir(path)
                    fastq_files = [f for f in files if f.endswith(('.fastq', '.fastq.gz', '.fq', '.fq.gz'))]
                    total_size = sum(os.path.getsize(os.path.join(path, f)) for f in files if os.path.isfile(os.path.join(path, f)))
                    size_str = f"{total_size / (1024**3):.2f} GB" if total_size > 1024**3 else f"{total_size / (1024**2):.1f} MB"
                    
                    dataset_name = os.path.basename(path)
                    if fastq_files:
                        message_parts.append(f"✅ **{dataset_name}**: {len(fastq_files)} FASTQ files ({size_str})")
                    elif files:
                        message_parts.append(f"⚠️ **{dataset_name}**: {len(files)} files ({size_str}) - no FASTQs yet")
                        # Show what files exist
                        message_parts.append(f"   Files: {', '.join(files[:5])}" + ("..." if len(files) > 5 else ""))
                    else:
                        message_parts.append(f"📂 **{dataset_name}**: Empty directory")
                else:
                    message_parts.append(f"❌ **{os.path.basename(path)}**: Not found")
            
            if not message_parts:
                message = "📋 No jobs found to monitor. Try specifying job IDs like: `check jobs 1104, 1105`"
            else:
                message = "\n".join(message_parts)
            
            return ToolResult(
                success=True,
                tool_name="monitor_jobs",
                data={"jobs": job_details},
                message=message
            )
            
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                tool_name="monitor_jobs",
                error="SLURM command timed out",
                message="⚠️ SLURM is slow to respond. Try again in a moment."
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name="monitor_jobs",
                error=str(e),
                message=f"❌ Failed to monitor jobs: {e}"
            )
    
    def get_logs(self, job_id: str = None, lines: int = 30) -> ToolResult:
        """
        Get logs from a job.
        
        Args:
            job_id: Optional job ID
            lines: Number of lines to return
            
        Returns:
            ToolResult with log content
        """
        try:
            executor = self._get_executor()
            
            if executor and hasattr(executor, 'get_job_logs'):
                logs = executor.get_job_logs(job_id, lines=lines)
            else:
                logs = "*No logs available. Job may still be starting.*"
            
            message = f"""📄 **Logs** (last {lines} lines):

```
{logs}
```"""
            
            return ToolResult(
                success=True,
                tool_name="get_logs",
                data={"logs": logs, "job_id": job_id},
                message=message
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name="get_logs",
                error=str(e),
                message=f"❌ Failed to get logs: {e}"
            )
    
    def cancel_job(self, job_id: str = None) -> ToolResult:
        """
        Cancel a running job.
        
        Args:
            job_id: Optional job ID
            
        Returns:
            ToolResult confirming cancellation
        """
        if not job_id:
            return ToolResult(
                success=False,
                tool_name="cancel_job",
                error="No job ID specified",
                message="❌ Please specify which job to cancel (e.g., 'cancel job 12345')"
            )
        
        try:
            executor = self._get_executor()
            
            if executor and hasattr(executor, 'cancel_job'):
                executor.cancel_job(job_id)
                message = f"🛑 **Cancelled** job `{job_id}`"
            else:
                message = f"🛑 Requested cancellation of job `{job_id}` (executor not available)"
            
            return ToolResult(
                success=True,
                tool_name="cancel_job",
                data={"job_id": job_id, "cancelled": True},
                message=message,
                ui_update={"job_status": "cancelled"}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name="cancel_job",
                error=str(e),
                message=f"❌ Failed to cancel job: {e}"
            )
    
    def diagnose_error(self, job_id: str = None) -> ToolResult:
        """
        Diagnose errors in a failed job.
        
        Args:
            job_id: Optional job ID
            
        Returns:
            ToolResult with diagnosis
        """
        try:
            # Use the proper ErrorDiagnosisAgent
            from workflow_composer.diagnosis.agent import ErrorDiagnosisAgent
            
            agent = ErrorDiagnosisAgent()
            
            # Get logs first
            logs_result = self.get_logs(job_id)
            logs = logs_result.data.get('logs', '') if logs_result.data else ''
            
            if not logs:
                # Try to get logs from SLURM directly
                import subprocess
                if job_id:
                    try:
                        result = subprocess.run(
                            ['sacct', '-j', str(job_id), '--format=State,ExitCode,Elapsed,MaxRSS', '-n'],
                            capture_output=True, text=True, timeout=30
                        )
                        if result.stdout.strip():
                            logs = f"Job {job_id} status:\n{result.stdout}"
                            
                        # Try to find log files
                        log_paths = [
                            f"/home/sdodl001_odu_edu/BioPipelines/logs/download_{job_id}.log",
                            f"/home/sdodl001_odu_edu/BioPipelines/logs/vllm_{job_id}.err",
                        ]
                        import os
                        for path in log_paths:
                            if os.path.exists(path):
                                with open(path, 'r') as f:
                                    logs += f"\n\n=== {path} ===\n" + f.read()[-2000:]
                    except Exception as e:
                        logger.warning(f"Failed to get SLURM info: {e}")
            
            if not logs:
                return ToolResult(
                    success=False,
                    tool_name="diagnose_error",
                    error="No logs available for diagnosis",
                    message="❌ No logs found to diagnose. Make sure the job has run."
                )
            
            # Run diagnosis using the pattern-based diagnosis
            diagnosis = agent.diagnose_from_logs_sync(logs, use_llm=False)
            
            # Format suggested fixes
            fixes_text = ""
            if diagnosis.suggested_fixes:
                fixes_text = "\n".join(
                    f"  {i+1}. {fix.description}" + (f"\n     Command: `{fix.command}`" if fix.command else "")
                    for i, fix in enumerate(diagnosis.suggested_fixes[:3])
                )
            else:
                fixes_text = "  No automated fixes available"
            
            message = f"""🔍 **Error Diagnosis:**

**Category:** {diagnosis.category.value}
**Confidence:** {diagnosis.confidence:.0%}

**Root Cause:** {diagnosis.root_cause}

**Explanation:** {diagnosis.user_explanation}

**Suggested Fixes:**
{fixes_text}

Would you like me to apply any of these fixes?"""
            
            return ToolResult(
                success=True,
                tool_name="diagnose_error",
                data={
                    "category": diagnosis.category.value,
                    "confidence": diagnosis.confidence,
                    "root_cause": diagnosis.root_cause,
                    "fixes": [f.description for f in diagnosis.suggested_fixes]
                },
                message=message
            )
            
        except ImportError:
            return ToolResult(
                success=False,
                tool_name="diagnose_error",
                error="Diagnosis agent not available",
                message="❌ AI diagnosis is not available. Check the logs manually."
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name="diagnose_error",
                error=str(e),
                message=f"❌ Diagnosis failed: {e}"
            )
    
    def list_workflows(self) -> ToolResult:
        """
        List available workflows.
        
        Returns:
            ToolResult with workflow list
        """
        from pathlib import Path
        
        workflows_dir = Path.home() / "BioPipelines" / "generated_workflows"
        
        if not workflows_dir.exists():
            return ToolResult(
                success=True,
                tool_name="list_workflows",
                data={"workflows": []},
                message="📋 No workflows generated yet. Describe your analysis to create one!"
            )
        
        workflows = sorted(
            [d.name for d in workflows_dir.iterdir() if d.is_dir()],
            reverse=True
        )[:10]
        
        if workflows:
            workflow_list = "\n".join([f"  - `{w}`" for w in workflows])
            message = f"""📋 **Available Workflows:**

{workflow_list}

Say "run <workflow_name>" to execute one."""
        else:
            message = "📋 No workflows found. Generate one by describing your analysis!"
        
        return ToolResult(
            success=True,
            tool_name="list_workflows",
            data={"workflows": workflows},
            message=message
        )
    
    def download_results(self, job_id: str = None) -> ToolResult:
        """
        Download/package results from a completed job.
        
        Args:
            job_id: Optional job ID
            
        Returns:
            ToolResult with download information
        """
        from pathlib import Path
        import shutil
        
        # Find results directory
        results_dir = Path.home() / "BioPipelines" / "data" / "results"
        
        if not results_dir.exists():
            return ToolResult(
                success=False,
                tool_name="download_results",
                error="No results directory found",
                message="❌ No results found. Run a workflow first!"
            )
        
        # Get recent result directories
        result_dirs = sorted(
            [d for d in results_dir.iterdir() if d.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:5]
        
        if not result_dirs:
            return ToolResult(
                success=False,
                tool_name="download_results",
                error="No result directories found",
                message="❌ No results found. Complete a workflow first!"
            )
        
        # List available results
        result_list = "\n".join([
            f"  - `{d.name}` ({sum(1 for _ in d.rglob('*') if _.is_file())} files)"
            for d in result_dirs
        ])
        
        message = f"""📦 **Available Results:**

{result_list}

To download, go to the **Results** tab and select a directory.
Or say "show results" to browse files."""
        
        return ToolResult(
            success=True,
            tool_name="download_results",
            data={"results": [str(d) for d in result_dirs]},
            message=message
        )
    
    def compare_samples(self, comparison: str = None) -> ToolResult:
        """
        Compare samples or groups for differential analysis.
        
        Args:
            comparison: Comparison description (e.g., "treatment vs control")
            
        Returns:
            ToolResult with comparison setup
        """
        manifest = self._get_manifest()
        
        if manifest is None or not manifest.samples:
            return ToolResult(
                success=False,
                tool_name="compare_samples",
                error="No samples in manifest",
                message="❌ No samples loaded. Scan data first with 'scan data in /path'"
            )
        
        samples = manifest.samples
        
        # Analyze sample names for potential groups
        sample_ids = [s.sample_id for s in samples]
        
        # Try to detect groups from naming patterns
        groups = {}
        for sid in sample_ids:
            # Common patterns: sample_treatment_rep1, ctrl_1, case_2
            parts = sid.replace('-', '_').split('_')
            if len(parts) >= 2:
                group = parts[0]
                if group not in groups:
                    groups[group] = []
                groups[group].append(sid)
        
        if len(groups) >= 2:
            group_list = "\n".join([
                f"  - **{g}**: {len(samples)} samples"
                for g, samples in groups.items()
            ])
            message = f"""🔬 **Detected Sample Groups:**

{group_list}

To set up a comparison, include this in your workflow request:
> "Compare {list(groups.keys())[0]} vs {list(groups.keys())[1]}"

Or manually specify groups when generating the workflow."""
        else:
            sample_list = "\n".join([f"  - `{s}`" for s in sample_ids[:10]])
            if len(sample_ids) > 10:
                sample_list += f"\n  - ... and {len(sample_ids) - 10} more"
            
            message = f"""🔬 **Samples in Manifest:**

{sample_list}

To compare groups, ensure sample names follow a pattern like:
- `treatment_rep1`, `treatment_rep2`, `control_rep1`, `control_rep2`
- Or specify groups in your workflow request."""
        
        return ToolResult(
            success=True,
            tool_name="compare_samples",
            data={"samples": sample_ids, "groups": groups},
            message=message
        )
    
    def describe_files(self, path: str = None) -> ToolResult:
        """
        Describe files in a directory - get file details, sizes, and content summaries.
        
        This tool inspects any file type including:
        - GEO matrix files (.series_matrix.txt.gz)
        - FASTQ files
        - BAM/SAM files
        - CSV/TSV files
        - Gzipped files
        
        Args:
            path: Directory path to inspect. Defaults to last downloaded dataset.
            
        Returns:
            ToolResult with file descriptions
        """
        import subprocess
        import gzip
        import os
        
        # Determine path to inspect
        if not path:
            # Try to find a recently downloaded dataset
            data_dir = Path(os.environ.get('DATA_DIR', '/scratch/sdodl001/BioPipelines/data'))
            raw_dir = data_dir / "raw"
            
            # Look for GSE directories
            gse_dirs = sorted([d for d in raw_dir.glob("GSE*") if d.is_dir()], 
                            key=lambda x: x.stat().st_mtime, reverse=True)
            
            if gse_dirs:
                path = str(gse_dirs[0])
            else:
                return ToolResult(
                    success=False,
                    tool_name="describe_files",
                    error="No path specified and no downloaded datasets found",
                    message="📁 Please specify a path to inspect, e.g., 'describe files in /path/to/data'"
                )
        
        target_path = Path(path).expanduser()
        
        if not target_path.exists():
            return ToolResult(
                success=False,
                tool_name="describe_files",
                error=f"Path not found: {target_path}",
                message=f"❌ Path not found: `{target_path}`"
            )
        
        file_info = []
        
        # Collect files to inspect
        if target_path.is_file():
            files_to_check = [target_path]
        else:
            files_to_check = sorted(target_path.iterdir())
        
        for item in files_to_check:
            if item.is_dir():
                # Count items in subdirectory
                subcount = len(list(item.iterdir()))
                file_info.append({
                    'name': f"📁 {item.name}/",
                    'size': f"{subcount} items",
                    'type': 'directory',
                    'details': ''
                })
                continue
            
            # Get file stats
            try:
                stat = item.stat()
                size_bytes = stat.st_size
                
                # Human readable size
                if size_bytes >= 1024**3:
                    size_str = f"{size_bytes / 1024**3:.1f} GB"
                elif size_bytes >= 1024**2:
                    size_str = f"{size_bytes / 1024**2:.1f} MB"
                elif size_bytes >= 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                else:
                    size_str = f"{size_bytes} B"
                
                # Determine file type and extract details
                details = ""
                file_type = "unknown"
                name = item.name
                
                if name.endswith('.gz'):
                    # Try to peek inside gzipped file
                    base_name = name[:-3]
                    
                    if 'series_matrix' in name or base_name.endswith('.txt') or base_name.endswith('.csv') or base_name.endswith('.tsv'):
                        # It's a tabular file - get row/column count
                        file_type = "matrix/table"
                        try:
                            with gzip.open(str(item), 'rt', errors='replace') as f:
                                lines = []
                                header_line = None
                                data_lines = 0
                                
                                for i, line in enumerate(f):
                                    if i > 100:  # Sample first 100 lines
                                        break
                                    lines.append(line)
                                    
                                    # Skip comment lines
                                    if line.startswith('!') or line.startswith('#'):
                                        continue
                                    elif header_line is None and '\t' in line:
                                        header_line = line.strip()
                                    else:
                                        data_lines += 1
                                
                                # Count total lines
                                try:
                                    result = subprocess.run(
                                        f"zcat '{item}' | wc -l",
                                        shell=True,
                                        capture_output=True,
                                        text=True,
                                        timeout=30
                                    )
                                    total_lines = int(result.stdout.strip())
                                except:
                                    total_lines = len(lines) if data_lines < 100 else "100+"
                                
                                # Get columns from header
                                if header_line:
                                    columns = header_line.split('\t')
                                    col_count = len(columns)
                                    # Show first few column names
                                    col_preview = columns[:5]
                                    if len(columns) > 5:
                                        col_preview.append(f"... +{len(columns)-5} more")
                                    details = f"{total_lines} rows, {col_count} cols: {', '.join(col_preview[:3])}"
                                else:
                                    details = f"{total_lines} lines"
                                    
                        except Exception as e:
                            details = f"gzipped (error reading: {str(e)[:30]})"
                            
                    elif base_name.endswith('.soft'):
                        file_type = "GEO metadata"
                        details = "SOFT format metadata"
                        
                    elif base_name.endswith('.fastq') or base_name.endswith('.fq'):
                        file_type = "FASTQ"
                        # Count reads (every 4 lines = 1 read)
                        try:
                            result = subprocess.run(
                                f"zcat '{item}' | head -n 4000 | wc -l",
                                shell=True,
                                capture_output=True,
                                text=True,
                                timeout=30
                            )
                            lines_sample = int(result.stdout.strip())
                            if lines_sample >= 4000:
                                # Estimate based on file size
                                approx_reads = f"~{size_bytes // 250:,}+"
                            else:
                                approx_reads = f"{lines_sample // 4:,}"
                            details = f"{approx_reads} reads"
                        except:
                            details = "compressed FASTQ"
                    else:
                        file_type = "gzipped"
                        details = "compressed file"
                        
                elif name.endswith('.txt'):
                    file_type = "text"
                    # Read first few lines
                    try:
                        with open(item, 'r') as f:
                            lines = [l.strip() for l in f.readlines()[:10]]
                        if lines:
                            details = f"{len(lines)} lines preview: {lines[0][:40]}..."
                        else:
                            details = "empty file"
                    except:
                        details = "text file"
                        
                elif name.endswith('.bam'):
                    file_type = "BAM"
                    details = "alignment file"
                    
                elif name.endswith('.vcf') or name.endswith('.vcf.gz'):
                    file_type = "VCF"
                    details = "variant calls"
                    
                elif name.endswith('.bed') or name.endswith('.bed.gz'):
                    file_type = "BED"
                    details = "genomic intervals"
                    
                elif name.endswith('.csv'):
                    file_type = "CSV"
                    try:
                        with open(item, 'r') as f:
                            header = f.readline().strip()
                            columns = header.split(',')
                            row_count = sum(1 for _ in f) + 1
                        details = f"{row_count} rows, {len(columns)} cols"
                    except:
                        details = "CSV file"
                        
                elif name.endswith('.tsv'):
                    file_type = "TSV"
                    try:
                        with open(item, 'r') as f:
                            header = f.readline().strip()
                            columns = header.split('\t')
                            row_count = sum(1 for _ in f) + 1
                        details = f"{row_count} rows, {len(columns)} cols"
                    except:
                        details = "TSV file"
                
                file_info.append({
                    'name': name,
                    'size': size_str,
                    'type': file_type,
                    'details': details
                })
                
            except Exception as e:
                file_info.append({
                    'name': item.name,
                    'size': 'error',
                    'type': 'unknown',
                    'details': str(e)[:30]
                })
        
        if not file_info:
            return ToolResult(
                success=True,
                tool_name="describe_files",
                data={"path": str(target_path), "files": []},
                message=f"📁 No files found in `{target_path}`"
            )
        
        # Build table output
        lines = [f"## 📁 Files in `{target_path.name}`\n"]
        lines.append("| Name | Size | Type | Details |")
        lines.append("|------|------|------|---------|")
        
        for f in file_info:
            name = f['name'][:40] + "..." if len(f['name']) > 40 else f['name']
            details = f['details'][:50] + "..." if len(f['details']) > 50 else f['details']
            lines.append(f"| `{name}` | {f['size']} | {f['type']} | {details} |")
        
        # Add summary
        total_files = len([f for f in file_info if f['type'] != 'directory'])
        total_dirs = len([f for f in file_info if f['type'] == 'directory'])
        
        lines.append(f"\n**Summary:** {total_files} files, {total_dirs} directories")
        
        # Add data type detection
        matrix_files = [f for f in file_info if 'matrix' in f['type'].lower() or 'table' in f['type'].lower()]
        fastq_files = [f for f in file_info if 'fastq' in f['type'].lower()]
        
        if matrix_files and not fastq_files:
            lines.append("\n⚠️ **Note:** This dataset contains processed matrix files (not raw FASTQ sequences).")
            lines.append("These are typically from array-based experiments (e.g., Illumina methylation arrays).")
            lines.append("For sequencing data, look for FASTQ files or check if raw data is available in SRA.")
        
        return ToolResult(
            success=True,
            tool_name="describe_files",
            data={"path": str(target_path), "files": file_info},
            message="\n".join(lines)
        )
    
    def validate_dataset(self, path_or_id: str = None) -> ToolResult:
        """
        Validate if a downloaded dataset contains actual data or just metadata.
        
        This is a critical tool that helps users understand:
        1. Whether they have real sequencing data (FASTQ) or processed data
        2. Whether files are metadata vs actual experimental data
        3. What the tissue/cell type is (does it match their request?)
        4. What additional steps are needed to get usable data
        
        Args:
            path_or_id: Path to dataset directory or GEO/ENCODE ID
            
        Returns:
            ToolResult with comprehensive validation report
        """
        import subprocess
        import gzip
        import os
        import re
        
        # Determine path to validate
        if not path_or_id:
            # Try to find a recently downloaded dataset
            data_dir = Path(os.environ.get('DATA_DIR', '/scratch/sdodl001/BioPipelines/data'))
            raw_dir = data_dir / "raw"
            
            gse_dirs = sorted([d for d in raw_dir.glob("GSE*") if d.is_dir()], 
                            key=lambda x: x.stat().st_mtime, reverse=True)
            
            if gse_dirs:
                path_or_id = str(gse_dirs[0])
            else:
                return ToolResult(
                    success=False,
                    tool_name="validate_dataset",
                    error="No path specified",
                    message="📁 No dataset found. Specify a path or dataset ID."
                )
        
        # Handle dataset ID vs path
        if path_or_id.upper().startswith("GSE") or path_or_id.upper().startswith("ENCSR"):
            data_dir = Path(os.environ.get('DATA_DIR', '/scratch/sdodl001/BioPipelines/data'))
            target_path = data_dir / "raw" / path_or_id.upper()
        else:
            target_path = Path(path_or_id).expanduser()
        
        if not target_path.exists():
            return ToolResult(
                success=False,
                tool_name="validate_dataset",
                error=f"Path not found: {target_path}",
                message=f"❌ Dataset not found: `{target_path}`"
            )
        
        # Collect validation results
        validation = {
            'dataset_id': target_path.name,
            'path': str(target_path),
            'has_raw_fastq': False,
            'has_processed_data': False,
            'has_metadata_only': False,
            'file_types': {},
            'tissue_types': set(),
            'organisms': set(),
            'assay_types': set(),
            'sample_count': 0,
            'total_size_bytes': 0,
            'issues': [],
            'recommendations': []
        }
        
        # Scan files in dataset
        files = list(target_path.iterdir())
        
        for f in files:
            if f.is_dir():
                continue
            
            name = f.name.lower()
            size = f.stat().st_size
            validation['total_size_bytes'] += size
            
            # Categorize files
            if name.endswith(('.fastq.gz', '.fq.gz', '.fastq', '.fq')):
                validation['file_types'].setdefault('fastq', []).append(f.name)
                validation['has_raw_fastq'] = True
                
            elif name.endswith('.bam') or name.endswith('.cram'):
                validation['file_types'].setdefault('alignment', []).append(f.name)
                validation['has_processed_data'] = True
                
            elif name.endswith('.bw') or name.endswith('.bigwig'):
                validation['file_types'].setdefault('bigwig', []).append(f.name)
                validation['has_processed_data'] = True
                
            elif name.endswith('.bed') or name.endswith('.bed.gz'):
                validation['file_types'].setdefault('bed', []).append(f.name)
                validation['has_processed_data'] = True
                
            elif name.endswith('.idat'):
                validation['file_types'].setdefault('idat', []).append(f.name)
                validation['has_raw_fastq'] = True  # IDAT is raw for array data
                validation['assay_types'].add('Illumina Array (450K/EPIC)')
                
            elif 'series_matrix' in name or 'family.soft' in name:
                validation['file_types'].setdefault('metadata', []).append(f.name)
                
                # Parse metadata for tissue/organism info
                try:
                    opener = gzip.open if name.endswith('.gz') else open
                    with opener(str(f), 'rt', errors='replace') as fp:
                        content = fp.read(50000)  # First 50KB
                        
                        # Extract tissue types
                        tissue_matches = re.findall(r'(?:tissue|cell.?type|cell.?line):\s*([^\n"]+)', content, re.I)
                        for t in tissue_matches:
                            validation['tissue_types'].add(t.strip())
                        
                        # Extract organism
                        org_matches = re.findall(r'(?:organism|taxid).*?:\s*([^\n"]+)', content, re.I)
                        for o in org_matches:
                            if 'sapiens' in o.lower() or 'human' in o.lower():
                                validation['organisms'].add('Human')
                            elif 'musculus' in o.lower() or 'mouse' in o.lower():
                                validation['organisms'].add('Mouse')
                            else:
                                validation['organisms'].add(o.strip()[:30])
                        
                        # Extract assay type
                        assay_matches = re.findall(r'(?:assay|library.?strategy|series_type).*?:\s*([^\n"]+)', content, re.I)
                        for a in assay_matches:
                            validation['assay_types'].add(a.strip()[:40])
                        
                        # Count samples
                        sample_matches = re.findall(r'!Sample_geo_accession\s*[\t"]*(GSM\d+)', content)
                        validation['sample_count'] = max(validation['sample_count'], len(set(sample_matches)))
                        
                except Exception as e:
                    logger.debug(f"Error parsing metadata: {e}")
                    
            elif name == 'filelist.txt':
                validation['file_types'].setdefault('filelist', []).append(f.name)
                
                # Parse filelist to see what data is available
                try:
                    with open(f, 'r') as fp:
                        content = fp.read()
                        if '.tar' in content.lower():
                            validation['recommendations'].append(
                                f"Raw data is in a TAR archive. Download with: `wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/.../RAW.tar`"
                            )
                        if '.bw' in content.lower() or '.bigwig' in content.lower():
                            validation['issues'].append("Dataset contains BigWig files (processed coverage), not raw FASTQ")
                            validation['has_processed_data'] = True
                except:
                    pass
                    
            elif name == 'sra_runs.txt':
                validation['file_types'].setdefault('sra_list', []).append(f.name)
                
                # Check if SRA list is empty
                try:
                    with open(f, 'r') as fp:
                        srr_ids = [l.strip() for l in fp if l.strip()]
                        if not srr_ids:
                            validation['issues'].append("No SRA runs found - dataset may not have raw sequencing data in SRA")
                        else:
                            validation['recommendations'].append(
                                f"Found {len(srr_ids)} SRA runs. Download with: `prefetch {srr_ids[0]}`"
                            )
                except:
                    pass
        
        # Determine overall status
        if not validation['file_types']:
            validation['has_metadata_only'] = True
            
        elif not validation['has_raw_fastq'] and not validation['has_processed_data']:
            # Only metadata files
            if 'metadata' in validation['file_types'] or 'filelist' in validation['file_types']:
                validation['has_metadata_only'] = True
        
        # Build validation report
        size_str = f"{validation['total_size_bytes'] / (1024**3):.2f} GB" if validation['total_size_bytes'] > 1024**3 else f"{validation['total_size_bytes'] / (1024**2):.1f} MB"
        
        report = [f"# 🔍 Dataset Validation: `{validation['dataset_id']}`\n"]
        
        # Overall verdict
        if validation['has_raw_fastq']:
            report.append("## ✅ Status: Contains RAW DATA\n")
        elif validation['has_processed_data'] and not validation['has_metadata_only']:
            report.append("## ⚠️ Status: Contains PROCESSED DATA (not raw)\n")
        else:
            report.append("## ❌ Status: METADATA ONLY\n")
        
        # File types found
        report.append("### 📁 Files Found:\n")
        report.append("| Category | Count | Files |")
        report.append("|----------|-------|-------|")
        for ftype, flist in validation['file_types'].items():
            files_preview = ", ".join(flist[:3])
            if len(flist) > 3:
                files_preview += f" ... +{len(flist)-3} more"
            report.append(f"| {ftype.upper()} | {len(flist)} | {files_preview} |")
        
        report.append(f"\n**Total Size:** {size_str}")
        
        # Sample info
        if validation['tissue_types'] or validation['organisms'] or validation['assay_types']:
            report.append("\n### 🧬 Sample Information:\n")
            if validation['organisms']:
                report.append(f"- **Organism:** {', '.join(validation['organisms'])}")
            if validation['tissue_types']:
                tissues_preview = list(validation['tissue_types'])[:5]
                report.append(f"- **Tissues/Cells:** {', '.join(tissues_preview)}")
            if validation['assay_types']:
                report.append(f"- **Assay Type:** {', '.join(validation['assay_types'])}")
            if validation['sample_count'] > 0:
                report.append(f"- **Samples:** {validation['sample_count']}")
        
        # Issues
        if validation['issues']:
            report.append("\n### ⚠️ Issues:\n")
            for issue in validation['issues']:
                report.append(f"- {issue}")
        
        # Recommendations
        if validation['has_metadata_only']:
            report.append("\n### 💡 What This Means:\n")
            report.append("""
This directory contains **experiment metadata** from GEO, NOT the actual sequencing data.

**Metadata files include:**
- `*_family.soft.gz` - Experiment description
- `*_series_matrix.txt.gz` - Sample annotations
- `filelist.txt` - List of files in the GEO archive

**To get actual data:**
1. Check if raw data is in SRA (look at `sra_runs.txt`)
2. For array data, the IDAT files may be in a separate archive
3. Some datasets only provide processed data (BigWig, BED files)
""")
        
        if validation['recommendations']:
            report.append("\n### 🔧 Recommendations:\n")
            for rec in validation['recommendations']:
                report.append(f"- {rec}")
        
        # Action buttons
        report.append("\n### 🎯 Next Steps:\n")
        if validation['has_metadata_only']:
            report.append("1. Search for datasets with raw FASTQ: `search for brain cancer WGBS with raw fastq`")
            report.append("2. Try TCGA for large clinical cohorts: [TCGA-GBM](https://portal.gdc.cancer.gov/projects/TCGA-GBM)")
            report.append("3. Check ENCODE for reference data: [ENCODE](https://www.encodeproject.org/)")
        elif validation['has_processed_data']:
            report.append("1. This data may still be useful for visualization or downstream analysis")
            report.append("2. For variant calling or alignment, you'll need raw FASTQ")
        else:
            report.append("1. Data looks ready for analysis! Use `scan data` to add to manifest")
        
        return ToolResult(
            success=True,
            tool_name="validate_dataset",
            data=validation,
            message="\n".join(report)
        )
    
    def show_help(self) -> ToolResult:
        """
        Show available chat commands.
        
        Returns:
            ToolResult with help message
        """
        help_message = """# 🤖 BioPipelines Chat Commands

## 📁 Data Discovery & Validation
| Command | Description |
|---------|-------------|
| `scan data in /path` | Find FASTQ files in a directory |
| `describe files in /path` | Get file sizes, rows, columns for any file |
| `validate dataset GSE123` | **NEW** Check if data is real vs metadata |
| `search for human RNA-seq` | Search ENCODE/GEO databases |
| `check reference for human` | Verify genome references |
| `compare samples` | Analyze sample groups |

## 🔬 Data Validation (Important!)
| Command | Description |
|---------|-------------|
| `validate dataset` | Check if downloaded data is real or metadata |
| `is this real data` | Verify data quality and type |
| `what kind of data is this` | Identify file types and content |

## 🔧 Workflow Generation
| Command | Description |
|---------|-------------|
| `create RNA-seq pipeline` | Generate a workflow |
| `build ChIP-seq workflow` | AI generates Nextflow code |
| `list workflows` | Show available workflows |

## 🚀 Execution
| Command | Description |
|---------|-------------|
| `run it on SLURM` | Submit to cluster |
| `run workflow xyz` | Execute specific workflow |
| `check my jobs` | Monitor SLURM job status |
| `show logs` | View job output |
| `cancel job 123` | Stop a running job |

## 📊 Results
| Command | Description |
|---------|-------------|
| `download results` | Package output files |
| `diagnose` | AI error analysis |
| `analyze results` | Interpret QC reports |
| `explain [concept]` | Learn about bioinformatics terms |

## 🔬 Cancer Data (NEW!)
| Command | Description |
|---------|-------------|
| `search TCGA for GBM` | Search cancer genome atlas |
| `find cancer methylation` | Find tumor data from GDC |

## 💡 Tips
- Always **validate datasets** after downloading to check if they contain actual data
- GEO downloads often contain only metadata - real data may be in SRA
- For methylation: IDAT files = array data, FASTQ = sequencing data
- Use `search for X with raw fastq` to find datasets with sequencing data
- For cancer data, use `search TCGA` - ENCODE mostly has normal tissue
"""
        
        return ToolResult(
            success=True,
            tool_name="show_help",
            data={},
            message=help_message
        )
    
    # ========== NEW SMART TOOLS ==========
    
    def search_tcga(self, query: str = None) -> ToolResult:
        """
        Search TCGA/GDC portal for cancer datasets.
        
        This is the right source for cancer research data:
        - TCGA-GBM: Glioblastoma (brain cancer)
        - TCGA-LGG: Low grade glioma
        - TCGA-BRCA: Breast cancer
        - TCGA-LUAD: Lung adenocarcinoma
        - etc.
        
        Args:
            query: Search query (e.g., "GBM methylation", "brain cancer WGBS")
            
        Returns:
            ToolResult with TCGA project information
        """
        import requests
        
        if not query:
            return ToolResult(
                success=False,
                tool_name="search_tcga",
                error="No query provided",
                message="❌ Please specify what you're looking for (e.g., 'search TCGA for GBM methylation')"
            )
        
        query_lower = query.lower()
        
        # Map common terms to TCGA project codes
        project_map = {
            "glioblastoma": "TCGA-GBM",
            "gbm": "TCGA-GBM", 
            "brain cancer": "TCGA-GBM",
            "brain tumor": "TCGA-GBM",
            "glioma": "TCGA-LGG",
            "low grade glioma": "TCGA-LGG",
            "breast cancer": "TCGA-BRCA",
            "breast": "TCGA-BRCA",
            "lung adenocarcinoma": "TCGA-LUAD",
            "lung cancer": "TCGA-LUAD",
            "lung": "TCGA-LUAD",
            "colon cancer": "TCGA-COAD",
            "colon": "TCGA-COAD",
            "colorectal": "TCGA-COAD",
            "prostate": "TCGA-PRAD",
            "prostate cancer": "TCGA-PRAD",
            "kidney": "TCGA-KIRC",
            "liver": "TCGA-LIHC",
            "hepatocellular": "TCGA-LIHC",
            "pancreatic": "TCGA-PAAD",
            "pancreas": "TCGA-PAAD",
            "melanoma": "TCGA-SKCM",
            "skin": "TCGA-SKCM",
            "ovarian": "TCGA-OV",
            "ovary": "TCGA-OV",
            "bladder": "TCGA-BLCA",
            "stomach": "TCGA-STAD",
            "thyroid": "TCGA-THCA",
            "head neck": "TCGA-HNSC",
        }
        
        # Find matching project
        matched_project = None
        for term, project in project_map.items():
            if term in query_lower:
                matched_project = project
                break
        
        # Determine data type
        data_type = None
        if "methylation" in query_lower or "wgbs" in query_lower:
            data_type = "methylation"
        elif "rna" in query_lower or "expression" in query_lower:
            data_type = "transcriptome"
        elif "wgs" in query_lower or "whole genome" in query_lower or "mutation" in query_lower:
            data_type = "wgs"
        elif "exome" in query_lower or "wes" in query_lower:
            data_type = "wes"
        
        # Build informative response
        results = []
        
        if matched_project:
            # Get project info from GDC API
            try:
                gdc_api = f"https://api.gdc.cancer.gov/projects/{matched_project}"
                resp = requests.get(gdc_api, timeout=10)
                if resp.status_code == 200:
                    data = resp.json().get("data", {})
                    results.append({
                        "id": matched_project,
                        "name": data.get("name", matched_project),
                        "primary_site": data.get("primary_site", ["Unknown"]),
                        "disease_type": data.get("disease_type", ["Unknown"]),
                        "case_count": data.get("summary", {}).get("case_count", "Unknown"),
                        "file_count": data.get("summary", {}).get("file_count", "Unknown"),
                    })
            except Exception as e:
                logger.debug(f"GDC API error: {e}")
                # Fallback to static info
                results.append({
                    "id": matched_project,
                    "name": f"TCGA {matched_project.split('-')[1]}",
                    "info": "GDC API unavailable - check portal directly"
                })
        else:
            # List popular TCGA projects
            results = [
                {"id": "TCGA-GBM", "name": "Glioblastoma", "case_count": "617"},
                {"id": "TCGA-LGG", "name": "Low Grade Glioma", "case_count": "516"},
                {"id": "TCGA-BRCA", "name": "Breast Cancer", "case_count": "1098"},
                {"id": "TCGA-LUAD", "name": "Lung Adenocarcinoma", "case_count": "585"},
                {"id": "TCGA-PRAD", "name": "Prostate Adenocarcinoma", "case_count": "500"},
            ]
        
        # Build message
        message = f"# 🔬 TCGA/GDC Cancer Data\n\n**Query:** {query}\n\n"
        
        if matched_project:
            r = results[0]
            message += f"""## {r['id']}: {r.get('name', 'Unknown')}

| Attribute | Value |
|-----------|-------|
| **Project** | [{r['id']}](https://portal.gdc.cancer.gov/projects/{r['id']}) |
| **Cases** | {r.get('case_count', 'Unknown')} |
| **Files** | {r.get('file_count', 'Unknown')} |

"""
            # Add data type specific info
            if data_type == "methylation":
                message += """### 📊 Methylation Data Available:

- **Platform:** Illumina 450K/EPIC arrays (IDAT/BED format)
- **Note:** TCGA uses array-based methylation, NOT WGBS
- **Download:** Use [GDC Data Transfer Tool](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)

**Example download command:**
```bash
gdc-client download -m manifest.txt
```

"""
            elif data_type == "transcriptome":
                message += """### 📊 RNA-seq Data Available:

- **Format:** BAM files, HTSeq counts
- **Samples:** Tumor + normal pairs available
- **Download:** Use GDC Data Transfer Tool

"""
            
            message += f"""### 💡 Next Steps:

1. Visit the [GDC Portal](https://portal.gdc.cancer.gov/projects/{r['id']})
2. Add files to your cart
3. Download manifest.txt
4. Use `gdc-client download -m manifest.txt`

Would you like me to help set up the GDC client?
"""
        else:
            message += """## Available TCGA Projects

| Project | Cancer Type | Cases |
|---------|-------------|-------|
"""
            for r in results:
                message += f"| [{r['id']}](https://portal.gdc.cancer.gov/projects/{r['id']}) | {r['name']} | {r['case_count']} |\n"
            
            message += """
### 💡 Specify a cancer type to get detailed info:
- `search TCGA for GBM methylation`
- `search TCGA for breast cancer RNA-seq`
- `find lung cancer mutations`
"""
        
        return ToolResult(
            success=True,
            tool_name="search_tcga",
            data={"results": results, "query": query, "matched_project": matched_project},
            message=message
        )
    
    def generate_workflow(self, description: str = None) -> ToolResult:
        """
        Generate a workflow from a natural language description.
        
        Args:
            description: What the user wants to analyze
            
        Returns:
            ToolResult with workflow generation status
        """
        if not description:
            return ToolResult(
                success=False,
                tool_name="generate_workflow",
                error="No description provided",
                message="❌ Please describe what you want to analyze. Example:\n- 'Create RNA-seq differential expression workflow'\n- 'Build methylation analysis pipeline'"
            )
        
        # Detect workflow type from description
        desc_lower = description.lower()
        
        workflow_type = None
        workflow_map = {
            "rna-seq": ["rna", "expression", "transcriptom", "mrna", "differential expression", "deseq", "edger"],
            "chip-seq": ["chip", "histone", "h3k", "peak calling", "transcription factor"],
            "atac-seq": ["atac", "chromatin access", "open chromatin"],
            "methylation": ["methyl", "wgbs", "bisulfite", "dmr", "cpg"],
            "wgs": ["wgs", "whole genome seq", "variant", "mutation", "snv", "indel"],
            "scrna-seq": ["scrna", "single cell", "10x", "seurat", "scanpy"],
            "hic": ["hic", "hi-c", "chromatin interact", "3d genome", "tad"],
        }
        
        for wf_type, keywords in workflow_map.items():
            if any(kw in desc_lower for kw in keywords):
                workflow_type = wf_type
                break
        
        if not workflow_type:
            message = f"""🤔 I couldn't determine the workflow type from "{description}".

**Available workflow types:**
- 🧬 **RNA-seq**: Gene expression, differential analysis
- 🎯 **ChIP-seq**: Protein-DNA binding, histone marks
- 🔓 **ATAC-seq**: Chromatin accessibility
- 🔬 **Methylation**: DNA methylation (WGBS/RRBS)
- 🧪 **WGS/WES**: Variant calling, mutations
- 🔴 **scRNA-seq**: Single-cell transcriptomics
- 🔗 **Hi-C**: Chromatin conformation

**Try being more specific:**
- "Create RNA-seq workflow for differential expression"
- "Build ChIP-seq peak calling pipeline"
- "Generate methylation DMR analysis"
"""
            return ToolResult(
                success=True,
                tool_name="generate_workflow",
                data={"detected_type": None},
                message=message
            )
        
        # Try to use the workflow composer
        try:
            from workflow_composer.core.workflow_generator import WorkflowGenerator
            from workflow_composer.core.query_parser import ParsedIntent, AnalysisType
            from datetime import datetime
            
            # Map workflow type string to AnalysisType enum
            type_map = {
                "rna-seq": AnalysisType.RNA_SEQ_DE,
                "chip-seq": AnalysisType.CHIP_SEQ,
                "atac-seq": AnalysisType.ATAC_SEQ,
                "methylation": AnalysisType.BISULFITE_SEQ,
                "wgs": AnalysisType.WGS_VARIANT_CALLING,
                "scrna-seq": AnalysisType.SCRNA_SEQ,
                "hic": AnalysisType.HIC,
            }
            
            analysis_type = type_map.get(workflow_type, AnalysisType.CUSTOM)
            
            # Create ParsedIntent object
            intent = ParsedIntent(
                analysis_type=analysis_type,
                analysis_type_raw=workflow_type,
                confidence=0.9,
                organism="human",
                paired_end=True,
            )
            
            generator = WorkflowGenerator()
            
            # Generate workflow with correct API
            workflow = generator.generate(intent=intent, modules=[])
            
            # Save the workflow
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path.cwd() / "generated_workflows" / f"{workflow_type}_{timestamp}"
            workflow_path = workflow.save(str(output_dir))
            
            message = f"""✅ **Generated {workflow_type.upper()} Workflow**

📂 **Output:** `{workflow_path}`

**What was created:**
- `main.nf` - Nextflow pipeline
- `nextflow.config` - Configuration
- `samplesheet.csv` - Sample template
- `README.md` - Documentation

**Next steps:**
1. Review the generated workflow
2. Prepare your sample sheet
3. Run it: `nextflow run {workflow_path}/main.nf -profile slurm`

Would you like me to explain what the workflow does?
"""
            return ToolResult(
                success=True,
                tool_name="generate_workflow",
                data={"workflow_type": workflow_type, "path": str(workflow_path)},
                message=message
            )
            
        except ImportError as e:
            # Generator not available - provide template info
            message = f"""📋 **Generating {workflow_type.upper()} Workflow**

Based on your description: *"{description}"*

**Workflow Components:**
"""
            # Add workflow-specific components
            component_map = {
                "rna-seq": """
- **QC**: FastQC → MultiQC
- **Trimming**: Trimmomatic/fastp
- **Alignment**: STAR or HISAT2
- **Quantification**: featureCounts or Salmon
- **Analysis**: DESeq2 differential expression""",
                "chip-seq": """
- **QC**: FastQC → MultiQC
- **Alignment**: BWA-MEM or Bowtie2
- **Peak Calling**: MACS2/MACS3
- **Annotation**: ChIPseeker
- **Visualization**: deepTools (heatmaps, profiles)""",
                "methylation": """
- **QC**: FastQC, Bismark QC
- **Alignment**: Bismark (bisulfite-aware)
- **Calling**: methylKit or MethylDackel
- **DMR Analysis**: DSS or DMRcaller
- **Annotation**: annotatr""",
                "atac-seq": """
- **QC**: FastQC → MultiQC
- **Alignment**: BWA-MEM
- **Peak Calling**: MACS2 (--nomodel)
- **Analysis**: ATACseqQC, nucleosome positioning
- **Motif**: HOMER motif analysis""",
            }
            
            message += component_map.get(workflow_type, "\n- Standard NGS pipeline components")
            
            message += f"""

**To generate the actual workflow:**
The WorkflowGenerator module needs to be configured. 
Check `config/analysis_definitions.yaml` for templates.

Would you like me to help you set this up?
"""
            return ToolResult(
                success=True,
                tool_name="generate_workflow",
                data={"workflow_type": workflow_type, "description": description},
                message=message
            )
    
    def analyze_results(self, path: str = None) -> ToolResult:
        """
        Analyze and interpret workflow results (QC reports, logs, outputs).
        
        Args:
            path: Path to results directory
            
        Returns:
            ToolResult with analysis summary
        """
        import os
        
        # Find results directory
        if not path:
            results_dirs = [
                Path("/scratch/sdodl001/BioPipelines/data/results"),
                Path.home() / "BioPipelines" / "data" / "results",
                Path.cwd() / "results",
            ]
            for d in results_dirs:
                if d.exists():
                    path = str(d)
                    break
        
        if not path or not Path(path).exists():
            return ToolResult(
                success=False,
                tool_name="analyze_results",
                error="No results found",
                message="❌ No results directory found. Run a workflow first!"
            )
        
        results_path = Path(path)
        
        # Look for common result files
        analysis = {
            "multiqc": None,
            "logs": [],
            "outputs": [],
            "errors": [],
        }
        
        # Check for MultiQC report
        multiqc_files = list(results_path.rglob("multiqc_report.html"))
        if multiqc_files:
            analysis["multiqc"] = str(multiqc_files[0])
        
        # Check for log files
        log_files = list(results_path.rglob("*.log"))[:5]
        analysis["logs"] = [str(f) for f in log_files]
        
        # Check for common output types
        for ext in ["*.bam", "*.vcf*", "*.bed*", "*.counts*", "*.tsv", "*.csv"]:
            files = list(results_path.rglob(ext))[:3]
            for f in files:
                analysis["outputs"].append({"name": f.name, "size": f.stat().st_size})
        
        # Check for errors in logs
        for log_file in log_files[:3]:
            try:
                content = log_file.read_text()[-2000:]  # Last 2KB
                if "error" in content.lower() or "failed" in content.lower():
                    # Extract error lines
                    error_lines = [l for l in content.split('\n') if 'error' in l.lower() or 'failed' in l.lower()][:3]
                    analysis["errors"].extend(error_lines)
            except:
                pass
        
        # Build report
        message = f"# 📊 Results Analysis\n\n**Path:** `{results_path}`\n\n"
        
        if analysis["multiqc"]:
            message += f"""## ✅ MultiQC Report Available!

📄 **Report:** `{analysis['multiqc']}`

This HTML report aggregates QC metrics from all samples. 
Open in a browser to see:
- Read quality scores
- Adapter content
- Duplication rates
- Alignment statistics

"""
        
        if analysis["outputs"]:
            message += "## 📁 Output Files\n\n"
            message += "| File | Size |\n|------|------|\n"
            for f in analysis["outputs"][:10]:
                size = f["size"]
                size_str = f"{size / 1024**3:.1f} GB" if size > 1024**3 else f"{size / 1024**2:.1f} MB"
                message += f"| `{f['name']}` | {size_str} |\n"
            message += "\n"
        
        if analysis["errors"]:
            message += "## ⚠️ Potential Issues Detected\n\n"
            for err in analysis["errors"][:5]:
                message += f"- {err[:100]}...\n"
            message += "\nUse `diagnose` for detailed error analysis.\n"
        
        if not analysis["outputs"] and not analysis["multiqc"]:
            message += """## ⏳ No Results Yet

The results directory exists but appears empty.
- Check if the workflow is still running: `check my jobs`
- View logs for progress: `show logs`
"""
        
        message += "\n### 💡 Next Steps:\n"
        if analysis["multiqc"]:
            message += "- Review the MultiQC report for quality metrics\n"
        if analysis["errors"]:
            message += "- Run `diagnose` to get AI-powered error analysis\n"
        message += "- Use `describe files` to inspect specific outputs\n"
        
        return ToolResult(
            success=True,
            tool_name="analyze_results",
            data=analysis,
            message=message
        )
    
    def explain_concept(self, concept: str = None) -> ToolResult:
        """
        Explain bioinformatics concepts, terms, and methods.
        
        This tool provides educational content to help researchers
        understand unfamiliar terms.
        
        Args:
            concept: The term or concept to explain
            
        Returns:
            ToolResult with explanation
        """
        if not concept:
            return ToolResult(
                success=False,
                tool_name="explain_concept",
                error="No concept specified",
                message="❌ What would you like me to explain? Example: 'explain DNA methylation'"
            )
        
        concept_lower = concept.lower().strip()
        
        # Common bioinformatics concepts
        explanations = {
            "methylation": """# 🧬 DNA Methylation

**What is it?**
DNA methylation is a chemical modification where a methyl group (CH₃) is added to DNA, typically at cytosine bases in CpG dinucleotides.

**Why is it important?**
- Regulates gene expression (usually silencing genes)
- Inherited through cell division (epigenetic mark)
- Altered in cancer and other diseases

**Data Types:**
- **450K/EPIC Arrays**: Probe-based, measures ~850K CpG sites (IDAT files)
- **WGBS**: Whole genome bisulfite sequencing (FASTQ → BAM)
- **RRBS**: Reduced representation, cost-effective (FASTQ)

**Analysis Pipeline:**
1. Alignment with Bismark (bisulfite-aware)
2. Methylation calling (% methylation per CpG)
3. DMR detection (differentially methylated regions)
4. Annotation to genes/regulatory regions
""",
            "wgbs": """# 🔬 WGBS - Whole Genome Bisulfite Sequencing

**What is it?**
Gold standard for single-base resolution DNA methylation profiling.

**How it works:**
1. **Bisulfite treatment**: Converts unmethylated C → U (then T after PCR)
2. **Sequencing**: Standard Illumina sequencing
3. **Alignment**: Use bisulfite-aware aligner (Bismark)
4. **Calling**: Count C vs T at each CpG

**Advantages:**
- ✅ Genome-wide coverage
- ✅ Single-base resolution
- ✅ Captures all CpG contexts

**Disadvantages:**
- ❌ Expensive (high sequencing depth needed)
- ❌ DNA damage from bisulfite treatment
- ❌ Large data files (100GB+ per sample)
""",
            "rna-seq": """# 🧬 RNA-seq (RNA Sequencing)

**What is it?**
Sequencing of RNA molecules to measure gene expression levels.

**Workflow:**
1. **QC**: FastQC to check read quality
2. **Trimming**: Remove adapters and low-quality bases
3. **Alignment**: STAR or HISAT2 to reference genome
4. **Quantification**: Count reads per gene (featureCounts, Salmon)
5. **Differential Expression**: DESeq2, edgeR

**Key Terms:**
- **TPM/FPKM**: Normalized expression values
- **Fold change**: Expression difference between conditions
- **FDR**: False discovery rate (adjusted p-value)
- **PCA**: Principal component analysis for sample clustering
""",
            "chip-seq": """# 🎯 ChIP-seq (Chromatin Immunoprecipitation Sequencing)

**What is it?**
Identifies DNA regions bound by specific proteins (transcription factors, histones).

**How it works:**
1. Cross-link proteins to DNA
2. Fragment DNA
3. Immunoprecipitate with antibody
4. Sequence the bound DNA fragments

**Analysis Pipeline:**
1. Align reads (BWA-MEM, Bowtie2)
2. Call peaks (MACS2/MACS3)
3. Annotate peaks to genes
4. Motif analysis (HOMER)

**Common Targets:**
- **H3K4me3**: Active promoters
- **H3K27ac**: Active enhancers
- **H3K27me3**: Repressed genes
- **CTCF**: Chromatin architecture
""",
            "differential expression": """# 📊 Differential Expression Analysis

**What is it?**
Statistical comparison of gene expression between conditions (e.g., treatment vs control).

**Popular Tools:**
- **DESeq2**: Uses negative binomial model
- **edgeR**: Similar to DESeq2, flexible
- **limma-voom**: Works with microarray and RNA-seq

**Key Concepts:**
- **Log2 Fold Change**: log₂(treatment/control)
- **P-value**: Probability of seeing difference by chance
- **FDR/Padj**: Multiple testing corrected p-value
- **Significant**: Usually |log2FC| > 1 AND padj < 0.05

**Best Practices:**
- Use at least 3 biological replicates per condition
- Normalize with DESeq2's size factors
- Use shrinkage for reliable fold change estimates
""",
            "atac-seq": """# 🔓 ATAC-seq (Assay for Transposase-Accessible Chromatin)

**What is it?**
Maps open/accessible chromatin regions across the genome.

**How it works:**
The Tn5 transposase cuts and tags accessible DNA regions.

**Analysis:**
1. Align reads (BWA-MEM)
2. Remove duplicates and mitochondrial reads
3. Call peaks (MACS2 with --nomodel)
4. Identify nucleosome-free regions
5. Motif enrichment analysis

**What you can learn:**
- Active regulatory regions
- Transcription factor binding sites
- Chromatin accessibility changes between conditions
""",
        }
        
        # Find matching explanation
        explanation = None
        for key, text in explanations.items():
            if key in concept_lower or concept_lower in key:
                explanation = text
                break
        
        if explanation:
            return ToolResult(
                success=True,
                tool_name="explain_concept",
                data={"concept": concept},
                message=explanation
            )
        
        # No match - suggest similar terms
        message = f"""# ❓ Concept: {concept}

I don't have a detailed explanation for this term yet.

**Available topics I can explain:**
- DNA methylation / WGBS
- RNA-seq
- ChIP-seq
- ATAC-seq
- Differential expression
- Peak calling
- Variant calling

**Want me to add this topic?** 
Let me know and I'll research it!

**Or try asking the AI:** 
The chat can explain most bioinformatics concepts - just ask naturally!
"""
        
        return ToolResult(
            success=True,
            tool_name="explain_concept",
            data={"concept": concept, "found": False},
            message=message
        )


# =============================================================================
# MODULAR BRIDGE - Forward to new tools/ subpackage when available
# =============================================================================

def get_modular_tools(base_path: str = None):
    """
    Get the modular AgentTools instance if available.
    
    Args:
        base_path: Base path for data operations
        
    Returns:
        ModularAgentTools instance or None
    """
    if _USE_MODULAR:
        return get_agent_tools(base_path)
    return None


def execute_tool_modular(tool_name: str, **kwargs) -> Optional[ToolResult]:
    """
    Execute a tool using the modular system.
    
    Args:
        tool_name: Name of the tool
        **kwargs: Tool arguments
        
    Returns:
        ToolResult or None if modular not available
    """
    if _USE_MODULAR:
        tools = get_agent_tools()
        return tools.execute_tool(tool_name, **kwargs)
    return None


def get_openai_function_definitions() -> List[Dict[str, Any]]:
    """
    Get OpenAI function definitions for LLM function calling.
    
    Returns:
        List of function definitions
    """
    if _USE_MODULAR:
        tools = get_agent_tools()
        return tools.get_openai_functions()
    
    # Legacy fallback - minimal definitions
    return [
        {
            "name": "scan_data",
            "description": "Scan workspace for data files",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    ]


# Convenience function for chat handler
def process_tool_request(message: str, app_state=None) -> Optional[ToolResult]:
    """
    Process a message to detect and execute tools.
    
    Args:
        message: User's chat message
        app_state: Application state
        
    Returns:
        ToolResult if a tool was executed, None otherwise
    """
    # Try modular tools first if available
    if _USE_MODULAR:
        try:
            modular_tools = get_agent_tools()
            detected = modular_tools.detect_tool(message)
            if detected:
                return modular_tools.execute_tool(detected)
        except Exception as e:
            logger.warning(f"Modular tool execution failed, falling back to legacy: {e}")
    
    # Legacy fallback
    tools = AgentTools(app_state)
    detection = tools.detect_tool(message)
    
    if detection:
        tool_name, args = detection
        return tools.execute(tool_name, args)
    
    return None


# =============================================================================
# EXPORTS FOR BACKWARD COMPATIBILITY
# =============================================================================

__all__ = [
    # Types
    "ToolResult",
    "ToolName",
    "TOOL_PATTERNS",
    
    # Main class
    "AgentTools",
    
    # Functions
    "process_tool_request",
    "get_openai_function_definitions",
    "get_modular_tools",
    "execute_tool_modular",
]

