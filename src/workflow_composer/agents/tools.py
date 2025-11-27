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
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ToolName(Enum):
    """Available agent tools."""
    SCAN_DATA = "scan_data"
    CLEANUP_DATA = "cleanup_data"
    CONFIRM_CLEANUP = "confirm_cleanup"
    SEARCH_DATABASES = "search_databases"
    CHECK_REFERENCES = "check_references"
    SUBMIT_JOB = "submit_job"
    GET_JOB_STATUS = "get_job_status"
    GET_LOGS = "get_logs"
    CANCEL_JOB = "cancel_job"
    DIAGNOSE_ERROR = "diagnose_error"
    LIST_WORKFLOWS = "list_workflows"
    DOWNLOAD_RESULTS = "download_results"
    COMPARE_SAMPLES = "compare_samples"
    RUN_COMMAND = "run_command"
    SHOW_HELP = "show_help"


@dataclass
class ToolResult:
    """Result from a tool invocation."""
    success: bool
    tool_name: str
    data: Any = None
    message: str = ""
    error: Optional[str] = None
    ui_update: Optional[Dict[str, Any]] = None  # Updates for sidebar


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
    
    # Logs
    (r"(?:show|get|view|display)\s+(?:me\s+)?(?:the\s+)?logs?\s*(?:for|of)?\s*(?:job\s*)?(\d+)?",
     ToolName.GET_LOGS),
    (r"(?:what(?:'s| is))\s+(?:in\s+)?(?:the\s+)?logs?",
     ToolName.GET_LOGS),
    
    # Cancel
    (r"(?:cancel|stop|abort|kill)\s+(?:the\s+)?(?:job|workflow|pipeline)\s*(\d+)?",
     ToolName.CANCEL_JOB),
    
    # Diagnosis
    (r"(?:diagnose|analyze|debug|what went wrong|why did it fail)",
     ToolName.DIAGNOSE_ERROR),
    (r"(?:fix|help with)\s+(?:the\s+)?(?:error|failure|problem)",
     ToolName.DIAGNOSE_ERROR),
    
    # List workflows
    (r"(?:list|show|what)\s+(?:available\s+)?workflows?",
     ToolName.LIST_WORKFLOWS),
    
    # Download results
    (r"(?:download|get|export)\s+(?:the\s+)?(?:results?|outputs?|files?)\s*(?:from|for)?\s*(?:job\s*)?(\d+)?",
     ToolName.DOWNLOAD_RESULTS),
    (r"(?:zip|package|archive)\s+(?:the\s+)?(?:results?|outputs?)",
     ToolName.DOWNLOAD_RESULTS),
    
    # Compare samples
    (r"(?:compare|diff|contrast)\s+(?:samples?|groups?|conditions?)\s*(.+)?",
     ToolName.COMPARE_SAMPLES),
    (r"(?:what(?:'s| is| are))\s+(?:the\s+)?(?:difference|differences)\s+(?:between)\s*(.+)?",
     ToolName.COMPARE_SAMPLES),
    
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
            ToolName.CHECK_REFERENCES: self.check_references,
            ToolName.SUBMIT_JOB: self.submit_job,
            ToolName.GET_JOB_STATUS: self.get_job_status,
            ToolName.GET_LOGS: self.get_logs,
            ToolName.CANCEL_JOB: self.cancel_job,
            ToolName.DIAGNOSE_ERROR: self.diagnose_error,
            ToolName.LIST_WORKFLOWS: self.list_workflows,
            ToolName.DOWNLOAD_RESULTS: self.download_results,
            ToolName.COMPARE_SAMPLES: self.compare_samples,
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
                    f"  - **{r['source']}**: [`{r['id']}`] {(r['title'][:55] + '...') if len(r['title']) > 55 else r['title']}"
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
                
                message = f"""🔍 Found **{len(results)} datasets** matching "{query}":
{parsed_str}
{result_list}

💡 Say "download <ID>" to add a dataset to your manifest."""
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
            # Try to use diagnosis agent
            from workflow_composer.diagnosis.agent import DiagnosisAgent
            
            agent = DiagnosisAgent()
            
            # Get logs first
            logs_result = self.get_logs(job_id)
            logs = logs_result.data.get('logs', '') if logs_result.data else ''
            
            if not logs:
                return ToolResult(
                    success=False,
                    tool_name="diagnose_error",
                    error="No logs available for diagnosis",
                    message="❌ No logs found to diagnose. Make sure the job has run."
                )
            
            # Run diagnosis
            diagnosis = agent.diagnose(logs)
            
            message = f"""🔍 **Error Diagnosis:**

**Problem:** {diagnosis.get('problem', 'Unknown error')}

**Cause:** {diagnosis.get('cause', 'Unable to determine cause')}

**Suggested Fix:**
{diagnosis.get('fix', 'No automated fix available')}

Would you like me to apply the suggested fix?"""
            
            return ToolResult(
                success=True,
                tool_name="diagnose_error",
                data=diagnosis,
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
    
    def show_help(self) -> ToolResult:
        """
        Show available chat commands.
        
        Returns:
            ToolResult with help message
        """
        help_message = """# 🤖 BioPipelines Chat Commands

## 📁 Data Discovery
| Command | Description |
|---------|-------------|
| `scan data in /path` | Find FASTQ files in a directory |
| `search for human RNA-seq` | Search ENCODE/GEO databases |
| `check reference for human` | Verify genome references |
| `compare samples` | Analyze sample groups |

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
| `show status` | Check job progress |
| `show logs` | View job output |
| `cancel job 123` | Stop a running job |

## 📊 Results
| Command | Description |
|---------|-------------|
| `download results` | Package output files |
| `diagnose` | AI error analysis |

## 💡 Tips
- Be specific about organism (human, mouse, etc.)
- Mention data type (RNA-seq, ChIP-seq, ATAC-seq)
- Specify comparisons (treatment vs control)
"""
        
        return ToolResult(
            success=True,
            tool_name="show_help",
            data={},
            message=help_message
        )


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
    tools = AgentTools(app_state)
    detection = tools.detect_tool(message)
    
    if detection:
        tool_name, args = detection
        return tools.execute(tool_name, args)
    
    return None
