"""
BioPipelines Façade
===================

The single, unified entry point for BioPipelines.

This provides a clean, simple API that hides the complexity of the underlying
components while providing access to all functionality.

Usage:
    from workflow_composer import BioPipelines
    
    # Initialize
    bp = BioPipelines()
    
    # Chat mode (natural language)
    response = bp.chat("Create an RNA-seq differential expression workflow")
    
    # Direct workflow generation
    workflow = bp.generate("RNA-seq differential expression", 
                           samples=["sample1.fq", "sample2.fq"])
    
    # Submit to cluster
    job = bp.submit(workflow)
    
    # Check status
    status = bp.status(job.id)
    
    # Scan for data
    samples = bp.scan_data("/path/to/fastq")
    
    # Search tools
    tools = bp.find_tools("alignment")
    
Advanced Usage:
    # Custom configuration
    bp = BioPipelines(
        llm_provider="openai",
        llm_model="gpt-4o",
        cluster="slurm",
    )
    
    # Access underlying components
    bp.composer.generate(...)
    bp.agent.process(...)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import logging

logger = logging.getLogger(__name__)

# Type checking imports
if TYPE_CHECKING:
    from .core import Workflow, Tool, ParsedIntent
    from .agents import AgentResponse
    from .data import DataManifest


@dataclass
class JobStatus:
    """Status of a submitted job."""
    
    job_id: str
    """SLURM job ID or local process ID."""
    
    state: str
    """Job state: pending, running, completed, failed, cancelled."""
    
    progress: float
    """Progress 0.0 to 1.0."""
    
    message: str
    """Human-readable status message."""
    
    outputs: List[str] = field(default_factory=list)
    """List of output file paths."""
    
    errors: List[str] = field(default_factory=list)
    """List of error messages if any."""
    
    @property
    def is_complete(self) -> bool:
        """Check if job is finished (success or failure)."""
        return self.state in ("completed", "failed", "cancelled")
    
    @property
    def is_success(self) -> bool:
        """Check if job completed successfully."""
        return self.state == "completed"


@dataclass
class ChatResponse:
    """Response from chat interaction."""
    
    message: str
    """The assistant's response message."""
    
    workflow: Optional["Workflow"] = None
    """Generated workflow if any."""
    
    tools_used: List[str] = field(default_factory=list)
    """Tools that were executed."""
    
    suggestions: List[str] = field(default_factory=list)
    """Follow-up suggestions."""
    
    correlation_id: str = ""
    """Request correlation ID for tracking."""


class BioPipelines:
    """
    The unified entry point for BioPipelines.
    
    Provides a simple, consistent API for:
    - Natural language workflow generation
    - Data discovery and management
    - Job submission and monitoring
    - Tool catalog access
    
    Example:
        bp = BioPipelines()
        
        # Generate a workflow
        workflow = bp.generate("RNA-seq analysis for human samples")
        
        # Or use chat for interactive workflow creation
        response = bp.chat("I need to analyze ChIP-seq data")
        
        # Submit to cluster
        job = bp.submit(workflow, cluster="slurm")
        
        # Monitor
        status = bp.status(job.id)
    """
    
    def __init__(
        self,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        cluster: str = "local",
        config_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize BioPipelines.
        
        Args:
            llm_provider: LLM provider ("lightning", "openai", "ollama", etc.)
            llm_model: Model name (provider-specific)
            cluster: Execution target ("local", "slurm")
            config_path: Path to configuration file
            **kwargs: Additional configuration options
        """
        # Lazy imports to avoid circular dependencies
        from .config import Config
        
        # Load configuration
        self._config = Config.load(config_path)
        
        # Override with explicit parameters
        if llm_provider:
            self._config.llm.default_provider = llm_provider
        if llm_model:
            self._config.llm.intent_parser_model = llm_model
        
        self._cluster = cluster
        self._kwargs = kwargs
        
        # Lazy initialization
        self._composer = None
        self._agent = None
        self._initialized = False
        
        logger.info(f"BioPipelines initialized (llm={llm_provider or 'default'})")
    
    def _ensure_initialized(self) -> None:
        """Lazily initialize components."""
        if self._initialized:
            return
        
        from .composer import Composer
        from .llm import get_llm
        
        # Initialize LLM
        provider = self._config.llm.default_provider
        model = self._config.llm.intent_parser_model
        llm = get_llm(provider, model)
        
        # Initialize composer
        self._composer = Composer(llm=llm, config=self._config)
        
        self._initialized = True
    
    @property
    def composer(self) -> "Composer":
        """Access the underlying Composer for advanced usage."""
        self._ensure_initialized()
        return self._composer
    
    @property
    def agent(self) -> "UnifiedAgent":
        """Access the underlying UnifiedAgent for advanced usage."""
        if self._agent is None:
            from .agents import UnifiedAgent
            self._agent = UnifiedAgent()
        return self._agent
    
    # =========================================================================
    # Core API - Workflow Generation
    # =========================================================================
    
    def generate(
        self,
        description: str,
        samples: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        auto_create_modules: bool = True,
        **options
    ) -> "Workflow":
        """
        Generate a workflow from a natural language description.
        
        This is the primary method for creating bioinformatics workflows.
        
        Args:
            description: Natural language description of the analysis
            samples: Optional list of sample file paths
            output_dir: Directory to save the workflow
            auto_create_modules: Auto-create missing modules with LLM
            **options: Additional options passed to workflow generator
            
        Returns:
            Generated Workflow object
            
        Example:
            workflow = bp.generate(
                "RNA-seq differential expression for mouse",
                samples=["sample1.fq.gz", "sample2.fq.gz"],
                output_dir="workflows/rnaseq"
            )
        """
        self._ensure_initialized()
        
        # Create data manifest if samples provided
        data_manifest = None
        if samples:
            from .data import DataManifest, Sample
            manifest_samples = [
                Sample(path=Path(s), sample_id=Path(s).stem)
                for s in samples
            ]
            data_manifest = DataManifest(samples=manifest_samples)
        
        return self._composer.generate(
            description=description,
            output_dir=output_dir,
            auto_create_modules=auto_create_modules,
            data_manifest=data_manifest,
            **options
        )
    
    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> ChatResponse:
        """
        Chat interface for natural language interaction.
        
        Provides a conversational interface for:
        - Asking questions about bioinformatics
        - Generating workflows step by step
        - Getting help with errors
        - Exploring the tool catalog
        
        Args:
            message: User's message
            history: Optional conversation history
            
        Returns:
            ChatResponse with the assistant's reply
            
        Example:
            response = bp.chat("What tools do I need for ChIP-seq analysis?")
            print(response.message)
            
            # Follow-up
            response = bp.chat("Generate that workflow", history=...)
        """
        from .agents import UnifiedAgent
        
        agent = self.agent
        
        # Process message
        result = agent.process_sync(message)
        
        return ChatResponse(
            message=result.message,
            workflow=result.workflow if hasattr(result, "workflow") else None,
            tools_used=result.tools_used if hasattr(result, "tools_used") else [],
            suggestions=result.suggestions if hasattr(result, "suggestions") else [],
        )
    
    def parse_intent(self, description: str) -> "ParsedIntent":
        """
        Parse a description to extract intent without generating workflow.
        
        Useful for understanding what the system detected from a query.
        
        Args:
            description: Natural language description
            
        Returns:
            ParsedIntent with analysis type, organism, etc.
        """
        self._ensure_initialized()
        return self._composer.parse_intent(description)
    
    # =========================================================================
    # Core API - Data Discovery
    # =========================================================================
    
    def scan_data(
        self,
        path: Union[str, Path],
        pattern: str = "*.f*q*",
    ) -> "DataManifest":
        """
        Scan a directory for sequencing data files.
        
        Automatically detects:
        - Sample names from filenames
        - Paired-end vs single-end
        - File formats (FASTQ, FASTA, BAM, etc.)
        
        Args:
            path: Directory to scan
            pattern: File pattern to match
            
        Returns:
            DataManifest with discovered samples
            
        Example:
            manifest = bp.scan_data("/data/raw/project1")
            print(f"Found {len(manifest.samples)} samples")
        """
        from .data import LocalSampleScanner, DataManifest
        
        scanner = LocalSampleScanner()
        samples = scanner.scan_directory(str(path), pattern=pattern)
        
        return DataManifest(samples=samples)
    
    # =========================================================================
    # Core API - Job Execution
    # =========================================================================
    
    def submit(
        self,
        workflow: "Workflow",
        cluster: Optional[str] = None,
        **slurm_options
    ) -> JobStatus:
        """
        Submit a workflow for execution.
        
        Args:
            workflow: Workflow to execute
            cluster: Execution target ("local", "slurm")
            **slurm_options: SLURM-specific options
            
        Returns:
            JobStatus with job_id for tracking
            
        Example:
            job = bp.submit(workflow, cluster="slurm", partition="gpu")
            print(f"Submitted job {job.job_id}")
        """
        cluster = cluster or self._cluster
        
        if cluster == "local":
            return self._submit_local(workflow)
        elif cluster == "slurm":
            return self._submit_slurm(workflow, **slurm_options)
        else:
            raise ValueError(f"Unknown cluster type: {cluster}")
    
    def _submit_local(self, workflow: "Workflow") -> JobStatus:
        """Submit workflow for local execution."""
        # For now, just generate the workflow
        import uuid
        job_id = f"local-{uuid.uuid4().hex[:8]}"
        
        return JobStatus(
            job_id=job_id,
            state="pending",
            progress=0.0,
            message=f"Workflow ready at {workflow.output_dir}",
        )
    
    def _submit_slurm(self, workflow: "Workflow", **options) -> JobStatus:
        """Submit workflow to SLURM."""
        from .agents import AgentTools
        
        # Generate SLURM submission script
        script_path = workflow.output_dir / "submit.sh"
        
        # Use agent tools to submit
        result = AgentTools.slurm_submit(str(script_path), **options)
        
        if result.success:
            return JobStatus(
                job_id=result.output.get("job_id", "unknown"),
                state="pending",
                progress=0.0,
                message="Job submitted to SLURM",
            )
        else:
            return JobStatus(
                job_id="",
                state="failed",
                progress=0.0,
                message=f"Submission failed: {result.error}",
                errors=[result.error],
            )
    
    def status(self, job_id: str) -> JobStatus:
        """
        Get status of a submitted job.
        
        Args:
            job_id: Job ID from submit()
            
        Returns:
            JobStatus with current state
        """
        if job_id.startswith("local-"):
            return JobStatus(
                job_id=job_id,
                state="completed",
                progress=1.0,
                message="Local job completed",
            )
        
        # Query SLURM
        from .agents import AgentTools
        
        result = AgentTools.slurm_status(job_id)
        
        if result.success:
            slurm_state = result.output.get("state", "UNKNOWN")
            state_map = {
                "PENDING": "pending",
                "RUNNING": "running",
                "COMPLETED": "completed",
                "FAILED": "failed",
                "CANCELLED": "cancelled",
            }
            
            return JobStatus(
                job_id=job_id,
                state=state_map.get(slurm_state, "unknown"),
                progress=1.0 if slurm_state == "COMPLETED" else 0.5,
                message=result.output.get("message", ""),
            )
        else:
            return JobStatus(
                job_id=job_id,
                state="unknown",
                progress=0.0,
                message=f"Status query failed: {result.error}",
            )
    
    def cancel(self, job_id: str) -> bool:
        """
        Cancel a running job.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if cancellation succeeded
        """
        if job_id.startswith("local-"):
            return True
        
        from .agents import AgentTools
        
        result = AgentTools.slurm_cancel(job_id)
        return result.success
    
    # =========================================================================
    # Core API - Tool Discovery
    # =========================================================================
    
    def find_tools(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List["Tool"]:
        """
        Search the tool catalog.
        
        Args:
            query: Search query (tool name, description, etc.)
            category: Optional category filter
            limit: Maximum results
            
        Returns:
            List of matching tools
            
        Example:
            tools = bp.find_tools("alignment")
            for tool in tools:
                print(f"{tool.name}: {tool.description}")
        """
        self._ensure_initialized()
        
        return self._composer.tool_selector.search(
            query,
            category=category,
            limit=limit
        )
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """
        List available tools.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tool names
        """
        self._ensure_initialized()
        
        if category:
            return self._composer.tool_selector.list_by_category(category)
        else:
            return self._composer.tool_selector.list_all()
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def check_readiness(self, description: str) -> Dict[str, Any]:
        """
        Check if system is ready to generate a workflow.
        
        Validates:
        - Required tools are available
        - Container images exist
        - Modules are present
        
        Args:
            description: Workflow description
            
        Returns:
            Dict with readiness status and any issues
        """
        self._ensure_initialized()
        return self._composer.check_readiness(description)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about available resources.
        
        Returns:
            Dict with tool count, module count, etc.
        """
        self._ensure_initialized()
        return self._composer.get_stats()
    
    def switch_llm(self, provider: str, model: Optional[str] = None) -> None:
        """
        Switch to a different LLM provider.
        
        Args:
            provider: Provider name ("lightning", "openai", etc.)
            model: Optional model name
        """
        self._ensure_initialized()
        self._composer.switch_llm(provider, model)
    
    # =========================================================================
    # Context Manager Support
    # =========================================================================
    
    def __enter__(self) -> "BioPipelines":
        """Support context manager usage."""
        return self
    
    def __exit__(self, *args) -> None:
        """Cleanup on exit."""
        pass
    
    def __repr__(self) -> str:
        provider = self._config.llm.default_provider if self._config else "unknown"
        return f"BioPipelines(llm={provider}, cluster={self._cluster})"


# =============================================================================
# Convenience Functions
# =============================================================================

_default_instance: Optional[BioPipelines] = None


def get_biopipelines(**kwargs) -> BioPipelines:
    """
    Get or create the default BioPipelines instance.
    
    For simple scripts that just need one instance.
    
    Example:
        bp = get_biopipelines()
        workflow = bp.generate("RNA-seq")
    """
    global _default_instance
    
    if _default_instance is None:
        _default_instance = BioPipelines(**kwargs)
    
    return _default_instance


def generate(description: str, **kwargs) -> "Workflow":
    """
    Quick function for workflow generation.
    
    Example:
        workflow = generate("RNA-seq for human samples")
    """
    return get_biopipelines().generate(description, **kwargs)


def chat(message: str) -> ChatResponse:
    """
    Quick function for chat interaction.
    
    Example:
        response = chat("What is RNA-seq?")
    """
    return get_biopipelines().chat(message)
