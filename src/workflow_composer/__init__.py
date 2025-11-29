"""
BioPipelines AI Workflow Composer
=================================

An intelligent system for generating Nextflow bioinformatics pipelines
from natural language descriptions.

RECOMMENDED USAGE (v2.0+):
==========================

    from workflow_composer import BioPipelines
    
    # Single entry point for everything
    bp = BioPipelines()
    
    # Generate workflow
    workflow = bp.generate("RNA-seq differential expression for human samples")
    
    # Or use chat interface
    response = bp.chat("Create a ChIP-seq workflow")
    
    # Scan for data
    manifest = bp.scan_data("/path/to/fastq")
    
    # Submit to cluster
    job = bp.submit(workflow, cluster="slurm")
    status = bp.status(job.id)

LEGACY USAGE (still supported):
==============================

    from workflow_composer import Composer
    from workflow_composer.llm import get_llm
    
    composer = Composer(llm=get_llm("ollama", "llama3"))
    workflow = composer.generate("RNA-seq differential expression")
    workflow.save("my_workflow/")

Features:
- Pluggable LLM backend (OpenAI, Anthropic, Ollama, HuggingFace, Lightning.ai)
- Intent parsing from natural language
- Tool selection from 9,909+ cataloged tools
- Automatic module mapping and creation
- Workflow generation using DSL2 patterns
- Reference data management
- SLURM job submission and monitoring
- Error diagnosis and auto-recovery

CLI Usage:
    $ biocomposer generate "RNA-seq analysis for mouse samples"
    $ biocomposer chat --llm ollama
    $ biocomposer tools --search star
    $ biocomposer modules --list
"""

__version__ = "2.0.0"
__author__ = "BioPipelines Team"

# Auto-load secrets from .secrets/ directory
from . import secrets

# ============================================================================
# NEW: Unified Entry Point (v2.0)
# ============================================================================
from .facade import (
    BioPipelines,
    JobStatus,
    ChatResponse,
    get_biopipelines,
    generate,
    chat,
)

# ============================================================================
# Infrastructure Layer (v2.0)
# ============================================================================
try:
    from .infrastructure import (
        # Exceptions
        BioPipelinesError,
        ConfigurationError,
        LLMError,
        ToolNotFoundError,
        SLURMError,
        # DI Container
        Container,
        get_container,
        inject,
        # Protocols
        LLMProtocol,
        # Settings
        Settings,
        get_settings,
        # Logging
        get_logger,
        operation_context,
    )
    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_AVAILABLE = False

# ============================================================================
# Legacy: Main Composer (still fully supported)
# ============================================================================
from .composer import Composer
from .config import Config

# LLM adapters
from .llm import (
    LLMAdapter,
    OllamaAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    HuggingFaceAdapter,
    get_llm,
    list_providers
)

# Core components
from .core import (
    IntentParser,
    ParsedIntent,
    AnalysisType,
    ToolSelector,
    Tool,
    ModuleMapper,
    Module,
    WorkflowGenerator,
    Workflow
)

# Data management
from .data import (
    DataDownloader,
    Reference
)

# Visualization
from .viz import (
    WorkflowVisualizer
)

# Monitoring
from .monitor import (
    WorkflowMonitor,
    WorkflowExecution,
    ProcessExecution,
    WorkflowStatus,
    ProcessStatus
)

__all__ = [
    # ========== NEW: Primary API (v2.0) ==========
    "BioPipelines",        # THE recommended entry point
    "JobStatus",
    "ChatResponse",
    "get_biopipelines",
    "generate",
    "chat",
    
    # Infrastructure
    "BioPipelinesError",
    "ConfigurationError",
    "LLMError",
    "ToolNotFoundError",
    "SLURMError",
    "Container",
    "get_container",
    "inject",
    "LLMProtocol",
    "Settings",
    "get_settings",
    "get_logger",
    "operation_context",
    
    # ========== Legacy API (still supported) ==========
    # Main
    "Composer",
    "Config",
    "__version__",
    
    # LLM
    "LLMAdapter",
    "OllamaAdapter", 
    "OpenAIAdapter",
    "AnthropicAdapter",
    "HuggingFaceAdapter",
    "get_llm",
    "list_providers",
    
    # Core
    "IntentParser",
    "ParsedIntent",
    "AnalysisType",
    "ToolSelector",
    "Tool",
    "ModuleMapper",
    "Module",
    "WorkflowGenerator",
    "Workflow",
    
    # Data
    "DataDownloader",
    "Reference",
    
    # Visualization
    "WorkflowVisualizer",
    
    # Monitoring
    "WorkflowMonitor",
    "WorkflowExecution",
    "ProcessExecution",
    "WorkflowStatus",
    "ProcessStatus",
]
