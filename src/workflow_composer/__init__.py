"""
BioPipelines AI Workflow Composer
=================================

An intelligent system for generating Nextflow bioinformatics pipelines
from natural language descriptions.

Features:
- Pluggable LLM backend (OpenAI, Anthropic, Ollama, HuggingFace)
- Intent parsing from natural language
- Tool selection from 9,909+ cataloged tools
- Automatic module mapping and creation
- Workflow generation using DSL2 patterns
- Reference data management
- Visualization and reporting

Example:
    from workflow_composer import Composer
    from workflow_composer.llm import get_llm
    
    # Use local Ollama (free, private)
    composer = Composer(llm=get_llm("ollama", "llama3"))
    
    # Or use OpenAI
    composer = Composer(llm=get_llm("openai", "gpt-4"))
    
    # Generate workflow from description
    workflow = composer.generate("RNA-seq differential expression analysis")
    workflow.save("my_workflow/")

Quick Start:
    # CLI usage
    $ biocomposer generate "RNA-seq analysis for mouse samples"
    $ biocomposer chat --llm ollama
    $ biocomposer tools --search star
    $ biocomposer modules --list
"""

__version__ = "0.1.0"
__author__ = "BioPipelines Team"

# Main composer
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
    "ProcessStatus"
]
