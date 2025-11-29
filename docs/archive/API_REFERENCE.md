# BioPipelines API Reference

## Core Classes

### Composer

The main orchestrator class that combines all components.

```python
class Composer:
    """
    AI-powered workflow composer that converts natural language
    to Nextflow pipelines.
    
    Attributes:
        config: Configuration object
        intent_parser: LLM-based intent parser
        tool_selector: Tool catalog query interface
        module_mapper: Nextflow module mapper
        workflow_generator: Nextflow code generator
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        llm_provider: Optional[str] = None
    ):
        """
        Initialize the composer.
        
        Args:
            config_path: Path to YAML configuration file
            llm_provider: Override default LLM provider
                         ('ollama', 'openai', 'anthropic', 'huggingface')
        """
    
    def generate(
        self,
        prompt: str,
        output_dir: Optional[str] = None,
        dry_run: bool = False
    ) -> Workflow:
        """
        Generate a workflow from natural language description.
        
        Args:
            prompt: Natural language description of desired analysis
            output_dir: Directory to save generated workflow
            dry_run: If True, don't save files, just return workflow
            
        Returns:
            Workflow object with generated code
            
        Example:
            >>> workflow = composer.generate(
            ...     "RNA-seq DE analysis for mouse",
            ...     output_dir="my_workflow/"
            ... )
        """
    
    def chat(self, message: str) -> str:
        """
        Interactive chat mode for iterative workflow refinement.
        
        Args:
            message: User message
            
        Returns:
            Assistant response
        """
```

---

### IntentParser

Parses natural language into structured analysis parameters.

```python
class IntentParser:
    """
    LLM-based parser that extracts analysis intent from natural language.
    
    Extracts:
        - Analysis type (rnaseq, chipseq, wgs, etc.)
        - Organism and genome
        - Specific tools requested
        - Analysis parameters
        - Quality thresholds
    """
    
    def __init__(self, llm: LLMAdapter):
        """
        Initialize with an LLM adapter.
        
        Args:
            llm: Any LLM adapter (Ollama, OpenAI, Anthropic, HuggingFace)
        """
    
    def parse(self, prompt: str) -> AnalysisIntent:
        """
        Parse natural language prompt into structured intent.
        
        Args:
            prompt: Natural language description
            
        Returns:
            AnalysisIntent with extracted parameters
            
        Example:
            >>> intent = parser.parse("RNA-seq for mouse liver samples")
            >>> print(intent.analysis_type)  # 'rnaseq'
            >>> print(intent.organism)       # 'mouse'
        """

class AnalysisIntent:
    """Structured representation of parsed analysis intent."""
    analysis_type: AnalysisType
    organism: str
    genome_build: str
    tools: List[str]
    params: Dict[str, Any]
    data_type: str
    paired_end: bool
```

---

### ToolSelector

Queries the tool catalog to find appropriate tools.

```python
class ToolSelector:
    """
    Interface to query the 9,909-tool catalog.
    
    Supports:
        - Fuzzy text search
        - Category filtering
        - Container-specific queries
        - Tool metadata retrieval
    """
    
    def __init__(self, catalog_path: str):
        """
        Initialize with path to tool catalog.
        
        Args:
            catalog_path: Path to tool_catalog directory or JSON file
        """
    
    def search(
        self,
        query: str,
        limit: int = 20,
        container: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Tool]:
        """
        Search for tools matching query.
        
        Args:
            query: Search query (tool name, keyword, description)
            limit: Maximum results to return
            container: Filter by container name
            category: Filter by category
            
        Returns:
            List of matching Tool objects
            
        Example:
            >>> tools = selector.search("alignment", container="rna-seq")
            >>> for t in tools:
            ...     print(f"{t.name}: {t.description}")
        """
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a specific tool by name."""
    
    def list_containers(self) -> List[str]:
        """List all available containers."""
    
    def list_categories(self) -> List[str]:
        """List all tool categories."""
    
    def get_tools_for_analysis(
        self,
        analysis_type: str
    ) -> Dict[str, List[Tool]]:
        """
        Get recommended tools for an analysis type.
        
        Args:
            analysis_type: Type of analysis (rnaseq, chipseq, wgs, etc.)
            
        Returns:
            Dict mapping workflow steps to recommended tools
        """

class Tool:
    """Tool metadata."""
    name: str
    version: str
    container: str
    category: str
    description: str
    binary_path: str
```

---

### ModuleMapper

Maps tools to Nextflow modules.

```python
class ModuleMapper:
    """
    Maps bioinformatics tools to Nextflow DSL2 modules.
    
    Features:
        - Automatic module discovery
        - Tool name aliasing
        - Process extraction
        - Module validation
    """
    
    def __init__(
        self,
        module_dir: str,
        additional_dirs: List[str] = None
    ):
        """
        Initialize with module directories.
        
        Args:
            module_dir: Primary modules directory
            additional_dirs: Additional directories to scan
        """
    
    def find_module(self, tool_name: str) -> Optional[Module]:
        """
        Find module for a tool.
        
        Args:
            tool_name: Name of tool (supports aliases)
            
        Returns:
            Module if found, None otherwise
            
        Example:
            >>> module = mapper.find_module("bwa")
            >>> print(module.name)       # 'bwamem'
            >>> print(module.processes)  # ['BWA_MEM']
        """
    
    def find_modules_for_tools(
        self,
        tool_names: List[str]
    ) -> Dict[str, Optional[Module]]:
        """Find modules for multiple tools."""
    
    def get_missing_tools(self, tool_names: List[str]) -> List[str]:
        """Get tools without available modules."""
    
    def list_modules(self) -> List[str]:
        """List all available modules."""
    
    def list_by_category(self) -> Dict[str, List[str]]:
        """List modules organized by category."""
    
    def create_module(
        self,
        tool_name: str,
        container: str,
        llm: LLMAdapter,
        description: str = ""
    ) -> Module:
        """
        Auto-generate a new module using LLM.
        
        Args:
            tool_name: Name of tool
            container: Container to use
            llm: LLM adapter for code generation
            description: Additional requirements
            
        Returns:
            Generated Module
        """

    # Class attribute - tool name aliases
    TOOL_ALIASES = {
        "bwa": "bwamem",
        "gatk": "gatk_haplotypecaller",
        "picard": "markduplicates",
        # ... more aliases
    }

class Module:
    """Nextflow module metadata."""
    name: str
    path: Path
    tool_name: str
    container: str
    processes: List[str]
    inputs: List[str]
    outputs: List[str]
    
    def get_import_statement(self) -> str:
        """Generate Nextflow import statement."""
```

---

### WorkflowGenerator

Generates complete Nextflow workflows.

```python
class WorkflowGenerator:
    """
    Generates production-ready Nextflow DSL2 workflows.
    
    Outputs:
        - main.nf: Main workflow file
        - nextflow.config: Configuration
        - params.yaml: Default parameters
    """
    
    def __init__(self, module_dir: str):
        """
        Initialize with module directory.
        
        Args:
            module_dir: Path to Nextflow modules
        """
    
    def generate(
        self,
        analysis_type: str,
        tools: List[str],
        organism: str,
        params: Dict[str, Any] = None,
        name: str = None
    ) -> Workflow:
        """
        Generate a complete workflow.
        
        Args:
            analysis_type: Type of analysis
            tools: List of tools to include
            organism: Target organism
            params: Additional parameters
            name: Workflow name
            
        Returns:
            Workflow object with generated code
        """
    
    def generate_main_nf(
        self,
        modules: List[Module],
        workflow_logic: str
    ) -> str:
        """Generate main.nf content."""
    
    def generate_config(
        self,
        containers: List[str],
        resources: Dict[str, Any] = None
    ) -> str:
        """Generate nextflow.config content."""

class Workflow:
    """Generated workflow."""
    name: str
    main_nf: str
    config: str
    params: Dict[str, Any]
    
    def save(self, output_dir: str) -> None:
        """Save workflow files to directory."""
    
    def validate(self) -> Dict[str, Any]:
        """Validate workflow syntax."""
```

---

## LLM Adapters

### Base Adapter

```python
class LLMAdapter(ABC):
    """Abstract base class for LLM adapters."""
    
    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.1,
        max_tokens: int = 4096
    ) -> Response:
        """Send chat messages and get response."""
    
    @abstractmethod
    def complete(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4096
    ) -> str:
        """Complete a prompt."""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""

class Message:
    """Chat message."""
    role: str  # 'system', 'user', 'assistant'
    content: str
    
    @classmethod
    def system(cls, content: str) -> "Message": ...
    
    @classmethod
    def user(cls, content: str) -> "Message": ...
    
    @classmethod
    def assistant(cls, content: str) -> "Message": ...

class Response:
    """LLM response."""
    content: str
    model: str
    usage: Dict[str, int]
```

### Available Adapters

```python
# Ollama (local)
from src.workflow_composer.llm import OllamaAdapter
adapter = OllamaAdapter(
    host="http://localhost:11434",
    model="llama3:8b"
)

# OpenAI
from src.workflow_composer.llm import OpenAIAdapter
adapter = OpenAIAdapter(
    api_key="sk-...",
    model="gpt-4-turbo-preview"
)

# Anthropic
from src.workflow_composer.llm import AnthropicAdapter
adapter = AnthropicAdapter(
    api_key="sk-ant-...",
    model="claude-3-opus-20240229"
)

# HuggingFace
from src.workflow_composer.llm import HuggingFaceAdapter
adapter = HuggingFaceAdapter(
    model="codellama/CodeLlama-13b-Instruct-hf",
    device="cuda"
)
```

---

## Data Utilities

### DataDownloader

```python
class DataDownloader:
    """Download reference data and annotations."""
    
    def __init__(self, cache_dir: str = "data/references/"):
        """Initialize with cache directory."""
    
    def download_genome(
        self,
        organism: str,
        build: str,
        source: str = "ensembl"
    ) -> Path:
        """
        Download genome FASTA.
        
        Args:
            organism: Species name
            build: Genome build (GRCh38, GRCm39, etc.)
            source: Data source (ensembl, ucsc, gencode)
            
        Returns:
            Path to downloaded file
        """
    
    def download_annotation(
        self,
        organism: str,
        build: str,
        format: str = "gtf"
    ) -> Path:
        """Download gene annotation."""
    
    def download_index(
        self,
        tool: str,
        organism: str,
        build: str
    ) -> Path:
        """Download pre-built aligner index."""
```

### WorkflowVisualizer

```python
class WorkflowVisualizer:
    """Generate workflow visualizations."""
    
    def __init__(self, output_dir: str = "visualizations/"):
        """Initialize with output directory."""
    
    def generate_dag(
        self,
        workflow: Workflow,
        format: str = "png"
    ) -> Path:
        """
        Generate DAG visualization.
        
        Args:
            workflow: Workflow to visualize
            format: Output format (png, svg, pdf)
            
        Returns:
            Path to generated image
        """
    
    def generate_mermaid(self, workflow: Workflow) -> str:
        """Generate Mermaid diagram code."""
```

---

## Configuration

### Config Class

```python
class Config:
    """Main configuration class."""
    
    llm: LLMConfig
    knowledge_base: KnowledgeBaseConfig
    data: DataConfig
    output: OutputConfig
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """Load configuration from YAML file."""
    
    def resolve_path(self, path: str) -> Path:
        """Resolve relative path against base path."""
    
    def get_llm_config(
        self,
        provider: Optional[str] = None
    ) -> LLMProviderConfig:
        """Get LLM provider configuration."""

class LLMConfig:
    default_provider: str
    providers: Dict[str, LLMProviderConfig]

class KnowledgeBaseConfig:
    tool_catalog: str
    module_library: str
    workflow_patterns: str
    container_images: str
```

---

## CLI Commands

```bash
# Generate workflow
biocomposer generate "description" [--output DIR] [--provider NAME]

# Interactive chat
biocomposer chat [--provider NAME]

# Search tools
biocomposer tools [--search QUERY] [--container NAME] [--limit N]

# List modules
biocomposer modules [--category NAME]

# Check LLM providers
biocomposer providers [--check]

# Visualize workflow
biocomposer visualize WORKFLOW_DIR [--format FORMAT]
```

---

## Enums

```python
class AnalysisType(Enum):
    """Supported analysis types."""
    RNASEQ = "rnaseq"
    CHIPSEQ = "chipseq"
    ATACSEQ = "atacseq"
    WGS = "wgs"
    WES = "wes"
    SCRNASEQ = "scrnaseq"
    METAGENOMICS = "metagenomics"
    METHYLATION = "methylation"
    LONGREAD = "longread"
    HIC = "hic"
    ASSEMBLY = "assembly"
    ANNOTATION = "annotation"
```
