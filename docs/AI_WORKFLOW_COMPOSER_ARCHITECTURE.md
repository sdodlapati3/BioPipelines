# AI Workflow Composer - Architecture & Implementation Plan

**Version:** 1.0  
**Date:** November 25, 2025  
**Status:** Phase 3 Implementation

---

## Executive Summary

The AI Workflow Composer is an intelligent system that transforms natural language descriptions into production-ready Nextflow bioinformatics pipelines. It leverages our existing infrastructure (12 containers, 9,909 tools, 62 modules, 20 workflow patterns) and adds AI-powered automation for workflow generation.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI WORKFLOW COMPOSER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         USER INTERFACE LAYER                           │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  CLI Interface  │  Python API  │  Web API (Future)  │  Chat Interface  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         LLM ADAPTER LAYER                              │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │ │
│  │  │ OpenAI   │ │Anthropic │ │ Ollama   │ │ HuggingFace│ │ Custom LLM  │  │ │
│  │  │ GPT-4    │ │ Claude   │ │ Local    │ │ Inference │ │ Endpoint    │  │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         CORE COMPONENTS                                │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                         │ │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │ │
│  │  │  INTENT PARSER  │───▶│  TOOL SELECTOR  │───▶│ MODULE MAPPER   │     │ │
│  │  │                 │    │                 │    │                 │     │ │
│  │  │ • Analysis type │    │ • Query catalog │    │ • Match modules │     │ │
│  │  │ • Data format   │    │ • Fuzzy match   │    │ • Gap detection │     │ │
│  │  │ • Parameters    │    │ • Alternatives  │    │ • Auto-create   │     │ │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘     │ │
│  │           │                                            │                │ │
│  │           ▼                                            ▼                │ │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │ │
│  │  │  DATA MANAGER   │    │WORKFLOW GENERATOR│    │  VISUALIZER     │     │ │
│  │  │                 │    │                 │    │                 │     │ │
│  │  │ • Download refs │    │ • Chain modules │    │ • DAG diagrams  │     │ │
│  │  │ • Validate data │    │ • Set params    │    │ • QC reports    │     │ │
│  │  │ • Preprocess    │    │ • Generate code │    │ • Result plots  │     │ │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘     │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         KNOWLEDGE BASE                                 │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │  Tool Catalog    │  Module Library  │  Workflow Patterns  │  Ref Data  │ │
│  │  (9,909 tools)   │  (62 modules)    │  (20 patterns)      │  (indexes) │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. LLM Adapter Layer

Pluggable interface supporting multiple LLM backends:

```python
# Abstract base class - all LLM providers implement this
class LLMAdapter(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str: ...
    
    @abstractmethod
    def chat(self, messages: List[Dict], **kwargs) -> str: ...
    
    @abstractmethod
    def embed(self, text: str) -> List[float]: ...
```

**Supported Providers:**

| Provider | Models | Use Case | Config |
|----------|--------|----------|--------|
| OpenAI | GPT-4, GPT-3.5 | High-quality intent parsing | `OPENAI_API_KEY` |
| Anthropic | Claude 3 | Complex reasoning | `ANTHROPIC_API_KEY` |
| Ollama | Llama3, Mistral, CodeLlama | Local/offline, no API costs | `OLLAMA_HOST` |
| HuggingFace | Any HF model | Custom fine-tuned models | `HF_TOKEN` |
| vLLM | Any GGUF/GPTQ | High-throughput local | `VLLM_ENDPOINT` |

### 2. Intent Parser

Extracts structured information from natural language:

**Input:** "I have paired-end RNA-seq data from mouse liver. I want to identify differentially expressed genes between wildtype and knockout samples."

**Output:**
```json
{
  "analysis_type": "rna_seq_differential_expression",
  "data_type": "paired_end_fastq",
  "organism": "mouse",
  "genome": "mm10",
  "tissue": "liver",
  "comparison": {
    "condition1": "wildtype",
    "condition2": "knockout"
  },
  "outputs_requested": ["de_genes", "normalized_counts"],
  "suggested_tools": ["star", "featurecounts", "deseq2"],
  "confidence": 0.95
}
```

### 3. Tool Selector

Queries tool catalog and selects appropriate tools:

```python
class ToolSelector:
    def find_tools(self, analysis_type: str, requirements: Dict) -> List[Tool]:
        """Query 9,909 tools in catalog, return ranked matches"""
        
    def check_availability(self, tool_name: str) -> ToolStatus:
        """Check if tool exists in containers"""
        
    def suggest_alternatives(self, tool_name: str) -> List[Tool]:
        """Find similar tools if requested one unavailable"""
```

### 4. Module Mapper

Maps tools to existing modules or creates new ones:

```python
class ModuleMapper:
    def find_module(self, tool_name: str) -> Optional[Module]:
        """Find existing module for tool"""
        
    def create_module(self, tool: Tool, llm: LLMAdapter) -> Module:
        """Auto-generate new module using LLM"""
        
    def validate_module(self, module: Module) -> ValidationResult:
        """Check module syntax and container compatibility"""
```

### 5. Workflow Generator

Chains modules into complete workflows:

```python
class WorkflowGenerator:
    def generate(self, 
                 intent: ParsedIntent,
                 modules: List[Module],
                 params: Dict) -> Workflow:
        """Generate Nextflow DSL2 workflow"""
        
    def apply_pattern(self, 
                      pattern_name: str,
                      customizations: Dict) -> Workflow:
        """Use existing pattern as template"""
```

### 6. Data Manager

Handles reference data and inputs:

```python
class DataManager:
    def download_reference(self, 
                          organism: str, 
                          genome_build: str) -> Path:
        """Download genome, GTF, indexes"""
        
    def validate_inputs(self, 
                       files: List[Path], 
                       expected_format: str) -> ValidationResult:
        """Check input data format and integrity"""
        
    def prepare_samplesheet(self, 
                           data_dir: Path,
                           pattern: str) -> Path:
        """Auto-generate sample sheet from files"""
```

### 7. Visualizer

Creates diagrams and reports:

```python
class Visualizer:
    def workflow_dag(self, workflow: Workflow) -> Path:
        """Generate workflow DAG diagram"""
        
    def qc_report(self, results_dir: Path) -> Path:
        """Generate QC summary report"""
        
    def plot_results(self, 
                    data: Path, 
                    plot_type: str) -> Path:
        """Generate analysis-specific plots"""
```

---

## Directory Structure

```
src/
└── workflow_composer/
    ├── __init__.py
    ├── config.py                 # Configuration management
    ├── cli.py                    # Command-line interface
    │
    ├── llm/                      # LLM Adapter Layer
    │   ├── __init__.py
    │   ├── base.py               # Abstract LLM adapter
    │   ├── openai_adapter.py     # OpenAI GPT
    │   ├── anthropic_adapter.py  # Anthropic Claude
    │   ├── ollama_adapter.py     # Local Ollama
    │   ├── huggingface_adapter.py# HuggingFace
    │   └── factory.py            # LLM provider factory
    │
    ├── core/                     # Core Components
    │   ├── __init__.py
    │   ├── intent_parser.py      # NL → structured intent
    │   ├── tool_selector.py      # Tool catalog queries
    │   ├── module_mapper.py      # Tool → module mapping
    │   ├── workflow_generator.py # Module → workflow
    │   └── validator.py          # Syntax validation
    │
    ├── data/                     # Data Management
    │   ├── __init__.py
    │   ├── reference_manager.py  # Download references
    │   ├── input_validator.py    # Validate inputs
    │   └── samplesheet.py        # Generate sample sheets
    │
    ├── viz/                      # Visualization
    │   ├── __init__.py
    │   ├── dag.py                # Workflow diagrams
    │   ├── qc_report.py          # QC reports
    │   └── plots.py              # Result plots
    │
    ├── knowledge/                # Knowledge Base
    │   ├── __init__.py
    │   ├── tool_catalog.py       # Tool catalog interface
    │   ├── module_library.py     # Module library interface
    │   ├── patterns.py           # Workflow patterns
    │   └── prompts/              # LLM prompt templates
    │       ├── intent_parsing.txt
    │       ├── module_generation.txt
    │       └── workflow_generation.txt
    │
    └── utils/                    # Utilities
        ├── __init__.py
        ├── nextflow.py           # Nextflow helpers
        └── logging.py            # Logging setup
```

---

## Configuration System

```yaml
# config/composer.yaml

# LLM Provider Configuration
llm:
  # Default provider (can be overridden per-component)
  default_provider: "ollama"  # openai, anthropic, ollama, huggingface
  
  # Provider-specific settings
  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      model: "gpt-4-turbo-preview"
      temperature: 0.1
      max_tokens: 4096
      
    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
      model: "claude-3-opus-20240229"
      temperature: 0.1
      max_tokens: 4096
      
    ollama:
      host: "http://localhost:11434"
      model: "llama3:70b"  # or mistral, codellama
      temperature: 0.1
      
    huggingface:
      token: "${HF_TOKEN}"
      model: "meta-llama/Llama-3-70b-chat-hf"
      device: "cuda"

# Component-specific LLM assignments
components:
  intent_parser:
    provider: "ollama"  # Use local for privacy
    model: "llama3:8b"  # Smaller model sufficient
    
  module_generator:
    provider: "anthropic"  # Use Claude for code generation
    model: "claude-3-opus-20240229"
    
  workflow_generator:
    provider: "openai"  # Use GPT-4 for complex workflows
    model: "gpt-4-turbo-preview"

# Knowledge base paths
knowledge_base:
  tool_catalog: "data/tool_catalog/tool_catalog_20251125_003207.json"
  module_library: "nextflow-modules/"
  workflow_patterns: "docs/COMPOSITION_PATTERNS.md"
  container_images: "containers/images/"

# Data management
data:
  reference_cache: "data/references/"
  download_sources:
    ensembl: "ftp://ftp.ensembl.org/pub/"
    ucsc: "https://hgdownload.soe.ucsc.edu/"
    gencode: "https://ftp.ebi.ac.uk/pub/databases/gencode/"

# Output settings
output:
  workflow_dir: "generated_workflows/"
  visualization_dir: "visualizations/"
  log_level: "INFO"
```

---

## Implementation Phases

### Phase 3.1: Core Framework (Week 1)
- [x] Architecture document
- [ ] Directory structure
- [ ] Configuration system
- [ ] LLM adapter base class
- [ ] Ollama adapter (local first)
- [ ] Basic CLI

### Phase 3.2: Intent Parser (Week 1-2)
- [ ] Prompt engineering for intent extraction
- [ ] Analysis type taxonomy
- [ ] Parameter inference
- [ ] Confidence scoring

### Phase 3.3: Tool & Module Selection (Week 2)
- [ ] Tool catalog interface
- [ ] Fuzzy matching
- [ ] Module library interface
- [ ] Gap detection

### Phase 3.4: Workflow Generator (Week 2-3)
- [ ] Pattern-based generation
- [ ] Module chaining logic
- [ ] Parameter configuration
- [ ] DSL2 code generation

### Phase 3.5: Data Manager (Week 3)
- [ ] Reference downloader
- [ ] Input validator
- [ ] Samplesheet generator

### Phase 3.6: Visualization (Week 3-4)
- [ ] DAG generator
- [ ] QC report templates
- [ ] Result plotting

### Phase 3.7: Integration & Testing (Week 4)
- [ ] End-to-end testing
- [ ] Error handling
- [ ] Documentation
- [ ] Performance optimization

---

## Example Usage

### CLI Interface

```bash
# Generate workflow from natural language
biocomposer generate "RNA-seq differential expression, mouse, paired-end" \
    --llm ollama \
    --output my_rnaseq_workflow/

# Interactive mode
biocomposer chat --llm openai

# Download reference data
biocomposer data download --organism mouse --genome mm10

# Visualize existing workflow
biocomposer viz dag pipelines/rna_seq/main.nf

# Create new module for missing tool
biocomposer module create salmon --container rna-seq
```

### Python API

```python
from workflow_composer import Composer
from workflow_composer.llm import OllamaAdapter

# Initialize with local LLM
llm = OllamaAdapter(model="llama3:8b")
composer = Composer(llm=llm)

# Generate workflow from description
workflow = composer.generate(
    description="I have ChIP-seq data for H3K4me3 in human cells. "
                "I want to call peaks and find enriched motifs.",
    output_dir="my_chipseq/"
)

# Access generated files
print(workflow.main_nf)      # Nextflow code
print(workflow.config)       # Configuration
print(workflow.samplesheet)  # Sample sheet template

# Visualize
workflow.visualize()  # Opens DAG diagram

# Run (if executor configured)
workflow.run(profile="slurm")
```

### Chat Interface

```
User: I need to analyze whole genome sequencing data to find variants

Composer: I understand you want to perform variant calling on WGS data. 
Let me gather some details:

1. What organism/species?
2. Do you have a reference genome, or should I download one?
3. Single sample or multiple samples for joint calling?
4. Germline or somatic variants?

User: Human, hg38, multiple samples, germline

Composer: I'll create a GATK Best Practices workflow:
- Alignment: BWA-MEM2
- Processing: MarkDuplicates, BQSR
- Calling: HaplotypeCaller (GVCF mode)
- Joint genotyping: GenotypeGVCFs
- Filtering: VQSR

Generating workflow... ✓

Files created:
- workflows/wgs_variant_calling/main.nf
- workflows/wgs_variant_calling/nextflow.config
- workflows/wgs_variant_calling/samplesheet.csv (template)

Would you like me to:
1. Download hg38 reference?
2. Show the workflow diagram?
3. Explain any step in detail?
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Intent parsing accuracy | ≥90% | Manual evaluation on 100 queries |
| Workflow generation success | ≥85% | Syntax-valid Nextflow code |
| Time to first workflow | <60 seconds | Average generation time |
| Tool coverage | ≥80% | % of requested tools found |
| User satisfaction | ≥4/5 | Feedback surveys |

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM hallucinations | Medium | Validate against tool catalog, syntax checking |
| API costs | Low | Default to local Ollama, cache responses |
| Complex workflows | Medium | Fall back to pattern templates |
| Missing tools | Low | Auto-suggest alternatives, module creation |
| Performance | Low | Async processing, caching |

---

## Next Steps

1. **Immediate**: Create directory structure and core framework
2. **This week**: Implement Ollama adapter and basic intent parsing
3. **Next week**: Tool selection and workflow generation
4. **Week 3**: Data management and visualization
5. **Week 4**: Testing and documentation

---

*Document maintained by BioPipelines development team*
