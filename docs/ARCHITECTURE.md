# BioPipelines Architecture

> Current codebase organization as of November 2024

## Related Documentation

- **[AGENTIC_SYSTEM_ARCHITECTURE.md](AGENTIC_SYSTEM_ARCHITECTURE.md)** — Detailed visual architecture of the AI agent system
- **[diagrams/](diagrams/)** — Mermaid diagrams for system flows and component relationships

## Project Overview

BioPipelines is an AI-powered bioinformatics workflow composition framework. It combines LLM-based query understanding with automated workflow generation, execution monitoring, and intelligent error diagnosis.

---

## Directory Structure

```
BioPipelines/
├── src/workflow_composer/     # Main Python package
├── config/                    # Configuration files
├── containers/                # Container definitions (Docker/Singularity)
├── data/                      # Data directories
├── tests/                     # Test suite
├── scripts/                   # Utility scripts
├── tools/                     # Standalone tools
├── generated_workflows/       # Output workflow directories
├── nextflow-modules/          # Nextflow module definitions
├── nextflow-pipelines/        # Nextflow pipeline implementations
├── docs/                      # Documentation
│   ├── archive/               # Archived documentation
│   ├── infrastructure/        # Infrastructure guides
│   └── tutorials/             # User tutorials
└── notebooks/                 # Jupyter notebooks
```

---

## Source Code Organization

### `src/workflow_composer/`

The main Python package containing all workflow composition logic.

```
workflow_composer/
├── __init__.py
├── cli.py                     # Command-line interface
├── composer.py                # Main workflow composition orchestrator
├── config.py                  # Configuration management
├── secrets.py                 # Secrets/credentials handling
│
├── agents/                    # Agent system (AI-powered components)
├── core/                      # Core workflow generation
├── data/                      # Data discovery and management
├── diagnosis/                 # Error diagnosis system
├── llm/                       # LLM adapters (legacy)
├── models/                    # Data models and registry
├── monitor/                   # Workflow monitoring
├── providers/                 # LLM provider implementations
├── results/                   # Results collection and processing
├── templates/                 # Workflow templates
├── viz/                       # Visualization
└── web/                       # Web interface
```

---

## Module Details

### `agents/` — Agent System

The AI-powered agent infrastructure for autonomous operations.

```
agents/
├── __init__.py
├── bridge.py                  # Legacy bridge for compatibility
├── coding_agent.py            # Code generation agent
├── unified_agent.py           # Unified orchestration agent
│
├── tools/                     # Tool implementations (32 total)
│   ├── base.py                # BaseTool abstract class
│   ├── registry.py            # ToolRegistry for tool management
│   ├── data_discovery.py      # 6 tools: search, browse, validate, etc.
│   ├── data_management.py     # 4 tools: download, transfer, cleanup, etc.
│   ├── diagnostics.py         # 5 tools: analyze, diagnose, fix, etc.
│   ├── education.py           # 4 tools: explain, recommend, compare, etc.
│   ├── execution.py           # 8 tools: run, monitor, stop, etc.
│   └── workflow.py            # 5 tools: generate, validate, optimize, etc.
│
├── autonomous/                # Autonomous agent capabilities
│   ├── agent.py               # Main autonomous agent
│   ├── health_checker.py      # System health monitoring
│   ├── job_monitor.py         # Job status tracking
│   └── recovery.py            # Error recovery strategies
│
└── executor/                  # Secure execution framework
    ├── audit.py               # Audit logging
    ├── file_ops.py            # Safe file operations
    ├── permissions.py         # Permission management
    ├── process_manager.py     # Process lifecycle management
    └── sandbox.py             # Sandboxed execution
```

**Key Classes:**
- `UnifiedAgent` — Central orchestration combining tools + autonomous capabilities
- `BaseTool` — Abstract base for all 32 tools
- `ToolRegistry` — Tool discovery and management
- `AutonomousAgent` — Self-directed task execution

---

### `core/` — Workflow Generation Core

Core logic for parsing queries and generating workflows.

```
core/
├── __init__.py
├── model_service_manager.py   # LLM service coordination
├── module_mapper.py           # Map tools to workflow modules
├── preflight_validator.py     # Pre-execution validation
├── query_parser.py            # User query parsing
├── query_parser_ensemble.py   # Multi-model query parsing
├── tool_selector.py           # Tool selection logic
└── workflow_generator.py      # Workflow code generation
```

**Key Classes:**
- `QueryParser` — Parse natural language to workflow intent
- `WorkflowGenerator` — Generate Nextflow/Snakemake code
- `ToolSelector` — Select appropriate tools for tasks

---

### `data/` — Data Discovery & Management

Data-related operations including discovery, downloading, and reference management.

```
data/
├── __init__.py
├── downloader.py              # File/dataset downloading
├── manifest.py                # Manifest file handling
├── reference_manager.py       # Reference genome management
├── scanner.py                 # Data directory scanning
│
├── browser/                   # Reference data browser
│   └── reference_browser.py   # Interactive reference browsing
│
└── discovery/                 # Data discovery system
    ├── adapters/              # Source-specific adapters
    ├── models.py              # Discovery data models
    ├── orchestrator.py        # Discovery orchestration
    └── query_parser.py        # Discovery query parsing
```

**Key Classes:**
- `ReferenceManager` — Download/manage reference genomes
- `DataDiscoveryOrchestrator` — Coordinate data searches
- `DataDownloader` — Handle file transfers

---

### `diagnosis/` — Error Diagnosis System

AI-powered error analysis and automated fixing.

```
diagnosis/
├── __init__.py
├── agent.py                   # Main diagnosis agent
├── auto_fix.py                # Automated error fixing
├── categories.py              # Error categorization
├── gemini_adapter.py          # Gemini API adapter
├── github_agent.py            # GitHub issue search
├── history.py                 # Diagnosis history
├── lightning_adapter.py       # Lightning AI adapter
├── log_collector.py           # Log aggregation
├── monitor.py                 # Error monitoring
├── patterns.py                # Error pattern matching
└── prompts.py                 # LLM prompt templates
```

**Key Classes:**
- `DiagnosisAgent` — Analyze and diagnose errors
- `AutoFixer` — Apply automated fixes
- `LogCollector` — Gather relevant logs

---

### `providers/` — LLM Provider Implementations

Unified interface for multiple LLM backends.

```
providers/
├── __init__.py
├── base.py                    # BaseProvider abstract class
├── factory.py                 # Provider factory
├── registry.py                # Provider registry
├── router.py                  # Request routing
│
├── anthropic.py               # Claude models
├── gemini.py                  # Google Gemini
├── lightning.py               # Lightning AI
├── ollama.py                  # Local Ollama
├── openai.py                  # OpenAI GPT models
├── vllm.py                    # vLLM serving
│
└── utils/                     # Provider utilities
    ├── health.py              # Health checks
    └── metrics.py             # Usage metrics
```

**Key Classes:**
- `BaseProvider` — Abstract LLM interface
- `ProviderFactory` — Create provider instances
- `ProviderRouter` — Route requests to providers

---

### `llm/` — Legacy LLM Adapters

Original LLM adapter implementations (deprecated, use `providers/`).

```
llm/
├── __init__.py
├── base.py                    # Base adapter
├── factory.py                 # Adapter factory
├── anthropic_adapter.py
├── huggingface_adapter.py
├── lightning_adapter.py
├── ollama_adapter.py
├── openai_adapter.py
└── vllm_adapter.py
```

---

### `results/` — Results Collection

Collect, organize, and transfer workflow results.

```
results/
├── __init__.py
├── archiver.py                # Result archiving
├── cloud_transfer.py          # Cloud storage transfer
├── collector.py               # Result collection
├── detector.py                # Output file detection
├── patterns.py                # File pattern matching
├── result_types.py            # Result type definitions
└── viewer.py                  # Result viewing
```

**Key Classes:**
- `ResultCollector` — Aggregate workflow outputs
- `CloudTransfer` — Upload to GCS/S3
- `ResultArchiver` — Create archives

---

### `monitor/` — Workflow Monitoring

Real-time workflow execution monitoring.

```
monitor/
├── __init__.py
└── workflow_monitor.py        # Main monitoring logic
```

---

### `viz/` — Visualization

Workflow and result visualization.

```
viz/
├── __init__.py
└── visualizer.py              # Visualization generation
```

---

### `web/` — Web Interface

Gradio-based web interface.

```
web/
├── __init__.py
├── app.py                     # Main Gradio application
├── chat_handler.py            # Chat interface handler
├── utils.py                   # Web utilities
├── archive/                   # Archived components
│
└── components/                # UI components
    ├── autonomous_panel.py    # Autonomous agent panel
    └── data_tab.py            # Data discovery tab
```

---

## Configuration

### `config/`

```
config/
├── README.md
├── analysis_definitions.yaml  # Workflow analysis types
├── composer.yaml              # Main composer config
├── defaults.yaml              # Default settings
├── ensemble.yaml              # Multi-model ensemble config
├── slurm.yaml                 # SLURM cluster config
├── tool_mappings.yaml         # Tool-to-module mappings
│
├── nextflow/                  # Nextflow-specific configs
└── snakemake_profiles/        # Snakemake profile configs
```

---

## Testing

### `tests/`

```
tests/
├── test_agentic_router.py     # Router integration tests
├── test_unified_agent.py      # UnifiedAgent tests (29 tests)
│
├── unit/                      # Unit tests
│   ├── test_data_discovery.py
│   ├── test_diagnosis.py
│   ├── test_llm_adapters.py
│   ├── test_results.py
│   └── test_workflow_composer.py
│
├── integration/               # Integration tests
│   └── test_integration.py
│
├── fixtures/                  # Test fixtures
└── config/                    # Test configurations
```

---

## Container Images

### `containers/`

Container definitions organized by analysis type.

```
containers/
├── base/                      # Base images
├── workflow-engine/           # Nextflow/Snakemake engines
│
├── rna-seq/                   # RNA-seq tools
├── dna-seq/                   # DNA-seq/variant calling
├── chip-seq/                  # ChIP-seq analysis
├── atac-seq/                  # ATAC-seq analysis
├── scrna-seq/                 # Single-cell RNA-seq
├── methylation/               # Methylation analysis
├── metagenomics/              # Metagenomic analysis
├── long-read/                 # Long-read sequencing
├── structural-variants/       # SV detection
├── hic/                       # Hi-C analysis
│
├── tier2/                     # Secondary tools
└── images/                    # Built image registry
```

---

## Key Architectural Patterns

### 1. Tool-Based Agent System
All capabilities exposed as 32 tools with OpenAI function-calling definitions:
- Tools inherit from `BaseTool`
- Registered via `ToolRegistry`
- Executed through `UnifiedAgent`

### 2. Multi-Provider LLM Support
Unified interface across 6 LLM providers:
- OpenAI, Anthropic, Google Gemini
- Local: Ollama, vLLM
- Cloud: Lightning AI

### 3. Autonomous Operations
Self-directed agent capabilities:
- Health monitoring
- Job tracking
- Error recovery
- Permission-gated execution

### 4. Workflow Composition
Query-to-workflow pipeline:
1. Parse natural language query
2. Select appropriate tools/modules
3. Generate Nextflow/Snakemake code
4. Validate and execute

---

## Entry Points

| Entry Point | Description |
|-------------|-------------|
| `python -m workflow_composer.cli` | Command-line interface |
| `python -m workflow_composer.web.app` | Web interface (Gradio) |
| `from workflow_composer.agents import UnifiedAgent` | Programmatic agent access |
| `from workflow_composer import Composer` | Direct composer access |

---

## Statistics

| Component | Count |
|-----------|-------|
| Total Tools | 32 |
| LLM Providers | 6 |
| Container Categories | 12 |
| Unit Tests | ~35 |
| Python Modules | ~80 |

---

*Document generated based on codebase analysis. See `docs/archive/` for historical documentation.*
