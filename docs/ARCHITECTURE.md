# BioPipelines v2.1 Architecture

**Version**: 2.1.0  
**Date**: November 30, 2025  
**Status**: Production

---

## Overview

BioPipelines is an AI-powered bioinformatics workflow automation platform that enables researchers to compose, execute, and monitor genomics analysis pipelines through natural language interaction.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BioPipelines v2.1                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│                         User Interface Layer                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐      │
│  │     Web UI      │  │      CLI        │  │       Python API            │      │
│  │    (Gradio)     │  │   (Terminal)    │  │    (import BioPipelines)    │      │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘      │
│           │                    │                          │                      │
│           └────────────────────┼──────────────────────────┘                      │
│                                ▼                                                 │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║                    BioPipelines Facade (facade.py)                         ║  │
│  ║   chat() │ generate_workflow() │ submit() │ status() │ diagnose()         ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                │                                                 │
│ ┌────────────────────────────────────────────────────────────────────────────┐  │
│ │                    ★ OBSERVABILITY LAYER (Phase 3) ★                        │  │
│ │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐  │  │
│ │  │  Tracer    │  │  Metrics   │  │ Structured │  │   Correlation IDs    │  │  │
│ │  │  (Spans)   │  │ (Counters) │  │    Logs    │  │   (trace_id, span)   │  │  │
│ │  └────────────┘  └────────────┘  └────────────┘  └──────────────────────┘  │  │
│ │  infrastructure/observability.py - @traced decorator, MetricsCollector      │  │
│ └────────────────────────────────────────────────────────────────────────────┘  │
│                                │                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                 Unified Agent Layer (agents/unified_agent.py)              │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │                     UnifiedAgent                                    │   │  │
│  │  │  • HybridQueryParser (intent + context)                            │   │  │
│  │  │  • PermissionManager (AutonomyLevel: READONLY→AUTONOMOUS)          │   │  │
│  │  │  • ConversationContext (multi-turn memory)                         │   │  │
│  │  │  • ToolMemory + RAGToolSelector (Phase 6: learning from history)   │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  │                                                                            │  │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────────┐  │  │
│  │  │ Classification  │ │ Tool Selection  │ │    Autonomy Control         │  │  │
│  │  │ (TaskType enum) │ │ (30+ tools)     │ │ (sandbox, audit, approval)  │  │  │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                │                                                 │
│ ┌────────────────────────────────────────────────────────────────────────────┐  │
│ │                    ★ RESILIENCE LAYER (Phase 1) ★                           │  │
│ │  ┌────────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────────┐   │  │
│ │  │  Circuit   │  │ Exponential  │  │    Rate    │  │    Timeout       │   │  │
│ │  │  Breaker   │  │   Backoff    │  │  Limiter   │  │    Manager       │   │  │
│ │  └────────────┘  └──────────────┘  └────────────┘  └──────────────────┘   │  │
│ │  infrastructure/resilience.py - Protects GEO, ENCODE, GDC adapters         │  │
│ └────────────────────────────────────────────────────────────────────────────┘  │
│                                │                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                     Tool Categories (agents/tools/)                        │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐   │  │
│  │  │   Data   │ │ Workflow │ │Execution │ │Diagnosis │ │   Education   │   │  │
│  │  │Discovery │ │ Generator│ │ (SLURM)  │ │  Agent   │ │               │   │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └───────────────┘   │  │
│  │                                                                            │  │
│  │  + PrefetchManager (Phase 5: background prefetch of search results)       │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                │                                                 │
│ ┌────────────────────────────────────────────────────────────────────────────┐  │
│ │                 ★ INTELLIGENT CACHING (Phase 4) ★                           │  │
│ │  ┌──────────────────────────────┐  ┌──────────────────────────────────┐   │  │
│ │  │      Semantic Cache          │  │     Proactive Prefetcher         │   │  │
│ │  │  (TTL + cosine similarity)   │  │     (background enrichment)      │   │  │
│ │  │  infrastructure/semantic_    │  │     agents/tools/prefetch.py     │   │  │
│ │  │  cache.py                    │  │                                   │   │  │
│ │  └──────────────────────────────┘  └──────────────────────────────────┘   │  │
│ └────────────────────────────────────────────────────────────────────────────┘  │
│                                │                                                 │
│ ┌────────────────────────────────────────────────────────────────────────────┐  │
│ │                 ★ PARALLEL EXECUTION (Phase 2) ★                            │  │
│ │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│ │  │            Federated Search (data/discovery/parallel.py)           │   │  │
│ │  │   ENCODE ──┐                                                       │   │  │
│ │  │   GEO ─────┼──► asyncio.gather ──► Merge + Deduplicate ──► Results │   │  │
│ │  │   TCGA ────┘                                                       │   │  │
│ │  │   (Each adapter has circuit breaker protection)                    │   │  │
│ │  └────────────────────────────────────────────────────────────────────┘   │  │
│ └────────────────────────────────────────────────────────────────────────────┘  │
│                                │                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                 Infrastructure Layer (infrastructure/)                     │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐    │  │
│  │  │Container │ │Protocols │ │ Logging  │ │ Settings │ │ Exceptions   │    │  │
│  │  │  (DI)    │ │          │ │          │ │          │ │              │    │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────┘    │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                │                                                 │
│ ┌────────────────────────────────────────────────────────────────────────────┐  │
│ │                 ★ EVALUATION FRAMEWORK (Phase 7) ★                          │  │
│ │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────────────┐    │  │
│ │  │ Benchmarks  │ │  Evaluator  │ │   Scorers   │ │ Report Generator  │    │  │
│ │  │ (queries)   │ │ (runner)    │ │ (rule/LLM)  │ │ (HTML/JSON/MD)    │    │  │
│ │  └─────────────┘ └─────────────┘ └─────────────┘ └───────────────────┘    │  │
│ │  evaluation/benchmarks.py, evaluator.py, scorer.py, report.py, metrics.py  │  │
│ └────────────────────────────────────────────────────────────────────────────┘  │
│                                │                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                         LLM Providers (llm/)                               │  │
│  │  ┌─────────────────────────────┐   ┌─────────────────────────────────┐   │  │
│  │  │      LOCAL (GPU)            │   │           CLOUD                  │   │  │
│  │  │  ┌───────────────────────┐  │   │  ┌───────────────────────────┐  │   │  │
│  │  │  │  vLLM (Primary)       │  │   │  │  Lightning.ai             │  │   │  │
│  │  │  │  Qwen, DeepSeek       │  │   │  │  (DeepSeek, GPT, Claude)  │  │   │  │
│  │  │  └───────────────────────┘  │   │  └───────────────────────────┘  │   │  │
│  │  └─────────────────────────────┘   └─────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Key Architecture Enhancements (v2.1)**:
- ★ **Observability Layer** - Distributed tracing with @traced decorator
- ★ **Resilience Layer** - Circuit breakers on all external API calls
- ★ **Semantic Caching** - 0.85 cosine similarity matching with TTL
- ★ **Parallel Search** - 3x faster federated queries with asyncio.gather
- ★ **Proactive Prefetching** - Background enrichment of top search results
- ★ **RAG Tool Memory** - Learning from successful past tool executions
- ★ **Evaluation Framework** - Automated benchmarks with rule-based and LLM scoring

---

## Core Principles

### 1. **Facade Pattern** - Single Entry Point
All external interactions go through the `BioPipelines` facade class, providing a clean, versioned API.

```python
from workflow_composer import BioPipelines

pipeline = BioPipelines()
result = pipeline.chat("Analyze RNA-seq data in /data/samples")
```

### 2. **Protocol-Based Interfaces** - Duck Typing with Type Safety
Components communicate through Python Protocols, enabling loose coupling and easy testing.

```python
from workflow_composer.infrastructure import LLMProtocol

class CustomLLM(LLMProtocol):
    def complete(self, prompt: str) -> str:
        return "response"
```

### 3. **Dependency Injection** - Testable Components
The DI container manages component lifecycles and dependencies.

```python
from workflow_composer.infrastructure import Container, Scope

container = Container()
container.register("llm", lambda: VLLMAdapter(), scope=Scope.SINGLETON)
llm = container.resolve("llm")
```

### 4. **Unified Error Hierarchy** - Consistent Error Handling
All errors inherit from `BioPipelinesError` with error codes and context.

```python
from workflow_composer.infrastructure import LLMError, ErrorCode

try:
    result = llm.complete(prompt)
except LLMError as e:
    print(f"[{e.error_code}] {e.message}")
    # Auto-recovery may trigger
```

---

## Directory Structure

```
src/workflow_composer/
├── __init__.py              # Package exports
├── facade.py                # BioPipelines entry point
│
├── infrastructure/          # Cross-cutting concerns (★ Enhanced in v2.1)
│   ├── __init__.py
│   ├── container.py         # Dependency injection
│   ├── protocols.py         # Interface definitions
│   ├── exceptions.py        # Error hierarchy
│   ├── logging.py           # Structured logging
│   ├── settings.py          # Configuration management
│   ├── resilience.py        # ★ Circuit breaker, retry, rate limiting (Phase 1)
│   ├── observability.py     # ★ Distributed tracing, metrics (Phase 3)
│   └── semantic_cache.py    # ★ TTL cache with similarity matching (Phase 4)
│
├── agents/                  # AI agent system
│   ├── unified_agent.py     # Main agent orchestrator
│   ├── classification.py    # Task type classification
│   ├── self_healing.py      # Auto-recovery agent
│   ├── tool_memory.py       # ★ RAG-enhanced tool selection (Phase 6)
│   ├── memory.py            # Agent memory with embeddings
│   ├── orchestrator.py      # Multi-agent orchestration
│   ├── coding_agent.py      # Code diagnosis/fixes
│   ├── react_agent.py       # ReAct reasoning agent
│   ├── intent/              # Intent parsing subsystem
│   │   ├── parser.py        # HybridQueryParser
│   │   ├── context.py       # ConversationContext
│   │   └── dialogue.py      # DialogueManager
│   ├── executor/            # Safe execution layer
│   │   ├── sandbox.py       # CommandSandbox
│   │   ├── permissions.py   # PermissionManager
│   │   └── audit.py         # AuditLogger
│   ├── autonomous/          # Full autonomy system
│   │   ├── agent.py         # AutonomousAgent
│   │   ├── health_checker.py# System health monitoring
│   │   └── recovery.py      # RecoveryManager
│   └── tools/               # Agent tool implementations
│       ├── base.py          # ToolResult, ToolName
│       ├── registry.py      # Tool registration
│       ├── prefetch.py      # ★ Proactive prefetching (Phase 5)
│       ├── data_discovery.py# Data scanning tools
│       ├── data_management.py# Download, index tools
│       ├── workflow.py      # Workflow generation
│       ├── execution/       # SLURM job management
│       ├── diagnostics.py   # Error diagnosis
│       └── education.py     # Concept explanation
│
├── core/                    # Core workflow logic
│   ├── workflow_generator.py# Nextflow generation
│   ├── module_mapper.py     # Tool → module mapping
│   ├── query_parser.py      # Query parsing
│   └── tool_selector.py     # LLM-based tool selection
│
├── evaluation/              # ★ Evaluation framework (Phase 7)
│   ├── __init__.py
│   ├── benchmarks.py        # Benchmark query definitions
│   ├── evaluator.py         # Evaluation runner
│   ├── scorer.py            # Rule-based and LLM scoring
│   ├── metrics.py           # Metrics aggregation
│   └── report.py            # HTML/JSON/Markdown reports
│
├── llm/                     # LLM adapters
│   ├── base.py              # BaseLLMAdapter
│   ├── factory.py           # Adapter factory
│   ├── vllm_adapter.py      # Local vLLM
│   ├── lightning_adapter.py # Lightning.ai cloud
│   ├── ollama_adapter.py    # Local Ollama
│   ├── openai_adapter.py    # OpenAI API
│   └── anthropic_adapter.py # Claude API
│
├── data/                    # Data management
│   ├── discovery/           # Multi-source search
│   │   ├── parallel.py      # ★ Parallel federated search (Phase 2)
│   │   ├── orchestrator.py  # Search orchestration
│   │   ├── query_parser.py  # Natural language → API query
│   │   └── adapters/        # Data source adapters
│   │       ├── base.py      # BaseDataAdapter
│   │       ├── encode.py    # ENCODE adapter (with circuit breaker)
│   │       ├── geo.py       # GEO adapter (with circuit breaker)
│   │       ├── gdc.py       # GDC/TCGA adapter (with circuit breaker)
│   │       └── ensembl.py   # Ensembl adapter
│   ├── scanner.py           # Local file scanning
│   ├── downloader.py        # Dataset download
│   └── reference_manager.py # Reference genomes
│
├── diagnosis/               # Error diagnosis system
│   ├── agent.py             # Diagnosis agent
│   ├── patterns.py          # 50+ error patterns
│   ├── auto_fix.py          # Automated fixes
│   └── history.py           # Learning from past
│
├── results/                 # Results management
│   ├── collector.py         # Result collection
│   ├── viewer.py            # Result visualization
│   └── archiver.py          # Result archiving
│
└── web/                     # Web interface
    ├── app.py               # Gradio application
    └── components/          # UI components
```

---

## Key Components

### 1. BioPipelines Facade (`facade.py`)

The unified entry point for all BioPipelines functionality.

| Method | Description |
|--------|-------------|
| `chat(query)` | Process natural language queries |
| `generate_workflow(type, input_dir)` | Create analysis pipelines |
| `submit(workflow_dir)` | Submit jobs to SLURM |
| `status(job_id)` | Check job status |
| `diagnose(job_id)` | Analyze job failures |
| `scan_data(path)` | Discover data files |
| `search_databases(query)` | Search GEO, ENCODE, SRA |

### 2. Unified Agent (`agents/unified_agent.py`)

Orchestrates all AI-powered operations with:
- **Query Classification**: Determines query intent and required tools
- **HybridQueryParser**: Combines pattern matching + LLM for intent extraction
- **Tool Selection**: Chooses appropriate tools with RAG enhancement
- **Autonomy Levels**: READONLY, SUPERVISED, ASSISTED, AUTONOMOUS
- **Permission System**: Sandboxed execution with audit logging
- **Conversation Context**: Multi-turn memory for follow-up queries

### 3. Infrastructure Layer (`infrastructure/`)

| Module | Purpose |
|--------|---------|
| `container.py` | Thread-safe DI container with SINGLETON/TRANSIENT/SCOPED lifecycles |
| `protocols.py` | LLMProtocol, ToolProtocol, EventPublisherProtocol |
| `exceptions.py` | BioPipelinesError hierarchy with error codes |
| `logging.py` | Structured logging with correlation IDs |
| `settings.py` | Pydantic-settings configuration with validation |
| `resilience.py` | ★ CircuitBreaker, RetryWithBackoff, RateLimiter |
| `observability.py` | ★ Tracer, Span, MetricsCollector, @traced decorator |
| `semantic_cache.py` | ★ SemanticCache with TTL and cosine similarity |

### 4. Tool System (`agents/tools/`)

30+ tools organized by category:

| Category | Tools |
|----------|-------|
| **Data Discovery** | scan_data, search_databases, describe_files, validate_dataset |
| **Data Management** | download_dataset, download_reference, build_index |
| **Workflow** | generate_workflow, list_workflows, check_references, visualize_workflow |
| **Execution** | submit_job, get_job_status, cancel_job, resubmit_job, watch_job, list_jobs |
| **Diagnostics** | diagnose_error, recover_error, analyze_results, check_system_health |
| **Education** | explain_concept, compare_samples, get_help |

### 5. ★ Resilience & Caching (NEW in v2.1)

| Component | Module | Purpose |
|-----------|--------|---------|
| **CircuitBreaker** | `infrastructure/resilience.py` | Prevents cascade failures (CLOSED→OPEN→HALF_OPEN) |
| **RetryWithBackoff** | `infrastructure/resilience.py` | Exponential backoff with jitter |
| **RateLimiter** | `infrastructure/resilience.py` | Token bucket rate limiting |
| **SemanticCache** | `infrastructure/semantic_cache.py` | TTL + 0.85 cosine similarity matching |
| **PrefetchManager** | `agents/tools/prefetch.py` | Background prefetch of top search results |

### 6. ★ Observability & Evaluation (NEW in v2.1)

| Component | Module | Purpose |
|-----------|--------|---------|
| **Tracer** | `infrastructure/observability.py` | Distributed tracing with spans |
| **MetricsCollector** | `infrastructure/observability.py` | Counters, gauges, histograms |
| **@traced decorator** | `infrastructure/observability.py` | Auto-instrument functions |
| **ToolMemory** | `agents/tool_memory.py` | RAG-enhanced tool selection from history |
| **Evaluator** | `evaluation/evaluator.py` | Benchmark runner with batch support |
| **RuleBasedScorer** | `evaluation/scorer.py` | Automated quality scoring |
| **LLMScorer** | `evaluation/scorer.py` | LLM-as-judge evaluation |
| **ReportGenerator** | `evaluation/report.py` | HTML, JSON, Markdown reports |

### 7. LLM Adapters (`llm/`)

Unified interface for LLM providers:

| Adapter | Provider | Status | Use Case |
|---------|----------|--------|----------|
| `VLLMAdapter` | Local vLLM | **Primary** | GPU cluster inference (Qwen, DeepSeek) |
| `LightningAdapter` | Lightning.ai | **Active** | Cloud access to DeepSeek, GPT, Claude (30M free tokens/month) |
| `OllamaAdapter` | Local Ollama | Available | Lightweight local fallback |
| `OpenAIAdapter` | OpenAI API | Available | Direct OpenAI access |
| `AnthropicAdapter` | Claude API | Available | Direct Anthropic access |

---

## Data Flow with Resilience

```
User Query: "search for human brain RNA-seq data"
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  UnifiedAgent.process()                                          │
│    1. @traced(name="agent.process")  ──────► Span created       │
│    2. HybridQueryParser.parse()      ──────► Intent extracted   │
│    3. ToolMemory.get_boosts()        ──────► RAG enhancement    │
│    4. SemanticCache.get()            ──────► Cache check        │
└─────────────────────────────────────────────────────────────────┘
    │ (cache miss)
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  ParallelSearchOrchestrator.search()                             │
│    asyncio.gather(                                               │
│      encode_adapter.search()  ◄── CircuitBreaker("encode")      │
│      geo_adapter.search()     ◄── CircuitBreaker("geo")         │
│      gdc_adapter.search()     ◄── CircuitBreaker("gdc")         │
│    )                                                             │
│    └─► merge + deduplicate ──► Results                          │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Post-Processing                                                 │
│    1. SemanticCache.set()         ──────► Cache result          │
│    2. PrefetchManager.prefetch()  ──────► Background enrichment │
│    3. ToolMemory.record()         ──────► Log successful exec   │
│    4. Tracer.end_span()           ──────► Complete trace        │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Response to User (with trace_id for debugging)
```

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Web UI** | Gradio 4.x |
| **Workflow Engine** | Nextflow DSL2, Snakemake |
| **Job Scheduler** | SLURM |
| **Containers** | Singularity/Apptainer |
| **LLM Runtime** | vLLM, Ollama |
| **Configuration** | Pydantic-settings, YAML |
| **Testing** | pytest, pytest-asyncio (467 tests) |
| **Logging** | structlog-compatible |
| **Observability** | Custom tracing (OpenTelemetry-compatible) |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.1.0 | Nov 30, 2025 | Resilience, observability, semantic cache, parallel search, prefetching, RAG tool memory, evaluation framework |
| 2.0.0 | Nov 2025 | Architecture modernization - DI, Protocols, Facade |
| 1.0.0 | Oct 2025 | Initial release with unified agent |
