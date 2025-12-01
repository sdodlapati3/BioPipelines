# BioPipelines v2.2 Architecture

**Version**: 2.2.0  
**Date**: December 2025  
**Status**: Production  
**Last Validated**: Against codebase December 2025

---

## Overview

BioPipelines is an AI-powered bioinformatics workflow automation platform that enables researchers to compose, execute, and monitor genomics analysis pipelines through natural language interaction.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BioPipelines v2.2                                      │
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
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                 Unified Agent Layer (agents/unified_agent.py)              │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │                     UnifiedAgent                                    │   │  │
│  │  │  • UnifiedIntentParser (RECOMMENDED: pattern + semantic + arbiter) │   │  │
│  │  │  • PermissionManager (AutonomyLevel: READONLY→AUTONOMOUS)          │   │  │
│  │  │  • ConversationContext (multi-turn memory)                         │   │  │
│  │  │  • RAGOrchestrator (tool selection from history)                   │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                │                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │        ★ INTENT PARSING LAYER (Hierarchical with LLM Arbiter) ★           │  │
│  │                                                                            │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                    UnifiedIntentParser                               │  │  │
│  │  │                  (agents/intent/unified_parser.py)                   │  │  │
│  │  │  ┌───────────────────────────────────────────────────────────────┐  │  │  │
│  │  │  │              Stage 1: Fast Methods (~15ms)                    │  │  │  │
│  │  │  │  ┌────────────┐ ┌────────────┐ ┌────────────────────┐        │  │  │  │
│  │  │  │  │  Pattern   │ │  Semantic  │ │      Entity        │        │  │  │  │
│  │  │  │  │  Matching  │ │   FAISS    │ │    Extraction      │        │  │  │  │
│  │  │  │  │ (parser.py)│ │(semantic.py│ │   (parser.py)      │        │  │  │  │
│  │  │  │  └────────────┘ └────────────┘ └────────────────────┘        │  │  │  │
│  │  │  └───────────────────────────────────────────────────────────────┘  │  │  │
│  │  │                              │                                       │  │  │
│  │  │              Agreement Check (80% of queries pass here)              │  │  │
│  │  │                              │                                       │  │  │
│  │  │  ┌───────────────────────────────────────────────────────────────┐  │  │  │
│  │  │  │    Stage 2: LLM Arbiter (20% - only complex/ambiguous)        │  │  │  │
│  │  │  │                    (arbiter.py)                                │  │  │  │
│  │  │  │  • Disagreement detection     • Negation handling             │  │  │  │
│  │  │  │  • Context-aware reasoning    • Provider cascade fallback     │  │  │  │
│  │  │  └───────────────────────────────────────────────────────────────┘  │  │  │
│  │  │                              │                                       │  │  │
│  │  │              Final Intent + Confidence + Reasoning                   │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                │                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                 ★ LLM PROVIDER CASCADE (Rate-Limit Resistant) ★           │  │
│  │                                                                            │  │
│  │       Priority 1      Priority 2       Priority 3       Priority 99       │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌─────────────┐     │  │
│  │  │ Lightning   │─►│GitHub Models │─►│   Gemini    │─►│   OpenAI    │     │  │
│  │  │ (DeepSeek)  │  │   (GPT-4o)   │  │(gemini-pro) │  │ (Fallback)  │     │  │
│  │  └─────────────┘  └──────────────┘  └─────────────┘  └─────────────┘     │  │
│  │           │                                                               │  │
│  │           ▼                                                               │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐   │  │
│  │  │  Local Providers (if available)                                   │   │  │
│  │  │  ┌───────────┐ ┌───────────┐                                     │   │  │
│  │  │  │   Ollama  │ │   vLLM    │  (GPU cluster inference)            │   │  │
│  │  │  │ Priority 5│ │Priority 6 │                                     │   │  │
│  │  │  └───────────┘ └───────────┘                                     │   │  │
│  │  └──────────────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                │                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                     Tool Categories (agents/tools/)                        │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐   │  │
│  │  │   Data   │ │ Workflow │ │Execution │ │Diagnosis │ │   Education   │   │  │
│  │  │Discovery │ │ Generator│ │ (SLURM)  │ │  Agent   │ │               │   │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └───────────────┘   │  │
│  │                                                                            │  │
│  │  + PrefetchManager (background prefetch of search results)                │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                │                                                 │
│ ┌────────────────────────────────────────────────────────────────────────────┐  │
│ │                    RESILIENCE LAYER (infrastructure/)                       │  │
│ │  ┌────────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────────────┐   │  │
│ │  │  Circuit   │  │ Exponential  │  │    Rate    │  │    Timeout       │   │  │
│ │  │  Breaker   │  │   Backoff    │  │  Limiter   │  │    Manager       │   │  │
│ │  └────────────┘  └──────────────┘  └────────────┘  └──────────────────┘   │  │
│ │  infrastructure/resilience.py - Protects GEO, ENCODE, GDC adapters         │  │
│ └────────────────────────────────────────────────────────────────────────────┘  │
│                                │                                                 │
│ ┌────────────────────────────────────────────────────────────────────────────┐  │
│ │                 OBSERVABILITY LAYER (infrastructure/)                       │  │
│ │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐  │  │
│ │  │  Tracer    │  │  Metrics   │  │ Structured │  │   Correlation IDs    │  │  │
│ │  │  (Spans)   │  │ (Counters) │  │    Logs    │  │   (trace_id, span)   │  │  │
│ │  └────────────┘  └────────────┘  └────────────┘  └──────────────────────┘  │  │
│ │  infrastructure/observability.py - @traced decorator, MetricsCollector      │  │
│ └────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Architecture Changes (v2.1 → v2.2)

| Component | v2.1 | v2.2 | Notes |
|-----------|------|------|-------|
| Intent Parser | HybridQueryParser | **UnifiedIntentParser** | Hierarchical with LLM arbiter |
| LLM Selection | Single provider | **Provider Cascade** | Rate-limit resistant, auto-fallback |
| LLM Rate | Always or Never | **~20% LLM** | Smart invocation only when needed |
| Accuracy | ~87% | **~87-95%** | Depends on arbiter strategy |
| Deleted Code | UnifiedEnsembleParser | N/A | 855 lines removed |

---

## Core Principles

### 1. **Facade Pattern** - Single Entry Point
All external interactions go through the `BioPipelines` facade class.

```python
from workflow_composer import BioPipelines

pipeline = BioPipelines()
result = pipeline.chat("Analyze RNA-seq data in /data/samples")
```

### 2. **Hierarchical Intent Parsing** - Fast + Smart
80% of queries are resolved by fast pattern/semantic matching. Only complex queries (negation, ambiguity, disagreement) invoke the LLM arbiter.

```python
from workflow_composer.agents.intent import UnifiedIntentParser

parser = UnifiedIntentParser(use_cascade=True)
result = parser.parse("search for human brain RNA-seq data")
print(result.primary_intent)  # IntentType.DATA_SEARCH
print(result.confidence)      # 0.92
print(result.method)          # "unanimous" or "llm_arbiter"
print(result.llm_invoked)     # False (80% of time)
```

### 3. **Provider Cascade** - Never Fail on Rate Limits
The arbiter uses a cascading provider router that automatically falls back when rate-limited:

```python
# Provider priority (lower = higher priority)
PROVIDER_PRIORITY = {
    "lightning": 1,     # Lightning.ai (DeepSeek)
    "github_models": 2, # GitHub Models (GPT-4o)
    "gemini": 3,        # Google Gemini
    "ollama": 5,        # Local Ollama
    "vllm": 6,          # Local vLLM
    "openai": 99,       # OpenAI (expensive fallback)
}
```

### 4. **Protocol-Based Interfaces** - Duck Typing with Type Safety
Components communicate through Python Protocols.

```python
from workflow_composer.infrastructure import LLMProtocol

class CustomLLM(LLMProtocol):
    def complete(self, prompt: str) -> str:
        return "response"
```

---

## Directory Structure

```
src/workflow_composer/
├── __init__.py              # Package exports
├── facade.py                # BioPipelines entry point
│
├── agents/                  # AI agent system
│   ├── unified_agent.py     # Main agent orchestrator
│   ├── classification.py    # Task type classification
│   ├── self_healing.py      # Auto-recovery agent
│   ├── tool_memory.py       # RAG-enhanced tool selection
│   ├── memory.py            # Agent memory with embeddings
│   ├── orchestrator.py      # Multi-agent orchestration
│   ├── coding_agent.py      # Code diagnosis/fixes
│   ├── react_agent.py       # ReAct reasoning agent
│   ├── intent/              # ★ Intent parsing subsystem
│   │   ├── __init__.py      # Exports UnifiedIntentParser
│   │   ├── parser.py        # IntentParser (patterns + entities)
│   │   ├── semantic.py      # SemanticIntentClassifier + BioinformaticsNER
│   │   ├── arbiter.py       # ★ IntentArbiter (LLM for complex queries)
│   │   ├── unified_parser.py# ★ UnifiedIntentParser (MAIN ENTRY POINT)
│   │   ├── dialogue.py      # DialogueManager
│   │   ├── context.py       # ConversationContext
│   │   ├── integration.py   # ChatIntegration
│   │   ├── negation_handler.py # Negation detection
│   │   └── learning.py      # Feedback system
│   ├── executor/            # Safe execution layer
│   │   ├── sandbox.py       # CommandSandbox
│   │   ├── permissions.py   # PermissionManager
│   │   └── audit.py         # AuditLogger
│   ├── autonomous/          # Full autonomy system
│   ├── rag/                 # RAG orchestration
│   └── tools/               # Agent tool implementations
│       ├── base.py          # ToolResult, ToolName
│       ├── registry.py      # Tool registration
│       ├── prefetch.py      # Proactive prefetching
│       └── (30+ tools)
│
├── providers/               # ★ LLM Provider Cascade
│   ├── __init__.py
│   ├── base.py              # Provider protocol
│   ├── factory.py           # Provider factory
│   ├── router.py            # ★ CascadingProviderRouter
│   ├── lightning.py         # Lightning.ai
│   ├── github_models.py     # GitHub Models
│   ├── gemini.py            # Google Gemini
│   ├── ollama.py            # Local Ollama
│   ├── openai.py            # OpenAI
│   ├── anthropic.py         # Anthropic Claude
│   └── vllm.py              # Local vLLM
│
├── llm/                     # LLM orchestration layer
│   ├── orchestrator.py      # ModelOrchestrator
│   ├── strategies.py        # Strategy, EnsembleMode
│   ├── task_router.py       # TaskRouter, TaskType
│   ├── cost_tracker.py      # CostTracker
│   └── providers/           # Provider abstractions
│
├── infrastructure/          # Cross-cutting concerns
│   ├── container.py         # Dependency injection
│   ├── protocols.py         # Interface definitions
│   ├── exceptions.py        # Error hierarchy
│   ├── logging.py           # Structured logging
│   ├── settings.py          # Configuration management
│   ├── resilience.py        # Circuit breaker, retry, rate limiting
│   ├── observability.py     # Distributed tracing, metrics
│   └── semantic_cache.py    # TTL cache with similarity matching
│
├── core/                    # Core workflow logic
│   ├── workflow_generator.py# Nextflow generation
│   ├── module_mapper.py     # Tool → module mapping
│   ├── query_parser.py      # Query parsing
│   └── tool_selector.py     # LLM-based tool selection
│
├── data/                    # Data management
│   ├── discovery/           # Multi-source search
│   │   ├── parallel.py      # Parallel federated search
│   │   ├── orchestrator.py  # Search orchestration
│   │   └── adapters/        # ENCODE, GEO, GDC, etc.
│   └── ...
│
├── evaluation/              # Evaluation framework
│   ├── benchmarks.py        # Benchmark definitions
│   ├── evaluator.py         # Evaluation runner
│   ├── scorer.py            # Rule-based and LLM scoring
│   └── report.py            # Report generation
│
└── web/                     # Web interface
    ├── app.py               # Gradio application
    └── components/          # UI components
```

---

## Intent Parsing Architecture

The intent parsing subsystem uses a **hierarchical approach**:

### When LLM is NOT invoked (~80% of queries)
- High confidence pattern match (≥0.85)
- Pattern and semantic classifiers agree
- No complexity indicators (negation, conditionals)

### When LLM IS invoked (~20% of queries)
- Disagreement between pattern and semantic
- Low confidence from both (<0.6)
- Complexity indicators detected:
  - Negation: "not", "don't", "without"
  - Conditional: "if", "unless", "when"
  - Comparative: "instead of", "rather than"
  - Change: "actually", "wait", "no,"

### Example Flow

```
Query: "search for human brain RNA-seq data"

1. Pattern Parser: DATA_SEARCH (0.92)
2. Semantic Classifier: DATA_SEARCH (0.88)
3. Agreement: YES, confidence > 0.85
4. LLM Invoked: NO
5. Result: DATA_SEARCH (unanimous), 15ms

---

Query: "NOT RNA-seq, I need ChIP-seq for brain tissue"

1. Pattern Parser: WORKFLOW_CREATE (0.45) 
2. Semantic Classifier: DATA_SEARCH (0.52)
3. Agreement: NO + negation detected
4. LLM Invoked: YES
5. Arbiter decides: DATA_SEARCH (ChIP-seq, brain)
6. Result: DATA_SEARCH (llm_arbiter), 600ms
```

---

## Provider Cascade

The `CascadingProviderRouter` ensures LLM calls never fail due to rate limits:

```python
class CascadingProviderRouter:
    """
    Routes LLM requests through providers in priority order.
    Automatically falls back on rate limits or errors.
    """
    
    PROVIDER_PRIORITY = {
        "lightning": 1,     # Primary: Lightning.ai (free tier)
        "github_models": 2, # Secondary: GitHub Models 
        "gemini": 3,        # Tertiary: Google Gemini
        "ollama": 5,        # Local: Ollama (if available)
        "vllm": 6,          # Local: vLLM (GPU cluster)
        "openai": 99,       # Fallback: OpenAI (paid)
    }
```

### Rate Limit Handling
1. Provider returns 429 → Mark temporarily unavailable
2. Cooldown period: 60 seconds
3. Next request routes to next priority provider
4. After cooldown, retry original provider

---

## Key Components

### 1. UnifiedIntentParser (`agents/intent/unified_parser.py`)

The **recommended** entry point for intent parsing.

```python
from workflow_composer.agents.intent import UnifiedIntentParser

parser = UnifiedIntentParser(
    use_cascade=True,      # Use cascading provider router
    arbiter_strategy="smart",  # smart | always | disagreement
    enable_cache=True,     # Cache LLM decisions
)

result = parser.parse("analyze my ChIP-seq data")
# Returns: UnifiedParseResult
```

**Features**:
- LRU cache for arbiter decisions (1000 entries)
- Metrics tracking (LLM rate, latency, intents)
- Graceful fallback to pattern parser

### 2. IntentArbiter (`agents/intent/arbiter.py`)

LLM-powered arbitration for complex queries.

```python
from workflow_composer.agents.intent import IntentArbiter

arbiter = IntentArbiter(use_cascade=True)
result = arbiter.arbitrate(
    query="NOT RNA-seq, I need ChIP-seq",
    votes=[
        ParserVote("pattern", "WORKFLOW_CREATE", 0.45),
        ParserVote("semantic", "DATA_SEARCH", 0.52),
    ]
)
```

### 3. CascadingProviderRouter (`providers/router.py`)

Rate-limit resistant LLM routing.

```python
from workflow_composer.providers import get_cascading_router

router = get_cascading_router()
response = router.complete("Classify this intent: ...")
# Automatically handles rate limits and fallbacks
```

### 4. BioPipelines Facade (`facade.py`)

Unified entry point.

```python
from workflow_composer import BioPipelines

bp = BioPipelines()
response = bp.chat("search for breast cancer RNA-seq")
```

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Web UI** | Gradio 4.x |
| **Workflow Engine** | Nextflow DSL2, Snakemake |
| **Job Scheduler** | SLURM |
| **Containers** | Singularity/Apptainer |
| **LLM Runtime** | vLLM, Ollama, Lightning.ai, OpenAI |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Vector Search** | FAISS |
| **Configuration** | Pydantic-settings, YAML |
| **Testing** | pytest, pytest-asyncio |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.2.0 | Dec 2025 | Hierarchical intent parsing with LLM arbiter, provider cascade, deleted UnifiedEnsembleParser |
| 2.1.0 | Nov 30, 2025 | Resilience, observability, semantic cache, parallel search |
| 2.0.0 | Nov 2025 | Architecture modernization - DI, Protocols, Facade |
| 1.0.0 | Oct 2025 | Initial release with unified agent |

---

## Related Documents

- [HIERARCHICAL_INTENT_PARSING_PLAN.md](HIERARCHICAL_INTENT_PARSING_PLAN.md) - Detailed arbiter design
- [LLM_ORCHESTRATION_PLAN.md](LLM_ORCHESTRATION_PLAN.md) - ModelOrchestrator implementation
- [RESILIENCE_OBSERVABILITY_PLAN.md](RESILIENCE_OBSERVABILITY_PLAN.md) - Infrastructure hardening
- [COMPONENTS.md](COMPONENTS.md) - Component details (needs update)

---

## Metrics & Monitoring

### Intent Parser Metrics

```python
parser = UnifiedIntentParser()
metrics = parser.get_metrics()
# {
#   "total_queries": 1000,
#   "pattern_only": 800,
#   "llm_invoked": 200,
#   "llm_rate": 20.0,
#   "avg_latency_ms": 45.2,
#   "cache_hit_rate": 0.35,
# }
```

### Provider Status

```python
from workflow_composer.providers import get_cascading_router
router = get_cascading_router()
status = router.get_status()
# {
#   "available": ["lightning", "github_models", "gemini"],
#   "unavailable": [],
#   "cooldowns": {},
# }
```
