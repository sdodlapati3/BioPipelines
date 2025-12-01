# BioPipelines v2.3 Architecture

**Version**: 2.3.0  
**Date**: December 2025  
**Status**: Production  
**Last Validated**: Against codebase December 2025

---

## Overview

BioPipelines is an AI-powered bioinformatics workflow automation platform that enables researchers to compose, execute, and monitor genomics analysis pipelines through natural language interaction.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BioPipelines v2.3                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│                         User Interface Layer                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐      │
│  │     Web UI      │  │      CLI        │  │       Python API            │      │
│  │    (Gradio)     │  │   (Terminal)    │  │    (import BioPipelines)    │      │
│  │  + Advanced Gen │  │  + --agents     │  │  + generate_with_agents()   │      │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┬──────────────┘      │
│           │                    │                          │                      │
│           └────────────────────┼──────────────────────────┘                      │
│                                ▼                                                 │
│  ╔═══════════════════════════════════════════════════════════════════════════╗  │
│  ║                    BioPipelines Facade (facade.py)                         ║  │
│  ║   chat() │ generate_workflow() │ submit() │ supervisor │ sessions         ║  │
│  ╚═══════════════════════════════════════════════════════════════════════════╝  │
│                                │                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │              ★ MULTI-AGENT SPECIALIST SYSTEM (Phase 2.4) ★                │  │
│  │                                                                            │  │
│  │  ┌──────────────────────────────────────────────────────────────────────┐ │  │
│  │  │                      SupervisorAgent                                  │ │  │
│  │  │         Orchestrates specialist agents for workflow generation        │ │  │
│  │  └───────────────────────────────┬──────────────────────────────────────┘ │  │
│  │                                  │                                         │  │
│  │    ┌─────────────┐  ┌────────────┴───────────┐  ┌─────────────────┐       │  │
│  │    │  Planner    │  │       CodeGen          │  │   Validator     │       │  │
│  │    │   Agent     │─►│        Agent           │─►│     Agent       │       │  │
│  │    │ (NL→Plan)   │  │  (Plan→Nextflow DSL2)  │  │ (Static+LLM)    │       │  │
│  │    └─────────────┘  └────────────────────────┘  └────────┬────────┘       │  │
│  │                                                          │                 │  │
│  │                        ┌─────────────────────────────────┘                 │  │
│  │                        ▼                                                   │  │
│  │    ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────────────┐ │  │
│  │    │    Doc      │  │    QC       │  │        Validation Loop           │ │  │
│  │    │   Agent     │  │   Agent     │  │  (up to 3 fix attempts)          │ │  │
│  │    │ (README,DAG)│  │ (Thresholds)│  │  ValidatorAgent ↔ CodeGenAgent   │ │  │
│  │    └─────────────┘  └─────────────┘  └──────────────────────────────────┘ │  │
│  │                                                                            │  │
│  │  Workflow States: IDLE→PLANNING→GENERATING→VALIDATING→FIXING→DOCUMENTING │  │
│  │                   →COMPLETE/FAILED                                         │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                │                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                 Unified Agent Layer (agents/unified_agent.py)              │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐   │  │
│  │  │                     UnifiedAgent                                    │   │  │
│  │  │  • UnifiedIntentParser (pattern + semantic + arbiter)              │   │  │
│  │  │  • PermissionManager (AutonomyLevel: READONLY→AUTONOMOUS)          │   │  │
│  │  │  • SessionManager (multi-turn memory, user profiles)               │   │  │
│  │  │  • RAGOrchestrator (tool selection from history)                   │   │  │
│  │  └────────────────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                │                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │            ★ RAG ENHANCEMENT LAYER (Phase 2.6) ★                          │  │
│  │                                                                            │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐   │  │
│  │  │  KnowledgeBase  │  │  ErrorPatternDB │  │  SemanticRetriever      │   │  │
│  │  │  (nf-core, tools│  │  (solutions)    │  │  (hybrid search)        │   │  │
│  │  │   best practices│  │                 │  │                         │   │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘   │  │
│  │                                                                            │  │
│  │  Sources: TOOL_CATALOG | NF_CORE_MODULES | ERROR_PATTERNS | BEST_PRACTICES│  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                │                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │        ★ INTENT PARSING LAYER (Hierarchical with LLM Arbiter) ★           │  │
│  │                                                                            │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                    UnifiedIntentParser                               │  │  │
│  │  │  ┌───────────────────────────────────────────────────────────────┐  │  │  │
│  │  │  │              Stage 1: Fast Methods (~15ms)                    │  │  │  │
│  │  │  │  ┌────────────┐ ┌────────────┐ ┌────────────────────┐        │  │  │  │
│  │  │  │  │  Pattern   │ │  Semantic  │ │      Entity        │        │  │  │  │
│  │  │  │  │  Matching  │ │   FAISS    │ │    Extraction      │        │  │  │  │
│  │  │  │  └────────────┘ └────────────┘ └────────────────────┘        │  │  │  │
│  │  │  └───────────────────────────────────────────────────────────────┘  │  │  │
│  │  │                              │                                       │  │  │
│  │  │              Agreement Check (80% of queries pass here)              │  │  │
│  │  │                              │                                       │  │  │
│  │  │  ┌───────────────────────────────────────────────────────────────┐  │  │  │
│  │  │  │    Stage 2: LLM Arbiter (20% - only complex/ambiguous)        │  │  │  │
│  │  │  └───────────────────────────────────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                │                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                 ★ LLM PROVIDER CASCADE (Rate-Limit Resistant) ★           │  │
│  │                                                                            │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐          │  │
│  │  │  Gemini    │─►│  Cerebras  │─►│   Groq     │─►│ OpenRouter │          │  │
│  │  │ (Primary)  │  │  (Fast)    │  │  (Fast)    │  │ (Fallback) │          │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘          │  │
│  │                                                                            │  │
│  │  + Streaming Support (Phase 2.1) │ + Fallback Chain                       │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                │                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │              ★ OBSERVABILITY LAYER (Phase 2.5) ★                          │  │
│  │                                                                            │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌──────────────────────────────┐ │  │
│  │  │ ProviderMetrics│  │ QueryAnalytics │  │      HealthEndpoints        │ │  │
│  │  │ (latency,cost) │  │ (patterns,perf)│  │  /health, /ready, /live     │ │  │
│  │  └────────────────┘  └────────────────┘  └──────────────────────────────┘ │  │
│  │                                                                            │  │
│  │  + Distributed Tracing │ + Metrics Collection │ + Alerting                │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                │                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │              ★ AUTO-PROVISIONING LAYER (Phase 2.3) ★                      │  │
│  │                                                                            │  │
│  │  ┌─────────────────────┐  ┌─────────────────────────────────────────────┐ │  │
│  │  │  ReferenceManager   │  │           ContainerManager                  │ │  │
│  │  │  • Genome downloads │  │  • Singularity/Docker support              │ │  │
│  │  │  • Index building   │  │  • Auto-pull from registries               │ │  │
│  │  │  • Version tracking │  │  • Build from Dockerfiles                  │ │  │
│  │  └─────────────────────┘  └─────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Architecture Changes (v2.2 → v2.3)

| Component | v2.2 | v2.3 | Notes |
|-----------|------|------|-------|
| Workflow Generation | Single-pass | **Multi-Agent Specialists** | SupervisorAgent + 5 specialists |
| Response Mode | Blocking | **Streaming** | Real-time token streaming |
| Session Handling | Stateless | **Session Memory** | User profiles, preferences |
| Resource Provisioning | Manual | **Auto-provisioning** | Reference + Container managers |
| Observability | Basic logging | **Full stack** | Metrics, analytics, health |
| Knowledge Retrieval | Simple search | **RAG Enhancement** | KnowledgeBase, ErrorPatternDB |
| CLI | Basic commands | **--agents flag** | Multi-agent generation mode |
| Web UI | Chat only | **Advanced Gen panel** | Streaming progress display |

---

## Phase 2 Features Summary

### 2.1 Streaming Responses
Real-time token streaming for improved UX:
```python
async for chunk in bp.chat_stream("Explain RNA-seq workflow"):
    print(chunk, end="", flush=True)
```

### 2.2 Session Memory
Persistent user sessions with preferences:
```python
session = bp.create_session("user123")
bp.chat("I prefer STAR over HISAT2", session_id=session)
# Future queries remember preference
```

### 2.3 Auto-Provisioning
Automatic download and indexing of references:
```python
from workflow_composer.agents.provisioning import ReferenceManager
ref_mgr = ReferenceManager()
await ref_mgr.ensure_available("GRCh38", tools=["STAR", "salmon"])
```

### 2.4 Multi-Agent Specialists
Coordinated workflow generation:
```python
result = await bp.generate_with_agents(
    "RNA-seq differential expression for human",
    output_dir="workflows/rnaseq"
)
# Uses: Planner → CodeGen → Validator → Doc → QC agents
```

### 2.5 Observability
Production-grade monitoring:
```python
from workflow_composer.agents.observability import ProviderMetrics
metrics = ProviderMetrics()
print(metrics.get_provider_stats())  # Latency, cost, success rates
```

### 2.6 RAG Enhancement
Context-enhanced generation:
```python
from workflow_composer.agents.rag import KnowledgeBase
kb = KnowledgeBase()
await kb.index_nf_core()  # Index nf-core modules
docs = kb.search("salmon quantification")
```

---

## Directory Structure

```
src/workflow_composer/
├── __init__.py              # Package exports
├── facade.py                # BioPipelines entry point
├── composer.py              # Workflow composition with multi-agent option
├── cli.py                   # CLI with --agents flag
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
│   │
│   ├── specialists/         # ★ Multi-Agent Specialists (Phase 2.4)
│   │   ├── __init__.py      # Exports all specialists
│   │   ├── supervisor.py    # SupervisorAgent - orchestrator
│   │   ├── planner.py       # PlannerAgent - NL → WorkflowPlan
│   │   ├── codegen.py       # CodeGenAgent - Plan → Nextflow DSL2
│   │   ├── validator.py     # ValidatorAgent - static + LLM review
│   │   ├── docs.py          # DocAgent - README, DAG generation
│   │   └── qc.py            # QCAgent - quality thresholds
│   │
│   ├── streaming/           # ★ Streaming Support (Phase 2.1)
│   │   ├── __init__.py
│   │   ├── stream_handler.py# Token streaming
│   │   └── adapters.py      # Provider-specific streaming
│   │
│   ├── session/             # ★ Session Memory (Phase 2.2)
│   │   ├── __init__.py
│   │   ├── session_manager.py # Session lifecycle
│   │   └── user_profile.py  # User preferences
│   │
│   ├── provisioning/        # ★ Auto-provisioning (Phase 2.3)
│   │   ├── __init__.py
│   │   ├── reference_manager.py # Genome downloads
│   │   └── container_manager.py # Singularity/Docker
│   │
│   ├── observability/       # ★ Observability (Phase 2.5)
│   │   ├── __init__.py
│   │   ├── provider_metrics.py  # LLM metrics
│   │   ├── query_analytics.py   # Query patterns
│   │   └── health.py        # Health endpoints
│   │
│   ├── rag/                 # ★ RAG Enhancement (Phase 2.6)
│   │   ├── __init__.py
│   │   ├── knowledge_base.py    # Multi-source KB
│   │   ├── error_pattern_db.py  # Error solutions
│   │   └── semantic_retriever.py# Hybrid search
│   │
│   ├── intent/              # Intent parsing subsystem
│   │   ├── __init__.py      # Exports UnifiedIntentParser
│   │   ├── parser.py        # IntentParser (patterns + entities)
│   │   ├── semantic.py      # SemanticIntentClassifier
│   │   ├── arbiter.py       # IntentArbiter (LLM for complex)
│   │   ├── unified_parser.py# UnifiedIntentParser (MAIN)
│   │   ├── dialogue.py      # DialogueManager
│   │   ├── context.py       # ConversationContext
│   │   └── learning.py      # Feedback system
│   │
│   ├── executor/            # Safe execution layer
│   │   ├── sandbox.py       # CommandSandbox
│   │   ├── permissions.py   # PermissionManager
│   │   └── audit.py         # AuditLogger
│   │
│   └── tools/               # Agent tool implementations
│       ├── base.py          # ToolResult, ToolName
│       ├── registry.py      # Tool registration
│       ├── prefetch.py      # Proactive prefetching
│       └── (30+ tools)
│
├── providers/               # LLM Provider Cascade
│   ├── __init__.py
│   ├── base.py              # Provider protocol
│   ├── factory.py           # Provider factory
│   ├── router.py            # CascadingProviderRouter
│   ├── gemini.py            # Google Gemini
│   ├── cerebras.py          # Cerebras (fast)
│   ├── groq.py              # Groq (fast)
│   ├── openrouter.py        # OpenRouter
│   ├── lightning.py         # Lightning.ai
│   ├── ollama.py            # Local Ollama
│   └── vllm.py              # Local vLLM
│
├── llm/                     # LLM orchestration layer
│   ├── orchestrator.py      # ModelOrchestrator
│   ├── strategies.py        # Strategy, EnsembleMode
│   ├── task_router.py       # TaskRouter, TaskType
│   └── cost_tracker.py      # CostTracker
│
├── infrastructure/          # Cross-cutting concerns
│   ├── container.py         # Dependency injection
│   ├── protocols.py         # Interface definitions
│   ├── exceptions.py        # Error hierarchy
│   ├── resilience.py        # Circuit breaker, retry
│   └── observability.py     # Distributed tracing
│
├── core/                    # Core workflow logic
│   ├── workflow_generator.py# Nextflow generation
│   ├── module_mapper.py     # Tool → module mapping
│   └── tool_selector.py     # LLM-based tool selection
│
├── data/                    # Data management
│   └── discovery/           # Multi-source search
│       ├── parallel.py      # Parallel federated search
│       └── adapters/        # ENCODE, GEO, GDC, etc.
│
└── web/                     # Web interface
    ├── app.py               # Gradio app + Advanced Gen panel
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
| 2.3.0 | Dec 2025 | Multi-agent specialists, streaming, session memory, auto-provisioning, observability, RAG enhancement |
| 2.2.0 | Dec 2025 | Hierarchical intent parsing with LLM arbiter, provider cascade, deleted UnifiedEnsembleParser |
| 2.1.0 | Nov 30, 2025 | Resilience, observability, semantic cache, parallel search |
| 2.0.0 | Nov 2025 | Architecture modernization - DI, Protocols, Facade |
| 1.0.0 | Oct 2025 | Initial release with unified agent |

---

## Related Documents

- [PHASE2_IMPLEMENTATION_PLAN.md](PHASE2_IMPLEMENTATION_PLAN.md) - Phase 2 feature implementation details
- [HIERARCHICAL_INTENT_PARSING_PLAN.md](HIERARCHICAL_INTENT_PARSING_PLAN.md) - Detailed arbiter design
- [LLM_ORCHESTRATION_PLAN.md](LLM_ORCHESTRATION_PLAN.md) - ModelOrchestrator implementation
- [RESILIENCE_OBSERVABILITY_PLAN.md](RESILIENCE_OBSERVABILITY_PLAN.md) - Infrastructure hardening
- [COMPONENTS.md](COMPONENTS.md) - Component details

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
