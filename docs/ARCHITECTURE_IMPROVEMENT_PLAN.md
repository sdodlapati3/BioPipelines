# BioPipelines Architecture Improvement Plan

**Version**: 1.0  
**Date**: November 30, 2025  
**Target**: Production-ready for 100s of users/week  

---

## Executive Summary

This document provides a critical analysis of each architectural component and a detailed improvement plan. The goal is to transform BioPipelines from a single-user research tool into a production-grade multi-user platform.

### Current State Assessment

| Component | Lines | Integration | Verdict |
|-----------|-------|-------------|---------|
| Core Agent System | ~3000 | Full | ✅ Keep as-is |
| Circuit Breakers (Phase 1) | 733 | 3 adapters | ✅ Strengthen |
| Parallel Search (Phase 2) | ~300 | Integrated | ✅ Keep |
| Observability (Phase 3) | 864 | 1 file | ⚠️ Extend integration |
| Semantic Cache (Phase 4) | 655 | 1 file | ⚠️ Add persistence |
| Prefetching (Phase 5) | 759 | 1 file | 🟡 Keep, make optional |
| RAG Tool Memory (Phase 6) | 1077 | 1 file | 🔴 Refocus |
| Evaluation (Phase 7) | 2566 | 0 files | 🟡 Move to dev tools |
| **Auth/Multi-tenant** | 0 | N/A | 🔴 **Missing - Critical** |
| **Persistent Storage** | 0 | N/A | 🔴 **Missing - Critical** |
| **REST API** | 0 | N/A | 🔴 **Missing - Critical** |
| **Job Queue** | 0 | N/A | 🔴 **Missing - Critical** |

---

## Part 1: RAG Tool Memory - Why It's Not Working & How to Fix

### The Problem: RAG is Solving the Wrong Problem

Your current tool selection is **deterministic by task type**:

```python
# Current flow (simplified from unified_agent.py)
TaskType.DATA → [search_databases, scan_data, download_dataset]
TaskType.WORKFLOW → [generate_workflow, visualize_workflow]
TaskType.JOB → [submit_job, get_job_status, cancel_job]
```

**The issue**: When a user says "find RNA-seq data", the system:
1. Classifies → `TaskType.DATA`
2. Selects tools → `[search_databases, scan_data, ...]`
3. Executes → `search_databases`

**RAG would boost** `search_databases` from 0.85 confidence to 0.90. But it was **already going to be selected**. The boost is meaningless.

### When RAG Tool Selection IS Valuable

RAG tool memory works when:
1. **Same task type has many competing tools** with different strengths
2. **Tool choice depends on subtle query features** the rule system misses
3. **Tools have different success rates** for different query patterns

**Example where RAG would help**:
```
Query: "search for methylation data from TCGA"

Without RAG:
  - search_databases: 0.80
  - search_tcga: 0.75        ← GDC-specific tool
  - search_geo: 0.70
  
With RAG (after 50 similar queries):
  - search_tcga: 0.75 + 0.20 boost = 0.95   ← RAG learned TCGA queries work better with search_tcga
  - search_databases: 0.80
  - search_geo: 0.70
```

**Your current problem**: You don't have enough tool overlap to make RAG selection valuable.

### The Fix: Refocus RAG for Real Value

#### Option A: Slim Down (Recommended for Now)

Remove `RAGToolSelector` and keep only `ToolMemory` for analytics:

```python
# Simplified tool_memory.py (~300 lines instead of 1077)

class ToolMemory:
    """Record tool executions for analytics, not selection."""
    
    def record_execution(self, query, tool, success, duration_ms):
        """Store for analytics dashboard."""
        
    def get_tool_stats(self) -> Dict[str, ToolStats]:
        """Get success rates, avg duration per tool."""
        
    def get_recent_errors(self) -> List[ToolExecutionRecord]:
        """Get recent failures for debugging."""
```

**Why**: Until you have 10+ tools competing for the same task type, RAG selection adds complexity without value.

#### Option B: Expand Tool Granularity (Future)

Make RAG valuable by having more specialized tools:

```python
# Future: More granular tools where RAG helps
TaskType.DATA → [
    search_databases,           # Generic federated search
    search_tcga,                # TCGA/GDC specialized
    search_geo_rnaseq,          # GEO RNA-seq optimized
    search_geo_chipseq,         # GEO ChIP-seq optimized
    search_encode_tf,           # ENCODE transcription factor
    search_encode_histone,      # ENCODE histone marks
    search_sra_metagenomics,    # SRA metagenomics
]

# Now RAG can learn:
# - "H3K4me3" queries → search_encode_histone (95% success)
# - "liver cancer" queries → search_tcga (90% success)
# - "microbiome" queries → search_sra_metagenomics (85% success)
```

#### Option C: Use RAG for Argument Optimization (Novel Approach)

Instead of tool selection, use RAG to learn optimal **arguments**:

```python
class ArgumentMemory:
    """Learn optimal search parameters from past successes."""
    
    def suggest_filters(self, query: str) -> Dict[str, Any]:
        """
        Learn from past successful queries.
        
        Example:
          Query: "human brain RNA-seq"
          Learned: {"organism": "Homo sapiens", "tissue": "brain", 
                   "assay": "RNA-seq", "limit": 50}
          
        Instead of user specifying all filters, RAG learns common patterns.
        """
```

**This is more valuable** because:
- Users often don't know all the filter options
- Past successful queries encode domain knowledge
- Reduces user friction significantly

---

### Option B + C: Hybrid Approach (RECOMMENDED)

**Options B and C are NOT mutually exclusive.** They operate at different layers and complement each other:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HYBRID RAG ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Layer 3: TOOL SELECTION (Option B - RAGToolSelector)               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Query: "find H3K4me3 ChIP-seq data"                        │   │
│  │                                                              │   │
│  │  Candidate Tools (same TaskType.DATA):                       │   │
│  │    • search_databases      (generic)     → 0.70             │   │
│  │    • search_encode_histone (specialized) → 0.75             │   │
│  │    • search_geo_chipseq    (specialized) → 0.72             │   │
│  │                                                              │   │
│  │  RAG Boost (from similar past queries):                      │   │
│  │    • search_encode_histone + 0.20 boost  → 0.95 ★ SELECTED  │   │
│  │                                                              │   │
│  │  VALUE: Selects BEST tool for the specific query pattern    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  Layer 2: ARGUMENT OPTIMIZATION (Option C - ArgumentMemory)         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Selected Tool: search_encode_histone                        │   │
│  │                                                              │   │
│  │  RAG-Suggested Arguments (from past successes):              │   │
│  │    {                                                         │   │
│  │      "assay": "ChIP-seq",                                   │   │
│  │      "target": "H3K4me3",                                   │   │
│  │      "output_type": "optimal IDR thresholded peaks",        │   │
│  │      "file_format": "bed narrowPeak",                       │   │
│  │      "assembly": "GRCh38"  ← User didn't specify!           │   │
│  │    }                                                         │   │
│  │                                                              │   │
│  │  VALUE: Fills in OPTIMAL parameters user didn't know        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                      │
│  Layer 1: EXECUTION (Current - ToolMemory for Analytics)           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Execute: search_encode_histone(args)                        │   │
│  │                                                              │   │
│  │  Record: {                                                   │   │
│  │    query, tool, args, success, duration, result_count       │   │
│  │  }                                                           │   │
│  │                                                              │   │
│  │  VALUE: Analytics, feeds back to Layers 2 & 3               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Why Hybrid Works Better**:

| Scenario | Tool Selection Only (B) | Argument Opt Only (C) | Hybrid (B+C) |
|----------|------------------------|----------------------|--------------|
| "H3K4me3 data" | Picks `search_encode_histone` ✓ | Generic tool, good args | Best tool + best args ✓✓ |
| "liver cancer RNA-seq" | Picks `search_tcga` ✓ | Generic tool, good args | Best tool + best args ✓✓ |
| Wrong tool selected | Bad results | Good args, wrong source | Rare - both layers compensate |

**Hybrid Implementation Architecture**:

```python
# src/workflow_composer/agents/rag/

# Layer 1: Analytics Foundation (build first)
class ToolMemory:
    """Records all executions. Foundation for Layers 2 & 3."""
    def record(self, query, tool, args, result) -> None
    def get_stats(self) -> Dict[str, ToolStats]

# Layer 2: Argument Optimization (build second)  
class ArgumentMemory:
    """Learns optimal arguments per tool."""
    def __init__(self, tool_memory: ToolMemory):
        self.tool_memory = tool_memory  # Reads from Layer 1
        
    def suggest_args(self, tool: str, query: str) -> Dict[str, Any]:
        """Find similar past queries for this tool, extract successful args."""
        similar = self.tool_memory.find_similar(query, tool_filter=tool)
        return self._merge_common_args(similar)

# Layer 3: Tool Selection (build third)
class RAGToolSelector:
    """Selects best tool from competing options."""
    def __init__(self, tool_memory: ToolMemory):
        self.tool_memory = tool_memory  # Reads from Layer 1
        
    def select(self, query: str, candidates: List[str]) -> Tuple[str, float]:
        """Boost tools that succeeded for similar queries."""

# Orchestrator ties it together
class RAGOrchestrator:
    """Coordinates all RAG layers."""
    def __init__(self):
        self.memory = ToolMemory()           # Layer 1
        self.arg_memory = ArgumentMemory(self.memory)  # Layer 2
        self.selector = RAGToolSelector(self.memory)   # Layer 3
    
    def enhance_execution(
        self, 
        query: str, 
        candidate_tools: List[str],
        base_args: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Returns (best_tool, optimized_args).
        
        1. Layer 3 selects best tool
        2. Layer 2 enhances arguments
        3. Layer 1 will record result after execution
        """
        # Tool selection (Layer 3)
        best_tool, confidence = self.selector.select(query, candidate_tools)
        
        # Argument optimization (Layer 2)
        suggested_args = self.arg_memory.suggest_args(best_tool, query)
        final_args = {**suggested_args, **base_args}  # User args override
        
        return best_tool, final_args
```

### Recommended Implementation Path (Updated for Hybrid)

| Phase | Layer | Action | Effort | Value |
|-------|-------|--------|--------|-------|
| Now | 1 | Keep `ToolMemory`, add persistence (PostgreSQL) | 2 days | Foundation |
| +1 month | 1 | Build analytics dashboard | 3 days | Insight |
| +2 months | 2 | Implement `ArgumentMemory` | 1 week | High user value |
| +3 months | B | Add 5-10 specialized tools per TaskType | 2 weeks | Enables Layer 3 |
| +4 months | 3 | Implement `RAGToolSelector` | 1 week | Full hybrid |
| +5 months | - | Integrate `RAGOrchestrator` in UnifiedAgent | 3 days | Complete |

**Key Insight**: Build Layer 1 first because Layers 2 and 3 both depend on it. Layer 2 (ArgumentMemory) provides value immediately even with few tools. Layer 3 (RAGToolSelector) only adds value once you have competing specialized tools.

---

## Part 2: Layered Architecture Principles

Before diving into component details, here's how all improvements fit together in a **clean, modular architecture**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BIOPIPELINES LAYERED ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LAYER 6: PRESENTATION                                                       │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │   Gradio UI    │  │   FastAPI      │  │   CLI          │                 │
│  │   (existing)   │  │   (NEW)        │  │   (existing)   │                 │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘                 │
│          └───────────────────┼───────────────────┘                          │
│                              ▼                                               │
│  LAYER 5: API GATEWAY                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Auth Middleware │ Rate Limiter │ Request Validation │ CORS          │   │
│  │  (NEW)           │ (extend)     │ (NEW)              │ (NEW)         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              ▼                                               │
│  LAYER 4: APPLICATION SERVICES (Orchestration)                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │   │
│  │  │ BioPipelines│  │ Unified     │  │ RAG         │                   │   │
│  │  │ Facade      │  │ Agent       │  │ Orchestrator│ (NEW - Hybrid)    │   │
│  │  │ (existing)  │  │ (existing)  │  │             │                   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │   │
│  │                                                                       │   │
│  │  Dependencies: Layer 3 (Domain), Layer 2 (Infrastructure)            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              ▼                                               │
│  LAYER 3: DOMAIN SERVICES (Business Logic)                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ Data        │  │ Workflow    │  │ Job         │  │ Diagnosis   │  │   │
│  │  │ Discovery   │  │ Generator   │  │ Manager     │  │ Agent       │  │   │
│  │  │ (existing)  │  │ (existing)  │  │ (existing)  │  │ (existing)  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  │                                                                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │   │
│  │  │ Tool        │  │ Argument    │  │ Tool        │                   │   │
│  │  │ Selector    │  │ Memory      │  │ Memory      │                   │   │
│  │  │ (RAG L3)    │  │ (RAG L2)    │  │ (RAG L1)    │                   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │   │
│  │                                                                       │   │
│  │  Dependencies: Layer 2 (Infrastructure) only                          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              ▼                                               │
│  LAYER 2: INFRASTRUCTURE (Cross-Cutting Concerns)                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ Circuit     │  │ Semantic    │  │ Observa-    │  │ Resilience  │  │   │
│  │  │ Breaker     │  │ Cache       │  │ bility      │  │ (retry,etc) │  │   │
│  │  │ (existing)  │  │ (+ Redis)   │  │ (extend)    │  │ (existing)  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  │                                                                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ LLM         │  │ DI          │  │ Settings    │  │ Exceptions  │  │   │
│  │  │ Adapters    │  │ Container   │  │             │  │             │  │   │
│  │  │ (existing)  │  │ (existing)  │  │ (existing)  │  │ (existing)  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  │                                                                       │   │
│  │  Dependencies: Layer 1 (External) only                                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              ▼                                               │
│  LAYER 1: EXTERNAL INTERFACES (I/O Boundaries)                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │ GEO         │  │ ENCODE      │  │ GDC/TCGA    │  │ SLURM       │  │   │
│  │  │ Adapter     │  │ Adapter     │  │ Adapter     │  │ Client      │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  │                                                                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │   │
│  │  │ PostgreSQL  │  │ Redis       │  │ Filesystem  │                   │   │
│  │  │ (NEW)       │  │ (NEW)       │  │ (existing)  │                   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  LAYER 0: BACKGROUND SERVICES (Async Processing)                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │   │
│  │  │ Celery      │  │ Prefetch    │  │ Scheduled   │                   │   │
│  │  │ Workers     │  │ Manager     │  │ Jobs        │                   │   │
│  │  │ (NEW)       │  │ (optional)  │  │ (future)    │                   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Dependency Rules (CRITICAL for Clean Architecture)

| Layer | Can Depend On | Cannot Depend On |
|-------|--------------|------------------|
| Layer 6 (Presentation) | Layers 5, 4 | Layers 3, 2, 1, 0 |
| Layer 5 (API Gateway) | Layer 4 | Layers 3, 2, 1, 0 |
| Layer 4 (Application) | Layers 3, 2 | Layers 6, 5, 1, 0 |
| Layer 3 (Domain) | Layer 2 | Layers 6, 5, 4, 1, 0 |
| Layer 2 (Infrastructure) | Layer 1 | Layers 6, 5, 4, 3, 0 |
| Layer 1 (External) | Nothing | All other layers |
| Layer 0 (Background) | Layers 3, 2, 1 | Layers 6, 5, 4 |

**Key Principle**: Lower layers NEVER import from higher layers. Use dependency injection and protocols/interfaces to invert dependencies when needed.

### Module Organization

```
src/workflow_composer/
│
├── api/                      # Layer 6 (Presentation) + Layer 5 (Gateway)
│   ├── app.py               # FastAPI app
│   ├── routes/              # Endpoint handlers
│   └── middleware/          # Auth, rate limit, validation
│
├── agents/                   # Layer 4 (Application)
│   ├── unified_agent.py     # Main orchestrator
│   ├── rag/                 # RAG subsystem (NEW)
│   │   ├── orchestrator.py  # RAGOrchestrator
│   │   ├── tool_selector.py # Layer 3 - Tool selection
│   │   ├── arg_memory.py    # Layer 2 - Argument optimization
│   │   └── memory.py        # Layer 1 - Execution recording
│   └── tools/               # Tool implementations
│
├── core/                     # Layer 3 (Domain)
│   ├── workflow_generator.py
│   ├── data_discovery.py
│   └── job_manager.py
│
├── infrastructure/           # Layer 2 (Infrastructure)
│   ├── resilience.py        # Circuit breaker, retry
│   ├── observability.py     # Tracing, metrics
│   ├── semantic_cache.py    # Caching layer
│   └── container.py         # DI container
│
├── adapters/                 # Layer 1 (External Interfaces)
│   ├── geo.py
│   ├── encode.py
│   ├── gdc.py
│   └── slurm.py
│
├── db/                       # Layer 1 (External Interfaces)
│   ├── models.py            # SQLAlchemy models
│   ├── repositories.py      # Data access
│   └── migrations/          # Alembic
│
└── tasks/                    # Layer 0 (Background)
    ├── celery.py            # Task queue config
    └── workflow_tasks.py    # Async tasks
```

---

## Part 3: Component-by-Component Improvement Plan

### 2.1 Circuit Breakers (Phase 1) — ✅ STRENGTHEN

**Current State**: Well-implemented, integrated in GEO/ENCODE/GDC adapters.

**Improvements**:

| Task | Priority | Effort | Description |
|------|----------|--------|-------------|
| Add per-user tracking | High | 2 days | Track failures per user to detect user-specific issues |
| Dashboard metrics | Medium | 1 day | Expose circuit state via `/health` endpoint |
| Recovery notifications | Low | 0.5 day | Notify when circuit closes after being open |
| Configurable thresholds | Medium | 0.5 day | Allow runtime config changes without restart |

**Code changes**:
```python
# Add to resilience.py

class CircuitBreakerRegistry:
    """Central registry for all circuit breakers with monitoring."""
    
    _breakers: Dict[str, CircuitBreaker] = {}
    
    @classmethod
    def get_status(cls) -> Dict[str, Dict]:
        """Get status of all breakers for /health endpoint."""
        return {
            name: {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "last_failure": cb.last_failure_time,
                "success_rate_last_hour": cb.get_success_rate(hours=1),
            }
            for name, cb in cls._breakers.items()
        }
```

---

### 2.2 Parallel Search (Phase 2) — ✅ KEEP AS-IS

**Current State**: Working well, provides 3x speedup.

**Minor improvements**:

| Task | Priority | Effort | Description |
|------|----------|--------|-------------|
| Add timeout per adapter | Low | 0.5 day | Don't wait forever for slow adapters |
| Result streaming | Low | 1 day | Stream results as they arrive (UX improvement) |

---

### 2.3 Observability (Phase 3) — ⚠️ EXTEND INTEGRATION

**Current State**: 864 lines of infrastructure, used in only 1 file.

**Problem**: You built enterprise observability but only instrumented the agent entry point.

**Fix**: Extend `@traced` to more components:

```python
# Files to instrument:

# High value (instrument now):
src/workflow_composer/agents/tools/data_discovery.py    # @traced on search_databases_impl
src/workflow_composer/data/discovery/adapters/geo.py    # @traced on search()
src/workflow_composer/data/discovery/adapters/encode.py # @traced on search()
src/workflow_composer/data/discovery/adapters/gdc.py    # @traced on search()
src/workflow_composer/core/workflow_generator.py        # @traced on generate()

# Medium value (instrument later):
src/workflow_composer/agents/tools/execution.py         # @traced on submit_job_impl
src/workflow_composer/diagnosis/agent.py                # @traced on diagnose()
```

**Implementation checklist**:

| Task | Priority | Effort | Description |
|------|----------|--------|-------------|
| Instrument data adapters | High | 1 day | Add @traced to GEO, ENCODE, GDC adapters |
| Instrument workflow generator | High | 0.5 day | Trace workflow generation time |
| Add trace context propagation | Medium | 1 day | Pass trace_id through async calls |
| Export to JSON file | Medium | 0.5 day | For later import to Jaeger/Zipkin |
| Add custom attributes | Low | 0.5 day | Tag traces with user_id, query_type |

---

### 2.4 Semantic Cache (Phase 4) — ⚠️ ADD PERSISTENCE

**Current State**: In-memory dict, lost on restart, single-process only.

**For multi-user production**:

| Task | Priority | Effort | Description |
|------|----------|--------|-------------|
| Redis backend | Critical | 2 days | Shared cache across processes |
| Cache invalidation API | High | 0.5 day | Manual invalidation for stale data |
| Per-user cache namespacing | Medium | 1 day | Isolate user caches if needed |
| Cache hit rate metrics | Medium | 0.5 day | Track effectiveness |

**Implementation**:

```python
# semantic_cache.py - Add Redis backend

class RedisSemanticCache(SemanticCacheProtocol):
    """Production semantic cache with Redis backend."""
    
    def __init__(self, redis_url: str, embedding_model: str = "bge-small"):
        self.redis = Redis.from_url(redis_url)
        self.embedder = SentenceTransformer(embedding_model)
        
    async def get(self, query: str) -> Optional[CacheEntry]:
        query_embedding = self.embedder.encode(query)
        
        # Use Redis vector similarity search (RediSearch module)
        results = await self.redis.ft("cache_idx").search(
            Query(f"@embedding:[VECTOR_RANGE 0.15 $vec]")
            .sort_by("similarity")
            .return_fields("result", "timestamp")
            .dialect(2)
        )
        # ...

    async def set(self, query: str, result: Any, ttl: int = 3600):
        embedding = self.embedder.encode(query).tolist()
        await self.redis.hset(f"cache:{query_hash}", mapping={
            "embedding": json.dumps(embedding),
            "result": json.dumps(result),
            "timestamp": datetime.now().isoformat(),
        })
        await self.redis.expire(f"cache:{query_hash}", ttl)
```

---

### 2.5 Prefetching (Phase 5) — 🟡 KEEP BUT MAKE OPTIONAL

**Current State**: 759 lines, integrated but unproven value.

**Recommendation**: Keep the code but disable by default until you have data.

| Task | Priority | Effort | Description |
|------|----------|--------|-------------|
| Add feature flag | High | 0.5 day | `PREFETCH_ENABLED=false` by default |
| Track prefetch hit rate | High | 0.5 day | Measure if prefetched data is actually used |
| Add memory limits | Medium | 0.5 day | Don't OOM with too many prefetches |
| Evaluate after 1 month | High | - | Review metrics, decide keep/remove |

**Configuration**:
```yaml
# config/defaults.yaml
prefetch:
  enabled: false  # Disabled until proven useful
  max_concurrent: 5
  max_cache_size_mb: 100
  strategies:
    - encode_details
    - geo_details
    - gdc_details
```

---

### 2.6 RAG Tool Memory (Phase 6) — 🔴 REFOCUS

See Part 1 above for detailed analysis.

**Immediate actions**:

| Task | Priority | Effort | Description |
|------|----------|--------|-------------|
| Remove `RAGToolSelector` | High | 1 hour | Delete class, remove imports |
| Keep `ToolMemory` | High | 0 | For analytics only |
| Add analytics endpoint | Medium | 0.5 day | `/api/tool-stats` for dashboard |
| Add persistence | Medium | 1 day | Save to SQLite/PostgreSQL |

**Slimmed down module** (~300 lines):

```python
# tool_memory.py - Simplified

@dataclass
class ToolExecutionRecord:
    query: str
    tool_name: str
    success: bool
    duration_ms: float
    timestamp: datetime
    error_message: Optional[str] = None

class ToolMemory:
    """Analytics-focused tool execution history."""
    
    def __init__(self, db_path: str = "tool_memory.db"):
        self._db = sqlite3.connect(db_path)
        self._init_schema()
    
    def record(self, record: ToolExecutionRecord) -> None:
        """Store execution for analytics."""
        
    def get_stats(self) -> Dict[str, ToolStats]:
        """Get aggregated stats per tool."""
        
    def get_slow_queries(self, threshold_ms: float = 5000) -> List[ToolExecutionRecord]:
        """Find slow queries for optimization."""
        
    def get_error_patterns(self) -> Dict[str, int]:
        """Group errors by pattern for debugging."""

# Remove: RAGToolSelector, RAGSelectorConfig, ToolBoost, get_rag_selector
```

---

### 2.7 Evaluation Framework (Phase 7) — 🟡 MOVE TO DEV TOOLS

**Current State**: 2566 lines, completely disconnected from production.

**Problem**: It's architected as a runtime component but used as a dev/test tool.

**Fix**: Move to `scripts/` or `tools/` as standalone CLI:

| Task | Priority | Effort | Description |
|------|----------|--------|-------------|
| Create `scripts/run_evaluation.py` | High | 0.5 day | CLI wrapper |
| Add to CI/CD | High | 0.5 day | Run on PRs |
| Remove from main exports | Medium | 0.5 hour | Clean up `__init__.py` |
| Add benchmark versioning | Low | 0.5 day | Track benchmark changes over time |

**New location**:
```
scripts/
  run_evaluation.py      # CLI entry point
  
tools/
  evaluation/            # Moved from src/workflow_composer/evaluation/
    benchmarks.py
    evaluator.py
    scorer.py
    ...
```

**Usage**:
```bash
# Run benchmarks in CI
python scripts/run_evaluation.py --category data_discovery --output reports/

# Compare with baseline
python scripts/run_evaluation.py --compare baseline.json --fail-on-regression
```

---

## Part 4: Critical Missing Components

### 3.1 Authentication & Multi-tenancy — 🔴 CRITICAL

**For 100s of users, you MUST have**:

| Task | Priority | Effort | Description |
|------|----------|--------|-------------|
| API key authentication | Critical | 2 days | Simple API keys for now |
| User model | Critical | 1 day | User table with quotas |
| Request context | Critical | 1 day | Thread-local user context |
| Per-user rate limiting | Critical | 1 day | Extend RateLimiter |

**Implementation**:

```python
# New: src/workflow_composer/auth/

from dataclasses import dataclass
from typing import Optional
import secrets

@dataclass
class User:
    user_id: str
    api_key: str
    email: str
    tier: str  # "free", "pro", "enterprise"
    quota_remaining: int
    created_at: datetime

class AuthService:
    def __init__(self, db: Database):
        self.db = db
        
    def create_api_key(self, user_id: str) -> str:
        key = f"bp_{secrets.token_urlsafe(32)}"
        self.db.store_api_key(user_id, hash_key(key))
        return key
    
    def authenticate(self, api_key: str) -> Optional[User]:
        user = self.db.find_by_api_key(hash_key(api_key))
        if user and user.quota_remaining > 0:
            return user
        return None

# Middleware for FastAPI
async def auth_middleware(request: Request, call_next):
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return JSONResponse({"error": "Missing API key"}, status_code=401)
    
    user = auth_service.authenticate(api_key)
    if not user:
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    
    request.state.user = user
    return await call_next(request)
```

---

### 3.2 REST API Layer — 🔴 CRITICAL

**Current**: Only Gradio web UI.  
**Needed**: REST API for programmatic access.

| Task | Priority | Effort | Description |
|------|----------|--------|-------------|
| FastAPI application | Critical | 2 days | Core API routes |
| OpenAPI documentation | High | 0.5 day | Auto-generated docs |
| Versioned endpoints | High | 0.5 day | `/api/v1/...` |
| Async handlers | High | 1 day | Non-blocking I/O |

**Implementation**:

```python
# New: src/workflow_composer/api/

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel

app = FastAPI(title="BioPipelines API", version="1.0.0")

class SearchRequest(BaseModel):
    query: str
    databases: list[str] = ["geo", "encode", "tcga"]
    limit: int = 50

class SearchResponse(BaseModel):
    results: list[DatasetResult]
    total: int
    cached: bool
    trace_id: str

@app.post("/api/v1/search", response_model=SearchResponse)
async def search_databases(
    request: SearchRequest,
    user: User = Depends(get_current_user),
):
    """Search federated databases for datasets."""
    with tracer.start_span("api.search") as span:
        span.set_attribute("user_id", user.user_id)
        span.set_attribute("query", request.query)
        
        results = await pipeline.search(
            query=request.query,
            databases=request.databases,
            limit=request.limit,
        )
        
        return SearchResponse(
            results=results,
            total=len(results),
            cached=results.from_cache,
            trace_id=span.trace_id,
        )

@app.post("/api/v1/workflows/generate")
async def generate_workflow(request: WorkflowRequest, user: User = Depends(get_current_user)):
    """Generate a Nextflow workflow from natural language."""
    
@app.post("/api/v1/jobs/submit")
async def submit_job(request: JobRequest, user: User = Depends(get_current_user)):
    """Submit a workflow to SLURM."""
    
@app.get("/api/v1/jobs/{job_id}/status")
async def get_job_status(job_id: str, user: User = Depends(get_current_user)):
    """Get job execution status."""
```

---

### 3.3 Persistent Storage — 🔴 CRITICAL

**Current**: In-memory dicts, filesystem.  
**Needed**: Database for users, jobs, cache, history.

| Task | Priority | Effort | Description |
|------|----------|--------|-------------|
| PostgreSQL schema | Critical | 1 day | Users, jobs, executions |
| SQLAlchemy models | Critical | 1 day | ORM layer |
| Redis for cache | Critical | 1 day | Semantic cache, sessions |
| Migration system | High | 0.5 day | Alembic for schema changes |

**Schema**:

```sql
-- Users & Auth
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    api_key_hash VARCHAR(255) NOT NULL,
    tier VARCHAR(50) DEFAULT 'free',
    quota_remaining INTEGER DEFAULT 100,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Job History
CREATE TABLE jobs (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    workflow_type VARCHAR(100),
    status VARCHAR(50),
    slurm_job_id VARCHAR(100),
    submitted_at TIMESTAMP,
    completed_at TIMESTAMP,
    result JSONB
);

-- Tool Execution History (for analytics)
CREATE TABLE tool_executions (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    tool_name VARCHAR(100),
    query TEXT,
    success BOOLEAN,
    duration_ms FLOAT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_jobs_user ON jobs(user_id);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_executions_tool ON tool_executions(tool_name);
```

---

### 3.4 Job Queue — 🔴 CRITICAL

**Current**: Synchronous execution in request thread.  
**Needed**: Background job queue for long-running tasks.

| Task | Priority | Effort | Description |
|------|----------|--------|-------------|
| Celery setup | Critical | 1 day | Task queue with Redis broker |
| Workflow generation task | Critical | 0.5 day | Async workflow creation |
| Job status polling | High | 0.5 day | WebSocket or polling endpoint |
| Retry logic | High | 0.5 day | Automatic retries for failures |

**Implementation**:

```python
# New: src/workflow_composer/tasks/

from celery import Celery

celery = Celery('biopipelines', broker='redis://localhost:6379/0')

@celery.task(bind=True, max_retries=3)
def generate_workflow_task(self, user_id: str, query: str) -> dict:
    """Background task for workflow generation."""
    try:
        result = composer.generate(query)
        return {"status": "completed", "workflow_path": result.path}
    except Exception as e:
        self.retry(exc=e, countdown=60)

@celery.task
def submit_job_task(user_id: str, workflow_path: str) -> dict:
    """Background task for SLURM submission."""
    
# API endpoint
@app.post("/api/v1/workflows/generate")
async def generate_workflow(request: WorkflowRequest, user: User = Depends(get_current_user)):
    task = generate_workflow_task.delay(user.user_id, request.query)
    return {"task_id": task.id, "status": "pending"}

@app.get("/api/v1/tasks/{task_id}")
async def get_task_status(task_id: str):
    task = AsyncResult(task_id)
    return {"status": task.status, "result": task.result if task.ready() else None}
```

---

## Part 5: Implementation Roadmap (Layered Build Order)

**Critical Principle**: Build from bottom up. Each phase completes a layer before moving to the next.

```
BUILD ORDER:
                                                                        
  Week 8  ┌─────────────────────────────────────────────────────────┐
          │  PHASE 4: Production Hardening                          │
          │  - Load testing, security audit, documentation          │
          └─────────────────────────────────────────────────────────┘
                                      ↑
  Week 6  ┌─────────────────────────────────────────────────────────┐
          │  PHASE 3: Observability & RAG Layer 2                   │
          │  - Extend tracing, ArgumentMemory, monitoring           │
          └─────────────────────────────────────────────────────────┘
                                      ↑
  Week 4  ┌─────────────────────────────────────────────────────────┐
          │  PHASE 2: API Layer (Layers 5-6)                        │
          │  - FastAPI routes, async handlers, Celery tasks         │
          └─────────────────────────────────────────────────────────┘
                                      ↑
  Week 2  ┌─────────────────────────────────────────────────────────┐
          │  PHASE 1: Foundation (Layers 1-2)                       │
          │  - PostgreSQL, Redis, Auth, ToolMemory persistence      │
          └─────────────────────────────────────────────────────────┘
                                      ↑
  Week 0  ┌─────────────────────────────────────────────────────────┐
          │  CURRENT STATE: Core working, missing production infra  │
          └─────────────────────────────────────────────────────────┘
```

### Phase 1: Foundation - Layers 1-2 (Weeks 1-2)

| Week | Task | Owner | Deliverable |
|------|------|-------|-------------|
| 1 | PostgreSQL + SQLAlchemy models | - | Database schema, migrations |
| 1 | Redis setup | - | Cache backend, session store |
| 1 | FastAPI skeleton | - | `/health`, `/api/v1/` routes |
| 2 | API key authentication | - | User registration, auth middleware |
| 2 | Migrate SemanticCache to Redis | - | Shared cache across processes |

### Phase 2: API Layer - Layers 5-6 (Weeks 3-4)

**Depends on**: Phase 1 complete (database, Redis, auth)

| Week | Task | Owner | Deliverable |
|------|------|-------|-------------|
| 3 | Search endpoint | - | `POST /api/v1/search` |
| 3 | Workflow generation endpoint | - | `POST /api/v1/workflows/generate` |
| 3 | Celery task queue | - | Background job processing |
| 4 | Job submission endpoint | - | `POST /api/v1/jobs/submit` |
| 4 | Job status endpoint | - | `GET /api/v1/jobs/{id}/status` |

### Phase 3: Observability & RAG Layer 2 (Weeks 5-6)

**Depends on**: Phase 2 complete (API working), ToolMemory has data

| Week | Task | Owner | Deliverable |
|------|------|-------|-------------|
| 5 | Extend @traced to adapters | - | Full distributed tracing |
| 5 | Implement ArgumentMemory (RAG L2) | - | Query argument optimization |
| 5 | Move evaluation to scripts/ | - | CI/CD integration |
| 6 | Per-user rate limiting | - | Quota enforcement |
| 6 | Monitoring dashboard | - | Grafana/custom dashboard |

### Phase 4: Production Hardening (Weeks 7-8)

**Depends on**: Phase 3 complete, system under real load

| Week | Task | Owner | Deliverable |
|------|------|-------|-------------|
| 7 | Load testing | - | 100 concurrent users test |
| 7 | Error handling audit | - | Consistent error responses |
| 8 | Documentation | - | API docs, deployment guide |
| 8 | Security audit | - | Input validation, rate limits |

### Future Phases: RAG Layer 3 & Specialized Tools (Months 3-6)

**Depends on**: Production stable, user data collected, ArgumentMemory proven

| Month | Task | Deliverable |
|-------|------|-------------|
| 3 | Analyze tool usage patterns | Report on which tools need specialization |
| 3-4 | Add 5+ specialized data tools | `search_tcga`, `search_encode_histone`, etc. |
| 4 | Implement RAGToolSelector (Layer 3) | Tool selection from competing options |
| 5 | Integrate RAGOrchestrator | Full hybrid B+C approach |
| 6 | Evaluate & tune | Measure improvement vs baseline |

---

## Part 6: Decision Log

### Decisions Made

| Decision | Rationale | Date |
|----------|-----------|------|
| Hybrid RAG (B+C) approach | Tool selection + argument optimization complement each other | Nov 30, 2025 |
| Build RAG in layers (L1→L2→L3) | Each layer provides value independently; L1 is foundation | Nov 30, 2025 |
| Keep ToolMemory, evolve to ArgumentMemory | Analytics first, then optimization, then selection | Nov 30, 2025 |
| Move evaluation to scripts/ | Not a runtime component; belongs in CI/CD | Nov 30, 2025 |
| Disable prefetching by default | Unproven value; enable after metrics show benefit | Nov 30, 2025 |
| Redis for semantic cache | Multi-process sharing required for production | Nov 30, 2025 |
| PostgreSQL for persistent state | Standard choice for user data, job history | Nov 30, 2025 |
| FastAPI for REST API | Async-native, OpenAPI support, modern Python | Nov 30, 2025 |
| Celery for job queue | Battle-tested, Redis integration | Nov 30, 2025 |
| Layer-by-layer build order | Ensures dependencies exist before dependents | Nov 30, 2025 |

### Decisions Deferred

| Decision | Reason for Deferral | Review Date |
|----------|---------------------|-------------|
| RAGToolSelector (Layer 3) | Need specialized tools first; ArgumentMemory provides interim value | Month 4 |
| Kubernetes deployment | Single-server sufficient initially | Q3 2026 |
| OpenTelemetry export | Current custom tracing sufficient | Q2 2026 |
| GraphQL API | REST sufficient for initial users | Q2 2026 |

---

## Appendix A: File Changes Summary

### Files to Modify

```
src/workflow_composer/agents/tool_memory.py        # Refactor: Split into rag/ module
src/workflow_composer/agents/__init__.py           # Update exports for rag module
src/workflow_composer/agents/unified_agent.py      # Integrate RAGOrchestrator
src/workflow_composer/infrastructure/semantic_cache.py  # Add Redis backend
config/defaults.yaml                                # Add prefetch.enabled=false
```

### Files to Add

```
src/workflow_composer/api/
  __init__.py
  app.py                    # FastAPI application
  routes/
    search.py
    workflows.py
    jobs.py
    health.py
  middleware/
    auth.py
    rate_limit.py
    
src/workflow_composer/auth/
  __init__.py
  service.py
  models.py
  
src/workflow_composer/tasks/
  __init__.py
  celery.py
  workflow_tasks.py
  job_tasks.py

src/workflow_composer/db/
  __init__.py
  models.py                 # SQLAlchemy models
  repositories.py           # Data access layer
  migrations/               # Alembic migrations

src/workflow_composer/agents/rag/    # NEW: Layered RAG subsystem
  __init__.py
  orchestrator.py           # RAGOrchestrator (ties layers together)
  memory.py                 # Layer 1: ToolMemory (execution recording)
  arg_memory.py             # Layer 2: ArgumentMemory (argument optimization)
  tool_selector.py          # Layer 3: RAGToolSelector (tool selection)
  
scripts/
  run_evaluation.py         # Moved from src/
```

### Files to Move/Delete

```
# Move
src/workflow_composer/evaluation/ → tools/evaluation/

# Delete (after slimming)
# None - just reduce code in existing files
```

---

## Appendix B: Metrics to Track

### System Health

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| API response time (p95) | < 2s | > 5s |
| API error rate | < 1% | > 5% |
| Circuit breaker opens | < 2/hour | > 5/hour |
| Cache hit rate | > 30% | < 10% |
| Job queue length | < 50 | > 200 |

### Business Metrics

| Metric | Purpose |
|--------|---------|
| Queries per user per day | Usage patterns |
| Workflow generations per week | Core value delivery |
| Tool success rates | Quality monitoring |
| Time to first result | UX quality |

---

## Appendix C: Modular Development Checklist

Each component should be independently testable and deployable. Use this checklist:

### Component Independence Verification

| Component | Has Interface? | Has Tests? | Can Mock Dependencies? | Can Deploy Alone? |
|-----------|---------------|------------|----------------------|------------------|
| ToolMemory (RAG L1) | ✓ Protocol | ✓ Unit | ✓ Mock DB | ✓ SQLite fallback |
| ArgumentMemory (RAG L2) | ✓ Protocol | ✓ Unit | ✓ Mock L1 | ✓ With L1 |
| RAGToolSelector (RAG L3) | ✓ Protocol | ✓ Unit | ✓ Mock L1 | ✓ With L1 |
| SemanticCache | ✓ Protocol | ✓ Unit | ✓ Mock Redis | ✓ Dict fallback |
| CircuitBreaker | ✓ Class | ✓ Unit | ✓ No deps | ✓ Standalone |
| AuthService | ✓ Protocol | ✓ Unit | ✓ Mock DB | ✓ With DB |
| FastAPI Routes | ✓ Endpoints | ✓ Integration | ✓ Mock services | ✓ With services |

### Feature Flag Strategy

Each new component should be behind a feature flag for gradual rollout:

```yaml
# config/features.yaml
features:
  # RAG Layers
  rag_layer_1_memory: true        # Always on (analytics)
  rag_layer_2_arguments: false    # Enable after Month 2
  rag_layer_3_selection: false    # Enable after Month 4
  
  # Infrastructure
  redis_cache: false              # Enable after Redis deployed
  prefetch: false                 # Enable after metrics prove value
  
  # API
  rest_api: false                 # Enable after Phase 2
  celery_tasks: false             # Enable after Phase 2
```

### Rollback Plan

Each phase should be independently rollback-able:

| Phase | Rollback Procedure | Data Loss Risk |
|-------|-------------------|----------------|
| Phase 1 | Revert migrations, switch to SQLite | Low (new tables) |
| Phase 2 | Disable FastAPI, route to Gradio only | None |
| Phase 3 | Disable ArgumentMemory feature flag | None |
| Phase 4 | N/A (hardening, not new features) | None |

---

*Document maintained by: BioPipelines Team*  
*Last updated: November 30, 2025*
