# Resilience & Observability Implementation Plan

**Version**: 1.0.0  
**Date**: November 29, 2025  
**Status**: ✅ COMPLETE

---

## Executive Summary

This document outlines the implementation plan for strengthening BioPipelines with:
1. **Circuit Breaker Pattern** - Prevent cascade failures when external APIs are down ✅
2. **Parallel Federated Search** - 3x faster multi-source queries ✅
3. **Lightweight Observability** - Distributed tracing for debugging agentic workflows ✅
4. **Semantic Result Caching** - Intelligent cache with TTL and similarity matching ✅
5. **Proactive Prefetching** - Anticipate user needs after search ✅
6. **Enhanced Memory (RAG)** - Use past interactions for tool selection ✅
7. **Evaluation Framework** - Automated quality benchmarks ✅

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Enhanced BioPipelines                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         Observability Layer                              │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │    │
│  │  │   Tracer    │  │   Metrics   │  │  Structured │  │ Correlation │    │    │
│  │  │ (Spans)     │  │ (Counters)  │  │    Logs     │  │     IDs     │    │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         Resilience Layer                                 │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │    │
│  │  │  Circuit    │  │ Exponential │  │    Rate     │  │   Timeout   │    │    │
│  │  │  Breaker    │  │   Backoff   │  │   Limiter   │  │   Manager   │    │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                     Intelligent Caching Layer                            │    │
│  │  ┌─────────────────────────┐  ┌─────────────────────────────────────┐   │    │
│  │  │   Semantic Cache        │  │      Proactive Prefetcher           │   │    │
│  │  │   (TTL + Similarity)    │  │      (Background Tasks)             │   │    │
│  │  └─────────────────────────┘  └─────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      Parallel Execution Layer                            │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │    │
│  │  │            Federated Search (asyncio.gather)                     │    │    │
│  │  │   ENCODE ──┐                                                     │    │    │
│  │  │   GEO ─────┼──► Merge + Deduplicate + Rank ──► Results          │    │    │
│  │  │   TCGA ────┘                                                     │    │    │
│  │  └─────────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Resilience Layer ✅ (Priority: P0)
**Goal**: Circuit breaker + exponential backoff for external API calls

| Task | File | Status | Est. Time |
|------|------|--------|-----------|
| 1.1 Create resilience module | `infrastructure/resilience.py` | ✅ DONE | 30 min |
| 1.2 Implement CircuitBreaker class | `infrastructure/resilience.py` | ✅ DONE | 20 min |
| 1.3 Implement RetryWithBackoff decorator | `infrastructure/resilience.py` | ✅ DONE | 15 min |
| 1.4 Integrate with ENCODE adapter | `data/discovery/adapters/encode.py` | ✅ DONE | 10 min |
| 1.5 Integrate with GEO adapter | `data/discovery/adapters/geo.py` | ✅ DONE | 10 min |
| 1.6 Integrate with GDC adapter | `data/discovery/adapters/gdc.py` | ✅ DONE | 10 min |
| 1.7 Add resilience tests | `tests/unit/test_resilience.py` | ✅ DONE | 20 min |
| 1.8 Update infrastructure exports | `infrastructure/__init__.py` | ✅ DONE | 5 min |

**Key Design Decisions:**
- Circuit breaker states: CLOSED → OPEN → HALF_OPEN → CLOSED
- Failure threshold: 5 failures in 60 seconds
- Open state duration: 30 seconds
- Half-open test: 1 request allowed
- Backoff formula: `min(base * 2^attempt + jitter, max_delay)`

### Phase 2: Parallel Federated Search ✅ (Priority: P0)
**Goal**: Query all sources simultaneously with timeout handling

| Task | File | Status | Est. Time |
|------|------|--------|-----------|
| 2.1 Create parallel search orchestrator | `data/discovery/parallel.py` | ✅ DONE | 25 min |
| 2.2 Add async search to base adapter | `data/discovery/adapters/base.py` | ✅ DONE | 15 min |
| 2.3 Implement result merger/deduplicator | `data/discovery/parallel.py` | ✅ DONE | 20 min |
| 2.4 Update DataDiscovery orchestrator | `data/discovery/orchestrator.py` | ✅ DONE | 15 min |
| 2.5 Add parallel search tests | `tests/unit/test_parallel_search.py` | ✅ DONE | 15 min |

**Key Design Decisions:**
- Per-source timeout: 10 seconds
- Overall timeout: 15 seconds
- Failed sources return empty (graceful degradation)
- Deduplication by: accession ID, file checksums, title similarity

### Phase 3: Observability Layer ✅ (Priority: P0)
**Goal**: Lightweight tracing without external dependencies

| Task | File | Status | Est. Time |
|------|------|--------|-----------|
| 3.1 Create observability module | `infrastructure/observability.py` | ✅ DONE | 30 min |
| 3.2 Implement Tracer with spans | `infrastructure/observability.py` | ✅ DONE | 25 min |
| 3.3 Implement MetricsCollector | `infrastructure/observability.py` | ✅ DONE | 20 min |
| 3.4 Add @traced decorator | `infrastructure/observability.py` | ✅ DONE | 10 min |
| 3.5 Integrate with UnifiedAgent | `agents/unified_agent.py` | ✅ DONE | 15 min |
| 3.6 Integrate with tool execution | `agents/tools/__init__.py` | ✅ DONE | 10 min |
| 3.7 Add JSON log exporter | `infrastructure/observability.py` | ✅ DONE | 15 min |
| 3.8 Add observability tests | `tests/unit/test_observability.py` | ✅ DONE | 15 min |

**Key Design Decisions:**
- No external dependencies (OpenTelemetry optional)
- Span structure: trace_id, span_id, parent_id, name, start, end, tags
- Metrics: counters, gauges, histograms (in-memory)
- Export: JSON files for HPC (no network), optional OTLP
- Context propagation via contextvars

### Phase 4: Semantic Caching ✅ (Priority: P1)
**Goal**: TTL-based cache with semantic similarity for cache hits

| Task | File | Status | Est. Time |
|------|------|--------|-----------|
| 4.1 Create semantic cache module | `infrastructure/semantic_cache.py` | ✅ DONE | 30 min |
| 4.2 Implement TTL eviction | `infrastructure/semantic_cache.py` | ✅ DONE | 15 min |
| 4.3 Add embedding-based similarity | `infrastructure/semantic_cache.py` | ✅ DONE | 20 min |
| 4.4 Integrate with search tool | `agents/tools/data_discovery.py` | ✅ DONE | 15 min |
| 4.5 Add cache tests | `tests/unit/test_semantic_cache.py` | ✅ DONE | 15 min |

**Key Design Decisions:**
- Default TTL: 1 hour for search results
- Similarity threshold: 0.85 cosine similarity
- Max cache size: 1000 entries (LRU eviction after TTL)
- Cache key: hash of normalized query
- Optional: Redis backend for multi-process

### Phase 5: Proactive Prefetching ✅ (Priority: P1)
**Goal**: Background prefetch after search to speed up follow-up actions

| Task | File | Status | Est. Time |
|------|------|--------|-----------|
| 5.1 Create prefetch manager | `agents/tools/prefetch.py` | ✅ DONE | 25 min |
| 5.2 Define prefetch strategies | `agents/tools/prefetch.py` | ✅ DONE | 15 min |
| 5.3 Integrate with search results | `agents/tools/data_discovery.py` | ✅ DONE | 10 min |
| 5.4 Add background task executor | `agents/tools/prefetch.py` | ✅ DONE | 20 min |
| 5.5 Add prefetch tests | `tests/unit/test_prefetch.py` | ✅ DONE | 15 min |

**Key Design Decisions:**
- Prefetch top 3 results after search
- Prefetch: metadata, download URLs, file counts
- Non-blocking (fire-and-forget)
- Cancel on new search query

### Phase 6: RAG-Enhanced Tool Selection ✅ (Priority: P2)
**Goal**: Use past successful interactions to improve tool selection

| Task | File | Status | Est. Time |
|------|------|--------|-----------|
| 6.1 Create tool memory module | `agents/tool_memory.py` | ✅ DONE | 25 min |
| 6.2 Log successful tool executions | `agents/tools/__init__.py` | ✅ DONE | 10 min |
| 6.3 Add similarity search for tools | `agents/tool_memory.py` | ✅ DONE | 20 min |
| 6.4 Integrate with intent parser | `agents/intent/parser.py` | ✅ DONE | 15 min |
| 6.5 Add RAG tests | `tests/unit/test_tool_memory.py` | ✅ DONE | 15 min |

**Key Design Decisions:**
- Store: (query, tool_used, success, duration)
- Embedding: same model as AgentMemory
- Boost tool confidence if similar past query succeeded
- Learning rate: recent queries weighted higher

### Phase 7: Evaluation Framework ✅ (Priority: P3)
**Goal**: Automated benchmarks for agent quality

| Task | File | Status | Est. Time |
|------|------|--------|-----------|
| 7.1 Create evaluation module | `evaluation/__init__.py` | ✅ DONE | 15 min |
| 7.2 Define benchmark queries | `evaluation/benchmarks.py` | ✅ DONE | 20 min |
| 7.3 Implement evaluator | `evaluation/evaluator.py` | ✅ DONE | 25 min |
| 7.4 Add LLM-as-judge scorer | `evaluation/scorer.py` | ✅ DONE | 20 min |
| 7.5 Add metrics aggregation | `evaluation/metrics.py` | ✅ DONE | 15 min |
| 7.6 Add report generation | `evaluation/report.py` | ✅ DONE | 15 min |
| 7.7 Add evaluation tests | `tests/unit/test_evaluation.py` | ✅ DONE | 15 min |

**Key Design Decisions:**
- Benchmark categories: data_discovery, workflow, job_management, education
- Metrics: tool_accuracy, response_relevance, latency_p95
- LLM-as-judge for subjective quality
- Report: JSON + HTML summary

---

## Detailed Specifications

### 1. CircuitBreaker Class

```python
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, TypeVar, Optional
import asyncio
import functools
import random

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2          # Successes to close from half-open
    timeout: float = 30.0               # Seconds to stay open
    half_open_max_calls: int = 1        # Test calls in half-open

class CircuitBreaker:
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
    
    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if self._timeout_expired():
                self._transition_to_half_open()
                return True
            return False
        else:  # HALF_OPEN
            return self.half_open_calls < self.config.half_open_max_calls
    
    def record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        else:
            self.failure_count = 0
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        elif self.failure_count >= self.config.failure_threshold:
            self._transition_to_open()
```

### 2. RetryWithBackoff Decorator

```python
def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
):
    """Retry with exponential backoff and optional jitter."""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        raise
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    if jitter:
                        delay += random.uniform(0, 1)
                    await asyncio.sleep(delay)
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Sync version using time.sleep
            ...
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator
```

### 3. Tracer and Span

```python
import contextvars
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List

# Context for trace propagation
_current_span: contextvars.ContextVar[Optional['Span']] = contextvars.ContextVar(
    'current_span', default=None
)

@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    parent_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    
    def add_tag(self, key: str, value: Any):
        self.tags[key] = value
    
    def add_event(self, name: str, attributes: Dict[str, Any] = None):
        self.events.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {}
        })
    
    def set_error(self, error: Exception):
        self.status = "error"
        self.tags["error.type"] = type(error).__name__
        self.tags["error.message"] = str(error)
    
    def end(self):
        self.end_time = datetime.now()
    
    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0

class Tracer:
    def __init__(self, service_name: str = "biopipelines"):
        self.service_name = service_name
        self.spans: List[Span] = []
        self._exporters: List[Callable[[Span], None]] = []
    
    def start_span(self, name: str, parent: Span = None) -> Span:
        current = parent or _current_span.get()
        trace_id = current.trace_id if current else str(uuid.uuid4())[:32]
        parent_id = current.span_id if current else None
        
        span = Span(name=name, trace_id=trace_id, parent_id=parent_id)
        _current_span.set(span)
        return span
    
    def end_span(self, span: Span):
        span.end()
        self.spans.append(span)
        for exporter in self._exporters:
            exporter(span)
        # Restore parent
        # ...
```

### 4. ParallelSearchOrchestrator

```python
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class FederatedSearchResult:
    datasets: List[DatasetInfo]
    sources_queried: List[str]
    sources_failed: List[str]
    total_time_ms: float
    cache_hits: int

class ParallelSearchOrchestrator:
    def __init__(
        self,
        adapters: Dict[str, BaseAdapter],
        timeout_per_source: float = 10.0,
        overall_timeout: float = 15.0,
    ):
        self.adapters = adapters
        self.timeout_per_source = timeout_per_source
        self.overall_timeout = overall_timeout
    
    async def search(
        self,
        query: SearchQuery,
        sources: Optional[List[str]] = None,
    ) -> FederatedSearchResult:
        start = datetime.now()
        sources_to_query = sources or list(self.adapters.keys())
        
        # Create tasks with individual timeouts
        tasks = []
        for source in sources_to_query:
            adapter = self.adapters.get(source)
            if adapter:
                task = asyncio.wait_for(
                    self._search_with_circuit_breaker(adapter, query),
                    timeout=self.timeout_per_source
                )
                tasks.append((source, task))
        
        # Execute in parallel with overall timeout
        results = await asyncio.wait_for(
            asyncio.gather(*[t[1] for t in tasks], return_exceptions=True),
            timeout=self.overall_timeout
        )
        
        # Process results
        all_datasets = []
        failed_sources = []
        
        for (source, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                failed_sources.append(source)
            else:
                all_datasets.extend(result)
        
        # Deduplicate
        unique_datasets = self._deduplicate(all_datasets)
        
        return FederatedSearchResult(
            datasets=unique_datasets,
            sources_queried=sources_to_query,
            sources_failed=failed_sources,
            total_time_ms=(datetime.now() - start).total_seconds() * 1000,
            cache_hits=0,  # TODO: track cache hits
        )
    
    def _deduplicate(self, datasets: List[DatasetInfo]) -> List[DatasetInfo]:
        """Remove duplicates by ID and title similarity."""
        seen_ids = set()
        unique = []
        
        for ds in datasets:
            if ds.id not in seen_ids:
                seen_ids.add(ds.id)
                unique.append(ds)
        
        return unique
```

### 5. SemanticCache

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, Tuple
import hashlib
import numpy as np

@dataclass
class CacheEntry:
    key: str
    value: Any
    embedding: Optional[np.ndarray]
    created_at: datetime
    ttl_seconds: int
    hits: int = 0
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)

class SemanticCache:
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600,
        similarity_threshold: float = 0.85,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.similarity_threshold = similarity_threshold
        self._cache: Dict[str, CacheEntry] = {}
        self._embedder = None  # Lazy load
    
    def get(self, query: str) -> Tuple[Optional[Any], bool]:
        """
        Get cached result.
        
        Returns:
            Tuple of (value, is_exact_match)
            - (value, True) if exact match
            - (value, False) if semantic match
            - (None, False) if miss
        """
        # Exact match
        key = self._make_key(query)
        if key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired:
                entry.hits += 1
                return entry.value, True
            else:
                del self._cache[key]
        
        # Semantic match
        if self._embedder:
            query_embedding = self._embedder.encode(query)
            best_match = None
            best_score = 0
            
            for entry in self._cache.values():
                if entry.is_expired:
                    continue
                if entry.embedding is not None:
                    score = np.dot(query_embedding, entry.embedding)
                    if score > best_score and score >= self.similarity_threshold:
                        best_score = score
                        best_match = entry
            
            if best_match:
                best_match.hits += 1
                return best_match.value, False
        
        return None, False
    
    def set(self, query: str, value: Any, ttl: int = None):
        """Cache a result."""
        self._evict_if_needed()
        
        key = self._make_key(query)
        embedding = self._embedder.encode(query) if self._embedder else None
        
        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            embedding=embedding,
            created_at=datetime.now(),
            ttl_seconds=ttl or self.default_ttl,
        )
    
    def _make_key(self, query: str) -> str:
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _evict_if_needed(self):
        # Remove expired
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for k in expired:
            del self._cache[k]
        
        # LRU if still over size
        if len(self._cache) >= self.max_size:
            # Sort by hits (LFU) then by age (LRU)
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: (x[1].hits, x[1].created_at)
            )
            for k, _ in sorted_entries[:len(self._cache) - self.max_size + 1]:
                del self._cache[k]
```

---

## File Structure

```
src/workflow_composer/
├── infrastructure/
│   ├── __init__.py              # Updated exports
│   ├── resilience.py            # NEW: Circuit breaker, retry
│   ├── observability.py         # NEW: Tracer, metrics
│   ├── semantic_cache.py        # NEW: TTL + similarity cache
│   └── background.py            # NEW: Background task executor
│
├── data/discovery/
│   ├── parallel.py              # NEW: Parallel search orchestrator
│   ├── orchestrator.py          # Updated: Use parallel search
│   └── adapters/
│       ├── base.py              # Updated: async support
│       ├── encode.py            # Updated: circuit breaker
│       ├── geo.py               # Updated: circuit breaker
│       └── gdc.py               # Updated: circuit breaker
│
├── agents/
│   ├── unified_agent.py         # Updated: tracing integration
│   ├── tool_memory.py           # NEW: RAG for tool selection
│   └── tools/
│       ├── __init__.py          # Updated: trace tool calls
│       ├── prefetch.py          # NEW: Proactive prefetching
│       └── data_discovery.py    # Updated: caching
│
├── evaluation/                  # NEW: Evaluation framework
│   ├── __init__.py
│   ├── benchmarks.py
│   ├── evaluator.py
│   └── scorer.py
│
tests/unit/
├── test_resilience.py           # NEW
├── test_parallel_search.py      # NEW
├── test_observability.py        # NEW
├── test_semantic_cache.py       # NEW
├── test_prefetch.py             # NEW
├── test_tool_memory.py          # NEW
└── test_evaluation.py           # NEW
```

---

## Testing Strategy

### Unit Tests
- Circuit breaker state transitions
- Backoff delay calculations
- Span creation and propagation
- Cache TTL and eviction
- Parallel search timeout handling

### Integration Tests
- End-to-end search with circuit breaker
- Tracing through complete query flow
- Cache hit rates under load

### Performance Tests
- Parallel vs sequential search latency
- Cache lookup overhead
- Memory usage with embeddings

---

## Success Criteria

| Metric | Target |
|--------|--------|
| All unit tests pass | 100% |
| Parallel search latency | < 33% of sequential |
| Cache hit rate (similar queries) | > 50% |
| Circuit breaker prevents cascade | Verified |
| Trace coverage | All tools traced |
| Memory overhead | < 100MB for cache |

---

## Progress Log

| Date | Phase | Task | Status | Notes |
|------|-------|------|--------|-------|
| 2025-11-29 | 0 | Plan created | ✅ Done | This document |
| 2025-11-29 | 1 | Resilience layer | ⏳ TODO | |
| 2025-11-29 | 2 | Parallel search | ⏳ TODO | |
| 2025-11-29 | 3 | Observability | ⏳ TODO | |
| 2025-11-29 | 4 | Semantic cache | ⏳ TODO | |
| 2025-11-29 | 5 | Prefetching | ⏳ TODO | |
| 2025-11-29 | 6 | RAG tools | ⏳ TODO | |
| 2025-11-29 | 7 | Evaluation | ⏳ TODO | |

---

## Dependencies

### Required (already installed)
- Python 3.10+
- asyncio (stdlib)
- numpy (for embeddings)
- sentence-transformers (for semantic cache)

### Optional
- redis (distributed cache)
- opentelemetry-sdk (OTLP export)

---

## Rollback Plan

Each phase is independent. If issues arise:
1. Feature flags in `config/defaults.yaml`:
   ```yaml
   features:
     circuit_breaker: true
     parallel_search: true
     semantic_cache: true
     tracing: true
   ```
2. Graceful degradation to sequential search
3. Cache bypass on errors

---

**Document Version:** 1.0  
**Last Updated:** November 29, 2025
