# Hierarchical Intent Parsing with LLM Arbiter

## Executive Summary

This document describes a **hierarchical intent parsing architecture** that combines fast pattern-based methods with intelligent LLM arbitration. The goal is to achieve **>95% accuracy** while keeping **LLM costs under control** by only invoking expensive LLM calls when truly needed.

**Key Insight**: ~80% of user queries are "easy" cases where traditional methods agree. Only ~20% require LLM intervention.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Proposed Solution](#proposed-solution)
3. [Architecture Overview](#architecture-overview)
4. [Component Details](#component-details)
5. [Decision Flow](#decision-flow)
6. [When LLM is Invoked](#when-llm-is-invoked)
7. [Implementation Status](#implementation-status)
8. [Integration Points](#integration-points)
9. [Expected Benefits](#expected-benefits)
10. [Performance Metrics](#performance-metrics)
11. [Next Steps](#next-steps)

---

## Problem Statement

### Current Challenges

1. **Pattern-Based Parsing Limitations**
   - Adding patterns for one category often causes regressions in others
   - Unsustainable as test set grows (currently 6,395 conversations)
   - Hard to handle edge cases without creating conflicts
   - Pattern maintenance becomes exponentially complex

2. **Pure LLM Approach Drawbacks**
   - Expensive (API costs for every query)
   - Slower latency (~500ms-2s per call)
   - Overkill for simple queries like "create RNA-seq workflow"

3. **Current Accuracy Gap**
   - Pattern-only: ~73% baseline on standardized benchmark
   - Target: >95% accuracy across all categories
   - Current best: 87.4% on full dataset (6,395 samples)

### Category-Specific Issues

| Category | Challenge |
|----------|-----------|
| adversarial | Malformed queries, edge cases |
| ambiguous | Multiple valid interpretations |
| negation | "NOT this", "but X" constructs |
| multi_turn | Context dependencies, coreference |
| data_discovery | Overlaps with workflow intents |

---

## Proposed Solution

### Hierarchical Intent Parsing

Instead of choosing between patterns OR LLM, we use **both in a hierarchy**:

```
┌─────────────────────────────────────────────────────────────┐
│                     User Query                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Stage 1: Fast Methods                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Pattern     │  │ Semantic    │  │ Entity      │          │
│  │ Matching    │  │ Similarity  │  │ Extraction  │          │
│  │ (~1ms)      │  │ (~10ms)     │  │ (~5ms)      │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Stage 2: Arbiter Logic                      │
│                                                              │
│  Check: Do all methods agree? Is confidence high?            │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ IF unanimous agreement AND confidence > 0.7:            ││
│  │    → Return immediately (no LLM needed)                  ││
│  │ ELSE:                                                    ││
│  │    → Proceed to Stage 3                                  ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                    (Only if needed)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Stage 3: LLM Arbiter                        │
│                                                              │
│  Input:                                                      │
│  - Original query                                            │
│  - Pattern parser result + confidence                        │
│  - Semantic parser result + confidence                       │
│  - Entity parser result                                      │
│  - Conversation context                                      │
│                                                              │
│  Output:                                                     │
│  - Final intent + confidence                                 │
│  - Reasoning for decision                                    │
│  - Validated entities                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture Overview

### New Module: `arbiter.py`

Located at: `src/workflow_composer/agents/intent/arbiter.py`

```python
# Key Classes

class ArbiterStrategy(Enum):
    """Controls when LLM is invoked."""
    ALWAYS_LLM = auto()    # Every query (expensive, debugging only)
    NEVER_LLM = auto()     # Never use LLM (fast, lower accuracy)
    SMART = auto()         # Intelligent invocation (recommended)

class ArbiterResult:
    """Result from the arbiter."""
    final_intent: str
    confidence: float
    method: str           # e.g., "unanimous", "llm_arbitration"
    llm_invoked: bool
    reasoning: Optional[str]

class IntentArbiter:
    """LLM-based final decision maker."""
    def arbitrate(
        self,
        query: str,
        pattern_result: IntentResult,
        semantic_result: Dict,
        context: Optional[ConversationContext] = None
    ) -> ArbiterResult

class UnifiedIntentParser:
    """Main entry point - orchestrates all parsers."""
    def parse(self, query: str) -> ArbiterResult
```

---

## Component Details

### 1. Pattern Parser (`parser.py`)

**Purpose**: Fast, deterministic matching using regex and keyword patterns.

**Strengths**:
- Extremely fast (~1ms)
- Zero cost (no API calls)
- Deterministic (same input = same output)
- Good for well-structured queries

**Weaknesses**:
- Brittle with novel phrasings
- Pattern conflicts as coverage increases
- Poor with negation and context

**Current Coverage**:
- 26 intent types
- ~200 pattern rules
- 87% accuracy on known patterns

### 2. Semantic Parser (`semantic.py`)

**Purpose**: Similarity-based matching using sentence embeddings.

**Strengths**:
- Handles paraphrases ("run RNA-seq" vs "execute RNA sequencing")
- Language-model aware
- Good generalization

**Weaknesses**:
- Requires embedding model
- Slower than patterns (~10ms)
- Less precise for exact matches

**Model**: `all-MiniLM-L6-v2` (384-dim embeddings)

### 3. Entity Parser

**Purpose**: Extract named entities (organisms, tissues, data formats).

**Strengths**:
- Provides structured information
- Helps disambiguate intent

**Entities Tracked**:
- Organisms (human, mouse, arabidopsis, etc.)
- Tissues (brain, heart, liver, etc.)
- Data formats (FASTQ, BAM, VCF, etc.)
- Tool names (STAR, hisat2, salmon, etc.)

### 4. LLM Arbiter (NEW)

**Purpose**: Make final decision when fast methods disagree or have low confidence.

**Strengths**:
- Understands complex context
- Handles ambiguity intelligently
- Can explain its reasoning

**Cost Control**:
- Only invoked when needed (~20% of queries)
- Uses fast models when possible (GPT-3.5-turbo first)
- Caches decisions for similar queries

---

## Decision Flow

### When Methods Agree (80% of cases)

```
Query: "create RNA-seq workflow for mouse brain"

Pattern:  WORKFLOW_CREATE (conf: 0.95)
Semantic: WORKFLOW_CREATE (conf: 0.88)
Entity:   organism=mouse, tissue=brain

Result: WORKFLOW_CREATE, confidence=0.91
        method="unanimous", llm_invoked=False
```

**Time**: ~15ms, **Cost**: $0

### When Methods Disagree (15% of cases)

```
Query: "I want to analyze some data"

Pattern:  WORKFLOW_CREATE (conf: 0.45)  # "analyze" triggered
Semantic: DATA_DISCOVERY  (conf: 0.52)  # "data" weighted
Entity:   (none extracted)

→ LLM Arbiter Invoked

LLM Input:
  - Query: "I want to analyze some data"
  - Pattern says: WORKFLOW_CREATE (0.45)
  - Semantic says: DATA_DISCOVERY (0.52)
  - No entities extracted

LLM Output:
  - Intent: AMBIGUOUS_QUERY
  - Reasoning: "Query lacks specifics about workflow type or data 
    location. Need clarification."
  - Confidence: 0.85

Result: AMBIGUOUS_QUERY, confidence=0.85
        method="llm_arbitration", llm_invoked=True
```

**Time**: ~600ms, **Cost**: ~$0.0003

### When Complexity Detected (5% of cases)

```
Query: "NOT RNA-seq, I need ChIP-seq for the data we discussed"

Complexity Detection:
  - Negation word: "NOT"
  - Context dependency: "we discussed"
  → Force LLM invocation

Result: WORKFLOW_CREATE (ChIP-seq), confidence=0.92
        method="llm_complexity", llm_invoked=True
```

---

## When LLM is Invoked

The arbiter uses **smart invocation** based on these signals:

### 1. Disagreement Detection

```python
def _needs_arbitration(self, pattern_result, semantic_result) -> bool:
    # Different intents
    if pattern_result.intent != semantic_result.get('intent'):
        return True
    
    # Low confidence from both
    if pattern_result.confidence < 0.6 and semantic_result['confidence'] < 0.6:
        return True
    
    return False
```

### 2. Complexity Detection

```python
COMPLEXITY_SIGNALS = {
    'negation': ['not', 'don\'t', 'without', 'except', 'but not'],
    'composite': ['after that', 'then also', 'and then'],
    'context': ['we discussed', 'mentioned earlier', 'the one'],
    'conditional': ['if', 'unless', 'depending on'],
}

def detect_complexity(query: str) -> bool:
    query_lower = query.lower()
    for category, patterns in COMPLEXITY_SIGNALS.items():
        if any(p in query_lower for p in patterns):
            return True
    return False
```

### 3. Confidence Threshold

```python
CONFIDENCE_THRESHOLD = 0.7  # Below this, consult LLM

if max_confidence < CONFIDENCE_THRESHOLD:
    return invoke_llm_arbiter()
```

---

## Implementation Status

### Completed ✅

1. **Core Arbiter Module** (`arbiter.py`)
   - `ArbiterStrategy` enum
   - `ArbiterResult` dataclass
   - `IntentArbiter` class with LLM integration
   - `CascadingProviderRouter` for rate-limit resistance
   - Complexity detection logic
   - LLM prompt templates

2. **UnifiedIntentParser** (`unified_parser.py`)
   - Orchestrates pattern + semantic + arbiter
   - LRU cache for LLM decisions (1000 entries)
   - Metrics tracking (LLM rate, latency)
   - Graceful fallback to pattern parser

3. **Frontend Integration** (`unified_agent.py`)
   - ✅ UnifiedAgent now uses UnifiedIntentParser
   - ✅ Entity attribute access fixed (type/value)
   - ✅ Deprecated UnifiedEnsembleParser deleted (855 lines)

4. **Pattern Parser Improvements**
   - 95.5% accuracy on 100-sample benchmark
   - Added workflow analysis patterns
   - Added education patterns
   - Added multi-turn context patterns

5. **Provider Cascade** (`providers/router.py`)
   - Lightning (1) → GitHub Models (2) → Gemini (3) → Ollama (5) → vLLM (6) → OpenAI (99)
   - Rate limit handling with cooldown periods
   - Automatic fallback on errors

### Performance Metrics

Based on actual evaluation (December 2025):
- **LLM Invocation Rate**: ~40% (targeting 20% with threshold tuning)
- **Overall Accuracy**: 87.4% on full dataset (6,395 samples)
- **Pattern-only accuracy**: 73% baseline
- **With arbiter accuracy**: 87-95% (depends on strategy)

---

## Integration Points

### Current Architecture (v2.2 - IMPLEMENTED)

> **Note**: The previous `UnifiedEnsembleParser` (855 lines) was deleted in December 2025.
> The frontend now uses `UnifiedIntentParser` which provides the hierarchical parsing with LLM arbiter.

### Current Architecture (v2.2 - IMPLEMENTED)

```
gradio_app.py
     │
     ▼
unified_agent.py
     │
     ├── _intent_parser (UnifiedIntentParser)  ← CURRENT (since Dec 2025)
     │        │
     │        ├── Pattern parser (IntentParser)
     │        ├── Semantic parser (SemanticIntentClassifier)
     │        ├── Entity extractor
     │        └── LLM Arbiter (IntentArbiter)  ← SMART INVOCATION (~20% of queries)
     │
     └── dialogue_manager (DialogueManager)
              │
              └── Uses context from UnifiedIntentParser
```

### Integration Code

```python
# In ChatIntegration.__init__

from .arbiter import UnifiedIntentParser, ArbiterStrategy

class ChatIntegration:
    def __init__(self, llm_client=None, strategy=ArbiterStrategy.SMART):
        # Replace simple parser with unified parser
        self.parser = UnifiedIntentParser(
            llm_client=llm_client,
            strategy=strategy
        )
        
# In DialogueManager

class DialogueManager:
    def __init__(self, intent_parser=None, context=None):
        if intent_parser is None:
            from .arbiter import UnifiedIntentParser
            intent_parser = UnifiedIntentParser()
        self.parser = intent_parser
```

---

## Expected Benefits

### 1. Accuracy Improvement

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Overall Accuracy | 87.4% | >95% |
| Adversarial | 88% | 98% |
| Ambiguous | 95% | 99% |
| Negation | 85% | 95% |
| Multi-turn | 96% | 99% |

### 2. Cost Efficiency

| Scenario | Pattern-Only | LLM-Only | Hierarchical |
|----------|--------------|----------|--------------|
| Queries/day | 1,000 | 1,000 | 1,000 |
| LLM calls | 0 | 1,000 | ~200 (20%) |
| Cost/day | $0 | ~$0.50 | ~$0.10 |
| Cost/month | $0 | ~$15 | ~$3 |

### 3. Latency Profile

| Query Type | Pattern-Only | Hierarchical |
|------------|--------------|--------------|
| Simple (80%) | 15ms | 15ms |
| Complex (20%) | 15ms | 600ms |
| Average | 15ms | ~130ms |

### 4. Maintainability

- **Reduced Pattern Burden**: No need to add patterns for every edge case
- **LLM Handles Edge Cases**: Adversarial and ambiguous queries resolved by LLM
- **Incremental Improvement**: Can tune thresholds without code changes
- **Explainable Decisions**: LLM provides reasoning for complex cases

---

## Performance Metrics

### Tracking Dashboard

```python
# Metrics to track

arbiter_metrics = {
    # Invocation stats
    "llm_invocation_rate": 0.0,  # % of queries needing LLM
    "unanimous_rate": 0.0,       # % of queries with agreement
    
    # Accuracy
    "overall_accuracy": 0.0,
    "per_category_accuracy": {},
    
    # Latency
    "avg_latency_ms": 0.0,
    "p95_latency_ms": 0.0,
    "p99_latency_ms": 0.0,
    
    # Cost
    "llm_cost_per_1k_queries": 0.0,
    "cache_hit_rate": 0.0,
}
```

### Success Criteria

1. **Accuracy**: ≥95% on standardized 100-sample benchmark
2. **LLM Rate**: ≤25% of queries invoke LLM
3. **Latency**: P95 ≤ 1s for all queries
4. **Cost**: ≤$0.15 per 1,000 queries

---

## Next Steps

### Phase 1: Integration (1-2 days)

1. Connect `UnifiedIntentParser` to `DialogueManager`
2. Update `ChatIntegration` to use arbiter
3. Add configuration for strategy selection
4. Basic logging and metrics

### Phase 2: Tuning (2-3 days)

1. Run full evaluation with arbiter enabled
2. Measure LLM invocation rate by category
3. Tune confidence thresholds
4. A/B test SMART vs ALWAYS_LLM

### Phase 3: Optimization (1-2 days)

1. Implement decision caching
2. Add semantic similarity for cache lookup
3. Optimize LLM prompts for speed
4. Add batch processing for multiple queries

### Phase 4: Production (1 day)

1. Add comprehensive error handling
2. Implement graceful degradation
3. Set up monitoring alerts
4. Documentation and runbook

---

## Appendix: LLM Prompt Template

```python
ARBITER_PROMPT = """You are an intent classification arbiter for a bioinformatics 
pipeline system. Multiple parsing methods have analyzed the user's query and 
produced different results. Your job is to make the final decision.

USER QUERY: {query}

PARSING RESULTS:
1. Pattern Parser: {pattern_intent} (confidence: {pattern_conf:.2f})
2. Semantic Parser: {semantic_intent} (confidence: {semantic_conf:.2f})
3. Entities Detected: {entities}

CONVERSATION CONTEXT:
{context_summary}

AVAILABLE INTENTS:
- WORKFLOW_CREATE: Create a bioinformatics workflow
- DATA_SEARCH: Search for datasets online
- DATA_SCAN: Scan local data directories
- JOB_STATUS: Check job status
- EDUCATION: Learn about bioinformatics concepts
- GREETING: Casual conversation
- (... full list ...)

Respond with JSON:
{{
    "intent": "<final_intent>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation>"
}}
"""
```

---

## Summary

The hierarchical intent parsing architecture provides:

1. **Best of both worlds**: Fast patterns for simple queries, LLM for complex ones
2. **Cost control**: Only ~20% of queries need LLM
3. **High accuracy**: Expected >95% across all categories
4. **Maintainability**: LLM handles edge cases without pattern explosion
5. **Explainability**: LLM provides reasoning for complex decisions

This approach directly addresses the scalability concerns with pattern-based parsing while avoiding the cost and latency issues of pure LLM approaches.
