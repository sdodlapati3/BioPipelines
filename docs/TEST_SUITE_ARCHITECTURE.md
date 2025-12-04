# BioPipelines Chat Agent Test Suite Architecture

## Executive Summary

This document provides a comprehensive technical specification of the BioPipelines chat agent testing infrastructure. The system is designed to systematically evaluate, measure, and improve the conversational AI capabilities of the workflow composer through progressive complexity tiers.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Components](#2-architecture-components)
3. [Test Data Structure](#3-test-data-structure)
4. [Evaluation Metrics](#4-evaluation-metrics)
5. [Execution Pipeline](#5-execution-pipeline)
6. [Baseline & Regression System](#6-baseline--regression-system)
7. [Error Analysis Framework](#7-error-analysis-framework)
8. [CI/CD Integration](#8-cicd-integration)
9. [Current Limitations](#9-current-limitations)
10. [Improvement Roadmap](#10-improvement-roadmap)

---

## 1. System Overview

### 1.1 Purpose

The test suite serves multiple critical functions:

| Function | Description |
|----------|-------------|
| **Validation** | Verify intent classification accuracy across query types |
| **Regression Detection** | Catch performance degradation before deployment |
| **Coverage Analysis** | Identify gaps in parser pattern coverage |
| **Benchmarking** | Track improvements over time with quantitative metrics |
| **Debugging** | Pinpoint specific failure patterns for targeted fixes |

### 1.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TEST SUITE SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │   Test Data      │    │   Evaluation     │    │   Reporting      │       │
│  │   Repository     │───▶│   Engine         │───▶│   & Analysis     │       │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘       │
│          │                       │                       │                   │
│          │                       │                       │                   │
│          ▼                       ▼                       ▼                   │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │ comprehensive_   │    │ unified_         │    │ error_pattern_   │       │
│  │ test_data.py     │    │ evaluation_      │    │ analyzer.py      │       │
│  │                  │    │ runner.py        │    │                  │       │
│  │ • 110+ tests     │    │                  │    │ • Pattern detect │       │
│  │ • 12 categories  │    │ • Async runner   │    │ • Root cause     │       │
│  │ • Multi-turn     │    │ • Metrics calc   │    │ • Fix suggest    │       │
│  └──────────────────┘    │ • HTML reports   │    └──────────────────┘       │
│                          └──────────────────┘                                │
│                                  │                                           │
│                                  ▼                                           │
│                          ┌──────────────────┐                                │
│                          │   Baseline       │                                │
│                          │   Comparison     │                                │
│                          │                  │                                │
│                          │ • Diff metrics   │                                │
│                          │ • Regression CI  │                                │
│                          └──────────────────┘                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Data Flow

```
User Query → UnifiedAgent.parser.parse_query() → ParseResult
                                                      │
                                                      ▼
                                              ┌───────────────┐
                                              │ Compare with  │
                                              │ Expected:     │
                                              │ • Intent      │
                                              │ • Entities    │
                                              │ • Tool        │
                                              └───────────────┘
                                                      │
                                                      ▼
                                              ┌───────────────┐
                                              │ TestResult    │
                                              │ Object        │
                                              └───────────────┘
```

---

## 2. Architecture Components

### 2.1 Component Inventory

| Component | File Path | Responsibility |
|-----------|-----------|----------------|
| Test Data Repository | `tests/evaluation/comprehensive_test_data.py` | 110+ curated test conversations |
| Evaluation Runner | `scripts/unified_evaluation_runner.py` | Execute tests, collect metrics |
| Error Analyzer | `scripts/error_pattern_analyzer.py` | Identify failure patterns |
| CI Runner | `scripts/ci_test.py` | CI/CD-friendly test execution |
| GitHub Workflow | `.github/workflows/chat-agent-evaluation.yml` | Automated CI pipeline |
| Documentation | `docs/TESTING_GUIDE.md` | User-facing documentation |

### 2.2 Test Data Repository Structure

```python
# tests/evaluation/comprehensive_test_data.py

# Category Constants (12 total)
DATA_DISCOVERY_CONVERSATIONS = [...]      # ~15 tests
WORKFLOW_GENERATION_CONVERSATIONS = [...]  # ~10 tests
JOB_MANAGEMENT_CONVERSATIONS = [...]       # ~10 tests
EDUCATION_CONVERSATIONS = [...]            # ~10 tests
ERROR_HANDLING_CONVERSATIONS = [...]       # ~10 tests
CONTEXT_DEPENDENT_CONVERSATIONS = [...]    # ~10 tests
PARAMETER_EXTRACTION_CONVERSATIONS = [...]  # ~10 tests
CROSS_DOMAIN_CONVERSATIONS = [...]         # ~10 tests
MULTI_TURN_CONVERSATIONS = [...]           # 5 full scenarios
COREFERENCE_CONVERSATIONS = [...]          # ~5 tests
AMBIGUOUS_CONVERSATIONS = [...]            # ~5 tests
EDGE_CASE_CONVERSATIONS = [...]            # ~10 tests
```

### 2.3 UnifiedEvaluationRunner Class

```python
class UnifiedEvaluationRunner:
    """Main evaluation orchestrator."""
    
    # Key Methods
    async def run() -> EvaluationReport
    async def run_single_test() -> TestResult
    async def run_multi_turn_test() -> list[TestResult]
    async def run_category() -> list[TestResult]
    
    def calculate_entity_metrics() -> tuple[precision, recall, f1]
    def calculate_category_metrics() -> CategoryMetrics
    def compare_with_baseline() -> tuple[comparison, regressions]
    def generate_html_report() -> Path
    def save_baseline() -> None
```

### 2.4 Data Classes

```python
@dataclass
class TestResult:
    test_id: str
    category: str
    query: str
    expected_intent: str
    actual_intent: str
    intent_correct: bool
    expected_entities: dict
    actual_entities: dict
    entity_precision: float
    entity_recall: float
    entity_f1: float
    expected_tool: Optional[str]
    actual_tool: Optional[str]
    tool_correct: bool
    confidence: float
    parse_time_ms: float
    total_time_ms: float
    llm_invoked: bool
    error: Optional[str]
    context: Optional[dict]

@dataclass
class CategoryMetrics:
    category: str
    total_tests: int
    intent_correct: int
    intent_accuracy: float
    entity_precision_avg: float
    entity_recall_avg: float
    entity_f1_avg: float
    tool_accuracy: float
    avg_confidence: float
    avg_parse_time_ms: float
    avg_total_time_ms: float
    llm_usage_rate: float
    error_count: int
    errors: list

@dataclass
class EvaluationReport:
    timestamp: str
    total_tests: int
    overall_intent_accuracy: float
    overall_entity_f1: float
    overall_tool_accuracy: float
    overall_avg_latency_ms: float
    overall_llm_usage_rate: float
    category_metrics: dict
    test_results: list
    baseline_comparison: Optional[dict]
    regressions: list
```

---

## 3. Test Data Structure

### 3.1 Single-Turn Test Case Schema

```python
{
    "id": "data_discovery_001",           # Unique identifier
    "query": "find RNA-seq data for breast cancer",  # User input
    "expected_intent": "DATA_SEARCH",     # Expected classification
    "expected_entities": {                # Expected extracted entities
        "data_type": "RNA-seq",
        "disease": "breast cancer"
    },
    "expected_tool": "search_data"        # Optional: expected tool routing
}
```

### 3.2 Multi-Turn Conversation Schema

```python
{
    "id": "multi_turn_001",
    "description": "Complete data discovery workflow",
    "turns": [
        {
            "query": "search for ATAC-seq datasets",
            "expected_intent": "DATA_SEARCH",
            "expected_entities": {"data_type": "ATAC-seq"}
        },
        {
            "query": "show me details of the first one",
            "expected_intent": "DATA_DESCRIBE",
            "requires_context": True,              # Flags context dependency
            "expected_entities": {}                # Resolved from context
        },
        {
            "query": "download it to my workspace",
            "expected_intent": "DATA_DOWNLOAD",
            "requires_context": True
        }
    ]
}
```

### 3.3 Category Taxonomy

| Category | Intent Coverage | Complexity | Purpose |
|----------|-----------------|------------|---------|
| `data_discovery` | DATA_SEARCH, DATA_DESCRIBE, DATA_DOWNLOAD, DATA_SCAN | Low-Medium | Core data operations |
| `workflow_generation` | WORKFLOW_GENERATE, WORKFLOW_CONFIGURE | Medium | Pipeline creation |
| `job_management` | JOB_SUBMIT, JOB_STATUS, JOB_CANCEL | Low | Job lifecycle |
| `education` | EXPLAIN, TUTORIAL | Low | Learning & help |
| `error_handling` | TROUBLESHOOT, DEBUG | Medium | Error recovery |
| `context_dependent` | Various | Medium | Requires prior context |
| `parameter_extraction` | Various | High | Complex entity extraction |
| `cross_domain` | Multiple intents | High | Multi-analysis queries |
| `multi_turn` | Sequential intents | High | Full conversations |
| `coreference` | Various | High | Pronoun/reference resolution |
| `ambiguous` | Unclear intent | High | Disambiguation handling |
| `edge_cases` | Various | Medium | Non-standard inputs |

### 3.4 Test Coverage Matrix

```
                          │ Single │ Multi │ Context │ Entity │ Edge │
                          │ Turn   │ Turn  │ Dep.    │ Heavy  │ Case │
──────────────────────────┼────────┼───────┼─────────┼────────┼──────┤
DATA_SEARCH               │   ✓    │   ✓   │    ✓    │   ✓    │  ✓   │
DATA_DESCRIBE             │   ✓    │   ✓   │    ✓    │   ✓    │  ✓   │
DATA_DOWNLOAD             │   ✓    │   ✓   │    ✓    │   ✓    │      │
DATA_SCAN                 │   ✓    │       │         │   ✓    │      │
WORKFLOW_GENERATE         │   ✓    │   ✓   │    ✓    │   ✓    │  ✓   │
WORKFLOW_CONFIGURE        │   ✓    │   ✓   │    ✓    │   ✓    │      │
JOB_SUBMIT                │   ✓    │   ✓   │    ✓    │   ✓    │      │
JOB_STATUS                │   ✓    │   ✓   │    ✓    │        │  ✓   │
JOB_CANCEL                │   ✓    │   ✓   │    ✓    │        │      │
EXPLAIN                   │   ✓    │       │         │   ✓    │      │
TUTORIAL                  │   ✓    │       │         │        │      │
TROUBLESHOOT              │   ✓    │   ✓   │    ✓    │   ✓    │      │
```

---

## 4. Evaluation Metrics

### 4.1 Intent Classification Metrics

#### 4.1.1 Intent Accuracy

$$\text{Intent Accuracy} = \frac{\text{Correct Intent Predictions}}{\text{Total Predictions}}$$

**Normalization Applied:**
- Case normalization: `"data_search"` → `"DATA_SEARCH"`
- Separator normalization: `"data-search"` → `"DATA_SEARCH"`

#### 4.1.2 Per-Category Intent Accuracy

Calculated separately for each of the 12 categories to identify weak areas.

### 4.2 Entity Extraction Metrics

#### 4.2.1 Entity Precision

$$\text{Precision} = \frac{|\text{Extracted} \cap \text{Expected}|}{|\text{Extracted}|}$$

#### 4.2.2 Entity Recall

$$\text{Recall} = \frac{|\text{Extracted} \cap \text{Expected}|}{|\text{Expected}|}$$

#### 4.2.3 Entity F1 Score

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Entity Matching Logic:**
```python
# Entities flattened to (key, value) pairs
# Case-insensitive comparison
expected_set = {("data_type", "rna-seq"), ("disease", "breast cancer")}
actual_set = {("data_type", "rna-seq")}
# Precision: 1/1 = 1.0, Recall: 1/2 = 0.5, F1: 0.67
```

### 4.3 Tool Selection Accuracy

$$\text{Tool Accuracy} = \frac{\text{Correct Tool Selections}}{\text{Total with Expected Tool}}$$

Only calculated when `expected_tool` is specified in test case.

### 4.4 Performance Metrics

| Metric | Description | Threshold |
|--------|-------------|-----------|
| `parse_time_ms` | Time for intent parsing only | Warning if > 1000ms |
| `total_time_ms` | End-to-end processing time | Warning if > 5000ms |
| `llm_usage_rate` | % of queries requiring LLM fallback | Target: < 20% |
| `confidence` | Parser confidence score | Track distribution |

### 4.5 Aggregation Formulas

```python
# Category-level aggregation
category_metrics = {
    "intent_accuracy": sum(r.intent_correct) / len(results),
    "entity_f1_avg": mean([r.entity_f1 for r in results if not r.error]),
    "avg_parse_time_ms": mean([r.parse_time_ms for r in results if not r.error]),
    "llm_usage_rate": sum(r.llm_invoked) / len(results)
}

# Overall aggregation
overall_metrics = {
    "overall_intent_accuracy": sum(all_intent_correct) / total_tests,
    "overall_entity_f1": mean(all_entity_f1_scores),
    "overall_tool_accuracy": sum(all_tool_correct) / tests_with_expected_tool
}
```

---

## 5. Execution Pipeline

### 5.1 Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. INITIALIZATION                                               │
│     ┌──────────────────┐                                        │
│     │ Load Test Data   │ ← comprehensive_test_data.py           │
│     │ Initialize Agent │ ← UnifiedAgent()                       │
│     │ Load Baseline    │ ← baseline.json (if exists)            │
│     └──────────────────┘                                        │
│              │                                                   │
│              ▼                                                   │
│  2. CATEGORY ITERATION                                           │
│     ┌──────────────────┐                                        │
│     │ For each category│                                        │
│     │   For each test  │                                        │
│     │     run_test()   │───┐                                    │
│     └──────────────────┘   │                                    │
│              │              │                                    │
│              │              ▼                                    │
│              │      ┌──────────────────┐                        │
│              │      │ agent.parser     │                        │
│              │      │ .parse_query()   │                        │
│              │      └──────────────────┘                        │
│              │              │                                    │
│              │              ▼                                    │
│              │      ┌──────────────────┐                        │
│              │      │ Compare Results  │                        │
│              │      │ Calculate Metrics│                        │
│              │      └──────────────────┘                        │
│              │              │                                    │
│              ▼              ▼                                    │
│  3. AGGREGATION                                                  │
│     ┌──────────────────┐                                        │
│     │ Aggregate by     │                                        │
│     │ Category         │                                        │
│     │ Calculate Overall│                                        │
│     └──────────────────┘                                        │
│              │                                                   │
│              ▼                                                   │
│  4. COMPARISON                                                   │
│     ┌──────────────────┐                                        │
│     │ Compare Baseline │                                        │
│     │ Detect Regressions│                                       │
│     └──────────────────┘                                        │
│              │                                                   │
│              ▼                                                   │
│  5. REPORTING                                                    │
│     ┌──────────────────┐                                        │
│     │ Generate JSON    │ → evaluation_YYYYMMDD_HHMMSS.json      │
│     │ Generate HTML    │ → report_YYYYMMDD_HHMMSS.html          │
│     │ Print Summary    │ → stdout                                │
│     └──────────────────┘                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Multi-Turn Execution

```python
async def run_multi_turn_test(conversation, category):
    results = []
    context = {}  # Accumulates across turns
    
    for i, turn in enumerate(conversation["turns"]):
        # Pass accumulated context
        result = await run_single_test(turn, category, context)
        results.append(result)
        
        # Update context for next turn
        context = {
            "previous_query": turn["query"],
            "previous_intent": result.actual_intent,
            "previous_entities": result.actual_entities,
            "turn_number": i + 1
        }
        
        # Validate context retention if required
        if turn.get("requires_context") and not result.actual_entities:
            result.error = "Context not retained from previous turn"
    
    return results
```

### 5.3 Async Execution Model

```python
# Single test is async for LLM fallback scenarios
async def run_single_test(test_case, category, context=None):
    start_time = time.time()
    
    # Main parsing call (may invoke LLM)
    result = await agent.parser.parse_query(
        query=test_case["query"],
        context=context
    )
    
    # Calculate metrics
    # ...
    
    return TestResult(...)
```

---

## 6. Baseline & Regression System

### 6.1 Baseline Schema

```json
{
    "timestamp": "2024-12-04T10:30:00",
    "overall_intent_accuracy": 0.85,
    "overall_entity_f1": 0.72,
    "overall_tool_accuracy": 0.90,
    "overall_avg_latency_ms": 450.5,
    "overall_llm_usage_rate": 0.15,
    "category_metrics": {
        "data_discovery": {
            "intent_accuracy": 0.90,
            "entity_f1_avg": 0.75,
            "error_count": 1
        },
        "workflow_generation": {
            "intent_accuracy": 0.80,
            "entity_f1_avg": 0.68,
            "error_count": 2
        }
        // ... other categories
    }
}
```

### 6.2 Regression Detection Thresholds

| Metric | Regression Threshold | Severity |
|--------|---------------------|----------|
| Intent Accuracy | Drop > 2% | High if > 5% |
| Entity F1 | Drop > 5% | High if > 10% |
| Tool Accuracy | Drop > 2% | High if > 5% |
| Latency | Increase > 100ms | High if > 200ms |

### 6.3 Comparison Logic

```python
def compare_with_baseline(current_metrics):
    baseline = load_baseline()
    
    comparison = {}
    regressions = []
    
    for metric in ["overall_intent_accuracy", "overall_entity_f1"]:
        baseline_val = baseline.get(metric, 0)
        current_val = current_metrics.get(metric, 0)
        diff = current_val - baseline_val
        
        comparison[metric] = {
            "baseline": baseline_val,
            "current": current_val,
            "diff": diff,
            "improved": diff > 0
        }
        
        if diff < -THRESHOLD:
            regressions.append({
                "metric": metric,
                "severity": "high" if diff < -0.05 else "medium",
                ...
            })
    
    return comparison, regressions
```

### 6.4 Baseline Lifecycle

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Development  │────▶│  Testing      │────▶│  Production   │
│               │     │               │     │               │
│  No baseline  │     │  Create       │     │  Compare      │
│               │     │  baseline     │     │  against      │
│               │     │               │     │  baseline     │
└───────────────┘     └───────────────┘     └───────────────┘
                              │
                              ▼
                      ┌───────────────┐
                      │  After major  │
                      │  improvements │
                      │  update       │
                      │  baseline     │
                      └───────────────┘
```

---

## 7. Error Analysis Framework

### 7.1 Error Pattern Types

| Pattern Type | Description | Detection Method |
|--------------|-------------|------------------|
| `intent_confusion` | Similar intents confused | Pair frequency analysis |
| `missing_entity` | Expected entities not extracted | Empty actual vs non-empty expected |
| `context_not_retained` | Multi-turn context lost | requires_context + empty entities |
| `edge_case` | Non-standard input failures | Category = edge_cases |
| `exception` | Runtime errors | Non-null error field |
| `llm_fallback_failure` | LLM also failed | llm_invoked=True + wrong intent |
| `tool_selection_error` | Correct intent, wrong tool | intent_correct + !tool_correct |

### 7.2 Pattern Detection Pipeline

```python
def analyze(results):
    failures = extract_failures(results)
    
    patterns = []
    
    # 1. Intent Confusion Detection
    intent_pairs = count_confusion_pairs(failures)
    if most_common_pair.count >= 2:
        patterns.append(IntentConfusionPattern(...))
    
    # 2. Missing Entity Detection
    missing_entity_cases = find_empty_extractions(failures)
    if len(missing_entity_cases) >= 2:
        patterns.append(MissingEntityPattern(...))
    
    # 3. Edge Case Detection
    edge_cases = filter_by_category(failures, "edge_cases")
    edge_types = classify_edge_types(edge_cases)
    
    # 4. Context Failure Detection
    context_failures = filter_by_category(failures, 
        ["multi_turn", "coreference", "context_dependent"])
    
    # 5. Exception Detection
    exception_cases = filter_by_error(failures)
    
    return AnalysisReport(patterns, root_causes, recommendations)
```

### 7.3 Root Cause Mapping

```python
ROOT_CAUSE_MAP = {
    "intent_confusion": 
        "Intent patterns need more distinctive keywords",
    "missing_entity": 
        "Entity extraction regex patterns incomplete",
    "context_not_retained": 
        "Conversation context not maintained between turns",
    "edge_case": 
        "Input normalization not handling non-standard formats",
    "exception": 
        "Error handling incomplete for edge inputs"
}
```

### 7.4 Fix Suggestion Generation

```python
def generate_suggestion(pattern):
    if pattern.type == "intent_confusion":
        return {
            "action": f"Add distinctive patterns for {pair[0]} vs {pair[1]}",
            "location": "agents/intent/parser.py",
            "priority": "high"
        }
    elif pattern.type == "missing_entity":
        missed_type = most_missed_entity_type(pattern)
        return {
            "action": f"Add regex for '{missed_type}' extraction",
            "location": "agents/intent/parser.py",
            "priority": "high"
        }
    # ... other patterns
```

---

## 8. CI/CD Integration

### 8.1 GitHub Actions Workflow

```yaml
# .github/workflows/chat-agent-evaluation.yml

name: Chat Agent Evaluation

on:
  push:
    branches: [main, develop]
    paths:
      - 'src/agents/**'
      - 'config/**'
  pull_request:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      
      - name: Run quick evaluation
        run: python scripts/ci_test.py --quick
      
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: evaluation-results
          path: reports/evaluation/*.json
```

### 8.2 Exit Code Semantics

| Exit Code | Meaning | CI Action |
|-----------|---------|-----------|
| 0 | All tests passed | ✅ Continue |
| 1 | Regressions detected | ⚠️ Investigate |
| 2 | Below minimum thresholds | ❌ Block merge |
| 3 | Execution error | ❌ Fix infrastructure |

### 8.3 CI Thresholds

```python
# scripts/ci_test.py
THRESHOLDS = {
    "intent_accuracy": 0.70,      # Minimum 70%
    "entity_f1": 0.50,            # Minimum 50%
    "tool_accuracy": 0.80,        # Minimum 80%
    "max_latency_ms": 5000,       # Max 5 seconds
    "regression_tolerance": 0.02  # 2% drop allowed
}
```

### 8.4 Artifact Structure

```
reports/evaluation/
├── baseline.json                    # Reference baseline
├── evaluation_20241204_103000.json  # Full results
├── report_20241204_103000.html      # Visual report
└── error_analysis_20241204_103000.md # Pattern analysis
```

---

## 9. Current Limitations

### 9.1 Test Coverage Gaps

| Gap | Impact | Severity |
|-----|--------|----------|
| No real API integration tests | May miss tool execution bugs | Medium |
| Limited error message testing | Error handling untested | Low |
| No performance stress testing | Unknown scaling behavior | Medium |
| Single-language only (English) | No i18n support testing | Low |
| No adversarial testing | Security/robustness unknown | Medium |

### 9.2 Metric Limitations

| Limitation | Impact |
|------------|--------|
| Entity matching is exact-string | Misses semantic equivalence ("RNA-seq" vs "RNAseq") |
| No confidence calibration | High confidence wrong predictions unpenalized |
| Binary intent matching | No partial credit for close intents |
| No conversation coherence score | Multi-turn quality unmeasured |

### 9.3 Infrastructure Gaps

| Gap | Description |
|-----|-------------|
| No historical trending | Can't see improvement over time |
| Manual baseline updates | Should auto-update on releases |
| Local-only execution | No distributed test execution |
| No test prioritization | All tests run every time |

### 9.4 Test Data Limitations

| Limitation | Description |
|------------|-------------|
| Static test set | No dynamic/generated tests |
| Limited entity diversity | Same entities repeated |
| No real user data | Synthetic only |
| English only | No multilingual support |

---

## 10. Improvement Roadmap

### 10.1 Phase 1: Foundation Strengthening (Current → +2 weeks)

**Goal:** Establish reliable baseline and fill critical gaps

| Task | Priority | Effort | Impact |
|------|----------|--------|--------|
| Expand test set to 200+ | High | Medium | +Coverage |
| Add entity normalization | High | Low | +Accuracy |
| Implement confidence tracking | Medium | Low | +Insights |
| Add historical DB | Medium | Medium | +Trending |

**Deliverables:**
- [ ] 200+ test conversations
- [ ] Semantic entity matching
- [ ] SQLite historical storage
- [ ] Confidence distribution reports

### 10.2 Phase 2: Intelligence Enhancement (+2 → +4 weeks)

**Goal:** Add smart test selection and better analysis

| Task | Priority | Effort | Impact |
|------|----------|--------|--------|
| Smart test selection | High | High | +Efficiency |
| Conversation coherence metric | High | Medium | +Quality |
| Auto-fix suggestions | Medium | Medium | +Dev velocity |
| LLM-based test generation | Medium | High | +Coverage |

**Deliverables:**
- [ ] Change-based test selection
- [ ] Coherence scoring for multi-turn
- [ ] Code-aware fix suggestions
- [ ] Synthetic test generator

### 10.3 Phase 3: Scale & Production (+4 → +8 weeks)

**Goal:** Production-grade testing infrastructure

| Task | Priority | Effort | Impact |
|------|----------|--------|--------|
| Parallel test execution | Medium | Medium | +Speed |
| Real user feedback loop | High | High | +Realism |
| Adversarial test suite | Medium | High | +Robustness |
| A/B testing framework | Low | High | +Experimentation |

**Deliverables:**
- [ ] Distributed test runner
- [ ] User feedback collection
- [ ] Red-team test suite
- [ ] Experiment tracking

### 10.4 Complexity Progression Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONVERSATION COMPLEXITY TIERS                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TIER 1: Basic                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ • Single intent, no entities                                           │ │
│  │ • Example: "help me get started"                                       │ │
│  │ • Target: 95% accuracy                                                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  TIER 2: Standard                                                           │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ • Single intent + 1-2 entities                                         │ │
│  │ • Example: "find RNA-seq data for lung cancer"                         │ │
│  │ • Target: 90% intent, 80% entity F1                                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  TIER 3: Complex                                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ • Multiple entities, context references                                │ │
│  │ • Example: "create a workflow for that data with 8 threads"            │ │
│  │ • Target: 85% intent, 70% entity F1                                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  TIER 4: Advanced                                                           │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ • Multi-turn with coreference, disambiguation needed                   │ │
│  │ • Example: 5+ turn workflow building conversation                      │ │
│  │ • Target: 80% intent, 65% entity F1, coherence > 0.7                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  TIER 5: Expert                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ • Cross-domain, error recovery, edge cases combined                    │ │
│  │ • Example: Complex debugging with partial information                  │ │
│  │ • Target: 75% intent, 60% entity F1, recovery success > 80%            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.5 Cyclic Improvement Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS IMPROVEMENT CYCLE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│       ┌────────────┐                                                        │
│       │   MEASURE  │ ◄────────────────────────────────────────┐             │
│       │            │                                          │             │
│       │ Run full   │                                          │             │
│       │ evaluation │                                          │             │
│       └─────┬──────┘                                          │             │
│             │                                                  │             │
│             ▼                                                  │             │
│       ┌────────────┐                                          │             │
│       │  ANALYZE   │                                          │             │
│       │            │                                          │             │
│       │ Error      │                                          │             │
│       │ patterns   │                                          │             │
│       └─────┬──────┘                                          │             │
│             │                                                  │             │
│             ▼                                                  │             │
│       ┌────────────┐                                          │             │
│       │ PRIORITIZE │                                          │             │
│       │            │                                          │             │
│       │ Rank fixes │                                          │             │
│       │ by impact  │                                          │             │
│       └─────┬──────┘                                          │             │
│             │                                                  │             │
│             ▼                                                  │             │
│       ┌────────────┐                                          │             │
│       │ IMPLEMENT  │                                          │             │
│       │            │                                          │             │
│       │ Apply top  │                                          │             │
│       │ 3 fixes    │                                          │             │
│       └─────┬──────┘                                          │             │
│             │                                                  │             │
│             ▼                                                  │             │
│       ┌────────────┐                                          │             │
│       │  VALIDATE  │                                          │             │
│       │            │                                          │             │
│       │ Run quick  │──────────────────────────────────────────┘             │
│       │ tests      │                                                        │
│       └────────────┘                                                        │
│                                                                              │
│  CADENCE: Weekly cycles with monthly baseline updates                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix A: Quick Reference Commands

```bash
# Run quick tests (CI mode)
make test
python scripts/ci_test.py --quick

# Run full evaluation
make evaluate
python scripts/unified_evaluation_runner.py

# Analyze errors
make analyze
python scripts/error_pattern_analyzer.py

# Create baseline
make baseline
python scripts/ci_test.py --create-baseline

# Run specific category
make test-category CATEGORY=data_discovery
python scripts/unified_evaluation_runner.py --category data_discovery
```

## Appendix B: File Locations

```
BioPipelines/
├── tests/
│   └── evaluation/
│       ├── comprehensive_test_data.py    # Test conversations
│       ├── conversation_generator.py     # Synthetic generation
│       ├── experiment_runner.py          # DB-backed runner
│       └── lifecycle_conversations.py    # Complex scenarios
├── scripts/
│   ├── unified_evaluation_runner.py      # Main evaluator
│   ├── error_pattern_analyzer.py         # Failure analysis
│   └── ci_test.py                        # CI runner
├── reports/
│   └── evaluation/
│       ├── baseline.json                 # Reference metrics
│       ├── evaluation_*.json             # Run results
│       ├── report_*.html                 # HTML reports
│       └── error_analysis_*.md           # Analysis reports
├── .github/
│   └── workflows/
│       └── chat-agent-evaluation.yml     # CI workflow
├── docs/
│   ├── TESTING_GUIDE.md                  # User guide
│   └── TEST_SUITE_ARCHITECTURE.md        # This document
└── Makefile                              # Convenience commands
```

## Appendix C: Metric Definitions Summary

| Metric | Formula | Range | Target |
|--------|---------|-------|--------|
| Intent Accuracy | Correct / Total | 0-1 | ≥ 0.85 |
| Entity Precision | TP / Extracted | 0-1 | ≥ 0.80 |
| Entity Recall | TP / Expected | 0-1 | ≥ 0.75 |
| Entity F1 | 2×P×R/(P+R) | 0-1 | ≥ 0.75 |
| Tool Accuracy | Correct / Total | 0-1 | ≥ 0.90 |
| Parse Latency | ms | 0-∞ | ≤ 500ms |
| LLM Usage Rate | LLM calls / Total | 0-1 | ≤ 0.15 |

---

*Document Version: 1.0*
*Last Updated: December 4, 2025*
*Author: BioPipelines Development Team*
