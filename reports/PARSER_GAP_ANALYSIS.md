# Parser Gap Analysis Report

**Generated:** 2025-12-04
**Evaluation Run:** eval_20251204_165923_4274fd
**Test Results:** 63/76 passed (82.9%)

## Executive Summary

The UnifiedIntentParser demonstrates strong performance on core bioinformatics tasks (workflow generation, data discovery, job management) but has systematic gaps in:

1. **Education vs Action disambiguation** - Struggles to distinguish "how to" questions from actual action requests
2. **Intent granularity** - Conflates similar intents (WATCH vs STATUS, RESUBMIT vs SUBMIT)
3. **Coreference context** - Loses context in multi-turn conversations
4. **Edge cases** - Single-word queries and ambiguous inputs

---

## Detailed Failure Analysis

### Category 1: Education vs Action Confusion (4 failures)

| Query | Expected | Actual | Issue |
|-------|----------|--------|-------|
| "How do I create a workflow?" | EDUCATION_HELP | WORKFLOW_CREATE | Question words not detected |
| "What are the best practices for differential expression?" | EDUCATION_EXPLAIN | WORKFLOW_CREATE | Explanation request misread |
| "RNA-seq" | WORKFLOW_CREATE | EDUCATION_EXPLAIN | Single word ambiguous |
| "I need help with RNA-seq analysis" | WORKFLOW_CREATE | EDUCATION_EXPLAIN | "help with" triggers education |

**Root Cause:** The parser doesn't distinguish between:
- "How do I X?" (education request)
- "Do X for me" (action request)
- "Help me with X" (ambiguous - context dependent)

**Fix Priority:** HIGH - Affects user experience significantly

**Recommended Fix:**
```python
# Add pattern matching for question words
QUESTION_PATTERNS = [
    r'^how\s+(do|can|should)',
    r'^what\s+(is|are|should)',
    r'^why\s+(do|does|should)',
    r'^can\s+you\s+explain',
    r'^tell\s+me\s+about',
]
# These should bias toward EDUCATION_* intents
```

---

### Category 2: Intent Granularity Issues (4 failures)

| Query | Expected | Actual | Issue |
|-------|----------|--------|-------|
| "Watch job 54321 and notify me" | JOB_WATCH | JOB_STATUS | WATCH is status + continuous |
| "Resubmit it with more memory" | JOB_RESUBMIT | JOB_SUBMIT | RESUBMIT vs SUBMIT |
| "Fix it and resubmit" | JOB_RESUBMIT | JOB_SUBMIT | Resubmit pattern not matched |
| "Add a quality control step" | WORKFLOW_CREATE | WORKFLOW_MODIFY | Context says modify existing |

**Root Cause:** Fine-grained intents (JOB_WATCH, JOB_RESUBMIT) are being mapped to coarser intents.

**Fix Priority:** MEDIUM - Affects power users

**Recommended Fix:**
```python
# Add specific keyword patterns for fine-grained intents
INTENT_KEYWORDS = {
    'JOB_WATCH': ['watch', 'monitor', 'notify when', 'alert when'],
    'JOB_RESUBMIT': ['resubmit', 're-submit', 'retry', 'run again'],
    'WORKFLOW_MODIFY': ['add step', 'insert', 'modify', 'change step'],
}
```

---

### Category 3: Reference/Context Issues (3 failures)

| Query | Expected | Actual | Issue |
|-------|----------|--------|-------|
| "Download the reference genome for me" | REFERENCE_DOWNLOAD | DATA_DOWNLOAD | "reference" not detected |
| "How many samples are there?" | DATA_DESCRIBE | DATA_SCAN | DESCRIBE vs SCAN |
| "Filter these results to only brain tissue" | DATA_SEARCH | META_UNKNOWN | Context lost |

**Root Cause:** 
- "reference genome" should trigger REFERENCE_* intent
- "How many" is descriptive, not scanning
- "these results" requires coreference resolution

**Fix Priority:** MEDIUM

**Recommended Fix:**
```python
# Entity-based intent boosting
if 'reference' in query.lower() and any(x in query.lower() for x in ['genome', 'annotation', 'index']):
    boost_intent('REFERENCE_*')
    
# Descriptive patterns
DESCRIBE_PATTERNS = ['how many', 'count of', 'number of', 'statistics']
```

---

### Category 4: Edge Cases (2 failures)

| Query | Expected | Actual | Issue |
|-------|----------|--------|-------|
| "Thanks, that's all I need!" | META_FAREWELL | META_CONFIRM | Gratitude = farewell |
| "Help me with RNA-seq" | EDUCATION_HELP | EDUCATION_EXPLAIN | Help vs Explain |

**Root Cause:** Ambiguous social/conversational inputs

**Fix Priority:** LOW - Doesn't affect core functionality

---

## Entity Extraction Issues

Several failures show entity extraction problems:

| Issue | Example | Expected | Actual |
|-------|---------|----------|--------|
| ORGANISM over-extraction | "RNA-seq" | - | `ORGANISM: RN` |
| Wrong entity type | "reference genome" | - | `reference: for` |
| Missing entities | "It failed" | Needs coreference | None |

**Root Cause:** Entity extraction is too aggressive with short strings

---

## Metrics Summary

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Intent Accuracy | 82.9% | 90% | ⚠️ BELOW TARGET |
| Entity F1 | 55.1% | 80% | ❌ NEEDS WORK |
| Tool Accuracy | 50.0% | 75% | ❌ NEEDS WORK |
| Semantic Similarity | 27.5% | 70% | ❌ NEEDS WORK |

---

## Recommended Improvements

### Immediate (Week 1)
1. Add question word detection for education intents
2. Add keyword patterns for JOB_WATCH, JOB_RESUBMIT
3. Fix "reference genome" entity extraction

### Short-term (Week 2-3)
1. Improve coreference resolution for multi-turn conversations
2. Add context-aware intent disambiguation
3. Reduce entity over-extraction for short queries

### Long-term (Month 1)
1. Train fine-tuned classifier on labeled data
2. Add confidence thresholds for uncertain classifications
3. Implement clarification prompts for ambiguous inputs

---

## Test Categories Performance

| Category | Pass Rate | Status |
|----------|-----------|--------|
| workflow_generation | 93% | ✅ Good |
| data_discovery | 92% | ✅ Good |
| coreference | 92% | ✅ Good |
| job_management | 83% | ⚠️ Needs Work |
| ambiguous | 75% | ⚠️ Needs Work |
| edge_cases | 67% | ⚠️ Needs Work |
| error_handling | 67% | ⚠️ Needs Work |
| education | 60% | ❌ Priority Fix |

---

## Files to Modify

1. `src/workflow_composer/agents/intent/unified_parser.py` - Add pattern matching
2. `src/workflow_composer/agents/intent/llm_parser.py` - Improve prompts
3. `config/nlu/intent_patterns.yaml` - Add missing patterns
4. `tests/evaluation/comprehensive_test_data.py` - Add regression tests

---

## Next Steps

1. [ ] Create regression tests for all 13 failures
2. [ ] Implement question word detection
3. [ ] Add JOB_WATCH and JOB_RESUBMIT patterns
4. [ ] Fix reference genome detection
5. [ ] Re-run evaluation to validate fixes
