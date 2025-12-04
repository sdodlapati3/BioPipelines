# Chat Agent Testing & Evaluation Guide

This document describes the comprehensive testing infrastructure for the BioPipelines Chat Agent system.

## Overview

The testing system provides:
- **110+ test conversations** across 12 categories
- **Unified evaluation runner** with metrics collection
- **Baseline comparison** for regression detection
- **Error pattern analysis** with fix suggestions
- **CI/CD integration** with GitHub Actions
- **HTML reports** with visualizations

## Quick Start

### Run Quick Tests
```bash
# Quick evaluation (core categories only, ~30 tests)
python scripts/ci_test.py --quick

# Full evaluation (~110+ tests)
python scripts/ci_test.py
```

### View Results
```bash
# Check the reports directory
ls reports/evaluation/

# Open HTML report (if generated)
open reports/evaluation/report_*.html
```

### Analyze Errors
```bash
# Run error pattern analysis
python scripts/error_pattern_analyzer.py

# View the markdown report
cat reports/evaluation/error_analysis_*.md
```

## Test Data Structure

Test conversations are defined in `tests/evaluation/comprehensive_test_data.py`:

### Categories

| Category | Description | Count |
|----------|-------------|-------|
| `data_discovery` | Search, download, describe, scan data | ~15 |
| `workflow_generation` | Create and configure workflows | ~10 |
| `job_management` | Submit, monitor, cancel jobs | ~10 |
| `education` | Explain concepts and tutorials | ~10 |
| `error_handling` | Troubleshoot and debug issues | ~10 |
| `context_dependent` | Previous context required | ~10 |
| `parameter_extraction` | Complex parameter parsing | ~10 |
| `cross_domain` | Multi-analysis type queries | ~10 |
| `multi_turn` | Full conversation sequences | 5 |
| `coreference` | Pronoun/reference resolution | ~5 |
| `ambiguous` | Unclear intent queries | ~5 |
| `edge_cases` | Typos, caps, special chars | ~10 |

### Test Case Format

```python
{
    "id": "data_discovery_001",
    "query": "find RNA-seq data for breast cancer",
    "expected_intent": "DATA_SEARCH",
    "expected_entities": {
        "data_type": "RNA-seq",
        "disease": "breast cancer"
    },
    "expected_tool": "search_data"  # optional
}
```

### Multi-Turn Format

```python
{
    "id": "multi_turn_001",
    "description": "Data discovery workflow",
    "turns": [
        {
            "query": "search for ATAC-seq datasets",
            "expected_intent": "DATA_SEARCH",
            "expected_entities": {"data_type": "ATAC-seq"}
        },
        {
            "query": "show me details of the first one",
            "expected_intent": "DATA_DESCRIBE",
            "requires_context": True  # flags context dependency
        }
    ]
}
```

## Evaluation Metrics

### Intent Accuracy
Percentage of queries where predicted intent matches expected intent.

### Entity F1 Score
Harmonic mean of precision and recall for entity extraction:
- **Precision**: Correct entities / All extracted entities
- **Recall**: Correct entities / All expected entities

### Tool Accuracy
Percentage of queries routed to the correct tool.

### Latency
Average time for intent parsing (in milliseconds).

### LLM Usage Rate
Percentage of queries that required LLM fallback (lower is better for efficiency).

## Scripts Reference

### `scripts/unified_evaluation_runner.py`

Main evaluation script.

```bash
# Run all tests with HTML report
python scripts/unified_evaluation_runner.py

# Run specific category
python scripts/unified_evaluation_runner.py --category data_discovery

# Save current results as baseline
python scripts/unified_evaluation_runner.py --save-baseline

# Skip baseline comparison
python scripts/unified_evaluation_runner.py --no-compare
```

### `scripts/error_pattern_analyzer.py`

Analyzes failures to identify patterns.

```bash
# Analyze latest results
python scripts/error_pattern_analyzer.py

# Analyze specific results file
python scripts/error_pattern_analyzer.py --results reports/evaluation/evaluation_20241215.json

# Output to specific file
python scripts/error_pattern_analyzer.py --output my_analysis.md
```

### `scripts/ci_test.py`

CI-friendly test runner with exit codes.

```bash
# Quick tests for PR checks
python scripts/ci_test.py --quick

# Full tests with strict mode
python scripts/ci_test.py --strict

# Create new baseline
python scripts/ci_test.py --create-baseline

# Custom thresholds
python scripts/ci_test.py --thresholds '{"intent_accuracy": 0.80}'
```

**Exit Codes:**
- `0` - All tests passed
- `1` - Regressions detected
- `2` - Below minimum thresholds
- `3` - Execution error

## Baseline Comparison

The system maintains a baseline (`reports/evaluation/baseline.json`) for regression detection.

### Setting a Baseline

```bash
# After verifying tests pass
python scripts/ci_test.py --create-baseline
```

### Regression Detection

When running evaluations, the system automatically:
1. Loads the baseline (if exists)
2. Compares current metrics against baseline
3. Flags regressions exceeding thresholds:
   - Intent accuracy drop > 2%
   - Entity F1 drop > 5%
   - Latency increase > 100ms

## HTML Reports

Generated reports include:
- Summary metrics cards
- Category breakdown table
- Baseline comparison (if available)
- Regression warnings
- Failed test details

Reports are saved to `reports/evaluation/report_*.html`.

## CI/CD Integration

### GitHub Actions

The workflow (`.github/workflows/chat-agent-evaluation.yml`) runs:
- On pushes to `main` or `develop` affecting agent code
- On pull requests to `main`
- Manually via workflow dispatch

### Manual Trigger Options

From GitHub Actions UI:
- **full_evaluation**: Run all tests instead of quick mode
- **create_baseline**: Save this run as new baseline

### Artifacts

Each run uploads:
- JSON results files
- HTML reports
- Error analysis markdown

Retention: 30 days (90 days for baselines)

## Adding New Tests

### 1. Add Test Case

Edit `tests/evaluation/comprehensive_test_data.py`:

```python
# In the appropriate category list
{
    "id": "data_discovery_new",
    "query": "your test query here",
    "expected_intent": "DATA_SEARCH",
    "expected_entities": {"key": "value"}
}
```

### 2. For New Categories

Add a new list and register it:

```python
NEW_CATEGORY_CONVERSATIONS = [
    {"query": "...", "expected_intent": "...", ...}
]

# Update ALL_CONVERSATIONS at the bottom
ALL_CONVERSATIONS = {
    ...
    "new_category": NEW_CATEGORY_CONVERSATIONS,
}
```

### 3. Run Tests

```bash
# Verify new tests work
python scripts/unified_evaluation_runner.py --category new_category
```

## Troubleshooting

### "No test data found"

Ensure `tests/evaluation/comprehensive_test_data.py` exists and is importable.

### "Agent not initialized"

Check that all agent dependencies are installed:
```bash
pip install -r requirements-composer.txt
```

### High LLM Usage Rate

If LLM fallback is triggered too often:
1. Run error analysis: `python scripts/error_pattern_analyzer.py`
2. Check which patterns are missing
3. Add regex patterns to `agents/intent/parser.py`

### Regression Detected

1. Download artifacts from CI run
2. Compare with baseline in HTML report
3. Check "Failed Tests" section for specific failures
4. Run error analyzer for fix suggestions

## Architecture

```
tests/evaluation/
├── comprehensive_test_data.py   # 110+ test conversations
├── conversation_generator.py     # Synthetic data generation
├── experiment_runner.py          # Database-backed runner
├── conversation_test_suite.py    # Evaluation metrics
└── lifecycle_conversations.py    # Complex scenarios

scripts/
├── unified_evaluation_runner.py  # Main evaluation script
├── error_pattern_analyzer.py     # Failure analysis
└── ci_test.py                    # CI-friendly runner

reports/evaluation/
├── baseline.json                 # Baseline metrics
├── evaluation_*.json             # Run results
├── report_*.html                 # HTML reports
└── error_analysis_*.md           # Error analysis

.github/workflows/
└── chat-agent-evaluation.yml     # CI workflow
```

## Metrics Thresholds

Default CI thresholds (in `scripts/ci_test.py`):

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Intent Accuracy | ≥ 70% | Minimum pass rate |
| Entity F1 | ≥ 50% | Minimum extraction quality |
| Tool Accuracy | ≥ 80% | Correct tool routing |
| Avg Latency | ≤ 5000ms | Maximum average parse time |
| Regression Tolerance | 2% | Drop before flagging |

## Best Practices

1. **Run tests before PRs**: `python scripts/ci_test.py --quick`
2. **Review HTML reports**: Check category breakdown for weak areas
3. **Use error analyzer**: Prioritize fixes based on recommendations
4. **Update baseline carefully**: Only after validating improvements
5. **Add edge cases**: Expand test coverage for discovered failures
