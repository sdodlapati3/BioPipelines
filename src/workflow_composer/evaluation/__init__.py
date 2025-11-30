"""Evaluation Framework for BioPipelines.

This package provides automated evaluation and benchmarking for the
BioPipelines agentic system, including:
    - Benchmark definitions for different task categories
    - Automated evaluator for running benchmarks
    - LLM-as-judge scoring for subjective quality
    - Metrics collection and reporting

Example:
    >>> from workflow_composer.evaluation import Evaluator, load_benchmarks
    >>> benchmarks = load_benchmarks()
    >>> evaluator = Evaluator()
    >>> results = await evaluator.run_benchmarks(benchmarks)
    >>> print(results.summary())
"""

from .benchmarks import (
    Benchmark,
    BenchmarkCategory,
    BenchmarkQuery,
    load_benchmarks,
    DISCOVERY_BENCHMARKS,
    WORKFLOW_BENCHMARKS,
    EDUCATION_BENCHMARKS,
)
from .evaluator import (
    Evaluator,
    EvaluatorConfig,
    EvaluationResult,
    EvaluationRun,
)
from .scorer import (
    Scorer,
    ScorerConfig,
    LLMJudgeScorer,
    RuleBasedScorer,
    Score,
)
from .metrics import (
    EvaluationMetrics,
    MetricAggregator,
)
from .report import (
    ReportGenerator,
    HTMLReportGenerator,
    JSONReportGenerator,
)

__all__ = [
    # Benchmarks
    "Benchmark",
    "BenchmarkCategory",
    "BenchmarkQuery",
    "load_benchmarks",
    "DISCOVERY_BENCHMARKS",
    "WORKFLOW_BENCHMARKS",
    "EDUCATION_BENCHMARKS",
    # Evaluator
    "Evaluator",
    "EvaluatorConfig",
    "EvaluationResult",
    "EvaluationRun",
    # Scorer
    "Scorer",
    "ScorerConfig",
    "LLMJudgeScorer",
    "RuleBasedScorer",
    "Score",
    # Metrics
    "EvaluationMetrics",
    "MetricAggregator",
    # Report
    "ReportGenerator",
    "HTMLReportGenerator",
    "JSONReportGenerator",
]
