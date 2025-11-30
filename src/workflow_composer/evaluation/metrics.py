"""Metrics aggregation for BioPipelines evaluation.

This module provides utilities for aggregating and analyzing
evaluation metrics across multiple runs.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .evaluator import EvaluationResult, EvaluationRun


@dataclass
class MetricStats:
    """Statistics for a single metric."""
    
    name: str
    count: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    p25: float
    p75: float
    p95: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "count": self.count,
            "mean": round(self.mean, 4),
            "median": round(self.median, 4),
            "std_dev": round(self.std_dev, 4),
            "min": round(self.min_value, 4),
            "max": round(self.max_value, 4),
            "p25": round(self.p25, 4),
            "p75": round(self.p75, 4),
            "p95": round(self.p95, 4),
        }
    
    @classmethod
    def from_values(cls, name: str, values: List[float]) -> "MetricStats":
        """Create from a list of values."""
        if not values:
            return cls(
                name=name,
                count=0,
                mean=0.0,
                median=0.0,
                std_dev=0.0,
                min_value=0.0,
                max_value=0.0,
                p25=0.0,
                p75=0.0,
                p95=0.0,
            )
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return cls(
            name=name,
            count=n,
            mean=statistics.mean(values),
            median=statistics.median(values),
            std_dev=statistics.stdev(values) if n > 1 else 0.0,
            min_value=min(values),
            max_value=max(values),
            p25=sorted_values[int(n * 0.25)] if n > 0 else 0.0,
            p75=sorted_values[int(n * 0.75)] if n > 0 else 0.0,
            p95=sorted_values[int(n * 0.95)] if n > 0 else 0.0,
        )


@dataclass
class CategoryMetrics:
    """Metrics for a specific category."""
    
    category: str
    total_queries: int
    successful_queries: int
    failed_queries: int
    timeout_queries: int
    success_rate: float
    avg_latency_ms: float
    avg_score: float
    score_breakdown: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "timeout_queries": self.timeout_queries,
            "success_rate": round(self.success_rate, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_score": round(self.avg_score, 4),
            "score_breakdown": {k: round(v, 4) for k, v in self.score_breakdown.items()},
        }


@dataclass
class EvaluationMetrics:
    """Comprehensive metrics from an evaluation run.
    
    Attributes:
        run_id: ID of the evaluation run
        total_queries: Total number of queries
        successful: Number of successful evaluations
        failed: Number of failed evaluations
        timeouts: Number of timeouts
        success_rate: Overall success rate
        latency_stats: Statistics for latency
        score_stats: Statistics for overall scores
        dimension_stats: Statistics per score dimension
        category_metrics: Metrics per category
    """
    
    run_id: str
    total_queries: int
    successful: int
    failed: int
    timeouts: int
    success_rate: float
    latency_stats: MetricStats
    score_stats: MetricStats
    dimension_stats: Dict[str, MetricStats] = field(default_factory=dict)
    category_metrics: Dict[str, CategoryMetrics] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "total_queries": self.total_queries,
            "successful": self.successful,
            "failed": self.failed,
            "timeouts": self.timeouts,
            "success_rate": round(self.success_rate, 2),
            "latency_stats": self.latency_stats.to_dict(),
            "score_stats": self.score_stats.to_dict(),
            "dimension_stats": {k: v.to_dict() for k, v in self.dimension_stats.items()},
            "category_metrics": {k: v.to_dict() for k, v in self.category_metrics.items()},
        }
    
    def summary_text(self) -> str:
        """Generate text summary."""
        lines = [
            f"=== Evaluation Metrics: {self.run_id} ===",
            f"Total Queries: {self.total_queries}",
            f"Success Rate: {self.success_rate:.1f}%",
            f"Average Score: {self.score_stats.mean:.3f}",
            f"Average Latency: {self.latency_stats.mean:.1f}ms",
            "",
            "Score Dimensions:",
        ]
        
        for dim, stats in self.dimension_stats.items():
            lines.append(f"  {dim}: {stats.mean:.3f} (±{stats.std_dev:.3f})")
        
        if self.category_metrics:
            lines.append("")
            lines.append("By Category:")
            for cat, metrics in self.category_metrics.items():
                lines.append(f"  {cat}:")
                lines.append(f"    Success: {metrics.success_rate:.1f}%")
                lines.append(f"    Avg Score: {metrics.avg_score:.3f}")
        
        return "\n".join(lines)


class MetricAggregator:
    """Aggregates metrics from evaluation runs.
    
    Example:
        >>> aggregator = MetricAggregator()
        >>> metrics = aggregator.aggregate(evaluation_run)
        >>> print(metrics.summary_text())
    """
    
    def aggregate(self, run: "EvaluationRun") -> EvaluationMetrics:
        """Aggregate metrics from an evaluation run.
        
        Args:
            run: The evaluation run to analyze
            
        Returns:
            Aggregated EvaluationMetrics
        """
        from .evaluator import EvaluationStatus
        
        results = run.results
        
        # Count statuses
        successful = sum(1 for r in results if r.status == EvaluationStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == EvaluationStatus.FAILED)
        timeouts = sum(1 for r in results if r.status == EvaluationStatus.TIMEOUT)
        total = len(results)
        
        # Collect latencies and scores
        latencies = [r.latency_ms for r in results if r.status == EvaluationStatus.COMPLETED]
        scores = [r.overall_score for r in results if r.scores]
        
        # Collect dimension scores
        dimension_values: Dict[str, List[float]] = {}
        for result in results:
            for dim, value in result.scores.items():
                if dim not in dimension_values:
                    dimension_values[dim] = []
                dimension_values[dim].append(value)
        
        dimension_stats = {
            dim: MetricStats.from_values(dim, values)
            for dim, values in dimension_values.items()
        }
        
        # Category metrics
        category_metrics = self._compute_category_metrics(results)
        
        return EvaluationMetrics(
            run_id=run.run_id,
            total_queries=total,
            successful=successful,
            failed=failed,
            timeouts=timeouts,
            success_rate=(successful / total * 100) if total > 0 else 0,
            latency_stats=MetricStats.from_values("latency_ms", latencies),
            score_stats=MetricStats.from_values("overall_score", scores),
            dimension_stats=dimension_stats,
            category_metrics=category_metrics,
        )
    
    def _compute_category_metrics(
        self,
        results: List["EvaluationResult"],
    ) -> Dict[str, CategoryMetrics]:
        """Compute metrics per category."""
        from .evaluator import EvaluationStatus
        
        # Group by category
        by_category: Dict[str, List["EvaluationResult"]] = {}
        for result in results:
            category = result.metadata.get("category", "unknown")
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(result)
        
        # Compute metrics per category
        metrics = {}
        for category, cat_results in by_category.items():
            successful = [r for r in cat_results if r.status == EvaluationStatus.COMPLETED]
            failed = sum(1 for r in cat_results if r.status == EvaluationStatus.FAILED)
            timeouts = sum(1 for r in cat_results if r.status == EvaluationStatus.TIMEOUT)
            
            latencies = [r.latency_ms for r in successful]
            scores = [r.overall_score for r in cat_results if r.scores]
            
            # Score breakdown per dimension
            score_breakdown: Dict[str, List[float]] = {}
            for result in cat_results:
                for dim, value in result.scores.items():
                    if dim not in score_breakdown:
                        score_breakdown[dim] = []
                    score_breakdown[dim].append(value)
            
            avg_breakdown = {
                dim: statistics.mean(values) if values else 0.0
                for dim, values in score_breakdown.items()
            }
            
            metrics[category] = CategoryMetrics(
                category=category,
                total_queries=len(cat_results),
                successful_queries=len(successful),
                failed_queries=failed,
                timeout_queries=timeouts,
                success_rate=(len(successful) / len(cat_results) * 100) if cat_results else 0,
                avg_latency_ms=statistics.mean(latencies) if latencies else 0,
                avg_score=statistics.mean(scores) if scores else 0,
                score_breakdown=avg_breakdown,
            )
        
        return metrics
    
    def compare_runs(
        self,
        runs: List["EvaluationRun"],
    ) -> Dict[str, Any]:
        """Compare metrics across multiple runs.
        
        Args:
            runs: List of evaluation runs to compare
            
        Returns:
            Comparison data
        """
        metrics_list = [self.aggregate(run) for run in runs]
        
        comparison = {
            "runs": [m.run_id for m in metrics_list],
            "success_rates": [m.success_rate for m in metrics_list],
            "avg_scores": [m.score_stats.mean for m in metrics_list],
            "avg_latencies": [m.latency_stats.mean for m in metrics_list],
        }
        
        # Best/worst per metric
        if metrics_list:
            comparison["best_success_rate"] = max(m.success_rate for m in metrics_list)
            comparison["best_score"] = max(m.score_stats.mean for m in metrics_list)
            comparison["best_latency"] = min(m.latency_stats.mean for m in metrics_list)
        
        return comparison


__all__ = [
    "MetricStats",
    "CategoryMetrics",
    "EvaluationMetrics",
    "MetricAggregator",
]
