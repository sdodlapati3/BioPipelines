#!/usr/bin/env python3
"""
Unified Evaluation Runner for BioPipelines Chat Agent

This script provides comprehensive testing of the chat agent system including:
- Single-turn intent classification accuracy
- Multi-turn conversation handling with context retention
- Entity extraction accuracy (F1 score)
- Tool selection accuracy
- Latency tracking and regression detection
- Baseline comparison for CI/CD integration

Usage:
    python scripts/unified_evaluation_runner.py                    # Run all tests
    python scripts/unified_evaluation_runner.py --category data    # Run specific category
    python scripts/unified_evaluation_runner.py --save-baseline    # Save current run as baseline
    python scripts/unified_evaluation_runner.py --compare-baseline # Compare against saved baseline
    python scripts/unified_evaluation_runner.py --report           # Generate HTML report
"""

import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test case."""
    test_id: str
    category: str
    query: str
    expected_intent: str
    actual_intent: str
    intent_correct: bool
    expected_entities: dict = field(default_factory=dict)
    actual_entities: dict = field(default_factory=dict)
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    entity_f1: float = 0.0
    expected_tool: Optional[str] = None
    actual_tool: Optional[str] = None
    tool_correct: bool = True
    confidence: float = 0.0
    parse_time_ms: float = 0.0
    total_time_ms: float = 0.0
    llm_invoked: bool = False
    error: Optional[str] = None
    context: Optional[dict] = None


@dataclass
class CategoryMetrics:
    """Aggregated metrics for a test category."""
    category: str
    total_tests: int = 0
    intent_correct: int = 0
    intent_accuracy: float = 0.0
    entity_precision_avg: float = 0.0
    entity_recall_avg: float = 0.0
    entity_f1_avg: float = 0.0
    tool_accuracy: float = 0.0
    avg_confidence: float = 0.0
    avg_parse_time_ms: float = 0.0
    avg_total_time_ms: float = 0.0
    llm_usage_rate: float = 0.0
    error_count: int = 0
    errors: list = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    total_tests: int = 0
    overall_intent_accuracy: float = 0.0
    overall_entity_f1: float = 0.0
    overall_tool_accuracy: float = 0.0
    overall_avg_latency_ms: float = 0.0
    overall_llm_usage_rate: float = 0.0
    category_metrics: dict = field(default_factory=dict)
    test_results: list = field(default_factory=list)
    baseline_comparison: Optional[dict] = None
    regressions: list = field(default_factory=list)


class UnifiedEvaluationRunner:
    """Runs comprehensive evaluation of the chat agent system."""
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        baseline_path: Optional[Path] = None
    ):
        self.output_dir = output_dir or project_root / "reports" / "evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.baseline_path = baseline_path or self.output_dir / "baseline.json"
        self.results_path = self.output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        self.test_results: list[TestResult] = []
        self.agent = None
        
    async def initialize_agent(self):
        """Initialize the UnifiedAgent for testing."""
        try:
            from agents.unified_agent import UnifiedAgent
            self.agent = UnifiedAgent()
            logger.info("UnifiedAgent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    def load_test_data(self) -> dict[str, list]:
        """Load test data from comprehensive_test_data.py."""
        try:
            from tests.evaluation.comprehensive_test_data import (
                DATA_DISCOVERY_CONVERSATIONS,
                WORKFLOW_GENERATION_CONVERSATIONS,
                JOB_MANAGEMENT_CONVERSATIONS,
                EDUCATION_CONVERSATIONS,
                ERROR_HANDLING_CONVERSATIONS,
                CONTEXT_DEPENDENT_CONVERSATIONS,
                PARAMETER_EXTRACTION_CONVERSATIONS,
                CROSS_DOMAIN_CONVERSATIONS,
                MULTI_TURN_CONVERSATIONS,
                COREFERENCE_CONVERSATIONS,
                AMBIGUOUS_CONVERSATIONS,
                EDGE_CASE_CONVERSATIONS,
            )
            
            return {
                "data_discovery": DATA_DISCOVERY_CONVERSATIONS,
                "workflow_generation": WORKFLOW_GENERATION_CONVERSATIONS,
                "job_management": JOB_MANAGEMENT_CONVERSATIONS,
                "education": EDUCATION_CONVERSATIONS,
                "error_handling": ERROR_HANDLING_CONVERSATIONS,
                "context_dependent": CONTEXT_DEPENDENT_CONVERSATIONS,
                "parameter_extraction": PARAMETER_EXTRACTION_CONVERSATIONS,
                "cross_domain": CROSS_DOMAIN_CONVERSATIONS,
                "multi_turn": MULTI_TURN_CONVERSATIONS,
                "coreference": COREFERENCE_CONVERSATIONS,
                "ambiguous": AMBIGUOUS_CONVERSATIONS,
                "edge_cases": EDGE_CASE_CONVERSATIONS,
            }
        except ImportError as e:
            logger.error(f"Failed to import test data: {e}")
            # Fall back to loading from JSON if available
            return self._load_from_json_fallback()
    
    def _load_from_json_fallback(self) -> dict[str, list]:
        """Load test data from JSON files as fallback."""
        test_data = {}
        json_path = project_root / "training_data" / "multi_turn_test_dataset.json"
        
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
                for category, conversations in data.items():
                    test_data[category] = conversations
        
        return test_data
    
    def calculate_entity_metrics(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any]
    ) -> tuple[float, float, float]:
        """Calculate precision, recall, and F1 for entity extraction."""
        if not expected and not actual:
            return 1.0, 1.0, 1.0
        if not expected:
            return 0.0, 1.0, 0.0 if actual else 1.0
        if not actual:
            return 0.0, 0.0, 0.0
        
        expected_set = set()
        actual_set = set()
        
        # Flatten entities to (key, value) pairs for comparison
        for key, value in expected.items():
            if isinstance(value, list):
                for v in value:
                    expected_set.add((key.lower(), str(v).lower()))
            else:
                expected_set.add((key.lower(), str(value).lower()))
        
        for key, value in actual.items():
            if isinstance(value, list):
                for v in value:
                    actual_set.add((key.lower(), str(v).lower()))
            else:
                actual_set.add((key.lower(), str(value).lower()))
        
        if not expected_set:
            return 1.0, 1.0, 1.0
        
        true_positives = len(expected_set & actual_set)
        
        precision = true_positives / len(actual_set) if actual_set else 0.0
        recall = true_positives / len(expected_set) if expected_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    async def run_single_test(
        self,
        test_case: dict,
        category: str,
        context: Optional[dict] = None
    ) -> TestResult:
        """Run a single test case and return the result."""
        query = test_case.get("query", "")
        expected_intent = test_case.get("expected_intent", "")
        expected_entities = test_case.get("expected_entities", {})
        expected_tool = test_case.get("expected_tool")
        test_id = test_case.get("id", f"{category}_{hash(query) % 10000}")
        
        start_time = time.time()
        
        try:
            # Run through the agent's parser
            if self.agent:
                parse_start = time.time()
                result = await self.agent.parser.parse_query(query, context=context)
                parse_time = (time.time() - parse_start) * 1000
                
                actual_intent = result.intent if hasattr(result, 'intent') else str(result.get('intent', ''))
                actual_entities = result.entities if hasattr(result, 'entities') else result.get('entities', {})
                confidence = result.confidence if hasattr(result, 'confidence') else result.get('confidence', 0.0)
                llm_invoked = result.llm_invoked if hasattr(result, 'llm_invoked') else result.get('llm_invoked', False)
                
                # Normalize intent names for comparison
                actual_intent_normalized = actual_intent.upper().replace("-", "_").replace(" ", "_")
                expected_intent_normalized = expected_intent.upper().replace("-", "_").replace(" ", "_")
                
                intent_correct = actual_intent_normalized == expected_intent_normalized
                
                # Calculate entity metrics
                precision, recall, f1 = self.calculate_entity_metrics(expected_entities, actual_entities)
                
                # Check tool selection if applicable
                actual_tool = result.tool if hasattr(result, 'tool') else result.get('tool')
                tool_correct = expected_tool is None or actual_tool == expected_tool
                
                total_time = (time.time() - start_time) * 1000
                
                return TestResult(
                    test_id=test_id,
                    category=category,
                    query=query,
                    expected_intent=expected_intent,
                    actual_intent=actual_intent,
                    intent_correct=intent_correct,
                    expected_entities=expected_entities,
                    actual_entities=actual_entities,
                    entity_precision=precision,
                    entity_recall=recall,
                    entity_f1=f1,
                    expected_tool=expected_tool,
                    actual_tool=actual_tool,
                    tool_correct=tool_correct,
                    confidence=confidence,
                    parse_time_ms=parse_time,
                    total_time_ms=total_time,
                    llm_invoked=llm_invoked,
                    context=context
                )
            else:
                # Agent not initialized - return failure result
                return TestResult(
                    test_id=test_id,
                    category=category,
                    query=query,
                    expected_intent=expected_intent,
                    actual_intent="",
                    intent_correct=False,
                    error="Agent not initialized"
                )
                
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            logger.error(f"Test failed for query '{query[:50]}...': {e}")
            
            return TestResult(
                test_id=test_id,
                category=category,
                query=query,
                expected_intent=expected_intent,
                actual_intent="",
                intent_correct=False,
                total_time_ms=total_time,
                error=str(e)
            )
    
    async def run_multi_turn_test(self, conversation: list[dict], category: str) -> list[TestResult]:
        """Run a multi-turn conversation test."""
        results = []
        context = {}
        
        for i, turn in enumerate(conversation):
            turn["id"] = f"{category}_turn_{i}"
            result = await self.run_single_test(turn, category, context)
            results.append(result)
            
            # Update context with previous turn info for context retention testing
            context = {
                "previous_query": turn.get("query"),
                "previous_intent": result.actual_intent,
                "previous_entities": result.actual_entities,
                "turn_number": i + 1
            }
            
            # Check context retention if expected
            if turn.get("requires_context"):
                # Verify that context was properly used
                if not result.actual_entities and turn.get("expected_entities"):
                    result.error = f"Context not retained from previous turn"
        
        return results
    
    async def run_category(self, category: str, test_cases: list) -> list[TestResult]:
        """Run all tests in a category."""
        logger.info(f"Running {len(test_cases)} tests in category: {category}")
        results = []
        
        for test_case in test_cases:
            # Check if this is a multi-turn conversation
            if isinstance(test_case, dict) and "turns" in test_case:
                turn_results = await self.run_multi_turn_test(test_case["turns"], category)
                results.extend(turn_results)
            else:
                result = await self.run_single_test(test_case, category)
                results.append(result)
        
        return results
    
    def calculate_category_metrics(self, category: str, results: list[TestResult]) -> CategoryMetrics:
        """Calculate aggregated metrics for a category."""
        if not results:
            return CategoryMetrics(category=category)
        
        total = len(results)
        intent_correct = sum(1 for r in results if r.intent_correct)
        tool_correct = sum(1 for r in results if r.tool_correct)
        error_count = sum(1 for r in results if r.error)
        llm_invoked = sum(1 for r in results if r.llm_invoked)
        
        entity_precisions = [r.entity_precision for r in results if not r.error]
        entity_recalls = [r.entity_recall for r in results if not r.error]
        entity_f1s = [r.entity_f1 for r in results if not r.error]
        confidences = [r.confidence for r in results if not r.error]
        parse_times = [r.parse_time_ms for r in results if not r.error]
        total_times = [r.total_time_ms for r in results if not r.error]
        
        errors = [{"test_id": r.test_id, "query": r.query, "error": r.error} 
                  for r in results if r.error]
        
        return CategoryMetrics(
            category=category,
            total_tests=total,
            intent_correct=intent_correct,
            intent_accuracy=intent_correct / total if total > 0 else 0.0,
            entity_precision_avg=sum(entity_precisions) / len(entity_precisions) if entity_precisions else 0.0,
            entity_recall_avg=sum(entity_recalls) / len(entity_recalls) if entity_recalls else 0.0,
            entity_f1_avg=sum(entity_f1s) / len(entity_f1s) if entity_f1s else 0.0,
            tool_accuracy=tool_correct / total if total > 0 else 0.0,
            avg_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            avg_parse_time_ms=sum(parse_times) / len(parse_times) if parse_times else 0.0,
            avg_total_time_ms=sum(total_times) / len(total_times) if total_times else 0.0,
            llm_usage_rate=llm_invoked / total if total > 0 else 0.0,
            error_count=error_count,
            errors=errors
        )
    
    def compare_with_baseline(self, current_metrics: dict) -> tuple[dict, list]:
        """Compare current metrics with baseline and detect regressions."""
        if not self.baseline_path.exists():
            logger.warning("No baseline found for comparison")
            return {}, []
        
        with open(self.baseline_path) as f:
            baseline = json.load(f)
        
        comparison = {}
        regressions = []
        
        # Define thresholds for regression detection
        REGRESSION_THRESHOLDS = {
            "intent_accuracy": 0.02,  # 2% drop is a regression
            "entity_f1_avg": 0.05,    # 5% drop in F1
            "avg_latency_ms": 100,    # 100ms increase
        }
        
        # Overall metrics comparison
        for metric in ["overall_intent_accuracy", "overall_entity_f1", "overall_tool_accuracy"]:
            baseline_val = baseline.get(metric, 0)
            current_val = current_metrics.get(metric, 0)
            diff = current_val - baseline_val
            comparison[metric] = {
                "baseline": baseline_val,
                "current": current_val,
                "diff": diff,
                "improved": diff > 0
            }
            
            if diff < -REGRESSION_THRESHOLDS.get("intent_accuracy", 0.02):
                regressions.append({
                    "metric": metric,
                    "baseline": baseline_val,
                    "current": current_val,
                    "diff": diff,
                    "severity": "high" if diff < -0.05 else "medium"
                })
        
        # Latency comparison (regression if slower)
        baseline_latency = baseline.get("overall_avg_latency_ms", 0)
        current_latency = current_metrics.get("overall_avg_latency_ms", 0)
        latency_diff = current_latency - baseline_latency
        comparison["overall_avg_latency_ms"] = {
            "baseline": baseline_latency,
            "current": current_latency,
            "diff": latency_diff,
            "improved": latency_diff < 0
        }
        
        if latency_diff > REGRESSION_THRESHOLDS["avg_latency_ms"]:
            regressions.append({
                "metric": "overall_avg_latency_ms",
                "baseline": baseline_latency,
                "current": current_latency,
                "diff": latency_diff,
                "severity": "medium" if latency_diff < 200 else "high"
            })
        
        # Per-category comparison
        comparison["categories"] = {}
        baseline_categories = baseline.get("category_metrics", {})
        current_categories = current_metrics.get("category_metrics", {})
        
        for cat in set(baseline_categories.keys()) | set(current_categories.keys()):
            baseline_cat = baseline_categories.get(cat, {})
            current_cat = current_categories.get(cat, {})
            
            comparison["categories"][cat] = {
                "intent_accuracy": {
                    "baseline": baseline_cat.get("intent_accuracy", 0),
                    "current": current_cat.get("intent_accuracy", 0),
                    "diff": current_cat.get("intent_accuracy", 0) - baseline_cat.get("intent_accuracy", 0)
                }
            }
        
        return comparison, regressions
    
    def save_baseline(self, report: EvaluationReport):
        """Save current run as the new baseline."""
        baseline_data = {
            "timestamp": report.timestamp,
            "overall_intent_accuracy": report.overall_intent_accuracy,
            "overall_entity_f1": report.overall_entity_f1,
            "overall_tool_accuracy": report.overall_tool_accuracy,
            "overall_avg_latency_ms": report.overall_avg_latency_ms,
            "overall_llm_usage_rate": report.overall_llm_usage_rate,
            "category_metrics": {
                cat: asdict(metrics) 
                for cat, metrics in report.category_metrics.items()
            }
        }
        
        with open(self.baseline_path, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        logger.info(f"Baseline saved to {self.baseline_path}")
    
    def generate_html_report(self, report: EvaluationReport) -> Path:
        """Generate an HTML report with visualizations."""
        html_path = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Calculate status colors
        def get_color(value: float, thresholds: tuple = (0.7, 0.9)) -> str:
            if value >= thresholds[1]:
                return "green"
            elif value >= thresholds[0]:
                return "orange"
            return "red"
        
        # Build category rows
        category_rows = ""
        for cat, metrics in sorted(report.category_metrics.items()):
            if isinstance(metrics, dict):
                metrics = CategoryMetrics(**metrics)
            
            intent_color = get_color(metrics.intent_accuracy)
            entity_color = get_color(metrics.entity_f1_avg)
            
            category_rows += f"""
            <tr>
                <td>{cat}</td>
                <td>{metrics.total_tests}</td>
                <td style="color: {intent_color}">{metrics.intent_accuracy:.1%}</td>
                <td style="color: {entity_color}">{metrics.entity_f1_avg:.1%}</td>
                <td>{metrics.tool_accuracy:.1%}</td>
                <td>{metrics.avg_parse_time_ms:.1f}ms</td>
                <td>{metrics.llm_usage_rate:.1%}</td>
                <td style="color: {'red' if metrics.error_count > 0 else 'green'}">{metrics.error_count}</td>
            </tr>
            """
        
        # Build regression warnings
        regression_html = ""
        if report.regressions:
            regression_html = """
            <div class="regression-warning">
                <h3>⚠️ Regressions Detected</h3>
                <ul>
            """
            for reg in report.regressions:
                regression_html += f"""
                <li class="severity-{reg['severity']}">
                    <strong>{reg['metric']}</strong>: 
                    {reg['baseline']:.3f} → {reg['current']:.3f} 
                    (diff: {reg['diff']:+.3f})
                </li>
                """
            regression_html += "</ul></div>"
        
        # Build comparison section
        comparison_html = ""
        if report.baseline_comparison:
            comparison_html = """
            <h2>Baseline Comparison</h2>
            <table>
                <tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Diff</th><th>Status</th></tr>
            """
            for metric, data in report.baseline_comparison.items():
                if metric == "categories":
                    continue
                status = "✅" if data.get("improved") else "⚠️"
                comparison_html += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{data['baseline']:.3f}</td>
                    <td>{data['current']:.3f}</td>
                    <td>{data['diff']:+.3f}</td>
                    <td>{status}</td>
                </tr>
                """
            comparison_html += "</table>"
        
        # Build failed tests section
        failed_tests_html = ""
        failed = [r for r in report.test_results if not r.get("intent_correct", True)]
        if failed:
            failed_tests_html = """
            <h2>Failed Tests</h2>
            <table>
                <tr><th>Category</th><th>Query</th><th>Expected</th><th>Actual</th><th>Error</th></tr>
            """
            for test in failed[:50]:  # Show first 50 failures
                failed_tests_html += f"""
                <tr>
                    <td>{test.get('category', 'N/A')}</td>
                    <td>{test.get('query', 'N/A')[:50]}...</td>
                    <td>{test.get('expected_intent', 'N/A')}</td>
                    <td>{test.get('actual_intent', 'N/A')}</td>
                    <td>{test.get('error', '-')}</td>
                </tr>
                """
            failed_tests_html += "</table>"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>BioPipelines Chat Agent Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f9f9f9; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #4CAF50; }}
        .metric-card h3 {{ margin: 0 0 10px 0; color: #666; font-size: 14px; }}
        .metric-card .value {{ font-size: 36px; font-weight: bold; color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .regression-warning {{ background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 4px; margin: 20px 0; }}
        .severity-high {{ color: #d32f2f; font-weight: bold; }}
        .severity-medium {{ color: #f57c00; }}
        .timestamp {{ color: #888; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>BioPipelines Chat Agent Evaluation Report</h1>
        <p class="timestamp">Generated: {report.timestamp}</p>
        
        {regression_html}
        
        <div class="summary">
            <div class="metric-card">
                <h3>Total Tests</h3>
                <div class="value">{report.total_tests}</div>
            </div>
            <div class="metric-card" style="border-color: {get_color(report.overall_intent_accuracy)}">
                <h3>Intent Accuracy</h3>
                <div class="value" style="color: {get_color(report.overall_intent_accuracy)}">{report.overall_intent_accuracy:.1%}</div>
            </div>
            <div class="metric-card" style="border-color: {get_color(report.overall_entity_f1)}">
                <h3>Entity F1</h3>
                <div class="value" style="color: {get_color(report.overall_entity_f1)}">{report.overall_entity_f1:.1%}</div>
            </div>
            <div class="metric-card">
                <h3>Tool Accuracy</h3>
                <div class="value">{report.overall_tool_accuracy:.1%}</div>
            </div>
            <div class="metric-card">
                <h3>Avg Latency</h3>
                <div class="value">{report.overall_avg_latency_ms:.0f}ms</div>
            </div>
            <div class="metric-card">
                <h3>LLM Usage</h3>
                <div class="value">{report.overall_llm_usage_rate:.1%}</div>
            </div>
        </div>
        
        <h2>Results by Category</h2>
        <table>
            <tr>
                <th>Category</th>
                <th>Tests</th>
                <th>Intent Acc.</th>
                <th>Entity F1</th>
                <th>Tool Acc.</th>
                <th>Avg Latency</th>
                <th>LLM Usage</th>
                <th>Errors</th>
            </tr>
            {category_rows}
        </table>
        
        {comparison_html}
        
        {failed_tests_html}
    </div>
</body>
</html>
        """
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {html_path}")
        return html_path
    
    async def run(
        self,
        categories: Optional[list[str]] = None,
        save_baseline: bool = False,
        compare_baseline: bool = True,
        generate_report: bool = True
    ) -> EvaluationReport:
        """Run the complete evaluation."""
        logger.info("Starting unified evaluation run")
        
        # Initialize agent
        await self.initialize_agent()
        
        # Load test data
        test_data = self.load_test_data()
        
        if categories:
            test_data = {k: v for k, v in test_data.items() if k in categories}
        
        if not test_data:
            logger.error("No test data found")
            return EvaluationReport(timestamp=datetime.now().isoformat())
        
        # Run tests for each category
        all_results = []
        category_metrics = {}
        
        for category, test_cases in test_data.items():
            results = await self.run_category(category, test_cases)
            all_results.extend(results)
            category_metrics[category] = self.calculate_category_metrics(category, results)
        
        # Calculate overall metrics
        total_tests = len(all_results)
        intent_correct = sum(1 for r in all_results if r.intent_correct)
        tool_correct = sum(1 for r in all_results if r.tool_correct)
        llm_invoked = sum(1 for r in all_results if r.llm_invoked)
        
        entity_f1s = [r.entity_f1 for r in all_results if not r.error]
        parse_times = [r.parse_time_ms for r in all_results if not r.error]
        
        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_tests=total_tests,
            overall_intent_accuracy=intent_correct / total_tests if total_tests > 0 else 0.0,
            overall_entity_f1=sum(entity_f1s) / len(entity_f1s) if entity_f1s else 0.0,
            overall_tool_accuracy=tool_correct / total_tests if total_tests > 0 else 0.0,
            overall_avg_latency_ms=sum(parse_times) / len(parse_times) if parse_times else 0.0,
            overall_llm_usage_rate=llm_invoked / total_tests if total_tests > 0 else 0.0,
            category_metrics={k: asdict(v) for k, v in category_metrics.items()},
            test_results=[asdict(r) for r in all_results]
        )
        
        # Compare with baseline if requested
        if compare_baseline:
            comparison, regressions = self.compare_with_baseline(asdict(report))
            report.baseline_comparison = comparison
            report.regressions = regressions
        
        # Save results
        with open(self.results_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        logger.info(f"Results saved to {self.results_path}")
        
        # Save as baseline if requested
        if save_baseline:
            self.save_baseline(report)
        
        # Generate HTML report
        if generate_report:
            html_path = self.generate_html_report(report)
            logger.info(f"HTML report: {html_path}")
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    def _print_summary(self, report: EvaluationReport):
        """Print a summary of the evaluation results."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Tests:        {report.total_tests}")
        print(f"Intent Accuracy:    {report.overall_intent_accuracy:.1%}")
        print(f"Entity F1:          {report.overall_entity_f1:.1%}")
        print(f"Tool Accuracy:      {report.overall_tool_accuracy:.1%}")
        print(f"Avg Latency:        {report.overall_avg_latency_ms:.1f}ms")
        print(f"LLM Usage Rate:     {report.overall_llm_usage_rate:.1%}")
        print("-"*60)
        
        if report.regressions:
            print("\n⚠️  REGRESSIONS DETECTED:")
            for reg in report.regressions:
                print(f"   - {reg['metric']}: {reg['baseline']:.3f} → {reg['current']:.3f}")
        
        print("\nBy Category:")
        for cat, metrics in sorted(report.category_metrics.items()):
            if isinstance(metrics, dict):
                acc = metrics.get("intent_accuracy", 0)
                total = metrics.get("total_tests", 0)
                errors = metrics.get("error_count", 0)
            else:
                acc = metrics.intent_accuracy
                total = metrics.total_tests
                errors = metrics.error_count
            
            status = "✓" if acc >= 0.9 else "○" if acc >= 0.7 else "✗"
            print(f"  {status} {cat}: {acc:.1%} ({total} tests, {errors} errors)")
        
        print("="*60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run BioPipelines chat agent evaluation")
    parser.add_argument("--category", "-c", nargs="+", help="Specific categories to test")
    parser.add_argument("--save-baseline", action="store_true", help="Save current run as baseline")
    parser.add_argument("--no-compare", action="store_true", help="Skip baseline comparison")
    parser.add_argument("--no-report", action="store_true", help="Skip HTML report generation")
    parser.add_argument("--output-dir", "-o", type=Path, help="Output directory for results")
    
    args = parser.parse_args()
    
    runner = UnifiedEvaluationRunner(
        output_dir=args.output_dir
    )
    
    # Run evaluation
    report = asyncio.run(runner.run(
        categories=args.category,
        save_baseline=args.save_baseline,
        compare_baseline=not args.no_compare,
        generate_report=not args.no_report
    ))
    
    # Exit with error code if regressions found
    if report.regressions:
        logger.warning("Regressions detected - exiting with error code 1")
        sys.exit(1)
    
    # Exit with error if accuracy below threshold
    if report.overall_intent_accuracy < 0.7:
        logger.warning(f"Intent accuracy {report.overall_intent_accuracy:.1%} below 70% threshold")
        sys.exit(1)


if __name__ == "__main__":
    main()
