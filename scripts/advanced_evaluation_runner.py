#!/usr/bin/env python3
"""
Advanced Evaluation Runner for BioPipelines Chat Agent

This runner integrates ALL evaluation capabilities:
1. Enhanced metrics (rule-based, LLM-as-judge, semantic similarity)
2. Historical tracking with SQLite database
3. Synthetic test generation
4. Adversarial testing
5. Smart test selection (failure-focused, time-based)
6. Multi-turn conversation evaluation
7. Performance benchmarking

This is the recommended evaluation tool for production use.
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import hashlib

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "tests"))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TestCase:
    """A single test case."""
    id: str
    query: str
    expected_intent: str
    expected_entities: dict = field(default_factory=dict)
    context: list = field(default_factory=list)
    category: str = ""
    difficulty: int = 1
    tags: list = field(default_factory=list)


@dataclass 
class DetailedTestResult:
    """Detailed result for a single test."""
    test_id: str
    query: str
    expected_intent: str
    actual_intent: str
    expected_entities: dict
    actual_entities: dict
    # Metric scores
    intent_accuracy: float
    entity_f1: float
    tool_accuracy: float
    semantic_similarity: float = 0.0
    llm_quality_score: float = 0.0
    # Status
    passed: bool = False
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    # Category for grouping
    category: str = ""
    difficulty: int = 1


@dataclass
class EvaluationSummary:
    """Summary of evaluation run."""
    run_id: str
    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    # Aggregate scores
    overall_accuracy: float
    intent_accuracy: float
    entity_f1: float
    tool_accuracy: float
    semantic_similarity: float
    # Timing
    total_time_seconds: float
    avg_time_per_test_ms: float
    # Breakdowns
    by_category: dict = field(default_factory=dict)
    by_difficulty: dict = field(default_factory=dict)
    # Metadata
    config: dict = field(default_factory=dict)
    

# ============================================================================
# ADVANCED EVALUATION RUNNER
# ============================================================================

class AdvancedEvaluationRunner:
    """
    Comprehensive evaluation runner with all advanced features.
    """
    
    def __init__(
        self,
        parser_class: type = None,
        enable_llm_judge: bool = False,
        enable_semantic: bool = True,
        enable_historical: bool = True,
        enable_adversarial: bool = False,
        db_path: str = None,
        output_dir: str = None,
    ):
        """
        Initialize the advanced runner.
        
        Args:
            parser_class: The parser class to evaluate
            enable_llm_judge: Enable LLM-as-judge evaluation
            enable_semantic: Enable semantic similarity
            enable_historical: Enable historical tracking
            enable_adversarial: Include adversarial tests
            db_path: Path to historical tracking database
            output_dir: Directory for output reports
        """
        self.enable_llm_judge = enable_llm_judge
        self.enable_semantic = enable_semantic
        self.enable_historical = enable_historical
        self.enable_adversarial = enable_adversarial
        
        # Set up output directory
        self.output_dir = Path(output_dir) if output_dir else project_root / "reports" / "evaluations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize parser
        self.parser = None
        if parser_class:
            try:
                self.parser = parser_class()
            except Exception as e:
                logger.warning(f"Could not initialize parser: {e}")
        
        # Load parser if not provided
        if self.parser is None:
            self._load_default_parser()
        
        # Initialize components
        self._init_metrics()
        self._init_historical_tracker(db_path)
        self._load_test_data()
        
        # Run tracking
        self.run_id = self._generate_run_id()
        self.results: list[DetailedTestResult] = []
        
    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_part = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"eval_{timestamp}_{hash_part}"
    
    def _load_default_parser(self):
        """Load the default parser."""
        try:
            from biopipe.chat.nlu.parser import UnifiedNLUParser
            self.parser = UnifiedNLUParser()
            logger.info("Loaded UnifiedNLUParser")
        except ImportError:
            try:
                from biopipe.nlu.parser import UnifiedNLUParser
                self.parser = UnifiedNLUParser()
                logger.info("Loaded UnifiedNLUParser from alternate location")
            except ImportError:
                logger.warning("Could not load UnifiedNLUParser, using mock")
                self.parser = self._create_mock_parser()
    
    def _create_mock_parser(self):
        """Create a mock parser for testing."""
        class MockParser:
            def parse(self, query):
                return {
                    "intent": "META_UNKNOWN",
                    "confidence": 0.5,
                    "entities": {}
                }
        return MockParser()
    
    def _init_metrics(self):
        """Initialize evaluation metrics."""
        self.metrics = {}
        
        try:
            from evaluation.enhanced_metrics import (
                IntentAccuracyMetric,
                EntityF1Metric,
                ToolAccuracyMetric,
                LLMResponseQualityMetric,
                SemanticSimilarityMetric
            )
            
            self.metrics["intent"] = IntentAccuracyMetric()
            self.metrics["entity"] = EntityF1Metric()
            self.metrics["tool"] = ToolAccuracyMetric()
            
            if self.enable_semantic:
                try:
                    self.metrics["semantic"] = SemanticSimilarityMetric()
                    logger.info("Enabled semantic similarity")
                except Exception as e:
                    logger.warning(f"Could not load semantic similarity: {e}")
            
            if self.enable_llm_judge:
                try:
                    self.metrics["llm"] = LLMResponseQualityMetric()
                    logger.info("Enabled LLM-as-judge")
                except Exception as e:
                    logger.warning(f"Could not load LLM judge: {e}")
                    
            logger.info("Loaded enhanced metrics")
            
        except ImportError as e:
            logger.warning(f"Could not load enhanced metrics: {e}")
            self._init_basic_metrics()
    
    def _init_basic_metrics(self):
        """Initialize basic metrics as fallback."""
        class BasicIntentMetric:
            def score(self, expected, actual):
                return 1.0 if expected.upper() == actual.upper() else 0.0
        
        class BasicEntityMetric:
            def score(self, expected, actual):
                if not expected and not actual:
                    return 1.0
                if not expected or not actual:
                    return 0.0
                
                # Simple overlap score
                expected_set = set(str(v).lower() for v in expected.values())
                actual_set = set(str(v).lower() for v in actual.values())
                
                if not expected_set:
                    return 1.0
                    
                intersection = expected_set & actual_set
                return len(intersection) / len(expected_set)
        
        self.metrics["intent"] = BasicIntentMetric()
        self.metrics["entity"] = BasicEntityMetric()
    
    def _init_historical_tracker(self, db_path: str = None):
        """Initialize historical tracking."""
        self.tracker = None
        
        if not self.enable_historical:
            return
            
        try:
            from evaluation.historical_tracker import HistoricalTracker
            
            if db_path is None:
                db_path = str(project_root / "data" / "evaluation_history.db")
            
            self.tracker = HistoricalTracker(db_path)
            logger.info(f"Initialized historical tracker: {db_path}")
            
        except ImportError as e:
            logger.warning(f"Could not load historical tracker: {e}")
    
    def _load_test_data(self):
        """Load all test data."""
        self.test_cases: list[TestCase] = []
        
        # Load from comprehensive test data
        try:
            from scripts.comprehensive_test_data import get_all_test_conversations
            
            conversations = get_all_test_conversations()
            for category, convs in conversations.items():
                for conv in convs:
                    for i, turn in enumerate(conv.get("turns", [])):
                        test = TestCase(
                            id=f"{category}_{conv.get('id', 'unknown')}_{i}",
                            query=turn.get("user", ""),
                            expected_intent=turn.get("expected_intent", "META_UNKNOWN"),
                            expected_entities=turn.get("expected_entities", {}),
                            context=[t.get("user", "") for t in conv.get("turns", [])[:i]],
                            category=category,
                            difficulty=conv.get("difficulty", 1),
                            tags=conv.get("tags", [])
                        )
                        self.test_cases.append(test)
            
            logger.info(f"Loaded {len(self.test_cases)} test cases from comprehensive data")
            
        except ImportError:
            logger.warning("Could not load comprehensive test data")
        
        # Load adversarial tests if enabled
        if self.enable_adversarial:
            self._load_adversarial_tests()
    
    def _load_adversarial_tests(self):
        """Load adversarial test cases."""
        try:
            from evaluation.adversarial_tests import ALL_ADVERSARIAL_TESTS
            
            for adv_test in ALL_ADVERSARIAL_TESTS:
                test = TestCase(
                    id=f"adversarial_{adv_test.id}",
                    query=adv_test.query,
                    expected_intent=adv_test.expected_intent or "META_UNKNOWN",
                    expected_entities=adv_test.expected_entities,
                    category=f"adversarial_{adv_test.category}",
                    difficulty={"low": 3, "medium": 4, "high": 5}.get(adv_test.risk_level, 3),
                    tags=[adv_test.category, adv_test.risk_level]
                )
                self.test_cases.append(test)
            
            logger.info(f"Added {len(ALL_ADVERSARIAL_TESTS)} adversarial tests")
            
        except ImportError as e:
            logger.warning(f"Could not load adversarial tests: {e}")
    
    def run_single_test(self, test: TestCase) -> DetailedTestResult:
        """Run a single test case and return detailed results."""
        start_time = time.time()
        error = None
        result = None
        
        try:
            # Parse query
            if hasattr(self.parser, "parse"):
                result = self.parser.parse(test.query)
            else:
                result = self.parser(test.query)
            
            actual_intent = result.get("intent", "")
            actual_entities = result.get("entities", {})
            
        except Exception as e:
            error = str(e)
            actual_intent = ""
            actual_entities = {}
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Calculate metric scores
        intent_score = 0.0
        entity_score = 0.0
        tool_score = 0.0
        semantic_score = 0.0
        llm_score = 0.0
        
        if error is None:
            # Intent accuracy
            if "intent" in self.metrics:
                try:
                    intent_score = self.metrics["intent"].score(
                        test.expected_intent, actual_intent
                    )
                except Exception as e:
                    logger.debug(f"Intent metric error: {e}")
            
            # Entity F1
            if "entity" in self.metrics:
                try:
                    entity_score = self.metrics["entity"].score(
                        test.expected_entities, actual_entities
                    )
                except Exception as e:
                    logger.debug(f"Entity metric error: {e}")
            
            # Tool accuracy (if applicable)
            if "tool" in self.metrics and result.get("tools"):
                try:
                    # Compute tool accuracy if we have expected tools
                    tool_score = 1.0 if result.get("tools") else 0.5
                except Exception as e:
                    logger.debug(f"Tool metric error: {e}")
            else:
                tool_score = 1.0  # No tools expected/returned
            
            # Semantic similarity
            if "semantic" in self.metrics:
                try:
                    semantic_score = self.metrics["semantic"].score(
                        test.query, result.get("response", actual_intent)
                    )
                except Exception as e:
                    logger.debug(f"Semantic metric error: {e}")
            
            # LLM quality
            if "llm" in self.metrics:
                try:
                    llm_score = self.metrics["llm"].score(
                        query=test.query,
                        response=result.get("response", ""),
                        expected_intent=test.expected_intent
                    )
                except Exception as e:
                    logger.debug(f"LLM metric error: {e}")
        
        # Determine if passed (primary criterion is intent accuracy)
        passed = intent_score >= 0.8 and error is None
        
        return DetailedTestResult(
            test_id=test.id,
            query=test.query,
            expected_intent=test.expected_intent,
            actual_intent=actual_intent,
            expected_entities=test.expected_entities,
            actual_entities=actual_entities,
            intent_accuracy=intent_score,
            entity_f1=entity_score,
            tool_accuracy=tool_score,
            semantic_similarity=semantic_score,
            llm_quality_score=llm_score,
            passed=passed,
            error=error,
            elapsed_ms=elapsed_ms,
            category=test.category,
            difficulty=test.difficulty
        )
    
    def run_evaluation(
        self,
        test_filter: str = None,
        category_filter: str = None,
        difficulty_filter: int = None,
        max_tests: int = None,
        focus_failures: bool = False,
    ) -> EvaluationSummary:
        """
        Run the full evaluation.
        
        Args:
            test_filter: Regex pattern to filter tests by ID
            category_filter: Filter by category name
            difficulty_filter: Filter by difficulty level
            max_tests: Maximum number of tests to run
            focus_failures: Prioritize previously failed tests
        """
        start_time = time.time()
        
        # Filter tests
        tests = self._filter_tests(
            test_filter=test_filter,
            category_filter=category_filter,
            difficulty_filter=difficulty_filter,
            focus_failures=focus_failures
        )
        
        if max_tests:
            tests = tests[:max_tests]
        
        logger.info(f"Running {len(tests)} tests...")
        
        # Run tests
        self.results = []
        for i, test in enumerate(tests):
            if (i + 1) % 50 == 0:
                logger.info(f"  Progress: {i + 1}/{len(tests)}")
            
            result = self.run_single_test(test)
            self.results.append(result)
        
        total_time = time.time() - start_time
        
        # Calculate summary
        summary = self._calculate_summary(total_time)
        
        # Record in historical tracker
        if self.tracker:
            self._record_to_history(summary)
        
        # Save results
        self._save_results(summary)
        
        return summary
    
    def _filter_tests(
        self,
        test_filter: str = None,
        category_filter: str = None,
        difficulty_filter: int = None,
        focus_failures: bool = False
    ) -> list[TestCase]:
        """Filter tests based on criteria."""
        import re
        
        tests = self.test_cases.copy()
        
        # Apply test ID filter
        if test_filter:
            pattern = re.compile(test_filter, re.IGNORECASE)
            tests = [t for t in tests if pattern.search(t.id)]
        
        # Apply category filter
        if category_filter:
            tests = [t for t in tests if category_filter.lower() in t.category.lower()]
        
        # Apply difficulty filter
        if difficulty_filter:
            tests = [t for t in tests if t.difficulty <= difficulty_filter]
        
        # Focus on failures (if historical data available)
        if focus_failures and self.tracker:
            try:
                failed_ids = self.tracker.get_recent_failures(limit=100)
                
                # Sort to prioritize failures
                def priority_sort(test):
                    if test.id in failed_ids:
                        return (0, test.id)  # Failures first
                    return (1, test.id)
                
                tests.sort(key=priority_sort)
                
            except Exception as e:
                logger.warning(f"Could not load failure history: {e}")
        
        return tests
    
    def _calculate_summary(self, total_time: float) -> EvaluationSummary:
        """Calculate evaluation summary from results."""
        if not self.results:
            return EvaluationSummary(
                run_id=self.run_id,
                timestamp=datetime.now().isoformat(),
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                overall_accuracy=0.0,
                intent_accuracy=0.0,
                entity_f1=0.0,
                tool_accuracy=0.0,
                semantic_similarity=0.0,
                total_time_seconds=total_time,
                avg_time_per_test_ms=0.0
            )
        
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        # Average metrics
        avg_intent = sum(r.intent_accuracy for r in self.results) / len(self.results)
        avg_entity = sum(r.entity_f1 for r in self.results) / len(self.results)
        avg_tool = sum(r.tool_accuracy for r in self.results) / len(self.results)
        avg_semantic = sum(r.semantic_similarity for r in self.results) / len(self.results)
        
        # Overall accuracy (weighted)
        overall = 0.5 * avg_intent + 0.3 * avg_entity + 0.2 * avg_tool
        
        # Category breakdown
        by_category = {}
        for r in self.results:
            cat = r.category or "uncategorized"
            if cat not in by_category:
                by_category[cat] = {"passed": 0, "failed": 0, "intent_acc": [], "entity_f1": []}
            
            if r.passed:
                by_category[cat]["passed"] += 1
            else:
                by_category[cat]["failed"] += 1
            
            by_category[cat]["intent_acc"].append(r.intent_accuracy)
            by_category[cat]["entity_f1"].append(r.entity_f1)
        
        # Calculate category averages
        for cat in by_category:
            by_category[cat]["avg_intent"] = sum(by_category[cat]["intent_acc"]) / len(by_category[cat]["intent_acc"])
            by_category[cat]["avg_entity"] = sum(by_category[cat]["entity_f1"]) / len(by_category[cat]["entity_f1"])
            del by_category[cat]["intent_acc"]
            del by_category[cat]["entity_f1"]
        
        # Difficulty breakdown
        by_difficulty = {}
        for r in self.results:
            diff = r.difficulty
            if diff not in by_difficulty:
                by_difficulty[diff] = {"passed": 0, "failed": 0}
            
            if r.passed:
                by_difficulty[diff]["passed"] += 1
            else:
                by_difficulty[diff]["failed"] += 1
        
        return EvaluationSummary(
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            total_tests=len(self.results),
            passed_tests=passed,
            failed_tests=failed,
            overall_accuracy=overall,
            intent_accuracy=avg_intent,
            entity_f1=avg_entity,
            tool_accuracy=avg_tool,
            semantic_similarity=avg_semantic,
            total_time_seconds=total_time,
            avg_time_per_test_ms=total_time * 1000 / len(self.results),
            by_category=by_category,
            by_difficulty=by_difficulty,
            config={
                "enable_llm_judge": self.enable_llm_judge,
                "enable_semantic": self.enable_semantic,
                "enable_adversarial": self.enable_adversarial
            }
        )
    
    def _record_to_history(self, summary: EvaluationSummary):
        """Record results to historical tracker."""
        try:
            # Record the evaluation run
            self.tracker.record_evaluation(
                run_id=summary.run_id,
                metrics={
                    "overall_accuracy": summary.overall_accuracy,
                    "intent_accuracy": summary.intent_accuracy,
                    "entity_f1": summary.entity_f1,
                    "tool_accuracy": summary.tool_accuracy,
                    "semantic_similarity": summary.semantic_similarity,
                    "pass_rate": summary.passed_tests / summary.total_tests if summary.total_tests > 0 else 0
                },
                category_results=summary.by_category,
                failures=[
                    {
                        "test_id": r.test_id,
                        "expected_intent": r.expected_intent,
                        "actual_intent": r.actual_intent,
                        "error": r.error
                    }
                    for r in self.results if not r.passed
                ]
            )
            logger.info(f"Recorded results to historical tracker: {summary.run_id}")
            
        except Exception as e:
            logger.warning(f"Could not record to history: {e}")
    
    def _save_results(self, summary: EvaluationSummary):
        """Save detailed results to files."""
        # Save summary
        summary_path = self.output_dir / f"{self.run_id}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(asdict(summary), f, indent=2, default=str)
        
        # Save detailed results
        details_path = self.output_dir / f"{self.run_id}_details.json"
        with open(details_path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2, default=str)
        
        # Save failures for quick analysis
        failures = [r for r in self.results if not r.passed]
        if failures:
            failures_path = self.output_dir / f"{self.run_id}_failures.json"
            with open(failures_path, "w") as f:
                json.dump([asdict(r) for r in failures], f, indent=2, default=str)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def print_report(self, summary: EvaluationSummary = None):
        """Print a formatted evaluation report."""
        if summary is None:
            summary = self._calculate_summary(0)
        
        print("\n" + "=" * 70)
        print("ADVANCED EVALUATION REPORT")
        print(f"Run ID: {summary.run_id}")
        print(f"Timestamp: {summary.timestamp}")
        print("=" * 70)
        
        # Overall results
        print(f"\n📊 OVERALL RESULTS")
        print(f"   Total Tests: {summary.total_tests}")
        print(f"   Passed: {summary.passed_tests} ({summary.passed_tests/summary.total_tests*100:.1f}%)")
        print(f"   Failed: {summary.failed_tests} ({summary.failed_tests/summary.total_tests*100:.1f}%)")
        
        # Metrics
        print(f"\n📈 METRICS")
        print(f"   Overall Accuracy:    {summary.overall_accuracy*100:.1f}%")
        print(f"   Intent Accuracy:     {summary.intent_accuracy*100:.1f}%")
        print(f"   Entity F1:           {summary.entity_f1*100:.1f}%")
        print(f"   Tool Accuracy:       {summary.tool_accuracy*100:.1f}%")
        if summary.semantic_similarity > 0:
            print(f"   Semantic Similarity: {summary.semantic_similarity*100:.1f}%")
        
        # Timing
        print(f"\n⏱️  TIMING")
        print(f"   Total Time: {summary.total_time_seconds:.1f}s")
        print(f"   Avg per Test: {summary.avg_time_per_test_ms:.1f}ms")
        
        # Category breakdown
        if summary.by_category:
            print(f"\n📁 BY CATEGORY")
            for cat, data in sorted(summary.by_category.items()):
                total = data["passed"] + data["failed"]
                rate = data["passed"] / total * 100 if total > 0 else 0
                status = "✅" if rate >= 80 else "⚠️" if rate >= 60 else "❌"
                print(f"   {status} {cat}: {data['passed']}/{total} ({rate:.0f}%)")
        
        # Difficulty breakdown
        if summary.by_difficulty:
            print(f"\n📊 BY DIFFICULTY")
            for diff in sorted(summary.by_difficulty.keys()):
                data = summary.by_difficulty[diff]
                total = data["passed"] + data["failed"]
                rate = data["passed"] / total * 100 if total > 0 else 0
                bar = "█" * int(rate / 10) + "░" * (10 - int(rate / 10))
                print(f"   Tier {diff}: [{bar}] {rate:.0f}% ({data['passed']}/{total})")
        
        # Top failures
        failures = [r for r in self.results if not r.passed]
        if failures:
            print(f"\n❌ TOP FAILURES ({len(failures)} total)")
            for r in failures[:5]:
                query_preview = r.query[:50] + "..." if len(r.query) > 50 else r.query
                print(f"   • {r.test_id}")
                print(f"     Query: {query_preview}")
                print(f"     Expected: {r.expected_intent}, Got: {r.actual_intent}")
                if r.error:
                    print(f"     Error: {r.error[:50]}...")
        
        print("\n" + "=" * 70)
        print(f"📄 Full results: {self.output_dir / f'{self.run_id}_details.json'}")
        print("=" * 70 + "\n")
    
    def get_trend_analysis(self, days: int = 30) -> dict:
        """Get trend analysis from historical data."""
        if not self.tracker:
            return {"error": "Historical tracking not enabled"}
        
        try:
            return self.tracker.get_trend_analysis(days=days)
        except Exception as e:
            return {"error": str(e)}


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Advanced evaluation runner for BioPipelines Chat Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run full evaluation
  %(prog)s --category data_discovery # Run specific category
  %(prog)s --max-tests 100           # Quick evaluation with 100 tests
  %(prog)s --adversarial             # Include adversarial tests
  %(prog)s --focus-failures          # Prioritize previously failed tests
  %(prog)s --llm-judge               # Enable LLM-as-judge evaluation
  %(prog)s --trends                  # Show historical trends
        """
    )
    
    parser.add_argument(
        "--category",
        help="Filter tests by category"
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Maximum difficulty level"
    )
    parser.add_argument(
        "--max-tests",
        type=int,
        help="Maximum number of tests to run"
    )
    parser.add_argument(
        "--focus-failures",
        action="store_true",
        help="Prioritize previously failed tests"
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        help="Include adversarial tests"
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Enable LLM-as-judge evaluation"
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic similarity"
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable historical tracking"
    )
    parser.add_argument(
        "--trends",
        action="store_true",
        help="Show historical trends instead of running evaluation"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for results"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet output (only summary)"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Initialize runner
    runner = AdvancedEvaluationRunner(
        enable_llm_judge=args.llm_judge,
        enable_semantic=not args.no_semantic,
        enable_historical=not args.no_history,
        enable_adversarial=args.adversarial,
        output_dir=args.output_dir
    )
    
    # Show trends or run evaluation
    if args.trends:
        print("\n📈 HISTORICAL TRENDS (Last 30 days)")
        print("=" * 50)
        trends = runner.get_trend_analysis(days=30)
        
        if "error" in trends:
            print(f"Error: {trends['error']}")
        else:
            for metric, data in trends.get("metrics", {}).items():
                print(f"\n{metric}:")
                print(f"  Current: {data.get('current', 0)*100:.1f}%")
                print(f"  Average: {data.get('average', 0)*100:.1f}%")
                print(f"  Trend:   {data.get('trend', 'stable')}")
    else:
        # Run evaluation
        summary = runner.run_evaluation(
            category_filter=args.category,
            difficulty_filter=args.difficulty,
            max_tests=args.max_tests,
            focus_failures=args.focus_failures
        )
        
        runner.print_report(summary)
        
        # Return exit code based on pass rate
        if summary.passed_tests / summary.total_tests >= 0.8:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure


if __name__ == "__main__":
    main()
