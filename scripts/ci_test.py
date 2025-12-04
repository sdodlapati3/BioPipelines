#!/usr/bin/env python3
"""
CI/CD Test Command for BioPipelines Chat Agent

A simple, CI-friendly script that runs evaluation tests and returns
appropriate exit codes for integration with GitHub Actions, GitLab CI, etc.

Exit Codes:
    0 - All tests passed, no regressions
    1 - Regressions detected (metrics worse than baseline)
    2 - Tests failed below minimum threshold
    3 - Error during test execution

Usage:
    python scripts/ci_test.py                    # Run all tests
    python scripts/ci_test.py --quick            # Run quick subset of tests
    python scripts/ci_test.py --strict           # Fail on any test failure
    python scripts/ci_test.py --create-baseline  # Create new baseline from this run
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Minimum thresholds for CI
THRESHOLDS = {
    "intent_accuracy": 0.70,      # 70% minimum intent accuracy
    "entity_f1": 0.50,            # 50% minimum entity F1
    "tool_accuracy": 0.80,        # 80% minimum tool accuracy
    "max_latency_ms": 5000,       # 5 second max average latency
    "regression_tolerance": 0.02,  # 2% drop allowed before regression
}

# Quick test categories for fast CI runs
QUICK_TEST_CATEGORIES = [
    "data_discovery",
    "workflow_generation",
    "job_management",
]


def print_banner():
    """Print CI test banner."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           BioPipelines Chat Agent - CI Test Suite                ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def print_result(name: str, passed: bool, value: str, threshold: str = None):
    """Print a formatted test result."""
    status = "✓" if passed else "✗"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    
    threshold_str = f" (threshold: {threshold})" if threshold else ""
    print(f"  {color}{status}{reset} {name}: {value}{threshold_str}")


async def run_tests(
    quick: bool = False,
    strict: bool = False,
    create_baseline: bool = False
) -> int:
    """Run the test suite and return exit code."""
    
    try:
        from scripts.unified_evaluation_runner import UnifiedEvaluationRunner
    except ImportError as e:
        print(f"\n✗ Failed to import evaluation runner: {e}")
        print("  Make sure all dependencies are installed.")
        return 3
    
    print(f"\n📅 Test run started: {datetime.now().isoformat()}")
    print(f"🔧 Mode: {'quick' if quick else 'full'}, {'strict' if strict else 'standard'}")
    
    runner = UnifiedEvaluationRunner()
    
    # Determine categories to test
    categories = QUICK_TEST_CATEGORIES if quick else None
    
    try:
        # Run evaluation
        report = await runner.run(
            categories=categories,
            save_baseline=create_baseline,
            compare_baseline=True,
            generate_report=True
        )
    except Exception as e:
        print(f"\n✗ Error during test execution: {e}")
        return 3
    
    # Analyze results
    print("\n" + "─"*60)
    print("TEST RESULTS")
    print("─"*60)
    
    all_passed = True
    
    # Check intent accuracy
    intent_passed = report.overall_intent_accuracy >= THRESHOLDS["intent_accuracy"]
    print_result(
        "Intent Accuracy",
        intent_passed,
        f"{report.overall_intent_accuracy:.1%}",
        f">= {THRESHOLDS['intent_accuracy']:.0%}"
    )
    if not intent_passed:
        all_passed = False
    
    # Check entity F1
    entity_passed = report.overall_entity_f1 >= THRESHOLDS["entity_f1"]
    print_result(
        "Entity F1",
        entity_passed,
        f"{report.overall_entity_f1:.1%}",
        f">= {THRESHOLDS['entity_f1']:.0%}"
    )
    if not entity_passed:
        all_passed = False
    
    # Check tool accuracy
    tool_passed = report.overall_tool_accuracy >= THRESHOLDS["tool_accuracy"]
    print_result(
        "Tool Accuracy",
        tool_passed,
        f"{report.overall_tool_accuracy:.1%}",
        f">= {THRESHOLDS['tool_accuracy']:.0%}"
    )
    if not tool_passed:
        all_passed = False
    
    # Check latency
    latency_passed = report.overall_avg_latency_ms <= THRESHOLDS["max_latency_ms"]
    print_result(
        "Avg Latency",
        latency_passed,
        f"{report.overall_avg_latency_ms:.0f}ms",
        f"<= {THRESHOLDS['max_latency_ms']}ms"
    )
    if not latency_passed:
        all_passed = False
    
    # Check for regressions
    regression_passed = len(report.regressions) == 0
    print_result(
        "Regression Check",
        regression_passed,
        f"{len(report.regressions)} regressions found"
    )
    
    if report.regressions:
        print("\n  ⚠️  Regressions detected:")
        for reg in report.regressions:
            print(f"      - {reg['metric']}: {reg['baseline']:.3f} → {reg['current']:.3f}")
    
    # Summary by category
    print("\n" + "─"*60)
    print("CATEGORY BREAKDOWN")
    print("─"*60)
    
    category_failures = []
    for cat, metrics in sorted(report.category_metrics.items()):
        if isinstance(metrics, dict):
            acc = metrics.get("intent_accuracy", 0)
            total = metrics.get("total_tests", 0)
            errors = metrics.get("error_count", 0)
        else:
            acc = metrics.intent_accuracy
            total = metrics.total_tests
            errors = metrics.error_count
        
        passed = acc >= THRESHOLDS["intent_accuracy"]
        print_result(
            f"{cat}",
            passed,
            f"{acc:.1%} ({total} tests, {errors} errors)"
        )
        
        if not passed:
            category_failures.append(cat)
    
    # Final verdict
    print("\n" + "═"*60)
    
    if create_baseline:
        print("📌 New baseline created from this run")
    
    if all_passed and regression_passed:
        print("✅ ALL TESTS PASSED")
        return 0
    elif report.regressions:
        print("⚠️  REGRESSIONS DETECTED - Pipeline should investigate")
        return 1
    else:
        print(f"❌ TESTS FAILED - Below minimum thresholds")
        if category_failures:
            print(f"   Failed categories: {', '.join(category_failures)}")
        return 2


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CI test runner for chat agent")
    parser.add_argument("--quick", "-q", action="store_true", help="Run quick subset of tests")
    parser.add_argument("--strict", "-s", action="store_true", help="Fail on any test failure")
    parser.add_argument("--create-baseline", "-b", action="store_true", help="Save this run as new baseline")
    parser.add_argument("--thresholds", type=str, help="JSON string with custom thresholds")
    
    args = parser.parse_args()
    
    # Override thresholds if provided
    if args.thresholds:
        try:
            custom = json.loads(args.thresholds)
            THRESHOLDS.update(custom)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse thresholds JSON: {args.thresholds}")
    
    print_banner()
    
    exit_code = asyncio.run(run_tests(
        quick=args.quick,
        strict=args.strict,
        create_baseline=args.create_baseline
    ))
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
