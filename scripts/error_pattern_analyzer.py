#!/usr/bin/env python3
"""
Error Pattern Analyzer for BioPipelines Chat Agent

Analyzes test failures to identify common patterns, categorize errors,
and suggest fixes for improving the chat agent.

Usage:
    python scripts/error_pattern_analyzer.py                           # Analyze latest results
    python scripts/error_pattern_analyzer.py --results path/to/file.json
    python scripts/error_pattern_analyzer.py --suggest-fixes           # Generate fix suggestions
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class ErrorPattern:
    """Represents a detected error pattern."""
    pattern_id: str
    pattern_type: str  # intent_mismatch, missing_entity, wrong_tool, exception, etc.
    description: str
    frequency: int
    affected_categories: list[str]
    example_queries: list[str]
    expected_vs_actual: list[dict]
    suggested_fix: Optional[str] = None
    fix_location: Optional[str] = None
    priority: str = "medium"  # high, medium, low


@dataclass
class AnalysisReport:
    """Complete error analysis report."""
    timestamp: str
    total_failures: int
    total_tests: int
    failure_rate: float
    patterns: list[ErrorPattern] = field(default_factory=list)
    category_breakdown: dict = field(default_factory=dict)
    root_causes: list[str] = field(default_factory=list)
    recommendations: list[dict] = field(default_factory=list)


class ErrorPatternAnalyzer:
    """Analyzes evaluation failures to identify patterns and suggest fixes."""
    
    # Known intent mapping issues
    INTENT_ALIASES = {
        "DATA_SEARCH": ["DATA_DISCOVERY", "SEARCH_DATA", "FIND_DATA"],
        "DATA_DESCRIBE": ["DESCRIBE_DATA", "DATA_INFO", "DATA_DETAILS"],
        "WORKFLOW_GENERATE": ["GENERATE_WORKFLOW", "CREATE_WORKFLOW", "BUILD_WORKFLOW"],
        "JOB_STATUS": ["CHECK_STATUS", "STATUS_CHECK", "GET_STATUS"],
    }
    
    # Pattern detection rules
    PATTERN_RULES = {
        "missing_entity_pattern": {
            "description": "Entity extraction failed due to missing regex pattern",
            "indicators": ["expected_entities not empty but actual_entities empty"],
            "fix_location": "agents/intent/parser.py",
        },
        "intent_confusion": {
            "description": "Similar intents being confused (e.g., DATA_SEARCH vs DATA_DESCRIBE)",
            "indicators": ["intent close but not exact match"],
            "fix_location": "agents/intent/parser.py",
        },
        "context_not_retained": {
            "description": "Multi-turn context not being passed correctly",
            "indicators": ["requires_context=True but context empty"],
            "fix_location": "agents/unified_agent.py",
        },
        "llm_fallback_failure": {
            "description": "Pattern matching and LLM fallback both failed",
            "indicators": ["llm_invoked=True but still wrong"],
            "fix_location": "agents/intent/semantic.py",
        },
        "tool_selection_error": {
            "description": "Correct intent but wrong tool selected",
            "indicators": ["intent_correct=True but tool_correct=False"],
            "fix_location": "config/tool_mappings.yaml",
        },
        "edge_case_failure": {
            "description": "Edge cases like typos, special chars not handled",
            "indicators": ["category=edge_cases"],
            "fix_location": "agents/intent/parser.py",
        },
    }
    
    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = results_dir or project_root / "reports" / "evaluation"
        
    def load_latest_results(self) -> Optional[dict]:
        """Load the most recent evaluation results."""
        if not self.results_dir.exists():
            return None
            
        result_files = list(self.results_dir.glob("evaluation_*.json"))
        if not result_files:
            return None
            
        latest = max(result_files, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            return json.load(f)
    
    def load_results(self, path: Path) -> Optional[dict]:
        """Load results from a specific file."""
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)
    
    def extract_failures(self, results: dict) -> list[dict]:
        """Extract all failed test cases."""
        failures = []
        for test in results.get("test_results", []):
            if not test.get("intent_correct", True) or test.get("error"):
                failures.append(test)
        return failures
    
    def detect_intent_confusion_pattern(self, failures: list[dict]) -> Optional[ErrorPattern]:
        """Detect cases where similar intents are being confused."""
        intent_pairs = defaultdict(list)
        
        for failure in failures:
            expected = failure.get("expected_intent", "").upper()
            actual = failure.get("actual_intent", "").upper()
            
            if expected and actual and expected != actual:
                # Check if they're related intents
                pair = tuple(sorted([expected, actual]))
                intent_pairs[pair].append(failure)
        
        if not intent_pairs:
            return None
        
        # Find most common confusion
        most_common = max(intent_pairs.items(), key=lambda x: len(x[1]))
        pair, cases = most_common
        
        if len(cases) >= 2:
            return ErrorPattern(
                pattern_id="intent_confusion_001",
                pattern_type="intent_confusion",
                description=f"Intents {pair[0]} and {pair[1]} are frequently confused ({len(cases)} cases)",
                frequency=len(cases),
                affected_categories=list(set(c.get("category", "unknown") for c in cases)),
                example_queries=[c.get("query", "")[:80] for c in cases[:5]],
                expected_vs_actual=[
                    {"expected": c.get("expected_intent"), "actual": c.get("actual_intent")}
                    for c in cases[:5]
                ],
                suggested_fix=f"Add more distinctive patterns for {pair[0]} vs {pair[1]} in parser.py",
                fix_location="agents/intent/parser.py",
                priority="high" if len(cases) >= 5 else "medium"
            )
        
        return None
    
    def detect_missing_entity_pattern(self, failures: list[dict]) -> Optional[ErrorPattern]:
        """Detect cases where entities are expected but not extracted."""
        missing_entity_cases = []
        
        for failure in failures:
            expected = failure.get("expected_entities", {})
            actual = failure.get("actual_entities", {})
            
            if expected and not actual:
                missing_entity_cases.append(failure)
            elif expected:
                # Check for partial extraction
                missing_keys = set(expected.keys()) - set(actual.keys())
                if missing_keys:
                    failure["missing_entity_keys"] = list(missing_keys)
                    missing_entity_cases.append(failure)
        
        if len(missing_entity_cases) >= 2:
            # Analyze which entity types are most commonly missed
            missed_types = defaultdict(int)
            for case in missing_entity_cases:
                for key in case.get("missing_entity_keys", case.get("expected_entities", {}).keys()):
                    missed_types[key] += 1
            
            most_missed = max(missed_types.items(), key=lambda x: x[1]) if missed_types else ("unknown", 0)
            
            return ErrorPattern(
                pattern_id="missing_entity_001",
                pattern_type="missing_entity",
                description=f"Entity extraction failing - most missed: {most_missed[0]} ({most_missed[1]} times)",
                frequency=len(missing_entity_cases),
                affected_categories=list(set(c.get("category", "unknown") for c in missing_entity_cases)),
                example_queries=[c.get("query", "")[:80] for c in missing_entity_cases[:5]],
                expected_vs_actual=[
                    {"expected": c.get("expected_entities"), "actual": c.get("actual_entities")}
                    for c in missing_entity_cases[:5]
                ],
                suggested_fix=f"Add regex patterns to extract '{most_missed[0]}' entities in parser.py",
                fix_location="agents/intent/parser.py",
                priority="high"
            )
        
        return None
    
    def detect_edge_case_failures(self, failures: list[dict]) -> Optional[ErrorPattern]:
        """Detect failures specific to edge cases."""
        edge_cases = [f for f in failures if f.get("category") == "edge_cases"]
        
        if len(edge_cases) >= 2:
            # Analyze edge case types
            edge_types = defaultdict(list)
            for case in edge_cases:
                query = case.get("query", "")
                
                # Detect edge case type
                if re.search(r'[A-Z]{3,}', query):
                    edge_types["unusual_caps"].append(case)
                if re.search(r'\d{3,}', query) or re.search(r'[!@#$%^&*()]+', query):
                    edge_types["special_chars"].append(case)
                if re.search(r'(?i)(pls|plz|thx|gonna|wanna)', query):
                    edge_types["informal_language"].append(case)
                if len(query.split()) <= 2:
                    edge_types["very_short"].append(case)
            
            most_common = max(edge_types.items(), key=lambda x: len(x[1])) if edge_types else ("general", edge_cases)
            
            return ErrorPattern(
                pattern_id="edge_case_001",
                pattern_type="edge_case",
                description=f"Edge case failures - most common type: {most_common[0]} ({len(most_common[1])} cases)",
                frequency=len(edge_cases),
                affected_categories=["edge_cases"],
                example_queries=[c.get("query", "")[:80] for c in edge_cases[:5]],
                expected_vs_actual=[
                    {"expected": c.get("expected_intent"), "actual": c.get("actual_intent")}
                    for c in edge_cases[:5]
                ],
                suggested_fix=f"Add normalization for {most_common[0]} cases before pattern matching",
                fix_location="agents/intent/parser.py",
                priority="medium"
            )
        
        return None
    
    def detect_context_failures(self, failures: list[dict]) -> Optional[ErrorPattern]:
        """Detect failures in multi-turn context handling."""
        context_failures = [
            f for f in failures 
            if f.get("category") in ["multi_turn", "coreference", "context_dependent"]
        ]
        
        if len(context_failures) >= 2:
            return ErrorPattern(
                pattern_id="context_001",
                pattern_type="context_not_retained",
                description=f"Multi-turn context not being handled correctly ({len(context_failures)} failures)",
                frequency=len(context_failures),
                affected_categories=list(set(c.get("category", "unknown") for c in context_failures)),
                example_queries=[c.get("query", "")[:80] for c in context_failures[:5]],
                expected_vs_actual=[
                    {"expected": c.get("expected_intent"), "actual": c.get("actual_intent")}
                    for c in context_failures[:5]
                ],
                suggested_fix="Implement proper context passing between turns in conversation handler",
                fix_location="agents/unified_agent.py",
                priority="high"
            )
        
        return None
    
    def detect_exception_pattern(self, failures: list[dict]) -> Optional[ErrorPattern]:
        """Detect failures due to exceptions."""
        exception_cases = [f for f in failures if f.get("error") and f["error"] != "-"]
        
        if len(exception_cases) >= 2:
            # Group by error type
            error_types = defaultdict(list)
            for case in exception_cases:
                error = case.get("error", "")
                # Extract error type (first word or exception name)
                error_type = error.split(":")[0].split()[-1] if error else "Unknown"
                error_types[error_type].append(case)
            
            most_common = max(error_types.items(), key=lambda x: len(x[1]))
            
            return ErrorPattern(
                pattern_id="exception_001",
                pattern_type="exception",
                description=f"Exceptions during processing - most common: {most_common[0]} ({len(most_common[1])} cases)",
                frequency=len(exception_cases),
                affected_categories=list(set(c.get("category", "unknown") for c in exception_cases)),
                example_queries=[c.get("query", "")[:80] for c in exception_cases[:5]],
                expected_vs_actual=[
                    {"query": c.get("query", "")[:50], "error": c.get("error", "")}
                    for c in exception_cases[:5]
                ],
                suggested_fix=f"Add error handling for {most_common[0]} exceptions",
                fix_location="agents/unified_agent.py",
                priority="high"
            )
        
        return None
    
    def analyze(self, results: Optional[dict] = None) -> AnalysisReport:
        """Run full error pattern analysis."""
        if results is None:
            results = self.load_latest_results()
        
        if not results:
            return AnalysisReport(
                timestamp=datetime.now().isoformat(),
                total_failures=0,
                total_tests=0,
                failure_rate=0.0
            )
        
        failures = self.extract_failures(results)
        total_tests = results.get("total_tests", len(results.get("test_results", [])))
        
        # Detect patterns
        patterns = []
        
        intent_pattern = self.detect_intent_confusion_pattern(failures)
        if intent_pattern:
            patterns.append(intent_pattern)
        
        entity_pattern = self.detect_missing_entity_pattern(failures)
        if entity_pattern:
            patterns.append(entity_pattern)
        
        edge_pattern = self.detect_edge_case_failures(failures)
        if edge_pattern:
            patterns.append(edge_pattern)
        
        context_pattern = self.detect_context_failures(failures)
        if context_pattern:
            patterns.append(context_pattern)
        
        exception_pattern = self.detect_exception_pattern(failures)
        if exception_pattern:
            patterns.append(exception_pattern)
        
        # Calculate category breakdown
        category_breakdown = defaultdict(lambda: {"failures": 0, "total": 0})
        for test in results.get("test_results", []):
            category = test.get("category", "unknown")
            category_breakdown[category]["total"] += 1
            if not test.get("intent_correct", True) or test.get("error"):
                category_breakdown[category]["failures"] += 1
        
        # Calculate failure rates
        for cat, data in category_breakdown.items():
            data["failure_rate"] = data["failures"] / data["total"] if data["total"] > 0 else 0.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(patterns, category_breakdown)
        
        # Identify root causes
        root_causes = self._identify_root_causes(patterns)
        
        return AnalysisReport(
            timestamp=datetime.now().isoformat(),
            total_failures=len(failures),
            total_tests=total_tests,
            failure_rate=len(failures) / total_tests if total_tests > 0 else 0.0,
            patterns=patterns,
            category_breakdown=dict(category_breakdown),
            root_causes=root_causes,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self,
        patterns: list[ErrorPattern],
        category_breakdown: dict
    ) -> list[dict]:
        """Generate prioritized recommendations based on patterns."""
        recommendations = []
        
        # Sort patterns by priority and frequency
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_patterns = sorted(
            patterns,
            key=lambda p: (priority_order.get(p.priority, 2), -p.frequency)
        )
        
        for i, pattern in enumerate(sorted_patterns[:5], 1):
            recommendations.append({
                "rank": i,
                "action": pattern.suggested_fix,
                "location": pattern.fix_location,
                "impact": f"Would fix ~{pattern.frequency} test failures",
                "priority": pattern.priority,
                "pattern_type": pattern.pattern_type
            })
        
        # Add category-specific recommendations
        worst_categories = sorted(
            category_breakdown.items(),
            key=lambda x: x[1].get("failure_rate", 0),
            reverse=True
        )[:3]
        
        for cat, data in worst_categories:
            if data.get("failure_rate", 0) > 0.3:  # More than 30% failure rate
                recommendations.append({
                    "rank": len(recommendations) + 1,
                    "action": f"Focus on improving {cat} category - {data['failure_rate']:.0%} failure rate",
                    "location": f"tests/evaluation/comprehensive_test_data.py - {cat} section",
                    "impact": f"{data['failures']} failures out of {data['total']} tests",
                    "priority": "medium"
                })
        
        return recommendations
    
    def _identify_root_causes(self, patterns: list[ErrorPattern]) -> list[str]:
        """Identify underlying root causes from patterns."""
        root_causes = []
        
        pattern_types = [p.pattern_type for p in patterns]
        
        if "intent_confusion" in pattern_types:
            root_causes.append(
                "Intent patterns need more distinctive keywords to differentiate similar intents"
            )
        
        if "missing_entity" in pattern_types:
            root_causes.append(
                "Entity extraction regex patterns are incomplete or too restrictive"
            )
        
        if "context_not_retained" in pattern_types:
            root_causes.append(
                "Conversation context is not being properly maintained between turns"
            )
        
        if "edge_case" in pattern_types:
            root_causes.append(
                "Input normalization is not handling non-standard text formats"
            )
        
        if "exception" in pattern_types:
            root_causes.append(
                "Error handling is incomplete - some input combinations cause crashes"
            )
        
        return root_causes
    
    def generate_report(self, analysis: AnalysisReport) -> str:
        """Generate a markdown report from the analysis."""
        lines = [
            "# Error Pattern Analysis Report",
            f"\n**Generated:** {analysis.timestamp}",
            f"\n**Total Tests:** {analysis.total_tests}",
            f"**Total Failures:** {analysis.total_failures}",
            f"**Overall Failure Rate:** {analysis.failure_rate:.1%}",
            "\n---\n",
        ]
        
        # Root Causes
        if analysis.root_causes:
            lines.append("## Root Causes Identified\n")
            for cause in analysis.root_causes:
                lines.append(f"- {cause}")
            lines.append("")
        
        # Patterns
        if analysis.patterns:
            lines.append("\n## Detected Error Patterns\n")
            for pattern in analysis.patterns:
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(pattern.priority, "⚪")
                lines.append(f"### {priority_emoji} {pattern.pattern_type.replace('_', ' ').title()}")
                lines.append(f"**Pattern ID:** {pattern.pattern_id}")
                lines.append(f"**Frequency:** {pattern.frequency} occurrences")
                lines.append(f"**Description:** {pattern.description}")
                lines.append(f"**Affected Categories:** {', '.join(pattern.affected_categories)}")
                lines.append("")
                lines.append("**Example Queries:**")
                for query in pattern.example_queries[:3]:
                    lines.append(f"- `{query}`")
                lines.append("")
                if pattern.suggested_fix:
                    lines.append(f"**Suggested Fix:** {pattern.suggested_fix}")
                    lines.append(f"**Fix Location:** `{pattern.fix_location}`")
                lines.append("\n---\n")
        
        # Category Breakdown
        lines.append("\n## Failure Rate by Category\n")
        lines.append("| Category | Failures | Total | Failure Rate |")
        lines.append("|----------|----------|-------|--------------|")
        for cat, data in sorted(
            analysis.category_breakdown.items(),
            key=lambda x: x[1].get("failure_rate", 0),
            reverse=True
        ):
            rate = data.get("failure_rate", 0)
            emoji = "🔴" if rate > 0.3 else "🟡" if rate > 0.1 else "🟢"
            lines.append(
                f"| {emoji} {cat} | {data['failures']} | {data['total']} | {rate:.0%} |"
            )
        
        # Recommendations
        if analysis.recommendations:
            lines.append("\n## Prioritized Recommendations\n")
            for rec in analysis.recommendations:
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec.get("priority", "medium"), "⚪")
                lines.append(f"{rec['rank']}. {priority_emoji} **{rec['action']}**")
                lines.append(f"   - Location: `{rec['location']}`")
                lines.append(f"   - Impact: {rec['impact']}")
                lines.append("")
        
        return "\n".join(lines)
    
    def save_report(self, analysis: AnalysisReport, output_path: Optional[Path] = None):
        """Save the analysis report to file."""
        if output_path is None:
            output_path = self.results_dir / f"error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        report = self.generate_report(analysis)
        output_path.write_text(report)
        
        # Also save JSON version
        json_path = output_path.with_suffix(".json")
        with open(json_path, 'w') as f:
            json.dump({
                "timestamp": analysis.timestamp,
                "total_failures": analysis.total_failures,
                "total_tests": analysis.total_tests,
                "failure_rate": analysis.failure_rate,
                "patterns": [
                    {
                        "pattern_id": p.pattern_id,
                        "pattern_type": p.pattern_type,
                        "description": p.description,
                        "frequency": p.frequency,
                        "affected_categories": p.affected_categories,
                        "suggested_fix": p.suggested_fix,
                        "fix_location": p.fix_location,
                        "priority": p.priority
                    }
                    for p in analysis.patterns
                ],
                "category_breakdown": analysis.category_breakdown,
                "root_causes": analysis.root_causes,
                "recommendations": analysis.recommendations
            }, f, indent=2)
        
        return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze error patterns in evaluation results")
    parser.add_argument("--results", "-r", type=Path, help="Path to results JSON file")
    parser.add_argument("--output", "-o", type=Path, help="Output path for report")
    parser.add_argument("--suggest-fixes", action="store_true", help="Focus on generating fix suggestions")
    
    args = parser.parse_args()
    
    analyzer = ErrorPatternAnalyzer()
    
    if args.results:
        results = analyzer.load_results(args.results)
    else:
        results = analyzer.load_latest_results()
    
    if not results:
        print("No evaluation results found. Run the evaluation first:")
        print("  python scripts/unified_evaluation_runner.py")
        sys.exit(1)
    
    analysis = analyzer.analyze(results)
    
    # Print summary
    print("\n" + "="*60)
    print("ERROR PATTERN ANALYSIS")
    print("="*60)
    print(f"Total Failures: {analysis.total_failures}/{analysis.total_tests} ({analysis.failure_rate:.1%})")
    print(f"Patterns Found: {len(analysis.patterns)}")
    print("-"*60)
    
    if analysis.root_causes:
        print("\n🔍 ROOT CAUSES:")
        for cause in analysis.root_causes:
            print(f"   • {cause}")
    
    if analysis.patterns:
        print("\n⚠️ DETECTED PATTERNS:")
        for p in analysis.patterns:
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(p.priority, "⚪")
            print(f"   {priority_emoji} {p.pattern_type}: {p.frequency} occurrences")
    
    if analysis.recommendations:
        print("\n📋 TOP RECOMMENDATIONS:")
        for rec in analysis.recommendations[:5]:
            print(f"   {rec['rank']}. {rec['action']}")
            print(f"      → {rec['location']}")
    
    print("="*60)
    
    # Save report
    output_path = analyzer.save_report(analysis, args.output)
    print(f"\n📄 Full report saved to: {output_path}")


if __name__ == "__main__":
    main()
