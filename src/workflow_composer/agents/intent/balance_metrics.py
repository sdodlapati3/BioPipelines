"""
Intent Balance Metrics
======================

Monitor training data quality and detect class imbalances that could
lead to biased intent classification.

Features:
- Training example counts per intent
- Imbalance warnings
- Diversity scores
- Coverage analysis
- Automated recommendations

Usage:
    analyzer = TrainingDataAnalyzer(training_data)
    
    # Get balance report
    report = analyzer.analyze_balance()
    
    for warning in report.warnings:
        print(f"⚠️  {warning}")
"""

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Thresholds and Constants
# =============================================================================

# Minimum examples recommended per intent
MIN_EXAMPLES_PER_INTENT = 10

# Maximum ratio between largest and smallest intent example counts
MAX_IMBALANCE_RATIO = 5.0

# Minimum unique words to consider diverse
MIN_VOCABULARY_PER_INTENT = 20

# Minimum example length (words)
MIN_EXAMPLE_LENGTH = 3

# Maximum overlap between intent vocabularies (indicates confusion risk)
MAX_VOCABULARY_OVERLAP = 0.3


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class IntentStats:
    """Statistics for a single intent."""
    intent: str
    example_count: int
    unique_words: int
    avg_length: float
    min_length: int
    max_length: int
    vocabulary: Set[str] = field(default_factory=set)
    duplicate_count: int = 0
    
    @property
    def is_underrepresented(self) -> bool:
        return self.example_count < MIN_EXAMPLES_PER_INTENT
    
    @property
    def has_low_diversity(self) -> bool:
        return self.unique_words < MIN_VOCABULARY_PER_INTENT


@dataclass
class BalanceReport:
    """Complete training data balance report."""
    total_intents: int
    total_examples: int
    min_examples: int
    max_examples: int
    imbalance_ratio: float
    
    # Per-intent stats
    intent_stats: Dict[str, IntentStats] = field(default_factory=dict)
    
    # Issues found
    warnings: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # Overlap analysis
    high_overlap_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    
    # Quality scores
    overall_score: float = 0.0
    balance_score: float = 0.0
    diversity_score: float = 0.0
    coverage_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_intents": self.total_intents,
            "total_examples": self.total_examples,
            "min_examples": self.min_examples,
            "max_examples": self.max_examples,
            "imbalance_ratio": self.imbalance_ratio,
            "intent_stats": {
                k: {
                    "example_count": v.example_count,
                    "unique_words": v.unique_words,
                    "avg_length": v.avg_length,
                    "duplicate_count": v.duplicate_count,
                }
                for k, v in self.intent_stats.items()
            },
            "warnings": self.warnings,
            "critical_issues": self.critical_issues,
            "suggestions": self.suggestions,
            "high_overlap_pairs": [
                {"intent1": p[0], "intent2": p[1], "overlap": p[2]}
                for p in self.high_overlap_pairs
            ],
            "scores": {
                "overall": self.overall_score,
                "balance": self.balance_score,
                "diversity": self.diversity_score,
                "coverage": self.coverage_score,
            }
        }
    
    def print_report(self) -> None:
        """Print a formatted report to console."""
        print("\n" + "=" * 60)
        print("  TRAINING DATA BALANCE REPORT")
        print("=" * 60)
        
        print(f"\n📊 Overview:")
        print(f"   Total Intents:  {self.total_intents}")
        print(f"   Total Examples: {self.total_examples}")
        print(f"   Min Examples:   {self.min_examples}")
        print(f"   Max Examples:   {self.max_examples}")
        print(f"   Imbalance:      {self.imbalance_ratio:.2f}x")
        
        print(f"\n📈 Quality Scores:")
        print(f"   Overall:    {self.overall_score:.0%}")
        print(f"   Balance:    {self.balance_score:.0%}")
        print(f"   Diversity:  {self.diversity_score:.0%}")
        print(f"   Coverage:   {self.coverage_score:.0%}")
        
        if self.critical_issues:
            print(f"\n🚨 Critical Issues ({len(self.critical_issues)}):")
            for issue in self.critical_issues:
                print(f"   • {issue}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings[:5]:  # Limit to 5
                print(f"   • {warning}")
            if len(self.warnings) > 5:
                print(f"   ... and {len(self.warnings) - 5} more")
        
        if self.suggestions:
            print(f"\n💡 Suggestions:")
            for suggestion in self.suggestions[:5]:
                print(f"   • {suggestion}")
        
        if self.high_overlap_pairs:
            print(f"\n⚡ High Vocabulary Overlap (confusion risk):")
            for intent1, intent2, overlap in self.high_overlap_pairs[:3]:
                print(f"   • {intent1} ↔ {intent2}: {overlap:.0%} overlap")
        
        print("\n" + "=" * 60)


# =============================================================================
# Training Data Analyzer
# =============================================================================

class TrainingDataAnalyzer:
    """
    Analyze training data for balance, diversity, and quality issues.
    """
    
    def __init__(
        self,
        training_data: Optional["NLUTrainingData"] = None,
        examples: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize the analyzer.
        
        Args:
            training_data: NLUTrainingData instance
            examples: Alternative: dict mapping intent -> list of example strings
        """
        self.training_data = training_data
        self._examples = examples or {}
        
        # Load from training data if provided
        if training_data is not None:
            self._load_from_training_data()
    
    def _load_from_training_data(self) -> None:
        """Load examples from NLUTrainingData."""
        # Import here to avoid circular dependency
        from .training_data import NLUTrainingData
        
        if self.training_data is None:
            return
        
        for intent in self.training_data.intent_examples:
            examples = self.training_data.get_intent_examples(intent)
            self._examples[intent] = [ex.text for ex in examples]
    
    def set_examples(self, intent: str, examples: List[str]) -> None:
        """Manually set examples for an intent."""
        self._examples[intent] = examples
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        return words
    
    def _compute_intent_stats(self, intent: str, examples: List[str]) -> IntentStats:
        """Compute statistics for a single intent."""
        if not examples:
            return IntentStats(
                intent=intent,
                example_count=0,
                unique_words=0,
                avg_length=0,
                min_length=0,
                max_length=0,
            )
        
        # Tokenize all examples
        all_tokens: List[List[str]] = []
        vocabulary: Set[str] = set()
        lengths: List[int] = []
        
        seen_examples: Set[str] = set()
        duplicate_count = 0
        
        for example in examples:
            # Check duplicates
            normalized = example.lower().strip()
            if normalized in seen_examples:
                duplicate_count += 1
            else:
                seen_examples.add(normalized)
            
            tokens = self._tokenize(example)
            all_tokens.append(tokens)
            vocabulary.update(tokens)
            lengths.append(len(tokens))
        
        return IntentStats(
            intent=intent,
            example_count=len(examples),
            unique_words=len(vocabulary),
            avg_length=sum(lengths) / len(lengths) if lengths else 0,
            min_length=min(lengths) if lengths else 0,
            max_length=max(lengths) if lengths else 0,
            vocabulary=vocabulary,
            duplicate_count=duplicate_count,
        )
    
    def _compute_vocabulary_overlap(
        self, 
        stats1: IntentStats, 
        stats2: IntentStats
    ) -> float:
        """Compute vocabulary overlap between two intents (Jaccard similarity)."""
        if not stats1.vocabulary or not stats2.vocabulary:
            return 0.0
        
        intersection = stats1.vocabulary & stats2.vocabulary
        union = stats1.vocabulary | stats2.vocabulary
        
        return len(intersection) / len(union) if union else 0.0
    
    def analyze_balance(self) -> BalanceReport:
        """
        Perform comprehensive balance analysis.
        
        Returns:
            BalanceReport with all metrics and issues
        """
        if not self._examples:
            return BalanceReport(
                total_intents=0,
                total_examples=0,
                min_examples=0,
                max_examples=0,
                imbalance_ratio=0.0,
                critical_issues=["No training data loaded"],
            )
        
        # Compute per-intent stats
        intent_stats: Dict[str, IntentStats] = {}
        for intent, examples in self._examples.items():
            intent_stats[intent] = self._compute_intent_stats(intent, examples)
        
        # Global metrics
        example_counts = [s.example_count for s in intent_stats.values()]
        total_examples = sum(example_counts)
        min_examples = min(example_counts) if example_counts else 0
        max_examples = max(example_counts) if example_counts else 0
        
        imbalance_ratio = max_examples / min_examples if min_examples > 0 else float('inf')
        
        # Initialize report
        report = BalanceReport(
            total_intents=len(intent_stats),
            total_examples=total_examples,
            min_examples=min_examples,
            max_examples=max_examples,
            imbalance_ratio=imbalance_ratio,
            intent_stats=intent_stats,
        )
        
        # Check for issues
        self._check_issues(report)
        
        # Compute vocabulary overlaps
        self._check_overlaps(report)
        
        # Compute quality scores
        self._compute_scores(report)
        
        # Generate suggestions
        self._generate_suggestions(report)
        
        return report
    
    def _check_issues(self, report: BalanceReport) -> None:
        """Check for balance and quality issues."""
        
        # Check for empty intents
        for intent, stats in report.intent_stats.items():
            if stats.example_count == 0:
                report.critical_issues.append(
                    f"Intent '{intent}' has no training examples"
                )
        
        # Check underrepresented intents
        for intent, stats in report.intent_stats.items():
            if stats.is_underrepresented and stats.example_count > 0:
                report.warnings.append(
                    f"Intent '{intent}' has only {stats.example_count} examples "
                    f"(recommended: {MIN_EXAMPLES_PER_INTENT}+)"
                )
        
        # Check low diversity
        for intent, stats in report.intent_stats.items():
            if stats.has_low_diversity and stats.example_count >= 5:
                report.warnings.append(
                    f"Intent '{intent}' has low vocabulary diversity "
                    f"({stats.unique_words} unique words)"
                )
        
        # Check overall imbalance
        if report.imbalance_ratio > MAX_IMBALANCE_RATIO:
            report.critical_issues.append(
                f"Severe class imbalance: {report.imbalance_ratio:.1f}x ratio "
                f"between largest ({report.max_examples}) and smallest ({report.min_examples}) intents"
            )
        
        # Check for duplicates
        for intent, stats in report.intent_stats.items():
            if stats.duplicate_count > 0:
                report.warnings.append(
                    f"Intent '{intent}' has {stats.duplicate_count} duplicate examples"
                )
        
        # Check for very short examples
        for intent, stats in report.intent_stats.items():
            if stats.min_length < MIN_EXAMPLE_LENGTH and stats.example_count > 0:
                report.warnings.append(
                    f"Intent '{intent}' has very short examples (min {stats.min_length} words)"
                )
    
    def _check_overlaps(self, report: BalanceReport) -> None:
        """Check vocabulary overlaps between intent pairs."""
        intents = list(report.intent_stats.keys())
        
        for i, intent1 in enumerate(intents):
            for intent2 in intents[i+1:]:
                stats1 = report.intent_stats[intent1]
                stats2 = report.intent_stats[intent2]
                
                overlap = self._compute_vocabulary_overlap(stats1, stats2)
                
                if overlap > MAX_VOCABULARY_OVERLAP:
                    report.high_overlap_pairs.append((intent1, intent2, overlap))
        
        # Sort by overlap descending
        report.high_overlap_pairs.sort(key=lambda x: -x[2])
        
        # Add warnings for very high overlaps
        for intent1, intent2, overlap in report.high_overlap_pairs:
            if overlap > 0.5:
                report.warnings.append(
                    f"High vocabulary overlap ({overlap:.0%}) between "
                    f"'{intent1}' and '{intent2}' - may cause confusion"
                )
    
    def _compute_scores(self, report: BalanceReport) -> None:
        """Compute quality scores."""
        
        if report.total_intents == 0:
            return
        
        # Balance score: penalize imbalance
        # Perfect = 1.0, infinite imbalance = 0.0
        balance_score = 1.0 / (1.0 + math.log(report.imbalance_ratio)) if report.imbalance_ratio > 0 else 0
        
        # Diversity score: average vocabulary size relative to example count
        diversity_scores = []
        for stats in report.intent_stats.values():
            if stats.example_count > 0:
                # More unique words per example = more diverse
                diversity = min(1.0, stats.unique_words / (stats.example_count * 3))
                diversity_scores.append(diversity)
        diversity_score = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0
        
        # Coverage score: what fraction of intents have sufficient examples
        sufficient_intents = sum(
            1 for s in report.intent_stats.values()
            if s.example_count >= MIN_EXAMPLES_PER_INTENT
        )
        coverage_score = sufficient_intents / report.total_intents
        
        # Overall score: weighted average minus penalties
        base_score = (
            balance_score * 0.35 +
            diversity_score * 0.30 +
            coverage_score * 0.35
        )
        
        # Penalties
        penalty = 0.0
        penalty += len(report.critical_issues) * 0.15
        penalty += min(len(report.warnings) * 0.02, 0.20)
        
        overall_score = max(0.0, base_score - penalty)
        
        report.balance_score = balance_score
        report.diversity_score = diversity_score
        report.coverage_score = coverage_score
        report.overall_score = overall_score
    
    def _generate_suggestions(self, report: BalanceReport) -> None:
        """Generate improvement suggestions."""
        
        # Suggest adding examples to underrepresented intents
        underrep = [
            (intent, stats.example_count)
            for intent, stats in report.intent_stats.items()
            if stats.is_underrepresented and stats.example_count > 0
        ]
        if underrep:
            underrep.sort(key=lambda x: x[1])
            worst_intent, count = underrep[0]
            needed = MIN_EXAMPLES_PER_INTENT - count
            report.suggestions.append(
                f"Add {needed} more examples to '{worst_intent}' "
                f"(currently only {count})"
            )
        
        # Suggest downsampling overrepresented intents
        if report.imbalance_ratio > MAX_IMBALANCE_RATIO:
            max_intent = max(
                report.intent_stats.items(),
                key=lambda x: x[1].example_count
            )[0]
            report.suggestions.append(
                f"Consider downsampling '{max_intent}' or adding examples "
                f"to other intents to reduce {report.imbalance_ratio:.1f}x imbalance"
            )
        
        # Suggest removing duplicates
        total_duplicates = sum(s.duplicate_count for s in report.intent_stats.values())
        if total_duplicates > 0:
            report.suggestions.append(
                f"Remove {total_duplicates} duplicate training examples"
            )
        
        # Suggest reviewing high-overlap pairs
        if report.high_overlap_pairs:
            intent1, intent2, _ = report.high_overlap_pairs[0]
            report.suggestions.append(
                f"Review '{intent1}' and '{intent2}' for potential intent consolidation "
                f"or add more distinctive examples"
            )
        
        # Suggest diversifying examples
        low_diversity = [
            intent for intent, stats in report.intent_stats.items()
            if stats.has_low_diversity and stats.example_count >= 5
        ]
        if low_diversity:
            report.suggestions.append(
                f"Add more varied phrasing to intents: {', '.join(low_diversity[:3])}"
            )


# =============================================================================
# Quick Analysis Function
# =============================================================================

def analyze_training_balance(
    examples: Optional[Dict[str, List[str]]] = None,
    training_data: Optional["NLUTrainingData"] = None,
    print_report: bool = True,
) -> BalanceReport:
    """
    Quick function to analyze training data balance.
    
    Args:
        examples: Dict mapping intent -> list of example strings
        training_data: NLUTrainingData instance
        print_report: Whether to print the report
        
    Returns:
        BalanceReport
    """
    analyzer = TrainingDataAnalyzer(
        training_data=training_data,
        examples=examples,
    )
    
    report = analyzer.analyze_balance()
    
    if print_report:
        report.print_report()
    
    return report
