"""
Evaluation Module for BioPipelines
==================================

Comprehensive testing and evaluation of chat agent capabilities.

Modules:
- conversation_test_suite: Basic conversation evaluation
- enhanced_metrics: Advanced metrics (LLM-as-judge, semantic similarity)
- historical_tracker: SQLite-based historical trend tracking
- synthetic_test_generator: Synthetic test case generation
- adversarial_tests: Security and robustness testing
"""

from .conversation_test_suite import (
    ConversationEvaluator,
    TestConversation,
    EvaluationReport,
    get_test_conversations,
    print_report,
    save_report,
)

# Enhanced metrics
try:
    from .enhanced_metrics import (
        IntentAccuracyMetric,
        EntityF1Metric,
        ToolAccuracyMetric,
        LLMResponseQualityMetric,
        SemanticSimilarityMetric,
        EnhancedEvaluator,
    )
except ImportError:
    IntentAccuracyMetric = None
    EntityF1Metric = None
    ToolAccuracyMetric = None
    LLMResponseQualityMetric = None
    SemanticSimilarityMetric = None
    EnhancedEvaluator = None

# Historical tracking
try:
    from .historical_tracker import HistoricalTracker
except ImportError:
    HistoricalTracker = None

# Synthetic test generation
try:
    from .synthetic_test_generator import (
        TemplateGenerator,
        LLMGenerator,
        ConversationGenerator,
        DataAugmentor,
        ChallengeGenerator,
    )
except ImportError:
    TemplateGenerator = None
    LLMGenerator = None
    ConversationGenerator = None
    DataAugmentor = None
    ChallengeGenerator = None

# Adversarial testing
try:
    from .adversarial_tests import (
        AdversarialTest,
        AdversarialTestRunner,
        ALL_ADVERSARIAL_TESTS,
        get_tests_by_category,
        get_tests_by_risk,
        get_high_priority_tests,
    )
except ImportError:
    AdversarialTest = None
    AdversarialTestRunner = None
    ALL_ADVERSARIAL_TESTS = []
    get_tests_by_category = None
    get_tests_by_risk = None
    get_high_priority_tests = None

__all__ = [
    # Core evaluation
    "ConversationEvaluator",
    "TestConversation", 
    "EvaluationReport",
    "get_test_conversations",
    "print_report",
    "save_report",
    # Enhanced metrics
    "IntentAccuracyMetric",
    "EntityF1Metric",
    "ToolAccuracyMetric",
    "LLMResponseQualityMetric",
    "SemanticSimilarityMetric",
    "EnhancedEvaluator",
    # Historical tracking
    "HistoricalTracker",
    # Synthetic generation
    "TemplateGenerator",
    "LLMGenerator",
    "ConversationGenerator",
    "DataAugmentor",
    "ChallengeGenerator",
    # Adversarial testing
    "AdversarialTest",
    "AdversarialTestRunner",
    "ALL_ADVERSARIAL_TESTS",
    "get_tests_by_category",
    "get_tests_by_risk",
    "get_high_priority_tests",
]
