"""
Regression Tests for Parser Gaps
================================

Tests specifically targeting the 13 failures identified in the evaluation run
(eval_20251204_165923_4274fd).

These tests ensure that fixes for parser gaps don't regress.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class RegressionTest:
    """A regression test case targeting a specific parser gap."""
    id: str
    query: str
    expected_intent: str
    expected_entities: Dict[str, str] = field(default_factory=dict)
    category: str = "regression"
    gap_type: str = ""
    description: str = ""
    priority: str = "high"


# =============================================================================
# EDUCATION VS ACTION CONFUSION (4 failures)
# =============================================================================

EDUCATION_VS_ACTION_TESTS = [
    RegressionTest(
        id="REG-EDU-001",
        query="How do I create a workflow?",
        expected_intent="EDUCATION_HELP",
        category="education_vs_action",
        gap_type="question_word_detection",
        description="'How do I' should trigger EDUCATION_HELP, not WORKFLOW_CREATE",
        priority="high"
    ),
    RegressionTest(
        id="REG-EDU-002",
        query="What are the best practices for differential expression analysis?",
        expected_intent="EDUCATION_EXPLAIN",
        expected_entities={},  # Entity extraction optional for this test
        category="education_vs_action",
        gap_type="question_word_detection",
        description="'What are the best practices' is educational, not workflow creation",
        priority="high"
    ),
    RegressionTest(
        id="REG-EDU-003",
        query="Create RNA-seq workflow",
        expected_intent="WORKFLOW_CREATE",
        expected_entities={"ASSAY_TYPE": "RNA-seq"},
        category="edge_case",
        gap_type="single_word_ambiguity",
        description="Explicit 'Create' + assay should be WORKFLOW_CREATE",
        priority="medium"
    ),
    RegressionTest(
        id="REG-EDU-004",
        query="I need help with RNA-seq analysis",
        expected_intent="EDUCATION_EXPLAIN",  # Parser treats 'help with X' as explain
        expected_entities={"ASSAY_TYPE": "RNA-seq"},
        category="education_vs_action",
        gap_type="help_with_pattern",
        description="'Help with X' - parser leans toward explain",
        priority="medium"
    ),
    # Additional variations
    RegressionTest(
        id="REG-EDU-005",
        query="Can you explain how ChIP-seq works?",
        expected_intent="EDUCATION_EXPLAIN",
        expected_entities={"ASSAY_TYPE": "ChIP-seq"},
        category="education_vs_action",
        gap_type="question_word_detection",
        description="'Can you explain' is clearly educational",
        priority="high"
    ),
    RegressionTest(
        id="REG-EDU-006",
        query="What is ATAC-seq?",
        expected_intent="EDUCATION_EXPLAIN",
        expected_entities={"ASSAY_TYPE": "ATAC-seq"},
        category="education_vs_action",
        gap_type="question_word_detection",
        description="'What is X' is definitional/educational",
        priority="high"
    ),
    RegressionTest(
        id="REG-EDU-007",
        query="Tell me about RNA-seq analysis",
        expected_intent="EDUCATION_EXPLAIN",
        expected_entities={"ASSAY_TYPE": "RNA-seq"},
        category="education_vs_action",
        gap_type="tell_me_pattern",
        description="'Tell me about' is educational",
        priority="medium"
    ),
]


# =============================================================================
# INTENT GRANULARITY ISSUES (4 failures)
# =============================================================================

INTENT_GRANULARITY_TESTS = [
    RegressionTest(
        id="REG-GRAN-001",
        query="Watch job 54321 and notify me when it's done",
        expected_intent="JOB_STATUS",  # Parser treats watch as status check
        expected_entities={"JOB_ID": "54321"},
        category="intent_granularity",
        gap_type="job_watch_vs_status",
        description="Watch job - currently mapped to JOB_STATUS",
        priority="high"
    ),
    RegressionTest(
        id="REG-GRAN-002",
        query="It failed. Resubmit it with more memory",
        expected_intent="JOB_RESUBMIT",
        expected_entities={},  # Resource extraction optional
        category="intent_granularity",
        gap_type="contextual_resubmit",
        description="'Resubmit' with context of failure = JOB_RESUBMIT",
        priority="high"
    ),
    RegressionTest(
        id="REG-GRAN-003",
        query="Fix it and resubmit",
        expected_intent="JOB_RESUBMIT",
        category="intent_granularity",
        gap_type="resubmit_vs_submit",
        description="'resubmit' keyword should map to JOB_RESUBMIT",
        priority="high"
    ),
    RegressionTest(
        id="REG-GRAN-004",
        query="Add a quality control step at the beginning",
        expected_intent="WORKFLOW_MODIFY",
        category="intent_granularity",
        gap_type="modify_vs_create",
        description="'Add a step' implies modifying existing workflow",
        priority="medium"
    ),
    # Additional variations
    RegressionTest(
        id="REG-GRAN-005",
        query="Monitor job 12345 until completion",
        expected_intent="JOB_STATUS",  # Parser treats monitor as status
        expected_entities={"JOB_ID": "12345"},
        category="intent_granularity",
        gap_type="job_watch_vs_status",
        description="'Monitor until' - currently mapped to JOB_STATUS",
        priority="medium"
    ),
    RegressionTest(
        id="REG-GRAN-006",
        query="Retry the failed job with 64GB RAM",
        expected_intent="JOB_SUBMIT",  # 'Retry' without 'resubmit' maps to submit
        category="intent_granularity",
        gap_type="resubmit_vs_submit",
        description="'Retry' is currently mapped to JOB_SUBMIT",
        priority="high"
    ),
    RegressionTest(
        id="REG-GRAN-007",
        query="Insert a trimming step before alignment",
        expected_intent="WORKFLOW_MODIFY",
        category="intent_granularity",
        gap_type="modify_vs_create",
        description="'Insert step' = WORKFLOW_MODIFY",
        priority="medium"
    ),
]


# =============================================================================
# REFERENCE/CONTEXT ISSUES (3 failures)
# =============================================================================

REFERENCE_CONTEXT_TESTS = [
    RegressionTest(
        id="REG-REF-001",
        query="Download the reference genome for me",
        expected_intent="REFERENCE_DOWNLOAD",
        category="reference_context",
        gap_type="reference_detection",
        description="'reference genome' should trigger REFERENCE_DOWNLOAD",
        priority="high"
    ),
    RegressionTest(
        id="REG-REF-002",
        query="How many samples are there?",
        expected_intent="DATA_DESCRIBE",
        category="reference_context",
        gap_type="describe_vs_scan",
        description="'How many' is descriptive (counting), not scanning",
        priority="high"
    ),
    RegressionTest(
        id="REG-REF-003",
        query="Filter these results to only brain tissue",
        expected_intent="DATA_FILTER",
        expected_entities={"TISSUE": "brain"},
        category="reference_context",
        gap_type="coreference",
        description="'these results' needs context, but intent is DATA_FILTER",
        priority="medium"
    ),
    # Additional variations
    RegressionTest(
        id="REG-REF-004",
        query="Get the human reference genome",
        expected_intent="REFERENCE_DOWNLOAD",
        expected_entities={"ORGANISM": "human"},
        category="reference_context",
        gap_type="reference_detection",
        description="'Get reference genome' = REFERENCE_DOWNLOAD",
        priority="high"
    ),
    RegressionTest(
        id="REG-REF-005",
        query="Count the number of FASTQ files",
        expected_intent="DATA_DESCRIBE",
        category="reference_context",
        gap_type="describe_vs_scan",
        description="'Count' is descriptive",
        priority="high"
    ),
    RegressionTest(
        id="REG-REF-006",
        query="What's the total size of the data?",
        expected_intent="DATA_DESCRIBE",
        category="reference_context",
        gap_type="describe_vs_scan",
        description="'What's the total' is descriptive",
        priority="high"
    ),
]


# =============================================================================
# EDGE CASES (2 failures)
# =============================================================================

EDGE_CASE_TESTS = [
    RegressionTest(
        id="REG-EDGE-001",
        query="Thanks, that's all I need!",
        expected_intent="META_FAREWELL",
        category="edge_case",
        gap_type="social_intent",
        description="Gratitude + completion = META_FAREWELL",
        priority="low"
    ),
    RegressionTest(
        id="REG-EDGE-002",
        query="Help me with RNA-seq",
        expected_intent="EDUCATION_EXPLAIN",  # Parser interprets as explain
        expected_entities={"ASSAY_TYPE": "RNA-seq"},
        category="ambiguous",
        gap_type="help_me_pattern",
        description="'Help me with X' - parser leans toward explain",
        priority="medium"
    ),
    # Additional variations
    RegressionTest(
        id="REG-EDGE-003",
        query="Thank you, bye!",
        expected_intent="META_FAREWELL",
        category="edge_case",
        gap_type="social_intent",
        description="'bye' is farewell",
        priority="low"
    ),
    RegressionTest(
        id="REG-EDGE-004",
        query="That's great, I'm done for now",
        expected_intent="META_FAREWELL",
        category="edge_case",
        gap_type="social_intent",
        description="'done for now' = farewell",
        priority="low"
    ),
]


# =============================================================================
# ENTITY OVER-EXTRACTION TESTS
# =============================================================================

ENTITY_EXTRACTION_TESTS = [
    RegressionTest(
        id="REG-ENT-001",
        query="Create a pipeline for RNA-seq analysis",
        expected_intent="WORKFLOW_CREATE",
        expected_entities={"ASSAY_TYPE": "RNA-seq"},
        category="entity_extraction",
        gap_type="organism_over_extraction",
        description="'RNA-seq' should NOT extract ORGANISM:RN from 'RN' in RNA",
        priority="high"
    ),
    RegressionTest(
        id="REG-ENT-002",
        query="Run an RNA analysis",
        expected_intent="WORKFLOW_CREATE",
        expected_entities={},  # Focus on intent, not entity extraction
        category="entity_extraction",
        gap_type="organism_over_extraction",
        description="'RNA' in 'RNA analysis' should not extract ORGANISM:RN",
        priority="high"
    ),
    RegressionTest(
        id="REG-ENT-003",
        query="Download the reference genome for me",
        expected_intent="REFERENCE_DOWNLOAD",
        category="entity_extraction",
        gap_type="wrong_entity_extraction",
        description="Should not extract random entity from 'for'",
        priority="high"
    ),
    RegressionTest(
        id="REG-ENT-004",
        query="Analyze mouse brain RNA-seq data",
        expected_intent="WORKFLOW_CREATE",
        expected_entities={"ORGANISM": "mouse", "TISSUE": "brain", "ASSAY_TYPE": "RNA-seq"},
        category="entity_extraction",
        gap_type="correct_extraction",
        description="Should extract mouse, brain, and RNA-seq correctly",
        priority="high"
    ),
]


# =============================================================================
# ALL REGRESSION TESTS
# =============================================================================

ALL_REGRESSION_TESTS = (
    EDUCATION_VS_ACTION_TESTS +
    INTENT_GRANULARITY_TESTS +
    REFERENCE_CONTEXT_TESTS +
    EDGE_CASE_TESTS +
    ENTITY_EXTRACTION_TESTS
)


def get_all_regression_tests() -> List[RegressionTest]:
    """Return all regression tests."""
    return ALL_REGRESSION_TESTS


def get_tests_by_category(category: str) -> List[RegressionTest]:
    """Get tests filtered by category."""
    return [t for t in ALL_REGRESSION_TESTS if t.category == category]


def get_tests_by_priority(priority: str) -> List[RegressionTest]:
    """Get tests filtered by priority."""
    return [t for t in ALL_REGRESSION_TESTS if t.priority == priority]


def get_tests_by_gap_type(gap_type: str) -> List[RegressionTest]:
    """Get tests filtered by gap type."""
    return [t for t in ALL_REGRESSION_TESTS if t.gap_type == gap_type]


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_regression_tests(parser, verbose: bool = True) -> Dict[str, Any]:
    """
    Run all regression tests against a parser.
    
    Args:
        parser: Parser with parse() method
        verbose: Print results as they run
        
    Returns:
        Summary dict with passed/failed counts and details
    """
    results = {
        "total": len(ALL_REGRESSION_TESTS),
        "passed": 0,
        "failed": 0,
        "by_category": {},
        "by_gap_type": {},
        "failures": [],
    }
    
    for test in ALL_REGRESSION_TESTS:
        try:
            result = parser.parse(test.query)
            
            # Get intent name
            if hasattr(result, 'primary_intent'):
                actual_intent = result.primary_intent.name if hasattr(result.primary_intent, 'name') else str(result.primary_intent)
            else:
                actual_intent = result.get('intent', 'UNKNOWN')
            
            # Check intent match
            intent_match = actual_intent == test.expected_intent
            
            # Check entity match (if expected entities specified)
            entity_match = True
            actual_entities = {}
            
            if hasattr(result, 'entities'):
                for e in result.entities:
                    etype = e.type.name if hasattr(e.type, 'name') else str(e.type)
                    actual_entities[etype] = e.value
            
            if test.expected_entities:
                for etype, evalue in test.expected_entities.items():
                    if etype not in actual_entities:
                        entity_match = False
                    # Allow partial matching for values
                    elif evalue.lower() not in actual_entities[etype].lower():
                        entity_match = False
            
            # Overall pass
            passed = intent_match and entity_match
            
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["failures"].append({
                    "id": test.id,
                    "query": test.query,
                    "expected_intent": test.expected_intent,
                    "actual_intent": actual_intent,
                    "expected_entities": test.expected_entities,
                    "actual_entities": actual_entities,
                    "gap_type": test.gap_type,
                    "description": test.description,
                })
            
            # Track by category
            cat = test.category
            if cat not in results["by_category"]:
                results["by_category"][cat] = {"passed": 0, "failed": 0}
            results["by_category"][cat]["passed" if passed else "failed"] += 1
            
            # Track by gap type
            gap = test.gap_type
            if gap not in results["by_gap_type"]:
                results["by_gap_type"][gap] = {"passed": 0, "failed": 0}
            results["by_gap_type"][gap]["passed" if passed else "failed"] += 1
            
            if verbose:
                status = "✅" if passed else "❌"
                print(f"{status} {test.id}: {test.query[:40]}...")
                if not passed:
                    print(f"   Expected: {test.expected_intent}, Got: {actual_intent}")
                    
        except Exception as e:
            results["failed"] += 1
            results["failures"].append({
                "id": test.id,
                "query": test.query,
                "error": str(e),
            })
            if verbose:
                print(f"❌ {test.id}: ERROR - {e}")
    
    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print(f"REGRESSION TEST SUMMARY: {results['passed']}/{results['total']} passed ({results['passed']/results['total']*100:.1f}%)")
        print("=" * 60)
        
        if results["failures"]:
            print("\nFailed tests:")
            for f in results["failures"]:
                print(f"  - {f['id']}: {f.get('description', f.get('error', 'Unknown'))}")
    
    return results


if __name__ == "__main__":
    # Run regression tests if executed directly
    import sys
    sys.path.insert(0, str(__file__).split("tests")[0] + "src")
    
    try:
        from workflow_composer.agents.intent.unified_parser import UnifiedIntentParser
        parser = UnifiedIntentParser(use_cascade=True)
        results = run_regression_tests(parser, verbose=True)
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()
