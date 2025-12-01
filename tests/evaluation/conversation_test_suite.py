"""
Conversation Test Suite for BioPipelines Chat Agent
====================================================

Comprehensive evaluation of conversational AI capabilities.

Metrics:
- Intent accuracy
- Entity extraction F1
- Multi-turn context retention
- Tool selection accuracy
- Response latency

Usage:
    python -m tests.evaluation.conversation_test_suite
    python -m tests.evaluation.conversation_test_suite --category data_discovery
    python -m tests.evaluation.conversation_test_suite --report html
"""

import json
import time
import asyncio
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pathlib import Path
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class ConversationCategory(Enum):
    """Categories of test conversations."""
    DATA_DISCOVERY = "data_discovery"
    WORKFLOW_GENERATION = "workflow_generation"
    JOB_MANAGEMENT = "job_management"
    EDUCATION = "education"
    MULTI_TURN = "multi_turn"
    AMBIGUOUS = "ambiguous"
    EDGE_CASES = "edge_cases"
    COREFERENCE = "coreference"
    ERROR_HANDLING = "error_handling"


@dataclass
class ExpectedResult:
    """Expected result for a query."""
    intent: str
    entities: Dict[str, str] = field(default_factory=dict)  # {entity_type: value}
    tool: Optional[str] = None
    contains_keywords: List[str] = field(default_factory=list)
    should_clarify: bool = False
    context_reference: Optional[str] = None  # e.g., "previous_search"


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    user_query: str
    expected: ExpectedResult
    description: str = ""


@dataclass
class TestConversation:
    """A complete test conversation with multiple turns."""
    id: str
    name: str
    category: ConversationCategory
    turns: List[ConversationTurn]
    description: str = ""
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class TurnResult:
    """Result of evaluating a single turn."""
    query: str
    expected_intent: str
    actual_intent: str
    intent_correct: bool
    expected_entities: Dict[str, str]
    actual_entities: Dict[str, str]
    entity_precision: float
    entity_recall: float
    entity_f1: float
    tool_expected: Optional[str]
    tool_actual: Optional[str]
    tool_correct: bool
    response_preview: str
    latency_ms: float
    confidence: float
    agreement: float
    errors: List[str] = field(default_factory=list)


@dataclass
class ConversationResult:
    """Result of evaluating a complete conversation."""
    conversation_id: str
    conversation_name: str
    category: str
    turns: List[TurnResult]
    overall_intent_accuracy: float
    overall_entity_f1: float
    overall_tool_accuracy: float
    avg_latency_ms: float
    context_retention_score: float
    passed: bool


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    total_conversations: int
    total_turns: int
    overall_intent_accuracy: float
    overall_entity_f1: float
    overall_tool_accuracy: float
    avg_latency_ms: float
    category_breakdown: Dict[str, Dict[str, float]]
    conversations: List[ConversationResult]
    failed_conversations: List[str]


# =============================================================================
# Test Conversation Definitions
# =============================================================================

def get_test_conversations() -> List[TestConversation]:
    """
    Get all test conversations.
    
    This is the main test suite with diverse conversation scenarios.
    """
    conversations = []
    
    # =========================================================================
    # DATA DISCOVERY CONVERSATIONS
    # =========================================================================
    
    # DD-001: Basic ENCODE search
    conversations.append(TestConversation(
        id="DD-001",
        name="Basic ENCODE ChIP-seq Search",
        category=ConversationCategory.DATA_DISCOVERY,
        difficulty="easy",
        description="Simple single-turn search for ChIP-seq data",
        turns=[
            ConversationTurn(
                user_query="Search ENCODE for human liver ChIP-seq H3K27ac data",
                expected=ExpectedResult(
                    intent="DATA_SEARCH",
                    entities={"ORGANISM": "human", "TISSUE": "liver", "ASSAY_TYPE": "ChIP-seq"},
                    tool="search_databases",
                    contains_keywords=["ENCODE", "ChIP-seq", "H3K27ac"],
                ),
                description="Clear search with all key entities"
            ),
        ]
    ))
    
    # DD-002: GEO RNA-seq search
    conversations.append(TestConversation(
        id="DD-002",
        name="GEO RNA-seq Search",
        category=ConversationCategory.DATA_DISCOVERY,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="Find RNA-seq data from mouse brain in GEO",
                expected=ExpectedResult(
                    intent="DATA_SEARCH",
                    entities={"ORGANISM": "mouse", "TISSUE": "brain", "ASSAY_TYPE": "RNA-seq"},
                    tool="search_databases",
                ),
            ),
        ]
    ))
    
    # DD-003: Cancer data search (TCGA)
    conversations.append(TestConversation(
        id="DD-003",
        name="TCGA Cancer Search",
        category=ConversationCategory.DATA_DISCOVERY,
        difficulty="medium",
        turns=[
            ConversationTurn(
                user_query="Search for glioblastoma methylation data in TCGA",
                expected=ExpectedResult(
                    intent="DATA_SEARCH",
                    entities={"DISEASE": "glioblastoma", "ASSAY_TYPE": "methylation"},
                    tool="search_databases",
                    contains_keywords=["TCGA", "GBM", "methylation"],
                ),
            ),
        ]
    ))
    
    # DD-004: Natural language paraphrase
    conversations.append(TestConversation(
        id="DD-004",
        name="Paraphrased Search Query",
        category=ConversationCategory.DATA_DISCOVERY,
        difficulty="medium",
        description="Tests semantic understanding of paraphrased queries",
        turns=[
            ConversationTurn(
                user_query="I need chromatin accessibility data from human heart cells",
                expected=ExpectedResult(
                    intent="DATA_SEARCH",
                    entities={"ORGANISM": "human", "TISSUE": "heart", "ASSAY_TYPE": "ATAC-seq"},
                    tool="search_databases",
                ),
                description="'chromatin accessibility' should map to ATAC-seq"
            ),
        ]
    ))
    
    # DD-005: Abbreviated query
    conversations.append(TestConversation(
        id="DD-005",
        name="Abbreviated Search Terms",
        category=ConversationCategory.DATA_DISCOVERY,
        difficulty="medium",
        turns=[
            ConversationTurn(
                user_query="find rnaseq mouse liver",
                expected=ExpectedResult(
                    intent="DATA_SEARCH",
                    entities={"ORGANISM": "mouse", "TISSUE": "liver", "ASSAY_TYPE": "RNA-seq"},
                    tool="search_databases",
                ),
                description="Lowercase, no hyphens - should still work"
            ),
        ]
    ))
    
    # DD-006: Dataset details
    conversations.append(TestConversation(
        id="DD-006",
        name="Get Dataset Details",
        category=ConversationCategory.DATA_DISCOVERY,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="Show details for ENCSR000ABC",
                expected=ExpectedResult(
                    intent="DATA_DESCRIBE",
                    entities={"DATASET_ID": "ENCSR000ABC"},
                    tool="get_dataset_details",
                ),
            ),
        ]
    ))
    
    # DD-007: Download dataset
    conversations.append(TestConversation(
        id="DD-007",
        name="Download Dataset",
        category=ConversationCategory.DATA_DISCOVERY,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="Download dataset GSE12345",
                expected=ExpectedResult(
                    intent="DATA_DOWNLOAD",
                    entities={"DATASET_ID": "GSE12345"},
                    tool="download_dataset",
                ),
            ),
        ]
    ))
    
    # DD-008: Complex multi-criteria search
    conversations.append(TestConversation(
        id="DD-008",
        name="Complex Multi-Criteria Search",
        category=ConversationCategory.DATA_DISCOVERY,
        difficulty="hard",
        turns=[
            ConversationTurn(
                user_query="Find H3K4me3 and H3K27me3 ChIP-seq from human embryonic stem cells with at least 2 replicates",
                expected=ExpectedResult(
                    intent="DATA_SEARCH",
                    entities={
                        "ORGANISM": "human",
                        "CELL_TYPE": "embryonic stem cells",
                        "ASSAY_TYPE": "ChIP-seq",
                    },
                    tool="search_databases",
                ),
                description="Complex query with multiple histone marks and criteria"
            ),
        ]
    ))
    
    # DD-009: Scan local data
    conversations.append(TestConversation(
        id="DD-009",
        name="Scan Local Directory",
        category=ConversationCategory.DATA_DISCOVERY,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="Scan /data/raw for FASTQ files",
                expected=ExpectedResult(
                    intent="DATA_SCAN",
                    entities={"PATH": "/data/raw"},
                    tool="scan_data",
                ),
            ),
        ]
    ))
    
    # DD-010: Reference check
    conversations.append(TestConversation(
        id="DD-010",
        name="Check Reference Genome",
        category=ConversationCategory.DATA_DISCOVERY,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="Do I have the hg38 reference genome?",
                expected=ExpectedResult(
                    intent="REFERENCE_CHECK",
                    entities={"GENOME": "hg38"},
                    tool="check_references",
                ),
            ),
        ]
    ))
    
    # =========================================================================
    # WORKFLOW GENERATION CONVERSATIONS
    # =========================================================================
    
    # WF-001: Basic RNA-seq workflow
    conversations.append(TestConversation(
        id="WF-001",
        name="Create RNA-seq Workflow",
        category=ConversationCategory.WORKFLOW_GENERATION,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="Create an RNA-seq workflow for human samples",
                expected=ExpectedResult(
                    intent="WORKFLOW_CREATE",
                    entities={"ASSAY_TYPE": "RNA-seq", "ORGANISM": "human"},
                    tool="generate_workflow",
                ),
            ),
        ]
    ))
    
    # WF-002: ChIP-seq pipeline
    conversations.append(TestConversation(
        id="WF-002",
        name="Create ChIP-seq Pipeline",
        category=ConversationCategory.WORKFLOW_GENERATION,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="Generate a ChIP-seq pipeline for mouse with input controls",
                expected=ExpectedResult(
                    intent="WORKFLOW_CREATE",
                    entities={"ASSAY_TYPE": "ChIP-seq", "ORGANISM": "mouse"},
                    tool="generate_workflow",
                ),
            ),
        ]
    ))
    
    # WF-003: ATAC-seq workflow
    conversations.append(TestConversation(
        id="WF-003",
        name="ATAC-seq Workflow",
        category=ConversationCategory.WORKFLOW_GENERATION,
        difficulty="medium",
        turns=[
            ConversationTurn(
                user_query="I want to analyze chromatin accessibility data using ATAC-seq pipeline",
                expected=ExpectedResult(
                    intent="WORKFLOW_CREATE",
                    entities={"ASSAY_TYPE": "ATAC-seq"},
                    tool="generate_workflow",
                ),
            ),
        ]
    ))
    
    # WF-004: Variant calling
    conversations.append(TestConversation(
        id="WF-004",
        name="Variant Calling Pipeline",
        category=ConversationCategory.WORKFLOW_GENERATION,
        difficulty="medium",
        turns=[
            ConversationTurn(
                user_query="Set up a variant calling pipeline for whole exome sequencing",
                expected=ExpectedResult(
                    intent="WORKFLOW_CREATE",
                    entities={"ASSAY_TYPE": "exome"},
                    tool="generate_workflow",
                ),
            ),
        ]
    ))
    
    # WF-005: Natural language workflow request
    conversations.append(TestConversation(
        id="WF-005",
        name="Natural Language Workflow",
        category=ConversationCategory.WORKFLOW_GENERATION,
        difficulty="hard",
        turns=[
            ConversationTurn(
                user_query="I have paired-end RNA samples from liver and want to find differentially expressed genes",
                expected=ExpectedResult(
                    intent="WORKFLOW_CREATE",
                    entities={"ASSAY_TYPE": "RNA-seq", "TISSUE": "liver"},
                    tool="generate_workflow",
                    contains_keywords=["differential", "expression"],
                ),
            ),
        ]
    ))
    
    # =========================================================================
    # JOB MANAGEMENT CONVERSATIONS
    # =========================================================================
    
    # JM-001: Submit job
    conversations.append(TestConversation(
        id="JM-001",
        name="Submit Workflow Job",
        category=ConversationCategory.JOB_MANAGEMENT,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="Submit the workflow in /projects/rnaseq",
                expected=ExpectedResult(
                    intent="JOB_SUBMIT",
                    entities={"PATH": "/projects/rnaseq"},
                    tool="submit_job",
                ),
            ),
        ]
    ))
    
    # JM-002: Check job status
    conversations.append(TestConversation(
        id="JM-002",
        name="Check Job Status",
        category=ConversationCategory.JOB_MANAGEMENT,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="What's the status of job 12345?",
                expected=ExpectedResult(
                    intent="JOB_STATUS",
                    entities={"JOB_ID": "12345"},
                    tool="get_job_status",
                ),
            ),
        ]
    ))
    
    # JM-003: List running jobs
    conversations.append(TestConversation(
        id="JM-003",
        name="List Running Jobs",
        category=ConversationCategory.JOB_MANAGEMENT,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="Show me all running jobs",
                expected=ExpectedResult(
                    intent="JOB_STATUS",
                    tool="list_jobs",
                ),
            ),
        ]
    ))
    
    # JM-004: Get job logs
    conversations.append(TestConversation(
        id="JM-004",
        name="Get Job Logs",
        category=ConversationCategory.JOB_MANAGEMENT,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="Show logs for job 67890",
                expected=ExpectedResult(
                    intent="JOB_LOGS",
                    entities={"JOB_ID": "67890"},
                    tool="get_logs",
                ),
            ),
        ]
    ))
    
    # JM-005: Diagnose error
    conversations.append(TestConversation(
        id="JM-005",
        name="Diagnose Job Error",
        category=ConversationCategory.JOB_MANAGEMENT,
        difficulty="medium",
        turns=[
            ConversationTurn(
                user_query="Job 54321 failed with OOM error, what went wrong?",
                expected=ExpectedResult(
                    intent="DIAGNOSE_ERROR",
                    entities={"JOB_ID": "54321"},
                    tool="diagnose_error",
                ),
            ),
        ]
    ))
    
    # JM-006: Cancel job
    conversations.append(TestConversation(
        id="JM-006",
        name="Cancel Running Job",
        category=ConversationCategory.JOB_MANAGEMENT,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="Cancel job 99999",
                expected=ExpectedResult(
                    intent="JOB_CANCEL",
                    entities={"JOB_ID": "99999"},
                    tool="cancel_job",
                ),
            ),
        ]
    ))
    
    # =========================================================================
    # EDUCATION CONVERSATIONS
    # =========================================================================
    
    # ED-001: Explain concept
    conversations.append(TestConversation(
        id="ED-001",
        name="Explain Bioinformatics Concept",
        category=ConversationCategory.EDUCATION,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="What is RNA-seq?",
                expected=ExpectedResult(
                    intent="EDUCATION_EXPLAIN",
                    entities={"ASSAY_TYPE": "RNA-seq"},
                    tool="explain_concept",
                ),
            ),
        ]
    ))
    
    # ED-002: Explain ChIP-seq
    conversations.append(TestConversation(
        id="ED-002",
        name="Explain ChIP-seq",
        category=ConversationCategory.EDUCATION,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="Explain ChIP-seq to me",
                expected=ExpectedResult(
                    intent="EDUCATION_EXPLAIN",
                    entities={"ASSAY_TYPE": "ChIP-seq"},
                    tool="explain_concept",
                ),
            ),
        ]
    ))
    
    # ED-003: How does X work
    conversations.append(TestConversation(
        id="ED-003",
        name="How Does ATAC-seq Work",
        category=ConversationCategory.EDUCATION,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="How does ATAC-seq work?",
                expected=ExpectedResult(
                    intent="EDUCATION_EXPLAIN",
                    entities={"ASSAY_TYPE": "ATAC-seq"},
                    tool="explain_concept",
                ),
            ),
        ]
    ))
    
    # ED-004: Complex concept
    conversations.append(TestConversation(
        id="ED-004",
        name="Complex Concept Explanation",
        category=ConversationCategory.EDUCATION,
        difficulty="medium",
        turns=[
            ConversationTurn(
                user_query="What's the difference between peak calling and motif analysis in ChIP-seq?",
                expected=ExpectedResult(
                    intent="EDUCATION_EXPLAIN",
                    contains_keywords=["peak", "motif", "ChIP"],
                    tool="explain_concept",
                ),
            ),
        ]
    ))
    
    # ED-005: Help request
    conversations.append(TestConversation(
        id="ED-005",
        name="General Help",
        category=ConversationCategory.EDUCATION,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="Help me understand what this system can do",
                expected=ExpectedResult(
                    intent="EDUCATION_HELP",
                    tool="show_help",
                ),
            ),
        ]
    ))
    
    # =========================================================================
    # MULTI-TURN CONVERSATIONS
    # =========================================================================
    
    # MT-001: Search then download
    conversations.append(TestConversation(
        id="MT-001",
        name="Search Then Download",
        category=ConversationCategory.MULTI_TURN,
        difficulty="medium",
        description="Tests context retention across search and download",
        turns=[
            ConversationTurn(
                user_query="Search for human liver RNA-seq data",
                expected=ExpectedResult(
                    intent="DATA_SEARCH",
                    entities={"ORGANISM": "human", "TISSUE": "liver", "ASSAY_TYPE": "RNA-seq"},
                    tool="search_databases",
                ),
            ),
            ConversationTurn(
                user_query="Download all of them",
                expected=ExpectedResult(
                    intent="DATA_DOWNLOAD",
                    tool="download_dataset",
                    context_reference="previous_search",
                ),
                description="Should use context from previous search"
            ),
        ]
    ))
    
    # MT-002: Search then get details
    conversations.append(TestConversation(
        id="MT-002",
        name="Search Then Details",
        category=ConversationCategory.MULTI_TURN,
        difficulty="medium",
        turns=[
            ConversationTurn(
                user_query="Find ATAC-seq data from mouse brain",
                expected=ExpectedResult(
                    intent="DATA_SEARCH",
                    entities={"ORGANISM": "mouse", "TISSUE": "brain", "ASSAY_TYPE": "ATAC-seq"},
                    tool="search_databases",
                ),
            ),
            ConversationTurn(
                user_query="Show me details for the first one",
                expected=ExpectedResult(
                    intent="DATA_DESCRIBE",
                    tool="get_dataset_details",
                    context_reference="previous_search",
                ),
            ),
        ]
    ))
    
    # MT-003: Create workflow then submit
    conversations.append(TestConversation(
        id="MT-003",
        name="Create Workflow Then Submit",
        category=ConversationCategory.MULTI_TURN,
        difficulty="medium",
        turns=[
            ConversationTurn(
                user_query="Create an RNA-seq pipeline for my human samples",
                expected=ExpectedResult(
                    intent="WORKFLOW_CREATE",
                    entities={"ASSAY_TYPE": "RNA-seq", "ORGANISM": "human"},
                    tool="generate_workflow",
                ),
            ),
            ConversationTurn(
                user_query="Now submit it",
                expected=ExpectedResult(
                    intent="JOB_SUBMIT",
                    tool="submit_job",
                    context_reference="previous_workflow",
                ),
            ),
        ]
    ))
    
    # MT-004: Submit then monitor
    conversations.append(TestConversation(
        id="MT-004",
        name="Submit Then Monitor",
        category=ConversationCategory.MULTI_TURN,
        difficulty="medium",
        turns=[
            ConversationTurn(
                user_query="Submit the workflow in /projects/chipseq",
                expected=ExpectedResult(
                    intent="JOB_SUBMIT",
                    tool="submit_job",
                ),
            ),
            ConversationTurn(
                user_query="Check on it",
                expected=ExpectedResult(
                    intent="JOB_STATUS",
                    tool="get_job_status",
                    context_reference="previous_job",
                ),
            ),
            ConversationTurn(
                user_query="Show me the logs",
                expected=ExpectedResult(
                    intent="JOB_LOGS",
                    tool="get_logs",
                    context_reference="previous_job",
                ),
            ),
        ]
    ))
    
    # MT-005: Learn then do
    conversations.append(TestConversation(
        id="MT-005",
        name="Learn Then Do",
        category=ConversationCategory.MULTI_TURN,
        difficulty="hard",
        turns=[
            ConversationTurn(
                user_query="What is differential expression analysis?",
                expected=ExpectedResult(
                    intent="EDUCATION_EXPLAIN",
                    tool="explain_concept",
                ),
            ),
            ConversationTurn(
                user_query="OK, set up a pipeline to do that with my RNA-seq data",
                expected=ExpectedResult(
                    intent="WORKFLOW_CREATE",
                    entities={"ASSAY_TYPE": "RNA-seq"},
                    tool="generate_workflow",
                ),
            ),
        ]
    ))
    
    # =========================================================================
    # COREFERENCE RESOLUTION
    # =========================================================================
    
    # CR-001: Pronoun resolution
    conversations.append(TestConversation(
        id="CR-001",
        name="Pronoun Resolution - It",
        category=ConversationCategory.COREFERENCE,
        difficulty="medium",
        turns=[
            ConversationTurn(
                user_query="Search for human ChIP-seq H3K27ac data",
                expected=ExpectedResult(
                    intent="DATA_SEARCH",
                    tool="search_databases",
                ),
            ),
            ConversationTurn(
                user_query="Download it",
                expected=ExpectedResult(
                    intent="DATA_DOWNLOAD",
                    tool="download_dataset",
                    context_reference="previous_search",
                ),
            ),
        ]
    ))
    
    # CR-002: "that data"
    conversations.append(TestConversation(
        id="CR-002",
        name="Demonstrative Resolution - That",
        category=ConversationCategory.COREFERENCE,
        difficulty="medium",
        turns=[
            ConversationTurn(
                user_query="Find ENCODE ATAC-seq mouse brain",
                expected=ExpectedResult(
                    intent="DATA_SEARCH",
                    tool="search_databases",
                ),
            ),
            ConversationTurn(
                user_query="Analyze that data with a standard pipeline",
                expected=ExpectedResult(
                    intent="WORKFLOW_CREATE",
                    context_reference="previous_search",
                ),
            ),
        ]
    ))
    
    # CR-003: "the dataset"
    conversations.append(TestConversation(
        id="CR-003",
        name="Definite Article Resolution",
        category=ConversationCategory.COREFERENCE,
        difficulty="medium",
        turns=[
            ConversationTurn(
                user_query="Get details for ENCSR000ABC",
                expected=ExpectedResult(
                    intent="DATA_DESCRIBE",
                    entities={"DATASET_ID": "ENCSR000ABC"},
                    tool="get_dataset_details",
                ),
            ),
            ConversationTurn(
                user_query="Download the dataset",
                expected=ExpectedResult(
                    intent="DATA_DOWNLOAD",
                    entities={"DATASET_ID": "ENCSR000ABC"},
                    tool="download_dataset",
                ),
            ),
        ]
    ))
    
    # =========================================================================
    # AMBIGUOUS QUERIES (should trigger clarification)
    # =========================================================================
    
    # AM-001: Vague query
    conversations.append(TestConversation(
        id="AM-001",
        name="Vague Search Query",
        category=ConversationCategory.AMBIGUOUS,
        difficulty="hard",
        turns=[
            ConversationTurn(
                user_query="data",
                expected=ExpectedResult(
                    intent="META_UNKNOWN",
                    should_clarify=True,
                ),
                description="Too vague, should ask for clarification"
            ),
        ]
    ))
    
    # AM-002: Mixed intent
    conversations.append(TestConversation(
        id="AM-002",
        name="Mixed Intent Query",
        category=ConversationCategory.AMBIGUOUS,
        difficulty="hard",
        turns=[
            ConversationTurn(
                user_query="Search and download RNA-seq",
                expected=ExpectedResult(
                    intent="DATA_SEARCH",  # Primary intent
                    entities={"ASSAY_TYPE": "RNA-seq"},
                ),
                description="Mixed intent - should handle gracefully"
            ),
        ]
    ))
    
    # AM-003: Incomplete download
    conversations.append(TestConversation(
        id="AM-003",
        name="Incomplete Download Request",
        category=ConversationCategory.AMBIGUOUS,
        difficulty="medium",
        turns=[
            ConversationTurn(
                user_query="Download",
                expected=ExpectedResult(
                    intent="DATA_DOWNLOAD",
                    should_clarify=True,
                ),
                description="Missing dataset ID - should ask which one"
            ),
        ]
    ))
    
    # =========================================================================
    # EDGE CASES
    # =========================================================================
    
    # EC-001: Typos
    conversations.append(TestConversation(
        id="EC-001",
        name="Query With Typos",
        category=ConversationCategory.EDGE_CASES,
        difficulty="medium",
        turns=[
            ConversationTurn(
                user_query="serch for human rnaseq liver",
                expected=ExpectedResult(
                    intent="DATA_SEARCH",
                    entities={"ORGANISM": "human", "TISSUE": "liver", "ASSAY_TYPE": "RNA-seq"},
                    tool="search_databases",
                ),
                description="Should handle typos gracefully"
            ),
        ]
    ))
    
    # EC-002: All caps
    conversations.append(TestConversation(
        id="EC-002",
        name="All Caps Query",
        category=ConversationCategory.EDGE_CASES,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="SEARCH FOR HUMAN CHIP-SEQ DATA",
                expected=ExpectedResult(
                    intent="DATA_SEARCH",
                    entities={"ORGANISM": "human", "ASSAY_TYPE": "ChIP-seq"},
                    tool="search_databases",
                ),
            ),
        ]
    ))
    
    # EC-003: Special characters
    conversations.append(TestConversation(
        id="EC-003",
        name="Special Characters",
        category=ConversationCategory.EDGE_CASES,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="Find H3K4me3 ChIP-seq (human, liver)",
                expected=ExpectedResult(
                    intent="DATA_SEARCH",
                    entities={"ORGANISM": "human", "TISSUE": "liver"},
                    tool="search_databases",
                ),
            ),
        ]
    ))
    
    # EC-004: Very long query
    conversations.append(TestConversation(
        id="EC-004",
        name="Very Long Query",
        category=ConversationCategory.EDGE_CASES,
        difficulty="hard",
        turns=[
            ConversationTurn(
                user_query="I am looking for publicly available ChIP-seq datasets that profile H3K27ac histone modifications in human liver tissue samples from healthy donors, preferably with at least two biological replicates and paired input controls, ideally from the ENCODE consortium or GEO database",
                expected=ExpectedResult(
                    intent="DATA_SEARCH",
                    entities={
                        "ORGANISM": "human",
                        "TISSUE": "liver",
                        "ASSAY_TYPE": "ChIP-seq",
                    },
                    tool="search_databases",
                ),
            ),
        ]
    ))
    
    # EC-005: Empty-ish query
    conversations.append(TestConversation(
        id="EC-005",
        name="Near-Empty Query",
        category=ConversationCategory.EDGE_CASES,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="hi",
                expected=ExpectedResult(
                    intent="META_GREETING",
                    tool="show_help",
                ),
            ),
        ]
    ))
    
    # EC-006: Non-English organism name
    conversations.append(TestConversation(
        id="EC-006",
        name="Scientific Name",
        category=ConversationCategory.EDGE_CASES,
        difficulty="medium",
        turns=[
            ConversationTurn(
                user_query="Search for Mus musculus RNA-seq in hippocampus",
                expected=ExpectedResult(
                    intent="DATA_SEARCH",
                    entities={"ORGANISM": "mouse", "ASSAY_TYPE": "RNA-seq"},
                    tool="search_databases",
                ),
                description="Should map 'Mus musculus' to 'mouse'"
            ),
        ]
    ))
    
    # =========================================================================
    # ERROR HANDLING
    # =========================================================================
    
    # EH-001: Invalid dataset ID
    conversations.append(TestConversation(
        id="EH-001",
        name="Invalid Dataset ID",
        category=ConversationCategory.ERROR_HANDLING,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="Download dataset INVALID123",
                expected=ExpectedResult(
                    intent="DATA_DOWNLOAD",
                    entities={"DATASET_ID": "INVALID123"},
                    tool="download_dataset",
                ),
                description="Should handle gracefully and provide helpful error"
            ),
        ]
    ))
    
    # EH-002: Invalid path
    conversations.append(TestConversation(
        id="EH-002",
        name="Invalid Path",
        category=ConversationCategory.ERROR_HANDLING,
        difficulty="easy",
        turns=[
            ConversationTurn(
                user_query="Scan /nonexistent/path for data",
                expected=ExpectedResult(
                    intent="DATA_SCAN",
                    entities={"PATH": "/nonexistent/path"},
                    tool="scan_data",
                ),
            ),
        ]
    ))
    
    return conversations


# =============================================================================
# Evaluation Functions
# =============================================================================

def calculate_entity_metrics(
    expected: Dict[str, str], 
    actual: Dict[str, str]
) -> Tuple[float, float, float]:
    """Calculate precision, recall, F1 for entity extraction."""
    if not expected and not actual:
        return 1.0, 1.0, 1.0
    
    if not expected:
        return 0.0, 1.0, 0.0  # No expected, but got some -> precision=0
    
    if not actual:
        return 1.0, 0.0, 0.0  # Expected some, got none -> recall=0
    
    # Count matches (type and value must match, case-insensitive)
    expected_lower = {k.lower(): v.lower() for k, v in expected.items()}
    actual_lower = {k.lower(): v.lower() for k, v in actual.items()}
    
    matches = 0
    for k, v in expected_lower.items():
        if k in actual_lower:
            # Fuzzy match on value
            if v in actual_lower[k] or actual_lower[k] in v:
                matches += 1
    
    precision = matches / len(actual_lower) if actual_lower else 0
    recall = matches / len(expected_lower) if expected_lower else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


class ConversationEvaluator:
    """Evaluates conversations against expected results."""
    
    def __init__(self, use_ensemble: bool = True):
        """
        Initialize evaluator.
        
        Args:
            use_ensemble: Use UnifiedIntentParser (recommended)
        """
        self.use_ensemble = use_ensemble
        self._agent = None
        self._parser = None
    
    @property
    def agent(self):
        """Lazy load agent."""
        if self._agent is None:
            from workflow_composer.agents import UnifiedAgent, AutonomyLevel
            self._agent = UnifiedAgent(autonomy_level=AutonomyLevel.READONLY)
        return self._agent
    
    @property
    def parser(self):
        """Lazy load intent parser."""
        if self._parser is None:
            from workflow_composer.agents.intent import UnifiedIntentParser
            self._parser = UnifiedIntentParser()
        return self._parser
    
    # Keep ensemble_parser as alias for backward compatibility
    @property
    def ensemble_parser(self):
        """Alias for parser (backward compatibility)."""
        return self.parser
    
    def evaluate_turn(
        self, 
        query: str, 
        expected: ExpectedResult,
        context: Optional[Dict] = None
    ) -> TurnResult:
        """Evaluate a single conversation turn."""
        errors = []
        start_time = time.time()
        
        try:
            # Parse with intent parser
            parse_result = self.parser.parse(query)
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract actual values - handle both result types
            if hasattr(parse_result, 'primary_intent'):
                actual_intent = parse_result.primary_intent.name if hasattr(parse_result.primary_intent, 'name') else str(parse_result.primary_intent)
            else:
                actual_intent = parse_result.intent
            actual_entities = {}
            
            # Convert entities to dict
            for entity in parse_result.entities:
                if hasattr(entity, 'entity_type'):
                    actual_entities[entity.entity_type] = entity.text
                elif isinstance(entity, dict):
                    actual_entities[entity.get('type', 'UNKNOWN')] = entity.get('text', '')
            
            # Check intent
            intent_correct = actual_intent == expected.intent
            
            # Calculate entity metrics
            precision, recall, f1 = calculate_entity_metrics(
                expected.entities, actual_entities
            )
            
            # Check tool (infer from intent)
            tool_mapping = {
                "DATA_SEARCH": "search_databases",
                "DATA_DOWNLOAD": "download_dataset",
                "DATA_SCAN": "scan_data",
                "DATA_DESCRIBE": "get_dataset_details",
                "WORKFLOW_CREATE": "generate_workflow",
                "JOB_SUBMIT": "submit_job",
                "JOB_STATUS": "get_job_status",
                "JOB_LOGS": "get_logs",
                "JOB_CANCEL": "cancel_job",
                "DIAGNOSE_ERROR": "diagnose_error",
                "EDUCATION_EXPLAIN": "explain_concept",
                "EDUCATION_HELP": "show_help",
                "REFERENCE_CHECK": "check_references",
            }
            actual_tool = tool_mapping.get(actual_intent)
            tool_correct = (expected.tool is None) or (actual_tool == expected.tool)
            
            return TurnResult(
                query=query,
                expected_intent=expected.intent,
                actual_intent=actual_intent,
                intent_correct=intent_correct,
                expected_entities=expected.entities,
                actual_entities=actual_entities,
                entity_precision=precision,
                entity_recall=recall,
                entity_f1=f1,
                tool_expected=expected.tool,
                tool_actual=actual_tool,
                tool_correct=tool_correct,
                response_preview="",  # Would need full agent call
                latency_ms=latency_ms,
                confidence=parse_result.confidence,
                agreement=parse_result.agreement_level,
                errors=errors,
            )
            
        except Exception as e:
            errors.append(str(e))
            return TurnResult(
                query=query,
                expected_intent=expected.intent,
                actual_intent="ERROR",
                intent_correct=False,
                expected_entities=expected.entities,
                actual_entities={},
                entity_precision=0,
                entity_recall=0,
                entity_f1=0,
                tool_expected=expected.tool,
                tool_actual=None,
                tool_correct=False,
                response_preview="",
                latency_ms=(time.time() - start_time) * 1000,
                confidence=0,
                agreement=0,
                errors=errors,
            )
    
    def evaluate_conversation(
        self, 
        conversation: TestConversation
    ) -> ConversationResult:
        """Evaluate a complete conversation."""
        turn_results = []
        context = {}
        
        for turn in conversation.turns:
            result = self.evaluate_turn(turn.user_query, turn.expected, context)
            turn_results.append(result)
            
            # Update context (simplified)
            if result.actual_intent == "DATA_SEARCH":
                context["previous_search"] = True
            elif result.actual_intent == "WORKFLOW_CREATE":
                context["previous_workflow"] = True
        
        # Calculate overall metrics
        intent_accuracy = sum(1 for t in turn_results if t.intent_correct) / len(turn_results)
        entity_f1 = statistics.mean([t.entity_f1 for t in turn_results]) if turn_results else 0
        tool_accuracy = sum(1 for t in turn_results if t.tool_correct) / len(turn_results)
        avg_latency = statistics.mean([t.latency_ms for t in turn_results]) if turn_results else 0
        
        # Context retention (simplified - check if multi-turn worked)
        context_score = 1.0  # TODO: implement proper context scoring
        
        # Overall pass/fail
        passed = intent_accuracy >= 0.8 and entity_f1 >= 0.6
        
        return ConversationResult(
            conversation_id=conversation.id,
            conversation_name=conversation.name,
            category=conversation.category.value,
            turns=turn_results,
            overall_intent_accuracy=intent_accuracy,
            overall_entity_f1=entity_f1,
            overall_tool_accuracy=tool_accuracy,
            avg_latency_ms=avg_latency,
            context_retention_score=context_score,
            passed=passed,
        )
    
    def run_evaluation(
        self,
        conversations: Optional[List[TestConversation]] = None,
        categories: Optional[List[ConversationCategory]] = None,
    ) -> EvaluationReport:
        """Run full evaluation."""
        if conversations is None:
            conversations = get_test_conversations()
        
        if categories:
            conversations = [c for c in conversations if c.category in categories]
        
        results = []
        failed = []
        
        print(f"\n{'='*60}")
        print(f"Running evaluation on {len(conversations)} conversations...")
        print(f"{'='*60}\n")
        
        for i, conv in enumerate(conversations):
            print(f"[{i+1}/{len(conversations)}] {conv.id}: {conv.name}...", end=" ")
            result = self.evaluate_conversation(conv)
            results.append(result)
            
            if result.passed:
                print(f"✅ PASS (intent={result.overall_intent_accuracy:.0%}, F1={result.overall_entity_f1:.2f})")
            else:
                print(f"❌ FAIL (intent={result.overall_intent_accuracy:.0%}, F1={result.overall_entity_f1:.2f})")
                failed.append(conv.id)
        
        # Calculate overall metrics
        total_turns = sum(len(r.turns) for r in results)
        overall_intent = statistics.mean([r.overall_intent_accuracy for r in results])
        overall_entity = statistics.mean([r.overall_entity_f1 for r in results])
        overall_tool = statistics.mean([r.overall_tool_accuracy for r in results])
        overall_latency = statistics.mean([r.avg_latency_ms for r in results])
        
        # Category breakdown
        category_breakdown = {}
        for category in ConversationCategory:
            cat_results = [r for r in results if r.category == category.value]
            if cat_results:
                category_breakdown[category.value] = {
                    "count": len(cat_results),
                    "intent_accuracy": statistics.mean([r.overall_intent_accuracy for r in cat_results]),
                    "entity_f1": statistics.mean([r.overall_entity_f1 for r in cat_results]),
                    "tool_accuracy": statistics.mean([r.overall_tool_accuracy for r in cat_results]),
                    "pass_rate": sum(1 for r in cat_results if r.passed) / len(cat_results),
                }
        
        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_conversations=len(conversations),
            total_turns=total_turns,
            overall_intent_accuracy=overall_intent,
            overall_entity_f1=overall_entity,
            overall_tool_accuracy=overall_tool,
            avg_latency_ms=overall_latency,
            category_breakdown=category_breakdown,
            conversations=results,
            failed_conversations=failed,
        )
        
        return report


def print_report(report: EvaluationReport):
    """Print evaluation report to console."""
    print(f"\n{'='*60}")
    print("EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"Timestamp: {report.timestamp}")
    print(f"Total Conversations: {report.total_conversations}")
    print(f"Total Turns: {report.total_turns}")
    print(f"\n--- OVERALL METRICS ---")
    print(f"Intent Accuracy:  {report.overall_intent_accuracy:.1%}")
    print(f"Entity F1 Score:  {report.overall_entity_f1:.3f}")
    print(f"Tool Accuracy:    {report.overall_tool_accuracy:.1%}")
    print(f"Avg Latency:      {report.avg_latency_ms:.1f}ms")
    
    print(f"\n--- CATEGORY BREAKDOWN ---")
    for category, metrics in report.category_breakdown.items():
        print(f"\n{category.upper()}:")
        print(f"  Conversations: {metrics['count']}")
        print(f"  Intent Acc:    {metrics['intent_accuracy']:.1%}")
        print(f"  Entity F1:     {metrics['entity_f1']:.3f}")
        print(f"  Pass Rate:     {metrics['pass_rate']:.1%}")
    
    if report.failed_conversations:
        print(f"\n--- FAILED CONVERSATIONS ({len(report.failed_conversations)}) ---")
        for conv_id in report.failed_conversations:
            print(f"  ❌ {conv_id}")
    
    passed = report.total_conversations - len(report.failed_conversations)
    print(f"\n{'='*60}")
    print(f"RESULT: {passed}/{report.total_conversations} PASSED ({passed/report.total_conversations:.1%})")
    print(f"{'='*60}\n")


def save_report(report: EvaluationReport, output_path: Path):
    """Save report to JSON file."""
    # Convert to serializable format
    data = {
        "timestamp": report.timestamp,
        "total_conversations": report.total_conversations,
        "total_turns": report.total_turns,
        "overall_intent_accuracy": report.overall_intent_accuracy,
        "overall_entity_f1": report.overall_entity_f1,
        "overall_tool_accuracy": report.overall_tool_accuracy,
        "avg_latency_ms": report.avg_latency_ms,
        "category_breakdown": report.category_breakdown,
        "failed_conversations": report.failed_conversations,
        "conversations": [
            {
                "id": c.conversation_id,
                "name": c.conversation_name,
                "category": c.category,
                "passed": c.passed,
                "intent_accuracy": c.overall_intent_accuracy,
                "entity_f1": c.overall_entity_f1,
                "tool_accuracy": c.overall_tool_accuracy,
                "latency_ms": c.avg_latency_ms,
                "turns": [
                    {
                        "query": t.query,
                        "expected_intent": t.expected_intent,
                        "actual_intent": t.actual_intent,
                        "intent_correct": t.intent_correct,
                        "entity_f1": t.entity_f1,
                        "tool_correct": t.tool_correct,
                        "latency_ms": t.latency_ms,
                        "confidence": t.confidence,
                        "agreement": t.agreement,
                    }
                    for t in c.turns
                ]
            }
            for c in report.conversations
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Report saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run conversation evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate chat agent conversations")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--output", type=str, default="evaluation_report.json", help="Output file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Filter by category if specified
    categories = None
    if args.category:
        try:
            categories = [ConversationCategory(args.category)]
        except ValueError:
            print(f"Unknown category: {args.category}")
            print(f"Available: {[c.value for c in ConversationCategory]}")
            return
    
    # Run evaluation
    evaluator = ConversationEvaluator()
    report = evaluator.run_evaluation(categories=categories)
    
    # Print and save
    print_report(report)
    
    output_path = Path(args.output)
    save_report(report, output_path)


if __name__ == "__main__":
    main()
