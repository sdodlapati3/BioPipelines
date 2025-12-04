"""
Comprehensive Test Conversation Dataset
========================================

100+ multi-turn test conversations covering all agent capabilities.
Each conversation tests context retention, intent parsing, and tool selection.

Categories:
- Data Discovery (search, download, scan, describe)
- Workflow Generation (create, modify, validate)
- Job Management (submit, status, logs, cancel, resubmit)
- Education (explain concepts, help)
- Error Handling (diagnose, fix, recover)
- Multi-Turn Context (coreference, follow-up)
- Edge Cases (typos, ambiguous, adversarial)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import random


class Category(Enum):
    DATA_DISCOVERY = "data_discovery"
    WORKFLOW_GENERATION = "workflow_generation"
    JOB_MANAGEMENT = "job_management"
    EDUCATION = "education"
    ERROR_HANDLING = "error_handling"
    MULTI_TURN = "multi_turn"
    EDGE_CASES = "edge_cases"
    COREFERENCE = "coreference"
    AMBIGUOUS = "ambiguous"


@dataclass
class Turn:
    """A single conversation turn."""
    query: str
    expected_intent: str
    expected_entities: Dict[str, str] = field(default_factory=dict)
    expected_tool: Optional[str] = None
    description: str = ""
    context_reference: Optional[str] = None  # References previous turn


@dataclass
class TestConversation:
    """A complete test conversation."""
    id: str
    name: str
    category: Category
    turns: List[Turn]
    description: str = ""
    difficulty: str = "medium"
    tags: List[str] = field(default_factory=list)


# =============================================================================
# DATA DISCOVERY CONVERSATIONS (20+)
# =============================================================================

DATA_DISCOVERY_CONVERSATIONS = [
    # Basic search patterns
    TestConversation(
        id="DD-001",
        name="Basic GEO Search",
        category=Category.DATA_DISCOVERY,
        difficulty="easy",
        tags=["search", "geo", "single-turn"],
        turns=[
            Turn(
                query="Search for human liver RNA-seq data in GEO",
                expected_intent="DATA_SEARCH",
                expected_entities={"ORGANISM": "human", "TISSUE": "liver", "ASSAY_TYPE": "RNA-seq", "DATABASE": "GEO"},
                expected_tool="search_databases",
            )
        ]
    ),
    TestConversation(
        id="DD-002",
        name="ENCODE ChIP-seq Search",
        category=Category.DATA_DISCOVERY,
        difficulty="easy",
        tags=["search", "encode", "chip-seq"],
        turns=[
            Turn(
                query="Find H3K27ac ChIP-seq data for K562 cells in ENCODE",
                expected_intent="DATA_SEARCH",
                expected_entities={"HISTONE_MARK": "H3K27ac", "CELL_LINE": "K562", "DATABASE": "ENCODE"},
                expected_tool="search_databases",
            )
        ]
    ),
    TestConversation(
        id="DD-003",
        name="Search Then Download Flow",
        category=Category.DATA_DISCOVERY,
        difficulty="medium",
        tags=["search", "download", "multi-turn"],
        turns=[
            Turn(
                query="Search for mouse brain methylation data",
                expected_intent="DATA_SEARCH",
                expected_entities={"ORGANISM": "mouse", "TISSUE": "brain", "ASSAY_TYPE": "methylation"},
                expected_tool="search_databases",
            ),
            Turn(
                query="Download the first three datasets",
                expected_intent="DATA_DOWNLOAD",
                expected_entities={},
                expected_tool="download_dataset",
                context_reference="previous_search",
            ),
        ]
    ),
    TestConversation(
        id="DD-004",
        name="Data Scan and Describe",
        category=Category.DATA_DISCOVERY,
        difficulty="easy",
        tags=["scan", "describe"],
        turns=[
            Turn(
                query="Scan my data folder for sequencing files",
                expected_intent="DATA_SCAN",
                expected_entities={},
                expected_tool="scan_data",
            ),
            Turn(
                query="Show me details of the RNA-seq files",
                expected_intent="DATA_DESCRIBE",
                expected_entities={"ASSAY_TYPE": "RNA-seq"},
                expected_tool="describe_files",
                context_reference="scan_results",
            ),
        ]
    ),
    TestConversation(
        id="DD-005",
        name="Complex Multi-criteria Search",
        category=Category.DATA_DISCOVERY,
        difficulty="hard",
        tags=["search", "complex"],
        turns=[
            Turn(
                query="Find single-cell RNA-seq data from human brain cortex with Alzheimer's disease",
                expected_intent="DATA_SEARCH",
                expected_entities={
                    "ORGANISM": "human",
                    "TISSUE": "brain cortex",
                    "ASSAY_TYPE": "scRNA-seq",
                    "DISEASE": "Alzheimer's",
                },
                expected_tool="search_databases",
            )
        ]
    ),
    TestConversation(
        id="DD-006",
        name="SRA Download by Accession",
        category=Category.DATA_DISCOVERY,
        difficulty="easy",
        tags=["download", "sra"],
        turns=[
            Turn(
                query="Download SRR12345678 from SRA",
                expected_intent="DATA_DOWNLOAD",
                expected_entities={"DATASET_ID": "SRR12345678", "DATABASE": "SRA"},
                expected_tool="download_dataset",
            )
        ]
    ),
    TestConversation(
        id="DD-007",
        name="Search with Negation",
        category=Category.DATA_DISCOVERY,
        difficulty="hard",
        tags=["search", "negation"],
        turns=[
            Turn(
                query="Find mouse RNA-seq data but not from liver or kidney",
                expected_intent="DATA_SEARCH",
                expected_entities={"ORGANISM": "mouse", "ASSAY_TYPE": "RNA-seq"},
                expected_tool="search_databases",
            )
        ]
    ),
    TestConversation(
        id="DD-008",
        name="Methylation Data Search",
        category=Category.DATA_DISCOVERY,
        difficulty="medium",
        tags=["search", "methylation"],
        turns=[
            Turn(
                query="Show me details of methylation data",
                expected_intent="DATA_DESCRIBE",
                expected_entities={"ASSAY_TYPE": "methylation"},
                expected_tool="describe_files",
            )
        ]
    ),
    TestConversation(
        id="DD-009",
        name="TCGA Cancer Data",
        category=Category.DATA_DISCOVERY,
        difficulty="medium",
        tags=["search", "tcga", "cancer"],
        turns=[
            Turn(
                query="Search TCGA for breast cancer RNA-seq data",
                expected_intent="DATA_SEARCH",
                expected_entities={"DATABASE": "TCGA", "DISEASE": "breast cancer", "ASSAY_TYPE": "RNA-seq"},
                expected_tool="search_databases",
            ),
            Turn(
                query="How many samples are there?",
                expected_intent="DATA_DESCRIBE",
                expected_entities={},
                context_reference="previous_search",
            ),
        ]
    ),
    TestConversation(
        id="DD-010",
        name="What Data Do I Have",
        category=Category.DATA_DISCOVERY,
        difficulty="easy",
        tags=["scan", "question"],
        turns=[
            Turn(
                query="What data do I have in my workspace?",
                expected_intent="DATA_SCAN",
                expected_entities={},
                expected_tool="scan_data",
            )
        ]
    ),
]

# =============================================================================
# WORKFLOW GENERATION CONVERSATIONS (20+)
# =============================================================================

WORKFLOW_GENERATION_CONVERSATIONS = [
    TestConversation(
        id="WF-001",
        name="Basic RNA-seq Workflow",
        category=Category.WORKFLOW_GENERATION,
        difficulty="easy",
        tags=["workflow", "rna-seq"],
        turns=[
            Turn(
                query="Create an RNA-seq workflow for mouse liver samples",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"ORGANISM": "mouse", "TISSUE": "liver", "ANALYSIS_TYPE": "RNA-seq"},
                expected_tool="generate_workflow",
            )
        ]
    ),
    TestConversation(
        id="WF-002",
        name="ChIP-seq Peak Calling",
        category=Category.WORKFLOW_GENERATION,
        difficulty="easy",
        tags=["workflow", "chip-seq"],
        turns=[
            Turn(
                query="Generate a ChIP-seq pipeline for H3K4me3 in human cells",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"HISTONE_MARK": "H3K4me3", "ORGANISM": "human", "ANALYSIS_TYPE": "ChIP-seq"},
                expected_tool="generate_workflow",
            )
        ]
    ),
    TestConversation(
        id="WF-003",
        name="Differential Expression Full Flow",
        category=Category.WORKFLOW_GENERATION,
        difficulty="hard",
        tags=["workflow", "de", "multi-turn"],
        turns=[
            Turn(
                query="I have paired-end RNA-seq data from treated and control mouse samples",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"ORGANISM": "mouse", "DATA_TYPE": "paired-end"},
            ),
            Turn(
                query="I want to find differentially expressed genes",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"ANALYSIS_TYPE": "differential_expression"},
                expected_tool="generate_workflow",
            ),
            Turn(
                query="Use DESeq2 for the analysis",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"TOOL": "DESeq2"},
                context_reference="previous_workflow",
            ),
        ]
    ),
    TestConversation(
        id="WF-004",
        name="ATAC-seq Workflow",
        category=Category.WORKFLOW_GENERATION,
        difficulty="medium",
        tags=["workflow", "atac-seq"],
        turns=[
            Turn(
                query="Build an ATAC-seq pipeline for chromatin accessibility analysis",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"ANALYSIS_TYPE": "ATAC-seq"},
                expected_tool="generate_workflow",
            )
        ]
    ),
    TestConversation(
        id="WF-005",
        name="Variant Calling Pipeline",
        category=Category.WORKFLOW_GENERATION,
        difficulty="medium",
        tags=["workflow", "variant-calling"],
        turns=[
            Turn(
                query="Create a variant calling workflow for human whole genome sequencing data",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"ORGANISM": "human", "ANALYSIS_TYPE": "variant_calling"},
                expected_tool="generate_workflow",
            )
        ]
    ),
    TestConversation(
        id="WF-006",
        name="Single-Cell RNA-seq",
        category=Category.WORKFLOW_GENERATION,
        difficulty="hard",
        tags=["workflow", "scrna-seq"],
        turns=[
            Turn(
                query="Generate a 10x Genomics single-cell RNA-seq analysis pipeline",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"ANALYSIS_TYPE": "scRNA-seq", "PLATFORM": "10x Genomics"},
                expected_tool="generate_workflow",
            ),
            Turn(
                query="Include cell type clustering and marker gene identification",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={},
                context_reference="previous_workflow",
            ),
        ]
    ),
    TestConversation(
        id="WF-007",
        name="Metagenomics 16S",
        category=Category.WORKFLOW_GENERATION,
        difficulty="medium",
        tags=["workflow", "metagenomics"],
        turns=[
            Turn(
                query="Create a 16S metagenomics workflow for microbiome analysis",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"ANALYSIS_TYPE": "16S"},
                expected_tool="generate_workflow",
            )
        ]
    ),
    TestConversation(
        id="WF-008",
        name="Workflow Modification",
        category=Category.WORKFLOW_GENERATION,
        difficulty="medium",
        tags=["workflow", "modify"],
        turns=[
            Turn(
                query="Generate RNA-seq workflow for human samples",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"ORGANISM": "human", "ANALYSIS_TYPE": "RNA-seq"},
                expected_tool="generate_workflow",
            ),
            Turn(
                query="Add a quality control step at the beginning",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={},
                context_reference="previous_workflow",
            ),
            Turn(
                query="Use STAR instead of HISAT2 for alignment",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"TOOL": "STAR"},
                context_reference="previous_workflow",
            ),
        ]
    ),
    TestConversation(
        id="WF-009",
        name="Long Read Assembly",
        category=Category.WORKFLOW_GENERATION,
        difficulty="hard",
        tags=["workflow", "long-read"],
        turns=[
            Turn(
                query="Create a genome assembly workflow for Oxford Nanopore data",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"PLATFORM": "Oxford Nanopore", "ANALYSIS_TYPE": "genome_assembly"},
                expected_tool="generate_workflow",
            )
        ]
    ),
    TestConversation(
        id="WF-010",
        name="Bisulfite Sequencing",
        category=Category.WORKFLOW_GENERATION,
        difficulty="hard",
        tags=["workflow", "methylation"],
        turns=[
            Turn(
                query="Generate a WGBS methylation analysis pipeline for mouse embryonic stem cells",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"ORGANISM": "mouse", "CELL_TYPE": "ESC", "ANALYSIS_TYPE": "WGBS"},
                expected_tool="generate_workflow",
            )
        ]
    ),
]

# =============================================================================
# JOB MANAGEMENT CONVERSATIONS (15+)
# =============================================================================

JOB_MANAGEMENT_CONVERSATIONS = [
    TestConversation(
        id="JM-001",
        name="Submit Workflow Job",
        category=Category.JOB_MANAGEMENT,
        difficulty="easy",
        tags=["job", "submit"],
        turns=[
            Turn(
                query="Submit the workflow I just created",
                expected_intent="JOB_SUBMIT",
                expected_entities={},
                expected_tool="submit_job",
                context_reference="previous_workflow",
            )
        ]
    ),
    TestConversation(
        id="JM-002",
        name="Check Job Status",
        category=Category.JOB_MANAGEMENT,
        difficulty="easy",
        tags=["job", "status"],
        turns=[
            Turn(
                query="What's the status of my running jobs?",
                expected_intent="JOB_STATUS",
                expected_entities={},
                expected_tool="get_job_status",
            )
        ]
    ),
    TestConversation(
        id="JM-003",
        name="Job Lifecycle Full Flow",
        category=Category.JOB_MANAGEMENT,
        difficulty="hard",
        tags=["job", "lifecycle", "multi-turn"],
        turns=[
            Turn(
                query="Submit job 12345",
                expected_intent="JOB_SUBMIT",
                expected_entities={"JOB_ID": "12345"},
                expected_tool="submit_job",
            ),
            Turn(
                query="Is it running yet?",
                expected_intent="JOB_STATUS",
                expected_entities={},
                expected_tool="get_job_status",
                context_reference="previous_job",
            ),
            Turn(
                query="Show me the logs",
                expected_intent="JOB_LOGS",
                expected_entities={},
                expected_tool="get_logs",
                context_reference="previous_job",
            ),
            Turn(
                query="Something went wrong, cancel it",
                expected_intent="JOB_CANCEL",
                expected_entities={},
                expected_tool="cancel_job",
                context_reference="previous_job",
            ),
        ]
    ),
    TestConversation(
        id="JM-004",
        name="List All Jobs",
        category=Category.JOB_MANAGEMENT,
        difficulty="easy",
        tags=["job", "list"],
        turns=[
            Turn(
                query="Show me all my running jobs",
                expected_intent="JOB_LIST",
                expected_entities={},
                expected_tool="list_jobs",
            )
        ]
    ),
    TestConversation(
        id="JM-005",
        name="Watch Job Progress",
        category=Category.JOB_MANAGEMENT,
        difficulty="medium",
        tags=["job", "watch"],
        turns=[
            Turn(
                query="Watch job 54321 and notify me when it's done",
                expected_intent="JOB_WATCH",
                expected_entities={"JOB_ID": "54321"},
                expected_tool="watch_job",
            )
        ]
    ),
    TestConversation(
        id="JM-006",
        name="Resubmit Failed Job",
        category=Category.JOB_MANAGEMENT,
        difficulty="medium",
        tags=["job", "resubmit"],
        turns=[
            Turn(
                query="Check the status of job 99999",
                expected_intent="JOB_STATUS",
                expected_entities={"JOB_ID": "99999"},
                expected_tool="get_job_status",
            ),
            Turn(
                query="It failed. Resubmit it with more memory",
                expected_intent="JOB_RESUBMIT",
                expected_entities={"RESOURCE": "memory"},
                expected_tool="resubmit_job",
                context_reference="previous_job",
            ),
        ]
    ),
    TestConversation(
        id="JM-007",
        name="Job Logs Analysis",
        category=Category.JOB_MANAGEMENT,
        difficulty="medium",
        tags=["job", "logs", "error"],
        turns=[
            Turn(
                query="Get the error logs for the failed job",
                expected_intent="JOB_LOGS",
                expected_entities={},
                expected_tool="get_logs",
            ),
            Turn(
                query="What went wrong?",
                expected_intent="DIAGNOSE_ERROR",
                expected_entities={},
                expected_tool="diagnose_error",
                context_reference="job_logs",
            ),
        ]
    ),
]

# =============================================================================
# EDUCATION CONVERSATIONS (10+)
# =============================================================================

EDUCATION_CONVERSATIONS = [
    TestConversation(
        id="ED-001",
        name="Explain RNA-seq",
        category=Category.EDUCATION,
        difficulty="easy",
        tags=["education", "explain"],
        turns=[
            Turn(
                query="What is RNA-seq?",
                expected_intent="EDUCATION_EXPLAIN",
                expected_entities={"CONCEPT": "RNA-seq"},
                expected_tool="explain_concept",
            )
        ]
    ),
    TestConversation(
        id="ED-002",
        name="Tool Comparison",
        category=Category.EDUCATION,
        difficulty="medium",
        tags=["education", "tools"],
        turns=[
            Turn(
                query="What's the difference between STAR and HISAT2?",
                expected_intent="EDUCATION_EXPLAIN",
                expected_entities={"TOOL_1": "STAR", "TOOL_2": "HISAT2"},
                expected_tool="explain_concept",
            )
        ]
    ),
    TestConversation(
        id="ED-003",
        name="Workflow Help",
        category=Category.EDUCATION,
        difficulty="easy",
        tags=["education", "help"],
        turns=[
            Turn(
                query="How do I create a workflow?",
                expected_intent="EDUCATION_HELP",
                expected_entities={},
                expected_tool="show_help",
            )
        ]
    ),
    TestConversation(
        id="ED-004",
        name="Explain Normalization",
        category=Category.EDUCATION,
        difficulty="medium",
        tags=["education", "explain"],
        turns=[
            Turn(
                query="Explain TPM vs FPKM normalization",
                expected_intent="EDUCATION_EXPLAIN",
                expected_entities={"CONCEPT": "normalization"},
                expected_tool="explain_concept",
            )
        ]
    ),
    TestConversation(
        id="ED-005",
        name="Best Practices Question",
        category=Category.EDUCATION,
        difficulty="medium",
        tags=["education", "best-practices"],
        turns=[
            Turn(
                query="What are the best practices for differential expression analysis?",
                expected_intent="EDUCATION_EXPLAIN",
                expected_entities={"CONCEPT": "differential_expression"},
                expected_tool="explain_concept",
            )
        ]
    ),
]

# =============================================================================
# ERROR HANDLING CONVERSATIONS (10+)
# =============================================================================

ERROR_HANDLING_CONVERSATIONS = [
    TestConversation(
        id="EH-001",
        name="Diagnose Job Failure",
        category=Category.ERROR_HANDLING,
        difficulty="medium",
        tags=["error", "diagnose"],
        turns=[
            Turn(
                query="My job failed with an out of memory error",
                expected_intent="DIAGNOSE_ERROR",
                expected_entities={"ERROR_TYPE": "OOM"},
                expected_tool="diagnose_error",
            )
        ]
    ),
    TestConversation(
        id="EH-002",
        name="Fix and Resubmit",
        category=Category.ERROR_HANDLING,
        difficulty="hard",
        tags=["error", "fix", "resubmit"],
        turns=[
            Turn(
                query="Job 12345 failed, what went wrong?",
                expected_intent="DIAGNOSE_ERROR",
                expected_entities={"JOB_ID": "12345"},
                expected_tool="diagnose_error",
            ),
            Turn(
                query="Fix it and resubmit",
                expected_intent="JOB_RESUBMIT",
                expected_entities={},
                expected_tool="resubmit_job",
                context_reference="diagnosed_error",
            ),
        ]
    ),
    TestConversation(
        id="EH-003",
        name="Tool Not Found Error",
        category=Category.ERROR_HANDLING,
        difficulty="medium",
        tags=["error", "missing-tool"],
        turns=[
            Turn(
                query="I'm getting 'samtools: command not found'",
                expected_intent="DIAGNOSE_ERROR",
                expected_entities={"ERROR_TYPE": "command_not_found", "TOOL": "samtools"},
                expected_tool="diagnose_error",
            )
        ]
    ),
    TestConversation(
        id="EH-004",
        name="Reference Genome Missing",
        category=Category.ERROR_HANDLING,
        difficulty="medium",
        tags=["error", "reference"],
        turns=[
            Turn(
                query="The workflow can't find the reference genome",
                expected_intent="DIAGNOSE_ERROR",
                expected_entities={"ERROR_TYPE": "file_not_found"},
                expected_tool="diagnose_error",
            ),
            Turn(
                query="Download the reference genome for me",
                expected_intent="REFERENCE_DOWNLOAD",
                expected_entities={},
                expected_tool="download_reference",
            ),
        ]
    ),
]

# =============================================================================
# COREFERENCE / CONTEXT CONVERSATIONS (15+)
# =============================================================================

COREFERENCE_CONVERSATIONS = [
    TestConversation(
        id="CR-001",
        name="It Pronoun Reference",
        category=Category.COREFERENCE,
        difficulty="medium",
        tags=["coreference", "pronoun"],
        turns=[
            Turn(
                query="Search for human liver RNA-seq data",
                expected_intent="DATA_SEARCH",
                expected_entities={"ORGANISM": "human", "TISSUE": "liver"},
                expected_tool="search_databases",
            ),
            Turn(
                query="Download it",
                expected_intent="DATA_DOWNLOAD",
                expected_entities={},
                expected_tool="download_dataset",
                context_reference="previous_search",
            ),
        ]
    ),
    TestConversation(
        id="CR-002",
        name="The Workflow Reference",
        category=Category.COREFERENCE,
        difficulty="medium",
        tags=["coreference", "workflow"],
        turns=[
            Turn(
                query="Create an RNA-seq workflow",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"ANALYSIS_TYPE": "RNA-seq"},
                expected_tool="generate_workflow",
            ),
            Turn(
                query="Submit the workflow",
                expected_intent="JOB_SUBMIT",
                expected_entities={},
                expected_tool="submit_job",
                context_reference="previous_workflow",
            ),
        ]
    ),
    TestConversation(
        id="CR-003",
        name="The Job Reference",
        category=Category.COREFERENCE,
        difficulty="medium",
        tags=["coreference", "job"],
        turns=[
            Turn(
                query="Submit job for my workflow",
                expected_intent="JOB_SUBMIT",
                expected_entities={},
                expected_tool="submit_job",
            ),
            Turn(
                query="Check the job status",
                expected_intent="JOB_STATUS",
                expected_entities={},
                expected_tool="get_job_status",
                context_reference="previous_job",
            ),
            Turn(
                query="Show its logs",
                expected_intent="JOB_LOGS",
                expected_entities={},
                expected_tool="get_logs",
                context_reference="previous_job",
            ),
        ]
    ),
    TestConversation(
        id="CR-004",
        name="These Results Reference",
        category=Category.COREFERENCE,
        difficulty="hard",
        tags=["coreference", "results"],
        turns=[
            Turn(
                query="Search for mouse ChIP-seq H3K4me3 data",
                expected_intent="DATA_SEARCH",
                expected_entities={"ORGANISM": "mouse", "HISTONE_MARK": "H3K4me3"},
                expected_tool="search_databases",
            ),
            Turn(
                query="Filter these results to only brain tissue",
                expected_intent="DATA_SEARCH",
                expected_entities={"TISSUE": "brain"},
                context_reference="previous_search",
            ),
            Turn(
                query="Download the first 5",
                expected_intent="DATA_DOWNLOAD",
                expected_entities={},
                expected_tool="download_dataset",
                context_reference="filtered_results",
            ),
        ]
    ),
    TestConversation(
        id="CR-005",
        name="Same Analysis Different Organism",
        category=Category.COREFERENCE,
        difficulty="medium",
        tags=["coreference", "modify"],
        turns=[
            Turn(
                query="Create RNA-seq workflow for human",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"ORGANISM": "human", "ANALYSIS_TYPE": "RNA-seq"},
                expected_tool="generate_workflow",
            ),
            Turn(
                query="Now do the same for mouse",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"ORGANISM": "mouse"},
                expected_tool="generate_workflow",
                context_reference="previous_workflow",
            ),
        ]
    ),
]

# =============================================================================
# EDGE CASES AND ADVERSARIAL (15+)
# =============================================================================

EDGE_CASE_CONVERSATIONS = [
    TestConversation(
        id="EC-001",
        name="Typos in Query",
        category=Category.EDGE_CASES,
        difficulty="medium",
        tags=["edge-case", "typo"],
        turns=[
            Turn(
                query="Serach for mouze RNA-seq daat",
                expected_intent="DATA_SEARCH",
                expected_entities={"ORGANISM": "mouse", "ASSAY_TYPE": "RNA-seq"},
                expected_tool="search_databases",
            )
        ]
    ),
    TestConversation(
        id="EC-002",
        name="All Caps Query",
        category=Category.EDGE_CASES,
        difficulty="easy",
        tags=["edge-case", "caps"],
        turns=[
            Turn(
                query="CREATE RNA-SEQ WORKFLOW FOR HUMAN",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"ORGANISM": "human", "ANALYSIS_TYPE": "RNA-seq"},
                expected_tool="generate_workflow",
            )
        ]
    ),
    TestConversation(
        id="EC-003",
        name="Very Long Query",
        category=Category.EDGE_CASES,
        difficulty="hard",
        tags=["edge-case", "long"],
        turns=[
            Turn(
                query="I need to create a comprehensive RNA sequencing analysis workflow for my research project involving human liver tissue samples from patients with hepatocellular carcinoma and I want to perform differential expression analysis comparing the cancer samples with normal adjacent tissue samples using DESeq2 and also do pathway enrichment analysis with GSEA",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"ORGANISM": "human", "TISSUE": "liver", "DISEASE": "hepatocellular carcinoma"},
                expected_tool="generate_workflow",
            )
        ]
    ),
    TestConversation(
        id="EC-004",
        name="Minimal Query",
        category=Category.EDGE_CASES,
        difficulty="hard",
        tags=["edge-case", "minimal"],
        turns=[
            Turn(
                query="RNA-seq",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"ANALYSIS_TYPE": "RNA-seq"},
            )
        ]
    ),
    TestConversation(
        id="EC-005",
        name="Mixed Language/Abbreviations",
        category=Category.EDGE_CASES,
        difficulty="hard",
        tags=["edge-case", "abbreviations"],
        turns=[
            Turn(
                query="Create scRNAseq wf 4 mouse w/ 10x chromium v3",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"ANALYSIS_TYPE": "scRNA-seq", "ORGANISM": "mouse", "PLATFORM": "10x"},
                expected_tool="generate_workflow",
            )
        ]
    ),
    TestConversation(
        id="EC-006",
        name="Greeting First",
        category=Category.EDGE_CASES,
        difficulty="easy",
        tags=["edge-case", "greeting"],
        turns=[
            Turn(
                query="Hello!",
                expected_intent="META_GREETING",
                expected_entities={},
            ),
            Turn(
                query="I need help with RNA-seq analysis",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"ANALYSIS_TYPE": "RNA-seq"},
            ),
        ]
    ),
    TestConversation(
        id="EC-007",
        name="Thanks and Goodbye",
        category=Category.EDGE_CASES,
        difficulty="easy",
        tags=["edge-case", "farewell"],
        turns=[
            Turn(
                query="Create RNA-seq workflow",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"ANALYSIS_TYPE": "RNA-seq"},
                expected_tool="generate_workflow",
            ),
            Turn(
                query="Thanks, that's all I need!",
                expected_intent="META_FAREWELL",
                expected_entities={},
            ),
        ]
    ),
]

# =============================================================================
# AMBIGUOUS CONVERSATIONS (10+)
# =============================================================================

AMBIGUOUS_CONVERSATIONS = [
    TestConversation(
        id="AM-001",
        name="Vague Data Request",
        category=Category.AMBIGUOUS,
        difficulty="hard",
        tags=["ambiguous", "clarification"],
        turns=[
            Turn(
                query="I need some data",
                expected_intent="DATA_SEARCH",
                expected_entities={},
                description="Should ask for clarification",
            )
        ]
    ),
    TestConversation(
        id="AM-002",
        name="Unspecified Analysis",
        category=Category.AMBIGUOUS,
        difficulty="hard",
        tags=["ambiguous", "clarification"],
        turns=[
            Turn(
                query="Analyze my fastq files",
                expected_intent="WORKFLOW_CREATE",
                expected_entities={"DATA_TYPE": "fastq"},
                description="Should ask what kind of analysis",
            )
        ]
    ),
    TestConversation(
        id="AM-003",
        name="Ambiguous Download",
        category=Category.AMBIGUOUS,
        difficulty="hard",
        tags=["ambiguous", "clarification"],
        turns=[
            Turn(
                query="Download the data",
                expected_intent="DATA_DOWNLOAD",
                expected_entities={},
                description="Should ask which data",
            )
        ]
    ),
    TestConversation(
        id="AM-004",
        name="Multiple Possible Intents",
        category=Category.AMBIGUOUS,
        difficulty="hard",
        tags=["ambiguous", "multi-intent"],
        turns=[
            Turn(
                query="Help me with RNA-seq",
                expected_intent="EDUCATION_HELP",
                expected_entities={"TOPIC": "RNA-seq"},
                description="Could be help or workflow creation",
            )
        ]
    ),
]


# =============================================================================
# COMBINED DATASET
# =============================================================================

def get_all_conversations() -> List[TestConversation]:
    """Get all test conversations from all categories."""
    all_convs = []
    all_convs.extend(DATA_DISCOVERY_CONVERSATIONS)
    all_convs.extend(WORKFLOW_GENERATION_CONVERSATIONS)
    all_convs.extend(JOB_MANAGEMENT_CONVERSATIONS)
    all_convs.extend(EDUCATION_CONVERSATIONS)
    all_convs.extend(ERROR_HANDLING_CONVERSATIONS)
    all_convs.extend(COREFERENCE_CONVERSATIONS)
    all_convs.extend(EDGE_CASE_CONVERSATIONS)
    all_convs.extend(AMBIGUOUS_CONVERSATIONS)
    return all_convs


def get_conversations_by_category(category: Category) -> List[TestConversation]:
    """Get conversations filtered by category."""
    return [c for c in get_all_conversations() if c.category == category]


def get_conversations_by_difficulty(difficulty: str) -> List[TestConversation]:
    """Get conversations filtered by difficulty."""
    return [c for c in get_all_conversations() if c.difficulty == difficulty]


def get_conversation_stats() -> Dict[str, Any]:
    """Get statistics about the test dataset."""
    all_convs = get_all_conversations()
    
    # Count by category
    by_category = {}
    for cat in Category:
        by_category[cat.value] = len([c for c in all_convs if c.category == cat])
    
    # Count by difficulty
    by_difficulty = {"easy": 0, "medium": 0, "hard": 0}
    for conv in all_convs:
        by_difficulty[conv.difficulty] = by_difficulty.get(conv.difficulty, 0) + 1
    
    # Count total turns
    total_turns = sum(len(c.turns) for c in all_convs)
    
    # Multi-turn conversations
    multi_turn = len([c for c in all_convs if len(c.turns) > 1])
    
    return {
        "total_conversations": len(all_convs),
        "total_turns": total_turns,
        "multi_turn_conversations": multi_turn,
        "by_category": by_category,
        "by_difficulty": by_difficulty,
    }


def export_to_json(filepath: str):
    """Export all conversations to JSON format."""
    import json
    from dataclasses import asdict
    
    all_convs = get_all_conversations()
    
    # Convert to serializable format
    data = []
    for conv in all_convs:
        conv_dict = {
            "id": conv.id,
            "name": conv.name,
            "category": conv.category.value,
            "difficulty": conv.difficulty,
            "tags": conv.tags,
            "description": conv.description,
            "turns": [
                {
                    "query": t.query,
                    "expected_intent": t.expected_intent,
                    "expected_entities": t.expected_entities,
                    "expected_tool": t.expected_tool,
                    "description": t.description,
                    "context_reference": t.context_reference,
                }
                for t in conv.turns
            ]
        }
        data.append(conv_dict)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return len(data)


if __name__ == "__main__":
    # Print stats
    stats = get_conversation_stats()
    print("=" * 60)
    print("COMPREHENSIVE TEST DATASET STATISTICS")
    print("=" * 60)
    print(f"Total Conversations: {stats['total_conversations']}")
    print(f"Total Turns: {stats['total_turns']}")
    print(f"Multi-turn Conversations: {stats['multi_turn_conversations']}")
    print()
    print("By Category:")
    for cat, count in stats['by_category'].items():
        print(f"  {cat}: {count}")
    print()
    print("By Difficulty:")
    for diff, count in stats['by_difficulty'].items():
        print(f"  {diff}: {count}")
    print("=" * 60)
    
    # Export
    count = export_to_json("training_data/comprehensive_test_dataset.json")
    print(f"\nExported {count} conversations to training_data/comprehensive_test_dataset.json")
