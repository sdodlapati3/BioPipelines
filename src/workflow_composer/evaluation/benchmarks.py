"""Benchmark definitions for BioPipelines evaluation.

This module defines benchmark queries across different categories
to test the agentic system's capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import json
from pathlib import Path


class BenchmarkCategory(Enum):
    """Categories of benchmark queries."""
    
    DATA_DISCOVERY = "data_discovery"
    WORKFLOW_GENERATION = "workflow_generation"
    JOB_MANAGEMENT = "job_management"
    EDUCATION = "education"
    MULTI_STEP = "multi_step"


class ExpectedBehavior(Enum):
    """Expected system behavior types."""
    
    TOOL_CALL = "tool_call"           # Should call a specific tool
    MULTI_TOOL = "multi_tool"         # Should call multiple tools
    TEXT_RESPONSE = "text_response"   # Should respond with text
    ERROR_HANDLING = "error_handling" # Should handle error gracefully


@dataclass
class BenchmarkQuery:
    """A single benchmark query.
    
    Attributes:
        id: Unique identifier for this query
        query: The user query text
        category: Category this query belongs to
        expected_behavior: What the system should do
        expected_tools: List of tools expected to be called
        expected_keywords: Keywords expected in the response
        ground_truth: Optional ground truth answer
        difficulty: Difficulty level (1-5)
        tags: Additional tags for filtering
    """
    
    id: str
    query: str
    category: BenchmarkCategory
    expected_behavior: ExpectedBehavior
    expected_tools: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
    ground_truth: Optional[str] = None
    difficulty: int = 1
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "query": self.query,
            "category": self.category.value,
            "expected_behavior": self.expected_behavior.value,
            "expected_tools": self.expected_tools,
            "expected_keywords": self.expected_keywords,
            "ground_truth": self.ground_truth,
            "difficulty": self.difficulty,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkQuery":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            query=data["query"],
            category=BenchmarkCategory(data["category"]),
            expected_behavior=ExpectedBehavior(data["expected_behavior"]),
            expected_tools=data.get("expected_tools", []),
            expected_keywords=data.get("expected_keywords", []),
            ground_truth=data.get("ground_truth"),
            difficulty=data.get("difficulty", 1),
            tags=data.get("tags", []),
        )


@dataclass
class Benchmark:
    """A collection of benchmark queries.
    
    Attributes:
        name: Name of the benchmark
        description: Description of what this benchmark tests
        version: Version string
        queries: List of benchmark queries
    """
    
    name: str
    description: str
    version: str
    queries: List[BenchmarkQuery]
    
    def filter_by_category(
        self, category: BenchmarkCategory
    ) -> List[BenchmarkQuery]:
        """Get queries for a specific category."""
        return [q for q in self.queries if q.category == category]
    
    def filter_by_difficulty(
        self, max_difficulty: int
    ) -> List[BenchmarkQuery]:
        """Get queries up to a difficulty level."""
        return [q for q in self.queries if q.difficulty <= max_difficulty]
    
    def filter_by_tags(self, tags: List[str]) -> List[BenchmarkQuery]:
        """Get queries that have any of the specified tags."""
        return [q for q in self.queries if any(t in q.tags for t in tags)]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "queries": [q.to_dict() for q in self.queries],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Benchmark":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            version=data["version"],
            queries=[BenchmarkQuery.from_dict(q) for q in data["queries"]],
        )
    
    def save(self, path: str) -> None:
        """Save benchmark to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "Benchmark":
        """Load benchmark from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


# -----------------------------------------------------------------------------
# Built-in Benchmark Queries
# -----------------------------------------------------------------------------

DISCOVERY_BENCHMARKS = [
    # Basic search queries
    BenchmarkQuery(
        id="disc_001",
        query="Find RNA-seq data for human liver",
        category=BenchmarkCategory.DATA_DISCOVERY,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["search_datasets"],
        expected_keywords=["RNA-seq", "liver", "human", "dataset"],
        difficulty=1,
        tags=["search", "rna-seq", "basic"],
    ),
    BenchmarkQuery(
        id="disc_002",
        query="Search for ChIP-seq data with H3K27ac in K562 cells from ENCODE",
        category=BenchmarkCategory.DATA_DISCOVERY,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["search_datasets"],
        expected_keywords=["ChIP-seq", "H3K27ac", "K562", "ENCODE"],
        difficulty=2,
        tags=["search", "chip-seq", "encode"],
    ),
    BenchmarkQuery(
        id="disc_003",
        query="Find ATAC-seq datasets for cancer samples in GDC",
        category=BenchmarkCategory.DATA_DISCOVERY,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["search_datasets"],
        expected_keywords=["ATAC-seq", "cancer", "GDC"],
        difficulty=2,
        tags=["search", "atac-seq", "gdc"],
    ),
    BenchmarkQuery(
        id="disc_004",
        query="What single-cell RNA-seq datasets are available for mouse brain?",
        category=BenchmarkCategory.DATA_DISCOVERY,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["search_datasets"],
        expected_keywords=["single-cell", "scRNA-seq", "mouse", "brain"],
        difficulty=2,
        tags=["search", "scrna-seq"],
    ),
    BenchmarkQuery(
        id="disc_005",
        query="List available data sources",
        category=BenchmarkCategory.DATA_DISCOVERY,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["list_data_sources"],
        expected_keywords=["ENCODE", "GEO", "GDC", "source"],
        difficulty=1,
        tags=["list", "sources"],
    ),
    
    # Detail queries
    BenchmarkQuery(
        id="disc_006",
        query="Get details for dataset ENCSR123ABC",
        category=BenchmarkCategory.DATA_DISCOVERY,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["get_dataset_details"],
        expected_keywords=["ENCSR123ABC", "details"],
        difficulty=1,
        tags=["details", "encode"],
    ),
    BenchmarkQuery(
        id="disc_007",
        query="How many files are in the GSE12345 series?",
        category=BenchmarkCategory.DATA_DISCOVERY,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["get_dataset_details"],
        expected_keywords=["files", "GSE12345"],
        difficulty=2,
        tags=["details", "geo"],
    ),
    
    # Complex search
    BenchmarkQuery(
        id="disc_008",
        query="Find paired-end RNA-seq data from healthy human lung tissue with at least 3 replicates",
        category=BenchmarkCategory.DATA_DISCOVERY,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["search_datasets"],
        expected_keywords=["paired-end", "RNA-seq", "lung", "replicate"],
        difficulty=4,
        tags=["search", "complex", "rna-seq"],
    ),
    BenchmarkQuery(
        id="disc_009",
        query="Search across all sources for methylation data in breast cancer",
        category=BenchmarkCategory.DATA_DISCOVERY,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["search_datasets"],
        expected_keywords=["methylation", "breast cancer"],
        difficulty=3,
        tags=["search", "methylation", "federated"],
    ),
    BenchmarkQuery(
        id="disc_010",
        query="Find Hi-C data for chromatin interactions in stem cells",
        category=BenchmarkCategory.DATA_DISCOVERY,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["search_datasets"],
        expected_keywords=["Hi-C", "chromatin", "stem cells"],
        difficulty=3,
        tags=["search", "hic"],
    ),
]

WORKFLOW_BENCHMARKS = [
    # Basic workflow generation
    BenchmarkQuery(
        id="wf_001",
        query="Create an RNA-seq analysis pipeline",
        category=BenchmarkCategory.WORKFLOW_GENERATION,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["generate_workflow"],
        expected_keywords=["workflow", "RNA-seq", "pipeline"],
        difficulty=2,
        tags=["workflow", "rna-seq"],
    ),
    BenchmarkQuery(
        id="wf_002",
        query="Generate a ChIP-seq peak calling workflow",
        category=BenchmarkCategory.WORKFLOW_GENERATION,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["generate_workflow"],
        expected_keywords=["ChIP-seq", "peak calling", "workflow"],
        difficulty=2,
        tags=["workflow", "chip-seq"],
    ),
    BenchmarkQuery(
        id="wf_003",
        query="Build a variant calling pipeline for whole genome sequencing",
        category=BenchmarkCategory.WORKFLOW_GENERATION,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["generate_workflow"],
        expected_keywords=["variant calling", "WGS", "pipeline"],
        difficulty=3,
        tags=["workflow", "wgs", "variant"],
    ),
    BenchmarkQuery(
        id="wf_004",
        query="Create a differential expression analysis workflow",
        category=BenchmarkCategory.WORKFLOW_GENERATION,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["generate_workflow"],
        expected_keywords=["differential expression", "DESeq2", "edgeR"],
        difficulty=2,
        tags=["workflow", "rna-seq", "de"],
    ),
    
    # Complex workflows
    BenchmarkQuery(
        id="wf_005",
        query="Design a multi-omics integration pipeline combining RNA-seq and ATAC-seq data",
        category=BenchmarkCategory.WORKFLOW_GENERATION,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["generate_workflow"],
        expected_keywords=["multi-omics", "integration", "RNA-seq", "ATAC-seq"],
        difficulty=4,
        tags=["workflow", "multi-omics", "complex"],
    ),
    BenchmarkQuery(
        id="wf_006",
        query="Generate a single-cell RNA-seq analysis workflow with clustering and trajectory analysis",
        category=BenchmarkCategory.WORKFLOW_GENERATION,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["generate_workflow"],
        expected_keywords=["scRNA-seq", "clustering", "trajectory"],
        difficulty=4,
        tags=["workflow", "scrna-seq", "complex"],
    ),
    
    # Tool-specific
    BenchmarkQuery(
        id="wf_007",
        query="What tools are available for RNA-seq analysis?",
        category=BenchmarkCategory.WORKFLOW_GENERATION,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["list_available_tools"],
        expected_keywords=["STAR", "HISAT2", "salmon", "tool"],
        difficulty=1,
        tags=["tools", "rna-seq"],
    ),
    BenchmarkQuery(
        id="wf_008",
        query="Show me the available containers for bioinformatics",
        category=BenchmarkCategory.WORKFLOW_GENERATION,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["list_containers"],
        expected_keywords=["container", "docker", "singularity"],
        difficulty=1,
        tags=["tools", "containers"],
    ),
]

EDUCATION_BENCHMARKS = [
    # Explanatory queries
    BenchmarkQuery(
        id="edu_001",
        query="What is RNA-seq and how is it used in research?",
        category=BenchmarkCategory.EDUCATION,
        expected_behavior=ExpectedBehavior.TEXT_RESPONSE,
        expected_tools=[],
        expected_keywords=["RNA-seq", "sequencing", "transcriptome", "expression"],
        difficulty=1,
        tags=["education", "rna-seq", "explanation"],
    ),
    BenchmarkQuery(
        id="edu_002",
        query="Explain the difference between ChIP-seq and ATAC-seq",
        category=BenchmarkCategory.EDUCATION,
        expected_behavior=ExpectedBehavior.TEXT_RESPONSE,
        expected_tools=[],
        expected_keywords=["ChIP-seq", "ATAC-seq", "chromatin", "accessibility"],
        difficulty=2,
        tags=["education", "epigenomics"],
    ),
    BenchmarkQuery(
        id="edu_003",
        query="What is the purpose of peak calling in ChIP-seq analysis?",
        category=BenchmarkCategory.EDUCATION,
        expected_behavior=ExpectedBehavior.TEXT_RESPONSE,
        expected_tools=[],
        expected_keywords=["peak", "enrichment", "binding", "signal"],
        difficulty=2,
        tags=["education", "chip-seq"],
    ),
    BenchmarkQuery(
        id="edu_004",
        query="Why do we need to normalize RNA-seq data?",
        category=BenchmarkCategory.EDUCATION,
        expected_behavior=ExpectedBehavior.TEXT_RESPONSE,
        expected_tools=[],
        expected_keywords=["normalization", "library size", "comparison", "bias"],
        difficulty=2,
        tags=["education", "rna-seq", "normalization"],
    ),
    BenchmarkQuery(
        id="edu_005",
        query="What are the key quality control steps in NGS data analysis?",
        category=BenchmarkCategory.EDUCATION,
        expected_behavior=ExpectedBehavior.TEXT_RESPONSE,
        expected_tools=[],
        expected_keywords=["QC", "quality", "FastQC", "adapter", "trimming"],
        difficulty=2,
        tags=["education", "qc"],
    ),
    
    # Best practices
    BenchmarkQuery(
        id="edu_006",
        query="What are best practices for RNA-seq experimental design?",
        category=BenchmarkCategory.EDUCATION,
        expected_behavior=ExpectedBehavior.TEXT_RESPONSE,
        expected_tools=[],
        expected_keywords=["replicates", "sequencing depth", "design", "control"],
        difficulty=3,
        tags=["education", "best-practices"],
    ),
    BenchmarkQuery(
        id="edu_007",
        query="How do I choose between STAR and HISAT2 for alignment?",
        category=BenchmarkCategory.EDUCATION,
        expected_behavior=ExpectedBehavior.TEXT_RESPONSE,
        expected_tools=[],
        expected_keywords=["STAR", "HISAT2", "alignment", "memory", "speed"],
        difficulty=3,
        tags=["education", "tools", "alignment"],
    ),
]

JOB_MANAGEMENT_BENCHMARKS = [
    BenchmarkQuery(
        id="job_001",
        query="Show me my running jobs",
        category=BenchmarkCategory.JOB_MANAGEMENT,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["list_jobs"],
        expected_keywords=["job", "running", "status"],
        difficulty=1,
        tags=["jobs", "list"],
    ),
    BenchmarkQuery(
        id="job_002",
        query="What is the status of job 12345?",
        category=BenchmarkCategory.JOB_MANAGEMENT,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["get_job_status"],
        expected_keywords=["status", "12345"],
        difficulty=1,
        tags=["jobs", "status"],
    ),
    BenchmarkQuery(
        id="job_003",
        query="Cancel job 12345",
        category=BenchmarkCategory.JOB_MANAGEMENT,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["cancel_job"],
        expected_keywords=["cancel", "12345"],
        difficulty=1,
        tags=["jobs", "cancel"],
    ),
    BenchmarkQuery(
        id="job_004",
        query="Submit the RNA-seq workflow to the cluster",
        category=BenchmarkCategory.JOB_MANAGEMENT,
        expected_behavior=ExpectedBehavior.TOOL_CALL,
        expected_tools=["submit_workflow"],
        expected_keywords=["submit", "workflow", "cluster"],
        difficulty=2,
        tags=["jobs", "submit"],
    ),
]

MULTI_STEP_BENCHMARKS = [
    BenchmarkQuery(
        id="multi_001",
        query="Find ChIP-seq data for H3K4me3 in human cells and create a peak calling workflow",
        category=BenchmarkCategory.MULTI_STEP,
        expected_behavior=ExpectedBehavior.MULTI_TOOL,
        expected_tools=["search_datasets", "generate_workflow"],
        expected_keywords=["ChIP-seq", "H3K4me3", "workflow", "peak calling"],
        difficulty=4,
        tags=["multi-step", "search", "workflow"],
    ),
    BenchmarkQuery(
        id="multi_002",
        query="Search for RNA-seq data from liver samples, download dataset ENCSR123ABC, and create an analysis workflow",
        category=BenchmarkCategory.MULTI_STEP,
        expected_behavior=ExpectedBehavior.MULTI_TOOL,
        expected_tools=["search_datasets", "get_dataset_details", "generate_workflow"],
        expected_keywords=["RNA-seq", "liver", "download", "workflow"],
        difficulty=5,
        tags=["multi-step", "complex"],
    ),
    BenchmarkQuery(
        id="multi_003",
        query="Find available ATAC-seq datasets, explain ATAC-seq analysis, and suggest a workflow",
        category=BenchmarkCategory.MULTI_STEP,
        expected_behavior=ExpectedBehavior.MULTI_TOOL,
        expected_tools=["search_datasets", "generate_workflow"],
        expected_keywords=["ATAC-seq", "chromatin", "accessibility", "workflow"],
        difficulty=4,
        tags=["multi-step", "education", "workflow"],
    ),
]


def load_benchmarks(
    categories: Optional[List[BenchmarkCategory]] = None,
    max_difficulty: int = 5,
) -> Benchmark:
    """Load built-in benchmarks.
    
    Args:
        categories: Optional list of categories to include
        max_difficulty: Maximum difficulty level to include
        
    Returns:
        Benchmark object with queries
    """
    all_queries = (
        DISCOVERY_BENCHMARKS +
        WORKFLOW_BENCHMARKS +
        EDUCATION_BENCHMARKS +
        JOB_MANAGEMENT_BENCHMARKS +
        MULTI_STEP_BENCHMARKS
    )
    
    # Filter by category
    if categories:
        all_queries = [q for q in all_queries if q.category in categories]
    
    # Filter by difficulty
    all_queries = [q for q in all_queries if q.difficulty <= max_difficulty]
    
    return Benchmark(
        name="BioPipelines Standard Benchmark",
        description="Standard benchmark suite for evaluating BioPipelines agent",
        version="1.0.0",
        queries=all_queries,
    )


def load_benchmark_from_file(path: str) -> Benchmark:
    """Load a custom benchmark from a JSON file.
    
    Args:
        path: Path to the JSON benchmark file
        
    Returns:
        Benchmark object
    """
    return Benchmark.load(path)


__all__ = [
    "BenchmarkCategory",
    "ExpectedBehavior",
    "BenchmarkQuery",
    "Benchmark",
    "load_benchmarks",
    "load_benchmark_from_file",
    "DISCOVERY_BENCHMARKS",
    "WORKFLOW_BENCHMARKS",
    "EDUCATION_BENCHMARKS",
    "JOB_MANAGEMENT_BENCHMARKS",
    "MULTI_STEP_BENCHMARKS",
]
