"""
Training Configuration
======================

Configuration classes for training data collection and processing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from enum import Enum


class VariationType(Enum):
    """Types of query variations to generate."""
    FORMAL = "formal"           # Technical, precise language
    CASUAL = "casual"           # Conversational language  
    MINIMAL = "minimal"         # Short, abbreviated queries
    DETAILED = "detailed"       # Long, comprehensive queries
    EDGE_CASE = "edge_case"     # Typos, incomplete, ambiguous


@dataclass
class GeneratorConfig:
    """Configuration for synthetic data generation."""
    
    # Variation settings
    variations_per_query: int = 50
    variation_types: List[VariationType] = field(
        default_factory=lambda: list(VariationType)
    )
    
    # LLM settings for generation
    llm_model: str = "gemini-1.5-flash"
    temperature: float = 0.8
    max_tokens: int = 1024
    
    # Validation settings
    validate_intents: bool = True
    validate_workflows: bool = True
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("data/training/synthetic"))
    
    # Batch settings
    batch_size: int = 10
    max_concurrent: int = 5


@dataclass
class LoggerConfig:
    """Configuration for interaction logging."""
    
    # Storage settings
    storage_dir: Path = field(default_factory=lambda: Path("data/training/interactions"))
    db_file: str = "interactions.db"
    
    # Quality thresholds
    min_confidence: float = 0.5
    min_quality_score: float = 0.6
    
    # Privacy settings
    anonymize: bool = True
    remove_pii: bool = True
    
    # Retention
    max_age_days: int = 365


@dataclass
class PipelineConfig:
    """Configuration for data processing pipeline."""
    
    # Input/output
    raw_dir: Path = field(default_factory=lambda: Path("data/training/raw"))
    processed_dir: Path = field(default_factory=lambda: Path("data/training/processed"))
    
    # Quality filtering
    min_quality_score: float = 0.7
    require_validated_workflow: bool = True
    require_tool_coverage: bool = True
    
    # Deduplication
    similarity_threshold: float = 0.95
    deduplicate: bool = True
    
    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Stratification
    stratify_by: str = "analysis_type"


@dataclass
class ExportConfig:
    """Configuration for data export."""
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("data/training/export"))
    
    # Format settings
    formats: List[str] = field(default_factory=lambda: ["jsonl", "huggingface"])
    
    # Content settings
    include_metadata: bool = True
    include_system_prompt: bool = True
    
    # System prompt for training
    system_prompt: str = """You are BioPipelines, an expert bioinformatics assistant that helps researchers create analysis workflows. You understand:
- Sequencing technologies (Illumina, PacBio, Nanopore)
- Analysis types (RNA-seq, ChIP-seq, variant calling, metagenomics, etc.)
- Bioinformatics tools (STAR, BWA, GATK, MACS2, etc.)
- Workflow engines (Nextflow, Snakemake)

When given a query, extract the analysis intent and generate appropriate Nextflow DSL2 workflows. Explain your tool choices and provide helpful guidance."""


@dataclass
class TrainingConfig:
    """Master configuration for training data collection."""
    
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    
    # Global settings
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent.parent)
    
    # Target dataset size
    target_synthetic: int = 9000
    target_interactions: int = 1000
    target_total: int = 10000
    
    def ensure_directories(self):
        """Create all required directories."""
        dirs = [
            self.generator.output_dir,
            self.logger.storage_dir,
            self.pipeline.raw_dir,
            self.pipeline.processed_dir,
            self.export.output_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    config = TrainingConfig()
    config.ensure_directories()
    return config
