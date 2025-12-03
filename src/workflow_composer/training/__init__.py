"""
Training Data Collection Module
===============================

This module provides infrastructure for collecting, generating, and
preparing training data for fine-tuning the BioPipelines LLM.

Components:
- TrainingDataGenerator: Generate synthetic training examples
- InteractionLogger: Log real user interactions
- TrainingDataPipeline: Process and validate training data
- DataExporter: Export to various training formats

Example:
    >>> from workflow_composer.training import TrainingDataGenerator
    >>> generator = TrainingDataGenerator()
    >>> examples = await generator.generate_from_golden_queries()
    >>> print(f"Generated {len(examples)} training examples")
"""

from .config import (
    TrainingConfig,
    GeneratorConfig,
    LoggerConfig,
    PipelineConfig,
    ExportConfig,
    VariationType,
    get_default_config,
)

from .data_generator import (
    TrainingDataGenerator,
)

from .interaction_logger import (
    InteractionLogger,
)

from .data_pipeline import (
    TrainingDataPipeline,
    DataValidator,
    QualityScorer,
    QualityMetrics,
    process_training_data,
)

from .export import (
    TrainingDataExporter,
    ExportResult,
    OpenAIChatExporter,
    AlpacaExporter,
    ShareGPTExporter,
    AxolotlExporter,
    export_training_data,
    export_all_formats,
)

__all__ = [
    # Config
    "TrainingConfig",
    "GeneratorConfig",
    "LoggerConfig",
    "PipelineConfig",
    "ExportConfig",
    "VariationType",
    "get_default_config",
    # Generator
    "TrainingDataGenerator",
    # Logger
    "InteractionLogger",
    # Pipeline
    "TrainingDataPipeline",
    "DataValidator",
    "QualityScorer",
    "QualityMetrics",
    "process_training_data",
    # Export
    "TrainingDataExporter",
    "ExportResult",
    "OpenAIChatExporter",
    "AlpacaExporter",
    "ShareGPTExporter",
    "AxolotlExporter",
    "export_training_data",
    "export_all_formats",
]


def get_data_generator(config: GeneratorConfig = None) -> TrainingDataGenerator:
    """Factory function to create a data generator."""
    return TrainingDataGenerator(config)


def get_interaction_logger(config: LoggerConfig = None) -> InteractionLogger:
    """Factory function to create an interaction logger."""
    return InteractionLogger(config)


def get_pipeline(config: PipelineConfig = None) -> TrainingDataPipeline:
    """Factory function to create a data pipeline."""
    return TrainingDataPipeline(config)


def get_exporter(config: ExportConfig = None) -> TrainingDataExporter:
    """Factory function to create a data exporter."""
    return TrainingDataExporter(config)
