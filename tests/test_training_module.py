"""
Tests for Training Data Collection Module
=========================================

Tests synthetic data generation, interaction logging, pipeline processing,
and export functionality.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from workflow_composer.training.config import (
    TrainingConfig,
    GeneratorConfig,
    LoggerConfig,
    PipelineConfig,
    ExportConfig,
    VariationType,
    get_default_config,
)
from workflow_composer.training.data_generator import TrainingDataGenerator
from workflow_composer.training.interaction_logger import InteractionLogger
from workflow_composer.training.data_pipeline import (
    TrainingDataPipeline,
    DataValidator,
    QualityScorer,
    QualityMetrics,
)
from workflow_composer.training.export import (
    TrainingDataExporter,
    OpenAIChatExporter,
    AlpacaExporter,
    ShareGPTExporter,
    AxolotlExporter,
    ExportResult,
)


# ============================================================================
# Config Tests
# ============================================================================

class TestConfig:
    """Tests for configuration classes."""
    
    def test_training_config_default(self):
        """Test default training configuration."""
        config = TrainingConfig()
        
        assert config.target_synthetic == 9000
        assert config.target_total == 10000
        assert isinstance(config.generator, GeneratorConfig)
        assert isinstance(config.logger, LoggerConfig)
        assert isinstance(config.pipeline, PipelineConfig)
        assert isinstance(config.export, ExportConfig)
    
    def test_generator_config(self):
        """Test generator configuration."""
        config = GeneratorConfig()
        
        assert config.variations_per_query == 50
        assert config.temperature == 0.8
        assert config.validate_intents is True
        assert len(config.variation_types) == len(VariationType)
    
    def test_pipeline_config(self):
        """Test pipeline configuration."""
        config = PipelineConfig()
        
        assert config.train_ratio == 0.8
        assert config.val_ratio == 0.1
        assert config.test_ratio == 0.1
        assert config.train_ratio + config.val_ratio + config.test_ratio == 1.0
    
    def test_export_config_system_prompt(self):
        """Test export configuration has system prompt."""
        config = ExportConfig()
        
        assert "bioinformatics" in config.system_prompt.lower()
        assert len(config.system_prompt) > 100


# ============================================================================
# Data Generator Tests
# ============================================================================

class TestTrainingDataGenerator:
    """Tests for TrainingDataGenerator."""
    
    @pytest.fixture
    def generator(self, tmp_path):
        """Create generator with temp output dir."""
        config = GeneratorConfig(output_dir=tmp_path / "synthetic")
        return TrainingDataGenerator(config)
    
    def test_generator_init(self, generator):
        """Test generator initialization."""
        assert generator.config is not None
        # Load golden queries to verify they can be loaded
        queries = generator._load_golden_queries()
        assert queries is not None
    
    def test_load_golden_queries(self, generator):
        """Test golden queries are loaded."""
        # Should load from tests/test_golden_queries.py
        queries = generator._load_golden_queries()
        assert len(queries) >= 50  # We have 100 golden queries
    
    @pytest.mark.asyncio
    async def test_generate_variation_prompt(self, generator):
        """Test variation prompt generation."""
        query = "Analyze RNA-seq data from mouse"
        
        prompt = generator._build_variation_prompt(query, VariationType.FORMAL, 5)
        
        assert "RNA-seq" in prompt or "rna" in prompt.lower()
        assert "formal" in prompt.lower() or "technical" in prompt.lower()
    
    def test_parse_variations(self, generator):
        """Test parsing LLM response into variations."""
        response = """1. Perform RNA-seq analysis on mouse samples
2. I need to analyze RNA sequencing data from mice
3. Run differential expression analysis on mouse RNA data"""
        
        base_query = "Analyze RNA-seq data from mouse"
        variations = generator._parse_variations(response, VariationType.FORMAL, base_query)
        
        assert len(variations) >= 2


# ============================================================================
# Interaction Logger Tests
# ============================================================================

class TestInteractionLogger:
    """Tests for InteractionLogger."""
    
    @pytest.fixture
    def logger(self, tmp_path):
        """Create logger with temp storage."""
        config = LoggerConfig(storage_dir=tmp_path / "interactions")
        return InteractionLogger(config)
    
    def test_logger_init(self, logger):
        """Test logger initialization."""
        assert logger.config is not None
        assert logger._db_path.exists()
    
    def test_log_query(self, logger):
        """Test logging a query."""
        session_id = "test_session"
        query = "Analyze ChIP-seq data"
        
        interaction_id = logger.log_query(session_id, query)
        
        # Verify by loading
        interactions = logger.get_session_interactions(session_id)
        assert len(interactions) == 1
        assert interactions[0]['query'] == query
        assert interaction_id is not None
    
    def test_log_intent(self, logger):
        """Test logging intent."""
        session_id = "test_session"
        query = "Analyze RNA-seq data"
        
        interaction_id = logger.log_query(session_id, query)
        
        intent = {
            "analysis_type": "rna_seq",
            "confidence": 0.95,
        }
        
        logger.log_intent(interaction_id, intent, confidence=0.95)
        
        interactions = logger.get_session_interactions(session_id)
        assert interactions[0]['intent'] is not None
        assert interactions[0]['intent']['analysis_type'] == "rna_seq"
    
    def test_log_workflow(self, logger):
        """Test logging workflow."""
        session_id = "test_session"
        query = "Run variant calling"
        
        interaction_id = logger.log_query(session_id, query)
        logger.log_workflow(interaction_id, "workflow { GATK() }")
        
        interactions = logger.get_session_interactions(session_id)
        assert "workflow" in interactions[0]['workflow_generated']
    
    def test_log_feedback(self, logger):
        """Test logging user feedback."""
        session_id = "test_session"
        query = "Analyze methylation data"
        
        interaction_id = logger.log_query(session_id, query)
        logger.log_feedback(interaction_id, "accept", rating=5)
        
        interactions = logger.get_session_interactions(session_id)
        assert interactions[0]['feedback_type'] == "accept"
        assert interactions[0]['rating'] == 5


# ============================================================================
# Data Pipeline Tests
# ============================================================================

class TestDataValidator:
    """Tests for DataValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create validator."""
        return DataValidator()
    
    def test_valid_example(self, validator):
        """Test validating a complete example."""
        example = {
            "id": "test_1",
            "query": "Analyze RNA-seq data",
            "intent": {
                "analysis_type": "rna_seq",
                "confidence": 0.9,
            },
            "tools": ["star", "deseq2"],
            "workflow": "workflow { STAR(reads) }",
        }
        
        is_valid, errors = validator.validate_example(example)
        
        # May have warnings but should be mostly valid
        assert "Missing query" not in errors
        assert "Missing id" not in errors
    
    def test_invalid_missing_query(self, validator):
        """Test validation catches missing query."""
        example = {"id": "test_1"}
        
        is_valid, errors = validator.validate_example(example)
        
        assert not is_valid
        assert any("query" in e.lower() for e in errors)


class TestQualityScorer:
    """Tests for QualityScorer."""
    
    @pytest.fixture
    def scorer(self):
        """Create scorer."""
        return QualityScorer()
    
    def test_score_complete_example(self, scorer):
        """Test scoring a complete example."""
        example = {
            "query": "Analyze data",
            "intent": {"analysis_type": "rna_seq", "confidence": 0.9},
            "tools": ["star", "deseq2"],
            "workflow": "workflow { }",
            "validated": True,
            "explanation": "This workflow does...",
        }
        
        score = scorer.score_example(example)
        
        assert score >= 0.8
    
    def test_score_minimal_example(self, scorer):
        """Test scoring minimal example."""
        example = {"query": "Analyze data"}
        
        score = scorer.score_example(example)
        
        assert score == 0.5  # Base score


class TestTrainingDataPipeline:
    """Tests for TrainingDataPipeline."""
    
    @pytest.fixture
    def pipeline(self, tmp_path):
        """Create pipeline with temp directories."""
        config = PipelineConfig(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
        )
        return TrainingDataPipeline(config)
    
    def test_pipeline_init(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.config.raw_dir.exists()
        assert pipeline.config.processed_dir.exists()
    
    def test_deduplicate(self, pipeline):
        """Test deduplication."""
        examples = [
            {"id": "1", "query": "Analyze RNA-seq data"},
            {"id": "2", "query": "Analyze RNA-seq data"},  # Duplicate
            {"id": "3", "query": "Run ChIP-seq analysis"},
        ]
        
        deduplicated = pipeline.deduplicate(examples)
        
        assert len(deduplicated) == 2
    
    def test_create_splits(self, pipeline):
        """Test train/val/test split creation."""
        examples = [
            {"id": str(i), "query": f"Query {i}", "analysis_type": "rna_seq"}
            for i in range(100)
        ]
        
        train, val, test = pipeline.create_splits(examples)
        
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
    
    def test_calculate_metrics(self, pipeline):
        """Test metrics calculation."""
        examples = [
            {
                "id": "1",
                "query": "Analyze RNA-seq",
                "intent": {"analysis_type": "rna_seq", "confidence": 0.9},
                "tools": ["star", "deseq2"],
                "workflow": "workflow { }",
                "category": "rna_seq",
            },
            {
                "id": "2", 
                "query": "Run ChIP-seq",
                "intent": {"analysis_type": "chip_seq", "confidence": 0.85},
                "tools": ["macs2"],
                "workflow": "workflow { }",
                "category": "chip_seq",
            },
        ]
        
        metrics = pipeline.calculate_metrics(examples)
        
        assert metrics.total_examples == 2
        assert metrics.intent_parse_rate == 1.0
        assert "rna_seq" in metrics.category_distribution
        assert "chip_seq" in metrics.category_distribution


# ============================================================================
# Export Tests
# ============================================================================

class TestExporters:
    """Tests for training data exporters."""
    
    @pytest.fixture
    def sample_examples(self):
        """Sample examples for export testing."""
        return [
            {
                "id": "1",
                "query": "Analyze RNA-seq data from mouse liver",
                "intent": {"analysis_type": "rna_seq", "confidence": 0.95},
                "tools": ["star", "deseq2", "fastp"],
                "workflow": 'include { STAR } from "./modules"\nworkflow { STAR(reads) }',
                "explanation": "This performs RNA-seq analysis.",
            },
            {
                "id": "2",
                "query": "Call variants in human exome data",
                "intent": {"analysis_type": "variant_calling", "confidence": 0.9},
                "tools": ["bwa", "gatk"],
                "workflow": 'workflow { BWA(reads) | GATK() }',
                "explanation": "Variant calling pipeline.",
            },
        ]
    
    def test_openai_chat_export(self, sample_examples, tmp_path):
        """Test OpenAI chat format export."""
        exporter = OpenAIChatExporter()
        output_path = tmp_path / "openai.jsonl"
        
        result = exporter.export(sample_examples, output_path)
        
        assert result.format == "openai_chat"
        assert result.example_count == 2
        assert output_path.exists()
        
        # Verify format
        with open(output_path) as f:
            line = json.loads(f.readline())
            assert "messages" in line
            assert len(line["messages"]) == 3  # system, user, assistant
            assert line["messages"][0]["role"] == "system"
    
    def test_alpaca_export(self, sample_examples, tmp_path):
        """Test Alpaca format export."""
        exporter = AlpacaExporter()
        output_path = tmp_path / "alpaca.json"
        
        result = exporter.export(sample_examples, output_path)
        
        assert result.format == "alpaca"
        assert output_path.exists()
        
        # Verify format
        with open(output_path) as f:
            data = json.load(f)
            assert isinstance(data, list)
            assert "instruction" in data[0]
            assert "output" in data[0]
    
    def test_sharegpt_export(self, sample_examples, tmp_path):
        """Test ShareGPT format export."""
        exporter = ShareGPTExporter()
        output_path = tmp_path / "sharegpt.json"
        
        result = exporter.export(sample_examples, output_path)
        
        assert result.format == "sharegpt"
        
        # Verify format
        with open(output_path) as f:
            data = json.load(f)
            assert "conversations" in data[0]
            assert data[0]["conversations"][0]["from"] == "human"
            assert data[0]["conversations"][1]["from"] == "gpt"
    
    def test_axolotl_export(self, sample_examples, tmp_path):
        """Test Axolotl format export."""
        exporter = AxolotlExporter()
        output_path = tmp_path / "axolotl.jsonl"
        
        result = exporter.export(sample_examples, output_path)
        
        assert result.format == "axolotl"
        
        # Verify format
        with open(output_path) as f:
            line = json.loads(f.readline())
            assert "text" in line
            assert "### Instruction:" in line["text"]
            assert "### Response:" in line["text"]
    
    def test_export_all_formats(self, sample_examples, tmp_path):
        """Test exporting to all formats."""
        exporter = TrainingDataExporter()
        
        results = exporter.export_all_formats(sample_examples, tmp_path)
        
        # Should have multiple formats
        assert len(results) >= 4
        assert "openai_chat" in results
        assert "alpaca" in results
        
        # All should succeed
        for name, result in results.items():
            assert result.example_count > 0


class TestTrainingDataExporter:
    """Tests for the main exporter class."""
    
    def test_available_formats(self):
        """Test available export formats."""
        exporter = TrainingDataExporter()
        
        assert "openai_chat" in exporter.EXPORTERS
        assert "alpaca" in exporter.EXPORTERS
        assert "sharegpt" in exporter.EXPORTERS
        assert "axolotl" in exporter.EXPORTERS
    
    def test_invalid_format(self):
        """Test error on invalid format."""
        exporter = TrainingDataExporter()
        
        with pytest.raises(ValueError):
            exporter.export([], "invalid_format")


# ============================================================================
# Integration Tests
# ============================================================================

class TestTrainingModuleIntegration:
    """Integration tests for the training module."""
    
    @pytest.fixture
    def setup_dirs(self, tmp_path):
        """Set up directory structure."""
        (tmp_path / "raw" / "synthetic").mkdir(parents=True)
        (tmp_path / "raw" / "interactions").mkdir(parents=True)
        (tmp_path / "processed").mkdir(parents=True)
        (tmp_path / "export").mkdir(parents=True)
        return tmp_path
    
    def test_end_to_end_pipeline(self, setup_dirs):
        """Test full pipeline from raw data to export."""
        tmp_path = setup_dirs
        
        # Create sample raw data
        raw_file = tmp_path / "raw" / "synthetic" / "batch_001.jsonl"
        examples = [
            {
                "id": f"example_{i}",
                "query": f"Query {i} for RNA-seq analysis",
                "intent": {"analysis_type": "rna_seq", "confidence": 0.9},
                "tools": ["star", "deseq2"],
                "workflow": "workflow { STAR(reads) }",
                "quality_score": 0.85,
                "analysis_type": "rna_seq",
            }
            for i in range(20)
        ]
        
        with open(raw_file, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')
        
        # Run pipeline
        config = PipelineConfig(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            min_quality_score=0.7,
        )
        pipeline = TrainingDataPipeline(config)
        result = pipeline.process_all()
        
        assert result["status"] == "success"
        assert result["raw_count"] == 20
        assert result["train_count"] > 0
        
        # Verify output files exist
        assert (tmp_path / "processed" / "train.jsonl").exists()
        assert (tmp_path / "processed" / "val.jsonl").exists()
        assert (tmp_path / "processed" / "test.jsonl").exists()
        assert (tmp_path / "processed" / "metrics.json").exists()
        
        # Export to various formats
        train_path = tmp_path / "processed" / "train.jsonl"
        exporter = TrainingDataExporter()
        
        train_examples = []
        with open(train_path) as f:
            for line in f:
                train_examples.append(json.loads(line))
        
        export_results = exporter.export_all_formats(
            train_examples, 
            tmp_path / "export"
        )
        
        assert len(export_results) >= 4


# ============================================================================
# Module Import Tests
# ============================================================================

class TestModuleImports:
    """Test module can be imported correctly."""
    
    def test_import_main_module(self):
        """Test importing main training module."""
        from workflow_composer import training
        
        assert hasattr(training, "TrainingDataGenerator")
        assert hasattr(training, "InteractionLogger")
        assert hasattr(training, "TrainingDataPipeline")
        assert hasattr(training, "TrainingDataExporter")
    
    def test_import_factory_functions(self):
        """Test factory functions are available."""
        from workflow_composer.training import (
            get_data_generator,
            get_interaction_logger,
            get_pipeline,
            get_exporter,
        )
        
        assert callable(get_data_generator)
        assert callable(get_interaction_logger)
        assert callable(get_pipeline)
        assert callable(get_exporter)
    
    def test_config_classes(self):
        """Test config classes are importable."""
        from workflow_composer.training import (
            TrainingConfig,
            GeneratorConfig,
            LoggerConfig,
            PipelineConfig,
            ExportConfig,
            VariationType,
        )
        
        # All should be instantiable
        TrainingConfig()
        GeneratorConfig()
        LoggerConfig()
        PipelineConfig()
        ExportConfig()
