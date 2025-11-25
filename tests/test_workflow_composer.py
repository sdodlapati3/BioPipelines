"""
Tests for Workflow Composer
===========================

Basic tests to verify the AI Workflow Composer structure.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestImports:
    """Test that all modules can be imported."""
    
    def test_import_main_package(self):
        """Test importing the main package."""
        import workflow_composer
        assert hasattr(workflow_composer, '__version__')
        assert hasattr(workflow_composer, 'Composer')
    
    def test_import_llm_adapters(self):
        """Test importing LLM adapters."""
        from workflow_composer.llm import (
            LLMAdapter,
            OllamaAdapter,
            OpenAIAdapter,
            AnthropicAdapter,
            HuggingFaceAdapter,
            get_llm,
            list_providers
        )
        
        # Check factory
        providers = list_providers()
        assert "ollama" in providers
        assert "openai" in providers
        assert "anthropic" in providers
        assert "huggingface" in providers
    
    def test_import_core_components(self):
        """Test importing core components."""
        from workflow_composer.core import (
            IntentParser,
            ParsedIntent,
            AnalysisType,
            ToolSelector,
            Tool,
            ModuleMapper,
            Module,
            WorkflowGenerator,
            Workflow
        )
        
        # Check AnalysisType enum
        assert hasattr(AnalysisType, 'RNA_SEQ_DE')
        assert hasattr(AnalysisType, 'CHIP_SEQ')
        assert hasattr(AnalysisType, 'SCRNA_SEQ')
    
    def test_import_data_module(self):
        """Test importing data download module."""
        from workflow_composer.data import (
            DataDownloader,
            Reference,
            REFERENCE_SOURCES,
            INDEX_SOURCES
        )
        
        # Check reference sources
        assert "ensembl" in REFERENCE_SOURCES
        assert "human" in REFERENCE_SOURCES["ensembl"]["species"]
    
    def test_import_viz_module(self):
        """Test importing visualization module."""
        from workflow_composer.viz import WorkflowVisualizer
        
        viz = WorkflowVisualizer()
        assert hasattr(viz, 'render_dag')
        assert hasattr(viz, 'generate_report')


class TestConfig:
    """Test configuration loading."""
    
    def test_config_exists(self):
        """Test that config file exists."""
        config_path = Path(__file__).parent.parent / "config" / "composer.yaml"
        assert config_path.exists(), f"Config not found: {config_path}"
    
    def test_config_load(self):
        """Test loading configuration."""
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "composer.yaml"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Check structure
        assert "llm" in config
        assert "paths" in config
        assert "analysis_types" in config
        
        # Check LLM config
        assert "default_provider" in config["llm"]
        assert "providers" in config["llm"]
        assert "ollama" in config["llm"]["providers"]


class TestLLMFactory:
    """Test LLM factory functionality."""
    
    def test_list_providers(self):
        """Test listing providers."""
        from workflow_composer.llm import list_providers
        
        providers = list_providers()
        assert isinstance(providers, dict)
        assert len(providers) >= 4  # At least 4 providers
    
    def test_create_adapter_no_api_key(self):
        """Test creating adapter without API key (should work for ollama)."""
        from workflow_composer.llm import get_llm, OllamaAdapter
        
        # Creating should not fail even without connection
        adapter = OllamaAdapter(model="llama3:8b")
        assert adapter.model == "llama3:8b"


class TestDataDownloader:
    """Test data downloader functionality."""
    
    def test_list_available(self):
        """Test listing available data."""
        from workflow_composer.data import DataDownloader
        
        downloader = DataDownloader()
        available = downloader.list_available()
        
        assert "reference_sources" in available
        assert "organisms" in available
        assert "sample_datasets" in available
    
    def test_organism_mappings(self):
        """Test organism mappings."""
        from workflow_composer.data import REFERENCE_SOURCES
        
        ensembl = REFERENCE_SOURCES["ensembl"]
        assert "human" in ensembl["species"]
        assert "mouse" in ensembl["species"]
        
        human = ensembl["species"]["human"]
        assert human["assembly"] == "GRCh38"


class TestAnalysisTypes:
    """Test analysis type parsing."""
    
    def test_analysis_type_enum(self):
        """Test AnalysisType enum values."""
        from workflow_composer.core import AnalysisType
        
        # Check major analysis types exist
        types = [
            AnalysisType.RNA_SEQ_DE,
            AnalysisType.RNA_SEQ_BASIC,
            AnalysisType.CHIP_SEQ,
            AnalysisType.ATAC_SEQ,
            AnalysisType.WGS_VARIANT_CALLING,
            AnalysisType.SCRNA_SEQ,
            AnalysisType.METAGENOMICS_PROFILING,
            AnalysisType.HIC,
            AnalysisType.LONG_READ_ASSEMBLY
        ]
        
        for atype in types:
            assert atype.value  # Each should have a string value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
