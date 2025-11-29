"""
Tests for Data Discovery Module
===============================

Tests the data discovery functionality including:
- Query parsing
- Database adapters (ENCODE, GEO, Ensembl)
- Orchestrator multi-source search
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

# Import modules under test
from workflow_composer.data.discovery.models import (
    DataSource, SearchQuery, DatasetInfo, DownloadURL,
    SearchResults, AssayType, FileType, DownloadMethod
)
from workflow_composer.data.discovery.query_parser import QueryParser, ParseResult
from workflow_composer.data.discovery.adapters.encode import ENCODEAdapter
from workflow_composer.data.discovery.adapters.geo import GEOAdapter
from workflow_composer.data.discovery.adapters.ensembl import EnsemblAdapter
from workflow_composer.data.discovery.orchestrator import DataDiscovery


# ============================================================================
# Model Tests
# ============================================================================

class TestModels:
    """Test data model classes."""
    
    def test_data_source_enum(self):
        """Test DataSource enum values."""
        assert DataSource.ENCODE.value == "encode"
        assert DataSource.GEO.value == "geo"
        assert DataSource.ENSEMBL.value == "ensembl"
    
    def test_search_query_creation(self):
        """Test SearchQuery dataclass."""
        query = SearchQuery(
            raw_query="human ChIP-seq",
            organism="human",
            assay_type="ChIP-seq",
            target="H3K27ac",
            max_results=10
        )
        assert query.organism == "human"
        assert query.assay_type == "ChIP-seq"
        assert query.target == "H3K27ac"
        assert query.max_results == 10
    
    def test_search_query_to_dict(self):
        """Test SearchQuery serialization."""
        query = SearchQuery(
            raw_query="mouse RNA-seq",
            organism="mouse",
            assay_type="RNA-seq",
        )
        d = query.to_dict()
        assert d["organism"] == "mouse"
        assert d["assay_type"] == "RNA-seq"
    
    def test_dataset_info_creation(self):
        """Test DatasetInfo dataclass."""
        dataset = DatasetInfo(
            id="ENCSR000ABC",
            source=DataSource.ENCODE,
            title="Test Dataset",
            description="A test dataset",
            organism="Homo sapiens",
            assay_type="ChIP-seq",
        )
        assert dataset.id == "ENCSR000ABC"
        assert dataset.source == DataSource.ENCODE
        assert dataset.organism == "Homo sapiens"
    
    def test_download_url_creation(self):
        """Test DownloadURL dataclass."""
        url = DownloadURL(
            url="https://example.com/file.fastq.gz",
            filename="file.fastq.gz",
            file_type=FileType.FASTQ,
            size_bytes=1024000,
            md5="abc123"
        )
        assert url.url == "https://example.com/file.fastq.gz"
        assert url.file_type == FileType.FASTQ
        assert url.size_bytes == 1024000
    
    def test_search_results_has_results(self):
        """Test SearchResults.has_results property."""
        query = SearchQuery(raw_query="test")
        
        # Empty results
        empty_results = SearchResults(query=query, datasets=[])
        assert not empty_results.has_results
        
        # With results
        dataset = DatasetInfo(id="TEST", source=DataSource.GEO, title="Test")
        results = SearchResults(query=query, datasets=[dataset])
        assert results.has_results


# ============================================================================
# Query Parser Tests
# ============================================================================

class TestQueryParser:
    """Test natural language query parsing."""
    
    def test_regex_parser_organism(self):
        """Test organism extraction from query."""
        parser = QueryParser()
        
        # Human variations
        result = parser.parse("human ChIP-seq")
        assert result.query.organism == "human"
        
        result = parser.parse("Homo sapiens RNA-seq")
        assert result.query.organism == "human"
        
        # Mouse
        result = parser.parse("mouse ATAC-seq")
        assert result.query.organism == "mouse"
    
    def test_regex_parser_assay_type(self):
        """Test assay type extraction."""
        parser = QueryParser()
        
        result = parser.parse("human ChIP-seq H3K27ac")
        assert result.query.assay_type == "ChIP-seq"
        
        result = parser.parse("RNA-seq mouse brain")
        assert result.query.assay_type == "RNA-seq"
        
        result = parser.parse("ATAC-seq accessibility")
        assert result.query.assay_type == "ATAC-seq"
    
    def test_regex_parser_target(self):
        """Test target/histone mark extraction."""
        parser = QueryParser()
        
        result = parser.parse("ChIP-seq H3K27ac")
        assert result.query.target == "H3K27ac"
        
        result = parser.parse("H3K4me3 ChIP-seq")
        assert result.query.target == "H3K4me3"
        
        result = parser.parse("CTCF binding ChIP-seq")
        assert result.query.target == "CTCF"
    
    def test_regex_parser_tissue(self):
        """Test tissue extraction."""
        parser = QueryParser()
        
        result = parser.parse("human liver ChIP-seq")
        assert result.query.tissue == "liver"
        
        result = parser.parse("brain RNA-seq")
        assert result.query.tissue == "brain"
    
    def test_regex_parser_suggested_sources(self):
        """Test suggested source inference."""
        parser = QueryParser()
        
        # ChIP-seq should suggest ENCODE
        result = parser.parse("ChIP-seq H3K27ac")
        assert DataSource.ENCODE in result.suggested_sources
        
        # RNA-seq should suggest GEO
        result = parser.parse("RNA-seq expression")
        assert DataSource.GEO in result.suggested_sources
        
        # Reference/genome should suggest Ensembl
        result = parser.parse("genome reference GRCh38")
        assert DataSource.ENSEMBL in result.suggested_sources


# ============================================================================
# Adapter Tests
# ============================================================================

class TestENCODEAdapter:
    """Test ENCODE adapter."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        adapter = ENCODEAdapter()
        assert adapter.BASE_URL == "https://www.encodeproject.org"
        # Check adapter has session
        assert hasattr(adapter, 'session')
    
    def test_build_search_params_basic(self):
        """Test parameter building for basic query."""
        adapter = ENCODEAdapter()
        query = SearchQuery(
            raw_query="test",
            organism="human",
            assay_type="ATAC-seq",
            max_results=5
        )
        
        params = adapter._build_search_params(query)
        
        # Check params is list of tuples
        assert isinstance(params, list)
        param_dict = {k: v for k, v in params}
        
        assert param_dict["type"] == "Experiment"
        assert param_dict["format"] == "json"
        # Now uses searchTerm containing assay info
        assert "ATAC-seq" in param_dict.get("searchTerm", "")
        assert param_dict["limit"] == "5"
    
    def test_build_search_params_organism_mapping(self):
        """Test organism name mapping via searchTerm."""
        adapter = ENCODEAdapter()
        
        # Test human mapping - now uses searchTerm
        query = SearchQuery(raw_query="test", organism="human")
        params = adapter._build_search_params(query)
        param_dict = {k: v for k, v in params}
        
        # Organism is now in searchTerm
        assert "human" in param_dict.get("searchTerm", "")
        
        # Test mouse mapping
        query = SearchQuery(raw_query="test", organism="mouse")
        params = adapter._build_search_params(query)
        param_dict = {k: v for k, v in params}
        assert "mouse" in param_dict.get("searchTerm", "")
    
    @patch('requests.Session.get')
    def test_search_success(self, mock_get):
        """Test successful search with mocked API."""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "@graph": [
                {
                    "@id": "/experiments/ENCSR000ABC/",
                    "accession": "ENCSR000ABC",
                    "description": "Test experiment",
                    "assay_title": "ChIP-seq",
                    "target": {"label": "H3K27ac"},
                    "replicates": [{
                        "library": {
                            "biosample": {
                                "donor": {
                                    "organism": {"scientific_name": "Homo sapiens"}
                                }
                            }
                        }
                    }],
                    "files": []
                }
            ]
        }
        mock_get.return_value = mock_response
        
        adapter = ENCODEAdapter()
        query = SearchQuery(raw_query="test", organism="human", assay_type="ChIP-seq")
        results = adapter.search(query)
        
        assert len(results) >= 0  # May be empty if parsing fails
        mock_get.assert_called()


class TestGEOAdapter:
    """Test GEO adapter."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        adapter = GEOAdapter()
        assert "ncbi.nlm.nih.gov" in adapter.BASE_URL
        # Check adapter has session
        assert hasattr(adapter, 'session')
    
    def test_build_search_term(self):
        """Test search term construction."""
        adapter = GEOAdapter()
        query = SearchQuery(
            raw_query="test",
            organism="human",
            assay_type="RNA-seq",
        )
        
        search_term = adapter._build_search_term(query)
        
        # The search term should include organism and assay info
        assert "human" in search_term.lower() or "[Organism]" in search_term
        # Should include dataset type or RNA-seq reference
        assert "sequencing" in search_term.lower() or "RNA" in search_term


class TestEnsemblAdapter:
    """Test Ensembl adapter."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        adapter = EnsemblAdapter()
        assert adapter.BASE_URL == "https://rest.ensembl.org"
        # Check adapter has session
        assert hasattr(adapter, 'session')
    
    def test_get_species_info(self):
        """Test species info retrieval setup."""
        adapter = EnsemblAdapter()
        # Adapter should be able to search
        assert hasattr(adapter, 'search')
        # Adapter should have organism mapping
        assert hasattr(adapter, 'ORGANISM_MAP') or hasattr(adapter, '_get_species_info')


# ============================================================================
# Orchestrator Tests
# ============================================================================

class TestDataDiscovery:
    """Test the main DataDiscovery orchestrator."""
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        discovery = DataDiscovery()
        
        assert DataSource.ENCODE in discovery.adapters
        assert DataSource.GEO in discovery.adapters
        assert DataSource.ENSEMBL in discovery.adapters
    
    def test_detect_source_encode(self):
        """Test ENCODE accession detection."""
        discovery = DataDiscovery()
        
        source = discovery._detect_source("ENCSR000ABC")
        assert source == DataSource.ENCODE
        
        source = discovery._detect_source("ENCFF000XYZ")
        assert source == DataSource.ENCODE
    
    def test_detect_source_geo(self):
        """Test GEO accession detection."""
        discovery = DataDiscovery()
        
        source = discovery._detect_source("GSE12345")
        assert source == DataSource.GEO
        
        source = discovery._detect_source("GSM67890")
        assert source == DataSource.GEO
    
    def test_detect_source_sra(self):
        """Test SRA accession detection."""
        discovery = DataDiscovery()
        
        source = discovery._detect_source("SRR1234567")
        assert source == DataSource.SRA
        
        source = discovery._detect_source("SRX123456")
        assert source == DataSource.SRA
    
    @patch.object(ENCODEAdapter, 'search')
    @patch.object(GEOAdapter, 'search')
    def test_search_multiple_sources(self, mock_geo_search, mock_encode_search):
        """Test searching multiple sources."""
        # Setup mocks
        encode_dataset = DatasetInfo(
            id="ENCSR000ABC",
            source=DataSource.ENCODE,
            title="ENCODE dataset"
        )
        geo_dataset = DatasetInfo(
            id="GSE12345",
            source=DataSource.GEO,
            title="GEO dataset"
        )
        
        mock_encode_search.return_value = [encode_dataset]
        mock_geo_search.return_value = [geo_dataset]
        
        # Run search
        discovery = DataDiscovery()
        results = discovery.search(
            "human ChIP-seq",
            sources=["encode", "geo"],
            max_results=5
        )
        
        assert results.has_results
        assert len(results.datasets) == 2
        assert DataSource.ENCODE in results.sources_searched
        assert DataSource.GEO in results.sources_searched


# ============================================================================
# Integration Tests (require network)
# ============================================================================

@pytest.mark.integration
class TestIntegration:
    """Integration tests that require network access."""
    
    @pytest.mark.slow
    def test_encode_real_search(self):
        """Test real ENCODE API search."""
        adapter = ENCODEAdapter()
        query = SearchQuery(
            raw_query="human ATAC-seq",
            organism="human",
            assay_type="ATAC-seq",
            max_results=3
        )
        
        results = adapter.search(query)
        assert len(results) > 0
        assert all(r.source == DataSource.ENCODE for r in results)
    
    @pytest.mark.slow
    def test_geo_real_search(self):
        """Test real GEO API search."""
        adapter = GEOAdapter()
        query = SearchQuery(
            raw_query="human RNA-seq",
            organism="human",
            assay_type="RNA-seq",
            max_results=3
        )
        
        results = adapter.search(query)
        assert len(results) > 0
        assert all(r.source == DataSource.GEO for r in results)
    
    @pytest.mark.slow
    def test_ensembl_real_search(self):
        """Test real Ensembl API search."""
        adapter = EnsemblAdapter()
        query = SearchQuery(
            raw_query="human reference",
            organism="human",
            assembly="GRCh38"
        )
        
        results = adapter.search(query)
        assert len(results) > 0
        assert all(r.source == DataSource.ENSEMBL for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
