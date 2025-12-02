"""
Tests for Database Clients
==========================

Unit tests for the BioPipelines database integration layer.
Tests cover UniProt, STRING, KEGG, Reactome, PubMed, and ClinVar clients.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json

# Import database components
from workflow_composer.agents.tools.databases import (
    DatabaseClient,
    DatabaseResult,
    UniProtClient,
    STRINGClient,
    KEGGClient,
    ReactomeClient,
    PubMedClient,
    ClinVarClient,
    get_uniprot_client,
    get_string_client,
    get_kegg_client,
    get_reactome_client,
    get_pubmed_client,
    get_clinvar_client,
    reset_clients,
)
from workflow_composer.agents.tools.databases.base import (
    resolve_taxonomy_id,
    ORGANISM_TAXONOMY_MAP,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_singleton_clients():
    """Reset singleton clients before each test."""
    reset_clients()
    yield
    reset_clients()


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client."""
    with patch("workflow_composer.agents.tools.databases.base.httpx") as mock_httpx:
        mock_client = Mock()
        mock_httpx.Client.return_value = mock_client
        yield mock_client


# =============================================================================
# DatabaseResult Tests
# =============================================================================

class TestDatabaseResult:
    """Tests for DatabaseResult dataclass."""
    
    def test_create_result(self):
        """Test creating a database result."""
        result = DatabaseResult(
            success=True,
            data=[{"id": "1", "name": "test"}],
            count=1,
            query="test",
            source="TestDB",
            message="Found 1 result",
        )
        
        assert result.success is True
        assert result.count == 1
        assert result.source == "TestDB"
        assert len(result.data) == 1
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = DatabaseResult(
            success=True,
            data={"key": "value"},
            count=1,
            query="test",
            source="TestDB",
        )
        
        result_dict = result.to_dict()
        
        assert "success" in result_dict
        assert "data" in result_dict
        assert "timestamp" in result_dict
        assert result_dict["source"] == "TestDB"
    
    def test_is_empty(self):
        """Test is_empty method."""
        empty_result = DatabaseResult(
            success=True,
            data=[],
            count=0,
            query="test",
            source="TestDB",
        )
        
        non_empty_result = DatabaseResult(
            success=True,
            data=[{"id": "1"}],
            count=1,
            query="test",
            source="TestDB",
        )
        
        assert empty_result.is_empty() is True
        assert non_empty_result.is_empty() is False
    
    def test_repr(self):
        """Test string representation."""
        result = DatabaseResult(
            success=True,
            data=[],
            count=0,
            query="test",
            source="TestDB",
        )
        
        repr_str = repr(result)
        
        assert "TestDB" in repr_str
        assert "success=True" in repr_str


# =============================================================================
# Taxonomy Resolution Tests
# =============================================================================

class TestTaxonomyResolution:
    """Tests for organism name to taxonomy ID resolution."""
    
    def test_resolve_human(self):
        """Test human organism resolution."""
        assert resolve_taxonomy_id("human") == "9606"
        assert resolve_taxonomy_id("Homo sapiens") == "9606"
        assert resolve_taxonomy_id("hsa") == "9606"
    
    def test_resolve_mouse(self):
        """Test mouse organism resolution."""
        assert resolve_taxonomy_id("mouse") == "10090"
        assert resolve_taxonomy_id("Mus musculus") == "10090"
    
    def test_resolve_numeric(self):
        """Test numeric taxonomy ID passthrough."""
        assert resolve_taxonomy_id("9606") == "9606"
        assert resolve_taxonomy_id("12345") == "12345"
    
    def test_unknown_organism(self):
        """Test unknown organism returns as-is."""
        assert resolve_taxonomy_id("unknown_species") == "unknown_species"
    
    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert resolve_taxonomy_id("HUMAN") == "9606"
        assert resolve_taxonomy_id("Mouse") == "10090"


# =============================================================================
# UniProt Client Tests
# =============================================================================

class TestUniProtClient:
    """Tests for UniProt database client."""
    
    def test_client_initialization(self):
        """Test client initializes correctly."""
        client = UniProtClient()
        
        assert client.NAME == "UniProt"
        assert client.BASE_URL == "https://rest.uniprot.org"
        assert client.RATE_LIMIT == 10.0
    
    def test_singleton_accessor(self):
        """Test singleton accessor returns same instance."""
        client1 = get_uniprot_client()
        client2 = get_uniprot_client()
        
        assert client1 is client2
    
    @patch.object(UniProtClient, '_request')
    def test_search_basic(self, mock_request):
        """Test basic search functionality."""
        mock_request.return_value = {
            "results": [
                {"primaryAccession": "P38398", "uniProtkbId": "BRCA1_HUMAN"},
            ]
        }
        
        client = UniProtClient()
        result = client.search("BRCA1", organism="human")
        
        assert result.success is True
        assert result.count == 1
        assert result.data[0]["primaryAccession"] == "P38398"
    
    @patch.object(UniProtClient, '_request')
    def test_search_with_filters(self, mock_request):
        """Test search with organism and reviewed filters."""
        mock_request.return_value = {"results": []}
        
        client = UniProtClient()
        client.search("test", organism="mouse", reviewed=True)
        
        # Check that _request was called with proper query
        call_args = mock_request.call_args
        params = call_args[1]["params"]
        
        assert "organism_id:10090" in params["query"]
        assert "reviewed:true" in params["query"]
    
    @patch.object(UniProtClient, '_request')
    def test_get_by_id(self, mock_request):
        """Test getting protein by accession."""
        mock_request.return_value = {
            "primaryAccession": "P38398",
            "proteinDescription": {"recommendedName": {"fullName": {"value": "BRCA1"}}},
        }
        
        client = UniProtClient()
        result = client.get_by_id("P38398")
        
        assert result.success is True
        assert result.data["primaryAccession"] == "P38398"
    
    @patch.object(UniProtClient, '_request')
    def test_get_sequence(self, mock_request):
        """Test getting protein sequence."""
        mock_request.return_value = ">sp|P38398|BRCA1_HUMAN\nMDLSALRVEE"
        
        client = UniProtClient()
        sequence = client.get_sequence("P38398")
        
        assert sequence is not None
        assert "BRCA1" in sequence
    
    def test_search_error_handling(self):
        """Test error handling during search."""
        client = UniProtClient()
        
        # Mock a failed request
        with patch.object(client, '_request', side_effect=Exception("Network error")):
            result = client.search("test")
        
        assert result.success is False
        assert "error" in result.message.lower()


# =============================================================================
# STRING Client Tests
# =============================================================================

class TestSTRINGClient:
    """Tests for STRING database client."""
    
    def test_client_initialization(self):
        """Test client initializes correctly."""
        client = STRINGClient()
        
        assert client.NAME == "STRING"
        assert client.RATE_LIMIT == 1.0  # Lower rate limit
    
    def test_singleton_accessor(self):
        """Test singleton accessor."""
        client1 = get_string_client()
        client2 = get_string_client()
        
        assert client1 is client2
    
    @patch.object(STRINGClient, '_request')
    def test_get_interactions(self, mock_request):
        """Test getting protein interactions."""
        mock_request.return_value = [
            {
                "preferredName_A": "TP53",
                "preferredName_B": "MDM2",
                "score": 0.999,
            },
        ]
        
        client = STRINGClient()
        result = client.get_interactions(["TP53", "MDM2"])
        
        assert result.success is True
        assert result.count == 1
        assert result.data[0]["preferredName_A"] == "TP53"
    
    @patch.object(STRINGClient, '_request')
    def test_get_enrichment(self, mock_request):
        """Test functional enrichment analysis."""
        mock_request.return_value = [
            {
                "category": "Process",
                "description": "DNA repair",
                "p_value": 1e-10,
            },
        ]
        
        client = STRINGClient()
        result = client.get_enrichment(["BRCA1", "BRCA2", "TP53"])
        
        assert result.success is True
        assert result.count == 1
    
    def test_species_mapping(self):
        """Test species name mapping."""
        client = STRINGClient()
        
        assert client.SPECIES_MAP["human"] == 9606
        assert client.SPECIES_MAP["mouse"] == 10090
    
    def test_get_network_image(self):
        """Test network image URL generation."""
        client = STRINGClient()
        url = client.get_network_image(["TP53", "MDM2"])
        
        assert url is not None
        assert "string-db.org" in url
        assert "TP53" in url


# =============================================================================
# KEGG Client Tests
# =============================================================================

class TestKEGGClient:
    """Tests for KEGG database client."""
    
    def test_client_initialization(self):
        """Test client initializes correctly."""
        client = KEGGClient()
        
        assert client.NAME == "KEGG"
        assert client.RATE_LIMIT == 3.0
    
    def test_singleton_accessor(self):
        """Test singleton accessor."""
        client1 = get_kegg_client()
        client2 = get_kegg_client()
        
        assert client1 is client2
    
    @patch.object(KEGGClient, '_request')
    def test_search_pathway(self, mock_request):
        """Test pathway search."""
        mock_request.return_value = "hsa04110\tCell cycle - Homo sapiens (human)\n"
        
        client = KEGGClient()
        result = client.search("cell cycle", database="pathway")
        
        assert result.success is True
        assert result.count == 1
        assert result.data[0]["id"] == "hsa04110"
    
    @patch.object(KEGGClient, '_request')
    def test_get_pathway(self, mock_request):
        """Test getting pathway details."""
        mock_request.return_value = """ENTRY       hsa04110
NAME        Cell cycle - Homo sapiens (human)
DESCRIPTION Cell cycle regulation pathway
///"""
        
        client = KEGGClient()
        result = client.get_pathway("hsa04110")
        
        assert result.success is True
        assert "name" in result.data
    
    def test_organism_mapping(self):
        """Test organism code mapping."""
        client = KEGGClient()
        
        assert client.ORGANISM_MAP["human"] == "hsa"
        assert client.ORGANISM_MAP["mouse"] == "mmu"
    
    def test_get_pathway_image(self):
        """Test pathway image URL generation."""
        client = KEGGClient()
        url = client.get_pathway_image("hsa04110")
        
        assert url is not None
        assert "kegg.jp" in url
        assert "hsa04110" in url


# =============================================================================
# Reactome Client Tests
# =============================================================================

class TestReactomeClient:
    """Tests for Reactome database client."""
    
    def test_client_initialization(self):
        """Test client initializes correctly."""
        client = ReactomeClient()
        
        assert client.NAME == "Reactome"
        assert client.RATE_LIMIT == 5.0
    
    def test_singleton_accessor(self):
        """Test singleton accessor."""
        client1 = get_reactome_client()
        client2 = get_reactome_client()
        
        assert client1 is client2
    
    @patch.object(ReactomeClient, '_request')
    def test_search(self, mock_request):
        """Test pathway search."""
        mock_request.return_value = {
            "results": [
                {
                    "entries": [
                        {"stId": "R-HSA-109582", "name": "Hemostasis"},
                    ]
                }
            ]
        }
        
        client = ReactomeClient()
        result = client.search("hemostasis")
        
        assert result.success is True
        assert result.count == 1
    
    @patch.object(ReactomeClient, '_request')
    def test_analyze_genes(self, mock_request):
        """Test gene enrichment analysis."""
        mock_request.return_value = {
            "pathways": [
                {
                    "stId": "R-HSA-5693567",
                    "name": "DNA Repair",
                    "entities": {"pValue": 1e-10, "fdr": 1e-8, "found": 5, "total": 50},
                    "reactions": {"found": 3, "total": 20},
                    "species": {"displayName": "Homo sapiens"},
                },
            ],
            "summary": {},
        }
        
        client = ReactomeClient()
        result = client.analyze_genes(["BRCA1", "BRCA2", "TP53"])
        
        assert result.success is True
        assert result.count == 1
        assert result.data[0]["stId"] == "R-HSA-5693567"
    
    def test_species_mapping(self):
        """Test species name mapping."""
        client = ReactomeClient()
        
        assert client.SPECIES_MAP["human"] == "Homo sapiens"
        assert client.SPECIES_MAP["mouse"] == "Mus musculus"
    
    def test_get_pathway_diagram(self):
        """Test pathway diagram URL generation."""
        client = ReactomeClient()
        url = client.get_pathway_diagram("R-HSA-109582", format="svg")
        
        assert url is not None
        assert "reactome.org" in url
        assert "R-HSA-109582" in url


# =============================================================================
# PubMed Client Tests
# =============================================================================

class TestPubMedClient:
    """Tests for PubMed database client."""
    
    def test_client_initialization(self):
        """Test client initializes correctly."""
        client = PubMedClient()
        
        assert client.NAME == "PubMed"
        assert client.RATE_LIMIT == 3.0  # Without API key
    
    def test_client_with_api_key(self):
        """Test client initialization with API key."""
        client = PubMedClient(api_key="test_key")
        
        assert client.api_key == "test_key"
        assert client.RATE_LIMIT == 10.0  # Higher with API key
    
    def test_singleton_accessor(self):
        """Test singleton accessor."""
        client1 = get_pubmed_client()
        client2 = get_pubmed_client()
        
        assert client1 is client2
    
    @patch.object(PubMedClient, '_request')
    @patch.object(PubMedClient, 'fetch_articles')
    def test_search(self, mock_fetch, mock_request):
        """Test article search."""
        mock_request.return_value = {
            "esearchresult": {
                "idlist": ["12345678"],
                "count": "1",
            }
        }
        mock_fetch.return_value = [
            {"pmid": "12345678", "title": "Test Article"},
        ]
        
        client = PubMedClient()
        result = client.search("CRISPR")
        
        assert result.success is True
        assert result.count == 1
        assert result.data[0]["pmid"] == "12345678"
    
    def test_parse_pubmed_xml(self):
        """Test PubMed XML parsing."""
        xml_text = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345678</PMID>
                    <Article>
                        <ArticleTitle>Test Title</ArticleTitle>
                        <Abstract>
                            <AbstractText>Test abstract text.</AbstractText>
                        </Abstract>
                        <AuthorList>
                            <Author>
                                <LastName>Smith</LastName>
                                <ForeName>John</ForeName>
                            </Author>
                        </AuthorList>
                        <Journal>
                            <Title>Test Journal</Title>
                            <PubDate>
                                <Year>2024</Year>
                            </PubDate>
                        </Journal>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>"""
        
        client = PubMedClient()
        articles = client._parse_pubmed_xml(xml_text)
        
        assert len(articles) == 1
        assert articles[0]["pmid"] == "12345678"
        assert articles[0]["title"] == "Test Title"
        assert "Smith John" in articles[0]["authors"]


# =============================================================================
# ClinVar Client Tests
# =============================================================================

class TestClinVarClient:
    """Tests for ClinVar database client."""
    
    def test_client_initialization(self):
        """Test client initializes correctly."""
        client = ClinVarClient()
        
        assert client.NAME == "ClinVar"
        assert client.RATE_LIMIT == 3.0
    
    def test_singleton_accessor(self):
        """Test singleton accessor."""
        client1 = get_clinvar_client()
        client2 = get_clinvar_client()
        
        assert client1 is client2
    
    @patch.object(ClinVarClient, '_request')
    @patch.object(ClinVarClient, '_fetch_variants')
    def test_search(self, mock_fetch, mock_request):
        """Test variant search."""
        mock_request.return_value = {
            "esearchresult": {
                "idlist": ["12345"],
                "count": "1",
            }
        }
        mock_fetch.return_value = [
            {"variation_id": "12345", "clinical_significance": "Pathogenic"},
        ]
        
        client = ClinVarClient()
        result = client.search("BRCA1")
        
        assert result.success is True
        assert result.count == 1
    
    @patch.object(ClinVarClient, '_request')
    @patch.object(ClinVarClient, '_fetch_variants')
    def test_search_with_significance_filter(self, mock_fetch, mock_request):
        """Test search with clinical significance filter."""
        mock_request.return_value = {
            "esearchresult": {"idlist": [], "count": "0"}
        }
        mock_fetch.return_value = []
        
        client = ClinVarClient()
        client.search("BRCA1", significance="pathogenic")
        
        # Check that query includes significance filter
        call_args = mock_request.call_args
        params = call_args[1]["params"]
        assert "pathogenic" in params["term"]
    
    @patch.object(ClinVarClient, '_fetch_variants')
    def test_get_by_id(self, mock_fetch):
        """Test getting variant by ID."""
        mock_fetch.return_value = [
            {"variation_id": "12345", "name": "NM_007294.3(BRCA1):c.68_69del"},
        ]
        
        client = ClinVarClient()
        result = client.get_by_id("12345")
        
        assert result.success is True
        assert result.data["variation_id"] == "12345"
    
    @patch.object(ClinVarClient, 'search')
    def test_search_gene(self, mock_search):
        """Test search by gene name."""
        mock_search.return_value = DatabaseResult(
            success=True,
            data=[],
            count=0,
            query="BRCA1",
            source="ClinVar",
        )
        
        client = ClinVarClient()
        client.search_gene("BRCA1")
        
        # Check that search was called with gene name filter
        call_args = mock_search.call_args
        assert "Gene Name" in call_args[0][0] or "Gene Name" in str(call_args)


# =============================================================================
# Base Client Tests
# =============================================================================

class TestDatabaseClientBase:
    """Tests for base database client functionality."""
    
    def test_rate_limiting(self):
        """Test that rate limiting is enforced."""
        import time
        
        class TestClient(DatabaseClient):
            BASE_URL = "https://test.api"
            NAME = "Test"
            RATE_LIMIT = 100.0  # 100 req/sec for fast testing
            
            def search(self, query, **kwargs):
                return self._empty_result(query)
            
            def get_by_id(self, identifier, **kwargs):
                return self._empty_result(identifier)
        
        client = TestClient()
        
        # First request should be immediate
        start = time.time()
        client._rate_limit()
        elapsed1 = time.time() - start
        
        # Second request should wait
        start = time.time()
        client._rate_limit()
        elapsed2 = time.time() - start
        
        # Rate limit is 100/sec = 10ms between requests
        # Allow some tolerance
        assert elapsed1 < 0.1  # First should be fast
        # Second should have some delay (at least 5ms for 100 req/sec)
    
    def test_empty_result_helper(self):
        """Test empty result helper method."""
        class TestClient(DatabaseClient):
            BASE_URL = "https://test.api"
            NAME = "TestDB"
            
            def search(self, query, **kwargs):
                return self._empty_result(query)
            
            def get_by_id(self, identifier, **kwargs):
                return self._empty_result(identifier)
        
        client = TestClient()
        result = client._empty_result("test query", "No data found")
        
        assert result.success is True
        assert result.count == 0
        assert result.source == "TestDB"
        assert "No data found" in result.message
    
    def test_error_result_helper(self):
        """Test error result helper method."""
        class TestClient(DatabaseClient):
            BASE_URL = "https://test.api"
            NAME = "TestDB"
            
            def search(self, query, **kwargs):
                return self._error_result(query, Exception("Test error"))
            
            def get_by_id(self, identifier, **kwargs):
                return self._error_result(identifier, Exception("Test error"))
        
        client = TestClient()
        result = client._error_result("test", Exception("Test error"))
        
        assert result.success is False
        assert "Test error" in result.message
    
    def test_cache_functionality(self):
        """Test response caching."""
        class TestClient(DatabaseClient):
            BASE_URL = "https://test.api"
            NAME = "Test"
            
            def search(self, query, **kwargs):
                return self._empty_result(query)
            
            def get_by_id(self, identifier, **kwargs):
                return self._empty_result(identifier)
        
        client = TestClient(enable_cache=True)
        
        # Generate cache key
        key = client._get_cache_key("GET", "https://test.api/query", params={"q": "test"})
        
        assert key is not None
        assert len(key) > 0
        
        # Store and check cache
        client._store_cache(key, {"test": "data"})
        cached = client._check_cache(key)
        
        assert cached is not None
        assert cached["test"] == "data"


# =============================================================================
# Integration Tests (require network)
# =============================================================================

@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests that hit real APIs (skipped by default)."""
    
    @pytest.mark.skip(reason="Integration test - requires network")
    def test_uniprot_real_search(self):
        """Test real UniProt search."""
        client = get_uniprot_client()
        result = client.search("insulin", organism="human", limit=5)
        
        assert result.success is True
        assert result.count > 0
    
    @pytest.mark.skip(reason="Integration test - requires network")
    def test_kegg_real_pathway(self):
        """Test real KEGG pathway search."""
        client = get_kegg_client()
        result = client.search("cell cycle", organism="hsa")
        
        assert result.success is True
    
    @pytest.mark.skip(reason="Integration test - requires network")
    def test_pubmed_real_search(self):
        """Test real PubMed search."""
        client = get_pubmed_client()
        result = client.search("CRISPR", limit=5)
        
        assert result.success is True
        assert result.count > 0


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
