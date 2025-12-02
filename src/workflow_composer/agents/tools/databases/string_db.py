"""
STRING Database Client
======================

Client for the STRING database API to retrieve protein-protein
interaction networks and functional enrichment.

STRING is a database of known and predicted protein-protein interactions.
The interactions include direct (physical) and indirect (functional)
associations derived from:
- Genomic context predictions
- High-throughput lab experiments  
- Coexpression data
- Automated textmining
- Previous knowledge in databases

API Documentation: https://string-db.org/cgi/help?subpage=api

Example:
    >>> from workflow_composer.agents.tools.databases import STRINGClient
    >>> 
    >>> client = STRINGClient()
    >>> result = client.get_interactions(["TP53", "MDM2"])
    >>> print(result.count)
    15
"""

from typing import Any, Dict, List, Optional
import logging

from .base import DatabaseClient, DatabaseResult, resolve_taxonomy_id

logger = logging.getLogger(__name__)


class STRINGClient(DatabaseClient):
    """
    Client for STRING protein interaction database.
    
    Provides access to:
    - Protein-protein interaction networks
    - Functional enrichment analysis
    - Network images and visualizations
    
    Note: STRING has strict rate limits. The client enforces 1 req/sec.
    """
    
    BASE_URL = "https://string-db.org/api"
    NAME = "STRING"
    RATE_LIMIT = 1.0  # STRING has lower rate limit
    
    # Common species taxonomy IDs
    SPECIES_MAP = {
        "human": 9606,
        "mouse": 10090,
        "rat": 10116,
        "zebrafish": 7955,
        "drosophila": 7227,
        "yeast": 4932,
        "e. coli": 511145,
    }
    
    def search(
        self,
        identifiers: List[str],
        species: int = 9606,
        limit: int = 10,
        **kwargs,
    ) -> DatabaseResult:
        """
        Map gene/protein names to STRING identifiers.
        
        Args:
            identifiers: Gene names or protein IDs to look up
            species: NCBI taxonomy ID (9606=human, 10090=mouse)
            limit: Maximum hits per identifier
            
        Returns:
            DatabaseResult with STRING identifier mappings
        """
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        
        try:
            response = self._request(
                "POST",
                f"{self.BASE_URL}/json/get_string_ids",
                data={
                    "identifiers": "\r".join(identifiers),
                    "species": species,
                    "limit": limit,
                    "echo_query": 1,
                },
            )
            
            results = response if isinstance(response, list) else []
            
            return DatabaseResult(
                success=True,
                data=results,
                count=len(results),
                query=", ".join(identifiers[:5]),
                source=self.NAME,
                message=f"Found {len(results)} STRING identifiers",
                metadata={"species": species},
            )
            
        except Exception as e:
            logger.error(f"STRING identifier lookup failed: {e}")
            return self._error_result(", ".join(identifiers[:5]), e)
    
    def get_by_id(
        self,
        identifier: str,
        species: int = 9606,
        **kwargs,
    ) -> DatabaseResult:
        """
        Get protein info by identifier.
        
        Args:
            identifier: Gene name or STRING ID
            species: NCBI taxonomy ID
            
        Returns:
            DatabaseResult with protein information
        """
        return self.search([identifier], species=species, limit=1)
    
    def get_interactions(
        self,
        identifiers: List[str],
        species: int = 9606,
        network_type: str = "functional",
        required_score: int = 400,
        limit: int = 50,
    ) -> DatabaseResult:
        """
        Get protein-protein interaction network.
        
        Args:
            identifiers: Gene names or protein IDs
            species: NCBI taxonomy ID (9606=human, 10090=mouse)
            network_type: "functional" or "physical"
            required_score: Minimum interaction score (0-1000)
                           - 150: low confidence
                           - 400: medium confidence
                           - 700: high confidence
                           - 900: highest confidence
            limit: Maximum interactions per protein
            
        Returns:
            DatabaseResult with interaction network
            
        Example:
            >>> result = client.get_interactions(["TP53", "MDM2", "CDKN1A"])
            >>> for interaction in result.data:
            ...     print(f"{interaction['preferredName_A']} <-> {interaction['preferredName_B']}")
        """
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        
        try:
            response = self._request(
                "POST",
                f"{self.BASE_URL}/json/network",
                data={
                    "identifiers": "\r".join(identifiers),
                    "species": species,
                    "network_type": network_type,
                    "required_score": required_score,
                    "limit": limit,
                },
            )
            
            interactions = response if isinstance(response, list) else []
            
            return DatabaseResult(
                success=True,
                data=interactions,
                count=len(interactions),
                query=", ".join(identifiers[:5]),
                source=self.NAME,
                message=f"Found {len(interactions)} interactions",
                metadata={
                    "species": species,
                    "network_type": network_type,
                    "required_score": required_score,
                },
            )
            
        except Exception as e:
            logger.error(f"STRING interaction network failed: {e}")
            return self._error_result(", ".join(identifiers[:5]), e)
    
    def get_enrichment(
        self,
        identifiers: List[str],
        species: int = 9606,
        background_count: Optional[int] = None,
    ) -> DatabaseResult:
        """
        Get functional enrichment for gene list.
        
        Performs enrichment analysis against:
        - GO Biological Process
        - GO Molecular Function
        - GO Cellular Component
        - KEGG Pathways
        - Reactome Pathways
        - UniProt Keywords
        - PFAM Domains
        - InterPro Domains
        
        Args:
            identifiers: Gene names or STRING IDs
            species: NCBI taxonomy ID
            background_count: Size of background gene set (optional)
            
        Returns:
            DatabaseResult with enrichment results including p-values
            
        Example:
            >>> result = client.get_enrichment(["BRCA1", "BRCA2", "TP53", "ATM"])
            >>> for term in result.data[:5]:
            ...     print(f"{term['category']}: {term['description']} (p={term['p_value']:.2e})")
        """
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        
        try:
            params = {
                "identifiers": "\r".join(identifiers),
                "species": species,
            }
            
            if background_count:
                params["background_count"] = background_count
            
            response = self._request(
                "POST",
                f"{self.BASE_URL}/json/enrichment",
                data=params,
            )
            
            enrichment = response if isinstance(response, list) else []
            
            return DatabaseResult(
                success=True,
                data=enrichment,
                count=len(enrichment),
                query=f"{len(identifiers)} genes",
                source=f"{self.NAME} Enrichment",
                message=f"Found {len(enrichment)} enriched terms",
                metadata={
                    "species": species,
                    "gene_count": len(identifiers),
                },
            )
            
        except Exception as e:
            logger.error(f"STRING enrichment failed: {e}")
            return self._error_result(f"{len(identifiers)} genes", e)
    
    def get_interaction_partners(
        self,
        identifier: str,
        species: int = 9606,
        limit: int = 10,
        required_score: int = 400,
    ) -> DatabaseResult:
        """
        Get interaction partners for a single protein.
        
        Args:
            identifier: Gene name or STRING ID
            species: NCBI taxonomy ID
            limit: Maximum number of partners
            required_score: Minimum confidence score
            
        Returns:
            DatabaseResult with list of interaction partners
        """
        try:
            response = self._request(
                "POST",
                f"{self.BASE_URL}/json/interaction_partners",
                data={
                    "identifiers": identifier,
                    "species": species,
                    "limit": limit,
                    "required_score": required_score,
                },
            )
            
            partners = response if isinstance(response, list) else []
            
            return DatabaseResult(
                success=True,
                data=partners,
                count=len(partners),
                query=identifier,
                source=self.NAME,
                message=f"Found {len(partners)} interaction partners for {identifier}",
                metadata={"species": species},
            )
            
        except Exception as e:
            logger.error(f"STRING interaction partners failed: {e}")
            return self._error_result(identifier, e)
    
    def get_network_image(
        self,
        identifiers: List[str],
        species: int = 9606,
        network_type: str = "functional",
        required_score: int = 400,
    ) -> Optional[str]:
        """
        Get URL for network visualization image.
        
        Args:
            identifiers: Gene names or STRING IDs
            species: NCBI taxonomy ID
            network_type: "functional" or "physical"
            required_score: Minimum confidence score
            
        Returns:
            URL to PNG image of the network
        """
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        
        # Build URL (doesn't require API call)
        params = [
            f"identifiers={'%0d'.join(identifiers)}",
            f"species={species}",
            f"network_type={network_type}",
            f"required_score={required_score}",
        ]
        
        return f"{self.BASE_URL}/image/network?{'&'.join(params)}"
    
    def get_ppi_enrichment(
        self,
        identifiers: List[str],
        species: int = 9606,
    ) -> DatabaseResult:
        """
        Test if the network has significantly more interactions
        than expected by chance.
        
        Args:
            identifiers: Gene names
            species: NCBI taxonomy ID
            
        Returns:
            DatabaseResult with PPI enrichment p-value
        """
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        
        try:
            response = self._request(
                "POST",
                f"{self.BASE_URL}/json/ppi_enrichment",
                data={
                    "identifiers": "\r".join(identifiers),
                    "species": species,
                },
            )
            
            results = response if isinstance(response, list) else [response] if response else []
            
            return DatabaseResult(
                success=True,
                data=results,
                count=len(results),
                query=f"{len(identifiers)} genes",
                source=f"{self.NAME} PPI Enrichment",
                message="PPI enrichment analysis complete",
                metadata={"species": species},
            )
            
        except Exception as e:
            logger.error(f"STRING PPI enrichment failed: {e}")
            return self._error_result(f"{len(identifiers)} genes", e)
    
    def get_homology(
        self,
        identifier: str,
        species: int = 9606,
        species_to: Optional[List[int]] = None,
    ) -> DatabaseResult:
        """
        Get homologs/orthologs for a protein in other species.
        
        Args:
            identifier: Gene name or STRING ID
            species: Source species taxonomy ID
            species_to: Target species (None = all)
            
        Returns:
            DatabaseResult with homolog information
        """
        try:
            params = {
                "identifiers": identifier,
                "species": species,
            }
            
            if species_to:
                params["species_b"] = "|".join(str(s) for s in species_to)
            
            response = self._request(
                "POST",
                f"{self.BASE_URL}/json/homology",
                data=params,
            )
            
            homologs = response if isinstance(response, list) else []
            
            return DatabaseResult(
                success=True,
                data=homologs,
                count=len(homologs),
                query=identifier,
                source=f"{self.NAME} Homology",
                message=f"Found {len(homologs)} homologs for {identifier}",
            )
            
        except Exception as e:
            logger.error(f"STRING homology lookup failed: {e}")
            return self._error_result(identifier, e)
    
    def build_network(
        self,
        genes: List[str],
        organism: str = "human",
        required_score: int = 400,
        include_enrichment: bool = True,
    ) -> Dict[str, Any]:
        """
        Convenience method to build a complete network analysis.
        
        Args:
            genes: List of gene symbols
            organism: Organism name
            required_score: Minimum confidence score
            include_enrichment: Whether to include enrichment analysis
            
        Returns:
            Dictionary with network, enrichment, and statistics
        """
        # Resolve species
        species = self.SPECIES_MAP.get(organism.lower(), 9606)
        
        result = {
            "genes": genes,
            "organism": organism,
            "species": species,
            "network": None,
            "enrichment": None,
            "ppi_enrichment": None,
            "image_url": None,
        }
        
        # Get interactions
        network = self.get_interactions(
            genes,
            species=species,
            required_score=required_score,
        )
        result["network"] = network.data if network.success else []
        
        # Get functional enrichment
        if include_enrichment:
            enrichment = self.get_enrichment(genes, species=species)
            result["enrichment"] = enrichment.data if enrichment.success else []
            
            ppi = self.get_ppi_enrichment(genes, species=species)
            result["ppi_enrichment"] = ppi.data if ppi.success else []
        
        # Get image URL
        result["image_url"] = self.get_network_image(
            genes,
            species=species,
            required_score=required_score,
        )
        
        return result
