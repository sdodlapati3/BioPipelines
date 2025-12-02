"""
Reactome Database Client
========================

Client for the Reactome REST API to retrieve pathway information,
reaction details, and perform pathway analysis.

Reactome is a free, open-source, curated and peer-reviewed pathway
database that provides intuitive bioinformatics tools for the
visualization, interpretation and analysis of pathway knowledge.

API Documentation: https://reactome.org/ContentService/

Example:
    >>> from workflow_composer.agents.tools.databases import ReactomeClient
    >>> 
    >>> client = ReactomeClient()
    >>> result = client.search("apoptosis", species="Homo sapiens")
    >>> print(result.count)
    42
"""

from typing import Any, Dict, List, Optional
import logging

from .base import DatabaseClient, DatabaseResult

logger = logging.getLogger(__name__)


class ReactomeClient(DatabaseClient):
    """
    Client for Reactome REST API.
    
    Provides access to:
    - Pathway hierarchies and details
    - Reaction information
    - Species-specific pathways
    - Pathway enrichment analysis
    - Interactor data
    
    Reactome uses stable identifiers (stIds) like:
    - R-HSA-109582 (Homo sapiens pathway)
    - R-MMU-109582 (Mus musculus pathway)
    """
    
    BASE_URL = "https://reactome.org/ContentService"
    ANALYSIS_URL = "https://reactome.org/AnalysisService"
    NAME = "Reactome"
    RATE_LIMIT = 5.0  # Reactome is generous with rate limits
    
    # Species name mappings
    SPECIES_MAP = {
        "human": "Homo sapiens",
        "homo sapiens": "Homo sapiens",
        "mouse": "Mus musculus",
        "mus musculus": "Mus musculus",
        "rat": "Rattus norvegicus",
        "rattus norvegicus": "Rattus norvegicus",
        "zebrafish": "Danio rerio",
        "danio rerio": "Danio rerio",
        "drosophila": "Drosophila melanogaster",
        "fly": "Drosophila melanogaster",
        "yeast": "Saccharomyces cerevisiae",
        "s. cerevisiae": "Saccharomyces cerevisiae",
    }
    
    def search(
        self,
        query: str,
        species: str = "Homo sapiens",
        types: Optional[List[str]] = None,
        limit: int = 25,
        **kwargs,
    ) -> DatabaseResult:
        """
        Search Reactome for pathways, reactions, and entities.
        
        Args:
            query: Search terms
            species: Species name (e.g., "Homo sapiens", "human")
            types: Filter by type(s): "Pathway", "Reaction", "Complex", etc.
            limit: Maximum results
            
        Returns:
            DatabaseResult with matching entries
        """
        # Resolve species name
        species_name = self.SPECIES_MAP.get(species.lower(), species)
        
        try:
            params = {
                "query": query,
                "species": species_name,
                "rows": limit,
                "cluster": True,  # Cluster similar results
            }
            
            if types:
                params["types"] = ",".join(types)
            
            response = self._request(
                "GET",
                f"{self.BASE_URL}/search/query",
                params=params,
            )
            
            results = response.get("results", []) if isinstance(response, dict) else []
            
            # Flatten grouped results
            entries = []
            for group in results:
                entries.extend(group.get("entries", []))
            
            return DatabaseResult(
                success=True,
                data=entries,
                count=len(entries),
                query=query,
                source=self.NAME,
                message=f"Found {len(entries)} results",
                metadata={
                    "species": species_name,
                    "types": types,
                },
            )
            
        except Exception as e:
            logger.error(f"Reactome search failed: {e}")
            return self._error_result(query, e)
    
    def get_by_id(
        self,
        identifier: str,
        **kwargs,
    ) -> DatabaseResult:
        """
        Get entry by Reactome stable ID.
        
        Args:
            identifier: Reactome stId (e.g., "R-HSA-109582")
            
        Returns:
            DatabaseResult with entry details
        """
        try:
            response = self._request(
                "GET",
                f"{self.BASE_URL}/data/query/{identifier}",
            )
            
            return DatabaseResult(
                success=True,
                data=response,
                count=1,
                query=identifier,
                source=self.NAME,
                message=f"Retrieved {identifier}",
            )
            
        except Exception as e:
            logger.error(f"Failed to get {identifier}: {e}")
            return self._error_result(identifier, e)
    
    def get_pathway(self, pathway_id: str) -> DatabaseResult:
        """
        Get detailed pathway information.
        
        Args:
            pathway_id: Reactome pathway stId
            
        Returns:
            DatabaseResult with pathway details
        """
        return self.get_by_id(pathway_id)
    
    def get_pathway_hierarchy(self, species: str = "Homo sapiens") -> DatabaseResult:
        """
        Get the pathway hierarchy tree for a species.
        
        Args:
            species: Species name
            
        Returns:
            DatabaseResult with pathway tree structure
        """
        species_name = self.SPECIES_MAP.get(species.lower(), species)
        
        try:
            response = self._request(
                "GET",
                f"{self.BASE_URL}/data/eventsHierarchy/{species_name}",
            )
            
            return DatabaseResult(
                success=True,
                data=response,
                count=len(response) if isinstance(response, list) else 1,
                query=f"hierarchy for {species_name}",
                source=self.NAME,
                message=f"Retrieved pathway hierarchy for {species_name}",
            )
            
        except Exception as e:
            logger.error(f"Failed to get pathway hierarchy: {e}")
            return self._error_result(f"hierarchy for {species}", e)
    
    def get_pathway_participants(self, pathway_id: str) -> DatabaseResult:
        """
        Get all participants (proteins, complexes, etc.) in a pathway.
        
        Args:
            pathway_id: Reactome pathway stId
            
        Returns:
            DatabaseResult with participant entities
        """
        try:
            response = self._request(
                "GET",
                f"{self.BASE_URL}/data/participants/{pathway_id}",
            )
            
            participants = response if isinstance(response, list) else []
            
            return DatabaseResult(
                success=True,
                data=participants,
                count=len(participants),
                query=pathway_id,
                source=self.NAME,
                message=f"Found {len(participants)} participants",
            )
            
        except Exception as e:
            logger.error(f"Failed to get participants for {pathway_id}: {e}")
            return self._error_result(pathway_id, e)
    
    def get_pathway_genes(self, pathway_id: str) -> List[str]:
        """
        Get gene symbols for all proteins in a pathway.
        
        Args:
            pathway_id: Reactome pathway stId
            
        Returns:
            List of gene symbols
        """
        participants = self.get_pathway_participants(pathway_id)
        
        if not participants.success:
            return []
        
        genes = set()
        
        for participant in participants.data:
            # Extract gene names from different entity types
            if "geneNames" in participant:
                genes.update(participant["geneNames"])
            if "displayName" in participant:
                # Sometimes display name is the gene symbol
                name = participant["displayName"]
                if "[" not in name:  # Skip complex names
                    genes.add(name)
        
        return list(genes)
    
    def analyze_genes(
        self,
        genes: List[str],
        include_interactors: bool = False,
        project_to_human: bool = True,
    ) -> DatabaseResult:
        """
        Perform pathway enrichment analysis on a gene list.
        
        Args:
            genes: List of gene symbols or UniProt IDs
            include_interactors: Include known interactors
            project_to_human: Project results to human pathways
            
        Returns:
            DatabaseResult with enrichment results
        """
        try:
            # Submit genes as newline-separated text
            gene_list = "\n".join(genes)
            
            params = {
                "interactors": str(include_interactors).lower(),
                "pageSize": 100,
                "page": 1,
                "sortBy": "ENTITIES_FDR",
                "order": "ASC",
            }
            
            if project_to_human:
                params["species"] = "Homo sapiens"
            
            response = self._request(
                "POST",
                f"{self.ANALYSIS_URL}/identifiers/projection",
                params=params,
                data=gene_list,
                headers={"Content-Type": "text/plain"},
            )
            
            # Parse results
            pathways = response.get("pathways", []) if isinstance(response, dict) else []
            
            # Extract key information
            results = []
            for pathway in pathways:
                results.append({
                    "stId": pathway.get("stId", ""),
                    "name": pathway.get("name", ""),
                    "entities_pvalue": pathway.get("entities", {}).get("pValue"),
                    "entities_fdr": pathway.get("entities", {}).get("fdr"),
                    "entities_found": pathway.get("entities", {}).get("found"),
                    "entities_total": pathway.get("entities", {}).get("total"),
                    "reactions_found": pathway.get("reactions", {}).get("found"),
                    "reactions_total": pathway.get("reactions", {}).get("total"),
                    "species": pathway.get("species", {}).get("displayName", ""),
                })
            
            return DatabaseResult(
                success=True,
                data=results,
                count=len(results),
                query=f"{len(genes)} genes",
                source=f"{self.NAME} Analysis",
                message=f"Found {len(results)} enriched pathways",
                metadata={
                    "input_genes": len(genes),
                    "include_interactors": include_interactors,
                    "summary": response.get("summary", {}),
                },
            )
            
        except Exception as e:
            logger.error(f"Reactome analysis failed: {e}")
            return self._error_result(f"{len(genes)} genes", e)
    
    def get_interactors(
        self,
        identifier: str,
        resource: str = "IntAct",
    ) -> DatabaseResult:
        """
        Get interaction data for a protein.
        
        Args:
            identifier: Protein identifier (gene symbol, UniProt ID)
            resource: Interaction database ("IntAct", "MINT", etc.)
            
        Returns:
            DatabaseResult with interactors
        """
        try:
            response = self._request(
                "GET",
                f"{self.BASE_URL}/interactors/static/molecule/{identifier}/details",
            )
            
            entities = response.get("entities", []) if isinstance(response, dict) else []
            
            return DatabaseResult(
                success=True,
                data=entities,
                count=len(entities),
                query=identifier,
                source=f"{self.NAME} Interactors",
                message=f"Found {len(entities)} interactors for {identifier}",
            )
            
        except Exception as e:
            logger.error(f"Failed to get interactors for {identifier}: {e}")
            return self._error_result(identifier, e)
    
    def get_pathway_diagram(self, pathway_id: str, format: str = "svg") -> Optional[str]:
        """
        Get URL for pathway diagram.
        
        Args:
            pathway_id: Reactome pathway stId
            format: Image format ("svg", "png")
            
        Returns:
            URL to diagram image
        """
        ext = "svg" if format.lower() == "svg" else "png"
        return f"https://reactome.org/ContentService/exporter/diagram/{pathway_id}.{ext}"
    
    def list_top_pathways(self, species: str = "Homo sapiens") -> DatabaseResult:
        """
        Get top-level pathways for a species.
        
        Args:
            species: Species name
            
        Returns:
            DatabaseResult with top-level pathways
        """
        species_name = self.SPECIES_MAP.get(species.lower(), species)
        
        try:
            response = self._request(
                "GET",
                f"{self.BASE_URL}/data/pathways/top/{species_name}",
            )
            
            pathways = response if isinstance(response, list) else []
            
            return DatabaseResult(
                success=True,
                data=pathways,
                count=len(pathways),
                query=f"top pathways for {species_name}",
                source=self.NAME,
                message=f"Found {len(pathways)} top-level pathways",
            )
            
        except Exception as e:
            logger.error(f"Failed to get top pathways: {e}")
            return self._error_result(f"top pathways for {species}", e)
    
    def get_complex_components(self, complex_id: str) -> DatabaseResult:
        """
        Get components of a protein complex.
        
        Args:
            complex_id: Reactome complex stId
            
        Returns:
            DatabaseResult with complex components
        """
        try:
            response = self._request(
                "GET",
                f"{self.BASE_URL}/data/complex/{complex_id}/subunits",
            )
            
            components = response if isinstance(response, list) else []
            
            return DatabaseResult(
                success=True,
                data=components,
                count=len(components),
                query=complex_id,
                source=self.NAME,
                message=f"Found {len(components)} complex components",
            )
            
        except Exception as e:
            logger.error(f"Failed to get complex components: {e}")
            return self._error_result(complex_id, e)
    
    def find_pathways_for_gene(
        self,
        gene: str,
        species: str = "Homo sapiens",
    ) -> DatabaseResult:
        """
        Find all pathways containing a specific gene.
        
        Args:
            gene: Gene symbol
            species: Species name
            
        Returns:
            DatabaseResult with pathways containing the gene
        """
        species_name = self.SPECIES_MAP.get(species.lower(), species)
        
        try:
            response = self._request(
                "GET",
                f"{self.BASE_URL}/data/pathways/low/entity/{gene}",
                params={"species": species_name},
            )
            
            pathways = response if isinstance(response, list) else []
            
            return DatabaseResult(
                success=True,
                data=pathways,
                count=len(pathways),
                query=gene,
                source=self.NAME,
                message=f"Found {len(pathways)} pathways containing {gene}",
                metadata={"species": species_name},
            )
            
        except Exception as e:
            logger.error(f"Failed to find pathways for {gene}: {e}")
            return self._error_result(gene, e)
