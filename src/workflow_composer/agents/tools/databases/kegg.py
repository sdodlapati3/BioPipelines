"""
KEGG Database Client
====================

Client for the KEGG REST API to retrieve pathway information,
metabolic pathways, and functional annotations.

KEGG (Kyoto Encyclopedia of Genes and Genomes) is a database
resource for understanding high-level functions and utilities
of biological systems from molecular-level information.

API Documentation: https://www.kegg.jp/kegg/rest/keggapi.html

Example:
    >>> from workflow_composer.agents.tools.databases import KEGGClient
    >>> 
    >>> client = KEGGClient()
    >>> result = client.search("apoptosis", organism="hsa")
    >>> print(result.count)
    5
"""

from typing import Any, Dict, List, Optional
import logging

from .base import DatabaseClient, DatabaseResult

logger = logging.getLogger(__name__)


class KEGGClient(DatabaseClient):
    """
    Client for KEGG REST API.
    
    Provides access to:
    - Pathway information
    - Gene annotations
    - Module definitions
    - Compound data
    - Disease information
    
    KEGG organism codes:
    - hsa: Homo sapiens (human)
    - mmu: Mus musculus (mouse)
    - rno: Rattus norvegicus (rat)
    - dre: Danio rerio (zebrafish)
    - dme: Drosophila melanogaster
    - cel: Caenorhabditis elegans
    - sce: Saccharomyces cerevisiae
    """
    
    BASE_URL = "https://rest.kegg.jp"
    NAME = "KEGG"
    RATE_LIMIT = 3.0  # KEGG rate limit
    
    # Common organism codes
    ORGANISM_MAP = {
        "human": "hsa",
        "homo sapiens": "hsa",
        "mouse": "mmu",
        "mus musculus": "mmu",
        "rat": "rno",
        "zebrafish": "dre",
        "drosophila": "dme",
        "fly": "dme",
        "worm": "cel",
        "c. elegans": "cel",
        "yeast": "sce",
        "s. cerevisiae": "sce",
    }
    
    def search(
        self,
        query: str,
        database: str = "pathway",
        organism: str = "hsa",
        **kwargs,
    ) -> DatabaseResult:
        """
        Search KEGG database.
        
        Args:
            query: Search terms
            database: Database to search:
                     - "pathway": Metabolic/signaling pathways
                     - "module": Functional modules
                     - "ko": KEGG Orthology
                     - "genome": Organism genomes
                     - "compound": Small molecules
                     - "drug": Drugs
                     - "disease": Diseases
            organism: KEGG organism code (hsa, mmu, etc.)
            
        Returns:
            DatabaseResult with matching entries
            
        Example:
            >>> result = client.search("cell cycle", database="pathway", organism="hsa")
        """
        # Resolve organism code
        org_code = self.ORGANISM_MAP.get(organism.lower(), organism)
        
        try:
            # For organism-specific databases, prefix with org code
            if database in ["pathway", "module", "disease"]:
                search_db = f"{org_code}" if database == "pathway" else database
            else:
                search_db = database
            
            response = self._request(
                "GET",
                f"{self.BASE_URL}/find/{search_db}/{query}",
            )
            
            # Parse KEGG text format
            results = self._parse_kegg_list(response)
            
            return DatabaseResult(
                success=True,
                data=results,
                count=len(results),
                query=query,
                source=self.NAME,
                message=f"Found {len(results)} entries in {database}",
                metadata={
                    "database": database,
                    "organism": org_code,
                },
            )
            
        except Exception as e:
            logger.error(f"KEGG search failed: {e}")
            return self._error_result(query, e)
    
    def get_by_id(
        self,
        identifier: str,
        **kwargs,
    ) -> DatabaseResult:
        """
        Get entry by KEGG ID.
        
        Args:
            identifier: KEGG ID (e.g., "hsa04110", "path:hsa04110", "hsa:7157")
            
        Returns:
            DatabaseResult with parsed entry data
        """
        return self.get_pathway(identifier)
    
    def get_pathway(self, pathway_id: str) -> DatabaseResult:
        """
        Get detailed pathway information.
        
        Args:
            pathway_id: KEGG pathway ID (e.g., "hsa04110" for cell cycle)
            
        Returns:
            DatabaseResult with pathway details
        """
        # Normalize pathway ID
        if not pathway_id.startswith("path:") and not pathway_id.startswith("map"):
            # Check if it looks like a pathway ID
            pass  # Keep as-is
        
        try:
            response = self._request(
                "GET",
                f"{self.BASE_URL}/get/{pathway_id}",
            )
            
            # Parse KEGG flat file format
            data = self._parse_kegg_entry(response)
            
            return DatabaseResult(
                success=True,
                data=data,
                count=1,
                query=pathway_id,
                source=self.NAME,
                message=f"Retrieved pathway {pathway_id}",
            )
            
        except Exception as e:
            logger.error(f"Failed to get pathway {pathway_id}: {e}")
            return self._error_result(pathway_id, e)
    
    def get_pathway_genes(self, pathway_id: str, organism: str = "hsa") -> List[str]:
        """
        Get genes in a pathway.
        
        Args:
            pathway_id: KEGG pathway ID
            organism: Organism code
            
        Returns:
            List of gene symbols
        """
        result = self.get_pathway(pathway_id)
        
        if not result.success or not result.data:
            return []
        
        genes = result.data.get("genes", [])
        
        # Extract gene symbols (format: "gene_id; symbol; description")
        symbols = []
        for gene in genes:
            parts = gene.split(";")
            if len(parts) >= 2:
                symbol = parts[1].strip()
                symbols.append(symbol)
            elif parts:
                # Try to extract from the first part
                symbols.append(parts[0].strip().split()[0])
        
        return symbols
    
    def list_pathways(self, organism: str = "hsa") -> DatabaseResult:
        """
        List all pathways for an organism.
        
        Args:
            organism: KEGG organism code
            
        Returns:
            DatabaseResult with all pathways
        """
        org_code = self.ORGANISM_MAP.get(organism.lower(), organism)
        
        try:
            response = self._request(
                "GET",
                f"{self.BASE_URL}/list/pathway/{org_code}",
            )
            
            results = self._parse_kegg_list(response)
            
            return DatabaseResult(
                success=True,
                data=results,
                count=len(results),
                query=f"pathways for {org_code}",
                source=self.NAME,
                message=f"Found {len(results)} pathways for {org_code}",
            )
            
        except Exception as e:
            logger.error(f"Failed to list pathways: {e}")
            return self._error_result(f"pathways for {organism}", e)
    
    def get_gene(self, gene_id: str) -> DatabaseResult:
        """
        Get gene information.
        
        Args:
            gene_id: KEGG gene ID (e.g., "hsa:7157" for TP53)
            
        Returns:
            DatabaseResult with gene details
        """
        try:
            response = self._request(
                "GET",
                f"{self.BASE_URL}/get/{gene_id}",
            )
            
            data = self._parse_kegg_entry(response)
            
            return DatabaseResult(
                success=True,
                data=data,
                count=1,
                query=gene_id,
                source=self.NAME,
                message=f"Retrieved gene {gene_id}",
            )
            
        except Exception as e:
            logger.error(f"Failed to get gene {gene_id}: {e}")
            return self._error_result(gene_id, e)
    
    def convert_ids(
        self,
        ids: List[str],
        source_db: str,
        target_db: str,
    ) -> DatabaseResult:
        """
        Convert identifiers between databases.
        
        Args:
            ids: List of identifiers
            source_db: Source database (e.g., "ncbi-geneid", "uniprot")
            target_db: Target database (e.g., "hsa", "kegg")
            
        Returns:
            DatabaseResult with converted IDs
        """
        try:
            response = self._request(
                "GET",
                f"{self.BASE_URL}/conv/{target_db}/{source_db}:" + "+".join(ids),
            )
            
            results = self._parse_kegg_list(response)
            
            return DatabaseResult(
                success=True,
                data=results,
                count=len(results),
                query=f"convert {source_db} to {target_db}",
                source=self.NAME,
                message=f"Converted {len(results)} IDs",
            )
            
        except Exception as e:
            logger.error(f"ID conversion failed: {e}")
            return self._error_result(f"convert {source_db} to {target_db}", e)
    
    def get_pathway_image(self, pathway_id: str) -> Optional[str]:
        """
        Get URL for pathway image.
        
        Args:
            pathway_id: KEGG pathway ID
            
        Returns:
            URL to pathway image (PNG)
        """
        # Clean pathway ID
        clean_id = pathway_id.replace("path:", "")
        return f"https://www.kegg.jp/kegg/pathway/{clean_id[:3]}/{clean_id}.png"
    
    def enrich_genes(
        self,
        genes: List[str],
        organism: str = "hsa",
        limit: int = 20,
    ) -> DatabaseResult:
        """
        Perform pathway enrichment analysis on a gene list.
        
        Note: This uses a simple overlap method. For proper statistical
        enrichment, use external tools like clusterProfiler or enrichR.
        
        Args:
            genes: List of gene symbols
            organism: Organism code
            limit: Maximum pathways to return
            
        Returns:
            DatabaseResult with pathways containing the input genes
        """
        org_code = self.ORGANISM_MAP.get(organism.lower(), organism)
        
        # Get all pathways
        all_pathways = self.list_pathways(org_code)
        if not all_pathways.success:
            return all_pathways
        
        # Check overlap for each pathway
        enriched = []
        genes_set = set(g.upper() for g in genes)
        
        for pathway in all_pathways.data[:100]:  # Limit to first 100 pathways
            pathway_id = pathway.get("id", "")
            
            # Get genes in this pathway
            pathway_genes = self.get_pathway_genes(pathway_id, org_code)
            pathway_genes_set = set(g.upper() for g in pathway_genes)
            
            # Calculate overlap
            overlap = genes_set & pathway_genes_set
            
            if overlap:
                enriched.append({
                    "pathway_id": pathway_id,
                    "pathway_name": pathway.get("name", ""),
                    "overlap_genes": list(overlap),
                    "overlap_count": len(overlap),
                    "pathway_size": len(pathway_genes),
                    "input_genes": len(genes),
                    "ratio": len(overlap) / len(pathway_genes) if pathway_genes else 0,
                })
        
        # Sort by overlap ratio
        enriched.sort(key=lambda x: x["ratio"], reverse=True)
        
        return DatabaseResult(
            success=True,
            data=enriched[:limit],
            count=len(enriched),
            query=f"{len(genes)} genes",
            source=f"{self.NAME} Enrichment",
            message=f"Found {len(enriched)} pathways with gene overlap",
        )
    
    def _parse_kegg_list(self, text: str) -> List[Dict[str, str]]:
        """Parse KEGG list format (tab-separated ID and name)."""
        if not isinstance(text, str):
            return []
        
        results = []
        for line in text.strip().split("\n"):
            if line:
                parts = line.split("\t")
                if len(parts) >= 2:
                    results.append({
                        "id": parts[0].strip(),
                        "name": parts[1].strip() if len(parts) > 1 else "",
                    })
                elif parts:
                    results.append({
                        "id": parts[0].strip(),
                        "name": "",
                    })
        
        return results
    
    def _parse_kegg_entry(self, text: str) -> Dict[str, Any]:
        """Parse KEGG flat file format."""
        if not isinstance(text, str):
            return {}
        
        entry: Dict[str, Any] = {}
        current_field = None
        current_value: List[str] = []
        
        for line in text.split("\n"):
            if line.startswith("///"):
                # End of entry
                break
            elif line.startswith(" ") and current_field:
                # Continuation of previous field
                current_value.append(line.strip())
            elif line:
                # New field
                # Save previous field
                if current_field:
                    if current_field in ["GENE", "COMPOUND", "MODULE", "DISEASE"]:
                        entry[current_field.lower() + "s"] = current_value
                    elif len(current_value) == 1:
                        entry[current_field.lower()] = current_value[0]
                    else:
                        entry[current_field.lower()] = current_value
                
                # Parse new field
                parts = line.split(None, 1)
                if parts:
                    current_field = parts[0].rstrip(":")
                    current_value = [parts[1]] if len(parts) > 1 else []
        
        # Save last field
        if current_field:
            if current_field in ["GENE", "COMPOUND", "MODULE", "DISEASE"]:
                entry[current_field.lower() + "s"] = current_value
            elif len(current_value) == 1:
                entry[current_field.lower()] = current_value[0]
            else:
                entry[current_field.lower()] = current_value
        
        return entry
