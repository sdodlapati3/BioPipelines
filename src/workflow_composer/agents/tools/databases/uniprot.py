"""
UniProt Database Client
=======================

Client for the UniProt REST API to retrieve protein sequences,
annotations, and functional information.

UniProt is the comprehensive protein sequence and functional database.
It includes:
- Swiss-Prot: Manually reviewed, high-quality annotations
- TrEMBL: Computationally analyzed, unreviewed entries

API Documentation: https://www.uniprot.org/help/api

Example:
    >>> from workflow_composer.agents.tools.databases import UniProtClient
    >>> 
    >>> client = UniProtClient()
    >>> result = client.search("BRCA1", organism="human")
    >>> print(result.count)
    25
    >>> print(result.data[0]["primaryAccession"])
    'P38398'
"""

from typing import Any, Dict, List, Optional
import logging

from .base import DatabaseClient, DatabaseResult, resolve_taxonomy_id

logger = logging.getLogger(__name__)


class UniProtClient(DatabaseClient):
    """
    Client for UniProt REST API.
    
    Provides access to protein sequences, annotations, and functional
    information from UniProt/Swiss-Prot.
    
    Features:
    - Search by gene name, protein name, or keywords
    - Filter by organism, reviewed status, etc.
    - Retrieve sequences in FASTA format
    - Get functional annotations and GO terms
    """
    
    BASE_URL = "https://rest.uniprot.org"
    NAME = "UniProt"
    RATE_LIMIT = 10.0  # UniProt allows higher rate
    
    # Default fields to retrieve
    DEFAULT_FIELDS = [
        "accession",
        "id",
        "protein_name",
        "gene_names",
        "organism_name",
        "length",
        "cc_function",
        "go_p",  # GO biological process
        "go_c",  # GO cellular component
        "go_f",  # GO molecular function
    ]
    
    def search(
        self,
        query: str,
        organism: Optional[str] = None,
        reviewed: bool = True,
        limit: int = 25,
        fields: Optional[List[str]] = None,
        **kwargs,
    ) -> DatabaseResult:
        """
        Search UniProt for proteins.
        
        Args:
            query: Search terms (gene name, protein name, keywords)
            organism: Organism filter (e.g., "human", "mouse", "9606")
            reviewed: If True, only return Swiss-Prot (reviewed) entries
            limit: Maximum results to return (max 500)
            fields: Specific fields to return (uses default if None)
            **kwargs: Additional UniProt query parameters
            
        Returns:
            DatabaseResult with protein entries
            
        Example:
            >>> result = client.search("TP53", organism="human")
            >>> print(result.data[0]["primaryAccession"])
        """
        # Build query
        query_parts = [query]
        
        if organism:
            tax_id = resolve_taxonomy_id(organism)
            query_parts.append(f"organism_id:{tax_id}")
        
        if reviewed:
            query_parts.append("reviewed:true")
        
        # Add any extra filters from kwargs
        for key, value in kwargs.items():
            if value is not None:
                query_parts.append(f"{key}:{value}")
        
        full_query = " AND ".join(query_parts)
        
        # Use default fields if not specified
        if fields is None:
            fields = self.DEFAULT_FIELDS
        
        try:
            response = self._request(
                "GET",
                f"{self.BASE_URL}/uniprotkb/search",
                params={
                    "query": full_query,
                    "format": "json",
                    "size": min(limit, 500),
                    "fields": ",".join(fields),
                },
            )
            
            results = response.get("results", [])
            
            return DatabaseResult(
                success=True,
                data=results,
                count=len(results),
                query=query,
                source=self.NAME,
                message=f"Found {len(results)} proteins",
                metadata={
                    "full_query": full_query,
                    "reviewed_only": reviewed,
                    "organism": organism,
                },
            )
            
        except Exception as e:
            logger.error(f"UniProt search failed: {e}")
            return self._error_result(query, e)
    
    def get_by_id(
        self,
        accession: str,
        format: str = "json",
        **kwargs,
    ) -> DatabaseResult:
        """
        Get protein by UniProt accession.
        
        Args:
            accession: UniProt accession (e.g., "P38398", "BRCA1_HUMAN")
            format: Response format ("json", "fasta", "txt")
            
        Returns:
            DatabaseResult with protein data
            
        Example:
            >>> result = client.get_by_id("P38398")
            >>> print(result.data["proteinDescription"]["recommendedName"]["fullName"]["value"])
            'Breast cancer type 1 susceptibility protein'
        """
        try:
            response = self._request(
                "GET",
                f"{self.BASE_URL}/uniprotkb/{accession}.{format}",
            )
            
            return DatabaseResult(
                success=True,
                data=response,
                count=1,
                query=accession,
                source=self.NAME,
                message=f"Retrieved {accession}",
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve {accession}: {e}")
            return self._error_result(accession, e)
    
    def get_sequence(self, accession: str) -> Optional[str]:
        """
        Get protein sequence in FASTA format.
        
        Args:
            accession: UniProt accession
            
        Returns:
            FASTA-formatted sequence string, or None if not found
        """
        try:
            response = self._request(
                "GET",
                f"{self.BASE_URL}/uniprotkb/{accession}.fasta",
            )
            return response if isinstance(response, str) else None
            
        except Exception as e:
            logger.error(f"Failed to get sequence for {accession}: {e}")
            return None
    
    def get_function(self, accession: str) -> Optional[str]:
        """
        Get the functional description for a protein.
        
        Args:
            accession: UniProt accession
            
        Returns:
            Functional description string, or None if not found
        """
        result = self.get_by_id(accession)
        
        if not result.success or not result.data:
            return None
        
        try:
            comments = result.data.get("comments", [])
            for comment in comments:
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        return texts[0].get("value", "")
        except Exception as e:
            logger.debug(f"Error extracting function for {accession}: {e}")
        
        return None
    
    def get_go_terms(
        self,
        accession: str,
        aspect: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Get Gene Ontology terms for a protein.
        
        Args:
            accession: UniProt accession
            aspect: Filter by aspect ("P"=process, "C"=component, "F"=function)
            
        Returns:
            List of GO term dictionaries with id and name
        """
        result = self.get_by_id(accession)
        
        if not result.success or not result.data:
            return []
        
        try:
            go_terms = []
            
            # Parse GO annotations from uniprotKBCrossReferences
            xrefs = result.data.get("uniProtKBCrossReferences", [])
            
            for xref in xrefs:
                if xref.get("database") == "GO":
                    go_id = xref.get("id", "")
                    properties = xref.get("properties", [])
                    
                    term_name = ""
                    term_aspect = ""
                    
                    for prop in properties:
                        if prop.get("key") == "GoTerm":
                            term_name = prop.get("value", "")
                            # Extract aspect from prefix
                            if term_name.startswith("P:"):
                                term_aspect = "P"
                                term_name = term_name[2:]
                            elif term_name.startswith("C:"):
                                term_aspect = "C"
                                term_name = term_name[2:]
                            elif term_name.startswith("F:"):
                                term_aspect = "F"
                                term_name = term_name[2:]
                    
                    # Apply aspect filter if specified
                    if aspect and term_aspect != aspect:
                        continue
                    
                    go_terms.append({
                        "id": go_id,
                        "name": term_name,
                        "aspect": term_aspect,
                    })
            
            return go_terms
            
        except Exception as e:
            logger.debug(f"Error extracting GO terms for {accession}: {e}")
            return []
    
    def search_by_gene(
        self,
        gene_name: str,
        organism: str = "human",
        reviewed: bool = True,
    ) -> DatabaseResult:
        """
        Search for proteins by gene name.
        
        Convenience method that searches specifically in gene name fields.
        
        Args:
            gene_name: Gene symbol (e.g., "BRCA1", "TP53")
            organism: Organism name or taxonomy ID
            reviewed: Only return reviewed entries
            
        Returns:
            DatabaseResult with matching proteins
        """
        return self.search(
            f'gene_exact:"{gene_name}"',
            organism=organism,
            reviewed=reviewed,
        )
    
    def batch_retrieve(
        self,
        accessions: List[str],
        fields: Optional[List[str]] = None,
    ) -> DatabaseResult:
        """
        Retrieve multiple proteins by accession.
        
        Args:
            accessions: List of UniProt accessions
            fields: Fields to retrieve
            
        Returns:
            DatabaseResult with all protein data
        """
        if not accessions:
            return self._empty_result("", "No accessions provided")
        
        # Build query with OR
        query = " OR ".join(f"accession:{acc}" for acc in accessions)
        
        return self.search(
            query,
            reviewed=False,  # Don't filter by reviewed for batch
            limit=len(accessions),
            fields=fields,
        )
    
    def get_protein_summary(self, accession: str) -> Optional[Dict[str, Any]]:
        """
        Get a concise summary of a protein.
        
        Args:
            accession: UniProt accession
            
        Returns:
            Dictionary with key protein information
        """
        result = self.get_by_id(accession)
        
        if not result.success or not result.data:
            return None
        
        try:
            data = result.data
            
            # Extract protein name
            protein_name = "Unknown"
            if "proteinDescription" in data:
                desc = data["proteinDescription"]
                if "recommendedName" in desc:
                    protein_name = desc["recommendedName"].get("fullName", {}).get("value", "Unknown")
                elif "submittedName" in desc and desc["submittedName"]:
                    protein_name = desc["submittedName"][0].get("fullName", {}).get("value", "Unknown")
            
            # Extract gene names
            gene_names = []
            for gene in data.get("genes", []):
                if "geneName" in gene:
                    gene_names.append(gene["geneName"].get("value", ""))
            
            # Extract organism
            organism = data.get("organism", {}).get("scientificName", "Unknown")
            
            # Get sequence length
            sequence_length = data.get("sequence", {}).get("length", 0)
            
            # Get function
            function = self.get_function(accession) or "No function description available"
            
            return {
                "accession": accession,
                "protein_name": protein_name,
                "gene_names": gene_names,
                "organism": organism,
                "sequence_length": sequence_length,
                "function": function[:500] if len(function) > 500 else function,  # Truncate long descriptions
            }
            
        except Exception as e:
            logger.debug(f"Error creating summary for {accession}: {e}")
            return None
