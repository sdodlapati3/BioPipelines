"""
ClinVar Database Client
=======================

Client for the NCBI ClinVar API to retrieve variant pathogenicity
and clinical significance information.

ClinVar is a freely accessible, public archive of reports of the
relationships among human variations and phenotypes, with supporting
evidence.

API Documentation: https://www.ncbi.nlm.nih.gov/clinvar/docs/api/

Example:
    >>> from workflow_composer.agents.tools.databases import ClinVarClient
    >>> 
    >>> client = ClinVarClient()
    >>> result = client.search("BRCA1", limit=10)
    >>> print(result.count)
    >>> print(result.data[0]["clinical_significance"])
"""

from typing import Any, Dict, List, Optional
import logging
import xml.etree.ElementTree as ET

from .base import DatabaseClient, DatabaseResult

logger = logging.getLogger(__name__)


class ClinVarClient(DatabaseClient):
    """
    Client for NCBI ClinVar database.
    
    Provides access to:
    - Variant pathogenicity information
    - Clinical significance annotations
    - Gene-disease associations
    - Supporting evidence
    
    Clinical significance categories:
    - Pathogenic
    - Likely pathogenic
    - Uncertain significance
    - Likely benign
    - Benign
    - Conflicting interpretations
    """
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    NAME = "ClinVar"
    RATE_LIMIT = 3.0  # NCBI rate limit
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize ClinVar client.
        
        Args:
            api_key: NCBI API key for higher rate limits
            email: Contact email (recommended by NCBI)
        """
        super().__init__(**kwargs)
        
        self.api_key = api_key
        self.email = email
        
        if api_key:
            self.RATE_LIMIT = 10.0
        
        # Try to load from environment
        if not self.api_key:
            import os
            self.api_key = os.environ.get("NCBI_API_KEY")
        
        if not self.email:
            import os
            self.email = os.environ.get("NCBI_EMAIL")
    
    def _add_api_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add API key and email to parameters."""
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        return params
    
    def search(
        self,
        query: str,
        limit: int = 20,
        significance: Optional[str] = None,
        **kwargs,
    ) -> DatabaseResult:
        """
        Search ClinVar for variants.
        
        Args:
            query: Search terms (gene name, variant, condition, etc.)
            limit: Maximum results
            significance: Filter by clinical significance:
                         "pathogenic", "likely_pathogenic", "uncertain",
                         "likely_benign", "benign"
            
        Returns:
            DatabaseResult with variant records
            
        Example:
            >>> result = client.search("BRCA1", significance="pathogenic")
        """
        # Build query
        full_query = query
        if significance:
            sig_map = {
                "pathogenic": "pathogenic[Clinical significance]",
                "likely_pathogenic": "likely pathogenic[Clinical significance]",
                "uncertain": "uncertain significance[Clinical significance]",
                "likely_benign": "likely benign[Clinical significance]",
                "benign": "benign[Clinical significance]",
            }
            sig_filter = sig_map.get(significance.lower().replace(" ", "_"))
            if sig_filter:
                full_query = f"({query}) AND {sig_filter}"
        
        try:
            # Search for variation IDs
            params = self._add_api_params({
                "db": "clinvar",
                "term": full_query,
                "retmax": limit,
                "retmode": "json",
            })
            
            response = self._request(
                "GET",
                f"{self.BASE_URL}/esearch.fcgi",
                params=params,
            )
            
            result = response.get("esearchresult", {})
            variation_ids = result.get("idlist", [])
            total_count = int(result.get("count", 0))
            
            # Fetch variant details
            if variation_ids:
                variants = self._fetch_variants(variation_ids)
            else:
                variants = []
            
            return DatabaseResult(
                success=True,
                data=variants,
                count=len(variants),
                query=query,
                source=self.NAME,
                message=f"Found {total_count} variants, returned {len(variants)}",
                metadata={
                    "total_count": total_count,
                    "significance_filter": significance,
                },
            )
            
        except Exception as e:
            logger.error(f"ClinVar search failed: {e}")
            return self._error_result(query, e)
    
    def get_by_id(
        self,
        variation_id: str,
        **kwargs,
    ) -> DatabaseResult:
        """
        Get variant by ClinVar variation ID.
        
        Args:
            variation_id: ClinVar variation ID
            
        Returns:
            DatabaseResult with variant details
        """
        variants = self._fetch_variants([variation_id])
        
        if variants:
            return DatabaseResult(
                success=True,
                data=variants[0],
                count=1,
                query=variation_id,
                source=self.NAME,
                message=f"Retrieved variant {variation_id}",
            )
        else:
            return self._empty_result(variation_id, f"Variant {variation_id} not found")
    
    def _fetch_variants(self, variation_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch variant details for a list of variation IDs."""
        if not variation_ids:
            return []
        
        try:
            params = self._add_api_params({
                "db": "clinvar",
                "id": ",".join(str(v) for v in variation_ids),
                "rettype": "vcv",
                "retmode": "xml",
            })
            
            response = self._request(
                "GET",
                f"{self.BASE_URL}/efetch.fcgi",
                params=params,
            )
            
            # Parse XML response
            return self._parse_clinvar_xml(response)
            
        except Exception as e:
            logger.error(f"Failed to fetch variants: {e}")
            return []
    
    def _parse_clinvar_xml(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse ClinVar XML response into variant dictionaries."""
        variants = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for record in root.findall(".//VariationArchive"):
                variant_data = {}
                
                # Variation ID
                variant_data["variation_id"] = record.get("VariationID", "")
                variant_data["accession"] = record.get("Accession", "")
                variant_data["version"] = record.get("Version", "")
                
                # Variation name
                name_elem = record.find(".//VariationName")
                variant_data["name"] = name_elem.text if name_elem is not None else ""
                
                # Variation type
                type_elem = record.find(".//VariationType")
                variant_data["type"] = type_elem.text if type_elem is not None else ""
                
                # Gene(s)
                genes = []
                for gene in record.findall(".//Gene"):
                    gene_data = {
                        "symbol": gene.get("Symbol", ""),
                        "gene_id": gene.get("GeneID", ""),
                        "name": gene.get("FullName", ""),
                    }
                    genes.append(gene_data)
                variant_data["genes"] = genes
                
                # Clinical significance (from Classifications)
                classifications = []
                for classification in record.findall(".//Classifications//GermlineClassification"):
                    desc = classification.find(".//Description")
                    review_status = classification.find(".//ReviewStatus")
                    
                    class_data = {
                        "description": desc.text if desc is not None else "",
                        "review_status": review_status.text if review_status is not None else "",
                    }
                    classifications.append(class_data)
                
                variant_data["classifications"] = classifications
                
                # Get primary clinical significance
                if classifications:
                    variant_data["clinical_significance"] = classifications[0].get("description", "")
                    variant_data["review_status"] = classifications[0].get("review_status", "")
                else:
                    variant_data["clinical_significance"] = ""
                    variant_data["review_status"] = ""
                
                # Conditions/Diseases
                conditions = []
                for trait in record.findall(".//TraitSet//Trait"):
                    name = trait.find(".//Name/ElementValue[@Type='Preferred']")
                    if name is not None:
                        conditions.append(name.text)
                variant_data["conditions"] = conditions
                
                # Genomic location (if available)
                for location in record.findall(".//SequenceLocation"):
                    assembly = location.get("Assembly", "")
                    if assembly.startswith("GRCh38"):
                        variant_data["chromosome"] = location.get("Chr", "")
                        variant_data["start"] = location.get("start", "")
                        variant_data["stop"] = location.get("stop", "")
                        variant_data["assembly"] = assembly
                        break
                
                # HGVS expressions
                hgvs = []
                for expr in record.findall(".//HGVSlist//HGVS"):
                    nucleotide = expr.find(".//NucleotideExpression/Expression")
                    protein = expr.find(".//ProteinExpression/Expression")
                    
                    if nucleotide is not None and nucleotide.text:
                        hgvs.append(nucleotide.text)
                    if protein is not None and protein.text:
                        hgvs.append(protein.text)
                
                variant_data["hgvs"] = hgvs[:5]  # Limit to first 5
                
                variants.append(variant_data)
                
        except ET.ParseError as e:
            logger.error(f"Failed to parse ClinVar XML: {e}")
        
        return variants
    
    def search_gene(
        self,
        gene_symbol: str,
        significance: Optional[str] = None,
        limit: int = 50,
    ) -> DatabaseResult:
        """
        Search for all variants in a gene.
        
        Args:
            gene_symbol: Gene symbol (e.g., "BRCA1")
            significance: Filter by clinical significance
            limit: Maximum results
            
        Returns:
            DatabaseResult with variants in the gene
        """
        query = f"{gene_symbol}[Gene Name]"
        return self.search(query, significance=significance, limit=limit)
    
    def search_condition(
        self,
        condition: str,
        significance: Optional[str] = None,
        limit: int = 50,
    ) -> DatabaseResult:
        """
        Search for variants associated with a condition/disease.
        
        Args:
            condition: Disease/condition name
            significance: Filter by clinical significance
            limit: Maximum results
            
        Returns:
            DatabaseResult with variants for the condition
        """
        query = f"{condition}[Disease/Phenotype]"
        return self.search(query, significance=significance, limit=limit)
    
    def search_variant(
        self,
        chromosome: str,
        position: int,
        ref: str = None,
        alt: str = None,
    ) -> DatabaseResult:
        """
        Search for a specific variant by genomic coordinates.
        
        Args:
            chromosome: Chromosome (e.g., "1", "X")
            position: Genomic position
            ref: Reference allele
            alt: Alternate allele
            
        Returns:
            DatabaseResult with matching variants
        """
        # Build coordinate query
        if ref and alt:
            query = f"chr{chromosome}:{position} {ref}>{alt}"
        else:
            query = f"chr{chromosome}:{position}"
        
        return self.search(query, limit=10)
    
    def get_pathogenic_variants(
        self,
        gene_symbol: str,
        limit: int = 100,
    ) -> DatabaseResult:
        """
        Get pathogenic and likely pathogenic variants for a gene.
        
        Args:
            gene_symbol: Gene symbol
            limit: Maximum results
            
        Returns:
            DatabaseResult with pathogenic variants
        """
        query = f'{gene_symbol}[Gene Name] AND ("pathogenic"[Clinical significance] OR "likely pathogenic"[Clinical significance])'
        return self.search(query, limit=limit)
    
    def get_variant_summary(self, variation_id: str) -> Optional[Dict[str, str]]:
        """
        Get a concise summary of a variant.
        
        Args:
            variation_id: ClinVar variation ID
            
        Returns:
            Dictionary with key variant information
        """
        result = self.get_by_id(variation_id)
        
        if not result.success or not result.data:
            return None
        
        variant = result.data
        
        # Format genes
        gene_str = ", ".join(g.get("symbol", "") for g in variant.get("genes", []))
        
        # Format conditions
        conditions_str = ", ".join(variant.get("conditions", [])[:3])
        if len(variant.get("conditions", [])) > 3:
            conditions_str += "..."
        
        # Format HGVS
        hgvs_str = variant.get("hgvs", [""])[0] if variant.get("hgvs") else ""
        
        return {
            "variation_id": variant.get("variation_id", ""),
            "accession": variant.get("accession", ""),
            "name": variant.get("name", ""),
            "type": variant.get("type", ""),
            "genes": gene_str,
            "clinical_significance": variant.get("clinical_significance", ""),
            "review_status": variant.get("review_status", ""),
            "conditions": conditions_str,
            "hgvs": hgvs_str,
            "clinvar_link": f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{variant.get('variation_id', '')}/",
        }
    
    def count_by_significance(self, gene_symbol: str) -> Dict[str, int]:
        """
        Count variants by clinical significance for a gene.
        
        Args:
            gene_symbol: Gene symbol
            
        Returns:
            Dictionary with counts per significance category
        """
        counts = {}
        
        categories = [
            ("pathogenic", "pathogenic"),
            ("likely_pathogenic", "likely pathogenic"),
            ("uncertain", "uncertain significance"),
            ("likely_benign", "likely benign"),
            ("benign", "benign"),
        ]
        
        for key, sig_term in categories:
            try:
                query = f'{gene_symbol}[Gene Name] AND "{sig_term}"[Clinical significance]'
                
                params = self._add_api_params({
                    "db": "clinvar",
                    "term": query,
                    "rettype": "count",
                    "retmode": "json",
                })
                
                response = self._request(
                    "GET",
                    f"{self.BASE_URL}/esearch.fcgi",
                    params=params,
                )
                
                count = int(response.get("esearchresult", {}).get("count", 0))
                counts[key] = count
                
            except Exception as e:
                logger.debug(f"Failed to count {key}: {e}")
                counts[key] = 0
        
        return counts
