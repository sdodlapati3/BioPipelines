"""
GEO/SRA Adapter
===============

Search and download data from NCBI GEO and SRA.

- GEO (Gene Expression Omnibus): High-throughput functional genomics data
- SRA (Sequence Read Archive): Raw sequencing data

Uses NCBI Entrez E-utilities API.
"""

import logging
import re
from typing import List, Optional, Dict, Any
from datetime import datetime

import requests

from .base import BaseAdapter
from ..models import (
    SearchQuery, DatasetInfo, DownloadURL, DataSource,
    FileType, DownloadMethod
)

logger = logging.getLogger(__name__)

# Try to import Biopython for Entrez
try:
    from Bio import Entrez
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    logger.warning("Biopython not installed. GEO/SRA search will use REST API fallback.")


class GEOAdapter(BaseAdapter):
    """
    Adapter for NCBI GEO and SRA databases.
    
    Usage:
        adapter = GEOAdapter(email="your@email.com")
        results = adapter.search(SearchQuery(
            organism="human",
            assay_type="RNA-seq",
            tissue="brain"
        ))
        
        for dataset in results:
            print(f"{dataset.id}: {dataset.title}")
    """
    
    SOURCE = DataSource.GEO
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    GEO_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    SRA_URL = "https://www.ncbi.nlm.nih.gov/sra"
    
    def __init__(
        self,
        email: str = "biopipelines@example.com",
        api_key: Optional[str] = None,
        cache_enabled: bool = True,
        timeout: int = 30
    ):
        """
        Initialize the GEO/SRA adapter.
        
        Args:
            email: Email for NCBI API (required by NCBI)
            api_key: NCBI API key (optional, increases rate limit)
            cache_enabled: Whether to cache results
            timeout: Request timeout in seconds
        """
        super().__init__(cache_enabled, timeout)
        self.email = email
        self.api_key = api_key
        
        if BIOPYTHON_AVAILABLE:
            Entrez.email = email
            if api_key:
                Entrez.api_key = api_key
        
        self.session = requests.Session()
    
    def search(self, query: SearchQuery) -> List[DatasetInfo]:
        """
        Search GEO for datasets.
        
        Args:
            query: Structured search query
            
        Returns:
            List of matching datasets (GSE series)
        """
        # Check cache
        cache_key = self._build_cache_key(query)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        # Build search term
        search_term = self._build_search_term(query)
        logger.info(f"Searching GEO: {search_term}")
        
        try:
            if BIOPYTHON_AVAILABLE:
                datasets = self._search_with_biopython(search_term, query.max_results)
            else:
                datasets = self._search_with_rest(search_term, query.max_results)
            
            self._set_cached(cache_key, datasets)
            logger.info(f"Found {len(datasets)} datasets from GEO")
            return datasets
            
        except Exception as e:
            logger.error(f"GEO search failed: {e}")
            return []
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetInfo]:
        """
        Get detailed information about a GEO dataset.
        
        Args:
            dataset_id: GEO accession (GSE, GSM, or SRX)
            
        Returns:
            Dataset info or None
        """
        cache_key = f"dataset:{dataset_id}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            # Determine ID type and fetch accordingly
            if dataset_id.startswith("GSE"):
                dataset = self._get_gse(dataset_id)
            elif dataset_id.startswith("GSM"):
                dataset = self._get_gsm(dataset_id)
            elif dataset_id.startswith("SRX") or dataset_id.startswith("SRR"):
                dataset = self._get_sra(dataset_id)
            else:
                logger.warning(f"Unknown GEO/SRA ID format: {dataset_id}")
                return None
            
            if dataset:
                self._set_cached(cache_key, dataset)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to get GEO dataset {dataset_id}: {e}")
            return None
    
    def get_download_urls(self, dataset_id: str) -> List[DownloadURL]:
        """
        Get download URLs for a GEO/SRA dataset.
        
        Args:
            dataset_id: GEO or SRA accession
            
        Returns:
            List of downloadable files
        """
        urls = []
        
        try:
            # For SRA data, we use fasterq-dump (SRA toolkit)
            if dataset_id.startswith("SRR"):
                urls.append(DownloadURL(
                    url=f"sra://{dataset_id}",
                    filename=f"{dataset_id}.fastq",
                    file_type=FileType.FASTQ,
                    download_method=DownloadMethod.SRA,
                ))
            
            elif dataset_id.startswith("GSE"):
                # Get supplementary files from GEO
                supp_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{dataset_id[:6]}nnn/{dataset_id}/suppl/"
                urls.append(DownloadURL(
                    url=supp_url,
                    filename=f"{dataset_id}_suppl.tar",
                    file_type=FileType.OTHER,
                    download_method=DownloadMethod.FTP,
                ))
                
                # Also get SRA runs associated with this GSE
                sra_runs = self._get_sra_runs_for_gse(dataset_id)
                for run_id in sra_runs[:10]:  # Limit to first 10
                    urls.append(DownloadURL(
                        url=f"sra://{run_id}",
                        filename=f"{run_id}.fastq",
                        file_type=FileType.FASTQ,
                        download_method=DownloadMethod.SRA,
                    ))
            
            return urls
            
        except Exception as e:
            logger.error(f"Failed to get download URLs for {dataset_id}: {e}")
            return []
    
    def _build_search_term(self, query: SearchQuery) -> str:
        """Build NCBI search term from query."""
        terms = []
        
        if query.organism:
            terms.append(f"{query.organism}[Organism]")
        
        if query.assay_type:
            # Map common assay names to GEO terms
            assay_map = {
                "RNA-seq": "RNA-Seq",
                "ChIP-seq": "ChIP-Seq",
                "ATAC-seq": "ATAC-Seq",
                "scRNA-seq": "single cell RNA-Seq",
                "WGS": "WGS",
                "WGBS": "bisulfite",
                "Bisulfite-seq": "bisulfite",
                "methylation": "methylation",
            }
            assay_term = assay_map.get(query.assay_type, query.assay_type)
            terms.append(f'"{assay_term}"[Title/Abstract]')
        
        if query.target:
            terms.append(f"{query.target}[Title/Abstract]")
        
        if query.tissue:
            terms.append(f"{query.tissue}[Title/Abstract]")
        
        if query.cell_line:
            terms.append(f"{query.cell_line}[Title/Abstract]")
        
        # For keywords, combine into a single term with OR to avoid over-filtering
        # Skip keywords that are already covered by organism/tissue/assay
        if query.keywords:
            # Filter out duplicates and already-used terms
            skip_terms = set()
            if query.organism:
                skip_terms.update(query.organism.lower().split())
            if query.tissue:
                skip_terms.update(query.tissue.lower().split())
            if query.assay_type:
                skip_terms.update(query.assay_type.lower().replace("-", "").split())
            
            unique_keywords = []
            for kw in query.keywords:
                kw_lower = kw.lower()
                if kw_lower not in skip_terms and kw not in unique_keywords:
                    unique_keywords.append(kw)
            
            # Only add a few keywords to avoid over-filtering
            # Use OR to be less restrictive
            if unique_keywords:
                if len(unique_keywords) <= 2:
                    for kw in unique_keywords:
                        terms.append(f"{kw}[Title/Abstract]")
                else:
                    # Combine with OR for flexibility
                    kw_term = " OR ".join(f"{kw}[Title/Abstract]" for kw in unique_keywords[:3])
                    terms.append(f"({kw_term})")
        
        # Only add dataset type filter if we have other terms
        if terms:
            terms.append('"Expression profiling by high throughput sequencing"[DataSet Type]')
        
        return " AND ".join(terms) if terms else "RNA-Seq[Title]"
    
    def _search_with_biopython(self, search_term: str, max_results: int) -> List[DatasetInfo]:
        """Search using Biopython Entrez."""
        # Search GEO DataSets database
        handle = Entrez.esearch(
            db="gds",
            term=search_term,
            retmax=max_results,
            sort="relevance"
        )
        search_results = Entrez.read(handle)
        handle.close()
        
        ids = search_results.get("IdList", [])
        if not ids:
            return []
        
        # Fetch details
        handle = Entrez.esummary(db="gds", id=",".join(ids))
        summaries = Entrez.read(handle)
        handle.close()
        
        datasets = []
        for summary in summaries:
            dataset = self._parse_gds_summary(summary)
            if dataset:
                datasets.append(dataset)
        
        return datasets
    
    def _search_with_rest(self, search_term: str, max_results: int) -> List[DatasetInfo]:
        """Search using REST API (fallback if Biopython unavailable)."""
        # ESearch
        search_url = f"{self.BASE_URL}/esearch.fcgi"
        params = {
            "db": "gds",
            "term": search_term,
            "retmax": max_results,
            "retmode": "json",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        
        response = self.session.get(search_url, params=params, timeout=self.timeout)
        response.raise_for_status()
        search_data = response.json()
        
        ids = search_data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        
        # ESummary
        summary_url = f"{self.BASE_URL}/esummary.fcgi"
        params = {
            "db": "gds",
            "id": ",".join(ids),
            "retmode": "json",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        
        response = self.session.get(summary_url, params=params, timeout=self.timeout)
        response.raise_for_status()
        summary_data = response.json()
        
        datasets = []
        result = summary_data.get("result", {})
        for uid in result.get("uids", []):
            summary = result.get(uid, {})
            dataset = self._parse_gds_summary_json(summary)
            if dataset:
                datasets.append(dataset)
        
        return datasets
    
    def _parse_gds_summary(self, summary: Dict[str, Any]) -> Optional[DatasetInfo]:
        """Parse a GDS summary from Biopython Entrez."""
        try:
            accession = summary.get("Accession", "")
            if not accession:
                return None
            
            return DatasetInfo(
                id=accession,
                source=DataSource.GEO,
                title=summary.get("title", ""),
                description=summary.get("summary", ""),
                organism=summary.get("taxon", ""),
                assay_type=summary.get("gdsType", ""),
                metadata={
                    "n_samples": summary.get("n_samples", 0),
                    "pubmed_ids": summary.get("pubmed_ids", []),
                    "platform": summary.get("GPL", ""),
                },
                web_url=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
            )
        except Exception as e:
            logger.warning(f"Failed to parse GDS summary: {e}")
            return None
    
    def _parse_gds_summary_json(self, summary: Dict[str, Any]) -> Optional[DatasetInfo]:
        """Parse a GDS summary from JSON response."""
        try:
            accession = summary.get("accession", "")
            if not accession:
                return None
            
            return DatasetInfo(
                id=accession,
                source=DataSource.GEO,
                title=summary.get("title", ""),
                description=summary.get("summary", ""),
                organism=summary.get("taxon", ""),
                assay_type=summary.get("gdstype", ""),
                file_count=int(summary.get("n_samples", 0)),
                metadata={
                    "n_samples": summary.get("n_samples", 0),
                    "platform": summary.get("gpl", ""),
                },
                web_url=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
            )
        except Exception as e:
            logger.warning(f"Failed to parse GDS summary JSON: {e}")
            return None
    
    def _get_gse(self, gse_id: str) -> Optional[DatasetInfo]:
        """Get details for a GSE series."""
        url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
        params = {"acc": gse_id, "targ": "self", "view": "brief", "form": "text"}
        
        response = self.session.get(url, params=params, timeout=self.timeout)
        if not response.ok:
            return None
        
        # Parse text response (simple key-value pairs)
        text = response.text
        title_match = re.search(r"!Series_title\s*=\s*(.+)", text)
        summary_match = re.search(r"!Series_summary\s*=\s*(.+)", text)
        organism_match = re.search(r"!Series_sample_organism\s*=\s*(.+)", text)
        
        return DatasetInfo(
            id=gse_id,
            source=DataSource.GEO,
            title=title_match.group(1).strip() if title_match else gse_id,
            description=summary_match.group(1).strip() if summary_match else "",
            organism=organism_match.group(1).strip() if organism_match else "",
            web_url=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}",
        )
    
    def _get_gsm(self, gsm_id: str) -> Optional[DatasetInfo]:
        """Get details for a GSM sample."""
        # Similar to _get_gse but for samples
        return DatasetInfo(
            id=gsm_id,
            source=DataSource.GEO,
            title=gsm_id,
            description="GEO Sample",
            web_url=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gsm_id}",
        )
    
    def _get_sra(self, sra_id: str) -> Optional[DatasetInfo]:
        """Get details for an SRA accession."""
        return DatasetInfo(
            id=sra_id,
            source=DataSource.SRA,
            title=sra_id,
            description="SRA Run",
            web_url=f"https://www.ncbi.nlm.nih.gov/sra/{sra_id}",
            download_urls=[
                DownloadURL(
                    url=f"sra://{sra_id}",
                    filename=f"{sra_id}.fastq",
                    file_type=FileType.FASTQ,
                    download_method=DownloadMethod.SRA,
                )
            ],
        )
    
    def _get_sra_runs_for_gse(self, gse_id: str) -> List[str]:
        """Get SRA run accessions associated with a GSE."""
        try:
            # Use ELink to find SRA entries linked to GDS
            if BIOPYTHON_AVAILABLE:
                # First get GDS ID from GSE
                handle = Entrez.esearch(db="gds", term=f"{gse_id}[Accession]")
                result = Entrez.read(handle)
                handle.close()
                
                gds_ids = result.get("IdList", [])
                if not gds_ids:
                    return []
                
                # Link to SRA
                handle = Entrez.elink(dbfrom="gds", db="sra", id=gds_ids[0])
                link_result = Entrez.read(handle)
                handle.close()
                
                sra_ids = []
                for linkset in link_result:
                    for link_db in linkset.get("LinkSetDb", []):
                        for link in link_db.get("Link", []):
                            sra_ids.append(link["Id"])
                
                if not sra_ids:
                    return []
                
                # Get SRA accessions
                handle = Entrez.efetch(db="sra", id=sra_ids[:10], rettype="runinfo", retmode="text")
                runinfo = handle.read()
                handle.close()
                
                # Parse runinfo CSV
                runs = []
                for line in runinfo.split("\n")[1:]:  # Skip header
                    parts = line.split(",")
                    if parts and parts[0].startswith("SRR"):
                        runs.append(parts[0])
                
                return runs
            
            return []
            
        except Exception as e:
            logger.warning(f"Failed to get SRA runs for {gse_id}: {e}")
            return []


# Convenience function
def search_geo(
    organism: str = None,
    assay_type: str = None,
    tissue: str = None,
    keywords: List[str] = None,
    **kwargs
) -> List[DatasetInfo]:
    """
    Quick search of GEO.
    
    Args:
        organism: Species (human, mouse, etc.)
        assay_type: Experiment type (RNA-seq, ChIP-seq, etc.)
        tissue: Tissue/organ
        keywords: Additional search terms
        **kwargs: Additional query parameters
        
    Returns:
        List of matching datasets
    """
    query = SearchQuery(
        organism=organism,
        assay_type=assay_type,
        tissue=tissue,
        keywords=keywords or [],
        **kwargs
    )
    adapter = GEOAdapter()
    return adapter.search(query)
