"""
ENCODE Portal Adapter
=====================

Search and download data from the ENCODE Project (https://www.encodeproject.org).

ENCODE provides high-quality functional genomics data including:
- ChIP-seq, ATAC-seq, DNase-seq
- RNA-seq, scRNA-seq
- Hi-C, ChIA-PET
- And more...

API Documentation: https://www.encodeproject.org/help/rest-api/
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

import requests

from .base import BaseAdapter
from ..models import (
    SearchQuery, DatasetInfo, DownloadURL, DataSource,
    FileType, DownloadMethod
)

logger = logging.getLogger(__name__)


# Map ENCODE file types to our FileType enum
ENCODE_FILE_TYPE_MAP = {
    "fastq": FileType.FASTQ_GZ,
    "bam": FileType.BAM,
    "bed": FileType.BED,
    "bigWig": FileType.BIGWIG,
    "bigBed": FileType.BIGBED,
    "vcf": FileType.VCF,
}

# Map ENCODE assay types to standard names
ENCODE_ASSAY_MAP = {
    "ChIP-seq": "ChIP-seq",
    "ATAC-seq": "ATAC-seq",
    "RNA-seq": "RNA-seq",
    "DNase-seq": "DNase-seq",
    "Hi-C": "Hi-C",
    "WGBS": "WGBS",
    "RRBS": "RRBS",
    "single-cell RNA sequencing assay": "scRNA-seq",
    "CUT&RUN": "CUT&RUN",
    "CUT&Tag": "CUT&Tag",
}


class ENCODEAdapter(BaseAdapter):
    """
    Adapter for the ENCODE Project portal.
    
    Usage:
        adapter = ENCODEAdapter()
        results = adapter.search(SearchQuery(
            organism="human",
            assay_type="ChIP-seq",
            target="H3K27ac"
        ))
        
        for dataset in results:
            print(f"{dataset.id}: {dataset.title}")
    """
    
    SOURCE = DataSource.ENCODE
    BASE_URL = "https://www.encodeproject.org"
    SEARCH_URL = f"{BASE_URL}/search/"
    
    def __init__(self, cache_enabled: bool = True, timeout: int = 30):
        """Initialize the ENCODE adapter."""
        super().__init__(cache_enabled, timeout)
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "BioPipelines/1.0"
        })
    
    def search(self, query: SearchQuery) -> List[DatasetInfo]:
        """
        Search ENCODE for datasets matching the query.
        
        Args:
            query: Structured search query
            
        Returns:
            List of matching datasets
        """
        # Check cache
        cache_key = self._build_cache_key(query)
        cached = self._get_cached(cache_key)
        if cached:
            logger.debug(f"Using cached results for {cache_key}")
            return cached
        
        # Build search parameters
        params = self._build_search_params(query)
        
        logger.info(f"Searching ENCODE: {params}")
        
        try:
            # Build URL manually to handle ENCODE's specific format
            # ENCODE expects: /search/?type=Experiment&field=accession&...
            url = self._build_search_url(params)
            logger.debug(f"ENCODE search URL: {url}")
            
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            # Parse results
            datasets = []
            for item in data.get("@graph", []):
                dataset = self._parse_experiment(item)
                if dataset:
                    datasets.append(dataset)
            
            # Cache results
            self._set_cached(cache_key, datasets)
            
            logger.info(f"Found {len(datasets)} datasets from ENCODE")
            return datasets
            
        except requests.RequestException as e:
            logger.error(f"ENCODE search failed: {e}")
            # Try alternative search approach
            return self._fallback_search(query)
    
    def _build_search_url(self, params: list) -> str:
        """
        Build ENCODE search URL with proper encoding.
        
        ENCODE API is picky about URL format - build it manually.
        """
        from urllib.parse import quote
        
        base = f"{self.BASE_URL}/search/?"
        param_parts = []
        
        for key, value in params:
            # URL encode the value properly
            encoded_value = quote(str(value), safe='')
            param_parts.append(f"{key}={encoded_value}")
        
        return base + "&".join(param_parts)
    
    def _fallback_search(self, query: SearchQuery) -> List[DatasetInfo]:
        """
        Fallback search using simpler ENCODE API parameters.
        
        When the full search fails, try a simpler approach.
        """
        try:
            # Use searchTerm for a broader search
            search_terms = []
            if query.assay_type:
                search_terms.append(query.assay_type)
            if query.tissue:
                search_terms.append(query.tissue)
            if query.target:
                search_terms.append(query.target)
            # Add organism last (less important for filtering)
            if query.organism:
                search_terms.append(query.organism)
            
            # Also add keywords
            if query.keywords:
                search_terms.extend(query.keywords)
            
            if not search_terms:
                return []
            
            # Simple search URL - combine terms with + for AND search
            search_term = " ".join(search_terms)
            # Limit to 25 results max for reliability
            max_results = min(query.max_results, 25)
            url = f"{self.BASE_URL}/search/?type=Experiment&format=json&limit={max_results}&status=released&searchTerm={requests.utils.quote(search_term)}"
            
            logger.info(f"ENCODE fallback search: {url}")
            
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            datasets = []
            for item in data.get("@graph", []):
                dataset = self._parse_experiment(item)
                if dataset:
                    datasets.append(dataset)
            
            logger.info(f"Fallback found {len(datasets)} datasets")
            return datasets
            
        except Exception as e:
            logger.error(f"ENCODE fallback search also failed: {e}")
            return []
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetInfo]:
        """
        Get detailed information about an ENCODE experiment.
        
        Args:
            dataset_id: ENCODE accession (e.g., ENCSR000ABC)
            
        Returns:
            Dataset info or None
        """
        # Check cache
        cache_key = f"dataset:{dataset_id}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        url = f"{self.BASE_URL}/experiments/{dataset_id}/"
        
        try:
            response = self.session.get(
                url,
                params={"format": "json"},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            dataset = self._parse_experiment(data)
            
            if dataset:
                # Get files
                dataset.download_urls = self.get_download_urls(dataset_id)
                self._set_cached(cache_key, dataset)
            
            return dataset
            
        except requests.RequestException as e:
            logger.error(f"Failed to get ENCODE experiment {dataset_id}: {e}")
            return None
    
    def get_download_urls(self, dataset_id: str) -> List[DownloadURL]:
        """
        Get download URLs for an ENCODE experiment.
        
        Args:
            dataset_id: ENCODE accession
            
        Returns:
            List of downloadable files
        """
        url = f"{self.BASE_URL}/experiments/{dataset_id}/"
        
        try:
            response = self.session.get(
                url,
                params={"format": "json"},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            files = []
            for file_ref in data.get("files", []):
                # Get file details
                file_url = f"{self.BASE_URL}{file_ref}"
                file_response = self.session.get(
                    file_url,
                    params={"format": "json"},
                    timeout=self.timeout
                )
                
                if file_response.ok:
                    file_data = file_response.json()
                    download_url = self._parse_file(file_data)
                    if download_url:
                        files.append(download_url)
            
            return files
            
        except requests.RequestException as e:
            logger.error(f"Failed to get ENCODE files for {dataset_id}: {e}")
            return []
    
    def _build_search_params(self, query: SearchQuery) -> List[tuple]:
        """Build ENCODE search parameters from query."""
        # Build params as list of tuples - keep it SIMPLE to avoid 404 errors
        params = [
            ("type", "Experiment"),
            ("format", "json"),
            ("limit", str(min(query.max_results, 25))),  # Cap at 25 for reliability
            ("status", "released"),
        ]
        
        # Build a single searchTerm combining all criteria
        search_parts = []
        
        # Assay type
        if query.assay_type:
            assay_lower = query.assay_type.lower()
            if "chip" in assay_lower:
                search_parts.append("ChIP-seq")
            elif "rna-seq" in assay_lower or "rnaseq" in assay_lower:
                search_parts.append("RNA-seq")
            elif "atac" in assay_lower:
                search_parts.append("ATAC-seq")
            elif "dnase" in assay_lower:
                search_parts.append("DNase-seq")
            elif "wgbs" in assay_lower or "methylation" in assay_lower or "bisulfite" in assay_lower:
                search_parts.append("methylation")
            elif "hi-c" in assay_lower or "hic" in assay_lower:
                search_parts.append("Hi-C")
            else:
                search_parts.append(query.assay_type)
        
        # Tissue
        if query.tissue:
            search_parts.append(query.tissue)
        
        # Cell line
        if query.cell_line:
            search_parts.append(query.cell_line)
            
        # Target
        if query.target:
            search_parts.append(query.target)
        
        # Organism - add last as it's less specific
        if query.organism:
            search_parts.append(query.organism)
        
        # Keywords
        if query.keywords:
            # Only use first 3 keywords to avoid overly long URLs
            search_parts.extend(query.keywords[:3])
        
        # Combine into single searchTerm
        if search_parts:
            # Use only unique terms
            unique_terms = list(dict.fromkeys(search_parts))  # Preserves order, removes dupes
            params.append(("searchTerm", " ".join(unique_terms)))
        
        return params
    
    def _parse_experiment(self, data: Dict[str, Any]) -> Optional[DatasetInfo]:
        """Parse an ENCODE experiment JSON to DatasetInfo."""
        try:
            accession = data.get("accession", data.get("@id", "").split("/")[-2])
            if not accession:
                return None
            
            # Get organism
            organism = ""
            replicates = data.get("replicates", [])
            if replicates:
                try:
                    organism = replicates[0].get("library", {}).get(
                        "biosample", {}
                    ).get("donor", {}).get("organism", {}).get("scientific_name", "")
                except (KeyError, IndexError, TypeError):
                    pass
            
            # Get biosample info
            biosample = data.get("biosample_ontology", {})
            tissue = biosample.get("term_name", "")
            cell_line = biosample.get("cell_slims", [""])[0] if biosample.get("cell_slims") else ""
            
            # Get target
            target_info = data.get("target", {})
            target = target_info.get("label", "") if isinstance(target_info, dict) else ""
            
            # Get assay
            assay_type = data.get("assay_title", "")
            assay_type = ENCODE_ASSAY_MAP.get(assay_type, assay_type)
            
            # Parse release date
            date_released = None
            if data.get("date_released"):
                try:
                    date_released = datetime.fromisoformat(data["date_released"])
                except ValueError:
                    pass
            
            return DatasetInfo(
                id=accession,
                source=DataSource.ENCODE,
                title=data.get("description", accession),
                description=data.get("description", ""),
                organism=organism,
                assembly=data.get("assembly", [""])[0] if data.get("assembly") else None,
                assay_type=assay_type,
                target=target,
                tissue=tissue,
                cell_line=cell_line,
                metadata={
                    "lab": data.get("lab", {}).get("title", ""),
                    "award": data.get("award", {}).get("project", ""),
                    "status": data.get("status", ""),
                },
                date_released=date_released,
                web_url=f"{self.BASE_URL}/experiments/{accession}/",
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse ENCODE experiment: {e}")
            return None
    
    def _parse_file(self, data: Dict[str, Any]) -> Optional[DownloadURL]:
        """Parse an ENCODE file JSON to DownloadURL."""
        try:
            # Only include released files
            if data.get("status") != "released":
                return None
            
            # Get file format
            file_format = data.get("file_format", "")
            file_type = ENCODE_FILE_TYPE_MAP.get(file_format, FileType.OTHER)
            
            # Get download URL
            href = data.get("href", "")
            if not href:
                return None
            
            # Full URL
            url = f"{self.BASE_URL}{href}" if href.startswith("/") else href
            
            # Get filename
            filename = href.split("/")[-1] if "/" in href else data.get("accession", "file")
            
            return DownloadURL(
                url=url,
                filename=filename,
                file_type=file_type,
                size_bytes=data.get("file_size"),
                md5=data.get("md5sum"),
                download_method=DownloadMethod.HTTPS,
                replicate=data.get("biological_replicates", [""])[0] if data.get("biological_replicates") else None,
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse ENCODE file: {e}")
            return None


# Convenience function
def search_encode(
    organism: str = None,
    assay_type: str = None,
    target: str = None,
    tissue: str = None,
    **kwargs
) -> List[DatasetInfo]:
    """
    Quick search of ENCODE.
    
    Args:
        organism: Species (human, mouse, etc.)
        assay_type: Experiment type (ChIP-seq, RNA-seq, etc.)
        target: ChIP target (H3K27ac, CTCF, etc.)
        tissue: Tissue/organ
        **kwargs: Additional query parameters
        
    Returns:
        List of matching datasets
    """
    query = SearchQuery(
        organism=organism,
        assay_type=assay_type,
        target=target,
        tissue=tissue,
        **kwargs
    )
    adapter = ENCODEAdapter()
    return adapter.search(query)
