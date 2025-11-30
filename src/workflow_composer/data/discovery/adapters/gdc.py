"""
GDC/TCGA Adapter
================

Adapter for NCI Genomic Data Commons (GDC) including TCGA, TARGET, and other projects.

API Documentation: https://docs.gdc.cancer.gov/API/Users_Guide/

Includes circuit breaker protection for API resilience.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from .base import BaseAdapter
from ..models import SearchQuery, DatasetInfo, DataSource, DownloadURL

# Import circuit breaker for resilience
from workflow_composer.infrastructure.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    get_circuit_breaker,
)

logger = logging.getLogger(__name__)

# Circuit breaker configuration for GDC API
GDC_CIRCUIT_NAME = "gdc_api"
GDC_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    success_threshold=2,
    timeout=30.0,
    failure_window=60.0,
    half_open_max_calls=1,
)


class GDCAdapter(BaseAdapter):
    """
    Adapter for searching and accessing GDC/TCGA data.
    
    Supports:
    - TCGA (The Cancer Genome Atlas) - 33 cancer types
    - TARGET (Therapeutically Applicable Research)
    - CPTAC (Clinical Proteomic Tumor Analysis Consortium)
    - Other GDC projects
    
    Data types:
    - Methylation Beta Value (450K, EPIC arrays)
    - Gene Expression Quantification (RNA-seq)
    - Aligned Reads (WGS, WXS)
    - Copy Number Variation
    - Simple Nucleotide Variation
    """
    
    BASE_URL = "https://api.gdc.cancer.gov"
    PORTAL_URL = "https://portal.gdc.cancer.gov"
    
    # Cancer type project mapping
    CANCER_TYPES = {
        "gbm": "TCGA-GBM",       # Glioblastoma Multiforme
        "glioblastoma": "TCGA-GBM",
        "brain": "TCGA-GBM",
        "lgg": "TCGA-LGG",       # Lower Grade Glioma
        "brca": "TCGA-BRCA",     # Breast Cancer
        "breast": "TCGA-BRCA",
        "luad": "TCGA-LUAD",     # Lung Adenocarcinoma
        "lusc": "TCGA-LUSC",     # Lung Squamous Cell
        "lung": "TCGA-LUAD",     # Default to adenocarcinoma
        "coad": "TCGA-COAD",     # Colon Adenocarcinoma
        "colon": "TCGA-COAD",
        "read": "TCGA-READ",     # Rectal Adenocarcinoma
        "prad": "TCGA-PRAD",     # Prostate Adenocarcinoma
        "prostate": "TCGA-PRAD",
        "stad": "TCGA-STAD",     # Stomach Adenocarcinoma
        "stomach": "TCGA-STAD",
        "hnsc": "TCGA-HNSC",     # Head and Neck Squamous Cell
        "kirc": "TCGA-KIRC",     # Kidney Renal Clear Cell
        "kidney": "TCGA-KIRC",
        "lihc": "TCGA-LIHC",     # Liver Hepatocellular
        "liver": "TCGA-LIHC",
        "blca": "TCGA-BLCA",     # Bladder Urothelial
        "bladder": "TCGA-BLCA",
        "skcm": "TCGA-SKCM",     # Skin Cutaneous Melanoma
        "melanoma": "TCGA-SKCM",
        "skin": "TCGA-SKCM",
        "thca": "TCGA-THCA",     # Thyroid Carcinoma
        "thyroid": "TCGA-THCA",
        "ov": "TCGA-OV",         # Ovarian Serous
        "ovarian": "TCGA-OV",
        "paad": "TCGA-PAAD",     # Pancreatic Adenocarcinoma
        "pancreatic": "TCGA-PAAD",
        "pancreas": "TCGA-PAAD",
        "ucec": "TCGA-UCEC",     # Uterine Corpus Endometrial
        "uterine": "TCGA-UCEC",
    }
    
    # Data type mapping
    DATA_TYPES = {
        "methylation": "Methylation Beta Value",
        "methyl": "Methylation Beta Value",
        "dna methylation": "Methylation Beta Value",
        "450k": "Methylation Beta Value",
        "epic": "Methylation Beta Value",
        "wgbs": "Methylation Beta Value",      # Whole genome bisulfite = methylation
        "rrbs": "Methylation Beta Value",      # Reduced rep bisulfite = methylation
        "bisulfite": "Methylation Beta Value",
        "expression": "Gene Expression Quantification",
        "gene expression": "Gene Expression Quantification",
        "rnaseq": "Gene Expression Quantification",
        "rna-seq": "Gene Expression Quantification",
        "rna seq": "Gene Expression Quantification",
        "aligned reads": "Aligned Reads",
        "bam": "Aligned Reads",
        "wgs": "Aligned Reads",
        "wes": "Aligned Reads",
        "whole genome": "Aligned Reads",
        "whole exome": "Aligned Reads",
        "exome": "Aligned Reads",
        "cnv": "Copy Number Variation",
        "copy number": "Copy Number Variation",
        "snv": "Simple Nucleotide Variation",
        "mutation": "Simple Nucleotide Variation",
        "mutations": "Simple Nucleotide Variation",
    }
    
    def __init__(self, timeout: float = 30.0):
        """Initialize the GDC adapter."""
        super().__init__(timeout=timeout)
        self.name = "GDC/TCGA"
    
    def search(
        self,
        query: SearchQuery,
        max_results: int = 50,
        **kwargs,
    ) -> List[DatasetInfo]:
        """
        Search GDC for datasets.
        
        Args:
            query: Search query with parameters
            max_results: Maximum number of results
            
        Returns:
            List of search results
        """
        # Check circuit breaker
        circuit = get_circuit_breaker(GDC_CIRCUIT_NAME, GDC_CIRCUIT_CONFIG)
        if not circuit.can_execute():
            logger.warning(f"GDC circuit breaker is OPEN - returning empty results")
            return []
        
        results = []
        
        try:
            # Detect project from keywords
            project_ids = self._detect_projects(query)
            data_type = self._detect_data_type(query)
            
            if project_ids:
                # Search for specific projects
                for project_id in project_ids[:3]:  # Limit to 3 projects
                    project_results = self._search_project_files(
                        project_id, data_type, max_results // len(project_ids)
                    )
                    results.extend(project_results)
            else:
                # General search
                results = self._general_search(query, data_type, max_results)
            
            # Record success
            circuit.record_success()
            
        except Exception as e:
            # Record failure
            circuit.record_failure()
            logger.error(f"GDC search failed: {e}")
            return []
        
        return results[:max_results]
    
    def _detect_projects(self, query: SearchQuery) -> List[str]:
        """Detect TCGA projects from query keywords."""
        projects = set()
        
        keywords = query.keywords or []
        if query.tissue:
            keywords.append(query.tissue)
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Check for TCGA-* pattern
            if keyword_lower.startswith("tcga-"):
                projects.add(keyword.upper())
                continue
            
            # Check cancer type mapping
            for key, project_id in self.CANCER_TYPES.items():
                if key in keyword_lower:
                    projects.add(project_id)
                    break
        
        return list(projects)
    
    def _detect_data_type(self, query: SearchQuery) -> Optional[str]:
        """Detect GDC data type from query."""
        # Check assay_type first
        if query.assay_type:
            assay_lower = query.assay_type.lower()
            for key, data_type in self.DATA_TYPES.items():
                if key in assay_lower:
                    return data_type
        
        # Check keywords
        for keyword in (query.keywords or []):
            keyword_lower = keyword.lower()
            for key, data_type in self.DATA_TYPES.items():
                if key in keyword_lower:
                    return data_type
        
        return None
    
    def _search_project_files(
        self,
        project_id: str,
        data_type: Optional[str],
        max_results: int,
    ) -> List[DatasetInfo]:
        """Search for files in a specific project."""
        results = []
        
        # Build filters
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "=",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": project_id,
                    },
                },
            ],
        }
        
        if data_type:
            filters["content"].append({
                "op": "=",
                "content": {
                    "field": "data_type",
                    "value": data_type,
                },
            })
        
        try:
            import json
            
            params = {
                "filters": json.dumps(filters),
                "fields": ",".join([
                    "file_id",
                    "file_name",
                    "data_type",
                    "data_category",
                    "experimental_strategy",
                    "file_size",
                    "access",
                    "cases.case_id",
                    "cases.primary_site",
                    "cases.disease_type",
                    "cases.project.project_id",
                    "cases.samples.sample_type",
                    "cases.samples.tissue_type",
                    "cases.samples.tumor_descriptor",
                ]),
                "size": max_results,
                "format": "json",
            }
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.BASE_URL}/files", params=params)
                response.raise_for_status()
                data = response.json()
            
            hits = data.get("data", {}).get("hits", [])
            
            for hit in hits:
                result = self._parse_file_hit(hit, project_id)
                if result:
                    results.append(result)
        
        except Exception as e:
            logger.warning(f"GDC project search failed: {e}")
        
        return results
    
    def _general_search(
        self,
        query: SearchQuery,
        data_type: Optional[str],
        max_results: int,
    ) -> List[DatasetInfo]:
        """General project search."""
        results = []
        
        try:
            # Search projects endpoint
            params = {
                "fields": ",".join([
                    "project_id",
                    "name",
                    "primary_site",
                    "disease_type",
                    "summary.case_count",
                    "summary.file_count",
                    "summary.data_categories.data_category",
                    "summary.data_categories.case_count",
                    "summary.data_categories.file_count",
                ]),
                "size": max_results,
                "format": "json",
            }
            
            # Filter by project prefix
            if query.organism and "human" in query.organism.lower():
                params["filters"] = '{"op":"like","content":{"field":"project_id","value":"TCGA%"}}'
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.BASE_URL}/projects", params=params)
                response.raise_for_status()
                data = response.json()
            
            hits = data.get("data", {}).get("hits", [])
            
            for hit in hits:
                result = self._parse_project_hit(hit, data_type)
                if result:
                    results.append(result)
        
        except Exception as e:
            logger.warning(f"GDC general search failed: {e}")
        
        return results
    
    def _parse_file_hit(
        self,
        hit: Dict[str, Any],
        project_id: str,
    ) -> Optional[DatasetInfo]:
        """Parse a file hit into DatasetInfo."""
        try:
            file_id = hit.get("file_id", "")
            file_name = hit.get("file_name", "")
            data_type = hit.get("data_type", "Unknown")
            data_category = hit.get("data_category", "")
            experimental_strategy = hit.get("experimental_strategy", "")
            file_size = hit.get("file_size", 0)
            access = hit.get("access", "controlled")
            
            # Get case/sample info
            cases = hit.get("cases", [{}])
            case_info = cases[0] if cases else {}
            
            primary_site = case_info.get("primary_site", "Unknown")
            disease_type = case_info.get("disease_type", "Unknown")
            
            # Format file size
            size_mb = file_size / (1024 * 1024) if file_size else 0
            size_str = f"{size_mb:.1f} MB" if size_mb < 1024 else f"{size_mb/1024:.1f} GB"
            
            title = f"{project_id}: {data_type} - {file_name[:40]}"
            
            description = (
                f"{experimental_strategy or data_category} data for {disease_type} ({primary_site}). "
                f"File size: {size_str}. Access: {access}."
            )
            
            # Determine organism (all TCGA is human)
            organism = "Homo sapiens"
            
            # Determine assay type
            assay_type = experimental_strategy or data_category or data_type
            
            return DatasetInfo(
                id=file_id,
                title=title,
                description=description,
                source=DataSource.GDC,
                web_url=f"{self.PORTAL_URL}/files/{file_id}",
                organism=organism,
                assay_type=assay_type,
                tissue=primary_site,
                metadata={
                    "file_name": file_name,
                    "data_type": data_type,
                    "data_category": data_category,
                    "experimental_strategy": experimental_strategy,
                    "file_size": file_size,
                    "file_size_formatted": size_str,
                    "access": access,
                    "project_id": project_id,
                    "primary_site": primary_site,
                    "disease_type": disease_type,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to parse GDC file hit: {e}")
            return None
    
    def _parse_project_hit(
        self,
        hit: Dict[str, Any],
        data_type: Optional[str],
    ) -> Optional[DatasetInfo]:
        """Parse a project hit into DatasetInfo."""
        try:
            project_id = hit.get("project_id", "")
            name = hit.get("name", project_id)
            primary_site = hit.get("primary_site", ["Unknown"])
            if isinstance(primary_site, list):
                primary_site = ", ".join(primary_site[:3])
            
            disease_type = hit.get("disease_type", ["Unknown"])
            if isinstance(disease_type, list):
                disease_type = ", ".join(disease_type[:3])
            
            summary = hit.get("summary", {})
            case_count = summary.get("case_count", 0)
            file_count = summary.get("file_count", 0)
            
            # Check if requested data type is available
            data_categories = summary.get("data_categories", [])
            has_requested_type = True
            if data_type:
                has_requested_type = any(
                    cat.get("data_category", "").lower() in data_type.lower()
                    or data_type.lower() in cat.get("data_category", "").lower()
                    for cat in data_categories
                )
            
            title = f"{project_id}: {name}"
            
            description = (
                f"{disease_type} study with {case_count} cases and {file_count} files. "
                f"Primary site: {primary_site}."
            )
            
            return DatasetInfo(
                id=project_id,
                title=title,
                description=description,
                source=DataSource.GDC,
                web_url=f"{self.PORTAL_URL}/projects/{project_id}",
                organism="Homo sapiens",
                assay_type="Multi-omics",
                tissue=primary_site,
                file_count=file_count,
                metadata={
                    "project_id": project_id,
                    "project_name": name,
                    "primary_site": primary_site,
                    "disease_type": disease_type,
                    "case_count": case_count,
                    "file_count": file_count,
                    "data_categories": [
                        cat.get("data_category")
                        for cat in data_categories
                    ],
                    "has_requested_type": has_requested_type,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to parse GDC project hit: {e}")
            return None
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetInfo]:
        """Get detailed information about a specific GDC file or project."""
        # Check circuit breaker
        circuit = get_circuit_breaker(GDC_CIRCUIT_NAME, GDC_CIRCUIT_CONFIG)
        if not circuit.can_execute():
            logger.warning(f"GDC circuit breaker is OPEN - cannot fetch dataset {dataset_id}")
            return None
        
        # Try as file first
        try:
            import json
            params = {
                "filters": json.dumps({
                    "op": "=",
                    "content": {"field": "file_id", "value": dataset_id}
                }),
                "format": "json",
            }
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.BASE_URL}/files", params=params)
                response.raise_for_status()
                data = response.json()
            
            hits = data.get("data", {}).get("hits", [])
            if hits:
                circuit.record_success()
                return self._parse_file_hit(hits[0], dataset_id.split("-")[0] if "-" in dataset_id else "TCGA")
        except Exception as e:
            logger.debug(f"File lookup failed: {e}")
        
        # Try as project
        try:
            params = {"format": "json"}
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.BASE_URL}/projects/{dataset_id}", params=params)
                response.raise_for_status()
                data = response.json()
            
            if data.get("data"):
                circuit.record_success()
                return self._parse_project_hit(data["data"], None)
        except Exception as e:
            logger.debug(f"Project lookup failed: {e}")
        
        # Record failure if both attempts failed
        circuit.record_failure()
        return None
    
    def get_download_urls(self, dataset_id: str) -> List[DownloadURL]:
        """Get download URLs for a GDC file."""
        return [
            DownloadURL(
                url=f"{self.BASE_URL}/data/{dataset_id}",
                filename=dataset_id,
            )
        ]
    
    def get_download_url(self, file_id: str) -> str:
        """Get download URL for a file."""
        return f"{self.BASE_URL}/data/{file_id}"
    
    def get_manifest(self, file_ids: List[str]) -> str:
        """
        Get a manifest for downloading multiple files.
        
        Returns the manifest content as a string.
        """
        try:
            import json
            
            payload = {
                "ids": file_ids,
                "format": "TSV",
            }
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.BASE_URL}/manifest",
                    json=payload,
                )
                response.raise_for_status()
                return response.text
        except Exception as e:
            logger.error(f"Failed to get GDC manifest: {e}")
            raise


def search_gdc(
    project: str = None,
    cancer_type: str = None,
    data_type: str = None,
    keywords: List[str] = None,
    max_results: int = 20,
) -> List[DatasetInfo]:
    """
    Convenience function for searching GDC.
    
    Args:
        project: Specific project ID (e.g., TCGA-GBM)
        cancer_type: Cancer type name (e.g., "brain", "breast")
        data_type: Data type (e.g., "methylation", "expression")
        keywords: Additional search keywords
        max_results: Maximum results to return
        
    Returns:
        List of search results
    """
    # Build query
    query_keywords = list(keywords or [])
    
    if project:
        query_keywords.append(project)
    
    if cancer_type:
        query_keywords.append(cancer_type)
    
    if data_type:
        query_keywords.append(data_type)
    
    query = SearchQuery(
        organism="Homo sapiens",
        assay_type=data_type,
        keywords=query_keywords,
    )
    
    adapter = GDCAdapter()
    return adapter.search(query, max_results=max_results)


def get_gdc_circuit_status() -> Dict[str, Any]:
    """Get the current status of the GDC circuit breaker.
    
    Returns:
        Dictionary with circuit breaker state and statistics
    """
    circuit = get_circuit_breaker(GDC_CIRCUIT_NAME, GDC_CIRCUIT_CONFIG)
    return {
        "name": GDC_CIRCUIT_NAME,
        "state": circuit.state.value,
        "is_open": circuit.state == CircuitState.OPEN,
        "failure_count": circuit.failure_count,
        "success_count": circuit.success_count,
        "can_execute": circuit.can_execute(),
    }
