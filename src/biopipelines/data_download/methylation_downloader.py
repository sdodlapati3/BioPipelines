"""
DNA Methylation Data Downloader
================================

Download whole-genome bisulfite sequencing (WGBS) and reduced representation
bisulfite sequencing (RRBS) datasets from public repositories.

Supports:
- ENCODE (human/mouse WGBS datasets)
- GEO/SRA (published methylation studies)
- BLUEPRINT Epigenome (human hematopoietic cells)
- Roadmap Epigenomics (diverse human tissues)

Author: BioPipelines Team
"""

import logging
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class MethylationDataset:
    """Metadata for methylation dataset"""
    accession: str
    title: str
    organism: str
    tissue: str
    assay_type: str  # WGBS, RRBS, RRBS-seq, etc.
    file_urls: List[str]
    file_names: List[str]
    file_sizes: List[int]
    read_length: int
    library_layout: str  # single, paired
    description: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'accession': self.accession,
            'title': self.title,
            'organism': self.organism,
            'tissue': self.tissue,
            'assay_type': self.assay_type,
            'file_urls': self.file_urls,
            'file_names': self.file_names,
            'file_sizes': self.file_sizes,
            'read_length': self.read_length,
            'library_layout': self.library_layout,
            'description': self.description
        }


class MethylationDownloader:
    """
    Download DNA methylation datasets from public repositories
    
    Parameters
    ----------
    output_dir : Path
        Base directory for downloads
    organism : str
        Target organism (human, mouse)
    """
    
    # ENCODE search endpoints
    ENCODE_BASE = "https://www.encodeproject.org"
    ENCODE_SEARCH = f"{ENCODE_BASE}/search/"
    
    # Roadmap Epigenomics FTP
    ROADMAP_FTP = "ftp://ftp.ncbi.nlm.nih.gov/pub/geo/DATA/roadmapepigenomics/"
    
    def __init__(self, output_dir: Path, organism: str = "human"):
        self.output_dir = Path(output_dir)
        self.organism = organism
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized MethylationDownloader for {organism}")
    
    def search_encode_wgbs(
        self,
        tissue: Optional[str] = None,
        limit: int = 10
    ) -> List[MethylationDataset]:
        """
        Search ENCODE for WGBS datasets
        
        Parameters
        ----------
        tissue : str, optional
            Tissue/cell type (e.g., "brain", "liver", "blood")
        limit : int
            Maximum results to return
            
        Returns
        -------
        List[MethylationDataset]
            Available datasets
        """
        logger.info(f"Searching ENCODE for WGBS datasets (tissue={tissue})")
        
        # Build search query
        params = {
            'type': 'Experiment',
            'assay_title': 'whole-genome shotgun bisulfite sequencing',
            'status': 'released',
            'replicates.library.biosample.organism.scientific_name': 
                'Homo sapiens' if self.organism == 'human' else 'Mus musculus',
            'limit': limit,
            'format': 'json'
        }
        
        if tissue:
            params['biosample_ontology.term_name'] = tissue
        
        try:
            response = requests.get(self.ENCODE_SEARCH, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            datasets = []
            for exp in data.get('@graph', []):
                dataset = self._parse_encode_experiment(exp)
                if dataset:
                    datasets.append(dataset)
            
            logger.info(f"Found {len(datasets)} WGBS datasets")
            return datasets
            
        except Exception as e:
            logger.error(f"ENCODE search failed: {e}")
            return []
    
    def download_encode_experiment(
        self,
        experiment_id: str,
        file_type: str = "fastq"
    ) -> List[Path]:
        """
        Download all files from an ENCODE experiment
        
        Parameters
        ----------
        experiment_id : str
            ENCODE experiment ID (e.g., ENCSR000AAA)
        file_type : str
            File type to download (fastq, bam, bed, bigWig)
            
        Returns
        -------
        List[Path]
            Downloaded file paths
        """
        logger.info(f"Downloading ENCODE experiment {experiment_id}")
        
        # Get experiment metadata
        exp_url = f"{self.ENCODE_BASE}/experiments/{experiment_id}/?format=json"
        
        try:
            response = requests.get(exp_url, timeout=30)
            response.raise_for_status()
            exp_data = response.json()
            
            # Find files matching criteria
            files_to_download = []
            for file_obj in exp_data.get('files', []):
                if (file_obj.get('file_format') == file_type and
                    file_obj.get('status') == 'released'):
                    files_to_download.append({
                        'accession': file_obj['accession'],
                        'url': f"{self.ENCODE_BASE}{file_obj['href']}",
                        'size': file_obj.get('file_size', 0)
                    })
            
            logger.info(f"Found {len(files_to_download)} {file_type} files")
            
            # Download files
            downloaded_paths = []
            for file_info in files_to_download:
                output_path = self.output_dir / f"{file_info['accession']}.{file_type}.gz"
                
                if output_path.exists():
                    logger.info(f"File already exists: {output_path}")
                    downloaded_paths.append(output_path)
                    continue
                
                self._download_file(file_info['url'], output_path, file_info['size'])
                downloaded_paths.append(output_path)
            
            return downloaded_paths
            
        except Exception as e:
            logger.error(f"Failed to download experiment: {e}")
            raise
    
    def download_roadmap_sample(
        self,
        sample_id: str,
        data_type: str = "WGBS"
    ) -> List[Path]:
        """
        Download from Roadmap Epigenomics Project
        
        Parameters
        ----------
        sample_id : str
            Roadmap sample ID (e.g., E001, E002)
        data_type : str
            Data type (WGBS, RRBS)
            
        Returns
        -------
        List[Path]
            Downloaded file paths
        """
        logger.info(f"Downloading Roadmap sample {sample_id}")
        
        # Roadmap uses GEO, need to map to GSM IDs
        # This is a simplified implementation
        # In practice, would query GEO API for sample metadata
        
        logger.warning("Roadmap download requires manual GEO accession mapping")
        logger.info("Use search_geo() to find GSM IDs for Roadmap samples")
        
        return []
    
    def _parse_encode_experiment(self, exp_data: Dict) -> Optional[MethylationDataset]:
        """Parse ENCODE experiment JSON into MethylationDataset"""
        try:
            accession = exp_data.get('accession', 'unknown')
            
            # Get biosample info
            biosample = exp_data.get('biosample_ontology', {})
            tissue = biosample.get('term_name', 'unknown')
            
            # Get file info
            files = []
            file_names = []
            file_sizes = []
            
            for file_obj in exp_data.get('files', []):
                if (file_obj.get('file_format') == 'fastq' and
                    file_obj.get('status') == 'released'):
                    files.append(f"{self.ENCODE_BASE}{file_obj['href']}")
                    file_names.append(file_obj['accession'])
                    file_sizes.append(file_obj.get('file_size', 0))
            
            if not files:
                return None
            
            # Determine library layout
            replicates = exp_data.get('replicates', [])
            layout = 'paired' if any(
                rep.get('library', {}).get('paired_ended', False)
                for rep in replicates
            ) else 'single'
            
            return MethylationDataset(
                accession=accession,
                title=exp_data.get('description', accession),
                organism=self.organism,
                tissue=tissue,
                assay_type='WGBS',
                file_urls=files,
                file_names=file_names,
                file_sizes=file_sizes,
                read_length=exp_data.get('read_length', 150),
                library_layout=layout,
                description=exp_data.get('description')
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse experiment: {e}")
            return None
    
    def _download_file(self, url: str, output_path: Path, expected_size: int):
        """Download file with progress tracking"""
        from ..data_download import download_from_url
        
        logger.info(f"Downloading {url} -> {output_path}")
        logger.info(f"Expected size: {expected_size / 1024 / 1024:.1f} MB")
        
        download_from_url(url, output_path)
        
        # Verify size
        actual_size = output_path.stat().st_size
        if abs(actual_size - expected_size) > 1024:  # Allow 1KB difference
            logger.warning(
                f"Size mismatch: expected {expected_size}, got {actual_size}"
            )


def search_methylation_datasets(
    query: str,
    organism: str = "human",
    tissue: Optional[str] = None,
    source: str = "encode",
    limit: int = 10
) -> List[MethylationDataset]:
    """
    Search for DNA methylation datasets
    
    Parameters
    ----------
    query : str
        Search keywords
    organism : str
        Target organism
    tissue : str, optional
        Tissue/cell type filter
    source : str
        Data source (encode, geo)
    limit : int
        Maximum results
        
    Returns
    -------
    List[MethylationDataset]
        Matching datasets
    """
    downloader = MethylationDownloader(Path("data/raw"), organism)
    
    if source.lower() == "encode":
        return downloader.search_encode_wgbs(tissue, limit)
    elif source.lower() == "geo":
        # Use GEO downloader
        from .sra_downloader import search_sra
        logger.info("Searching GEO/SRA for methylation datasets")
        # Would integrate with GEO API here
        return []
    else:
        raise ValueError(f"Unknown source: {source}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    downloader = MethylationDownloader(Path("data/raw/methylation"))
    
    # Search for brain WGBS datasets
    datasets = downloader.search_encode_wgbs(tissue="brain", limit=5)
    
    for ds in datasets:
        print(f"Found: {ds.accession} - {ds.title}")
        print(f"  Tissue: {ds.tissue}")
        print(f"  Files: {len(ds.file_urls)}")
        print(f"  Layout: {ds.library_layout}")
