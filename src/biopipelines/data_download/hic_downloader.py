"""
Hi-C Data Downloader
====================

Download Hi-C (chromosome conformation capture) datasets from public repositories.

Supports:
- 4DN Data Portal (primary Hi-C repository)
- ENCODE (Hi-C experiments)
- GEO/SRA (published Hi-C studies)
- Aiden Lab (in situ Hi-C datasets)

Handles various Hi-C protocols:
- in situ Hi-C
- Dilution Hi-C
- DNase Hi-C
- Micro-C (ultra-high resolution)

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
class HiCDataset:
    """Metadata for Hi-C dataset"""
    accession: str
    title: str
    organism: str
    cell_type: str
    protocol: str  # in-situ, dilution, DNase, Micro-C
    enzyme: str  # MboI, DpnII, HindIII, etc.
    file_urls: List[str]
    file_names: List[str]
    file_types: List[str]  # fastq, hic, cool, mcool
    file_sizes: List[int]
    read_length: int
    sequencing_depth: Optional[int] = None  # Total read pairs
    resolution: Optional[str] = None  # e.g., "5kb", "10kb"
    description: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'accession': self.accession,
            'title': self.title,
            'organism': self.organism,
            'cell_type': self.cell_type,
            'protocol': self.protocol,
            'enzyme': self.enzyme,
            'file_urls': self.file_urls,
            'file_names': self.file_names,
            'file_types': self.file_types,
            'file_sizes': self.file_sizes,
            'read_length': self.read_length,
            'sequencing_depth': self.sequencing_depth,
            'resolution': self.resolution,
            'description': self.description
        }


class HiCDownloader:
    """
    Download Hi-C datasets from public repositories
    
    Parameters
    ----------
    output_dir : Path
        Base directory for downloads
    organism : str
        Target organism (human, mouse, fly)
    prefer_processed : bool
        Prefer processed files (.hic, .cool) over raw FASTQ
    """
    
    # Data portal endpoints
    FOURD_BASE = "https://data.4dnucleome.org"
    ENCODE_BASE = "https://www.encodeproject.org"
    AIDENLAB_BASE = "https://www.aidenlab.org/data.html"
    
    def __init__(
        self,
        output_dir: Path,
        organism: str = "human",
        prefer_processed: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.organism = organism
        self.prefer_processed = prefer_processed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized HiCDownloader for {organism}")
    
    def search_4dn(
        self,
        cell_type: Optional[str] = None,
        protocol: Optional[str] = None,
        limit: int = 10
    ) -> List[HiCDataset]:
        """
        Search 4DN Data Portal for Hi-C datasets
        
        Parameters
        ----------
        cell_type : str, optional
            Cell type/tissue (e.g., "GM12878", "IMR90")
        protocol : str, optional
            Hi-C protocol type
        limit : int
            Maximum results
            
        Returns
        -------
        List[HiCDataset]
            Available datasets
        """
        logger.info(f"Searching 4DN for Hi-C datasets (cell_type={cell_type})")
        
        # 4DN API endpoint
        search_url = f"{self.FOURD_BASE}/search/"
        
        params = {
            'type': 'ExperimentHiC',
            'status': 'released',
            'limit': limit,
            'format': 'json'
        }
        
        if self.organism == 'human':
            params['organism.name'] = 'human'
        elif self.organism == 'mouse':
            params['organism.name'] = 'mouse'
        
        if cell_type:
            params['biosample.biosource_summary'] = cell_type
        
        try:
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            datasets = []
            for exp in data.get('@graph', []):
                dataset = self._parse_4dn_experiment(exp)
                if dataset:
                    datasets.append(dataset)
            
            logger.info(f"Found {len(datasets)} Hi-C datasets on 4DN")
            return datasets
            
        except Exception as e:
            logger.error(f"4DN search failed: {e}")
            logger.info("4DN may require authentication for some datasets")
            return []
    
    def search_encode_hic(
        self,
        cell_line: Optional[str] = None,
        limit: int = 10
    ) -> List[HiCDataset]:
        """
        Search ENCODE for Hi-C datasets
        
        Parameters
        ----------
        cell_line : str, optional
            Cell line name (e.g., "GM12878", "K562")
        limit : int
            Maximum results
            
        Returns
        -------
        List[HiCDataset]
            Available datasets
        """
        logger.info(f"Searching ENCODE for Hi-C datasets (cell_line={cell_line})")
        
        params = {
            'type': 'Experiment',
            'assay_title': 'Hi-C',
            'status': 'released',
            'replicates.library.biosample.organism.scientific_name':
                'Homo sapiens' if self.organism == 'human' else 'Mus musculus',
            'limit': limit,
            'format': 'json'
        }
        
        if cell_line:
            params['biosample_ontology.term_name'] = cell_line
        
        try:
            response = requests.get(
                f"{self.ENCODE_BASE}/search/",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            datasets = []
            for exp in data.get('@graph', []):
                dataset = self._parse_encode_hic(exp)
                if dataset:
                    datasets.append(dataset)
            
            logger.info(f"Found {len(datasets)} Hi-C datasets on ENCODE")
            return datasets
            
        except Exception as e:
            logger.error(f"ENCODE search failed: {e}")
            return []
    
    def download_4dn_experiment(
        self,
        experiment_id: str,
        file_format: str = "pairs"
    ) -> List[Path]:
        """
        Download files from 4DN experiment
        
        Parameters
        ----------
        experiment_id : str
            4DN experiment ID
        file_format : str
            Preferred format: fastq, pairs, hic, cool, mcool
            
        Returns
        -------
        List[Path]
            Downloaded file paths
        """
        logger.info(f"Downloading 4DN experiment {experiment_id}")
        
        exp_url = f"{self.FOURD_BASE}/experiments-hi-c/{experiment_id}/?format=json"
        
        try:
            response = requests.get(exp_url, timeout=30)
            response.raise_for_status()
            exp_data = response.json()
            
            # Extract file information
            files_to_download = []
            
            for file_ref in exp_data.get('processed_files', []):
                # Get detailed file info
                file_url = f"{self.FOURD_BASE}{file_ref}?format=json"
                file_resp = requests.get(file_url, timeout=30)
                file_data = file_resp.json()
                
                if file_data.get('file_format', {}).get('display_title') == file_format:
                    files_to_download.append({
                        'accession': file_data['accession'],
                        'url': f"{self.FOURD_BASE}{file_data.get('href', '')}",
                        'size': file_data.get('file_size', 0),
                        'format': file_format
                    })
            
            logger.info(f"Found {len(files_to_download)} {file_format} files")
            
            downloaded_paths = []
            for file_info in files_to_download:
                output_path = self.output_dir / f"{file_info['accession']}.{file_format}.gz"
                
                if output_path.exists():
                    logger.info(f"File already exists: {output_path}")
                    downloaded_paths.append(output_path)
                    continue
                
                self._download_file(file_info['url'], output_path, file_info['size'])
                downloaded_paths.append(output_path)
            
            return downloaded_paths
            
        except Exception as e:
            logger.error(f"Failed to download 4DN experiment: {e}")
            raise
    
    def download_encode_hic(
        self,
        experiment_id: str,
        file_type: str = "fastq"
    ) -> List[Path]:
        """
        Download Hi-C files from ENCODE
        
        Parameters
        ----------
        experiment_id : str
            ENCODE experiment ID
        file_type : str
            File type (fastq, bam, hic)
            
        Returns
        -------
        List[Path]
            Downloaded file paths
        """
        logger.info(f"Downloading ENCODE Hi-C experiment {experiment_id}")
        
        exp_url = f"{self.ENCODE_BASE}/experiments/{experiment_id}/?format=json"
        
        try:
            response = requests.get(exp_url, timeout=30)
            response.raise_for_status()
            exp_data = response.json()
            
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
            logger.error(f"Failed to download ENCODE Hi-C: {e}")
            raise
    
    def download_aidenlab_dataset(
        self,
        dataset_name: str,
        resolution: str = "5kb"
    ) -> Path:
        """
        Download pre-processed Hi-C data from Aiden Lab
        
        Parameters
        ----------
        dataset_name : str
            Dataset name (e.g., "GM12878_combined")
        resolution : str
            Matrix resolution (1kb, 5kb, 10kb, etc.)
            
        Returns
        -------
        Path
            Downloaded file path
        """
        logger.info(f"Downloading Aiden Lab dataset: {dataset_name}")
        
        # Aiden Lab hosts data on GEO and their website
        # This is a template - actual URLs would need to be mapped
        
        base_url = "https://hicfiles.s3.amazonaws.com/hiseq"
        
        # Construct likely URL (this is simplified)
        file_url = f"{base_url}/{dataset_name}/{dataset_name}_{resolution}.hic"
        
        output_path = self.output_dir / f"{dataset_name}_{resolution}.hic"
        
        try:
            self._download_file(file_url, output_path, 0)
            return output_path
        except Exception as e:
            logger.error(f"Failed to download Aiden Lab dataset: {e}")
            logger.info("Visit https://aidenlab.org/data.html for available datasets")
            raise
    
    def _parse_4dn_experiment(self, exp_data: Dict) -> Optional[HiCDataset]:
        """Parse 4DN experiment JSON"""
        try:
            accession = exp_data.get('accession', 'unknown')
            
            biosample = exp_data.get('biosample', {})
            cell_type = biosample.get('biosource_summary', 'unknown')
            
            # Get protocol info
            protocol_info = exp_data.get('experiment_type', {})
            protocol = protocol_info.get('display_title', 'in situ Hi-C')
            
            # Get enzyme
            digestion = exp_data.get('digestion_enzyme', {})
            enzyme = digestion.get('name', 'unknown')
            
            files = []
            file_names = []
            file_types = []
            file_sizes = []
            
            # Process files based on preference
            for file_ref in exp_data.get('processed_files', [])[:5]:  # Limit for metadata
                files.append(f"{self.FOURD_BASE}{file_ref}")
                file_names.append(file_ref.split('/')[-2])  # Extract accession
                file_types.append('processed')
                file_sizes.append(0)  # Would need separate API call
            
            if not files:
                return None
            
            return HiCDataset(
                accession=accession,
                title=exp_data.get('description', accession),
                organism=self.organism,
                cell_type=cell_type,
                protocol=protocol,
                enzyme=enzyme,
                file_urls=files,
                file_names=file_names,
                file_types=file_types,
                file_sizes=file_sizes,
                read_length=exp_data.get('read_length', 150),
                description=exp_data.get('description')
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse 4DN experiment: {e}")
            return None
    
    def _parse_encode_hic(self, exp_data: Dict) -> Optional[HiCDataset]:
        """Parse ENCODE Hi-C experiment JSON"""
        try:
            accession = exp_data.get('accession', 'unknown')
            
            biosample = exp_data.get('biosample_ontology', {})
            cell_type = biosample.get('term_name', 'unknown')
            
            files = []
            file_names = []
            file_types = []
            file_sizes = []
            
            for file_obj in exp_data.get('files', [])[:5]:
                if file_obj.get('status') == 'released':
                    files.append(f"{self.ENCODE_BASE}{file_obj['href']}")
                    file_names.append(file_obj['accession'])
                    file_types.append(file_obj.get('file_format', 'unknown'))
                    file_sizes.append(file_obj.get('file_size', 0))
            
            if not files:
                return None
            
            return HiCDataset(
                accession=accession,
                title=exp_data.get('description', accession),
                organism=self.organism,
                cell_type=cell_type,
                protocol='in situ Hi-C',
                enzyme='MboI',  # Common default
                file_urls=files,
                file_names=file_names,
                file_types=file_types,
                file_sizes=file_sizes,
                read_length=exp_data.get('read_length', 150),
                description=exp_data.get('description')
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse ENCODE Hi-C: {e}")
            return None
    
    def _download_file(self, url: str, output_path: Path, expected_size: int):
        """Download file with progress tracking"""
        from . import download_from_url
        
        logger.info(f"Downloading {url} -> {output_path}")
        if expected_size > 0:
            logger.info(f"Expected size: {expected_size / 1024 / 1024:.1f} MB")
        
        download_from_url(url, output_path)
        
        actual_size = output_path.stat().st_size
        logger.info(f"Downloaded {actual_size / 1024 / 1024:.1f} MB")


def search_hic_datasets(
    query: str,
    organism: str = "human",
    cell_type: Optional[str] = None,
    source: str = "4dn",
    limit: int = 10
) -> List[HiCDataset]:
    """
    Search for Hi-C datasets
    
    Parameters
    ----------
    query : str
        Search keywords
    organism : str
        Target organism
    cell_type : str, optional
        Cell type/tissue filter
    source : str
        Data source (4dn, encode, aidenlab)
    limit : int
        Maximum results
        
    Returns
    -------
    List[HiCDataset]
        Matching datasets
    """
    downloader = HiCDownloader(Path("data/raw"), organism)
    
    if source.lower() == "4dn":
        return downloader.search_4dn(cell_type, limit=limit)
    elif source.lower() == "encode":
        return downloader.search_encode_hic(cell_type, limit)
    elif source.lower() == "aidenlab":
        logger.info("Aiden Lab datasets require manual selection")
        logger.info("Visit: https://aidenlab.org/data.html")
        return []
    else:
        raise ValueError(f"Unknown source: {source}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    downloader = HiCDownloader(Path("data/raw/hic"))
    
    # Search for GM12878 Hi-C datasets
    datasets = downloader.search_encode_hic(cell_line="GM12878", limit=5)
    
    for ds in datasets:
        print(f"Found: {ds.accession} - {ds.title}")
        print(f"  Cell type: {ds.cell_type}")
        print(f"  Protocol: {ds.protocol}")
        print(f"  Enzyme: {ds.enzyme}")
        print(f"  Files: {len(ds.file_urls)}")
