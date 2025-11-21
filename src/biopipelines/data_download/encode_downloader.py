"""
ENCODE Downloader Module
=========================

Download datasets from ENCODE portal using their REST API.

Documentation: https://www.encodeproject.org/help/rest-api/
"""

from pathlib import Path
from typing import List, Dict, Optional
import requests
import logging

logger = logging.getLogger(__name__)


class ENCODEDownloader:
    """
    Download datasets from ENCODE portal
    """
    
    BASE_URL = "https://www.encodeproject.org"
    HEADERS = {'accept': 'application/json'}
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download(
        self,
        file_id: str,
        dataset_type: str
    ) -> Path:
        """
        Download a file from ENCODE
        
        Parameters
        ----------
        file_id : str
            ENCODE file ID (e.g., ENCFF001NQP)
        dataset_type : str
            Dataset type for organizing files
            
        Returns
        -------
        Path
            Path to downloaded file
        """
        logger.info(f"Downloading {file_id} from ENCODE")
        
        # Get file metadata
        metadata = self.get_file_metadata(file_id)
        
        if not metadata:
            raise ValueError(f"File {file_id} not found in ENCODE")
        
        # Check file status
        status = metadata.get('status')
        if status not in ['released', 'in progress']:
            logger.warning(f"File status is '{status}', may not be available")
        
        # Get download URL
        download_url = f"{self.BASE_URL}{metadata['href']}"
        
        # Determine output filename (extract just the basename, not full path)
        from pathlib import Path as PathLib
        submitted_name = metadata.get('submitted_file_name', '')
        if submitted_name:
            filename = PathLib(submitted_name).name  # Extract just the filename
        else:
            filename = f"{file_id}.fastq.gz"
        
        output_subdir = self.output_dir / dataset_type
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_path = output_subdir / filename
        
        if output_path.exists():
            logger.info(f"File already exists: {output_path}")
            return output_path
        
        # Download file
        logger.info(f"Downloading from: {download_url}")
        
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Progress reporting every 100MB
                    if downloaded % (1024 * 1024 * 100) == 0 and total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"Progress: {progress:.1f}%")
        
        logger.info(f"Downloaded to: {output_path}")
        return output_path
    
    def get_file_metadata(self, file_id: str) -> Dict:
        """
        Get metadata for a file
        
        Parameters
        ----------
        file_id : str
            ENCODE file ID
            
        Returns
        -------
        dict
            File metadata
        """
        url = f"{self.BASE_URL}/files/{file_id}/"
        
        response = requests.get(url, headers=self.HEADERS)
        
        if response.status_code == 404:
            logger.error(f"File {file_id} not found")
            return {}
        
        response.raise_for_status()
        return response.json()
    
    def download_experiment(
        self,
        experiment_id: str,
        dataset_type: str,
        file_format: str = "fastq"
    ) -> List[Path]:
        """
        Download all files from an experiment
        
        Parameters
        ----------
        experiment_id : str
            ENCODE experiment ID (e.g., ENCSR000AED)
        dataset_type : str
            Dataset type
        file_format : str
            File format to download (fastq, bam, etc.)
            
        Returns
        -------
        List[Path]
            Downloaded file paths
        """
        logger.info(f"Downloading experiment {experiment_id}")
        
        # Get experiment metadata
        url = f"{self.BASE_URL}/experiments/{experiment_id}/"
        response = requests.get(url, headers=self.HEADERS)
        response.raise_for_status()
        
        experiment = response.json()
        
        # Get files of requested format
        files = [
            f for f in experiment.get('files', [])
            if f.get('file_format') == file_format and
            f.get('status') == 'released'
        ]
        
        logger.info(f"Found {len(files)} {file_format} files")
        
        downloaded = []
        for file_info in files:
            file_id = file_info['accession']
            try:
                path = self.download(file_id, dataset_type)
                downloaded.append(path)
            except Exception as e:
                logger.error(f"Failed to download {file_id}: {e}")
        
        return downloaded


def search_encode(
    query: str,
    organism: str = "human",
    dataset_type: Optional[str] = None,
    limit: int = 10
) -> List[Dict]:
    """
    Search ENCODE database
    
    Parameters
    ----------
    query : str
        Search query
    organism : str
        Organism name
    dataset_type : str, optional
        Filter by assay type
    limit : int
        Maximum results
        
    Returns
    -------
    List[dict]
        Matching datasets metadata
    """
    base_url = "https://www.encodeproject.org"
    
    # Map dataset types to ENCODE assay terms
    assay_map = {
        "rna_seq": "RNA-seq",
        "chip_seq": "ChIP-seq",
        "atac_seq": "ATAC-seq",
        "dna_seq": "whole-genome shotgun bisulfite sequencing"
    }
    
    # Build search URL
    search_url = f"{base_url}/search/"
    params = {
        'type': 'Experiment',
        'status': 'released',
        'limit': limit,
        'searchTerm': query
    }
    
    if organism == "human":
        params['replicates.library.biosample.donor.organism.scientific_name'] = 'Homo sapiens'
    elif organism == "mouse":
        params['replicates.library.biosample.donor.organism.scientific_name'] = 'Mus musculus'
    
    if dataset_type and dataset_type in assay_map:
        params['assay_title'] = assay_map[dataset_type]
    
    headers = {'accept': 'application/json'}
    
    logger.info(f"Searching ENCODE with params: {params}")
    
    response = requests.get(search_url, params=params, headers=headers)
    response.raise_for_status()
    
    results = response.json()
    
    experiments = results.get('@graph', [])
    
    logger.info(f"Found {len(experiments)} experiments")
    
    # Format results
    formatted = []
    for exp in experiments:
        formatted.append({
            'accession': exp.get('accession'),
            'assay_title': exp.get('assay_title'),
            'description': exp.get('description'),
            'organism': exp.get('organism', [{}])[0].get('scientific_name'),
            'biosample_summary': exp.get('biosample_summary'),
            'files_count': len(exp.get('files', [])),
            'url': f"{base_url}{exp.get('@id')}"
        })
    
    return formatted


def get_encode_experiment_metadata(experiment_id: str) -> Dict:
    """
    Get detailed metadata for an ENCODE experiment
    
    Parameters
    ----------
    experiment_id : str
        ENCODE experiment ID
        
    Returns
    -------
    dict
        Experiment metadata
    """
    url = f"https://www.encodeproject.org/experiments/{experiment_id}/"
    headers = {'accept': 'application/json'}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return response.json()
