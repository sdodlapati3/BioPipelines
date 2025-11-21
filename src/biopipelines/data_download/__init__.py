"""
BioPipelines Data Download Module
==================================

Unified interface for downloading genomics datasets from public repositories.

Supported databases:
- ENCODE (https://www.encodeproject.org/)
- GEO/SRA (NCBI)
- ENA (European Nucleotide Archive)
- 1000 Genomes Project
- UCSC Genome Browser

Key packages used:
- pysradb: SRA/GEO metadata and download
- ffq: Fast metadata fetching for SRA
- requests: HTTP downloads from ENCODE/UCSC
- urllib: FTP downloads from ENA

Example usage:
--------------
```python
from biopipelines.data_download import DataDownloader

# Initialize downloader
downloader = DataDownloader(output_dir="data/raw")

# Download from SRA
downloader.download_sra("SRR891268", dataset_type="atac_seq")

# Download from ENCODE
downloader.download_encode("ENCFF001NQP", dataset_type="chip_seq")

# Search and download from GEO
datasets = downloader.search_geo("H3K4me3", organism="human", limit=5)
downloader.download_geo_series("GSE12345")
```
"""

from pathlib import Path
from typing import List, Dict, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum

__version__ = "0.1.0"
__all__ = [
    "DataDownloader",
    "DatasetType",
    "DataSource",
    "search_encode",
    "search_sra",
    "search_geo",
    "download_from_url",
    "MethylationDownloader",
    "HiCDownloader",
    "search_methylation_datasets",
    "search_hic_datasets"
]


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetType(Enum):
    """Supported dataset types"""
    DNA_SEQ = "dna_seq"
    RNA_SEQ = "rna_seq"
    CHIP_SEQ = "chip_seq"
    ATAC_SEQ = "atac_seq"
    WGS = "wgs"
    EXOME = "exome"
    METHYLATION = "methylation"
    WGBS = "wgbs"
    RRBS = "rrbs"
    HIC = "hic"
    MICRO_C = "micro_c"


class DataSource(Enum):
    """Supported data sources"""
    ENCODE = "encode"
    SRA = "sra"
    GEO = "geo"
    ENA = "ena"
    GENOMES_1000 = "1000genomes"
    UCSC = "ucsc"
    FOURD = "4dn"
    ROADMAP = "roadmap"
    AIDENLAB = "aidenlab"


@dataclass
class DatasetMetadata:
    """
    Metadata for a genomics dataset
    """
    accession: str
    title: str
    organism: str
    dataset_type: DatasetType
    source: DataSource
    file_urls: List[str]
    file_sizes: List[int]
    file_types: List[str]
    sample_count: int
    experiment_type: str
    library_layout: str  # single-end or paired-end
    description: Optional[str] = None
    publication: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'accession': self.accession,
            'title': self.title,
            'organism': self.organism,
            'dataset_type': self.dataset_type.value,
            'source': self.source.value,
            'file_urls': self.file_urls,
            'file_sizes': self.file_sizes,
            'file_types': self.file_types,
            'sample_count': self.sample_count,
            'experiment_type': self.experiment_type,
            'library_layout': self.library_layout,
            'description': self.description,
            'publication': self.publication
        }


class DataDownloader:
    """
    Main class for downloading genomics datasets from various sources
    
    Parameters
    ----------
    output_dir : Path or str
        Base directory for downloaded files
    verify_downloads : bool
        Verify file integrity after download (default: True)
    retry_attempts : int
        Number of retry attempts for failed downloads (default: 3)
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "data/raw",
        verify_downloads: bool = True,
        retry_attempts: int = 3
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verify_downloads = verify_downloads
        self.retry_attempts = retry_attempts
        
        logger.info(f"Initialized DataDownloader with output_dir: {self.output_dir}")
    
    def download_sra(
        self,
        accession: str,
        dataset_type: Union[str, DatasetType],
        use_aspera: bool = False
    ) -> List[Path]:
        """
        Download from SRA/ENA using accession ID
        
        Parameters
        ----------
        accession : str
            SRA accession (SRR, ERR, DRR)
        dataset_type : str or DatasetType
            Type of dataset (rna_seq, chip_seq, etc.)
        use_aspera : bool
            Use Aspera for faster download (requires ascp)
            
        Returns
        -------
        List[Path]
            Paths to downloaded files
        """
        from .sra_downloader import SRADownloader
        
        downloader = SRADownloader(self.output_dir)
        return downloader.download(accession, dataset_type, use_aspera)
    
    def download_encode(
        self,
        file_id: str,
        dataset_type: Union[str, DatasetType]
    ) -> Path:
        """
        Download from ENCODE using file ID
        
        Parameters
        ----------
        file_id : str
            ENCODE file ID (e.g., ENCFF001NQP)
        dataset_type : str or DatasetType
            Type of dataset
            
        Returns
        -------
        Path
            Path to downloaded file
        """
        from .encode_downloader import ENCODEDownloader
        
        downloader = ENCODEDownloader(self.output_dir)
        return downloader.download(file_id, dataset_type)
    
    def download_geo_series(
        self,
        geo_id: str,
        dataset_type: Union[str, DatasetType]
    ) -> List[Path]:
        """
        Download all samples from a GEO series
        
        Parameters
        ----------
        geo_id : str
            GEO series ID (e.g., GSE12345)
        dataset_type : str or DatasetType
            Type of dataset
            
        Returns
        -------
        List[Path]
            Paths to downloaded files
        """
        from .geo_downloader import GEODownloader
        
        downloader = GEODownloader(self.output_dir)
        return downloader.download_series(geo_id, dataset_type)
    
    def download_methylation(
        self,
        experiment_id: str,
        source: str = "encode"
    ) -> List[Path]:
        """
        Download DNA methylation dataset
        
        Parameters
        ----------
        experiment_id : str
            Experiment ID
        source : str
            Data source (encode, roadmap)
            
        Returns
        -------
        List[Path]
            Paths to downloaded files
        """
        from .methylation_downloader import MethylationDownloader
        
        downloader = MethylationDownloader(self.output_dir)
        
        if source.lower() == "encode":
            return downloader.download_encode_experiment(experiment_id)
        elif source.lower() == "roadmap":
            return downloader.download_roadmap_sample(experiment_id)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def download_hic(
        self,
        experiment_id: str,
        source: str = "4dn",
        file_format: str = "pairs"
    ) -> List[Path]:
        """
        Download Hi-C dataset
        
        Parameters
        ----------
        experiment_id : str
            Experiment ID
        source : str
            Data source (4dn, encode, aidenlab)
        file_format : str
            Preferred format (fastq, pairs, hic, cool)
            
        Returns
        -------
        List[Path]
            Paths to downloaded files
        """
        from .hic_downloader import HiCDownloader
        
        downloader = HiCDownloader(self.output_dir)
        
        if source.lower() == "4dn":
            return downloader.download_4dn_experiment(experiment_id, file_format)
        elif source.lower() == "encode":
            return downloader.download_encode_hic(experiment_id, file_format)
        elif source.lower() == "aidenlab":
            return [downloader.download_aidenlab_dataset(experiment_id)]
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def search_datasets(
        self,
        query: str,
        source: Union[str, DataSource],
        dataset_type: Optional[Union[str, DatasetType]] = None,
        organism: str = "human",
        limit: int = 10
    ) -> List[DatasetMetadata]:
        """
        Search for datasets matching criteria
        
        Parameters
        ----------
        query : str
            Search query (e.g., "H3K4me3", "RNA-seq liver")
        source : str or DataSource
            Data source to search
        dataset_type : str or DatasetType, optional
            Filter by dataset type
        organism : str
            Organism name (human, mouse, etc.)
        limit : int
            Maximum number of results
            
        Returns
        -------
        List[DatasetMetadata]
            List of matching datasets
        """
        if isinstance(source, str):
            source = DataSource(source)
        
        if source == DataSource.SRA:
            from .sra_downloader import search_sra
            return search_sra(query, organism, dataset_type, limit)
        elif source == DataSource.ENCODE:
            from .encode_downloader import search_encode
            return search_encode(query, organism, dataset_type, limit)
        elif source == DataSource.GEO:
            from .geo_downloader import search_geo
            return search_geo(query, organism, dataset_type, limit)
        else:
            raise ValueError(f"Search not implemented for source: {source}")


# Convenience functions
def download_from_url(
    url: str,
    output_path: Path,
    retry_attempts: int = 3,
    timeout: int = 300
) -> Path:
    """
    Download file from URL with retry logic
    
    Parameters
    ----------
    url : str
        URL to download from
    output_path : Path
        Output file path
    retry_attempts : int
        Number of retry attempts
    timeout : int
        Timeout in seconds
        
    Returns
    -------
    Path
        Path to downloaded file
    """
    import requests
    from time import sleep
    
    for attempt in range(retry_attempts):
        try:
            logger.info(f"Downloading from {url} (attempt {attempt + 1}/{retry_attempts})")
            
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Progress reporting
                            progress = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024 * 10) == 0:  # Every 10MB
                                logger.info(f"Progress: {progress:.1f}%")
            
            logger.info(f"Successfully downloaded to {output_path}")
            return output_path
            
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < retry_attempts - 1:
                sleep(5 * (attempt + 1))  # Exponential backoff
            else:
                raise
    
    raise RuntimeError(f"Failed to download after {retry_attempts} attempts")
