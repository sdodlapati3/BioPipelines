"""
SRA/ENA Downloader Module
==========================

Download datasets from NCBI SRA and EBI ENA using pysradb.

Key features:
- Search SRA metadata
- Download FASTQ files from ENA (faster than SRA)
- Convert SRA to FASTQ using fasterq-dump
- Metadata extraction and filtering
"""

from pathlib import Path
from typing import List, Optional, Union
import subprocess
import logging

logger = logging.getLogger(__name__)


class SRADownloader:
    """
    Download datasets from SRA/ENA
    
    Uses ENA FTP for faster downloads when possible, falls back to SRA tools.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if tools are available
        self.has_pysradb = self._check_tool("pysradb")
        self.has_sratools = self._check_tool("fasterq-dump")
    
    def _check_tool(self, tool_name: str) -> bool:
        """Check if a command-line tool is available"""
        try:
            subprocess.run([tool_name, "--version"], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def download(
        self,
        accession: str,
        dataset_type: str,
        use_aspera: bool = False
    ) -> List[Path]:
        """
        Download SRA dataset
        
        Parameters
        ----------
        accession : str
            SRA accession (SRR, ERR, DRR)
        dataset_type : str
            Dataset type for organizing files
        use_aspera : bool
            Use Aspera for faster download
            
        Returns
        -------
        List[Path]
            Paths to downloaded FASTQ files
        """
        logger.info(f"Downloading {accession} from SRA/ENA")
        
        output_subdir = self.output_dir / dataset_type
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Try ENA first (faster)
        try:
            return self._download_from_ena(accession, output_subdir)
        except Exception as e:
            logger.warning(f"ENA download failed: {e}, trying SRA tools")
            
            if self.has_sratools:
                return self._download_from_sra(accession, output_subdir)
            else:
                raise RuntimeError(
                    "SRA tools not available. Install with: "
                    "conda install -c bioconda sra-tools"
                )
    
    def _download_from_ena(
        self,
        accession: str,
        output_dir: Path
    ) -> List[Path]:
        """Download from ENA using FTP"""
        try:
            import pysradb
        except ImportError:
            raise ImportError(
                "pysradb not installed. Install with: pip install pysradb"
            )
        
        db = pysradb.SRAweb()
        
        # Get metadata including FTP links
        metadata = db.sra_metadata(accession, detailed=True)
        
        if metadata.empty:
            raise ValueError(f"No metadata found for {accession}")
        
        # Extract ENA FTP URLs
        fastq_ftp = metadata['fastq_ftp'].iloc[0]
        
        if pd.isna(fastq_ftp):
            raise ValueError(f"No FTP URLs found for {accession}")
        
        # Parse FTP URLs (semicolon-separated for paired-end)
        ftp_urls = [f"ftp://{url}" for url in fastq_ftp.split(';')]
        
        downloaded_files = []
        for i, url in enumerate(ftp_urls, 1):
            filename = Path(url).name
            output_path = output_dir / filename
            
            if output_path.exists():
                logger.info(f"File already exists: {output_path}")
                downloaded_files.append(output_path)
                continue
            
            logger.info(f"Downloading from ENA: {url}")
            
            # Download using wget or curl
            cmd = ["wget", "-q", "--show-progress", "-O", str(output_path), url]
            try:
                subprocess.run(cmd, check=True)
                downloaded_files.append(output_path)
            except subprocess.CalledProcessError:
                # Try curl as fallback
                cmd = ["curl", "-L", "-o", str(output_path), url]
                subprocess.run(cmd, check=True)
                downloaded_files.append(output_path)
        
        logger.info(f"Downloaded {len(downloaded_files)} files from ENA")
        return downloaded_files
    
    def _download_from_sra(
        self,
        accession: str,
        output_dir: Path
    ) -> List[Path]:
        """Download and convert from SRA using sra-tools"""
        logger.info(f"Using fasterq-dump to download {accession}")
        
        cmd = [
            "fasterq-dump",
            accession,
            "--outdir", str(output_dir),
            "--split-files",  # Split into R1/R2 for paired-end
            "--progress",
            "--threads", "4"
        ]
        
        subprocess.run(cmd, check=True)
        
        # Find downloaded files
        downloaded = list(output_dir.glob(f"{accession}*.fastq"))
        
        # Compress files
        compressed = []
        for fastq in downloaded:
            gzipped = fastq.with_suffix(fastq.suffix + '.gz')
            logger.info(f"Compressing {fastq}")
            subprocess.run(["gzip", str(fastq)], check=True)
            compressed.append(gzipped)
        
        return compressed


def search_sra(
    query: str,
    organism: str = "human",
    dataset_type: Optional[str] = None,
    limit: int = 10
) -> List:
    """
    Search SRA database
    
    Parameters
    ----------
    query : str
        Search query
    organism : str
        Organism name
    dataset_type : str, optional
        Filter by experiment type
    limit : int
        Maximum results
        
    Returns
    -------
    List[DatasetMetadata]
        Matching datasets
    """
    try:
        import pysradb
        import pandas as pd
    except ImportError:
        raise ImportError("pysradb not installed. Install with: pip install pysradb")
    
    db = pysradb.SRAweb()
    
    # Search SRA
    search_term = f"{query} AND {organism}"
    
    if dataset_type:
        # Map dataset types to library strategies
        strategy_map = {
            "rna_seq": "RNA-Seq",
            "chip_seq": "ChIP-Seq",
            "atac_seq": "ATAC-seq",
            "dna_seq": "WGS OR WXS"
        }
        strategy = strategy_map.get(dataset_type, dataset_type.upper())
        search_term += f" AND {strategy}"
    
    logger.info(f"Searching SRA: {search_term}")
    
    # Use esearch to get IDs, then fetch metadata
    results = db.search(search_term, max_results=limit)
    
    if results.empty:
        logger.warning("No results found")
        return []
    
    # Get detailed metadata for top results
    accessions = results['run_accession'].head(limit).tolist()
    
    metadata_list = []
    for acc in accessions:
        try:
            meta = db.sra_metadata(acc, detailed=True)
            metadata_list.append(meta.iloc[0].to_dict())
        except Exception as e:
            logger.warning(f"Failed to get metadata for {acc}: {e}")
    
    return metadata_list


def get_sra_metadata(accession: str) -> dict:
    """
    Get detailed metadata for an SRA accession
    
    Parameters
    ----------
    accession : str
        SRA accession
        
    Returns
    -------
    dict
        Metadata dictionary
    """
    try:
        import pysradb
    except ImportError:
        raise ImportError("pysradb not installed")
    
    db = pysradb.SRAweb()
    metadata = db.sra_metadata(accession, detailed=True)
    
    if metadata.empty:
        raise ValueError(f"No metadata found for {accession}")
    
    return metadata.iloc[0].to_dict()
