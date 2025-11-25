"""
Data Downloader Module
======================

Download reference data, indexes, and sample datasets.

Supports:
- Reference genomes (Ensembl, UCSC, NCBI)
- Gene annotations (GTF/GFF)
- Pre-built indexes (STAR, HISAT2, BWA, etc.)
- Sample datasets for testing

Example:
    from workflow_composer.data import DataDownloader
    
    downloader = DataDownloader()
    genome = downloader.get_genome("human", "GRCh38")
    annotation = downloader.get_annotation("human", "GRCh38")
"""

import os
import logging
import hashlib
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from urllib.parse import urlparse
import json

logger = logging.getLogger(__name__)


# Known reference sources
REFERENCE_SOURCES = {
    "ensembl": {
        "genome_url": "http://ftp.ensembl.org/pub/release-{release}/fasta/{species}/dna/{species}.{assembly}.dna.primary_assembly.fa.gz",
        "gtf_url": "http://ftp.ensembl.org/pub/release-{release}/gtf/{species}/{species_cap}.{assembly}.{release}.gtf.gz",
        "releases": [108, 109, 110, 111, 112],
        "species": {
            "human": {"name": "homo_sapiens", "cap": "Homo_sapiens", "assembly": "GRCh38"},
            "mouse": {"name": "mus_musculus", "cap": "Mus_musculus", "assembly": "GRCm39"},
            "zebrafish": {"name": "danio_rerio", "cap": "Danio_rerio", "assembly": "GRCz11"},
            "fly": {"name": "drosophila_melanogaster", "cap": "Drosophila_melanogaster", "assembly": "BDGP6.46"},
            "worm": {"name": "caenorhabditis_elegans", "cap": "Caenorhabditis_elegans", "assembly": "WBcel235"},
            "yeast": {"name": "saccharomyces_cerevisiae", "cap": "Saccharomyces_cerevisiae", "assembly": "R64-1-1"},
        }
    },
    "gencode": {
        "genome_url": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_{species}/release_{release}/{species_short}.{assembly}.genome.fa.gz",
        "gtf_url": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_{species}/release_{release}/gencode.v{release}.annotation.gtf.gz",
        "species": {
            "human": {"name": "human", "short": "GRCh38", "assembly": "GRCh38.primary_assembly"},
            "mouse": {"name": "mouse", "short": "GRCm39", "assembly": "GRCm39.primary_assembly"}
        }
    },
    "ucsc": {
        "genome_url": "https://hgdownload.soe.ucsc.edu/goldenPath/{assembly}/bigZips/{assembly}.fa.gz",
        "species": {
            "human": {"assembly": "hg38"},
            "human_hg19": {"assembly": "hg19"},
            "mouse": {"assembly": "mm39"},
            "mouse_mm10": {"assembly": "mm10"}
        }
    }
}

# Pre-built index URLs
INDEX_SOURCES = {
    "star": {
        "human_GRCh38": "s3://ngi-igenomes/igenomes/Homo_sapiens/Ensembl/GRCh38/Sequence/STARIndex/",
        "mouse_GRCm39": "s3://ngi-igenomes/igenomes/Mus_musculus/Ensembl/GRCm39/Sequence/STARIndex/"
    },
    "salmon": {
        "human_GRCh38": "https://refgenie.databio.org/v3/assets/archive/hg38/salmon_partial_sa_index",
        "mouse_GRCm39": "https://refgenie.databio.org/v3/assets/archive/mm10/salmon_partial_sa_index"
    },
    "hisat2": {
        "human_GRCh38": "https://genome-idx.s3.amazonaws.com/hisat/grch38_snptran.tar.gz",
        "mouse_GRCm38": "https://genome-idx.s3.amazonaws.com/hisat/grcm38_snptran.tar.gz"
    },
    "bowtie2": {
        "human_GRCh38": "https://genome-idx.s3.amazonaws.com/bt/GRCh38_noalt_as.zip",
        "mouse_GRCm38": "https://genome-idx.s3.amazonaws.com/bt/GRCm38_noalt_as.zip"
    },
    "bwa": {
        "human_GRCh38": "s3://ngi-igenomes/igenomes/Homo_sapiens/Ensembl/GRCh38/Sequence/BWAIndex/",
        "mouse_GRCm39": "s3://ngi-igenomes/igenomes/Mus_musculus/Ensembl/GRCm39/Sequence/BWAIndex/"
    }
}

# Sample datasets for testing
SAMPLE_DATASETS = {
    "rnaseq_test": {
        "description": "Small RNA-seq test dataset (E. coli)",
        "url": "https://github.com/nf-core/test-datasets/raw/rnaseq/testdata/",
        "files": ["SRR3191542_1.fastq.gz", "SRR3191542_2.fastq.gz"]
    },
    "chipseq_test": {
        "description": "Small ChIP-seq test dataset",
        "url": "https://github.com/nf-core/test-datasets/raw/chipseq/testdata/",
        "files": ["SRR1822153_1.fastq.gz", "SRR1822153_2.fastq.gz"]
    },
    "atacseq_test": {
        "description": "Small ATAC-seq test dataset",
        "url": "https://github.com/nf-core/test-datasets/raw/atacseq/testdata/",
        "files": ["SRR5204809_Hep_1.mLb.clN.sorted.bam"]
    },
    "scrna_test": {
        "description": "Small 10x scRNA-seq test dataset",
        "url": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_1k_v3/",
        "files": ["pbmc_1k_v3_fastqs.tar"]
    }
}


@dataclass
class DownloadedFile:
    """Represents a downloaded file."""
    path: Path
    source_url: str
    checksum: Optional[str] = None
    size: int = 0
    
    def verify(self) -> bool:
        """Verify file integrity."""
        if not self.path.exists():
            return False
        if self.checksum:
            actual = hashlib.md5(self.path.read_bytes()).hexdigest()
            return actual == self.checksum
        return True


@dataclass  
class Reference:
    """Reference data bundle."""
    organism: str
    assembly: str
    genome_fasta: Optional[Path] = None
    annotation_gtf: Optional[Path] = None
    indexes: Dict[str, Path] = field(default_factory=dict)


class DataDownloader:
    """
    Download reference data and sample datasets.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        prefer_source: str = "ensembl"
    ):
        """
        Initialize the data downloader.
        
        Args:
            cache_dir: Directory to cache downloads
            prefer_source: Preferred reference source (ensembl, gencode, ucsc)
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".biopipelines" / "references"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prefer_source = prefer_source
        
        # Track downloads
        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict:
        """Load download manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {"downloads": {}}
    
    def _save_manifest(self) -> None:
        """Save download manifest."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2, default=str)
    
    def get_genome(
        self,
        organism: str,
        assembly: Optional[str] = None,
        release: Optional[int] = None,
        source: Optional[str] = None
    ) -> Path:
        """
        Get genome FASTA file, downloading if necessary.
        
        Args:
            organism: Organism name (human, mouse, etc.)
            assembly: Assembly name (GRCh38, GRCm39, etc.)
            release: Ensembl release number
            source: Reference source (ensembl, gencode, ucsc)
            
        Returns:
            Path to genome FASTA file
        """
        source = source or self.prefer_source
        organism = organism.lower()
        
        # Get species info
        source_info = REFERENCE_SOURCES.get(source, {})
        species_info = source_info.get("species", {}).get(organism)
        
        if not species_info:
            raise ValueError(f"Unknown organism '{organism}' for source '{source}'")
        
        assembly = assembly or species_info.get("assembly")
        release = release or source_info.get("releases", [111])[-1]
        
        # Check cache
        cache_key = f"genome_{source}_{organism}_{assembly}"
        cached = self._get_cached(cache_key)
        if cached and cached.exists():
            logger.info(f"Using cached genome: {cached}")
            return cached
        
        # Build URL
        url = source_info["genome_url"].format(
            species=species_info.get("name", organism),
            species_cap=species_info.get("cap", organism.capitalize()),
            assembly=assembly,
            release=release
        )
        
        # Download
        output_path = self.cache_dir / source / organism / f"{assembly}.genome.fa.gz"
        self._download(url, output_path)
        
        # Cache
        self._set_cached(cache_key, output_path)
        
        return output_path
    
    def get_annotation(
        self,
        organism: str,
        assembly: Optional[str] = None,
        release: Optional[int] = None,
        source: Optional[str] = None
    ) -> Path:
        """
        Get annotation GTF file, downloading if necessary.
        
        Args:
            organism: Organism name
            assembly: Assembly name
            release: Ensembl release number
            source: Reference source
            
        Returns:
            Path to GTF file
        """
        source = source or self.prefer_source
        organism = organism.lower()
        
        source_info = REFERENCE_SOURCES.get(source, {})
        species_info = source_info.get("species", {}).get(organism)
        
        if not species_info:
            raise ValueError(f"Unknown organism '{organism}' for source '{source}'")
        
        assembly = assembly or species_info.get("assembly")
        release = release or source_info.get("releases", [111])[-1]
        
        # Check cache
        cache_key = f"gtf_{source}_{organism}_{assembly}"
        cached = self._get_cached(cache_key)
        if cached and cached.exists():
            logger.info(f"Using cached annotation: {cached}")
            return cached
        
        # Build URL
        url = source_info.get("gtf_url", "").format(
            species=species_info.get("name", organism),
            species_cap=species_info.get("cap", organism.capitalize()),
            assembly=assembly,
            release=release
        )
        
        if not url:
            raise ValueError(f"GTF not available from source '{source}'")
        
        # Download
        output_path = self.cache_dir / source / organism / f"{assembly}.annotation.gtf.gz"
        self._download(url, output_path)
        
        # Cache
        self._set_cached(cache_key, output_path)
        
        return output_path
    
    def get_index(
        self,
        aligner: str,
        organism: str,
        assembly: str
    ) -> Path:
        """
        Get pre-built aligner index.
        
        Args:
            aligner: Aligner name (star, hisat2, bwa, salmon, bowtie2)
            organism: Organism name
            assembly: Assembly name
            
        Returns:
            Path to index directory/file
        """
        aligner = aligner.lower()
        key = f"{organism}_{assembly}"
        
        if aligner not in INDEX_SOURCES:
            raise ValueError(f"Unknown aligner: {aligner}")
        
        if key not in INDEX_SOURCES[aligner]:
            available = list(INDEX_SOURCES[aligner].keys())
            raise ValueError(f"Index not available for {key}. Available: {available}")
        
        # Check cache
        cache_key = f"index_{aligner}_{key}"
        cached = self._get_cached(cache_key)
        if cached and Path(cached).exists():
            logger.info(f"Using cached index: {cached}")
            return Path(cached)
        
        url = INDEX_SOURCES[aligner][key]
        output_path = self.cache_dir / "indexes" / aligner / key
        
        # Download (handle different URL schemes)
        if url.startswith("s3://"):
            self._download_s3(url, output_path)
        else:
            self._download(url, output_path)
        
        self._set_cached(cache_key, output_path)
        return output_path
    
    def get_sample_dataset(
        self,
        dataset: str,
        output_dir: Optional[str] = None
    ) -> Path:
        """
        Download a sample dataset for testing.
        
        Args:
            dataset: Dataset name (rnaseq_test, chipseq_test, etc.)
            output_dir: Output directory
            
        Returns:
            Path to downloaded data directory
        """
        if dataset not in SAMPLE_DATASETS:
            available = list(SAMPLE_DATASETS.keys())
            raise ValueError(f"Unknown dataset: {dataset}. Available: {available}")
        
        info = SAMPLE_DATASETS[dataset]
        
        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = self.cache_dir / "samples" / dataset
        
        out_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading sample dataset: {info['description']}")
        
        for filename in info["files"]:
            url = info["url"] + filename
            self._download(url, out_path / filename)
        
        return out_path
    
    def get_reference(
        self,
        organism: str,
        assembly: Optional[str] = None,
        include_indexes: Optional[List[str]] = None
    ) -> Reference:
        """
        Get a complete reference bundle.
        
        Args:
            organism: Organism name
            assembly: Assembly name (optional)
            include_indexes: List of aligners to include indexes for
            
        Returns:
            Reference object with all paths
        """
        genome = self.get_genome(organism, assembly)
        annotation = self.get_annotation(organism, assembly)
        
        # Determine actual assembly
        source_info = REFERENCE_SOURCES.get(self.prefer_source, {})
        species_info = source_info.get("species", {}).get(organism.lower(), {})
        actual_assembly = assembly or species_info.get("assembly", "unknown")
        
        ref = Reference(
            organism=organism,
            assembly=actual_assembly,
            genome_fasta=genome,
            annotation_gtf=annotation
        )
        
        # Download indexes
        if include_indexes:
            for aligner in include_indexes:
                try:
                    idx = self.get_index(aligner, organism, actual_assembly)
                    ref.indexes[aligner] = idx
                except Exception as e:
                    logger.warning(f"Could not get {aligner} index: {e}")
        
        return ref
    
    def list_available(self) -> Dict[str, Any]:
        """List available references and datasets."""
        return {
            "reference_sources": list(REFERENCE_SOURCES.keys()),
            "organisms": {
                source: list(info.get("species", {}).keys())
                for source, info in REFERENCE_SOURCES.items()
            },
            "index_aligners": list(INDEX_SOURCES.keys()),
            "sample_datasets": {
                name: info["description"]
                for name, info in SAMPLE_DATASETS.items()
            }
        }
    
    def _download(self, url: str, output_path: Path) -> None:
        """Download a file using wget or curl."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.exists():
            logger.info(f"File exists: {output_path}")
            return
        
        logger.info(f"Downloading: {url}")
        logger.info(f"To: {output_path}")
        
        # Try wget first, fall back to curl
        try:
            subprocess.run(
                ["wget", "-q", "-O", str(output_path), url],
                check=True,
                capture_output=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            subprocess.run(
                ["curl", "-sL", "-o", str(output_path), url],
                check=True
            )
        
        logger.info(f"Downloaded: {output_path.stat().st_size / 1e6:.1f} MB")
    
    def _download_s3(self, url: str, output_path: Path) -> None:
        """Download from S3 using aws cli."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading from S3: {url}")
        
        try:
            subprocess.run(
                ["aws", "s3", "sync", url, str(output_path), "--no-sign-request"],
                check=True
            )
        except FileNotFoundError:
            raise RuntimeError("AWS CLI not found. Install with: pip install awscli")
    
    def _get_cached(self, key: str) -> Optional[Path]:
        """Get cached path for a key."""
        cached = self.manifest.get("downloads", {}).get(key)
        return Path(cached) if cached else None
    
    def _set_cached(self, key: str, path: Path) -> None:
        """Set cached path for a key."""
        if "downloads" not in self.manifest:
            self.manifest["downloads"] = {}
        self.manifest["downloads"][key] = str(path)
        self._save_manifest()
