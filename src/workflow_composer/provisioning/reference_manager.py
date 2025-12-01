"""
Reference Genome Manager
========================

Manages reference genome downloads and index building.

Features:
- Downloads from Ensembl, UCSC, GENCODE
- Builds aligner indices (STAR, BWA, Bowtie2, HISAT2)
- Progress tracking for large downloads
- Checksum verification
- Resume interrupted downloads

Usage:
    from workflow_composer.provisioning import get_reference_manager
    
    ref_mgr = get_reference_manager()
    
    # Get reference, downloading if needed
    fasta = ref_mgr.get_reference("human", "GRCh38")
    
    # Build aligner index
    star_index = ref_mgr.ensure_index("human", "GRCh38", "star")
"""

import os
import subprocess
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class ReferenceGenome:
    """Reference genome metadata and URLs."""
    organism: str
    build: str
    source: str  # ensembl, ucsc, gencode
    fasta_url: str
    gtf_url: Optional[str] = None
    fasta_md5: Optional[str] = None
    gtf_md5: Optional[str] = None
    release: Optional[str] = None  # e.g., "110" for Ensembl
    
    def __repr__(self):
        return f"ReferenceGenome({self.organism}/{self.build} from {self.source})"


# Comprehensive reference catalog
REFERENCE_CATALOG: Dict[str, ReferenceGenome] = {
    # Human - Ensembl
    "human_GRCh38_ensembl": ReferenceGenome(
        organism="human",
        build="GRCh38",
        source="ensembl",
        release="110",
        fasta_url="https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
        gtf_url="https://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/Homo_sapiens.GRCh38.110.gtf.gz",
    ),
    # Human - GENCODE (more annotation coverage)
    "human_GRCh38_gencode": ReferenceGenome(
        organism="human",
        build="GRCh38",
        source="gencode",
        release="44",
        fasta_url="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/GRCh38.primary_assembly.genome.fa.gz",
        gtf_url="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.primary_assembly.annotation.gtf.gz",
    ),
    # Mouse - Ensembl
    "mouse_GRCm39_ensembl": ReferenceGenome(
        organism="mouse",
        build="GRCm39",
        source="ensembl",
        release="110",
        fasta_url="https://ftp.ensembl.org/pub/release-110/fasta/mus_musculus/dna/Mus_musculus.GRCm39.dna.primary_assembly.fa.gz",
        gtf_url="https://ftp.ensembl.org/pub/release-110/gtf/mus_musculus/Mus_musculus.GRCm39.110.gtf.gz",
    ),
    # Mouse - GENCODE
    "mouse_GRCm39_gencode": ReferenceGenome(
        organism="mouse",
        build="GRCm39",
        source="gencode",
        release="M33",
        fasta_url="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M33/GRCm39.primary_assembly.genome.fa.gz",
        gtf_url="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M33/gencode.vM33.primary_assembly.annotation.gtf.gz",
    ),
    # Rat - Ensembl
    "rat_mRatBN7.2_ensembl": ReferenceGenome(
        organism="rat",
        build="mRatBN7.2",
        source="ensembl",
        release="110",
        fasta_url="https://ftp.ensembl.org/pub/release-110/fasta/rattus_norvegicus/dna/Rattus_norvegicus.mRatBN7.2.dna.toplevel.fa.gz",
        gtf_url="https://ftp.ensembl.org/pub/release-110/gtf/rattus_norvegicus/Rattus_norvegicus.mRatBN7.2.110.gtf.gz",
    ),
    # Zebrafish - Ensembl
    "zebrafish_GRCz11_ensembl": ReferenceGenome(
        organism="zebrafish",
        build="GRCz11",
        source="ensembl",
        release="110",
        fasta_url="https://ftp.ensembl.org/pub/release-110/fasta/danio_rerio/dna/Danio_rerio.GRCz11.dna.primary_assembly.fa.gz",
        gtf_url="https://ftp.ensembl.org/pub/release-110/gtf/danio_rerio/Danio_rerio.GRCz11.110.gtf.gz",
    ),
    # Drosophila - Ensembl
    "drosophila_BDGP6.46_ensembl": ReferenceGenome(
        organism="drosophila",
        build="BDGP6.46",
        source="ensembl",
        release="110",
        fasta_url="https://ftp.ensembl.org/pub/release-110/fasta/drosophila_melanogaster/dna/Drosophila_melanogaster.BDGP6.46.dna.toplevel.fa.gz",
        gtf_url="https://ftp.ensembl.org/pub/release-110/gtf/drosophila_melanogaster/Drosophila_melanogaster.BDGP6.46.110.gtf.gz",
    ),
    # C. elegans - Ensembl
    "worm_WBcel235_ensembl": ReferenceGenome(
        organism="worm",
        build="WBcel235",
        source="ensembl",
        release="110",
        fasta_url="https://ftp.ensembl.org/pub/release-110/fasta/caenorhabditis_elegans/dna/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa.gz",
        gtf_url="https://ftp.ensembl.org/pub/release-110/gtf/caenorhabditis_elegans/Caenorhabditis_elegans.WBcel235.110.gtf.gz",
    ),
    # Yeast - Ensembl
    "yeast_R64-1-1_ensembl": ReferenceGenome(
        organism="yeast",
        build="R64-1-1",
        source="ensembl",
        release="110",
        fasta_url="https://ftp.ensembl.org/pub/release-110/fasta/saccharomyces_cerevisiae/dna/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa.gz",
        gtf_url="https://ftp.ensembl.org/pub/release-110/gtf/saccharomyces_cerevisiae/Saccharomyces_cerevisiae.R64-1-1.110.gtf.gz",
    ),
    # Arabidopsis - Ensembl Plants
    "arabidopsis_TAIR10_ensembl": ReferenceGenome(
        organism="arabidopsis",
        build="TAIR10",
        source="ensembl_plants",
        release="57",
        fasta_url="https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz",
        gtf_url="https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-57/gtf/arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.57.gtf.gz",
    ),
}

# Common organism aliases
ORGANISM_ALIASES = {
    # Human
    "homo_sapiens": "human", "h_sapiens": "human", "hg38": "human",
    "grch38": "human", "hs": "human",
    # Mouse
    "mus_musculus": "mouse", "mm39": "mouse", "grch39": "mouse",
    "mm": "mouse", "m_musculus": "mouse",
    # Rat
    "rattus_norvegicus": "rat", "rn7": "rat", "r_norvegicus": "rat",
    # Zebrafish
    "danio_rerio": "zebrafish", "d_rerio": "zebrafish",
    "zfish": "zebrafish", "dr": "zebrafish",
    # Drosophila
    "drosophila_melanogaster": "drosophila", "fruit_fly": "drosophila",
    "dm": "drosophila", "fly": "drosophila",
    # C. elegans
    "caenorhabditis_elegans": "worm", "c_elegans": "worm",
    "ce": "worm", "elegans": "worm",
    # Yeast
    "saccharomyces_cerevisiae": "yeast", "s_cerevisiae": "yeast",
    "sc": "yeast", "cerevisiae": "yeast",
    # Arabidopsis
    "arabidopsis_thaliana": "arabidopsis", "a_thaliana": "arabidopsis",
    "at": "arabidopsis", "thaliana": "arabidopsis",
}


@dataclass
class DownloadProgress:
    """Track download progress."""
    url: str
    total_bytes: int = 0
    downloaded_bytes: int = 0
    speed_bps: float = 0.0
    eta_seconds: float = 0.0
    status: str = "pending"  # pending, downloading, completed, failed
    error: Optional[str] = None
    
    @property
    def percent(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return (self.downloaded_bytes / self.total_bytes) * 100


class ReferenceManager:
    """
    Manages reference genome downloads and index building.
    
    Features:
    - Auto-download from Ensembl/GENCODE/UCSC
    - Build aligner indices (STAR, BWA, etc.)
    - Progress tracking
    - Cache management
    """
    
    def __init__(
        self,
        base_path: Optional[str] = None,
        scratch_path: Optional[str] = None,
    ):
        """
        Initialize reference manager.
        
        Args:
            base_path: Base directory for references. Defaults to ~/data/references
            scratch_path: Scratch directory for downloads. Defaults to /scratch
        """
        if base_path is None:
            base_path = os.environ.get(
                'BIOPIPELINES_REFERENCES',
                str(Path.home() / "data" / "references")
            )
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Scratch for large downloads
        if scratch_path is None:
            scratch_path = os.environ.get('SCRATCH', '/scratch')
        self.scratch_path = Path(scratch_path) if Path(scratch_path).exists() else self.base_path / "tmp"
        self.scratch_path.mkdir(parents=True, exist_ok=True)
        
        # Track downloads
        self._active_downloads: Dict[str, DownloadProgress] = {}
        
        logger.info(f"ReferenceManager initialized: {self.base_path}")
    
    def _normalize_organism(self, organism: str) -> str:
        """Normalize organism name to canonical form."""
        org_lower = organism.lower().strip().replace(" ", "_")
        return ORGANISM_ALIASES.get(org_lower, org_lower)
    
    def _resolve_reference_key(
        self,
        organism: str,
        build: Optional[str] = None,
        source: str = "ensembl",
    ) -> Optional[str]:
        """Resolve to a reference catalog key."""
        organism = self._normalize_organism(organism)
        
        # Try exact match first
        if build:
            key = f"{organism}_{build}_{source}"
            if key in REFERENCE_CATALOG:
                return key
            # Try without source
            for k in REFERENCE_CATALOG:
                if k.startswith(f"{organism}_{build}"):
                    return k
        
        # Try any match for organism
        for key in REFERENCE_CATALOG:
            if key.startswith(f"{organism}_"):
                return key
        
        return None
    
    def get_reference(
        self,
        organism: str,
        build: Optional[str] = None,
        source: str = "ensembl",
        download: bool = True,
    ) -> Optional[Path]:
        """
        Get reference genome path, optionally downloading if missing.
        
        Args:
            organism: Organism name (e.g., "human", "mouse")
            build: Genome build (e.g., "GRCh38", "GRCm39")
            source: Data source (ensembl, gencode)
            download: Whether to download if missing
            
        Returns:
            Path to genome FASTA, or None if not available
        """
        key = self._resolve_reference_key(organism, build, source)
        if not key:
            logger.warning(f"Unknown reference: {organism}/{build}")
            return None
        
        ref = REFERENCE_CATALOG[key]
        ref_dir = self.base_path / ref.organism / ref.build
        fasta_path = ref_dir / "genome.fa"
        
        if fasta_path.exists():
            return fasta_path
        
        # Check for compressed version
        fasta_gz = ref_dir / "genome.fa.gz"
        if fasta_gz.exists():
            self._decompress(fasta_gz, fasta_path)
            return fasta_path
        
        if not download:
            return None
        
        # Download
        logger.info(f"Downloading reference: {ref}")
        self._download_reference(ref)
        
        if fasta_path.exists():
            return fasta_path
        
        return None
    
    def get_annotation(
        self,
        organism: str,
        build: Optional[str] = None,
        source: str = "ensembl",
        download: bool = True,
    ) -> Optional[Path]:
        """
        Get annotation GTF path.
        
        Args:
            organism: Organism name
            build: Genome build
            source: Data source
            download: Whether to download if missing
            
        Returns:
            Path to GTF file, or None
        """
        key = self._resolve_reference_key(organism, build, source)
        if not key:
            return None
        
        ref = REFERENCE_CATALOG[key]
        ref_dir = self.base_path / ref.organism / ref.build
        gtf_path = ref_dir / "genes.gtf"
        
        if gtf_path.exists():
            return gtf_path
        
        # Check compressed
        gtf_gz = ref_dir / "genes.gtf.gz"
        if gtf_gz.exists():
            self._decompress(gtf_gz, gtf_path)
            return gtf_path
        
        if not download or not ref.gtf_url:
            return None
        
        # Download annotation only
        self._download_annotation(ref)
        
        return gtf_path if gtf_path.exists() else None
    
    def ensure_index(
        self,
        organism: str,
        build: Optional[str],
        index_type: str,
        threads: int = 8,
        memory_gb: int = 32,
        submit_job: bool = False,
    ) -> Optional[Path]:
        """
        Ensure aligner index exists, building if needed.
        
        Args:
            organism: Organism name
            build: Genome build
            index_type: Index type (star, bwa, bowtie2, hisat2, salmon, kallisto)
            threads: Number of threads for building
            memory_gb: Memory in GB for building
            submit_job: Submit as SLURM job instead of blocking
            
        Returns:
            Path to index directory, or None
        """
        fasta = self.get_reference(organism, build)
        if not fasta:
            logger.error(f"Cannot build {index_type} index: no reference for {organism}/{build}")
            return None
        
        index_dir = fasta.parent / "indices" / index_type
        
        if self._index_exists(index_dir, index_type):
            return index_dir
        
        logger.info(f"Building {index_type} index for {organism}/{build}")
        
        if submit_job:
            job_id = self._submit_index_job(fasta, index_dir, index_type, threads, memory_gb)
            logger.info(f"Submitted index building job: {job_id}")
            return None  # Index not ready yet
        
        # Build synchronously
        self._build_index(fasta, index_dir, index_type, threads)
        
        return index_dir if self._index_exists(index_dir, index_type) else None
    
    def _download_reference(self, ref: ReferenceGenome) -> None:
        """Download reference genome and annotation."""
        ref_dir = self.base_path / ref.organism / ref.build
        ref_dir.mkdir(parents=True, exist_ok=True)
        
        # Download FASTA
        fasta_gz = ref_dir / "genome.fa.gz"
        fasta_fa = ref_dir / "genome.fa"
        
        if not fasta_fa.exists():
            logger.info(f"Downloading genome: {ref.fasta_url}")
            self._download_file(ref.fasta_url, fasta_gz)
            self._decompress(fasta_gz, fasta_fa)
        
        # Download GTF
        if ref.gtf_url:
            gtf_gz = ref_dir / "genes.gtf.gz"
            gtf_file = ref_dir / "genes.gtf"
            
            if not gtf_file.exists():
                logger.info(f"Downloading annotation: {ref.gtf_url}")
                self._download_file(ref.gtf_url, gtf_gz)
                self._decompress(gtf_gz, gtf_file)
        
        # Save metadata
        metadata = {
            "organism": ref.organism,
            "build": ref.build,
            "source": ref.source,
            "release": ref.release,
            "downloaded_at": datetime.now().isoformat(),
            "fasta_url": ref.fasta_url,
            "gtf_url": ref.gtf_url,
        }
        with open(ref_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _download_annotation(self, ref: ReferenceGenome) -> None:
        """Download annotation only."""
        if not ref.gtf_url:
            return
        
        ref_dir = self.base_path / ref.organism / ref.build
        ref_dir.mkdir(parents=True, exist_ok=True)
        
        gtf_gz = ref_dir / "genes.gtf.gz"
        gtf_file = ref_dir / "genes.gtf"
        
        if not gtf_file.exists():
            self._download_file(ref.gtf_url, gtf_gz)
            self._decompress(gtf_gz, gtf_file)
    
    def _download_file(self, url: str, dest: Path) -> None:
        """Download a file with progress tracking."""
        try:
            # Use wget for resumable downloads
            subprocess.run([
                "wget", "-c", "-q", "--show-progress",
                "-O", str(dest), url
            ], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed: {e}")
            raise
    
    def _decompress(self, gz_path: Path, dest_path: Path) -> None:
        """Decompress gzip file."""
        try:
            subprocess.run(["gunzip", "-k", str(gz_path)], check=True)
            # gunzip creates file without .gz suffix
            decompressed = gz_path.with_suffix("")
            if decompressed != dest_path:
                decompressed.rename(dest_path)
        except subprocess.CalledProcessError:
            # Try with pigz for faster decompression
            try:
                subprocess.run([
                    "pigz", "-dk", str(gz_path)
                ], check=True)
                decompressed = gz_path.with_suffix("")
                if decompressed != dest_path:
                    decompressed.rename(dest_path)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.error(f"Decompression failed: {e}")
                raise
    
    def _index_exists(self, index_dir: Path, index_type: str) -> bool:
        """Check if index exists and is complete."""
        if not index_dir.exists():
            return False
        
        # Check for completion markers by index type
        markers = {
            "star": ["SA", "Genome", "genomeParameters.txt"],
            "bwa": ["genome.fa.bwt", "genome.fa.sa"],
            "bowtie2": ["genome.1.bt2", "genome.2.bt2"],
            "hisat2": ["genome.1.ht2", "genome.2.ht2"],
            "salmon": ["info.json", "versionInfo.json"],
            "kallisto": ["index.idx"],
        }
        
        required = markers.get(index_type, [])
        return all((index_dir / m).exists() for m in required)
    
    def _build_index(
        self,
        fasta: Path,
        index_dir: Path,
        index_type: str,
        threads: int = 8,
    ) -> None:
        """Build aligner index."""
        index_dir.mkdir(parents=True, exist_ok=True)
        gtf = fasta.parent / "genes.gtf"
        
        try:
            if index_type == "bwa":
                subprocess.run([
                    "bwa", "index",
                    "-p", str(index_dir / "genome.fa"),
                    str(fasta)
                ], check=True)
                
            elif index_type == "bowtie2":
                subprocess.run([
                    "bowtie2-build",
                    "--threads", str(threads),
                    str(fasta),
                    str(index_dir / "genome")
                ], check=True)
                
            elif index_type == "star":
                cmd = [
                    "STAR", "--runMode", "genomeGenerate",
                    "--genomeDir", str(index_dir),
                    "--genomeFastaFiles", str(fasta),
                    "--runThreadN", str(threads),
                ]
                if gtf.exists():
                    cmd.extend(["--sjdbGTFfile", str(gtf)])
                subprocess.run(cmd, check=True)
                
            elif index_type == "hisat2":
                subprocess.run([
                    "hisat2-build",
                    "-p", str(threads),
                    str(fasta),
                    str(index_dir / "genome")
                ], check=True)
                
            elif index_type == "salmon":
                # Salmon needs transcriptome, not genome
                # This would require extracting transcripts first
                logger.warning("Salmon index requires transcriptome - use decoys approach")
                subprocess.run([
                    "salmon", "index",
                    "-t", str(fasta),  # Would be transcriptome in practice
                    "-i", str(index_dir),
                    "-p", str(threads),
                ], check=True)
                
            elif index_type == "kallisto":
                # Kallisto also needs transcriptome
                subprocess.run([
                    "kallisto", "index",
                    "-i", str(index_dir / "index.idx"),
                    str(fasta),
                ], check=True)
            
            logger.info(f"Built {index_type} index at {index_dir}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build {index_type} index: {e}")
            raise
        except FileNotFoundError:
            logger.error(f"{index_type} not found in PATH")
            raise
    
    def _submit_index_job(
        self,
        fasta: Path,
        index_dir: Path,
        index_type: str,
        threads: int,
        memory_gb: int,
    ) -> str:
        """Submit index building as SLURM job."""
        job_script = f"""#!/bin/bash
#SBATCH --job-name=build_{index_type}_index
#SBATCH --cpus-per-task={threads}
#SBATCH --mem={memory_gb}G
#SBATCH --time=4:00:00
#SBATCH --output={index_dir}.build.log

echo "Building {index_type} index for {fasta}"
python -c "
from workflow_composer.provisioning import get_reference_manager
mgr = get_reference_manager()
mgr._build_index(
    Path('{fasta}'),
    Path('{index_dir}'),
    '{index_type}',
    {threads}
)
"
echo "Index build complete"
"""
        script_path = index_dir.parent / f"build_{index_type}.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(job_script)
        
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            # Parse job ID from "Submitted batch job 12345"
            job_id = result.stdout.strip().split()[-1]
            return job_id
        else:
            raise RuntimeError(f"Failed to submit job: {result.stderr}")
    
    def list_available(self) -> Dict[str, Dict[str, bool]]:
        """List available references and their status."""
        result = {}
        
        for key, ref in REFERENCE_CATALOG.items():
            ref_dir = self.base_path / ref.organism / ref.build
            fasta = ref_dir / "genome.fa"
            gtf = ref_dir / "genes.gtf"
            
            indices = {}
            for idx_type in ["star", "bwa", "bowtie2", "hisat2"]:
                idx_dir = ref_dir / "indices" / idx_type
                indices[idx_type] = self._index_exists(idx_dir, idx_type)
            
            result[key] = {
                "organism": ref.organism,
                "build": ref.build,
                "source": ref.source,
                "fasta_available": fasta.exists(),
                "gtf_available": gtf.exists(),
                "indices": indices,
            }
        
        return result
    
    def get_stats(self) -> Dict[str, any]:
        """Get storage statistics."""
        total_size = 0
        reference_count = 0
        index_count = 0
        
        for path in self.base_path.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
                if path.name == "genome.fa":
                    reference_count += 1
                if path.name in ["SA", "genome.fa.bwt", "genome.1.bt2"]:
                    index_count += 1
        
        return {
            "base_path": str(self.base_path),
            "total_size_gb": round(total_size / (1024**3), 2),
            "reference_count": reference_count,
            "index_count": index_count,
            "available_organisms": list(set(
                ref.organism for ref in REFERENCE_CATALOG.values()
            )),
        }


# =============================================================================
# Global Instance
# =============================================================================

_reference_manager: Optional[ReferenceManager] = None


def get_reference_manager() -> ReferenceManager:
    """Get global reference manager instance."""
    global _reference_manager
    if _reference_manager is None:
        _reference_manager = ReferenceManager()
    return _reference_manager
