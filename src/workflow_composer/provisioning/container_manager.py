"""
Container Manager
=================

Manages Singularity container provisioning for BioPipelines.

Features:
- Pull containers from registry (GHCR, Docker Hub)
- Build containers from Dockerfiles
- Verify container integrity
- Cache management

Usage:
    from workflow_composer.provisioning import get_container_manager
    
    cm = get_container_manager()
    
    # Get container, pulling if needed
    sif_path = cm.get_container("rna-seq")
    
    # Verify container
    if cm.verify_container(sif_path):
        print("Container is valid")
"""

import os
import subprocess
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


# Container registry mapping
CONTAINER_REGISTRY: Dict[str, Dict[str, str]] = {
    # Base containers
    "base": {
        "uri": "docker://ghcr.io/sdodlapa/biopipelines-base:latest",
        "description": "Base tools (samtools, bedtools, fastqc, multiqc)",
        "tools": ["fastqc", "multiqc", "samtools", "bedtools", "pigz"],
    },
    # Analysis-specific containers
    "rna-seq": {
        "uri": "docker://ghcr.io/sdodlapa/biopipelines-rnaseq:latest",
        "description": "RNA-seq tools (STAR, HISAT2, salmon, featureCounts, DESeq2)",
        "tools": ["star", "hisat2", "salmon", "kallisto", "featurecounts", "deseq2", "edger", "stringtie"],
    },
    "chip-seq": {
        "uri": "docker://ghcr.io/sdodlapa/biopipelines-chipseq:latest",
        "description": "ChIP-seq tools (bowtie2, macs2, homer, deeptools)",
        "tools": ["bowtie2", "macs2", "macs3", "homer", "deeptools", "sicer2"],
    },
    "dna-seq": {
        "uri": "docker://ghcr.io/sdodlapa/biopipelines-dnaseq:latest",
        "description": "DNA-seq/variant calling (BWA, GATK, bcftools, freebayes)",
        "tools": ["bwa", "bwa-mem2", "gatk", "bcftools", "freebayes", "picard", "deepvariant"],
    },
    "scrna-seq": {
        "uri": "docker://ghcr.io/sdodlapa/biopipelines-scrnaseq:latest",
        "description": "Single-cell RNA-seq (Seurat, Scanpy, CellRanger)",
        "tools": ["cellranger", "seurat", "scanpy", "scvelo", "velocyto"],
    },
    "metagenomics": {
        "uri": "docker://ghcr.io/sdodlapa/biopipelines-metagenomics:latest",
        "description": "Metagenomics (Kraken2, MetaPhlAn, HUMAnN)",
        "tools": ["kraken2", "metaphlan", "humann", "bracken", "megahit"],
    },
    "methylation": {
        "uri": "docker://ghcr.io/sdodlapa/biopipelines-methylation:latest",
        "description": "Methylation analysis (Bismark, methylpy)",
        "tools": ["bismark", "bsmap", "methylpy", "methyldackel"],
    },
    "long-read": {
        "uri": "docker://ghcr.io/sdodlapa/biopipelines-longread:latest",
        "description": "Long-read sequencing (minimap2, flye, canu)",
        "tools": ["minimap2", "flye", "canu", "medaka", "racon", "wtdbg2"],
    },
    "structural-variants": {
        "uri": "docker://ghcr.io/sdodlapa/biopipelines-sv:latest",
        "description": "Structural variant calling (Manta, DELLY, SURVIVOR)",
        "tools": ["manta", "delly", "lumpy", "survivor", "svaba"],
    },
    "hic": {
        "uri": "docker://ghcr.io/sdodlapa/biopipelines-hic:latest",
        "description": "Hi-C/3D genome analysis",
        "tools": ["hicup", "cooler", "juicer", "hicexplorer", "homer"],
    },
    # Fallback - nf-core containers (don't include in TOOL_CONTAINER_MAP)
    "nf-core-rnaseq": {
        "uri": "docker://quay.io/nf-core/rnaseq:3.14.0",
        "description": "nf-core RNA-seq pipeline container",
        "tools": [],  # Don't override primary mappings
        "fallback_tools": ["star", "salmon", "hisat2"],
    },
    "nf-core-chipseq": {
        "uri": "docker://quay.io/nf-core/chipseq:2.0.0",
        "description": "nf-core ChIP-seq pipeline container",
        "tools": [],  # Don't override primary mappings
        "fallback_tools": ["bowtie2", "macs2"],
    },
}

# Tool to container mapping
TOOL_CONTAINER_MAP: Dict[str, str] = {}
for container, info in CONTAINER_REGISTRY.items():
    for tool in info.get("tools", []):
        TOOL_CONTAINER_MAP[tool] = container


@dataclass
class ContainerInfo:
    """Container metadata."""
    name: str
    path: Optional[Path]
    uri: str
    size_mb: float
    pulled_at: Optional[datetime]
    verified: bool
    tools: List[str]


class ContainerManager:
    """
    Manages Singularity container provisioning.
    
    Features:
    - Pull containers from registries
    - Build from local Dockerfiles
    - Verify container integrity
    - Cache management
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        project_dir: Optional[str] = None,
    ):
        """
        Initialize container manager.
        
        Args:
            cache_dir: Directory for cached container images
            project_dir: BioPipelines project directory
        """
        if cache_dir is None:
            cache_dir = os.environ.get(
                'SINGULARITY_CACHEDIR',
                str(Path.home() / ".biopipelines" / "containers")
            )
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if project_dir is None:
            project_dir = os.environ.get(
                'BIOPIPELINES_HOME',
                '/home/sdodl001_odu_edu/BioPipelines'
            )
        self.project_dir = Path(project_dir)
        self.containers_dir = self.project_dir / "containers"
        
        logger.info(f"ContainerManager initialized: {self.cache_dir}")
    
    def get_container(
        self,
        name: str,
        pull: bool = True,
        build_local: bool = False,
    ) -> Optional[Path]:
        """
        Get container path, pulling if needed.
        
        Args:
            name: Container name (e.g., "rna-seq", "base")
            pull: Pull from registry if not cached
            build_local: Build from local Dockerfile if pull fails
            
        Returns:
            Path to .sif file, or None if not available
        """
        sif_path = self.cache_dir / f"{name}.sif"
        
        # Check cache
        if sif_path.exists():
            if self.verify_container(sif_path):
                return sif_path
            else:
                logger.warning(f"Cached container {name} is invalid, removing")
                sif_path.unlink()
        
        if not pull:
            return None
        
        # Check if we have this container in registry
        if name not in CONTAINER_REGISTRY:
            logger.warning(f"Unknown container: {name}")
            if build_local:
                return self._build_from_dockerfile(name)
            return None
        
        # Pull from registry
        try:
            self._pull_container(name, sif_path)
            return sif_path
        except Exception as e:
            logger.error(f"Failed to pull container {name}: {e}")
            
            if build_local:
                return self._build_from_dockerfile(name)
            
            return None
    
    def get_container_for_tool(self, tool: str) -> Optional[Path]:
        """
        Get container that includes a specific tool.
        
        Args:
            tool: Tool name (e.g., "star", "bwa")
            
        Returns:
            Path to container with that tool
        """
        container_name = TOOL_CONTAINER_MAP.get(tool.lower())
        if not container_name:
            # Fall back to base container
            logger.warning(f"Unknown tool {tool}, using base container")
            container_name = "base"
        
        return self.get_container(container_name)
    
    def verify_container(self, sif_path: Path) -> bool:
        """
        Verify container is valid and runnable.
        
        Args:
            sif_path: Path to .sif file
            
        Returns:
            True if container is valid
        """
        if not sif_path.exists():
            return False
        
        try:
            result = subprocess.run(
                ["singularity", "exec", str(sif_path), "echo", "OK"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0 and "OK" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"Container verification failed: {e}")
            return False
    
    def list_tools_in_container(self, sif_path: Path) -> List[str]:
        """
        List tools available in a container.
        
        Args:
            sif_path: Path to .sif file
            
        Returns:
            List of available tool names
        """
        # Check common bioinformatics tools
        tools_to_check = [
            "fastqc", "multiqc", "samtools", "bedtools",
            "star", "hisat2", "salmon", "kallisto",
            "bowtie2", "bwa", "minimap2",
            "gatk", "bcftools", "freebayes",
            "macs2", "macs3", "homer",
            "featureCounts", "htseq-count",
            "R", "python", "python3",
        ]
        
        available = []
        for tool in tools_to_check:
            try:
                result = subprocess.run(
                    ["singularity", "exec", str(sif_path), "which", tool],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    available.append(tool)
            except (subprocess.TimeoutExpired, Exception):
                continue
        
        return available
    
    def _pull_container(self, name: str, dest: Path) -> None:
        """Pull container from registry."""
        if name not in CONTAINER_REGISTRY:
            raise ValueError(f"Unknown container: {name}")
        
        uri = CONTAINER_REGISTRY[name]["uri"]
        logger.info(f"Pulling container: {uri}")
        
        # Create temp file for pull
        temp_sif = dest.with_suffix(".sif.tmp")
        
        try:
            subprocess.run(
                ["singularity", "pull", "--force", str(temp_sif), uri],
                check=True,
                capture_output=True,
                text=True,
            )
            
            # Verify before moving to final location
            if self.verify_container(temp_sif):
                temp_sif.rename(dest)
                logger.info(f"Pulled container: {dest}")
            else:
                temp_sif.unlink()
                raise RuntimeError("Container verification failed")
                
        except subprocess.CalledProcessError as e:
            if temp_sif.exists():
                temp_sif.unlink()
            raise RuntimeError(f"Pull failed: {e.stderr}")
    
    def _build_from_dockerfile(self, name: str) -> Optional[Path]:
        """Build container from local Dockerfile."""
        dockerfile_dir = self.containers_dir / name
        dockerfile = dockerfile_dir / "Dockerfile"
        
        if not dockerfile.exists():
            logger.error(f"No Dockerfile found: {dockerfile}")
            return None
        
        sif_path = self.cache_dir / f"{name}.sif"
        
        logger.info(f"Building container from {dockerfile}")
        
        try:
            # Build Docker image first
            image_tag = f"biopipelines-{name}:local"
            subprocess.run(
                ["docker", "build", "-t", image_tag, str(dockerfile_dir)],
                check=True,
            )
            
            # Convert to Singularity
            subprocess.run(
                ["singularity", "build", str(sif_path), f"docker-daemon://{image_tag}"],
                check=True,
            )
            
            return sif_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Build failed: {e}")
            return None
        except FileNotFoundError as e:
            logger.error(f"Docker/Singularity not found: {e}")
            return None
    
    def list_available(self) -> Dict[str, ContainerInfo]:
        """List all containers and their status."""
        result = {}
        
        for name, info in CONTAINER_REGISTRY.items():
            sif_path = self.cache_dir / f"{name}.sif"
            
            size_mb = 0.0
            pulled_at = None
            verified = False
            
            if sif_path.exists():
                stat = sif_path.stat()
                size_mb = stat.st_size / (1024 * 1024)
                pulled_at = datetime.fromtimestamp(stat.st_mtime)
                verified = self.verify_container(sif_path)
            
            result[name] = ContainerInfo(
                name=name,
                path=sif_path if sif_path.exists() else None,
                uri=info["uri"],
                size_mb=round(size_mb, 1),
                pulled_at=pulled_at,
                verified=verified,
                tools=info.get("tools", []),
            )
        
        return result
    
    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        total_size = 0
        container_count = 0
        
        for sif in self.cache_dir.glob("*.sif"):
            total_size += sif.stat().st_size
            container_count += 1
        
        return {
            "cache_dir": str(self.cache_dir),
            "total_size_gb": round(total_size / (1024**3), 2),
            "container_count": container_count,
            "registry_count": len(CONTAINER_REGISTRY),
        }
    
    def clear_cache(self, keep: Optional[List[str]] = None) -> int:
        """
        Clear container cache.
        
        Args:
            keep: List of container names to keep
            
        Returns:
            Number of containers removed
        """
        keep = keep or []
        removed = 0
        
        for sif in self.cache_dir.glob("*.sif"):
            name = sif.stem
            if name not in keep:
                sif.unlink()
                removed += 1
                logger.info(f"Removed cached container: {name}")
        
        return removed


# =============================================================================
# Global Instance
# =============================================================================

_container_manager: Optional[ContainerManager] = None


def get_container_manager() -> ContainerManager:
    """Get global container manager instance."""
    global _container_manager
    if _container_manager is None:
        _container_manager = ContainerManager()
    return _container_manager
