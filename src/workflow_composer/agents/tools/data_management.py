"""
Data Management Tools
=====================

Tools for downloading, cleaning, and managing datasets.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ToolResult

logger = logging.getLogger(__name__)


# =============================================================================
# DOWNLOAD_DATASET
# =============================================================================

DOWNLOAD_DATASET_PATTERNS = [
    r"download\s+(GSE\d+|ENCSR[A-Z0-9]+|TCGA-[A-Z]+)",
    r"(?:download|get|fetch)\s+(?:the\s+)?(?:dataset\s+)?(GSE\d+|ENCSR[A-Z0-9]+|TCGA-[A-Z]+)",
    r"(?:add|queue)\s+(GSE\d+|ENCSR[A-Z0-9]+|TCGA-[A-Z]+)\s+(?:to\s+)?(?:manifest|download)",
    r"(?:download|get|fetch)\s+(?:encode|geo|tcga|gdc)\s+(?:dataset\s+)?(ENCSR[A-Z0-9]+|GSE\d+|TCGA-[A-Z]+)",
    r"(?:download|get)\s+(?:this|that)\s+(?:dataset|data)",
    r"(?:download|get)\s+(?:brain|gbm|brca|lung|colon)\s+(?:cancer\s+)?(?:methylation|data)",
]


def download_dataset_impl(
    dataset_id: str = None,
    output_dir: str = None,
    data_type: str = None,
    execute: bool = True,
) -> ToolResult:
    """
    Download a dataset from GEO, ENCODE, or TCGA/GDC.
    
    Args:
        dataset_id: Dataset ID (GSE*, ENCSR*, or TCGA-*)
        output_dir: Output directory for downloaded files
        data_type: Type of data to download (methylation, rnaseq, etc.)
        execute: Whether to actually run download commands
        
    Returns:
        ToolResult with download status
    """
    if not dataset_id:
        return ToolResult(
            success=False,
            tool_name="download_dataset",
            error="No dataset ID provided",
            message="❌ Please specify a dataset ID (e.g., GSE12345, ENCSRXYZ123, or TCGA-GBM)"
        )
    
    dataset_id = dataset_id.strip().upper()
    
    # Determine source and build download command
    if dataset_id.startswith("GSE"):
        return _download_geo(dataset_id, output_dir, execute)
    elif dataset_id.startswith("ENCSR"):
        return _download_encode(dataset_id, output_dir, execute)
    elif dataset_id.startswith("TCGA"):
        return _download_tcga(dataset_id, output_dir, data_type, execute)
    else:
        return ToolResult(
            success=False,
            tool_name="download_dataset",
            error=f"Unknown dataset format: {dataset_id}",
            message="❌ Dataset ID should start with GSE (GEO), ENCSR (ENCODE), or TCGA-* (GDC)"
        )


def _download_geo(dataset_id: str, output_dir: str, execute: bool) -> ToolResult:
    """Download from GEO using prefetch or wget."""
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={dataset_id}"
    
    if not output_dir:
        output_dir = Path.cwd() / "data" / "raw" / dataset_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for prefetch (SRA toolkit)
    has_prefetch = shutil.which("prefetch") is not None
    
    if execute and has_prefetch:
        try:
            result = subprocess.run(
                ["prefetch", dataset_id, "-O", str(output_dir)],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                message = f"""✅ **Download Complete**

| Property | Value |
|----------|-------|
| **Dataset** | [{dataset_id}]({url}) |
| **Source** | GEO/SRA |
| **Output** | `{output_dir}` |
| **Status** | Downloaded successfully |

Run `scan data` to verify the downloaded files.
"""
                return ToolResult(
                    success=True,
                    tool_name="download_dataset",
                    data={"id": dataset_id, "source": "GEO", "output": str(output_dir), "status": "completed"},
                    message=message
                )
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            logger.warning(f"prefetch failed: {e}")
    
    # Fallback to instructions
    message = f"""📥 **Download Dataset: {dataset_id}**

| Property | Value |
|----------|-------|
| **Dataset** | [{dataset_id}]({url}) |
| **Source** | GEO |
| **Output** | `{output_dir}` |

**Download Commands:**

```bash
# Option 1: SRA Toolkit (recommended)
prefetch {dataset_id} -O {output_dir}
fasterq-dump {dataset_id} -O {output_dir}

# Option 2: Direct FTP
wget -r -np -nd -P {output_dir} \\
  ftp://ftp.ncbi.nlm.nih.gov/geo/series/{dataset_id[:6]}nnn/{dataset_id}/suppl/
```
"""
    return ToolResult(
        success=True,
        tool_name="download_dataset",
        data={"id": dataset_id, "source": "GEO", "output": str(output_dir), "status": "instructions"},
        message=message
    )


def _download_encode(dataset_id: str, output_dir: str, execute: bool) -> ToolResult:
    """Download from ENCODE."""
    url = f"https://www.encodeproject.org/experiments/{dataset_id}"
    
    if not output_dir:
        output_dir = Path.cwd() / "data" / "raw" / dataset_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    message = f"""📥 **Download Dataset: {dataset_id}**

| Property | Value |
|----------|-------|
| **Dataset** | [{dataset_id}]({url}) |
| **Source** | ENCODE |
| **Output** | `{output_dir}` |

**Download Commands:**

```bash
cd {output_dir}

# Get file list from ENCODE
curl -s "{url}/?format=json" | jq -r '.files[].href' > files.txt

# Download all files
xargs -L 1 curl -O -J -L < files.txt
```
"""
    return ToolResult(
        success=True,
        tool_name="download_dataset",
        data={"id": dataset_id, "source": "ENCODE", "output": str(output_dir)},
        message=message
    )


def _download_tcga(dataset_id: str, output_dir: str, data_type: str, execute: bool) -> ToolResult:
    """Download from TCGA/GDC."""
    project = dataset_id if dataset_id.startswith("TCGA-") else f"TCGA-{dataset_id}"
    url = f"https://portal.gdc.cancer.gov/projects/{project}"
    
    if not output_dir:
        output_dir = Path.cwd() / "data" / "raw" / project
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for gdc-client
    has_gdc = shutil.which("gdc-client") is not None
    
    # Data type filter
    data_type_filter = ""
    if data_type:
        dt_lower = data_type.lower()
        if "methyl" in dt_lower:
            data_type_filter = "Methylation Beta Value"
        elif "rna" in dt_lower or "expression" in dt_lower:
            data_type_filter = "Gene Expression Quantification"
        elif "wgs" in dt_lower or "wes" in dt_lower or "exome" in dt_lower:
            data_type_filter = "Aligned Reads"
    
    # Build GDC API query
    api_url = "https://api.gdc.cancer.gov/files"
    
    message = f"""📥 **Download TCGA Data: {project}**

| Property | Value |
|----------|-------|
| **Project** | [{project}]({url}) |
| **Source** | GDC/TCGA |
| **Output** | `{output_dir}` |
| **Data Type** | {data_type_filter or 'All'} |

**Download Steps:**

### 1. Create Manifest (via GDC Portal)
1. Go to [{project}]({url})
2. Click **Files** tab
3. Filter by **Data Type** → {data_type_filter or 'Methylation Beta Value'}
4. Click **Add All Files to Cart**
5. Go to Cart → **Download Manifest**

### 2. Download with gdc-client
```bash
cd {output_dir}

# Download using manifest
gdc-client download -m gdc_manifest.txt -d .

# Or download specific file IDs
gdc-client download <file-uuid>
```

### 3. Alternative: API Query
```bash
# Query available methylation files
curl '{api_url}?filters={{"op":"and","content":[{{"op":"=","content":{{"field":"cases.project.project_id","value":"{project}"}}}},{{"op":"=","content":{{"field":"data_type","value":"Methylation Beta Value"}}}}]}}&format=json&size=10' | jq '.data.hits[].file_id'
```

**Note:** TCGA data requires authentication for controlled-access data.
Get a token from [GDC Portal](https://portal.gdc.cancer.gov/auth).
"""

    return ToolResult(
        success=True,
        tool_name="download_dataset",
        data={
            "id": project,
            "source": "TCGA/GDC",
            "output": str(output_dir),
            "data_type": data_type_filter,
            "gdc_client_available": has_gdc,
        },
        message=message
    )


# =============================================================================
# CLEANUP_DATA
# =============================================================================

CLEANUP_DATA_PATTERNS = [
    r"(?:can you\s+)?(?:clean\s*up|remove|delete)\s+(?:the\s+)?(?:corrupted|invalid|bad|broken)\s+(?:data|files?)",
    r"(?:clean\s*up|fix)\s+(?:the\s+)?(?:data\s+)?(?:folder|directory)",
    r"(?:remove|delete)\s+(?:the\s+)?(?:html|corrupted|invalid)\s+(?:fastq|files?)",
]

# Store pending cleanup info globally
_pending_cleanup = {}


def cleanup_data_impl(
    path: str = None,
    confirm: bool = False,
) -> ToolResult:
    """
    Clean up corrupted data files.
    
    Args:
        path: Path to clean up
        confirm: Whether to actually delete files
        
    Returns:
        ToolResult with cleanup status
    """
    global _pending_cleanup
    
    if not path:
        default_paths = [
            Path("/scratch/sdodl001/BioPipelines/data/raw"),
            Path.home() / "BioPipelines" / "data" / "raw",
            Path.cwd() / "data" / "raw",
        ]
        for p in default_paths:
            if p.exists():
                path = str(p)
                break
    
    if not path:
        return ToolResult(
            success=False,
            tool_name="cleanup_data",
            error="No path specified",
            message="❌ Please specify a path to clean up"
        )
    
    path = Path(path).expanduser().resolve()
    
    if not path.exists():
        return ToolResult(
            success=False,
            tool_name="cleanup_data",
            error=f"Path not found: {path}",
            message=f"❌ Path not found: `{path}`"
        )
    
    # Find problematic files
    issues = []
    
    try:
        for f in path.rglob("*"):
            if f.is_file():
                # Check for HTML error pages masquerading as data
                try:
                    with open(f, 'rb') as fp:
                        header = fp.read(200)
                        if b'<!DOCTYPE' in header or b'<html' in header.lower():
                            issues.append(("html_error", f))
                        elif f.suffix in ['.fastq', '.fq', '.fastq.gz', '.fq.gz']:
                            if not (header.startswith(b'@') or header[:2] == b'\x1f\x8b'):
                                issues.append(("invalid_fastq", f))
                except Exception:
                    pass
        
        if not issues:
            return ToolResult(
                success=True,
                tool_name="cleanup_data",
                data={"path": str(path), "issues": []},
                message=f"✅ No corrupted files found in `{path}`"
            )
        
        if confirm:
            # Actually delete files
            deleted = []
            for issue_type, f in issues:
                try:
                    f.unlink()
                    deleted.append(str(f))
                except Exception as e:
                    logger.error(f"Failed to delete {f}: {e}")
            
            _pending_cleanup = {}
            
            return ToolResult(
                success=True,
                tool_name="cleanup_data",
                data={"deleted": deleted},
                message=f"✅ Deleted {len(deleted)} corrupted files"
            )
        else:
            # Store for confirmation
            _pending_cleanup = {"path": str(path), "issues": issues}
            
            issue_list = "\n".join([
                f"  - `{f.name}` ({issue_type})"
                for issue_type, f in issues[:10]
            ])
            if len(issues) > 10:
                issue_list += f"\n  - ... and {len(issues) - 10} more"
            
            return ToolResult(
                success=True,
                tool_name="cleanup_data",
                data={"path": str(path), "count": len(issues)},
                message=f"""⚠️ Found **{len(issues)} problematic files**:

{issue_list}

Say **"yes, delete them"** to remove these files.
"""
            )
    
    except Exception as e:
        return ToolResult(
            success=False,
            tool_name="cleanup_data",
            error=str(e),
            message=f"❌ Cleanup error: {e}"
        )


# =============================================================================
# CONFIRM_CLEANUP
# =============================================================================

CONFIRM_CLEANUP_PATTERNS = [
    r"^(?:yes|yep|yeah|y)\s*[,.]?\s*(?:delete|remove|confirm|do it|go ahead|proceed)",
    r"confirm\s+(?:cleanup|deletion|removal)",
    r"(?:yes|ok|okay|sure|please)\s*[,.]?\s*(?:delete|remove)\s+(?:them|these|those|the files?)",
    r"^(?:delete|remove)\s+(?:them|these|those|the files?)$",
    r"^proceed(?:\s+with\s+(?:cleanup|deletion))?$",
]


def confirm_cleanup_impl() -> ToolResult:
    """
    Confirm and execute pending cleanup.
    
    Returns:
        ToolResult with cleanup status
    """
    global _pending_cleanup
    
    if not _pending_cleanup:
        return ToolResult(
            success=False,
            tool_name="confirm_cleanup",
            error="No pending cleanup",
            message="❌ No pending cleanup operation. Run `cleanup data` first."
        )
    
    issues = _pending_cleanup.get("issues", [])
    
    deleted = []
    failed = []
    
    for issue_type, f in issues:
        try:
            if f.exists():
                f.unlink()
                deleted.append(str(f))
        except Exception as e:
            failed.append((str(f), str(e)))
    
    _pending_cleanup = {}
    
    if failed:
        fail_list = "\n".join([f"  - `{f}`: {e}" for f, e in failed[:5]])
        message = f"""⚠️ Cleanup completed with errors:

**Deleted:** {len(deleted)} files
**Failed:** {len(failed)} files

{fail_list}
"""
    else:
        message = f"✅ Successfully deleted {len(deleted)} corrupted files"
    
    return ToolResult(
        success=len(failed) == 0,
        tool_name="confirm_cleanup",
        data={"deleted": deleted, "failed": failed},
        message=message
    )


# =============================================================================
# DOWNLOAD_REFERENCE
# =============================================================================

DOWNLOAD_REFERENCE_PATTERNS = [
    r"download\s+(?:the\s+)?(?:reference|genome|annotation|gtf|transcriptome)(?:\s+for)?\s+(\w+)",
    r"(?:get|fetch|download)\s+(?:human|mouse|rat|zebrafish)\s+(?:GRCh38|GRCh37|GRCm39|GRCm38|mRatBN7\.2|GRCz11)",
    r"download\s+(?:GRCh38|GRCh37|GRCm39|GRCm38)\s+(?:reference|genome|gtf)",
    r"(?:need|get|download)\s+(?:star|salmon|bwa|hisat2)\s+index",
    r"build\s+(?:star|salmon|bwa|hisat2)\s+index\s+for\s+(\w+)",
]


def download_reference_impl(
    organism: str = "human",
    assembly: str = "GRCh38",
    resource: str = "genome",  # genome, gtf, transcriptome
    output_dir: str = None,
) -> ToolResult:
    """
    Download reference genome, annotation, or transcriptome from Ensembl.
    
    Uses ReferenceManager for robust downloads with progress tracking.
    
    Args:
        organism: Organism name (human, mouse, rat, zebrafish)
        assembly: Genome assembly (GRCh38, GRCh37, GRCm39, etc.)
        resource: What to download (genome, gtf, transcriptome)
        output_dir: Optional output directory
        
    Returns:
        ToolResult with download status and file path
    """
    try:
        # Import ReferenceManager
        try:
            from workflow_composer.data.reference_manager import ReferenceManager, REFERENCE_SOURCES
        except ImportError as e:
            return ToolResult(
                success=False,
                tool_name="download_reference",
                error=str(e),
                message=f"❌ Could not import ReferenceManager: {e}"
            )
        
        # Normalize inputs
        organism = organism.lower().strip()
        assembly = assembly.strip()
        resource = resource.lower().strip()
        
        # Validate organism
        if organism not in REFERENCE_SOURCES:
            available = ", ".join(REFERENCE_SOURCES.keys())
            return ToolResult(
                success=False,
                tool_name="download_reference",
                error=f"Unknown organism: {organism}",
                message=f"❌ Unknown organism: **{organism}**\n\nAvailable: {available}"
            )
        
        # Validate assembly
        if assembly not in REFERENCE_SOURCES[organism]:
            available = ", ".join(REFERENCE_SOURCES[organism].keys())
            return ToolResult(
                success=False,
                tool_name="download_reference",
                error=f"Unknown assembly: {assembly}",
                message=f"❌ Unknown assembly: **{assembly}** for {organism}\n\nAvailable: {available}"
            )
        
        # Validate resource
        if resource not in REFERENCE_SOURCES[organism][assembly]:
            available = ", ".join(REFERENCE_SOURCES[organism][assembly].keys())
            return ToolResult(
                success=False,
                tool_name="download_reference",
                error=f"Unknown resource: {resource}",
                message=f"❌ Unknown resource: **{resource}**\n\nAvailable: {available}"
            )
        
        # Setup reference manager
        if output_dir:
            base_dir = Path(output_dir)
        else:
            # Try standard locations
            possible_dirs = [
                Path("/scratch/sdodl001/BioPipelines/data/references"),
                Path.home() / "BioPipelines" / "data" / "references",
                Path.cwd() / "data" / "references",
            ]
            base_dir = None
            for d in possible_dirs:
                if d.exists() or d.parent.exists():
                    base_dir = d
                    break
            if base_dir is None:
                base_dir = Path.cwd() / "data" / "references"
        
        manager = ReferenceManager(base_dir=base_dir)
        
        # Get download info
        info = REFERENCE_SOURCES[organism][assembly][resource]
        url = info["url"]
        expected_path = base_dir / organism / info.get("decompressed", info["filename"].replace(".gz", ""))
        
        # Check if already exists
        if expected_path.exists():
            size_mb = expected_path.stat().st_size / (1024 * 1024)
            return ToolResult(
                success=True,
                tool_name="download_reference",
                data={
                    "path": str(expected_path),
                    "already_exists": True,
                    "size_mb": round(size_mb, 1),
                },
                message=f"✅ Reference already exists:\n\n`{expected_path}`\n\n**Size:** {size_mb:.1f} MB"
            )
        
        # Download
        logger.info(f"Downloading {organism}/{assembly}/{resource} from {url}")
        
        result_path = manager.download_reference(
            organism=organism,
            assembly=assembly,
            resource=resource,
        )
        
        if result_path and result_path.exists():
            size_mb = result_path.stat().st_size / (1024 * 1024)
            return ToolResult(
                success=True,
                tool_name="download_reference",
                data={
                    "path": str(result_path),
                    "organism": organism,
                    "assembly": assembly,
                    "resource": resource,
                    "size_mb": round(size_mb, 1),
                },
                message=f"""✅ **Download Complete**

**File:** `{result_path}`
**Size:** {size_mb:.1f} MB
**Organism:** {organism}
**Assembly:** {assembly}
**Resource:** {resource}

You can now build an index with:
- `build star index for {assembly}`
- `build salmon index for {assembly}`
"""
            )
        else:
            return ToolResult(
                success=False,
                tool_name="download_reference",
                error="Download failed",
                message=f"❌ Failed to download {resource} for {organism}/{assembly}"
            )
            
    except Exception as e:
        logger.exception("download_reference failed")
        return ToolResult(
            success=False,
            tool_name="download_reference",
            error=str(e),
            message=f"❌ Download error: {e}"
        )


# =============================================================================
# BUILD_INDEX
# =============================================================================

BUILD_INDEX_PATTERNS = [
    r"build\s+(star|salmon|bwa|hisat2|kallisto)\s+index",
    r"(?:create|generate|make)\s+(star|salmon|bwa|hisat2|kallisto)\s+index",
    r"index\s+(?:the\s+)?genome\s+(?:with|for|using)\s+(star|salmon|bwa|hisat2|kallisto)",
]


def build_index_impl(
    aligner: str,
    genome_path: str = None,
    gtf_path: str = None,
    organism: str = "human",
    assembly: str = "GRCh38",
    output_dir: str = None,
    threads: int = 8,
    memory_gb: int = 32,
) -> ToolResult:
    """
    Build an aligner index for a genome.
    
    Uses ReferenceManager to build STAR, Salmon, BWA, HISAT2, or Kallisto indexes.
    
    Args:
        aligner: Aligner to build index for (star, salmon, bwa, hisat2, kallisto)
        genome_path: Path to genome FASTA (if not provided, uses organism/assembly)
        gtf_path: Path to GTF annotation (optional, but recommended for STAR)
        organism: Organism name (used if genome_path not provided)
        assembly: Genome assembly (used if genome_path not provided)
        output_dir: Output directory for index
        threads: Number of threads
        memory_gb: Memory limit in GB
        
    Returns:
        ToolResult with build status and index path
    """
    try:
        # Import ReferenceManager
        try:
            from workflow_composer.data.reference_manager import ReferenceManager
        except ImportError as e:
            return ToolResult(
                success=False,
                tool_name="build_index",
                error=str(e),
                message=f"❌ Could not import ReferenceManager: {e}"
            )
        
        aligner = aligner.lower().strip()
        valid_aligners = ["star", "salmon", "bwa", "hisat2", "kallisto"]
        
        if aligner not in valid_aligners:
            return ToolResult(
                success=False,
                tool_name="build_index",
                error=f"Unknown aligner: {aligner}",
                message=f"❌ Unknown aligner: **{aligner}**\n\nSupported: {', '.join(valid_aligners)}"
            )
        
        # Find genome if not provided
        if genome_path:
            genome = Path(genome_path)
        else:
            # Try to find genome using ReferenceManager
            base_dir = None
            possible_dirs = [
                Path("/scratch/sdodl001/BioPipelines/data/references"),
                Path.home() / "BioPipelines" / "data" / "references",
                Path.cwd() / "data" / "references",
            ]
            for d in possible_dirs:
                if d.exists():
                    base_dir = d
                    break
            
            if base_dir is None:
                return ToolResult(
                    success=False,
                    tool_name="build_index",
                    error="No reference directory found",
                    message="❌ Could not find reference directory. Please specify `genome_path`."
                )
            
            manager = ReferenceManager(base_dir=base_dir)
            ref_info = manager.check_references(organism, assembly)
            
            if not ref_info.genome_fasta:
                return ToolResult(
                    success=False,
                    tool_name="build_index",
                    error="Genome not found",
                    message=f"""❌ Genome not found for {organism}/{assembly}

**Download first with:**
```
download reference genome for {organism} {assembly}
```
"""
                )
            
            genome = ref_info.genome_fasta
            
            # Use discovered GTF if not provided
            if not gtf_path and ref_info.annotation_gtf:
                gtf_path = str(ref_info.annotation_gtf)
        
        if not genome.exists():
            return ToolResult(
                success=False,
                tool_name="build_index",
                error=f"Genome not found: {genome}",
                message=f"❌ Genome file not found: `{genome}`"
            )
        
        gtf = Path(gtf_path) if gtf_path else None
        
        # Warn if STAR without GTF
        if aligner == "star" and (gtf is None or not gtf.exists()):
            logger.warning("Building STAR index without GTF - splice junctions won't be annotated")
        
        # Setup output directory
        if output_dir:
            out_dir = Path(output_dir)
        else:
            out_dir = genome.parent / "indexes" / f"{aligner}_{assembly}"
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if index already exists
        manager = ReferenceManager(base_dir=genome.parent.parent)
        if manager._validate_index(aligner, out_dir):
            return ToolResult(
                success=True,
                tool_name="build_index",
                data={
                    "path": str(out_dir),
                    "aligner": aligner,
                    "already_exists": True,
                },
                message=f"✅ Index already exists:\n\n`{out_dir}`"
            )
        
        # Build index
        logger.info(f"Building {aligner} index: {genome} -> {out_dir}")
        
        result_path = manager.build_index(
            aligner=aligner,
            genome_path=genome,
            gtf_path=gtf,
            output_dir=out_dir,
            threads=threads,
            memory_gb=memory_gb,
        )
        
        if result_path and result_path.exists():
            return ToolResult(
                success=True,
                tool_name="build_index",
                data={
                    "path": str(result_path),
                    "aligner": aligner,
                    "genome": str(genome),
                    "gtf": str(gtf) if gtf else None,
                },
                message=f"""✅ **Index Build Complete**

**Aligner:** {aligner}
**Index Path:** `{result_path}`
**Genome:** `{genome.name}`
"""
            )
        else:
            return ToolResult(
                success=False,
                tool_name="build_index",
                error="Index build failed",
                message=f"❌ Failed to build {aligner} index\n\nCheck logs for details."
            )
            
    except Exception as e:
        logger.exception("build_index failed")
        return ToolResult(
            success=False,
            tool_name="build_index",
            error=str(e),
            message=f"❌ Index build error: {e}"
        )
