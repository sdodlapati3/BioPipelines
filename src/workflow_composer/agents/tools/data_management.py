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
