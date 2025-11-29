"""
Data Discovery Tools
====================

Tools for scanning local data and searching remote databases.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ToolResult, ToolParameter
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# SCAN_DATA
# =============================================================================

SCAN_DATA_PATTERNS = [
    # Pattern 1: "scan/find data in /path" - path must start with / or ~
    r"(?:can you\s+)?(?:scan|find|look for|check|discover|list|show)\s+(?:the\s+)?(?:local\s+)?(?:data|files?|samples?|fastq|folders?|directories?|datasets?)\s+(?:in|at|from|under|within)\s+['\"]?([\/~][^\s'\"\?]+)['\"]?",
    # Pattern 2: "in data dir /path" - captures path after "dir"  
    r"(?:in|at|from)\s+(?:data\s+)?(?:dir|directory|folder)\s+['\"]?([\/~][^\s'\"\?]+)['\"]?",
    # Pattern 3: Simple "scan /path"
    r"(?:scan|check|look in)\s+['\"]?([\/~][^\s'\"\?]+)['\"]?",
    # Pattern 4: "what data is available in /path"
    r"(?:what|which)\s+(?:data|files?|samples?|datasets?)\s+(?:are|is\s+available|is|do i have|exist|available)\s+(?:in|at)\s+['\"]?([\/~][^\s'\"\?]+)['\"]?",
    # Pattern 5: "scan local folders" without path
    r"(?:can you\s+)?(?:scan|find|look for|check|discover|list|show)\s+(?:me\s+)?(?:my\s+)?(?:the\s+)?(?:local\s+)?(?:data|files?|samples?|folders?|directories?|datasets?)",
    # Pattern 6: "what data is available" without path
    r"(?:what|which)\s+(?:data|files?|samples?|datasets?)\s+(?:are|is|do i have)\s*(?:available|there)?",
    # Pattern 7: "show me my datasets/data"
    r"show\s+(?:me\s+)?(?:my\s+)?(?:what\s+)?(?:data|datasets?|samples?|files?)",
]


def scan_data_impl(
    path: str = None,
    scanner=None,
    manifest=None,
) -> ToolResult:
    """
    Scan a directory for FASTQ files.
    
    Args:
        path: Directory path to scan. Defaults to current directory.
        scanner: DataScanner instance
        manifest: DataManifest instance
        
    Returns:
        ToolResult with discovered samples
    """
    if scanner is None:
        return ToolResult(
            success=False,
            tool_name="scan_data",
            error="Scanner not available",
            message="❌ Data scanner is not available. Please check installation."
        )
    
    # Smart default paths
    if not path:
        default_paths = [
            Path("/scratch/sdodl001/BioPipelines"),
            Path("/scratch/sdodl001/BioPipelines/data"),
            Path.home() / "BioPipelines" / "data",
            Path.home() / "data",
            Path.cwd() / "data",
            Path.cwd(),
        ]
        for p in default_paths:
            if p.exists() and p.is_dir():
                path = str(p)
                break
        else:
            path = str(Path.cwd())
    
    # Clean up path
    path = path.strip().strip("'\"")
    scan_path = Path(path).expanduser().resolve()
    
    if not scan_path.exists():
        return ToolResult(
            success=False,
            tool_name="scan_data",
            error=f"Path not found: {scan_path}",
            message=f"❌ Directory not found: `{scan_path}`"
        )
    
    try:
        result = scanner.scan_directory(scan_path, recursive=True)
        samples = result.samples if hasattr(result, 'samples') else []
        
        # Add samples to manifest
        if manifest and samples:
            for sample in samples:
                manifest.add_sample(sample)
        
        # Build response message
        if samples:
            sample_list = []
            for s in samples[:10]:
                file_count = 2 if (hasattr(s, 'is_paired') and s.is_paired) or (hasattr(s, 'fastq_2') and s.fastq_2) else 1
                layout = "paired" if file_count == 2 else "single"
                if hasattr(s, 'library_layout'):
                    layout = s.library_layout.value if hasattr(s.library_layout, 'value') else str(s.library_layout)
                sample_list.append(f"  - `{s.sample_id}`: {file_count} files ({layout})")
            
            sample_str = "\n".join(sample_list)
            if len(samples) > 10:
                sample_str += f"\n  - ... and {len(samples) - 10} more"
            
            message = f"""✅ Found **{len(samples)} samples** in `{scan_path}`:

{sample_str}

Added to data manifest. Ready for workflow generation!"""
        else:
            message = f"⚠️ No FASTQ samples found in `{scan_path}`"
        
        return ToolResult(
            success=True,
            tool_name="scan_data",
            data={
                "samples": samples,
                "path": str(scan_path),
                "count": len(samples)
            },
            message=message,
            ui_update={
                "manifest_sample_count": len(samples),
                "manifest_path": str(scan_path)
            }
        )
        
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        return ToolResult(
            success=False,
            tool_name="scan_data",
            error=str(e),
            message=f"❌ Failed to scan directory: {e}"
        )


# =============================================================================
# SEARCH_DATABASES
# =============================================================================

SEARCH_DATABASES_PATTERNS = [
    r"(?:search|query)\s+(?:for\s+)?(.+?)\s+(?:data|datasets?|samples?)\s+(?:in|on|from)\s+(?:encode|geo|sra|databases?)",
    r"(?:search|query)\s+(?:in\s+)?(?:encode|geo|sra|ensembl|databases?)\s+(?:for)?\s*(.+)",
    r"(?:search|query)\s+(?:for\s+)(.+?)(?:\s+data|\s+datasets?)?$",
]


def search_databases_impl(query: str = None, include_tcga: bool = True) -> ToolResult:
    """
    Search ALL remote databases for datasets in parallel using DataDiscovery orchestrator.
    
    Searches: ENCODE, GEO, SRA, Ensembl, and optionally TCGA
    Uses parallel execution for speed and deduplicates results.
    
    Args:
        query: Search query (e.g., "human RNA-seq liver", "brain cancer methylation")
        include_tcga: Also search TCGA for cancer-related queries (default: True)
        
    Returns:
        ToolResult with deduplicated search results from all databases
    """
    if not query:
        return ToolResult(
            success=False,
            tool_name="search_databases",
            error="No search query provided",
            message="❌ Please specify what to search for (e.g., 'search for human RNA-seq liver data')"
        )
    
    try:
        # Use the DataDiscovery orchestrator for parallel multi-database search
        from workflow_composer.data.discovery import DataDiscovery
        
        results = []
        errors = []
        
        # Initialize the orchestrator (searches ENCODE, GEO, SRA, Ensembl in parallel)
        discovery = DataDiscovery(max_workers=4, timeout=30)
        
        # Perform federated search across ALL major databases (ENCODE, GEO, SRA)
        # Explicitly pass sources to ensure comprehensive search
        all_sources = ["encode", "geo"]  # SRA uses same adapter as GEO
        
        logger.info(f"Starting comprehensive search for: {query}")
        search_results = discovery.search(query, sources=all_sources, max_results=10)
        
        # Convert DatasetInfo objects to result dicts
        for dataset in search_results.datasets:
            results.append({
                "source": dataset.source.value.upper() if dataset.source else "UNKNOWN",
                "id": dataset.id,
                "title": dataset.title or dataset.id,
                "organism": dataset.organism or "",
                "assay": dataset.assay_type or "",
                "tissue": getattr(dataset, 'tissue', '') or "",
                "url": dataset.web_url or ""
            })
        
        # Add any search errors
        if search_results.errors:
            errors.extend(search_results.errors)
        
        # Also search TCGA for cancer-related queries
        cancer_keywords = ['cancer', 'tumor', 'carcinoma', 'adenocarcinoma', 'glioblastoma', 
                           'glioma', 'leukemia', 'lymphoma', 'melanoma', 'sarcoma', 'metastatic']
        is_cancer_query = any(kw in query.lower() for kw in cancer_keywords)
        
        if include_tcga and is_cancer_query:
            logger.info("Cancer-related query detected, also searching TCGA...")
            try:
                tcga_result = search_tcga_impl(
                    cancer_type=query,  # Let TCGA search parse it
                    data_type="methylation" if "methylation" in query.lower() else None
                )
                if tcga_result.success and tcga_result.data:
                    tcga_projects = tcga_result.data.get('projects', [])
                    for proj in tcga_projects[:5]:
                        results.append({
                            "source": "TCGA",
                            "id": proj.get('project_id', ''),
                            "title": proj.get('name', proj.get('project_id', '')),
                            "organism": "Homo sapiens",
                            "assay": proj.get('data_types', [''])[0] if proj.get('data_types') else '',
                            "tissue": proj.get('primary_site', ''),
                            "url": f"https://portal.gdc.cancer.gov/projects/{proj.get('project_id', '')}"
                        })
            except Exception as e:
                logger.debug(f"TCGA search failed: {e}")
                errors.append(f"TCGA: {e}")
        
        if results:
            # Deduplicate by ID
            seen_ids = set()
            unique_results = []
            for r in results:
                if r['id'] not in seen_ids:
                    seen_ids.add(r['id'])
                    unique_results.append(r)
            results = unique_results
            
            def get_url(r: dict) -> str:
                """Get URL from result or generate it."""
                if r.get('url'):
                    return r['url']
                source = r.get('source', '').upper()
                id = r.get('id', '')
                if source == "ENCODE":
                    return f"https://www.encodeproject.org/experiments/{id}"
                elif source == "GEO":
                    return f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={id}"
                elif source == "TCGA":
                    return f"https://portal.gdc.cancer.gov/projects/{id}"
                elif source == "SRA":
                    return f"https://www.ncbi.nlm.nih.gov/sra/{id}"
                elif source == "ENSEMBL":
                    return f"https://www.ensembl.org"
                return "#"
            
            # Group by source for better display
            sources_found = list(set(r['source'] for r in results))
            
            result_list = "\n".join([
                f"  - **{r['source']}**: [{r['id']}]({get_url(r)}) | {r.get('assay', '') or r.get('tissue', '')} | {(r['title'][:40] + '...') if len(r.get('title', '')) > 40 else r.get('title', '')}"
                for r in results[:15]
            ])
            
            sources_info = f"Searched: {', '.join(sources_found)}"
            if errors:
                sources_info += f" (errors: {len(errors)})"
            
            message = f"""🔍 Found **{len(results)} unique datasets** matching "{query}":

{sources_info}

{result_list}

💡 Say "download <ID>" to download a dataset."""
        else:
            message = f"""⚠️ No datasets found matching '{query}'.

**Tips for better results:**
- Include organism (human, mouse)
- Specify assay type (RNA-seq, ChIP-seq, ATAC-seq)
- Add tissue or cell line"""
        
        return ToolResult(
            success=True,
            tool_name="search_databases",
            data={"results": results, "query": query},
            message=message
        )
        
    except ImportError as e:
        return ToolResult(
            success=False,
            tool_name="search_databases",
            error=f"Discovery adapters not available: {e}",
            message="❌ Database search is not available."
        )
    except Exception as e:
        return ToolResult(
            success=False,
            tool_name="search_databases",
            error=str(e),
            message=f"❌ Search failed: {e}"
        )


# =============================================================================
# SEARCH_TCGA
# =============================================================================

SEARCH_TCGA_PATTERNS = [
    r"(?:search|find|look for)\s+(?:in\s+)?(?:tcga|gdc|cancer)\s+(?:for\s+)?(.+)",
    r"(?:search|find)\s+(.+?)\s+(?:in|on|from)\s+(?:tcga|gdc|cancer\s+portal)",
    r"(?:get|find)\s+(?:cancer|tumor)\s+data\s+(?:for|from)\s+(.+)",
]


def search_tcga_impl(query: str = None, cancer_type: str = None, data_type: str = None) -> ToolResult:
    """
    Search TCGA/GDC for cancer genomics data.
    
    Args:
        query: Cancer type or search query
        cancer_type: Specific cancer type (e.g., TCGA-BRCA, GBM, brain)
        data_type: Data type filter (e.g., methylation, RNA-seq, WXS)
        
    Returns:
        ToolResult with TCGA project information
    """
    # Combine query parameters - use cancer_type if provided, otherwise use query
    search_query = cancer_type or query or ""
    if data_type:
        search_query = f"{search_query} {data_type}".strip()
    
    # TCGA project mapping
    TCGA_PROJECTS = {
        "gbm": ("TCGA-GBM", "Glioblastoma Multiforme", "Brain"),
        "glioblastoma": ("TCGA-GBM", "Glioblastoma Multiforme", "Brain"),
        "brain cancer": ("TCGA-GBM", "Glioblastoma Multiforme", "Brain"),
        "lgg": ("TCGA-LGG", "Lower Grade Glioma", "Brain"),
        "luad": ("TCGA-LUAD", "Lung Adenocarcinoma", "Lung"),
        "lung": ("TCGA-LUAD", "Lung Adenocarcinoma", "Lung"),
        "brca": ("TCGA-BRCA", "Breast Cancer", "Breast"),
        "breast": ("TCGA-BRCA", "Breast Cancer", "Breast"),
        "coad": ("TCGA-COAD", "Colon Adenocarcinoma", "Colon"),
        "colon": ("TCGA-COAD", "Colon Adenocarcinoma", "Colon"),
        "prad": ("TCGA-PRAD", "Prostate Adenocarcinoma", "Prostate"),
        "prostate": ("TCGA-PRAD", "Prostate Adenocarcinoma", "Prostate"),
        "thca": ("TCGA-THCA", "Thyroid Carcinoma", "Thyroid"),
        "thyroid": ("TCGA-THCA", "Thyroid Carcinoma", "Thyroid"),
        "ov": ("TCGA-OV", "Ovarian Cancer", "Ovary"),
        "ovarian": ("TCGA-OV", "Ovarian Cancer", "Ovary"),
        "kirc": ("TCGA-KIRC", "Kidney Renal Clear Cell Carcinoma", "Kidney"),
        "kidney": ("TCGA-KIRC", "Kidney Renal Clear Cell Carcinoma", "Kidney"),
        "lihc": ("TCGA-LIHC", "Liver Hepatocellular Carcinoma", "Liver"),
        "liver": ("TCGA-LIHC", "Liver Hepatocellular Carcinoma", "Liver"),
        "skcm": ("TCGA-SKCM", "Skin Cutaneous Melanoma", "Skin"),
        "melanoma": ("TCGA-SKCM", "Skin Cutaneous Melanoma", "Skin"),
        "paad": ("TCGA-PAAD", "Pancreatic Adenocarcinoma", "Pancreas"),
        "pancreatic": ("TCGA-PAAD", "Pancreatic Adenocarcinoma", "Pancreas"),
    }
    
    if not search_query:
        # Show available projects
        message = """# 🔬 TCGA/GDC Cancer Data Portal

**Available Cancer Types:**

| Project | Cancer Type | Organ |
|---------|-------------|-------|
| TCGA-GBM | Glioblastoma | Brain |
| TCGA-LGG | Lower Grade Glioma | Brain |
| TCGA-BRCA | Breast Cancer | Breast |
| TCGA-LUAD | Lung Adenocarcinoma | Lung |
| TCGA-COAD | Colon Adenocarcinoma | Colon |
| TCGA-PRAD | Prostate Adenocarcinoma | Prostate |
| TCGA-KIRC | Kidney Clear Cell | Kidney |
| TCGA-LIHC | Liver Hepatocellular | Liver |

**Usage:**
- `search TCGA for GBM` - Find glioblastoma data
- `search TCGA for breast cancer` - Find breast cancer data
- `search TCGA for methylation` - Find methylation studies

🔗 [GDC Data Portal](https://portal.gdc.cancer.gov/)
"""
        return ToolResult(
            success=True,
            tool_name="search_tcga",
            data={"projects": list(TCGA_PROJECTS.keys())},
            message=message
        )
    
    query_lower = search_query.lower()
    matched_projects = []
    
    # Match any cancer type mentioned in the query
    for key, (project_id, cancer_name, organ) in TCGA_PROJECTS.items():
        if key in query_lower or organ.lower() in query_lower:
            matched_projects.append({
                "project_id": project_id,
                "name": cancer_name,
                "primary_site": organ,
                "data_types": ["WGS", "WXS", "RNA-seq", "Methylation", "Clinical"]
            })
    
    if matched_projects:
        # Remove duplicates (multiple keywords can match same project)
        seen = set()
        unique_projects = []
        for p in matched_projects:
            if p["project_id"] not in seen:
                seen.add(p["project_id"])
                unique_projects.append(p)
        matched_projects = unique_projects
        
        # Format message
        project_rows = []
        for p in matched_projects:
            portal_url = f"https://portal.gdc.cancer.gov/projects/{p['project_id']}"
            project_rows.append(f"| [{p['project_id']}]({portal_url}) | {p['name']} | {p['primary_site']} |")
        
        project_table = "\n".join(project_rows)
        data_type_info = f"\n**Filtered by:** {data_type}" if data_type else ""
        
        message = f"""# 🔬 TCGA/GDC Cancer Data

**Query:** {search_query}{data_type_info}

| Project | Cancer Type | Organ |
|---------|-------------|-------|
{project_table}

### 💡 Download Instructions:

1. Visit the [GDC Portal](https://portal.gdc.cancer.gov/projects/{matched_projects[0]['project_id']})
2. Add files to your cart based on data type
3. Download manifest and use GDC client:
   ```bash
   gdc-client download -m gdc_manifest.txt
   ```

📊 **Data types available:** WGS, WXS, RNA-seq, Methylation, Clinical
"""
        return ToolResult(
            success=True,
            tool_name="search_tcga",
            data={"query": search_query, "matched": True, "projects": matched_projects},
            message=message
        )
    else:
        message = f"""# 🔬 TCGA/GDC Cancer Data

**Query:** {search_query}

No specific cancer type matched. Try:
- `brain cancer` or `GBM` for glioblastoma
- `breast cancer` or `BRCA` for breast
- `lung cancer` or `LUAD` for lung

🔗 [Browse All Projects](https://portal.gdc.cancer.gov/projects)
"""
    
    return ToolResult(
        success=True,
        tool_name="search_tcga",
        data={"query": search_query, "matched": matched_project is not None},
        message=message
    )


# =============================================================================
# DESCRIBE_FILES
# =============================================================================

DESCRIBE_FILES_PATTERNS = [
    r"(?:describe|inspect|examine|analyze|check)\s+(?:the\s+)?(?:files?|data)\s+(?:in|at|from)\s+['\"]?([\/~][^\s'\"\?]+)['\"]?",
    r"(?:give me\s+)?(?:details?|info|information|summary|stats|statistics|metadata)\s+(?:of|about|for)\s+(?:the\s+)?(?:files?|data)\s*(?:in|at|from)?\s*['\"]?([\/~][^\s'\"\?]+)?['\"]?",
    r"(?:what(?:'s| is| are))\s+(?:in\s+)?(?:these|the|those)\s+(?:files?|data\s+files?)",
]


def describe_files_impl(path: str = None) -> ToolResult:
    """
    Get detailed information about files in a directory.
    
    Args:
        path: Path to file or directory
        
    Returns:
        ToolResult with file details
    """
    import os
    
    if not path:
        default_paths = [
            Path("/scratch/sdodl001/BioPipelines/data"),
            Path.home() / "BioPipelines" / "data",
            Path.cwd() / "data",
        ]
        for p in default_paths:
            if p.exists():
                path = str(p)
                break
    
    if not path:
        return ToolResult(
            success=False,
            tool_name="describe_files",
            error="No path specified",
            message="❌ Please specify a path to describe"
        )
    
    path = Path(path).expanduser().resolve()
    
    if not path.exists():
        return ToolResult(
            success=False,
            tool_name="describe_files",
            error=f"Path not found: {path}",
            message=f"❌ Path not found: `{path}`"
        )
    
    try:
        if path.is_file():
            size = path.stat().st_size
            size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024*1024 else f"{size / 1024:.1f} KB"
            
            message = f"""📄 **File:** `{path.name}`

| Property | Value |
|----------|-------|
| **Path** | `{path}` |
| **Size** | {size_str} |
| **Type** | {path.suffix or 'No extension'} |
"""
        else:
            # Directory
            files = list(path.iterdir())
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            size_str = f"{total_size / 1024 / 1024:.1f} MB" if total_size > 1024*1024 else f"{total_size / 1024:.1f} KB"
            
            file_types = {}
            for f in files:
                if f.is_file():
                    ext = f.suffix or "no_ext"
                    file_types[ext] = file_types.get(ext, 0) + 1
            
            type_str = ", ".join([f"{ext}: {count}" for ext, count in sorted(file_types.items())])
            
            message = f"""📁 **Directory:** `{path}`

| Property | Value |
|----------|-------|
| **Files** | {len([f for f in files if f.is_file()])} |
| **Folders** | {len([f for f in files if f.is_dir()])} |
| **Total Size** | {size_str} |
| **File Types** | {type_str or 'N/A'} |
"""
        
        return ToolResult(
            success=True,
            tool_name="describe_files",
            data={"path": str(path)},
            message=message
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            tool_name="describe_files",
            error=str(e),
            message=f"❌ Error describing files: {e}"
        )


# =============================================================================
# VALIDATE_DATASET
# =============================================================================

VALIDATE_DATASET_PATTERNS = [
    r"(?:validate|verify|check)\s+(?:the\s+)?(?:downloaded\s+)?(?:data|dataset|files?)(?:\s+(?:in|at|from)\s+['\"]?([^\s'\"\?]+)['\"]?)?",
    r"(?:is\s+)?(?:the\s+)?(?:data|dataset)\s+(?:valid|correct|good|real|actual)",
]


def validate_dataset_impl(path_or_id: str = None) -> ToolResult:
    """
    Validate a downloaded dataset.
    
    Args:
        path_or_id: Path to data or dataset ID
        
    Returns:
        ToolResult with validation results
    """
    if not path_or_id:
        # Look for recent downloads
        default_paths = [
            Path("/scratch/sdodl001/BioPipelines/data/raw"),
            Path.home() / "BioPipelines" / "data" / "raw",
        ]
        for p in default_paths:
            if p.exists() and any(p.iterdir()):
                path_or_id = str(p)
                break
    
    if not path_or_id:
        return ToolResult(
            success=False,
            tool_name="validate_dataset",
            error="No path or ID specified",
            message="❌ Please specify a path or dataset ID to validate"
        )
    
    path = Path(path_or_id).expanduser().resolve()
    
    if not path.exists():
        return ToolResult(
            success=False,
            tool_name="validate_dataset",
            error=f"Path not found: {path}",
            message=f"❌ Path not found: `{path}`"
        )
    
    try:
        issues = []
        warnings = []
        
        # Check for FASTQ files
        fastq_files = list(path.rglob("*.fastq*")) + list(path.rglob("*.fq*"))
        
        if not fastq_files:
            issues.append("No FASTQ files found")
        else:
            # Check for HTML error pages
            for f in fastq_files[:5]:
                try:
                    with open(f, 'rb') as fp:
                        header = fp.read(100)
                        if b'<!DOCTYPE' in header or b'<html' in header.lower():
                            issues.append(f"HTML error page instead of data: {f.name}")
                except Exception:
                    pass
        
        if issues:
            message = f"""❌ **Validation Failed**

**Path:** `{path}`

**Issues:**
""" + "\n".join([f"- ⚠️ {issue}" for issue in issues])
        else:
            message = f"""✅ **Validation Passed**

**Path:** `{path}`
**FASTQ Files:** {len(fastq_files)}

Data looks valid and ready for analysis!
"""
        
        return ToolResult(
            success=len(issues) == 0,
            tool_name="validate_dataset",
            data={"path": str(path), "issues": issues, "fastq_count": len(fastq_files)},
            message=message
        )
        
    except Exception as e:
        return ToolResult(
            success=False,
            tool_name="validate_dataset",
            error=str(e),
            message=f"❌ Validation error: {e}"
        )
