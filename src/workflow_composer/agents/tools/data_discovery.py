"""
Data Discovery Tools
====================

Tools for scanning local data and searching remote databases.

Integrates with:
- SemanticCache: Caches search results with TTL and similarity matching
- PrefetchManager: Background prefetch of top search result details
- Observability: Distributed tracing for debugging
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ToolResult, ToolParameter
from .registry import ToolRegistry

logger = logging.getLogger(__name__)

# Optional integrations (graceful degradation if not available)
_semantic_cache = None
_prefetch_manager = None

def _get_semantic_cache():
    """Get the semantic cache instance (lazy load)."""
    global _semantic_cache
    if _semantic_cache is None:
        try:
            from workflow_composer.infrastructure.semantic_cache import get_cache
            _semantic_cache = get_cache()
        except ImportError:
            logger.debug("SemanticCache not available")
            _semantic_cache = False  # Mark as unavailable
    return _semantic_cache if _semantic_cache else None

def _get_prefetch_manager():
    """Get the prefetch manager instance (lazy load)."""
    global _prefetch_manager
    if _prefetch_manager is None:
        try:
            from .prefetch import get_prefetch_manager
            _prefetch_manager = get_prefetch_manager()
        except ImportError:
            logger.debug("PrefetchManager not available")
            _prefetch_manager = False  # Mark as unavailable
    return _prefetch_manager if _prefetch_manager else None


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
    data_type: str = None,
    scanner=None,
    manifest=None,
) -> ToolResult:
    """
    Scan a directory for FASTQ files and display hierarchical folder structure.
    
    Args:
        path: Directory path to scan. Defaults to current directory.
        data_type: Optional filter for specific data type (e.g., "scRNA-seq", "RNA-seq", "ChIP-seq")
        scanner: DataScanner instance
        manifest: DataManifest instance
        
    Returns:
        ToolResult with discovered samples organized by folder
    """
    if scanner is None:
        return ToolResult(
            success=False,
            tool_name="scan_data",
            error="Scanner not available",
            message="❌ Data scanner is not available. Please check installation."
        )
    
    # Normalize data_type filter
    data_type_filter = None
    if data_type:
        data_type_filter = _normalize_data_type(data_type)
    
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
        
        # Filter by data type if specified
        if data_type_filter and samples:
            filtered_samples = []
            for sample in samples:
                sample_type = _infer_data_type_from_path(
                    str(sample.fastq_1) if hasattr(sample, 'fastq_1') else "",
                    sample.sample_id
                )
                if sample_type and sample_type.lower() == data_type_filter.lower():
                    filtered_samples.append(sample)
            
            # If filtering returns no results, report that
            if not filtered_samples:
                return ToolResult(
                    success=True,
                    tool_name="scan_data",
                    data={"samples": [], "path": str(scan_path), "count": 0, "filter": data_type_filter},
                    message=f"⚠️ **No {data_type_filter} samples found** in `{scan_path}`\n\n"
                            f"The scan found {len(samples)} total samples, but none matched '{data_type_filter}'.\n\n"
                            f"**Available data types**: {', '.join(sorted(set(_infer_data_type_from_path(str(s.fastq_1), s.sample_id) or 'Unknown' for s in samples)))}\n\n"
                            f"💡 Try: *\"What data types do we have?\"* to see all available data.",
                )
            samples = filtered_samples
        
        # Add samples to manifest
        if manifest and samples:
            for sample in samples:
                manifest.add_sample(sample)
        
        # Build hierarchical folder summary
        message = _build_folder_summary(scan_path, samples, result, data_type_filter)
        
        return ToolResult(
            success=True,
            tool_name="scan_data",
            data={
                "samples": samples,
                "path": str(scan_path),
                "count": len(samples),
                "filter": data_type_filter
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


def _normalize_data_type(data_type: str) -> str:
    """Normalize data type string to standard format."""
    dt_lower = data_type.lower().strip()
    # Remove extra spaces
    dt_lower = " ".join(dt_lower.split())
    
    # Map common variations to standard names
    type_map = {
        "scrna": "scRNA-seq",
        "scrna-seq": "scRNA-seq",
        "scrnaseq": "scRNA-seq",
        "sc-rna-seq": "scRNA-seq",
        "sc rna seq": "scRNA-seq",
        "single cell": "scRNA-seq",
        "single-cell": "scRNA-seq",
        "singlecell": "scRNA-seq",
        "single cell rna": "scRNA-seq",
        "single cell rna seq": "scRNA-seq",
        "single-cell rna-seq": "scRNA-seq",
        "10x": "scRNA-seq",
        "rna": "RNA-seq",
        "rna-seq": "RNA-seq",
        "rnaseq": "RNA-seq",
        "rna seq": "RNA-seq",
        "chip": "ChIP-seq",
        "chip-seq": "ChIP-seq",
        "chipseq": "ChIP-seq",
        "chip seq": "ChIP-seq",
        "atac": "ATAC-seq",
        "atac-seq": "ATAC-seq",
        "atacseq": "ATAC-seq",
        "atac seq": "ATAC-seq",
        "methyl": "Methylation",
        "methylation": "Methylation",
        "wgbs": "Methylation",
        "bisulfite": "Methylation",
        "wgs": "WGS",
        "whole genome": "WGS",
        "wes": "WES",
        "exome": "WES",
        "hic": "Hi-C",
        "hi-c": "Hi-C",
        "hi c": "Hi-C",
    }
    
    for key, value in type_map.items():
        if key in dt_lower:
            return value
    
    return data_type  # Return as-is if no match


def _build_folder_summary(scan_path: Path, samples: list, scan_result, data_type_filter: str = None) -> str:
    """
    Build an intelligent folder summary that:
    1. Separates INPUT data from OUTPUT/work directories
    2. Groups by data type (RNA-seq, ChIP-seq, etc.)
    3. Shows meaningful folder names (skips hash directories)
    4. Includes file sizes to distinguish real data from test files
    """
    from collections import defaultdict
    import os
    
    if not samples:
        return _build_empty_folder_summary(scan_path)
    
    # Categorize folders
    input_folders = {}   # Actual input data (raw/, data/, etc.)
    output_folders = {}  # Pipeline outputs (work/, results/, test_*)
    
    # Folders that are clearly pipeline outputs or temporary
    output_patterns = ['work', 'results', 'output', 'tmp', 'temp', 'cache', 
                       'nextflow', '.nextflow', 'test_', 'logs']
    
    # Group samples by folder with metadata
    folder_data = defaultdict(lambda: {
        "paired": 0, "single": 0, "total_size_bytes": 0,
        "data_types": set(), "sample_ids": [], "subfolders": defaultdict(int)
    })
    
    for sample in samples:
        sample_path = Path(sample.fastq_1) if hasattr(sample, 'fastq_1') else None
        if not sample_path:
            continue
        
        try:
            rel_path = sample_path.relative_to(scan_path)
            parts = rel_path.parts
            top_folder = parts[0] if len(parts) > 1 else "."
            
            # Get file size
            try:
                file_size = sample_path.stat().st_size
                if hasattr(sample, 'fastq_2') and sample.fastq_2:
                    file_size += Path(sample.fastq_2).stat().st_size
            except:
                file_size = 0
            
            # Determine paired/single
            is_paired = (hasattr(sample, 'is_paired') and sample.is_paired) or \
                       (hasattr(sample, 'fastq_2') and sample.fastq_2)
            
            # Detect data type from path or sample name
            data_type = _infer_data_type_from_path(str(sample_path), sample.sample_id)
            if data_type:
                folder_data[top_folder]["data_types"].add(data_type)
            
            # Update stats
            if is_paired:
                folder_data[top_folder]["paired"] += 1
            else:
                folder_data[top_folder]["single"] += 1
            
            folder_data[top_folder]["total_size_bytes"] += file_size
            folder_data[top_folder]["sample_ids"].append(sample.sample_id)
            
            # Track meaningful subfolders (skip hash-like names)
            if len(parts) > 2:
                sub = parts[1]
                if not _is_hash_folder(sub):
                    folder_data[top_folder]["subfolders"][sub] += 1
                    
        except ValueError:
            continue
    
    # Categorize folders as input vs output
    for folder, stats in folder_data.items():
        is_output = any(p in folder.lower() for p in output_patterns)
        # Also check if it's mostly small files (likely intermediate)
        avg_size = stats["total_size_bytes"] / max(1, stats["paired"] + stats["single"])
        is_small = avg_size < 1_000_000  # Less than 1MB average = likely not real FASTQ
        
        if is_output or (is_small and folder not in ['data', 'raw', '.']):
            output_folders[folder] = stats
        else:
            input_folders[folder] = stats
    
    # Build output message
    if data_type_filter:
        lines = [f"📁 **{data_type_filter} Data** in `{scan_path}`\n"]
    else:
        lines = [f"📁 **Data Overview**: `{scan_path}`\n"]
    
    # Summary
    total_input = sum(s["paired"] + s["single"] for s in input_folders.values())
    total_output = sum(s["paired"] + s["single"] for s in output_folders.values())
    total_size = sum(s["total_size_bytes"] for s in input_folders.values())
    
    if data_type_filter:
        lines.append(f"**Found**: {total_input} {data_type_filter} samples ({_format_size(total_size)})")
    else:
        lines.append(f"**Input Data**: {total_input} samples ({_format_size(total_size)})")
    if total_output > 0:
        lines.append(f"**Pipeline Outputs**: {total_output} intermediate files (in work/results folders)")
    lines.append("")
    
    # Show INPUT folders prominently
    if input_folders:
        if data_type_filter:
            lines.append(f"### 📥 {data_type_filter} Samples")
        else:
            lines.append("### 📥 Input Data")
        lines.append("")
        
        for folder in sorted(input_folders.keys(), key=lambda f: -(input_folders[f]["paired"] + input_folders[f]["single"])):
            stats = input_folders[folder]
            _add_folder_section(lines, folder, stats, show_details=True)
    
    # Collect all data types
    all_data_types = set()
    for stats in folder_data.values():
        all_data_types.update(stats["data_types"])
    
    # Show OUTPUT folders collapsed (just summary)
    if output_folders:
        lines.append("")
        lines.append("### ⚙️ Pipeline Outputs (intermediate files)")
        output_types = set()
        for stats in output_folders.values():
            output_types.update(stats["data_types"])
        
        if output_types:
            lines.append(f"   Contains: {', '.join(sorted(output_types))} workflow outputs")
        lines.append(f"   Location: `work/`, `results/`, `test_*/` folders")
        lines.append(f"   *These are intermediate files, not input data*")
    
    # Suggestions
    lines.append("")
    lines.append("---")
    lines.append(_generate_smart_suggestions(input_folders, all_data_types, total_input))
    
    return "\n".join(lines)


def _infer_data_type_from_path(path: str, sample_id: str) -> str:
    """Infer data type from file path or sample name."""
    combined = (path + " " + sample_id).lower()
    
    # Check for explicit data type markers
    type_patterns = [
        (["rna-seq", "rnaseq", "rna_seq", "transcriptome"], "RNA-seq"),
        (["chip-seq", "chipseq", "chip_seq", "h3k", "h3k4", "h3k27", "h3k9"], "ChIP-seq"),
        (["atac-seq", "atacseq", "atac_seq", "atac"], "ATAC-seq"),
        (["methyl", "bisulfite", "wgbs", "rrbs"], "Methylation"),
        (["wgs", "whole-genome", "wholegenome"], "WGS"),
        (["wes", "exome", "whole-exome"], "WES"),
        (["hic", "hi-c"], "Hi-C"),
        (["scrna", "single-cell", "singlecell", "10x"], "scRNA-seq"),
        (["cut&run", "cutrun", "cut-run", "cut&tag", "cuttag"], "CUT&RUN"),
    ]
    
    for patterns, dtype in type_patterns:
        if any(p in combined for p in patterns):
            return dtype
    
    return None


def _is_hash_folder(name: str) -> bool:
    """Check if folder name looks like a hash/UUID (not meaningful to display)."""
    # Hash-like: all hex chars, 2 chars, long random strings
    if len(name) <= 2 and all(c in '0123456789abcdef' for c in name.lower()):
        return True
    if len(name) >= 20 and all(c in '0123456789abcdef-_' for c in name.lower()):
        return True
    # Nextflow work directory pattern
    if len(name) == 2:
        return True
    return False


def _format_size(size_bytes: int) -> str:
    """Format bytes to human readable."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.1f} MB"
    else:
        return f"{size_bytes/1024**3:.1f} GB"


def _add_folder_section(lines: list, folder: str, stats: dict, show_details: bool = True):
    """Add a folder section to the output."""
    emoji = _get_folder_emoji(folder)
    folder_display = folder if folder != "." else "(root)"
    total_samples = stats["paired"] + stats["single"]
    size_str = _format_size(stats["total_size_bytes"])
    
    # Data type tag
    type_tag = ""
    if stats["data_types"]:
        type_tag = f" **[{', '.join(sorted(stats['data_types']))}]**"
    
    lines.append(f"{emoji} **{folder_display}/**{type_tag}")
    lines.append(f"   {total_samples} samples ({stats['paired']} paired, {stats['single']} single) • {size_str}")
    
    if show_details and stats["subfolders"]:
        # Show meaningful subfolders
        sorted_subs = sorted(stats["subfolders"].items(), key=lambda x: -x[1])[:4]
        for sub_name, count in sorted_subs:
            lines.append(f"   └── `{sub_name}/`: {count} samples")
        
        remaining = len(stats["subfolders"]) - 4
        if remaining > 0:
            lines.append(f"   └── ... and {remaining} more")


def _generate_smart_suggestions(input_folders: dict, data_types: set, total_input: int) -> str:
    """Generate contextual suggestions based on data."""
    suggestions = []
    
    if data_types:
        type_list = ", ".join(sorted(data_types))
        suggestions.append(f"📊 **Detected data types**: {type_list}")
        
        # Suggest workflows for each type
        suggestions.append("")
        suggestions.append("💡 **Suggested next steps:**")
        
        for dtype in sorted(data_types):
            if dtype == "RNA-seq":
                suggestions.append(f'   • *"Create an RNA-seq differential expression workflow"*')
            elif dtype == "ChIP-seq":
                suggestions.append(f'   • *"Create a ChIP-seq peak calling workflow"*')
            elif dtype == "ATAC-seq":
                suggestions.append(f'   • *"Create an ATAC-seq accessibility analysis"*')
            elif dtype == "Methylation":
                suggestions.append(f'   • *"Create a methylation analysis workflow"*')
            elif dtype == "scRNA-seq":
                suggestions.append(f'   • *"Create a single-cell RNA-seq clustering workflow"*')
            else:
                suggestions.append(f'   • *"Create a {dtype} analysis workflow"*')
    elif total_input > 0:
        suggestions.append(f"📊 **Found {total_input} samples** ready for analysis")
        suggestions.append("")
        suggestions.append("💡 **To determine data type:**")
        suggestions.append('   • *"What type of sequencing data is in sample X?"*')
        suggestions.append('   • *"Describe the samples in the raw folder"*')
    else:
        suggestions.append("⚠️ No input data found. Upload FASTQ files or specify a path.")
    
    return "\n".join(suggestions)


def _build_empty_folder_summary(scan_path: Path) -> str:
    """Build a folder structure summary when no FASTQ files found."""
    from collections import defaultdict
    
    lines = [f"📁 **Scanning**: `{scan_path}`\n"]
    lines.append("⚠️ **No FASTQ samples found**\n")
    
    # Show what folders exist
    try:
        subdirs = [d for d in scan_path.iterdir() if d.is_dir()]
        if subdirs:
            lines.append("**Existing folders:**")
            for d in sorted(subdirs)[:10]:
                # Count files in each
                file_count = sum(1 for _ in d.rglob("*") if _.is_file())
                lines.append(f"  📂 `{d.name}/` ({file_count} files)")
            if len(subdirs) > 10:
                lines.append(f"  ... and {len(subdirs) - 10} more folders")
        
        # Check for other data types
        other_files = list(scan_path.rglob("*.bam"))[:5]
        other_files += list(scan_path.rglob("*.vcf*"))[:5]
        other_files += list(scan_path.rglob("*.bed"))[:5]
        
        if other_files:
            lines.append("\n**Other data files found:**")
            file_types = defaultdict(int)
            for f in scan_path.rglob("*"):
                if f.is_file():
                    ext = "".join(f.suffixes[-2:]) if len(f.suffixes) > 1 else f.suffix
                    if ext in ['.bam', '.vcf', '.vcf.gz', '.bed', '.bigwig', '.bw', '.gtf', '.gff']:
                        file_types[ext] += 1
            
            for ext, count in sorted(file_types.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  • {ext}: {count} files")
                
    except PermissionError:
        lines.append("  (Permission denied to list folders)")
    
    lines.append("\n💡 **Tip**: Upload FASTQ files or specify a different path.")
    
    return "\n".join(lines)


def _get_folder_emoji(folder_name: str) -> str:
    """Get an emoji based on the folder name/data type."""
    name_lower = folder_name.lower()
    
    emoji_map = {
        "rna": "🧬",
        "chip": "🎯",
        "atac": "🔓",
        "wgs": "🧬",
        "wes": "🔬",
        "methyl": "⚗️",
        "hic": "🔗",
        "scrna": "🦠",
        "single": "🦠",
        "raw": "📥",
        "process": "⚙️",
        "result": "📊",
        "reference": "📚",
        "genome": "🧬",
        "annotation": "📝",
    }
    
    for key, emoji in emoji_map.items():
        if key in name_lower:
            return emoji
    
    return "📂"


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
    
    Integrations:
    - SemanticCache: Checks for cached similar queries first
    - PrefetchManager: Background prefetch of top result details
    
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
    
    # Check semantic cache first
    cache = _get_semantic_cache()
    cache_key = f"search:{query.lower().strip()}"
    if cache:
        try:
            cached = cache.get(cache_key)
            if cached:
                logger.info(f"Cache hit for search query: {query[:50]}...")
                cached_result = cached.get("result")
                if cached_result:
                    cached_result["message"] = cached_result.get("message", "") + "\n\n_📦 Cached result_"
                    return ToolResult(**cached_result)
        except Exception as e:
            logger.debug(f"Cache lookup failed: {e}")
    
    try:
        # Use the DataDiscovery orchestrator for parallel multi-database search
        from workflow_composer.data.discovery import DataDiscovery
        
        results = []
        errors = []
        
        # Detect if this is a cancer-related query
        cancer_keywords = ['cancer', 'tumor', 'carcinoma', 'adenocarcinoma', 'glioblastoma', 
                           'glioma', 'leukemia', 'lymphoma', 'melanoma', 'sarcoma', 'metastatic',
                           'gbm', 'brca', 'luad', 'lihc', 'tcga', 'gdc']
        is_cancer_query = any(kw in query.lower() for kw in cancer_keywords)
        
        # Initialize the orchestrator (searches all databases in parallel)
        discovery = DataDiscovery(max_workers=4, timeout=30)
        
        # Perform federated search across ALL major databases
        # Include GDC for cancer queries
        all_sources = ["encode", "geo"]  # SRA uses same adapter as GEO
        if is_cancer_query and include_tcga:
            all_sources.append("gdc")  # Add GDC for cancer-related queries
        
        logger.info(f"Starting comprehensive search for: {query} (sources: {all_sources})")
        search_results = discovery.search(query, sources=all_sources, max_results=10)
        
        # Convert DatasetInfo objects to result dicts (including file metadata)
        for dataset in search_results.datasets:
            source_name = dataset.source.value.upper() if dataset.source else "UNKNOWN"
            # Map GDC to TCGA for display consistency
            if source_name == "GDC":
                source_name = "TCGA"
            
            # Get file metadata
            file_count = dataset.file_count or len(dataset.download_urls)
            size_str = dataset.total_size_human if hasattr(dataset, 'total_size_human') else "Unknown"
            
            # Get file types with counts and sizes
            file_type_info = {}  # {format: {"count": n, "size": bytes}}
            for url in dataset.download_urls:
                fmt = (getattr(url, 'file_format', None) or 'OTHER').upper()
                size = getattr(url, 'size_bytes', 0) or 0
                if fmt not in file_type_info:
                    file_type_info[fmt] = {"count": 0, "size": 0}
                file_type_info[fmt]["count"] += 1
                file_type_info[fmt]["size"] += size
            
            # Format file breakdown
            def format_size(bytes):
                if bytes == 0:
                    return ""
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if bytes < 1024:
                        return f"{bytes:.1f}{unit}"
                    bytes /= 1024
                return f"{bytes:.1f}TB"
            
            file_breakdown = []
            for fmt, info in sorted(file_type_info.items(), key=lambda x: -x[1]["size"]):
                size_str_fmt = format_size(info["size"])
                file_breakdown.append({
                    "format": fmt,
                    "count": info["count"],
                    "size": size_str_fmt,
                    "size_bytes": info["size"]
                })
            
            results.append({
                "source": source_name,
                "id": dataset.id,
                "title": dataset.title or dataset.id,
                "organism": dataset.organism or "",
                "assay": dataset.assay_type or "",
                "tissue": getattr(dataset, 'tissue', '') or "",
                "url": dataset.web_url or "",
                "file_count": file_count,
                "total_size": size_str,
                "file_types": [f["format"] for f in file_breakdown[:5]],
                "file_breakdown": file_breakdown[:5],  # Detailed breakdown
            })
        
        # Add any search errors
        if search_results.errors:
            errors.extend(search_results.errors)
        
        # Note: GDC/TCGA is now searched via the orchestrator when is_cancer_query=True
        
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
                elif source in ("TCGA", "GDC"):
                    return f"https://portal.gdc.cancer.gov/projects/{id}"
                elif source == "SRA":
                    return f"https://www.ncbi.nlm.nih.gov/sra/{id}"
                elif source == "ENSEMBL":
                    return f"https://www.ensembl.org"
                return "#"
            
            # Group by source for better display
            sources_found = list(set(r['source'] for r in results))
            
            def format_result(r):
                """Format a single result with file type breakdown."""
                assay_info = r.get('assay', '') or r.get('tissue', '')
                title = (r['title'][:30] + '...') if len(r.get('title', '')) > 30 else r.get('title', '')
                
                # Main line
                size_info = f" ({r['total_size']})" if r.get('total_size') and r['total_size'] != 'Unknown' else ""
                main_line = f"**{r['source']}**: [{r['id']}]({get_url(r)}) | {assay_info}{size_info}"
                
                # File breakdown line
                breakdown = r.get('file_breakdown', [])
                if breakdown:
                    parts = []
                    for fb in breakdown[:4]:
                        if fb['size']:
                            parts.append(f"{fb['count']}× {fb['format']} ({fb['size']})")
                        else:
                            parts.append(f"{fb['count']}× {fb['format']}")
                    file_line = f"  └─ {' | '.join(parts)}"
                    return f"{main_line}\n{file_line}"
                
                return main_line
            
            result_list = "\n".join([format_result(r) for r in results[:12]])
            
            sources_info = f"Searched: {', '.join(sources_found)}"
            if errors:
                sources_info += f" (errors: {len(errors)})"
            
            message = f"""🔍 Found **{len(results)} unique datasets** matching "{query}":

{sources_info}

{result_list}

---
**What's next?**
- 📋 `show details for <ID>` - View metadata, sample info, and file list
- 📥 `download <ID>` - Download a single dataset
- 📥 `download all` - Preview all datasets before downloading
- 🔍 `refine search: <more specific terms>` - Narrow results"""
        else:
            message = f"""⚠️ No datasets found matching '{query}'.

**Tips for better results:**
- Include organism (human, mouse)
- Specify assay type (RNA-seq, ChIP-seq, ATAC-seq)
- Add tissue or cell line
- For cancer data, include: cancer, tumor, TCGA, GBM, BRCA, etc."""
        
        # Build the result
        tool_result = ToolResult(
            success=True,
            tool_name="search_databases",
            data={"results": results, "query": query},
            message=message
        )
        
        # Cache the successful result
        if cache and results:
            try:
                cache.set(cache_key, {
                    "result": {
                        "success": True,
                        "tool_name": "search_databases",
                        "data": {"results": results, "query": query},
                        "message": message,
                    }
                }, ttl=3600)  # 1 hour TTL
                logger.debug(f"Cached search results for: {query[:50]}...")
            except Exception as e:
                logger.debug(f"Cache storage failed: {e}")
        
        # Trigger background prefetch for top results
        prefetch = _get_prefetch_manager()
        if prefetch and results:
            try:
                # Extract dataset IDs for prefetching
                top_ids = [r['id'] for r in results[:3]]
                sources = [r['source'] for r in results[:3]]
                import asyncio
                asyncio.create_task(
                    prefetch.prefetch_dataset_details(top_ids, sources)
                ) if asyncio.get_event_loop().is_running() else None
                logger.debug(f"Triggered prefetch for: {top_ids}")
            except Exception as e:
                logger.debug(f"Prefetch trigger failed: {e}")
        
        return tool_result
        
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
# GET_DATASET_DETAILS
# =============================================================================

GET_DATASET_DETAILS_PATTERNS = [
    r"(?:show|get|view|display)\s+(?:the\s+)?(?:details?|info|information|metadata)\s+(?:for|of|about)\s+(GSE\d+|ENCSR[A-Z0-9]+|TCGA-[A-Z]+|[a-f0-9-]{36})",
    r"(?:what(?:'s| is| are))\s+(?:in\s+)?(GSE\d+|ENCSR[A-Z0-9]+|TCGA-[A-Z]+|[a-f0-9-]{36})",
    r"(?:describe|inspect|examine)\s+(GSE\d+|ENCSR[A-Z0-9]+|TCGA-[A-Z]+|[a-f0-9-]{36})",
    r"(?:more\s+)?(?:details?|info)\s+(?:on|about)\s+(GSE\d+|ENCSR[A-Z0-9]+|TCGA-[A-Z]+|[a-f0-9-]{36})",
    r"(GSE\d+|ENCSR[A-Z0-9]+)\s+(?:details?|info|metadata)",
]


def get_dataset_details_impl(dataset_id: str = None) -> ToolResult:
    """
    Get detailed metadata and file information for a dataset.
    
    Args:
        dataset_id: Dataset ID (GSE*, ENCSR*, TCGA-*, or GDC UUID)
        
    Returns:
        ToolResult with detailed dataset information
    """
    if not dataset_id:
        return ToolResult(
            success=False,
            tool_name="get_dataset_details",
            error="No dataset ID provided",
            message="❌ Please specify a dataset ID (e.g., `show details for ENCSR123ABC`)"
        )
    
    dataset_id = dataset_id.strip()
    
    # Normalize case for known ID formats
    if dataset_id.upper().startswith(("GSE", "ENCSR", "TCGA")):
        dataset_id = dataset_id.upper()
    
    # Determine source from ID format
    if dataset_id.startswith("GSE"):
        source = "geo"
    elif dataset_id.startswith("ENCSR"):
        source = "encode"
    elif dataset_id.startswith("TCGA-"):
        source = "gdc"
    elif "-" in dataset_id and len(dataset_id) > 30:
        # UUID format - likely TCGA/GDC
        source = "gdc"
    else:
        source = None
    
    try:
        from workflow_composer.data.discovery import DataDiscovery
        from workflow_composer.data.discovery.models import DataSource
        
        discovery = DataDiscovery()
        source_enum = DataSource(source) if source else None
        dataset = discovery.get_dataset(dataset_id, source_enum)
        
        if not dataset:
            return ToolResult(
                success=False,
                tool_name="get_dataset_details",
                error=f"Dataset {dataset_id} not found",
                message=f"❌ Could not find dataset **{dataset_id}**. Check the ID and try again."
            )
        
        # Format detailed output
        source_name = dataset.source.value.upper() if dataset.source else "Unknown"
        
        # Basic info table
        info_lines = [
            f"## 📋 {dataset.title or dataset_id}",
            "",
            "| Property | Value |",
            "|----------|-------|",
            f"| **ID** | `{dataset.id}` |",
            f"| **Source** | {source_name} |",
        ]
        
        if dataset.organism:
            info_lines.append(f"| **Organism** | {dataset.organism} |")
        if dataset.assembly:
            info_lines.append(f"| **Assembly** | {dataset.assembly} |")
        if dataset.assay_type:
            info_lines.append(f"| **Assay** | {dataset.assay_type} |")
        if dataset.target:
            info_lines.append(f"| **Target** | {dataset.target} |")
        if dataset.tissue:
            info_lines.append(f"| **Tissue** | {dataset.tissue} |")
        if dataset.cell_line:
            info_lines.append(f"| **Cell Line** | {dataset.cell_line} |")
        if dataset.date_released:
            info_lines.append(f"| **Released** | {dataset.date_released.strftime('%Y-%m-%d')} |")
        
        # Description
        if dataset.description:
            info_lines.extend([
                "",
                "### Description",
                dataset.description[:500] + ("..." if len(dataset.description) > 500 else ""),
            ])
        
        # Files section
        if dataset.download_urls:
            info_lines.extend([
                "",
                f"### 📁 Files ({len(dataset.download_urls)} available)",
                "",
            ])
            
            # Group by file type
            by_type = {}
            total_size = 0
            for url in dataset.download_urls:
                fmt = url.file_type.value if url.file_type else "OTHER"
                if fmt not in by_type:
                    by_type[fmt] = {"count": 0, "size": 0, "files": []}
                by_type[fmt]["count"] += 1
                by_type[fmt]["size"] += url.size_bytes or 0
                if len(by_type[fmt]["files"]) < 3:
                    by_type[fmt]["files"].append(url.filename)
                total_size += url.size_bytes or 0
            
            def format_size(bytes):
                if bytes == 0:
                    return "Unknown"
                for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                    if bytes < 1024:
                        return f"{bytes:.1f} {unit}"
                    bytes /= 1024
                return f"{bytes:.1f} PB"
            
            info_lines.append(f"**Total Size:** {format_size(total_size)}")
            info_lines.append("")
            info_lines.append("| Type | Count | Size | Examples |")
            info_lines.append("|------|-------|------|----------|")
            
            for fmt, info in sorted(by_type.items(), key=lambda x: -x[1]["size"]):
                examples = ", ".join(info["files"][:2])
                if len(info["files"]) > 2:
                    examples += ", ..."
                info_lines.append(f"| {fmt} | {info['count']} | {format_size(info['size'])} | {examples} |")
        
        # Links and actions
        info_lines.extend([
            "",
            "---",
            f"🔗 [View on {source_name}]({dataset.web_url})" if dataset.web_url else "",
            "",
            "**Actions:**",
            f"- `download {dataset.id}` - Download this dataset",
            f"- `download {dataset.id} without fastq` - Download processed files only",
        ])
        
        message = "\n".join(info_lines)
        
        return ToolResult(
            success=True,
            tool_name="get_dataset_details",
            data={
                "id": dataset.id,
                "source": source_name,
                "title": dataset.title,
                "organism": dataset.organism,
                "assay_type": dataset.assay_type,
                "file_count": len(dataset.download_urls) if dataset.download_urls else 0,
            },
            message=message
        )
        
    except ImportError as e:
        return ToolResult(
            success=False,
            tool_name="get_dataset_details",
            error=f"Discovery not available: {e}",
            message="❌ Dataset details feature not available."
        )
    except Exception as e:
        logger.error(f"Failed to get dataset details: {e}")
        return ToolResult(
            success=False,
            tool_name="get_dataset_details",
            error=str(e),
            message=f"❌ Error getting details: {e}"
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
