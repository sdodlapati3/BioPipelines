# Results Visualization & Download - Implementation Design

**Created:** November 26, 2025  
**Status:** ✅ Complete  
**Priority:** 🔴 CRITICAL

## Implementation Summary

The Results Visualization feature has been fully implemented with the following components:

### Core Package: `src/workflow_composer/results/`
- `models.py` - Data classes (FileType, ResultCategory, ResultFile, ResultSummary, FileTreeNode)
- `patterns.py` - File discovery patterns (PIPELINE_PATTERNS, EXCLUDE_PATTERNS)
- `detector.py` - File type detection (50+ extension mappings, magic bytes)
- `collector.py` - ResultCollector class for directory scanning
- `viewer.py` - ResultViewer class for file rendering
- `archiver.py` - ResultArchiver for ZIP creation
- `cloud_transfer.py` - GCS/S3 upload stubs
- `__init__.py` - Package exports

### UI Integration: `src/workflow_composer/web/gradio_app.py`
- Added "📊 Results" tab between Execute and Advanced tabs
- Directory selection dropdown with auto-discovery
- File scanning with category summary
- Interactive file tree view
- Multi-format file viewer (HTML, images, tables, text)
- Download functionality (all files or QC reports only)

---

---

## Overview

This document describes the implementation of the Result Visualization & Download system for BioPipelines. This feature allows users to view pipeline outputs (QC reports, plots, data files) directly in the browser and download results as archives.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RESULTS VISUALIZATION SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────┐ │
│  │ Completed    │────▶│ Result       │────▶│ ResultSummary                │ │
│  │ Pipeline Job │     │ Collector    │     │ ├── qc_reports: List[File]   │ │
│  └──────────────┘     └──────────────┘     │ ├── visualizations: List     │ │
│                              │              │ ├── data_files: List         │ │
│                              ▼              │ ├── alignments: List         │ │
│                       ┌──────────────┐     │ └── logs: List               │ │
│                       │ FileType     │     └──────────────────────────────┘ │
│                       │ Detector     │                    │                  │
│                       └──────────────┘                    ▼                  │
│                                                   ┌──────────────────┐      │
│                                                   │ Result Viewer    │      │
│                                                   │ ├── HTML embed   │      │
│                                                   │ ├── Image view   │      │
│                                                   │ ├── Table view   │      │
│                                                   │ └── Text view    │      │
│                                                   └──────────────────┘      │
│                                                           │                  │
│                                                           ▼                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        GRADIO UI                                      │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────────────┐ │   │
│  │  │ QC Reports │ │ Plots      │ │ File Tree  │ │ Download Options   │ │   │
│  │  │ (iframe)   │ │ (gallery)  │ │ (browser)  │ │ [ZIP] [Cloud]      │ │   │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        DOWNLOAD SYSTEM                                │   │
│  │  ResultArchiver ──▶ ZIP Archive ──▶ Gradio File Download             │   │
│  │  CloudTransfer ──▶ GCS/S3 ──▶ Signed URL                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
src/workflow_composer/results/
├── __init__.py              # Package exports
├── collector.py             # ResultCollector - scan output directories  
├── detector.py              # FileTypeDetector - identify file types
├── categorizer.py           # ResultCategorizer - group files by category
├── viewer.py                # ResultViewer - render for display
├── archiver.py              # ResultArchiver - create ZIP downloads
├── cloud_transfer.py        # CloudTransfer - GCS/S3 upload (optional)
├── patterns.py              # File patterns for different pipelines
└── models.py                # Data classes (ResultFile, ResultSummary)

src/workflow_composer/web/components/
├── __init__.py
├── result_browser.py        # Main result browsing component
└── file_tree.py             # File tree widget
```

---

## Implementation Checklist

### Phase 2.1: Core Results Package

- [ ] `results/models.py` - Data classes
  - [ ] FileType enum (QC_REPORT, IMAGE, TABLE, ALIGNMENT, etc.)
  - [ ] ResultCategory enum (QC, VISUALIZATION, DATA, ALIGNMENT, LOG)
  - [ ] ResultFile dataclass
  - [ ] ResultSummary dataclass

- [ ] `results/patterns.py` - File discovery patterns
  - [ ] Generic patterns (MultiQC, FastQC, images, tables)
  - [ ] Pipeline-specific patterns (RNA-seq, ChIP-seq, DNA-seq, scRNA)

- [ ] `results/detector.py` - File type detection
  - [ ] Extension-based detection
  - [ ] Magic byte detection for compressed files
  - [ ] Content sniffing for ambiguous files

- [ ] `results/collector.py` - Result collection
  - [ ] scan() - Scan output directory
  - [ ] get_qc_reports() - Get QC files
  - [ ] get_visualizations() - Get plot files
  - [ ] get_data_files() - Get data files
  - [ ] get_file_tree() - Get hierarchical tree

### Phase 2.2: Result Viewing

- [ ] `results/viewer.py` - Result rendering
  - [ ] render_html() - For QC reports
  - [ ] render_image() - For plots
  - [ ] render_table() - For TSV/CSV
  - [ ] render_text() - For logs
  - [ ] render_h5ad_summary() - For scRNA data

### Phase 2.3: Download & Archive

- [ ] `results/archiver.py` - ZIP creation
  - [ ] create_archive() - Full results ZIP
  - [ ] create_selective_archive() - Selected files
  - [ ] get_archive_size() - Estimate size

- [ ] `results/cloud_transfer.py` - Cloud upload (optional)
  - [ ] upload_to_gcs() - Google Cloud Storage
  - [ ] upload_to_s3() - AWS S3
  - [ ] get_signed_url() - Temporary download link

### Phase 2.4: UI Integration

- [ ] `web/components/result_browser.py` - Main component
  - [ ] QC reports panel (iframe embed)
  - [ ] Plot gallery
  - [ ] File browser tree
  - [ ] File preview panel
  - [ ] Download buttons

- [ ] Update `web/gradio_app.py`
  - [ ] Add "Results" tab
  - [ ] Connect to job completion
  - [ ] Wire up download functionality

---

## Data Classes

```python
# results/models.py

from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

class FileType(Enum):
    """Types of result files."""
    QC_REPORT = "qc_report"      # HTML reports (MultiQC, FastQC)
    IMAGE = "image"              # PNG, JPG, SVG plots
    PDF = "pdf"                  # PDF documents
    TABLE = "table"              # TSV, CSV data tables
    ALIGNMENT = "alignment"      # BAM, CRAM files
    VARIANT = "variant"          # VCF, BED files
    MATRIX = "matrix"            # H5AD, RDS, MTX count matrices
    LOG = "log"                  # Log files
    CONFIG = "config"            # Config/parameter files
    ARCHIVE = "archive"          # Compressed archives
    UNKNOWN = "unknown"

class ResultCategory(Enum):
    """Categories for grouping results in UI."""
    QC_REPORTS = "qc_reports"           # Priority 1
    VISUALIZATIONS = "visualizations"   # Priority 2
    DATA_FILES = "data_files"           # Priority 3
    ALIGNMENTS = "alignments"           # Priority 4
    LOGS = "logs"                       # Priority 5

@dataclass
class ResultFile:
    """A single result file with metadata."""
    path: Path
    name: str
    relative_path: str          # Relative to output directory
    size: int                   # Size in bytes
    file_type: FileType
    category: ResultCategory
    modified: datetime
    preview_available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size_human(self) -> str:
        """Human-readable file size."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if self.size < 1024:
                return f"{self.size:.1f} {unit}"
            self.size /= 1024
        return f"{self.size:.1f} TB"

@dataclass  
class ResultSummary:
    """Summary of all results for a pipeline run."""
    output_dir: Path
    pipeline_type: Optional[str]
    job_id: str
    total_files: int
    total_size: int
    
    # Categorized files
    qc_reports: List[ResultFile] = field(default_factory=list)
    visualizations: List[ResultFile] = field(default_factory=list)
    data_files: List[ResultFile] = field(default_factory=list)
    alignments: List[ResultFile] = field(default_factory=list)
    logs: List[ResultFile] = field(default_factory=list)
    
    # Quick access
    multiqc_report: Optional[ResultFile] = None
    fastqc_reports: List[ResultFile] = field(default_factory=list)
    
    @property
    def has_qc_reports(self) -> bool:
        return len(self.qc_reports) > 0
    
    @property
    def size_human(self) -> str:
        size = self.total_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
```

---

## File Patterns

```python
# results/patterns.py

# Generic patterns (work for all pipelines)
GENERIC_PATTERNS = {
    # QC Reports
    "multiqc": ["**/multiqc_report.html", "**/multiqc*.html"],
    "fastqc": ["**/*_fastqc.html", "**/fastqc/**/*.html"],
    "picard": ["**/*_metrics.txt", "**/picard/**/*.txt"],
    
    # Images/Plots
    "images": ["**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.svg"],
    "pdfs": ["**/*.pdf"],
    
    # Data Tables
    "tables": ["**/*.tsv", "**/*.csv", "**/*.txt"],
    
    # Alignments
    "bam": ["**/*.bam", "**/*.cram"],
    "bai": ["**/*.bai", "**/*.crai"],
    
    # Variants
    "vcf": ["**/*.vcf", "**/*.vcf.gz"],
    "bed": ["**/*.bed", "**/*.narrowPeak", "**/*.broadPeak"],
    
    # Matrices
    "h5ad": ["**/*.h5ad"],
    "rds": ["**/*.rds", "**/*.rda"],
    "mtx": ["**/*.mtx", "**/*.mtx.gz"],
    
    # Logs
    "logs": ["**/*.log", "**/pipeline_info/**", "**/.command.log"],
}

# Pipeline-specific patterns
PIPELINE_PATTERNS = {
    "rna_seq": {
        "salmon": ["**/salmon_quant/**", "**/quant.sf"],
        "star": ["**/star_align/**", "**/*Aligned*.bam"],
        "counts": ["**/counts/**", "**/featureCounts/**"],
        "deseq2": ["**/deseq2/**", "**/differential_expression/**"],
    },
    "chip_seq": {
        "peaks": ["**/macs2/**", "**/peaks/**", "**/*peaks.bed"],
        "bigwig": ["**/*.bw", "**/*.bigWig"],
        "homer": ["**/homer/**", "**/motifs/**"],
    },
    "dna_seq": {
        "variants": ["**/gatk/**", "**/deepvariant/**", "**/variants/**"],
        "bqsr": ["**/bqsr/**", "**/recalibration/**"],
    },
    "scrna_seq": {
        "cellranger": ["**/cellranger/**", "**/outs/**"],
        "scanpy": ["**/scanpy/**", "**/*.h5ad"],
        "seurat": ["**/seurat/**", "**/*.rds"],
    },
}

# Exclude patterns (skip these directories)
EXCLUDE_PATTERNS = [
    "**/work/**",           # Nextflow work directory
    "**/.nextflow/**",      # Nextflow cache
    "**/.snakemake/**",     # Snakemake cache
    "**/tmp/**",            # Temporary files
    "**/__pycache__/**",    # Python cache
]
```

---

## UI Design

### Results Tab Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  📊 Results Viewer                                          [Job: abc123]   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─ Job Selector ─────────────────────────────────────────────────────────┐ │
│  │ Select Completed Job: [Dropdown: job_abc123 (RNA-seq) ▼]  [Refresh]    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─ Summary ──────────────────────────────────────────────────────────────┐ │
│  │ 📁 Output: /data/results/rna_seq/abc123/                               │ │
│  │ 📊 Files: 47 files (2.3 GB total)                                      │ │
│  │ ✅ QC Reports: 5 | 📈 Plots: 12 | 📋 Data: 20 | 💾 Alignments: 8      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─ Content ──────────────────────────────────────────────────────────────┐ │
│  │ [📊 QC Reports] [📈 Plots] [📁 Files] [📝 Logs] [⬇️ Download]         │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │                                                                  │   │ │
│  │  │                    MultiQC Report                                │   │ │
│  │  │                    (embedded iframe)                             │   │ │
│  │  │                                                                  │   │ │
│  │  │                    [Open in New Tab]                             │   │ │
│  │  │                                                                  │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                         │ │
│  │  Other QC Reports:                                                     │ │
│  │  • sample1_fastqc.html [View]                                          │ │
│  │  • sample2_fastqc.html [View]                                          │ │
│  │  • alignment_metrics.txt [View]                                        │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─ Download Options ─────────────────────────────────────────────────────┐ │
│  │ [📥 Download All (ZIP)] [📥 Download QC Only] [☁️ Upload to Cloud]     │ │
│  │                                                                         │ │
│  │ Estimated ZIP size: 1.8 GB (excludes large BAM files)                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Order

1. **models.py** - Data classes (no dependencies)
2. **patterns.py** - File patterns (no dependencies)  
3. **detector.py** - File type detection (uses models)
4. **collector.py** - Result collection (uses models, patterns, detector)
5. **viewer.py** - Result rendering (uses models)
6. **archiver.py** - ZIP creation (uses collector)
7. **__init__.py** - Package exports
8. **web/components/result_browser.py** - UI component
9. **Update gradio_app.py** - Integration

---

## Testing

```python
# Test with actual pipeline output
from workflow_composer.results import ResultCollector

collector = ResultCollector(pipeline_type="rna_seq")
summary = collector.scan("/data/results/rna_seq/test_job/")

print(f"Total files: {summary.total_files}")
print(f"QC reports: {len(summary.qc_reports)}")
print(f"MultiQC: {summary.multiqc_report}")

# Test viewer
from workflow_composer.results import ResultViewer

viewer = ResultViewer()
if summary.multiqc_report:
    html = viewer.render_html(summary.multiqc_report)
    
# Test archiver
from workflow_composer.results import ResultArchiver

archiver = ResultArchiver()
zip_path = archiver.create_archive(summary, exclude_bam=True)
print(f"Created: {zip_path}")
```

---

## Notes

- **Security**: HTML reports are sandboxed in iframes
- **Performance**: Large files (>100MB) are not previewed
- **Caching**: File scans are cached for 5 minutes
- **Cloud Transfer**: Optional, requires GCS/S3 credentials

---

**Document Version:** 1.0  
**Last Updated:** November 26, 2025
