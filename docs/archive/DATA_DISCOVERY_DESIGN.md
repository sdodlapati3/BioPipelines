# Data Discovery & Reference Browser - Implementation Design

**Created:** November 26, 2025  
**Status:** ✅ IMPLEMENTED  
**Priority:** 🟡 MEDIUM  
**Completed:** January 2025

## Overview

A hybrid data discovery system that combines:
1. **Local browser** - Browse and validate existing data
2. **Database adapters** - Search ENCODE, GEO, SRA, Ensembl with structured APIs
3. **LLM-powered query parsing** - Natural language → structured parameters
4. **Dynamic download code generation** - For custom/unknown data sources

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        DATA DISCOVERY SYSTEM                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  User Input: "Download H3K27ac ChIP-seq data for human liver from ENCODE"       │
│                                      │                                           │
│                                      ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                   1. QUERY PARSER (LLM-powered)                            │  │
│  │                                                                            │  │
│  │   Natural Language → Structured Query                                      │  │
│  │   {                                                                        │  │
│  │     "organism": "human",                                                   │  │
│  │     "assay": "ChIP-seq",                                                   │  │
│  │     "target": "H3K27ac",                                                   │  │
│  │     "tissue": "liver",                                                     │  │
│  │     "source": "ENCODE"                                                     │  │
│  │   }                                                                        │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                           │
│                                      ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                   2. DATABASE ADAPTERS                                     │  │
│  │                                                                            │  │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │  │
│  │   │  ENCODE     │  │  GEO/SRA    │  │  Ensembl    │  │  Custom     │      │  │
│  │   │  Adapter    │  │  Adapter    │  │  Adapter    │  │  URL Parser │      │  │
│  │   │             │  │             │  │             │  │             │      │  │
│  │   │ REST API    │  │ Entrez API  │  │ REST API    │  │ LLM-powered │      │  │
│  │   │ portal.org  │  │ NCBI        │  │ ensembl.org │  │ scraping    │      │  │
│  │   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │  │
│  │                                                                            │  │
│  │   Common Interface:                                                        │  │
│  │   - search(query) → List[DatasetInfo]                                     │  │
│  │   - get_download_urls(id) → List[DownloadURL]                             │  │
│  │   - get_metadata(id) → Dict                                               │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                           │
│                                      ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                   3. RESULT AGGREGATOR                                     │  │
│  │                                                                            │  │
│  │   • Merge results from multiple sources                                    │  │
│  │   • Deduplicate (same dataset in GEO and ENCODE)                          │  │
│  │   • Rank by relevance, quality, size                                       │  │
│  │   • Present unified results                                                │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                           │
│                                      ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                   4. DOWNLOAD ORCHESTRATOR                                 │  │
│  │                                                                            │  │
│  │   Leverages existing: src/workflow_composer/data/downloader.py            │  │
│  │                                                                            │  │
│  │   ┌─────────────────────────────────────────────────────────────┐         │  │
│  │   │  Download Strategies                                        │         │  │
│  │   │                                                             │         │  │
│  │   │  1. Direct HTTP/FTP:  wget, curl, requests                 │         │  │
│  │   │  2. S3 buckets:       aws s3 sync                          │         │  │
│  │   │  3. GCS buckets:      gsutil cp                            │         │  │
│  │   │  4. SRA toolkit:      prefetch, fasterq-dump               │         │  │
│  │   │  5. Aspera:           ascp (high-speed for NCBI)           │         │  │
│  │   │  6. Custom (LLM):     Generated code for unusual sources   │         │  │
│  │   └─────────────────────────────────────────────────────────────┘         │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                           │
│                                      ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                   5. LOCAL DATA BROWSER                                    │  │
│  │                                                                            │  │
│  │   📁 /data/references/                                                     │  │
│  │   ├── genomes/                                                             │  │
│  │   │   ├── human/hg38.fa     ✅ Valid FASTA (3.1 GB)                       │  │
│  │   │   └── mouse/mm39.fa     ✅ Valid FASTA (2.7 GB)                       │  │
│  │   ├── indexes/                                                             │  │
│  │   │   ├── star_hg38/        ✅ 27 GB, compatible                          │  │
│  │   │   └── bwa_hg38/         ❌ Missing .amb file                          │  │
│  │   └── annotations/                                                         │  │
│  │       └── human/gencode.v44.gtf  ✅ 1.4 GB                                │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Package Structure

```
src/workflow_composer/data/
├── __init__.py                 # Exports (updated)
├── downloader.py               # EXISTING - base download functionality
├── discovery/
│   ├── __init__.py
│   ├── query_parser.py         # LLM-powered natural language parsing
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract base adapter
│   │   ├── encode.py           # ENCODE portal adapter
│   │   ├── geo.py              # GEO/SRA adapter (Entrez)
│   │   ├── ensembl.py          # Ensembl REST adapter
│   │   └── custom.py           # LLM-powered custom URL parser
│   ├── aggregator.py           # Merge & rank results
│   └── models.py               # DatasetInfo, SearchQuery, etc.
├── browser/
│   ├── __init__.py
│   ├── local_scanner.py        # Scan local directories
│   ├── validator.py            # Validate files (FASTA, GTF, BAM, etc.)
│   └── index_checker.py        # Check aligner index completeness
└── orchestrator.py             # Download queue & progress tracking
```

### Phase 1: Core Models & Base Adapter (2 hours)

**File: `data/discovery/models.py`**

```python
@dataclass
class SearchQuery:
    """Structured search query."""
    organism: Optional[str] = None
    assembly: Optional[str] = None
    assay_type: Optional[str] = None  # RNA-seq, ChIP-seq, ATAC-seq
    target: Optional[str] = None      # H3K27ac, CTCF, etc.
    tissue: Optional[str] = None
    cell_line: Optional[str] = None
    source: Optional[str] = None      # ENCODE, GEO, Ensembl
    keywords: List[str] = field(default_factory=list)
    raw_query: str = ""               # Original natural language

@dataclass
class DatasetInfo:
    """Information about a discovered dataset."""
    id: str                           # e.g., ENCSR000ABC, GSE12345
    source: str                       # ENCODE, GEO, SRA, Ensembl
    title: str
    description: str
    organism: str
    assay_type: Optional[str] = None
    download_urls: List[DownloadURL] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    size_bytes: Optional[int] = None
    file_count: int = 0
    quality_score: float = 0.0        # For ranking

@dataclass
class DownloadURL:
    """A downloadable file."""
    url: str
    filename: str
    file_type: str                    # fastq, bam, bed, etc.
    size_bytes: Optional[int] = None
    md5: Optional[str] = None
    download_method: str = "http"     # http, ftp, s3, sra, aspera
```

### Phase 2: Database Adapters (4 hours)

**ENCODE Adapter** - Uses REST API
- Endpoint: `https://www.encodeproject.org/search/`
- Rich metadata, well-structured
- Direct download URLs

**GEO/SRA Adapter** - Uses NCBI Entrez
- Search via Entrez E-utilities
- Convert GSE → SRR accessions
- Download via SRA toolkit

**Ensembl Adapter** - Uses REST API
- Reference genomes, annotations
- Known gene sets

### Phase 3: LLM Query Parser (2 hours)

**File: `data/discovery/query_parser.py`**

```python
class QueryParser:
    """Parse natural language queries into structured SearchQuery."""
    
    SYSTEM_PROMPT = '''You are a bioinformatics data query parser.
    Extract structured information from natural language queries about genomic data.
    
    Return JSON with these fields (use null if not mentioned):
    - organism: species name (human, mouse, etc.)
    - assembly: genome build (GRCh38, mm39, etc.)
    - assay_type: experimental type (RNA-seq, ChIP-seq, ATAC-seq, WGS, etc.)
    - target: for ChIP/CUT&RUN (H3K27ac, CTCF, p53, etc.)
    - tissue: tissue/organ type
    - cell_line: cell line name
    - source: preferred database (ENCODE, GEO, SRA, Ensembl)
    - keywords: other relevant search terms
    '''
    
    def parse(self, query: str) -> SearchQuery:
        """Parse natural language to structured query."""
        # Use LLM to extract structured data
        # Fallback to rule-based parsing if LLM fails
```

### Phase 4: Local Data Browser (3 hours)

**File: `data/browser/local_scanner.py`**

```python
class LocalDataScanner:
    """Scan local directories for reference data."""
    
    def scan_references(self, base_dir: Path) -> Dict[str, ReferenceInfo]:
        """Scan for reference genomes, annotations, indexes."""
        
    def validate_fasta(self, path: Path) -> ValidationResult:
        """Validate FASTA file."""
        
    def validate_gtf(self, path: Path) -> ValidationResult:
        """Validate GTF/GFF file."""
        
    def check_index_completeness(self, path: Path, aligner: str) -> IndexStatus:
        """Check if aligner index is complete."""
```

### Phase 5: UI Integration (3 hours)

**Add to `web/gradio_app.py`:**

```python
# New "📦 Data" tab with sub-tabs:
# - Local Browser: Browse existing data
# - Search: Natural language search
# - Download: Queue and progress
```

## Database Adapter Details

### ENCODE REST API

```python
# Search endpoint
GET https://www.encodeproject.org/search/?type=Experiment&assay_title=ChIP-seq&target.label=H3K27ac&biosample_ontology.term_name=liver&organism.scientific_name=Homo+sapiens&format=json

# Response includes:
# - Experiment accession (ENCSR000ABC)
# - Files with download URLs
# - Comprehensive metadata
```

### NCBI Entrez (GEO/SRA)

```python
from Bio import Entrez
Entrez.email = "user@example.com"

# Search GEO
handle = Entrez.esearch(db="gds", term="ChIP-seq H3K27ac human liver")
results = Entrez.read(handle)

# Get details
handle = Entrez.efetch(db="gds", id=results["IdList"])
```

### Ensembl REST API

```python
# Get available species
GET https://rest.ensembl.org/info/species?content-type=application/json

# Get genome info
GET https://rest.ensembl.org/info/assembly/homo_sapiens?content-type=application/json
```

## LLM for Custom Sources

For unknown or unusual data sources, the LLM generates download code:

```python
class CustomURLParser:
    """LLM-powered parser for custom data sources."""
    
    SYSTEM_PROMPT = '''You are a bioinformatics download expert.
    Given a URL or website description, generate Python code to:
    1. Parse the page/API to find download links
    2. Extract file metadata (size, type, checksums)
    3. Download the files
    
    Use requests, BeautifulSoup, or appropriate libraries.
    Return safe, executable Python code.
    '''
    
    def generate_download_code(self, url: str, description: str) -> str:
        """Generate download code for a custom source."""
        # LLM generates code
        # Code is sandboxed and validated before execution
```

## Success Metrics

| Metric | Target |
|--------|--------|
| ENCODE search accuracy | >95% |
| GEO/SRA search coverage | >90% |
| Query parsing accuracy | >85% |
| Local scan speed | <5s for 1000 files |
| Download reliability | >99% |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| API rate limits | Caching, exponential backoff |
| LLM hallucination | Validation layer, user confirmation |
| Large downloads fail | Resume support, chunked downloads |
| Custom code unsafe | Sandboxing, code review |

## Dependencies

```
# Required
requests>=2.28.0
biopython>=1.80          # For Entrez API
beautifulsoup4>=4.11.0   # For HTML parsing

# Optional  
boto3>=1.26.0           # For S3 downloads
google-cloud-storage    # For GCS downloads
```
