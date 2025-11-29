# BioPipelines: Data-First Architecture Implementation Plan

**Created:** November 27, 2025  
**Version:** 1.0  
**Status:** 📋 PLANNING  
**Priority:** 🔴 HIGH  

---

## Executive Summary

This document outlines a comprehensive plan to refactor BioPipelines from a **workflow-first** to a **data-first** architecture. The fundamental shift is:

**Current:** User Query → Parse Intent → Select Tools → Generate Workflow → [User manually finds data] → Run  
**Proposed:** User Query → Parse Intent → **Discover/Validate Data** → Select Tools (informed by data) → Generate Workflow (with paths) → Run

This change will dramatically improve user experience by:
1. Automatically discovering and validating data before workflow generation
2. Informing tool selection based on actual data characteristics
3. Generating ready-to-run workflows with real file paths
4. Eliminating manual data path configuration

---

## Current State Analysis

### Existing Components (Already Implemented)

| Component | Location | Status | Notes |
|-----------|----------|--------|-------|
| Data Discovery Models | `src/workflow_composer/data/discovery/models.py` | ✅ Complete | SearchQuery, DatasetInfo, DownloadURL |
| Query Parser | `src/workflow_composer/data/discovery/query_parser.py` | ✅ Complete | LLM-powered natural language parsing |
| ENCODE Adapter | `src/workflow_composer/data/discovery/adapters/encode.py` | ✅ Complete | REST API integration |
| GEO Adapter | `src/workflow_composer/data/discovery/adapters/geo.py` | ✅ Complete | Entrez API integration |
| Ensembl Adapter | `src/workflow_composer/data/discovery/adapters/ensembl.py` | ✅ Complete | REST API for references |
| Orchestrator | `src/workflow_composer/data/discovery/orchestrator.py` | ✅ Complete | DataDiscovery class |
| Reference Browser UI | `src/workflow_composer/data/browser/reference_browser.py` | ✅ Complete | Gradio component |
| Gradio Web UI | `src/workflow_composer/web/gradio_app.py` | ✅ Complete | Main interface (~2900 lines) |
| Intent Parser | `src/workflow_composer/core/query_parser.py` | ✅ Complete | Analysis type detection |
| Tool Selector | `src/workflow_composer/core/tool_selector.py` | ✅ Complete | Tool catalog matching |
| Module Mapper | `src/workflow_composer/core/module_mapper.py` | ✅ Complete | Nextflow module mapping |
| Workflow Generator | `src/workflow_composer/core/workflow_generator.py` | ✅ Complete | DSL2 code generation |
| Composer | `src/workflow_composer/composer.py` | ✅ Complete | Main orchestrator |

### Gaps to Address

| Gap | Priority | Complexity |
|-----|----------|------------|
| **UI**: Data tab not integrated into main flow | 🔴 HIGH | Medium |
| **UI**: No "data-first" wizard flow | 🔴 HIGH | Medium |
| **Backend**: DataManifest not connected to Composer | 🔴 HIGH | Medium |
| **Backend**: Tool selection not informed by data | 🟡 MEDIUM | Low |
| **Backend**: Workflow generator doesn't inject paths | 🔴 HIGH | Medium |
| **Backend**: Local sample scanner not implemented | 🟡 MEDIUM | Medium |
| **Backend**: Reference manager for auto-downloads | 🟡 MEDIUM | High |
| **Integration**: End-to-end data→workflow flow | 🔴 HIGH | High |

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    BIOPIPELINES DATA-FIRST FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

                                         USER INTERFACE
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                  │
│    ┌────────────────────────────────────────────────────────────────────────────────────────┐   │
│    │                              STEP 1: DESCRIBE ANALYSIS                                  │   │
│    │                                                                                         │   │
│    │   "I want to do RNA-seq differential expression analysis on human liver samples        │   │
│    │    comparing treated vs control"                                                        │   │
│    │                                                                                         │   │
│    │   [Parse Query]                                                                         │   │
│    └────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                │                                                 │
│                                                ▼                                                 │
│    ┌────────────────────────────────────────────────────────────────────────────────────────┐   │
│    │                              STEP 2: DATA DISCOVERY                                     │   │
│    │                                                                                         │   │
│    │   ┌─────────────────────┐    ┌─────────────────────────────────────────────────────┐   │   │
│    │   │   LOCAL DATA TAB    │    │              REMOTE DATA TAB                        │   │   │
│    │   │                     │    │                                                      │   │   │
│    │   │ Scanning: data/raw  │    │  🔍 Search: "human liver RNA-seq"                  │   │   │
│    │   │                     │    │                                                      │   │   │
│    │   │ Found Samples:      │    │  Source: [ENCODE ▼]  [GEO ▼]  [SRA ▼]              │   │   │
│    │   │ ┌─────────────────┐ │    │                                                      │   │   │
│    │   │ │ sample1_R1.fq.gz│ │    │  Results:                                           │   │   │
│    │   │ │ sample1_R2.fq.gz│ │    │  ┌────────────────────────────────────────────────┐ │   │   │
│    │   │ │ sample2_R1.fq.gz│ │    │  │ GSE123456 - Human Liver RNA-seq (10 samples)   │ │   │   │
│    │   │ │ sample2_R2.fq.gz│ │    │  │ ENCSR000ABC - Liver tissue RNA-seq (4 reps)   │ │   │   │
│    │   │ └─────────────────┘ │    │  │ SRP654321 - Liver treatment study (20 samples)│ │   │   │
│    │   │                     │    │  └────────────────────────────────────────────────┘ │   │   │
│    │   │ [Use Local Data]    │    │                                                      │   │   │
│    │   │                     │    │  [ ] GSE123456   [Download Selected]                │   │   │
│    │   └─────────────────────┘    │  [✓] ENCSR000ABC                                    │   │   │
│    │                               └─────────────────────────────────────────────────────┘   │   │
│    │                                                                                         │   │
│    │   ┌─────────────────────────────────────────────────────────────────────────────────┐   │   │
│    │   │                        REFERENCE DATA                                            │   │   │
│    │   │                                                                                  │   │   │
│    │   │   Organism: Human (GRCh38)                                                       │   │   │
│    │   │                                                                                  │   │   │
│    │   │   Genome:     [✅ Found: /data/references/human/GRCh38.fa    ] [Download]       │   │   │
│    │   │   Annotation: [✅ Found: /data/references/human/gencode.v44.gtf] [Download]     │   │   │
│    │   │   STAR Index: [❌ Missing                                     ] [Build Now]     │   │   │
│    │   │   Salmon Idx: [✅ Found: /data/references/salmon_index/       ]                 │   │   │
│    │   │                                                                                  │   │   │
│    │   └─────────────────────────────────────────────────────────────────────────────────┘   │   │
│    │                                                                                         │   │
│    │   [Continue with Selected Data →]                                                       │   │
│    └────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                │                                                 │
│                                                ▼                                                 │
│    ┌────────────────────────────────────────────────────────────────────────────────────────┐   │
│    │                              STEP 3: CONFIGURE PIPELINE                                 │   │
│    │                                                                                         │   │
│    │   Data Summary:                                                                         │   │
│    │   ├── 4 samples (paired-end, 150bp reads)                                              │   │
│    │   ├── Reference: GRCh38 + GENCODE v44                                                  │   │
│    │   └── Comparison: treated (2) vs control (2)                                           │   │
│    │                                                                                         │   │
│    │   Recommended Pipeline:                                                                 │   │
│    │   ┌──────────────────────────────────────────────────────────────────────────────────┐ │   │
│    │   │ FastQC → Trim Galore → STAR → FeatureCounts → DESeq2 → MultiQC                   │ │   │
│    │   │   ↓         ↓          ↓           ↓            ↓          ↓                     │ │   │
│    │   │  QC      Trimming  Alignment   Counting    Diff.Expr.   Report                  │ │   │
│    │   └──────────────────────────────────────────────────────────────────────────────────┘ │   │
│    │                                                                                         │   │
│    │   Tool Alternatives:                                                                    │   │
│    │   ├── Aligner:     [STAR ▼]  (alternatives: HISAT2, Salmon)                            │   │
│    │   ├── Quantifier:  [FeatureCounts ▼]  (alternatives: Salmon, HTSeq)                    │   │
│    │   └── DE Analysis: [DESeq2 ▼]  (alternatives: edgeR, limma)                            │   │
│    │                                                                                         │   │
│    │   Sample Sheet (auto-generated):                                                        │   │
│    │   ┌────────────────────────────────────────────────────────────────────────────────┐   │   │
│    │   │ sample,fastq_1,fastq_2,condition                                               │   │   │
│    │   │ ctrl_1,/data/raw/ctrl1_R1.fq.gz,/data/raw/ctrl1_R2.fq.gz,control              │   │   │
│    │   │ ctrl_2,/data/raw/ctrl2_R1.fq.gz,/data/raw/ctrl2_R2.fq.gz,control              │   │   │
│    │   │ treat_1,/data/raw/treat1_R1.fq.gz,/data/raw/treat1_R2.fq.gz,treated           │   │   │
│    │   │ treat_2,/data/raw/treat2_R1.fq.gz,/data/raw/treat2_R2.fq.gz,treated           │   │   │
│    │   └────────────────────────────────────────────────────────────────────────────────┘   │   │
│    │                                                                                         │   │
│    │   [Generate Workflow →]                                                                │   │
│    └────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                │                                                 │
│                                                ▼                                                 │
│    ┌────────────────────────────────────────────────────────────────────────────────────────┐   │
│    │                              STEP 4: REVIEW & EXECUTE                                   │   │
│    │                                                                                         │   │
│    │   Generated: rnaseq_liver_20251127_143022/                                             │   │
│    │   ├── main.nf           ✅ Ready                                                       │   │
│    │   ├── nextflow.config   ✅ Ready (paths configured)                                    │   │
│    │   ├── samplesheet.csv   ✅ Ready (4 samples)                                          │   │
│    │   └── modules/          ✅ Ready (6 modules)                                          │   │
│    │                                                                                         │   │
│    │   Pre-flight Checks:                                                                   │   │
│    │   ├── ✅ All samples accessible                                                        │   │
│    │   ├── ✅ Reference genome found                                                        │   │
│    │   ├── ✅ Annotation file found                                                         │   │
│    │   ├── ⚠️ STAR index will be built (est. 2 hours)                                      │   │
│    │   └── ✅ All containers available                                                      │   │
│    │                                                                                         │   │
│    │   Resource Estimate: 32GB RAM, 8 CPUs, ~6 hours                                        │   │
│    │                                                                                         │   │
│    │   [🚀 Execute on SLURM]  [Download ZIP]  [Edit Manually]                               │   │
│    └────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 0: Foundation & Data Models (Day 1)
**Goal:** Establish core data structures and the DataManifest

#### 0.1 Create DataManifest Model
**File:** `src/workflow_composer/data/manifest.py`

```python
@dataclass
class SampleInfo:
    """Information about a single sample."""
    sample_id: str
    fastq_1: Path                    # R1 file path
    fastq_2: Optional[Path] = None   # R2 for paired-end
    condition: Optional[str] = None  # Experimental condition
    replicate: Optional[int] = None
    
    # Detected properties
    is_paired: bool = False
    read_length: Optional[int] = None
    instrument: Optional[str] = None
    
    # Source
    source: str = "local"            # local, sra, encode, geo
    accession: Optional[str] = None  # SRR/GSM ID if remote

@dataclass
class ReferenceInfo:
    """Information about reference data."""
    organism: str
    assembly: str
    
    # File paths (None if not available)
    genome_fasta: Optional[Path] = None
    annotation_gtf: Optional[Path] = None
    transcriptome_fasta: Optional[Path] = None
    
    # Aligner indexes (dir paths)
    star_index: Optional[Path] = None
    hisat2_index: Optional[Path] = None
    bwa_index: Optional[Path] = None
    salmon_index: Optional[Path] = None
    kallisto_index: Optional[Path] = None
    
    # What's missing and needs download/build
    missing: List[str] = field(default_factory=list)
    download_urls: Dict[str, str] = field(default_factory=dict)

@dataclass
class DataManifest:
    """Complete data manifest for workflow generation."""
    # Samples
    samples: List[SampleInfo] = field(default_factory=list)
    sample_count: int = 0
    
    # References
    reference: Optional[ReferenceInfo] = None
    
    # Detected characteristics
    is_paired_end: bool = False
    avg_read_length: int = 0
    total_size_bytes: int = 0
    
    # Experimental design
    conditions: List[str] = field(default_factory=list)
    comparisons: List[Tuple[str, str]] = field(default_factory=list)
    
    # Validation status
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Source tracking
    created_from: str = ""  # "local_scan", "sra_download", "encode_download"
    
    def to_samplesheet(self) -> str:
        """Generate CSV samplesheet content."""
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
```

#### 0.2 Create LocalSampleScanner
**File:** `src/workflow_composer/data/scanner.py`

```python
class LocalSampleScanner:
    """Scan local directories for FASTQ samples."""
    
    def scan_directory(self, path: Path) -> List[SampleInfo]:
        """Scan directory for FASTQ files and pair them."""
        
    def detect_read_length(self, fastq_path: Path) -> int:
        """Sample first N reads to detect read length."""
        
    def detect_pairs(self, files: List[Path]) -> List[Tuple[Path, Optional[Path]]]:
        """Match R1/R2 pairs using naming conventions."""
        
    def infer_conditions(self, samples: List[SampleInfo]) -> List[str]:
        """Infer experimental conditions from sample names."""
```

---

### Phase 1: UI Redesign - Data Discovery Tab (Day 1-2)
**Goal:** Integrate data discovery into the main Gradio UI

#### 1.1 Restructure UI Tabs
**Current:**
```
[Workspace] [Execute] [Results] [Advanced]
```

**New:**
```
[Workspace] [📦 Data] [Execute] [Results] [Advanced]
```

#### 1.2 Create Data Tab Component
**File:** `src/workflow_composer/web/components/data_tab.py`

```python
def create_data_tab() -> Dict[str, gr.Component]:
    """Create the Data Discovery tab with three sub-tabs."""
    
    with gr.Tab("📦 Data", id="data"):
        with gr.Tabs() as data_subtabs:
            # Sub-tab 1: Local Data
            with gr.Tab("📁 Local Files", id="local"):
                local_scanner_ui()
            
            # Sub-tab 2: Remote Search
            with gr.Tab("🔍 Search Databases", id="remote"):
                remote_search_ui()
            
            # Sub-tab 3: References
            with gr.Tab("📚 References", id="references"):
                reference_manager_ui()
        
        # Data summary panel (always visible)
        with gr.Accordion("📊 Data Summary", open=True):
            data_manifest_display()
```

#### 1.3 Local Scanner UI
```python
def local_scanner_ui():
    """UI for scanning local directories."""
    
    gr.Markdown("### 📁 Scan Local Data Directory")
    
    with gr.Row():
        scan_dir = gr.Textbox(
            value="./data/raw",
            label="Directory to Scan",
            scale=3
        )
        scan_btn = gr.Button("🔍 Scan", variant="primary", scale=1)
    
    # Results table
    samples_table = gr.Dataframe(
        headers=["Sample", "R1", "R2", "Size", "Status"],
        label="Detected Samples"
    )
    
    # Condition assignment
    gr.Markdown("### Assign Conditions")
    condition_editor = gr.Dataframe(
        headers=["Sample", "Condition"],
        interactive=True
    )
    
    use_local_btn = gr.Button("✓ Use Selected Samples", variant="primary")
```

#### 1.4 Remote Search UI
```python
def remote_search_ui():
    """UI for searching remote databases."""
    
    gr.Markdown("### 🔍 Search Public Databases")
    
    with gr.Row():
        search_query = gr.Textbox(
            placeholder="e.g., human liver RNA-seq treated vs control",
            label="Natural Language Query",
            scale=4
        )
        search_btn = gr.Button("Search", variant="primary", scale=1)
    
    with gr.Row():
        source_filter = gr.CheckboxGroup(
            choices=["ENCODE", "GEO", "SRA"],
            value=["ENCODE", "GEO"],
            label="Sources"
        )
    
    # Results with checkboxes
    results_display = gr.Dataframe(
        headers=["Select", "Source", "ID", "Title", "Samples", "Size"],
        interactive=True
    )
    
    download_btn = gr.Button("📥 Download Selected", variant="primary")
    download_status = gr.Markdown("")
```

#### 1.5 Reference Manager UI
```python
def reference_manager_ui():
    """UI for managing reference data."""
    
    gr.Markdown("### 📚 Reference Data Manager")
    
    with gr.Row():
        organism_select = gr.Dropdown(
            choices=["Human", "Mouse", "Rat", "Zebrafish", "Fly", "Worm"],
            label="Organism",
            value="Human"
        )
        assembly_select = gr.Dropdown(
            choices=["GRCh38", "GRCh37", "T2T-CHM13"],
            label="Assembly",
            value="GRCh38"
        )
        check_btn = gr.Button("Check Availability", scale=1)
    
    # Reference status table
    ref_status = gr.Dataframe(
        headers=["Resource", "Status", "Path", "Size", "Action"],
        label="Reference Files"
    )
    
    # Action buttons
    with gr.Row():
        download_missing_btn = gr.Button("📥 Download Missing")
        build_indexes_btn = gr.Button("🔨 Build Indexes")
```

---

### Phase 2: Backend Integration (Day 2-3)
**Goal:** Connect data discovery to workflow generation

#### 2.1 Modify Composer Class
**File:** `src/workflow_composer/composer.py`

```python
class Composer:
    def __init__(self, config_path: Optional[Path] = None):
        # Existing initialization...
        
        # Add data discovery components
        self.data_discovery = DataDiscovery()
        self.local_scanner = LocalSampleScanner()
        self.reference_manager = ReferenceManager()
        self.data_manifest: Optional[DataManifest] = None
    
    def generate(
        self,
        query: str,
        config: Optional[Dict[str, Any]] = None,
        data_manifest: Optional[DataManifest] = None  # NEW
    ) -> Workflow:
        """
        Generate a workflow from natural language query.
        
        If data_manifest is provided, uses it to inform tool selection
        and inject actual file paths. Otherwise, generates with
        placeholder paths.
        """
        # Step 1: Parse intent (existing)
        intent = self.intent_parser.parse(query)
        
        # Step 2: Data validation (NEW)
        if data_manifest:
            self.data_manifest = data_manifest
            # Enrich intent with data characteristics
            intent = self._enrich_intent_with_data(intent, data_manifest)
        
        # Step 3: Select tools (modified to use data info)
        tools = self.tool_selector.select_tools(
            intent,
            data_characteristics=self._get_data_characteristics(data_manifest)
        )
        
        # Step 4: Map to modules (existing)
        modules = self.module_mapper.map_tools(tools)
        
        # Step 5: Generate workflow (modified to inject paths)
        workflow = self.workflow_generator.generate(
            intent,
            modules,
            data_manifest=data_manifest  # NEW
        )
        
        return workflow
    
    def _enrich_intent_with_data(
        self, 
        intent: ParsedIntent, 
        manifest: DataManifest
    ) -> ParsedIntent:
        """Add data-derived information to intent."""
        intent.parameters["is_paired_end"] = manifest.is_paired_end
        intent.parameters["sample_count"] = manifest.sample_count
        intent.parameters["conditions"] = manifest.conditions
        if manifest.reference:
            intent.organism = intent.organism or manifest.reference.organism
        return intent
    
    def _get_data_characteristics(
        self, 
        manifest: Optional[DataManifest]
    ) -> Dict[str, Any]:
        """Extract characteristics for tool selection."""
        if not manifest:
            return {}
        return {
            "is_paired": manifest.is_paired_end,
            "read_length": manifest.avg_read_length,
            "sample_count": manifest.sample_count,
            "has_replicates": len(manifest.conditions) < manifest.sample_count,
        }
```

#### 2.2 Modify ToolSelector
**File:** `src/workflow_composer/core/tool_selector.py`

```python
class ToolSelector:
    def select_tools(
        self,
        intent: ParsedIntent,
        data_characteristics: Optional[Dict[str, Any]] = None  # NEW
    ) -> List[ToolMatch]:
        """
        Select appropriate tools based on intent AND data characteristics.
        """
        # Existing selection logic...
        
        # NEW: Adjust tool selection based on data
        if data_characteristics:
            selected_tools = self._adjust_for_data(
                selected_tools, 
                data_characteristics
            )
        
        return selected_tools
    
    def _adjust_for_data(
        self, 
        tools: List[ToolMatch], 
        data: Dict[str, Any]
    ) -> List[ToolMatch]:
        """Adjust tool selection based on actual data."""
        adjusted = []
        
        for tool_match in tools:
            tool = tool_match.tool
            
            # Example: Prefer Salmon for long reads
            if data.get("read_length", 0) > 100:
                if tool.name == "kallisto":
                    # Suggest salmon as better alternative
                    tool_match.alternatives.insert(0, self._get_tool("salmon"))
            
            # Example: Use STAR for paired-end, HISAT2 for single-end
            if tool.category == "aligner":
                if not data.get("is_paired", True):
                    if tool.name == "star":
                        # Add note about single-end
                        tool_match.notes = "Consider HISAT2 for single-end data"
            
            adjusted.append(tool_match)
        
        return adjusted
```

#### 2.3 Modify WorkflowGenerator
**File:** `src/workflow_composer/core/workflow_generator.py`

```python
class WorkflowGenerator:
    def generate(
        self,
        intent: ParsedIntent,
        modules: List[NextflowModule],
        data_manifest: Optional[DataManifest] = None  # NEW
    ) -> Workflow:
        """Generate workflow with actual data paths if manifest provided."""
        
        # Generate base workflow (existing)
        workflow = self._generate_base(intent, modules)
        
        # NEW: Inject data paths if manifest provided
        if data_manifest:
            workflow = self._inject_data_paths(workflow, data_manifest)
            workflow.samplesheet = data_manifest.to_samplesheet()
            workflow.params_yaml = self._generate_params(data_manifest)
        
        return workflow
    
    def _inject_data_paths(
        self, 
        workflow: Workflow, 
        manifest: DataManifest
    ) -> Workflow:
        """Replace placeholder paths with actual paths from manifest."""
        config = workflow.config
        
        # Replace paths in config
        if manifest.reference:
            ref = manifest.reference
            config = config.replace(
                "params.genome = ''",
                f"params.genome = '{ref.genome_fasta}'"
            )
            config = config.replace(
                "params.gtf = ''",
                f"params.gtf = '{ref.annotation_gtf}'"
            )
            if ref.star_index:
                config = config.replace(
                    "params.star_index = ''",
                    f"params.star_index = '{ref.star_index}'"
                )
        
        workflow.config = config
        return workflow
    
    def _generate_params(self, manifest: DataManifest) -> str:
        """Generate params.yaml with actual values."""
        params = {
            "input": "./samplesheet.csv",
            "outdir": "./results",
        }
        
        if manifest.reference:
            ref = manifest.reference
            params["genome"] = str(ref.genome_fasta) if ref.genome_fasta else None
            params["gtf"] = str(ref.annotation_gtf) if ref.annotation_gtf else None
            params["star_index"] = str(ref.star_index) if ref.star_index else None
        
        return yaml.dump(params, default_flow_style=False)
```

---

### Phase 3: Reference Manager (Day 3-4)
**Goal:** Automatic reference data discovery and download

#### 3.1 Create ReferenceManager
**File:** `src/workflow_composer/data/reference_manager.py`

```python
class ReferenceManager:
    """Manage reference data (genomes, annotations, indexes)."""
    
    # Known reference locations
    REFERENCE_SOURCES = {
        "human": {
            "GRCh38": {
                "genome": "https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
                "gtf": "https://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/Homo_sapiens.GRCh38.110.gtf.gz",
                "transcriptome": "https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz",
            },
            "T2T-CHM13": {
                "genome": "https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/analysis_set/chm13v2.0.fa.gz",
            }
        },
        "mouse": {
            "GRCm39": {
                "genome": "https://ftp.ensembl.org/pub/release-110/fasta/mus_musculus/dna/Mus_musculus.GRCm39.dna.primary_assembly.fa.gz",
                "gtf": "https://ftp.ensembl.org/pub/release-110/gtf/mus_musculus/Mus_musculus.GRCm39.110.gtf.gz",
            }
        }
    }
    
    def __init__(self, base_dir: Path = Path("data/references")):
        self.base_dir = base_dir
    
    def check_references(
        self, 
        organism: str, 
        assembly: str
    ) -> ReferenceInfo:
        """Check what references are available locally."""
        
    def download_reference(
        self, 
        organism: str, 
        assembly: str, 
        resource: str  # "genome", "gtf", "transcriptome"
    ) -> Path:
        """Download a reference file."""
        
    def build_index(
        self, 
        aligner: str,  # "star", "bwa", "salmon", etc.
        genome_path: Path,
        gtf_path: Optional[Path] = None
    ) -> Path:
        """Build aligner index (runs as background job)."""
        
    def get_reference_for_organism(
        self, 
        organism: str
    ) -> ReferenceInfo:
        """Get best available reference for organism."""
```

---

### Phase 4: Wizard Flow Integration (Day 4-5)
**Goal:** Connect all pieces into guided workflow

#### 4.1 Create WizardState Manager
**File:** `src/workflow_composer/web/wizard.py`

```python
@dataclass
class WizardState:
    """Track state through the data-first wizard."""
    step: int = 1  # Current step (1-4)
    
    # Step 1: Query
    raw_query: str = ""
    parsed_intent: Optional[ParsedIntent] = None
    
    # Step 2: Data
    data_manifest: Optional[DataManifest] = None
    data_source: str = "local"  # local, remote
    
    # Step 3: Pipeline
    selected_tools: List[ToolMatch] = field(default_factory=list)
    tool_overrides: Dict[str, str] = field(default_factory=dict)
    
    # Step 4: Generated
    workflow: Optional[Workflow] = None
    validation_report: Optional[ValidationReport] = None
    
    def can_proceed(self) -> Tuple[bool, str]:
        """Check if current step is complete and can proceed."""
        if self.step == 1:
            return bool(self.parsed_intent), "Please describe your analysis"
        elif self.step == 2:
            if not self.data_manifest:
                return False, "Please select data source"
            if not self.data_manifest.is_valid:
                return False, f"Data validation failed: {self.data_manifest.validation_errors}"
            return True, ""
        elif self.step == 3:
            return bool(self.selected_tools), "Please confirm tool selection"
        elif self.step == 4:
            return bool(self.workflow), "Please generate workflow"
        return False, "Unknown step"


class WizardController:
    """Control wizard flow."""
    
    def __init__(self, composer: Composer):
        self.composer = composer
        self.state = WizardState()
    
    def step1_parse_query(self, query: str) -> Dict[str, Any]:
        """Step 1: Parse the user's query."""
        self.state.raw_query = query
        self.state.parsed_intent = self.composer.intent_parser.parse(query)
        self.state.step = 2
        
        return {
            "analysis_type": self.state.parsed_intent.analysis_type.value,
            "organism": self.state.parsed_intent.organism,
            "tasks": self.state.parsed_intent.tasks,
            "suggested_organism": self._suggest_organism(),
        }
    
    def step2_set_data(
        self, 
        manifest: DataManifest
    ) -> Dict[str, Any]:
        """Step 2: Set data manifest."""
        self.state.data_manifest = manifest
        
        # Validate
        manifest = self._validate_manifest(manifest)
        
        if manifest.is_valid:
            # Select tools based on data
            self.state.selected_tools = self.composer.tool_selector.select_tools(
                self.state.parsed_intent,
                self.composer._get_data_characteristics(manifest)
            )
            self.state.step = 3
        
        return {
            "is_valid": manifest.is_valid,
            "errors": manifest.validation_errors,
            "warnings": manifest.warnings,
            "tools": [t.tool.name for t in self.state.selected_tools],
        }
    
    def step3_confirm_tools(
        self, 
        overrides: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Step 3: Confirm or override tool selection."""
        if overrides:
            self.state.tool_overrides = overrides
            # Apply overrides to tool selection
            self._apply_tool_overrides()
        
        self.state.step = 4
        return {
            "final_tools": [t.tool.name for t in self.state.selected_tools],
        }
    
    def step4_generate(self) -> Workflow:
        """Step 4: Generate the workflow."""
        self.state.workflow = self.composer.generate(
            self.state.raw_query,
            data_manifest=self.state.data_manifest
        )
        
        # Run preflight validation
        self.state.validation_report = PreflightValidator().validate(
            self.state.parsed_intent,
            self.state.selected_tools
        )
        
        return self.state.workflow
```

#### 4.2 Integrate Wizard into Gradio App
**File:** `src/workflow_composer/web/gradio_app.py` (modifications)

```python
# Add wizard state to app state
app_state = gr.State({
    "wizard": None,  # WizardController instance
    "current_step": 1,
})

# Modify chat handler to start wizard flow
def chat_with_composer(message, history, provider, state):
    """Enhanced chat that starts wizard flow."""
    
    # Initialize wizard if not exists
    if state.get("wizard") is None:
        state["wizard"] = WizardController(get_composer())
    
    wizard = state["wizard"]
    
    # Step 1: Parse query
    if wizard.state.step == 1:
        result = wizard.step1_parse_query(message)
        
        response = f"""
## Analysis Detected ✓

**Type:** {result['analysis_type']}
**Organism:** {result['organism'] or 'Not specified'}
**Tasks:** {', '.join(result['tasks'])}

### Next Step: Select Your Data

Click the **📦 Data** tab to:
1. Scan local files, or
2. Search public databases

Or continue chatting to refine your analysis.
"""
        history.append((message, response))
        
    return history, "", state
```

---

### Phase 5: Testing & Validation (Day 5-6)

#### 5.1 Unit Tests
**File:** `tests/unit/test_data_manifest.py`

```python
class TestDataManifest:
    def test_samplesheet_generation(self):
        """Test CSV samplesheet generation."""
        
    def test_paired_end_detection(self):
        """Test paired-end sample pairing."""
        
    def test_condition_inference(self):
        """Test condition inference from names."""


class TestLocalScanner:
    def test_fastq_discovery(self):
        """Test FASTQ file discovery."""
        
    def test_pair_matching(self):
        """Test R1/R2 pair matching."""
        
    def test_read_length_detection(self):
        """Test read length detection."""


class TestReferenceManager:
    def test_check_references(self):
        """Test reference availability check."""
        
    def test_download_url_generation(self):
        """Test correct download URLs."""
```

#### 5.2 Integration Tests
**File:** `tests/integration/test_data_first_flow.py`

```python
class TestDataFirstFlow:
    def test_local_to_workflow(self):
        """Test: local scan → manifest → workflow generation."""
        
    def test_remote_to_workflow(self):
        """Test: search → download → manifest → workflow."""
        
    def test_reference_resolution(self):
        """Test: detect organism → find references → build index."""
        
    def test_path_injection(self):
        """Test: paths correctly injected into workflow."""
```

---

## File Changes Summary

### New Files to Create

| File | Purpose | LOC Est. |
|------|---------|----------|
| `src/workflow_composer/data/manifest.py` | DataManifest, SampleInfo, ReferenceInfo | ~200 |
| `src/workflow_composer/data/scanner.py` | LocalSampleScanner | ~150 |
| `src/workflow_composer/data/reference_manager.py` | ReferenceManager | ~300 |
| `src/workflow_composer/web/wizard.py` | WizardState, WizardController | ~250 |
| `src/workflow_composer/web/components/data_tab.py` | Data tab Gradio components | ~400 |
| `tests/unit/test_data_manifest.py` | Unit tests | ~150 |
| `tests/unit/test_local_scanner.py` | Unit tests | ~100 |
| `tests/integration/test_data_first_flow.py` | Integration tests | ~200 |

### Files to Modify

| File | Changes |
|------|---------|
| `src/workflow_composer/composer.py` | Add data_manifest parameter, enrich intent |
| `src/workflow_composer/core/tool_selector.py` | Add data_characteristics parameter |
| `src/workflow_composer/core/workflow_generator.py` | Add data_manifest parameter, path injection |
| `src/workflow_composer/web/gradio_app.py` | Add Data tab, integrate wizard flow |
| `src/workflow_composer/data/__init__.py` | Export new classes |

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 0: Data Models | 0.5 days | None |
| Phase 1: UI Redesign | 1.5 days | Phase 0 |
| Phase 2: Backend Integration | 1.5 days | Phase 0, 1 |
| Phase 3: Reference Manager | 1 day | Phase 0 |
| Phase 4: Wizard Flow | 1.5 days | Phase 1, 2, 3 |
| Phase 5: Testing | 1 day | Phase 4 |

**Total: ~7 days**

---

## Success Criteria

1. **User can scan local data** and see detected samples with pairing
2. **User can search remote databases** and queue downloads
3. **Reference data is automatically checked** and download links provided
4. **Tool selection is informed by data** characteristics
5. **Generated workflows have real paths** from manifest
6. **Sample sheet is auto-generated** with conditions
7. **Pre-flight validation confirms** all data accessible
8. **End-to-end test passes** for RNA-seq local data scenario

---

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Remote API rate limits | Medium | Medium | Implement caching, respect limits |
| Large file downloads fail | Medium | Medium | Resume support, chunked downloads |
| UI complexity too high | Low | High | Progressive disclosure, defaults |
| Reference index build time | High | Medium | Background jobs, status updates |

---

## Open Questions

1. Should downloads happen in foreground or background?
2. How to handle multi-organism comparisons?
3. Should we support cloud storage (S3, GCS) as data source?
4. How much auto-detection is too much (user override needs)?

---

## Next Steps

1. **Review this plan** and approve direction
2. **Start Phase 0** with DataManifest model
3. **Parallel: Design UI mockups** for Data tab
4. **Set up feature branch** for development
