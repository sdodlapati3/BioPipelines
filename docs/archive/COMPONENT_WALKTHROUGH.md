# BioPipelines - Component Walkthrough Guide

**Purpose:** Detailed technical walkthrough of each component from query entry to result visualization.

---

## Table of Contents

1. [Query Entry & Parsing](#1-query-entry--parsing)
2. [Intent Understanding & Validation](#2-intent-understanding--validation)
3. [Tool Selection & Module Mapping](#3-tool-selection--module-mapping)
4. [Workflow Generation](#4-workflow-generation)
5. [Data Preparation](#5-data-preparation)
6. [Pipeline Execution](#6-pipeline-execution)
7. [Monitoring & Status](#7-monitoring--status)
8. [Result Analysis](#8-result-analysis) ⚠️ NEEDS IMPLEMENTATION
9. [Visualization](#9-visualization) ⚠️ NEEDS IMPLEMENTATION

---

## 1. Query Entry & Parsing

### 1.1 Entry Points

```
┌─────────────────────────────────────────────────────────────┐
│                      Query Entry Points                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Gradio UI   │  │  REST API    │  │     CLI      │       │
│  │  (Primary)   │  │  (FastAPI)   │  │  (argparse)  │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            ▼                                 │
│                    ┌──────────────┐                          │
│                    │   Composer   │                          │
│                    │  (Main API)  │                          │
│                    └──────────────┘                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Query Examples

| Query Type | Example | Complexity |
|------------|---------|------------|
| Simple | "RNA-seq for mouse" | Low |
| Detailed | "ChIP-seq peak calling for H3K4me3 in human ES cells" | Medium |
| Complex | "Compare RNA-seq between treated and control samples with DESeq2, mouse GRCm39, paired-end" | High |

### 1.3 Code Flow

```python
# src/workflow_composer/web/gradio_app.py

def chat_with_composer(message: str, history: list) -> Tuple[list, str]:
    """Process user query and generate workflow."""
    
    # 1. Get LLM adapter
    llm = app_state.get_llm()
    
    # 2. Parse intent (determine what user wants)
    intent = app_state.composer.parse_intent(message)
    
    # 3. Check for workflow generation trigger
    if intent.analysis_type and intent.confidence > 0.5:
        # Generate workflow
        workflow = app_state.composer.generate(message, output_dir=output_dir)
```

### 1.4 Current Limitations

- ❌ No spell correction for organism/tool names
- ❌ No auto-complete/suggestions
- ❌ No query history

### 1.5 Proposed Improvements

```python
# PROPOSED: Enhanced query preprocessing

class QueryPreprocessor:
    """Preprocess and enhance user queries."""
    
    def __init__(self):
        self.organism_aliases = {
            "human": ["homo sapiens", "h. sapiens", "hsa"],
            "mouse": ["mus musculus", "m. musculus", "mmu"],
        }
        self.tool_corrections = {
            "deseq": "DESeq2",
            "star aligner": "STAR",
        }
    
    def preprocess(self, query: str) -> str:
        """Normalize and enhance query."""
        query = self._normalize_organisms(query)
        query = self._normalize_tools(query)
        return query
```

---

## 2. Intent Understanding & Validation

### 2.1 Intent Parser Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Intent Parsing Pipeline                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌─────────────────┐    ┌───────────────┐   │
│  │  Query   │───▶│   LLM Parser    │───▶│ ParsedIntent  │   │
│  │  String  │    │ (Structured)    │    │   Object      │   │
│  └──────────┘    └─────────────────┘    └───────────────┘   │
│                          │                      │            │
│                          │ OR (Ensemble)        │            │
│                          ▼                      ▼            │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Ensemble Parser (Optional - Higher Accuracy)           ││
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    ││
│  │  │  BioMistral  │ │  BiomedBERT  │ │   SciBERT    │    ││
│  │  │  (Intent)    │ │  (Entities)  │ │  (Terms)     │    ││
│  │  └──────────────┘ └──────────────┘ └──────────────┘    ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 ParsedIntent Structure

```python
@dataclass
class ParsedIntent:
    """Parsed user intent from natural language."""
    
    # Core identification
    analysis_type: AnalysisType      # e.g., RNA_SEQ_DE
    confidence: float                 # 0.0 - 1.0
    
    # Biological context
    organism: Optional[str]           # e.g., "mouse"
    genome_build: Optional[str]       # e.g., "GRCm39"
    data_type: Optional[str]          # e.g., "paired-end RNA-seq"
    
    # Experimental design
    paired_end: bool                  # True/False
    has_comparison: bool              # Has conditions to compare
    conditions: List[str]             # e.g., ["treated", "control"]
    
    # Extracted entities
    tools_mentioned: List[str]        # e.g., ["STAR", "DESeq2"]
    parameters: Dict[str, Any]        # Custom parameters
    
    # Original query
    original_query: str
```

### 2.3 Validation Steps

```python
# CURRENT: Basic validation
if intent.confidence < 0.5:
    return "I'm not sure what analysis you want. Could you clarify?"

# PROPOSED: Comprehensive validation
class IntentValidator:
    """Validate parsed intent against system capabilities."""
    
    def validate(self, intent: ParsedIntent) -> ValidationResult:
        errors = []
        warnings = []
        
        # 1. Check if analysis type is supported
        if intent.analysis_type not in SUPPORTED_ANALYSES:
            errors.append(f"Analysis type not supported: {intent.analysis_type}")
        
        # 2. Check if required tools are available
        for tool in intent.tools_mentioned:
            if not self.tool_selector.has_tool(tool):
                warnings.append(f"Tool may not be available: {tool}")
        
        # 3. Check if reference data exists
        if intent.organism and intent.genome_build:
            if not self.data_manager.has_reference(intent.organism, intent.genome_build):
                warnings.append(f"Reference may need download: {intent.organism}/{intent.genome_build}")
        
        return ValidationResult(errors=errors, warnings=warnings)
```

---

## 3. Tool Selection & Module Mapping

### 3.1 Tool Selection Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   Tool Selection Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────┐    ┌─────────────────┐    ┌──────────────┐  │
│  │AnalysisType│───▶│ Analysis→Tools  │───▶│  Tool List   │  │
│  │            │    │     Mapping     │    │              │  │
│  └────────────┘    └─────────────────┘    └──────────────┘  │
│                                                   │          │
│                                                   ▼          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                   Tool Catalog                          ││
│  │  ┌─────────────────────────────────────────────────────┐││
│  │  │ QC:         FastQC, MultiQC, Fastp, Trim_Galore    │││
│  │  │ Alignment:  STAR, HISAT2, BWA, Bowtie2, Minimap2   │││
│  │  │ Quant:      featureCounts, Salmon, kallisto        │││
│  │  │ Variants:   GATK, bcftools, FreeBayes, DeepVariant │││
│  │  │ Peaks:      MACS2, HOMER, SICER                    │││
│  │  │ scRNA:      STARsolo, Cellranger, Scanpy           │││
│  │  │ Meta:       Kraken2, MetaPhlAn, MEGAHIT            │││
│  │  └─────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Analysis → Tools Mapping

```python
# src/workflow_composer/core/tool_selector.py

ANALYSIS_TOOLS = {
    AnalysisType.RNA_SEQ_DE: {
        "qc": ["FastQC", "Fastp"],
        "alignment": ["STAR"],
        "quantification": ["featureCounts", "Salmon"],
        "analysis": ["DESeq2"],
        "reporting": ["MultiQC"]
    },
    AnalysisType.CHIP_SEQ: {
        "qc": ["FastQC"],
        "alignment": ["Bowtie2", "BWA"],
        "peak_calling": ["MACS2"],
        "annotation": ["HOMER"],
        "reporting": ["MultiQC"]
    },
    # ... more mappings
}
```

### 3.3 Module Mapping

```
┌─────────────────────────────────────────────────────────────┐
│                   Module Mapping Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────┐    ┌─────────────────┐    ┌──────────────┐  │
│  │  Tool      │───▶│ Tool→Module Map │───▶│   Module     │  │
│  │  (STAR)    │    │   (Registry)    │    │  (Nextflow)  │  │
│  └────────────┘    └─────────────────┘    └──────────────┘  │
│                                                   │          │
│                                                   ▼          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Nextflow Module Structure                  ││
│  │                                                         ││
│  │  nextflow-pipelines/modules/                            ││
│  │  ├── alignment/                                         ││
│  │  │   ├── star/main.nf          ← STAR_ALIGN process    ││
│  │  │   ├── bwa/main.nf           ← BWA_MEM process       ││
│  │  │   └── bowtie2/main.nf       ← BOWTIE2_ALIGN process ││
│  │  ├── qc/                                                ││
│  │  │   ├── fastqc/main.nf        ← FASTQC process        ││
│  │  │   └── multiqc/main.nf       ← MULTIQC process       ││
│  │  └── quantification/                                    ││
│  │      ├── salmon/main.nf        ← SALMON_QUANT process  ││
│  │      └── featurecounts/main.nf ← FEATURECOUNTS process ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Workflow Generation

### 4.1 Generation Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                 Workflow Generation Strategy                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Priority 1: File-Based Templates (Most Reliable)           │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ nextflow-pipelines/templates/                           ││
│  │ ├── chipseq_template.nf    ✅ Tested, parameterized     ││
│  │ ├── rnaseq_template.nf     ✅ Tested, parameterized     ││
│  │ └── ...                                                 ││
│  └─────────────────────────────────────────────────────────┘│
│                          │                                   │
│                          ▼ (if not found)                    │
│  Priority 2: Embedded Templates                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ WORKFLOW_TEMPLATES dict in workflow_generator.py       ││
│  └─────────────────────────────────────────────────────────┘│
│                          │                                   │
│                          ▼ (if not found)                    │
│  Priority 3: LLM Generation (Flexible, Less Reliable)       │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Generate via LLM prompt with module info                ││
│  │ ⚠️ May produce incorrect paths                          ││
│  └─────────────────────────────────────────────────────────┘│
│                          │                                   │
│                          ▼ (if no LLM)                       │
│  Priority 4: Generic Template                                │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Minimal workflow skeleton                               ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Generated Workflow Package

```
generated_workflows/
└── rnaseq_de_mouse_20251125_123456/
    ├── main.nf              # Main Nextflow workflow
    ├── nextflow.config      # Execution configuration
    ├── samplesheet.csv      # Sample metadata template
    ├── README.md            # Usage instructions
    ├── modules/             # (empty, uses symlink)
    └── logs/                # Execution logs (created at runtime)
```

### 4.3 Template Customization

```python
# Template with placeholders
template_content = """
/*
 * {{ANALYSIS_TYPE}} Workflow
 * Organism: {{ORGANISM}}
 * Genome: {{GENOME_BUILD}}
 * Generated: {{DATE}}
 */

params.reads = "${projectDir}/data/*_R{1,2}.fastq.gz"
params.genome = "${projectDir}/references/{{GENOME_BUILD}}/genome.fa"
params.single_end = {{SINGLE_END}}
"""

# Customization
params = {
    "ANALYSIS_TYPE": "RNA-seq Differential Expression",
    "ORGANISM": "mouse",
    "GENOME_BUILD": "GRCm39",
    "DATE": "2025-11-25",
    "SINGLE_END": "false"
}

main_nf = template.customize(params)
```

---

## 5. Data Preparation

### 5.1 Data Management Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Data Management System                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                Reference Data Sources                   ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   ││
│  │  │ Ensembl  │ │ GENCODE  │ │   UCSC   │ │ iGenomes │   ││
│  │  │ Genomes  │ │ Annots   │ │ Genomes  │ │ Indexes  │   ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   ││
│  └─────────────────────────────────────────────────────────┘│
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                  DataDownloader                         ││
│  │  • get_genome(organism, assembly)                       ││
│  │  • get_annotation(organism, assembly)                   ││
│  │  • get_index(aligner, organism, assembly)               ││
│  │  • get_sample_dataset(name)                             ││
│  └─────────────────────────────────────────────────────────┘│
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                  Local Cache                            ││
│  │  ~/.biopipelines/references/                            ││
│  │  ├── ensembl/                                           ││
│  │  │   ├── human/GRCh38.genome.fa.gz                     ││
│  │  │   └── mouse/GRCm39.annotation.gtf.gz                ││
│  │  ├── indexes/                                           ││
│  │  │   ├── star/human_GRCh38/                            ││
│  │  │   └── bwa/mouse_GRCm39/                             ││
│  │  └── manifest.json                                      ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Missing: UI Integration

```python
# PROPOSED: Data Browser Component for Gradio UI

class DataBrowserUI:
    """UI component for data management."""
    
    def render(self) -> gr.Column:
        with gr.Column() as data_browser:
            gr.Markdown("## 📁 Data Browser")
            
            with gr.Tab("Reference Data"):
                organism = gr.Dropdown(
                    choices=["human", "mouse", "zebrafish"],
                    label="Organism"
                )
                assembly = gr.Dropdown(
                    choices=[],  # Populated based on organism
                    label="Assembly"
                )
                download_btn = gr.Button("Download Reference")
                status = gr.Markdown()
            
            with gr.Tab("Sample Data"):
                gr.Markdown("Available test datasets:")
                datasets = gr.Dataframe(
                    headers=["Name", "Description", "Size"],
                    value=self._list_sample_datasets()
                )
            
            with gr.Tab("My Data"):
                gr.Markdown("Upload or register your data:")
                upload = gr.File(label="Upload FASTQ files")
                # Or register existing paths
                path_input = gr.Textbox(label="Path to existing data")
        
        return data_browser
```

---

## 6. Pipeline Execution

### 6.1 Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   Pipeline Execution Flow                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User clicks "Submit"                                        │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                            │
│  │ Validate     │ Check workflow exists, params valid        │
│  │ Workflow     │                                            │
│  └──────┬───────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                            │
│  │ Generate     │ Create SLURM batch script                  │
│  │ SBATCH       │ with Nextflow command                      │
│  └──────┬───────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                            │
│  │ Submit to    │ sbatch run_job_xxx.sbatch                  │
│  │ SLURM        │ Returns SLURM job ID                       │
│  └──────┬───────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                            │
│  │ Start        │ Background thread monitors                 │
│  │ Monitor      │ job status every 5 seconds                 │
│  └──────────────┘                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 SBATCH Script Generation

```bash
#!/bin/bash
#SBATCH --job-name=nf_rnaseq_workflow
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cpuspot

# Load required modules
module load nextflow 2>/dev/null || true
module load singularity 2>/dev/null || true

# Activate conda if available
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate biopipelines 2>/dev/null || true

# Change to workflow directory
cd /path/to/generated_workflows/rnaseq_xxx

# Run Nextflow
nextflow run main.nf \
    -profile slurm,singularity \
    -with-report logs/report.html \
    -with-timeline logs/timeline.html \
    -with-dag logs/dag.png \
    2>&1 | tee logs/nextflow_job_xxx.log

echo "Pipeline finished at $(date)"
```

---

## 7. Monitoring & Status

### 7.1 Monitoring Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Monitoring Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                  Monitor Thread                         ││
│  │  (runs every 5 seconds while job active)                ││
│  │                                                         ││
│  │  1. Check SLURM status (squeue -j JOB_ID)              ││
│  │  2. If not in queue, check sacct for final status      ││
│  │  3. Parse Nextflow log for progress                    ││
│  │  4. Detect errors from log patterns                    ││
│  │  5. Update job status in memory                        ││
│  └─────────────────────────────────────────────────────────┘│
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                  UI Auto-Refresh                        ││
│  │  (gr.Timer every 15 seconds)                            ││
│  │                                                         ││
│  │  • Re-parse logs for pending/running jobs              ││
│  │  • Update jobs table                                    ││
│  │  • Update job selector dropdown                        ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Status Detection Patterns

```python
# Error patterns to detect failure
ERROR_PATTERNS = [
    r"Error executing process",
    r"Pipeline failed",
    r"ERROR\s*[~\-]",
    r"No such file or directory",
    r"Command error:",
    r"Execution halted",
    r"FATAL:",
    r"Exception:",
]

# Success patterns
SUCCESS_PATTERNS = [
    "Pipeline completed",
    "Workflow completed",
    "Succeeded   :",
    "Workflow finished successfully",
]
```

---

## 8. Result Analysis ⚠️ NEEDS IMPLEMENTATION

### 8.1 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Result Analysis Pipeline                   │
│                      (TO BE IMPLEMENTED)                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                  Output Discovery                       ││
│  │  • Scan workflow output directories                     ││
│  │  • Identify file types (BAM, VCF, counts, etc.)        ││
│  │  • Catalog with metadata                                ││
│  └─────────────────────────────────────────────────────────┘│
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                  QC Report Parser                       ││
│  │  • Parse MultiQC JSON/HTML                              ││
│  │  • Extract FastQC metrics                               ││
│  │  • Aggregate per-sample QC                              ││
│  └─────────────────────────────────────────────────────────┘│
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                 Analysis-Specific                       ││
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       ││
│  │  │ RNA-seq │ │ChIP-seq │ │Variants │ │ scRNA   │       ││
│  │  │  DESeq2 │ │  Peaks  │ │  Stats  │ │ Clusters│       ││
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘       ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Proposed Implementation

```python
# src/workflow_composer/analysis/__init__.py

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class AnalysisResult:
    """Container for analysis results."""
    workflow_id: str
    analysis_type: str
    output_dir: Path
    qc_summary: Optional[Dict] = None
    result_files: List[Path] = None
    statistics: Dict = None

class ResultAnalyzer:
    """Analyze pipeline outputs."""
    
    def __init__(self, workflow_dir: str):
        self.workflow_dir = Path(workflow_dir)
    
    def discover_outputs(self) -> List[Path]:
        """Find all output files."""
        outputs = []
        for pattern in ["**/*.bam", "**/*.vcf*", "**/*counts*", "**/*.html"]:
            outputs.extend(self.workflow_dir.glob(pattern))
        return outputs
    
    def parse_multiqc(self) -> Dict:
        """Parse MultiQC report."""
        multiqc_data = self.workflow_dir / "results" / "multiqc" / "multiqc_data.json"
        if multiqc_data.exists():
            return json.load(open(multiqc_data))
        return {}
    
    def get_deseq2_results(self) -> Optional[pd.DataFrame]:
        """Parse DESeq2 results for RNA-seq."""
        deseq_file = self.workflow_dir / "results" / "deseq2" / "results.csv"
        if deseq_file.exists():
            return pd.read_csv(deseq_file)
        return None
    
    def get_peak_statistics(self) -> Optional[Dict]:
        """Get peak calling statistics for ChIP-seq."""
        peak_file = self.workflow_dir / "results" / "peaks" / "*_peaks.narrowPeak"
        # Count peaks, average width, etc.
        pass
```

---

## 9. Visualization ⚠️ NEEDS IMPLEMENTATION

### 9.1 Proposed Visualization Components

```
┌─────────────────────────────────────────────────────────────┐
│                   Visualization System                       │
│                      (TO BE IMPLEMENTED)                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Workflow Visualization                     ││
│  │  • DAG diagram (already exists, needs UI integration)  ││
│  │  • Process timeline                                     ││
│  │  • Resource usage graphs                                ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              QC Visualization                           ││
│  │  • Read quality distributions                           ││
│  │  • Mapping rate bar charts                              ││
│  │  • Coverage plots                                       ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Analysis-Specific Plots                    ││
│  │  ┌───────────────────────────────────────────────────┐ ││
│  │  │ RNA-seq: Volcano plot, PCA, heatmap               │ ││
│  │  │ ChIP-seq: Peak distribution, motif logos         │ ││
│  │  │ Variants: Allele frequency, transition/transversion│││
│  │  │ scRNA: UMAP/tSNE, cluster markers                 │ ││
│  │  └───────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Proposed UI Integration

```python
# New Results tab in Gradio UI

with gr.TabItem("📊 Results", id="results"):
    gr.Markdown("## Pipeline Results & Analysis")
    
    with gr.Row():
        # Left: Result browser
        with gr.Column(scale=4):
            completed_jobs = gr.Dropdown(
                choices=get_completed_jobs(),
                label="Select Completed Job"
            )
            
            with gr.Accordion("📁 Output Files", open=True):
                file_tree = gr.Dataframe(
                    headers=["File", "Type", "Size"],
                    interactive=False
                )
            
            with gr.Accordion("📋 QC Summary", open=True):
                qc_summary = gr.HTML()  # MultiQC iframe or parsed
        
        # Right: Visualizations
        with gr.Column(scale=6):
            with gr.Tab("QC Plots"):
                qc_plot = gr.Plot()
            
            with gr.Tab("Analysis Results"):
                analysis_plot = gr.Plot()  # Volcano, heatmap, etc.
            
            with gr.Tab("Reports"):
                report_viewer = gr.HTML()  # MultiQC HTML embed
```

---

## 10. Summary: Complete Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        BioPipelines Complete Flow                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  1. QUERY    "ChIP-seq for H3K4me3 in human"                                 │
│      │                                                                        │
│      ▼                                                                        │
│  2. PARSE    → AnalysisType.CHIP_SEQ, organism="human", antibody="H3K4me3"   │
│      │                                                                        │
│      ▼                                                                        │
│  3. TOOLS    → FastQC, Bowtie2, MACS2, HOMER, MultiQC                        │
│      │                                                                        │
│      ▼                                                                        │
│  4. MODULES  → modules/qc/fastqc, modules/alignment/bowtie2, ...             │
│      │                                                                        │
│      ▼                                                                        │
│  5. GENERATE → chipseq_20251125_123456/main.nf + config + samplesheet        │
│      │                                                                        │
│      ▼                                                                        │
│  6. DATA     → Download hg38 genome, bowtie2 index (if needed) ⚠️ MANUAL     │
│      │                                                                        │
│      ▼                                                                        │
│  7. EXECUTE  → sbatch → SLURM job → Nextflow → Singularity containers        │
│      │                                                                        │
│      ▼                                                                        │
│  8. MONITOR  → Auto-refresh, log parsing, status updates                     │
│      │                                                                        │
│      ▼                                                                        │
│  9. RESULTS  → Output files in results/ directory ❌ NO UI                    │
│      │                                                                        │
│      ▼                                                                        │
│  10. ANALYZE → Parse QC, count peaks, generate statistics ❌ NOT IMPLEMENTED  │
│      │                                                                        │
│      ▼                                                                        │
│  11. VISUALIZE → Plots, reports, interactive exploration ❌ PARTIAL          │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Phase 1 (Priority):** Implement Result Analysis module
2. **Phase 2:** Integrate Data Downloader into UI
3. **Phase 3:** Add Visualization components
4. **Phase 4:** Improve monitoring reliability

---

*Document generated: November 25, 2025*
