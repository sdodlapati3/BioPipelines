# Container Architecture for AI-Agentic Multi-Omics Pipelines

**Date**: November 23, 2025  
**Version**: 1.0  
**Status**: 🚀 Implementation Phase

---

## Vision: AI-Agent Ready Containers

### Design Principles

1. **Modularity**: Each container = one capability unit (e.g., "RNA-seq QC", "Variant Calling")
2. **Discoverability**: Containers have machine-readable metadata for AI agent discovery
3. **Composability**: Agents can chain containers to build custom workflows
4. **Standardization**: Consistent I/O contracts (FASTQ → BAM → VCF, etc.)
5. **Observability**: Containers emit structured logs/metrics for agent monitoring
6. **Extensibility**: Easy to add new tools without breaking existing workflows

### AI Agent Integration Pattern

```
┌─────────────┐
│ AI Agent    │ ← User: "Analyze my RNA-seq data"
│ Orchestrator│
└──────┬──────┘
       │ 1. Query available containers
       │ 2. Build execution plan
       │ 3. Compose workflow
       ▼
┌──────────────────────────────────────┐
│  Container Registry + Metadata       │
│  ┌────────┐ ┌────────┐ ┌────────┐  │
│  │RNA-seq │ │DNA-seq │ │ChIP-seq│  │
│  │v1.0    │ │v1.0    │ │v1.0    │  │
│  └────────┘ └────────┘ └────────┘  │
└──────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  Execution Engine (Snakemake/Nextflow│
│  or Direct Container Invocation)     │
└──────────────────────────────────────┘
```

---

## Container Hierarchy

### Tier 1: Base Container (Foundation)
**Purpose**: Common tools shared across all pipelines  
**Size**: ~1GB  
**Updated**: Quarterly

```dockerfile
# containers/base/Dockerfile
FROM mambaorg/micromamba:1.5.8
LABEL description="BioPipelines base container with common tools"
LABEL version="1.0.0"
LABEL maintainer="sdodlapa"

# Common tools used across all pipelines
RUN micromamba install -y -n base -c conda-forge -c bioconda \
    # Core utilities
    samtools=1.17 \
    bcftools=1.17 \
    htslib=1.17 \
    bedtools=2.30.0 \
    parallel \
    pigz \
    # QC tools (universal)
    fastqc=0.12.1 \
    multiqc=1.14 \
    # Python essentials
    python=3.11 \
    numpy \
    pandas \
    biopython \
    pysam \
    && micromamba clean --all --yes
```

### Tier 2: Pipeline Containers (Specialized)
**Purpose**: Complete environment for one pipeline  
**Inherits**: Base container  
**Size**: ~2-4GB each

```
containers/
├── base/
│   ├── Dockerfile
│   └── Singularity.def
├── rna-seq/
│   ├── Dockerfile          # Extends base
│   ├── Singularity.def
│   ├── manifest.json       # AI-readable metadata
│   └── README.md
├── dna-seq/
│   ├── Dockerfile
│   ├── Singularity.def
│   ├── manifest.json
│   └── README.md
├── chip-seq/
├── atac-seq/
├── methylation/
├── hic/
├── scrna-seq/
├── long-read/
├── metagenomics/
└── structural-variants/
```

### Tier 3: Tool-Specific Containers (Micro-services)
**Purpose**: Single-tool containers for AI agent composition  
**Size**: ~500MB-1GB each

```
containers/tools/
├── fastqc/           # Just FastQC
├── star/             # Just STAR aligner
├── salmon/           # Just Salmon quantification
├── gatk/             # Just GATK
├── macs2/            # Just MACS2
└── deseq2/           # Just DESeq2 R environment
```

---

## Container Manifest (AI-Agent Interface)

Each container includes a `manifest.json` for AI agent discovery:

```json
{
  "name": "biopipelines-rna-seq",
  "version": "1.0.0",
  "type": "pipeline",
  "category": "transcriptomics",
  "description": "Complete RNA-seq analysis pipeline from FASTQ to differential expression",
  
  "capabilities": [
    "quality_control",
    "adapter_trimming",
    "alignment",
    "quantification",
    "differential_expression"
  ],
  
  "input_formats": ["fastq", "fastq.gz"],
  "output_formats": ["bam", "counts_matrix", "deseq2_results"],
  
  "parameters": {
    "genome": {
      "type": "string",
      "required": true,
      "description": "Reference genome (hg38, mm10, etc.)"
    },
    "strandedness": {
      "type": "enum",
      "options": ["unstranded", "forward", "reverse"],
      "default": "unstranded"
    },
    "threads": {
      "type": "integer",
      "default": 8,
      "min": 1,
      "max": 64
    }
  },
  
  "resources": {
    "min_memory_gb": 16,
    "recommended_memory_gb": 32,
    "min_cores": 4,
    "recommended_cores": 8,
    "disk_space_gb": 100
  },
  
  "execution": {
    "entrypoint": "/analysis/pipeline.sh",
    "container_uri": "docker://biopipelines/rna-seq:1.0.0",
    "singularity_uri": "oras://ghcr.io/biopipelines/rna-seq:1.0.0"
  },
  
  "dependencies": {
    "reference_genome": "required",
    "gene_annotation": "required"
  },
  
  "tools": [
    {"name": "FastQC", "version": "0.12.1"},
    {"name": "fastp", "version": "0.23.4"},
    {"name": "STAR", "version": "2.7.10b"},
    {"name": "Salmon", "version": "1.10.1"},
    {"name": "DESeq2", "version": "1.38.0"}
  ],
  
  "metadata": {
    "created": "2025-11-23",
    "updated": "2025-11-23",
    "authors": ["BioPipelines Team"],
    "license": "MIT",
    "repository": "https://github.com/sdodlapa/BioPipelines",
    "documentation": "https://biopipelines.readthedocs.io/rna-seq"
  }
}
```

### AI Agent Usage Example

```python
# AI Agent discovers and invokes containers
from biopipelines_agent import ContainerRegistry, PipelineOrchestrator

# 1. Discover available pipelines
registry = ContainerRegistry("https://ghcr.io/biopipelines")
pipelines = registry.search(category="transcriptomics")

# 2. Agent selects appropriate pipeline
rna_pipeline = pipelines["rna-seq"]

# 3. Agent validates inputs and parameters
plan = rna_pipeline.create_execution_plan(
    inputs=["sample1_R1.fastq.gz", "sample1_R2.fastq.gz"],
    genome="hg38",
    output_dir="/results"
)

# 4. Execute with monitoring
orchestrator = PipelineOrchestrator()
result = orchestrator.execute(plan, monitor=True)

# 5. Agent interprets results
if result.success:
    agent.report(f"Found {result.metrics['de_genes']} differentially expressed genes")
```

---

## Implementation Plan

### Phase 1: Foundation (Days 1-2)

#### 1.1 Create Base Container
```bash
cd ~/BioPipelines
mkdir -p containers/base
```

**Dockerfile**:
```dockerfile
FROM mambaorg/micromamba:1.5.8

LABEL org.opencontainers.image.title="BioPipelines Base"
LABEL org.opencontainers.image.description="Common tools for multi-omics analysis"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="BioPipelines Team"
LABEL org.opencontainers.image.source="https://github.com/sdodlapa/BioPipelines"

# Install common tools
RUN micromamba install -y -n base -c conda-forge -c bioconda \
    samtools=1.17 \
    bcftools=1.17 \
    htslib=1.17 \
    bedtools=2.30.0 \
    fastqc=0.12.1 \
    multiqc=1.14 \
    python=3.11 \
    numpy \
    pandas \
    biopython \
    pysam \
    pybedtools \
    && micromamba clean --all --yes

ENV PATH="/opt/conda/bin:$PATH"
WORKDIR /analysis
```

**Build**:
```bash
cd containers/base
docker build -t biopipelines/base:1.0.0 .
singularity build base_1.0.0.sif docker-daemon://biopipelines/base:1.0.0
```

#### 1.2 Create Container Build System
```bash
mkdir -p scripts/containers
```

**scripts/containers/build_all.sh**:
```bash
#!/bin/bash
# Automated container building

set -e

REGISTRY="biopipelines"
VERSION="1.0.0"

echo "Building BioPipelines containers..."

# Build base first
docker build -t ${REGISTRY}/base:${VERSION} containers/base/

# Build pipeline containers
for pipeline in rna-seq dna-seq chip-seq atac-seq; do
    echo "Building ${pipeline}..."
    docker build -t ${REGISTRY}/${pipeline}:${VERSION} containers/${pipeline}/
done

echo "Converting to Singularity..."
mkdir -p /scratch/sdodl001/containers

for pipeline in base rna-seq dna-seq chip-seq atac-seq; do
    singularity build \
        /scratch/sdodl001/containers/${pipeline}_${VERSION}.sif \
        docker-daemon://${REGISTRY}/${pipeline}:${VERSION}
done

echo "✓ All containers built successfully"
```

### Phase 2: RNA-seq Pilot (Days 2-3)

#### 2.1 Create RNA-seq Container
```dockerfile
# containers/rna-seq/Dockerfile
FROM biopipelines/base:1.0.0

LABEL org.opencontainers.image.title="BioPipelines RNA-seq"
LABEL org.opencontainers.image.description="RNA-seq analysis from FASTQ to differential expression"
LABEL org.opencontainers.image.version="1.0.0"

# RNA-seq specific tools
RUN micromamba install -y -n base -c conda-forge -c bioconda \
    fastp=0.23.4 \
    trimmomatic=0.39 \
    star=2.7.10b \
    hisat2=2.2.1 \
    salmon=1.10.1 \
    subread=2.0.3 \
    r-base=4.2 \
    bioconductor-deseq2 \
    bioconductor-edger \
    r-ggplot2 \
    r-pheatmap \
    && micromamba clean --all --yes

# Copy pipeline scripts
COPY scripts/ /opt/biopipelines/scripts/
COPY pipelines/rna_seq/ /opt/biopipelines/rna_seq/

# Set entrypoint
COPY entrypoint.sh /opt/biopipelines/
RUN chmod +x /opt/biopipelines/entrypoint.sh

ENTRYPOINT ["/opt/biopipelines/entrypoint.sh"]
CMD ["--help"]
```

**containers/rna-seq/entrypoint.sh**:
```bash
#!/bin/bash
# Smart entrypoint for RNA-seq container

set -e

if [ "$1" = "--help" ]; then
    cat << EOF
BioPipelines RNA-seq Container v1.0.0

Usage:
  # Run full pipeline
  singularity run rna-seq.sif --input /data/fastq --output /data/results --genome hg38

  # Run specific tool
  singularity exec rna-seq.sif fastqc sample.fastq.gz

  # AI Agent mode (JSON config)
  singularity run rna-seq.sif --config config.json

Parameters:
  --input DIR       Input directory with FASTQ files
  --output DIR      Output directory
  --genome STR      Reference genome (hg38, mm10, etc.)
  --threads INT     Number of threads (default: 8)
  --config FILE     JSON configuration for AI agents
EOF
    exit 0
fi

# Handle AI agent mode
if [ "$1" = "--config" ]; then
    python3 /opt/biopipelines/scripts/ai_agent_handler.py "$2"
    exit $?
fi

# Regular Snakemake execution
cd /opt/biopipelines/rna_seq
exec snakemake --cores "${THREADS:-8}" "$@"
```

#### 2.2 Update RNA-seq Snakefile
```python
# pipelines/rna_seq/Snakefile
configfile: "config.yaml"

# Use container for all rules
container: "docker://biopipelines/rna-seq:1.0.0"

rule all:
    input:
        "results/multiqc_report.html",
        "results/counts_matrix.txt",
        "results/deseq2/de_results.csv"

rule fastqc:
    input:
        "data/raw/{sample}.fastq.gz"
    output:
        "results/qc/{sample}_fastqc.html"
    shell:
        "fastqc {input} -o $(dirname {output})"

rule fastp:
    input:
        r1="data/raw/{sample}_R1.fastq.gz",
        r2="data/raw/{sample}_R2.fastq.gz"
    output:
        r1="results/trimmed/{sample}_R1.fastq.gz",
        r2="results/trimmed/{sample}_R2.fastq.gz",
        json="results/trimmed/{sample}_fastp.json"
    threads: 4
    shell:
        """
        fastp -i {input.r1} -I {input.r2} \
              -o {output.r1} -O {output.r2} \
              -j {output.json} \
              --thread {threads}
        """

# ... rest of rules
```

### Phase 3: Container Registry & Discovery (Day 3)

#### 3.1 Create Container Metadata System
```python
# src/biopipelines/containers/registry.py
"""
Container registry for AI agent discovery
"""
import json
from pathlib import Path
from typing import List, Dict, Optional

class ContainerManifest:
    def __init__(self, manifest_path: Path):
        with open(manifest_path) as f:
            self.data = json.load(f)
    
    def matches_query(self, 
                     category: Optional[str] = None,
                     capability: Optional[str] = None,
                     input_format: Optional[str] = None) -> bool:
        """Check if container matches AI agent query"""
        if category and self.data.get("category") != category:
            return False
        if capability and capability not in self.data.get("capabilities", []):
            return False
        if input_format and input_format not in self.data.get("input_formats", []):
            return False
        return True
    
    def get_execution_command(self, params: Dict) -> str:
        """Generate execution command for AI agent"""
        entrypoint = self.data["execution"]["entrypoint"]
        # Build command from params
        cmd_parts = [entrypoint]
        for key, value in params.items():
            cmd_parts.append(f"--{key} {value}")
        return " ".join(cmd_parts)

class ContainerRegistry:
    def __init__(self, containers_dir: Path):
        self.containers_dir = containers_dir
        self.manifests = self._load_manifests()
    
    def _load_manifests(self) -> List[ContainerManifest]:
        """Load all container manifests"""
        manifests = []
        for manifest_file in self.containers_dir.glob("*/manifest.json"):
            manifests.append(ContainerManifest(manifest_file))
        return manifests
    
    def search(self, **query) -> List[ContainerManifest]:
        """AI agent queries available containers"""
        return [m for m in self.manifests if m.matches_query(**query)]
    
    def get_container(self, name: str) -> Optional[ContainerManifest]:
        """Get specific container by name"""
        for manifest in self.manifests:
            if manifest.data["name"] == name:
                return manifest
        return None
```

#### 3.2 AI Agent Integration Layer
```python
# src/biopipelines/agents/orchestrator.py
"""
AI agent orchestration for container-based pipelines
"""
from typing import Dict, List
from pathlib import Path
import json

class PipelineOrchestrator:
    """
    Enables AI agents to discover, compose, and execute pipelines
    """
    
    def __init__(self, registry_path: Path):
        self.registry = ContainerRegistry(registry_path)
    
    def discover_pipeline(self, user_query: str) -> Dict:
        """
        AI agent analyzes user request and finds appropriate pipeline
        
        Example:
            query: "I have RNA-seq data and want to find differentially expressed genes"
            -> Returns: rna-seq container manifest
        """
        # Simple keyword matching (can be enhanced with LLM)
        if "rna" in user_query.lower() or "transcriptom" in user_query.lower():
            return self.registry.get_container("biopipelines-rna-seq")
        elif "variant" in user_query.lower() or "dna" in user_query.lower():
            return self.registry.get_container("biopipelines-dna-seq")
        # ... more logic
    
    def compose_workflow(self, steps: List[str]) -> str:
        """
        AI agent composes custom workflow from individual tools
        
        Example:
            steps = ["fastqc", "trim", "align", "quantify"]
            -> Generates Snakemake workflow using appropriate containers
        """
        workflow_template = """
rule all:
    input: {final_outputs}

{rules}
        """
        # Generate rules for each step
        rules = []
        for step in steps:
            container = self.registry.search(capability=step)[0]
            # Generate rule using container
            rules.append(f"# Rule for {step} using {container.data['name']}")
        
        return workflow_template.format(
            final_outputs="...",
            rules="\n".join(rules)
        )
    
    def execute(self, manifest: ContainerManifest, params: Dict) -> Dict:
        """
        Execute pipeline with monitoring
        """
        command = manifest.get_execution_command(params)
        # Execute and monitor
        # Return structured results for agent interpretation
        return {
            "success": True,
            "metrics": {},
            "outputs": []
        }
```

### Phase 4: Multi-Container Orchestration (Day 4)

#### 4.1 Create Multi-Omics Integration Container
```dockerfile
# containers/multi-omics/Dockerfile
FROM biopipelines/base:1.0.0

LABEL org.opencontainers.image.title="BioPipelines Multi-Omics Integrator"
LABEL org.opencontainers.image.description="AI agent for multi-omics data integration"

# Integration tools
RUN micromamba install -y -n base -c conda-forge -c bioconda \
    # Multi-omics integration
    mofa2 \
    mixomics \
    # Machine learning
    scikit-learn \
    xgboost \
    pytorch \
    # Visualization
    matplotlib \
    seaborn \
    plotly \
    && micromamba clean --all --yes

# AI agent framework
RUN pip install langchain openai anthropic

COPY scripts/multi_omics_agent.py /opt/biopipelines/
```

#### 4.2 Example: AI Agent Composes Multi-Omics Workflow
```python
# Example: AI agent receives complex query
user_query = """
I have RNA-seq and ChIP-seq data from the same samples.
I want to:
1. Identify differentially expressed genes
2. Find which transcription factors regulate them
3. Integrate both datasets to find TF-target relationships
"""

# AI agent workflow
orchestrator = PipelineOrchestrator(registry_path="containers/")

# Step 1: Discover required containers
rna_container = orchestrator.discover_pipeline("rna-seq differential expression")
chip_container = orchestrator.discover_pipeline("chip-seq peak calling")
integration_container = orchestrator.discover_pipeline("multi-omics integration")

# Step 2: Compose workflow
workflow = orchestrator.compose_workflow([
    {"container": rna_container, "input": "rna_samples/", "output": "rna_results/"},
    {"container": chip_container, "input": "chip_samples/", "output": "chip_results/"},
    {"container": integration_container, "input": ["rna_results/", "chip_results/"], "output": "integrated/"}
])

# Step 3: Execute with monitoring
result = orchestrator.execute(workflow, monitor=True)

# Step 4: AI agent interprets results
summary = f"""
Analysis complete:
- {result.rna.de_genes} differentially expressed genes
- {result.chip.peaks} transcription factor binding sites
- {result.integration.tf_target_pairs} TF-target regulatory relationships identified
"""
```

---

## Directory Structure

```
BioPipelines/
├── containers/
│   ├── base/
│   │   ├── Dockerfile
│   │   ├── Singularity.def
│   │   └── manifest.json
│   ├── rna-seq/
│   │   ├── Dockerfile
│   │   ├── Singularity.def
│   │   ├── manifest.json
│   │   ├── entrypoint.sh
│   │   └── README.md
│   ├── dna-seq/
│   ├── chip-seq/
│   ├── atac-seq/
│   ├── methylation/
│   ├── hic/
│   ├── scrna-seq/
│   ├── long-read/
│   ├── metagenomics/
│   ├── structural-variants/
│   └── multi-omics/        # AI integration container
├── src/biopipelines/
│   ├── containers/
│   │   ├── __init__.py
│   │   ├── registry.py     # Container discovery
│   │   └── builder.py      # Automated builds
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py # AI agent orchestration
│   │   ├── composer.py     # Workflow composition
│   │   └── monitor.py      # Execution monitoring
│   └── core/
├── scripts/
│   ├── containers/
│   │   ├── build_all.sh
│   │   ├── push_to_registry.sh
│   │   └── convert_to_singularity.sh
│   └── submit_containerized.sh   # New SLURM submission
└── docs/
    ├── CONTAINER_ARCHITECTURE.md (this file)
    └── containers/
        ├── building.md
        ├── ai_integration.md
        └── multi_omics.md
```

---

## Advantages for AI Agents

### 1. **Discoverability**
Agents can query container registry to find capabilities:
```python
agent.search(category="transcriptomics") → [rna-seq, scrna-seq]
agent.search(capability="variant_calling") → [dna-seq, long-read]
```

### 2. **Composability**
Agents can chain containers to build custom workflows:
```
QC → Trim → Align → Call Variants → Annotate
 ↓      ↓      ↓         ↓            ↓
[base] [base] [align] [variant]  [annotation]
```

### 3. **Observability**
Containers emit structured logs for agent monitoring:
```json
{
  "step": "alignment",
  "progress": 0.75,
  "metrics": {"mapped_reads": 45000000, "unmapped": 5000000},
  "status": "running"
}
```

### 4. **Extensibility**
Adding new tools doesn't break existing agent workflows:
```dockerfile
# New container for single-cell multiomics
FROM biopipelines/base:1.0.0
RUN micromamba install -y seurat cellranger
# Agents automatically discover via manifest.json
```

### 5. **Reproducibility**
Agents can specify exact container versions:
```python
# Agent ensures reproducibility
agent.execute(
    container="biopipelines/rna-seq:1.0.0",  # Pinned version
    params={...}
)
```

---

## Migration Path

### Week 1: Foundation
- ✅ Create base container
- ✅ Containerize RNA-seq (pilot)
- ✅ Test on cluster
- ✅ Validate success

### Week 2: Scale Out
- Containerize DNA-seq, ChIP-seq, ATAC-seq
- Build container registry system
- Create AI agent integration layer

### Week 3: Advanced Features
- Containerize remaining pipelines
- Implement multi-omics integration
- Add AI agent orchestration

### Week 4: Production
- CI/CD for automated builds
- Documentation and training
- Full deployment

---

## Success Metrics

- ✅ 100% pipeline success rate (no conda issues)
- ✅ <30 second pipeline startup time
- ✅ AI agents can discover and invoke containers
- ✅ Multi-omics workflows composable
- ✅ Reproducible results (container version pinning)
- ✅ <10% time on environment issues

---

**Next Steps**: Begin implementation with RNA-seq pilot container.
