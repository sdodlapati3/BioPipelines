# BioPipelines Tutorials

Complete tutorials for using the AI Workflow Composer to generate bioinformatics pipelines.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Tutorial 1: RNA-seq Workflow](#tutorial-1-your-first-rna-seq-workflow)
3. [Tutorial 2: ChIP-seq Peak Calling](#tutorial-2-chip-seq-peak-calling)
4. [Tutorial 3: Variant Calling](#tutorial-3-variant-calling-pipeline)
5. [Tutorial 4: Single-cell RNA-seq](#tutorial-4-single-cell-rna-seq)
6. [Tutorial 5: Metagenomics](#tutorial-5-metagenomics-analysis)
7. [Tutorial 6: CLI Usage](#tutorial-6-using-the-cli)
8. [Tutorial 7: LLM Providers](#tutorial-7-llm-provider-selection)
9. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- BioPipelines installed (\`pip install -e .\`)
- Python 3.10+
- At least one LLM provider configured:
  - **OpenAI**: Set \`OPENAI_API_KEY\` environment variable
  - **vLLM**: Start vLLM server on GPU (see [LLM Setup](LLM_SETUP.md))
  - **Ollama**: Install and run Ollama locally

### Quick Test

\`\`\`python
# Test that workflow_composer is installed
from workflow_composer import Composer
from workflow_composer.llm import check_providers

# Check available LLM providers
available = check_providers()
print(f"Available providers: {available}")
# Example: {'openai': True, 'vllm': False, 'ollama': True, ...}
\`\`\`

---

## Tutorial 1: Your First RNA-seq Workflow

This tutorial walks through creating a complete RNA-seq differential expression workflow.

### Step 1: Import and Initialize

\`\`\`python
from workflow_composer import Composer
from workflow_composer.llm import get_llm

# Option A: Use OpenAI (requires OPENAI_API_KEY)
llm = get_llm("openai", model="gpt-4o")

# Option B: Use vLLM on GPU cluster
# llm = get_llm("vllm", model="meta-llama/Llama-3.1-8B-Instruct")

# Option C: Use Ollama locally
# llm = get_llm("ollama", model="llama3")

# Initialize composer
composer = Composer(llm=llm)
print(f"Using LLM: {composer.llm}")
\`\`\`

### Step 2: Describe Your Analysis

\`\`\`python
# Natural language description
description = """
RNA-seq differential expression analysis:
- Organism: Mouse (Mus musculus)
- Genome: GRCm39
- Data: Paired-end Illumina reads
- Comparison: Treatment vs Control (3 replicates each)
- Steps needed:
  1. Quality control with FastQC
  2. Adapter trimming with fastp
  3. Alignment with STAR to mouse genome
  4. Read counting with featureCounts
  5. Differential expression with DESeq2
  6. Generate MultiQC report
"""
\`\`\`

### Step 3: Generate the Workflow

\`\`\`python
# Generate workflow
workflow = composer.generate(
    description,
    output_dir="tutorials/rnaseq_de/"
)

print(f"Workflow generated: {workflow.name}")
print(f"Modules used: {[m.name for m in workflow.modules_used]}")
\`\`\`

### Step 4: Review Generated Files

\`\`\`
tutorials/rnaseq_de/
├── main.nf           # Main Nextflow workflow
├── nextflow.config   # Configuration (containers, resources)
├── samplesheet.csv   # Template sample sheet
├── README.md         # Documentation
└── modules/          # Symlinked/copied modules
\`\`\`

### Step 5: Run the Workflow

\`\`\`bash
# Local execution with Singularity
nextflow run tutorials/rnaseq_de/main.nf \
  -profile singularity \
  --input samplesheet.csv \
  --outdir results/

# On SLURM cluster
nextflow run tutorials/rnaseq_de/main.nf \
  -profile slurm,singularity \
  --input samplesheet.csv \
  --outdir results/
\`\`\`

---

## Tutorial 2: ChIP-seq Peak Calling

### Goal
Identify H3K4me3 peaks in human cells with input control.

### Generate Workflow

\`\`\`python
from workflow_composer import Composer
from workflow_composer.llm import get_llm

composer = Composer(llm=get_llm("openai"))

workflow = composer.generate("""
ChIP-seq peak calling analysis:
- Human samples (GRCh38)
- Histone mark: H3K4me3 (narrow peaks)
- Single-end 50bp reads
- Have input control samples
""", output_dir="tutorials/chipseq_peaks/")

print(f"Generated: {workflow.name}")
\`\`\`

---

## Tutorial 3: Variant Calling Pipeline

### Goal
Germline variant calling from whole genome sequencing data.

\`\`\`python
from workflow_composer import Composer
from workflow_composer.llm import get_llm

# Using vLLM with Llama model
llm = get_llm("vllm", model="llama3.1-8b")  # Model alias
composer = Composer(llm=llm)

workflow = composer.generate("""
WGS germline variant calling:
- Human samples, GRCh38 reference
- Paired-end 150bp Illumina
- Pipeline: FastQC, fastp, BWA-MEM, GATK HaplotypeCaller
""", output_dir="tutorials/wgs_variants/")
\`\`\`

---

## Tutorial 4: Single-cell RNA-seq

### Goal
Process 10X Genomics single-cell data.

\`\`\`python
from workflow_composer import Composer

composer = Composer()  # Uses default provider from config

workflow = composer.generate("""
10X Genomics single-cell RNA-seq analysis:
- Human PBMC samples
- 10X Genomics 3' v3 chemistry
- STARsolo quantification
- Clustering and cell type annotation
""", output_dir="tutorials/scrna_10x/")
\`\`\`

---

## Tutorial 5: Metagenomics Analysis

### Goal
Taxonomic profiling of microbiome samples.

\`\`\`python
from workflow_composer import Composer
from workflow_composer.llm import get_llm

composer = Composer(llm=get_llm("openai", model="gpt-4o"))

workflow = composer.generate("""
Shotgun metagenomics analysis:
- Human gut microbiome samples
- Paired-end Illumina data
- Kraken2 classification, Bracken abundance
""", output_dir="tutorials/metagenomics/")
\`\`\`

---

## Tutorial 6: Using the CLI

The \`biocomposer\` command-line tool provides quick access to workflow generation.

### Generate Workflow

\`\`\`bash
# Basic generation (uses default LLM from config)
biocomposer generate "RNA-seq DE analysis for mouse"

# Specify output directory
biocomposer generate "ChIP-seq peak calling" --output chipseq_workflow/

# Use specific LLM provider
biocomposer generate "WGS variant calling" --llm openai --model gpt-4o

# Use vLLM server
biocomposer generate "scRNA-seq analysis" --llm vllm --model llama3.1-8b
\`\`\`

### Search Tools

\`\`\`bash
# Search all tools
biocomposer tools --search "alignment"

# List all tools in container
biocomposer tools --container rna-seq
\`\`\`

### Interactive Chat

\`\`\`bash
biocomposer chat --llm openai

# Chat session:
You: I need to analyze ATAC-seq data
Assistant: Creating ATAC-seq workflow...
\`\`\`

### Check Providers

\`\`\`bash
biocomposer providers --check
# ✓ openai
# ✗ ollama
# ✓ vllm
\`\`\`

---

## Tutorial 7: LLM Provider Selection

### OpenAI (Recommended for Quality)

\`\`\`python
from workflow_composer.llm import get_llm

llm = get_llm("openai", model="gpt-4o")
# Models: gpt-4o (recommended), gpt-4-turbo, gpt-3.5-turbo
\`\`\`

### vLLM (Self-hosted GPU Inference)

\`\`\`python
from workflow_composer.llm import get_llm, VLLMAdapter

# Use model alias
llm = get_llm("vllm", model="llama3.1-8b")

# Get recommended models
print(VLLMAdapter.get_recommended_models())
\`\`\`

### HuggingFace (Multiple Backends)

\`\`\`python
from workflow_composer.llm import HuggingFaceAdapter

# Via vLLM server
llm = HuggingFaceAdapter(
    model="meta-llama/Llama-3.1-8B-Instruct",
    backend="vllm",
    vllm_url="http://localhost:8000"
)
\`\`\`

---

## Troubleshooting

### Common Issues

**1. "LLM provider not available"**

\`\`\`bash
# Check which providers work
biocomposer providers --check

# For OpenAI, set API key
export OPENAI_API_KEY="sk-..."

# For vLLM, ensure server is running
curl http://localhost:8000/health
\`\`\`

**2. "Module not found for tool X"**

\`\`\`python
from workflow_composer import Composer

composer = Composer()
modules = composer.module_mapper.list_by_category()
print(modules)
\`\`\`

**3. "Nextflow execution failed"**

\`\`\`bash
# Check logs and resume
cat .nextflow.log
nextflow run main.nf -resume
\`\`\`

---

## Next Steps

1. **Configure LLM** - [LLM Setup Guide](LLM_SETUP.md)
2. **Explore Patterns** - [Composition Patterns](COMPOSITION_PATTERNS.md) (27 examples)
3. **API Reference** - [Workflow Composer Guide](WORKFLOW_COMPOSER_GUIDE.md)
4. **Examples** - Check \`examples/generated/\` for pre-generated workflows
