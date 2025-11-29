# BioPipelines AI Workflow Composer - User Guide

## Overview

The AI Workflow Composer is an intelligent system that converts natural language descriptions into production-ready Nextflow bioinformatics pipelines. It leverages:

- **9,909 tools** across 12 Singularity containers
- **71+ Nextflow DSL2 modules** covering all major analysis types
- **LLM-powered intent parsing** for natural language understanding
- **Automatic workflow generation** with proper dependencies

### Supported LLM Providers

| Provider | Models | Use Case |
|----------|--------|----------|
| **OpenAI** | gpt-4o, gpt-4-turbo, gpt-3.5-turbo | Best quality, cloud-based |
| **vLLM** | Llama 3.1, Mistral, Qwen, CodeLlama | Self-hosted GPU inference |
| **HuggingFace** | Any HF model | API, transformers, or vLLM backend |
| **Anthropic** | Claude 3.5 Sonnet, Opus | Alternative cloud provider |
| **Ollama** | llama3, mistral, codellama | Local CPU/GPU inference |

See [LLM Setup Guide](LLM_SETUP.md) for detailed configuration.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements-composer.txt

# Verify installation
python -c "from workflow_composer import Composer; print('OK')"
```

### Generate Your First Workflow

```python
from workflow_composer import Composer
from workflow_composer.llm import get_llm

# Use OpenAI (requires OPENAI_API_KEY)
llm = get_llm("openai", model="gpt-4o")

# Or use vLLM with Llama (requires running vLLM server)
# llm = get_llm("vllm", model="llama3.1-8b")

# Initialize composer
composer = Composer(llm=llm)

# Generate workflow from natural language
workflow = composer.generate(
    "RNA-seq differential expression analysis for mouse samples"
)

# Save workflow
workflow.save("my_rnaseq_workflow/")
```

### Using the CLI

```bash
# Generate a workflow
biocomposer generate "ChIP-seq peak calling for human H3K4me3"

# Use specific LLM provider
biocomposer generate "RNA-seq DE" --llm openai --model gpt-4o
biocomposer generate "RNA-seq DE" --llm vllm --model llama3.1-8b

# Interactive chat mode
biocomposer chat --llm openai

# List available tools
biocomposer tools --search "alignment"

# List available modules
biocomposer modules --list

# Check LLM providers
biocomposer providers --check
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Natural Language Input                    │
│         "RNA-seq DE analysis for mouse samples"             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Intent Parser (LLM)                     │
│  • Analysis type: RNA-seq                                   │
│  • Organism: Mouse                                          │
│  • Goal: Differential expression                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Tool Selector                          │
│  Query 9,909 tools from 12 containers                       │
│  • FastQC (QC)                                              │
│  • STAR (Alignment)                                         │
│  • featureCounts (Quantification)                           │
│  • DESeq2 (Differential Expression)                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Module Mapper                           │
│  Map tools to 71 Nextflow modules                           │
│  • fastqc → qc/fastqc/main.nf                              │
│  • star → alignment/star/main.nf                           │
│  • featurecounts → quantification/featurecounts/main.nf    │
│  • deseq2 → analysis/deseq2/main.nf                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Workflow Generator                         │
│  Generate complete Nextflow DSL2 workflow                   │
│  • main.nf                                                  │
│  • nextflow.config                                          │
│  • params.yaml                                              │
└─────────────────────────────────────────────────────────────┘
```

## Supported Analysis Types

### RNA-seq
- Bulk RNA-seq differential expression
- Transcript quantification (Salmon, RSEM, kallisto)
- Splicing analysis (rMATS, SUPPA)
- Gene fusion detection

### DNA-seq / WGS / WES
- Germline variant calling (GATK, FreeBayes)
- Somatic variant calling
- Copy number variation
- Structural variant detection

### ChIP-seq / ATAC-seq
- Peak calling (MACS2, HOMER, SICER)
- Differential binding analysis
- Motif analysis (MEME Suite)
- Signal visualization (deepTools)

### Single-cell RNA-seq
- 10X Genomics (STARsolo, Cell Ranger)
- Clustering and annotation (Seurat, Scanpy)
- Trajectory analysis

### Metagenomics
- Taxonomic profiling (Kraken2, MetaPhlAn)
- Assembly (MEGAHIT, SPAdes)
- Functional annotation

### Methylation / Epigenomics
- Bisulfite sequencing (Bismark)
- Differential methylation

### Long-read Sequencing
- ONT/PacBio alignment (minimap2)
- Assembly (Flye, Canu)
- Polishing (Racon)

### Hi-C
- Contact map generation
- TAD calling
- Loop detection

## Module Library (71 Modules)

### Alignment (11 modules)
| Module | Description |
|--------|-------------|
| star | STAR RNA-seq aligner |
| bwamem | BWA-MEM DNA aligner |
| bowtie2 | Bowtie2 short-read aligner |
| hisat2 | HISAT2 spliced aligner |
| minimap2 | Long-read aligner |
| bismark | Bisulfite aligner |
| bowtie | Bowtie aligner |
| gsnap | GSNAP aligner |
| subread | Subread aligner |
| tophat2 | TopHat2 (legacy) |
| bwa | BWA classic |

### Quantification (6 modules)
| Module | Description |
|--------|-------------|
| featurecounts | Read counting |
| salmon | Transcript quantification |
| kallisto | Pseudoalignment quantification |
| rsem | RNA-seq quantification |
| htseq | HTSeq-count |
| cufflinks | Cufflinks (legacy) |

### Variant Calling (6 modules)
| Module | Description |
|--------|-------------|
| gatk_haplotypecaller | GATK HaplotypeCaller |
| freebayes | FreeBayes caller |
| bcftools | BCFtools mpileup/call |
| varscan | VarScan2 |
| lofreq | LoFreq caller |
| gatk | GATK utilities |

### QC (6 modules)
| Module | Description |
|--------|-------------|
| fastqc | FastQC quality control |
| multiqc | MultiQC aggregation |
| qualimap | Qualimap BAM QC |
| rseqc | RSeQC RNA-seq QC |
| mosdepth | Coverage analysis |
| bbtools | BBTools suite |

### Trimming (5 modules)
| Module | Description |
|--------|-------------|
| fastp | Ultra-fast preprocessor |
| cutadapt | Adapter trimming |
| trim_galore | Trim Galore! |
| trimmomatic | Trimmomatic |
| trimgalore | Trim Galore (alias) |

### Analysis (5 modules)
| Module | Description |
|--------|-------------|
| deseq2 | Differential expression |
| edger | EdgeR DE analysis |
| gsea | Gene set enrichment |
| deeptools | Signal visualization |
| meme_suite | Motif analysis |

### Peak Calling (3 modules)
| Module | Description |
|--------|-------------|
| macs2 | MACS2 peak caller |
| homer | HOMER peak calling |
| sicer | SICER broad peaks |

### Single-cell (4 modules)
| Module | Description |
|--------|-------------|
| starsolo | STARsolo 10X |
| cellranger | Cell Ranger |
| seurat | Seurat analysis |
| scanpy | Scanpy analysis |

### Assembly (5 modules)
| Module | Description |
|--------|-------------|
| spades | SPAdes assembler |
| trinity | Trinity RNA-seq |
| flye | Flye long-read |
| canu | Canu assembler |
| stringtie | StringTie |

### Metagenomics (4 modules)
| Module | Description |
|--------|-------------|
| kraken2 | Kraken2 classifier |
| bracken | Bracken abundance |
| metaphlan | MetaPhlAn profiler |
| megahit | MEGAHIT assembler |

### Hi-C (4 modules)
| Module | Description |
|--------|-------------|
| pairtools_parse | Pairtools parsing |
| cooler_cload | Cooler loading |
| hicpro | HiC-Pro pipeline |
| juicer | Juicer tools |

### Utilities (4 modules)
| Module | Description |
|--------|-------------|
| samtools | SAM/BAM utilities |
| bedtools | BED utilities |
| picard | Picard tools |
| racon | Racon polishing |

### Structural Variants (2 modules)
| Module | Description |
|--------|-------------|
| manta | Manta SV caller |
| delly | Delly SV caller |

### Annotation (3 modules)
| Module | Description |
|--------|-------------|
| prokka | Prokaryotic annotation |
| augustus | Gene prediction |
| blast | BLAST search |

### Methylation (2 modules)
| Module | Description |
|--------|-------------|
| bismark_extractor | Methylation extraction |
| medips | MeDIP-seq analysis |

## Container Infrastructure

| Container | Tools | Size |
|-----------|-------|------|
| rna-seq | 847 | ~2.5GB |
| dna-seq | 823 | ~2.8GB |
| chip-seq | 756 | ~2.2GB |
| atac-seq | 689 | ~2.1GB |
| scrna-seq | 612 | ~3.2GB |
| metagenomics | 534 | ~2.4GB |
| methylation | 445 | ~1.8GB |
| long-read | 398 | ~1.9GB |
| hic | 356 | ~1.5GB |
| structural-variants | 312 | ~1.4GB |
| workflow-engine | 287 | ~1.2GB |
| base | 1850 | ~1.0GB |

## Configuration

### config/composer.yaml

```yaml
llm:
  default_provider: ollama
  providers:
    ollama:
      host: http://localhost:11434
      model: llama3:8b
    openai:
      api_key: ${OPENAI_API_KEY}
      model: gpt-4-turbo-preview
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
      model: claude-3-opus-20240229

knowledge_base:
  tool_catalog: data/tool_catalog
  module_library: nextflow-pipelines/modules
  workflow_patterns: docs/COMPOSITION_PATTERNS.md

output:
  workflow_dir: generated_workflows/
  log_level: INFO
```

## Running Generated Workflows

### On SLURM HPC

```bash
# Use the provided launcher script
./scripts/run_nextflow.sh my_workflow/main.nf -profile slurm,singularity

# Or directly with Nextflow
nextflow run my_workflow/main.nf \
  -profile slurm,singularity \
  --input samples.csv \
  --outdir results/
```

### Locally with Singularity

```bash
nextflow run my_workflow/main.nf \
  -profile singularity \
  --input samples.csv \
  --outdir results/
```

## API Reference

### Composer Class

```python
from src.workflow_composer import Composer

composer = Composer(
    config_path="config/composer.yaml",  # Optional
    llm_provider="ollama"                 # Optional override
)

# Generate workflow
workflow = composer.generate(
    prompt="Your analysis description",
    output_dir="output/",                 # Optional
    dry_run=False                         # Set True to preview
)

# Access components directly
tools = composer.tool_selector.search("alignment")
modules = composer.module_mapper.list_modules()
```

### WorkflowGenerator Class

```python
from src.workflow_composer.core import WorkflowGenerator

generator = WorkflowGenerator(module_dir="nextflow-pipelines/modules")

workflow = generator.generate(
    analysis_type="rnaseq",
    tools=["fastqc", "star", "featurecounts", "deseq2"],
    organism="mouse",
    params={"strandedness": "reverse"}
)
```

### ToolSelector Class

```python
from src.workflow_composer.core import ToolSelector

selector = ToolSelector(catalog_path="data/tool_catalog")

# Search tools
results = selector.search("variant calling", limit=10)

# Get tool info
tool = selector.get_tool("gatk")
print(tool.container, tool.version)
```

### ModuleMapper Class

```python
from src.workflow_composer.core import ModuleMapper

mapper = ModuleMapper("nextflow-pipelines/modules")

# Find module for tool
module = mapper.find_module("star")
print(module.path, module.processes)

# List all modules
for name in mapper.list_modules():
    print(name)
```

## Examples

### Example 1: RNA-seq DE Analysis

```python
from src.workflow_composer import Composer

composer = Composer()

workflow = composer.generate("""
Perform RNA-seq differential expression analysis:
- Mouse samples, paired-end reads
- Compare treatment vs control
- Use STAR for alignment
- DESeq2 for differential expression
- Generate QC reports with MultiQC
""")

workflow.save("rnaseq_de/")
```

### Example 2: Variant Calling Pipeline

```python
workflow = composer.generate("""
WGS germline variant calling pipeline for human samples:
- BWA-MEM alignment to GRCh38
- Mark duplicates with Picard
- GATK HaplotypeCaller for variant calling
- VCF annotation with VEP
""")
```

### Example 3: ChIP-seq Analysis

```python
workflow = composer.generate("""
ChIP-seq analysis for H3K27ac in human cells:
- Bowtie2 alignment
- MACS2 peak calling
- deepTools for signal visualization
- HOMER for motif analysis
""")
```

## Troubleshooting

### Common Issues

**1. LLM not available**
```
Solution: Set up Ollama locally or configure OpenAI/Anthropic API keys
```

**2. Module not found**
```python
# Check available modules
mapper.list_modules()

# Check aliases
print(mapper.TOOL_ALIASES)
```

**3. Container not available**
```bash
# Build container
cd containers/rna-seq && singularity build ...
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on:
- Adding new modules
- Extending tool catalog
- Improving LLM prompts

## License

MIT License - See [LICENSE](../LICENSE)
