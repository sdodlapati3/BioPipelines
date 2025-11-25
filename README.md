# BioPipelines

A comprehensive, production-ready bioinformatics repository for NGS and genomic data analysis pipelines.

## Features

### 🤖 AI-Powered Workflow Composer

Generate production-ready Nextflow pipelines from natural language descriptions:

```python
from workflow_composer import Composer
from workflow_composer.llm import get_llm

# Use OpenAI GPT-4o or local vLLM with Llama/Mistral
composer = Composer(llm=get_llm("openai"))  # or "vllm"

workflow = composer.generate(
    "RNA-seq differential expression for mouse, paired-end, treatment vs control"
)
workflow.save("my_rnaseq_workflow/")
```

**LLM Providers:**
- ✅ **OpenAI** - GPT-4o, GPT-4-turbo (cloud)
- ✅ **vLLM** - Llama 3.1, Mistral, Qwen on GPUs (self-hosted)
- ✅ **HuggingFace** - API, transformers, or vLLM backend
- ✅ **Anthropic** - Claude 3.5 Sonnet, Opus
- ✅ **Ollama** - Local models

See [LLM Setup Guide](docs/LLM_SETUP.md) for configuration.

### 🧬 10 Production-Ready Pipelines

(8 fully validated, 2 core complete):

- ✅ **DNA-seq**: Variant calling, structural variant detection (VALIDATED)
- ✅ **RNA-seq**: Differential expression, isoform analysis (VALIDATED)
- ✅ **scRNA-seq**: Single-cell analysis, clustering, cell-type annotation (VALIDATED)
- ✅ **ChIP-seq**: Peak calling, motif analysis, differential binding (VALIDATED)
- ✅ **ATAC-seq**: Chromatin accessibility, footprinting (VALIDATED)
- ⚠️ **Methylation**: WGBS/RRBS bisulfite sequencing analysis (CODE VALIDATED - needs production data)
- ⚠️ **Hi-C**: 3D genome organization, contact matrices (CORE COMPLETE - advanced tools optional)
- ✅ **Long-read**: Nanopore/PacBio structural variant detection (VALIDATED)
- ✅ **Metagenomics**: Taxonomic profiling with Kraken2 (VALIDATED)
- ✅ **Structural Variants**: Multi-tool SV calling pipeline (VALIDATED)

**Achievement**: 80% fully validated (8/10), 100% core functional (10/10)  
See `PIPELINE_STATUS_FINAL.md` for detailed validation report.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/BioPipelines.git
cd BioPipelines

# Create conda environment
conda env create -f environment.yml
conda activate biopipelines

# Install Python package
pip install -e .
```

### Running Your First Pipeline

**Option 1: Using Unified Scripts (Recommended)**

```bash
# Download test data
conda activate biopipelines
./scripts/download_data.py chipseq --test --output data/raw/chip_seq/

# Submit pipeline to SLURM
./scripts/submit_pipeline.sh --pipeline chip_seq --mem 32G --cores 8

# Check job status
squeue -u $USER

# View results
ls data/results/chip_seq/
```

**Option 2: Manual Execution**

```bash
# Navigate to pipeline directory
cd pipelines/dna_seq/variant_calling

# Edit config.yaml with your sample information
vim config.yaml

# Run with Snakemake
snakemake --cores 4
```

**Available Pipelines:**
- `atac_seq`, `chip_seq`, `dna_seq`, `rna_seq`, `scrna_seq`
- `methylation`, `hic`, `long_read`, `metagenomics`, `sv`

See `scripts/README.md` for detailed usage of unified scripts.

## Project Structure

```
BioPipelines/
├── src/workflow_composer/  # AI Workflow Composer (main package)
│   ├── llm/               # LLM adapters (OpenAI, vLLM, HuggingFace)
│   ├── core/              # Intent parsing, tool selection, workflow generation
│   ├── cli.py             # biocomposer CLI
│   └── composer.py        # Main Composer class
├── pipelines/             # Analysis pipelines (Snakemake workflows)
│   ├── dna_seq/           # Variant calling with GATK
│   ├── rna_seq/           # Differential expression with DESeq2
│   ├── scrna_seq/         # Single-cell analysis with Scanpy
│   ├── chip_seq/          # Peak calling with MACS2
│   └── ...                # More pipelines
├── containers/            # Singularity container definitions
├── config/                # Configuration files
│   └── composer.yaml      # Workflow Composer config
├── scripts/               # Utility scripts
│   └── llm/               # vLLM server scripts
├── data/                  # Data directory (gitignored)
├── docs/                  # Documentation
│   ├── LLM_SETUP.md       # LLM integration guide
│   ├── TUTORIALS.md       # Workflow Composer tutorials
│   └── COMPOSITION_PATTERNS.md  # 27 workflow patterns
├── examples/              # Example workflows
│   └── generated/         # AI-generated workflow examples
├── logs/                  # Job logs
└── tests/                 # Test suite
```

## AI Workflow Composer

### CLI Usage

```bash
# Generate workflow from natural language
biocomposer generate "ChIP-seq peak calling for human H3K4me3" -o chipseq_workflow/

# Interactive chat mode
biocomposer chat --llm openai

# Search available tools
biocomposer tools --search "alignment"

# List modules
biocomposer modules --list

# Check LLM providers
biocomposer providers --check
```

### Python API

```python
from workflow_composer import Composer
from workflow_composer.llm import get_llm, check_providers

# Check available providers
print(check_providers())
# {'openai': True, 'vllm': True, 'ollama': False, ...}

# Create composer with specific LLM
llm = get_llm("openai", model="gpt-4o")
composer = Composer(llm=llm)

# Generate and save workflow
workflow = composer.generate(
    "WGS germline variant calling for human samples"
)
workflow.save("variants_workflow/")
```

See [Workflow Composer Guide](docs/WORKFLOW_COMPOSER_GUIDE.md) for detailed documentation.

## Pipelines (Snakemake)

### DNA-seq Variant Calling
- Quality control (FastQC, MultiQC)
- Read trimming (fastp)
- Alignment (BWA-MEM)
- Variant calling (GATK, FreeBayes)
- Annotation (SnpEff, VEP)

### RNA-seq Differential Expression
- QC and trimming
- Alignment (STAR) or pseudo-alignment (Salmon)
- Quantification (featureCounts, RSEM)
- Differential expression (DESeq2, edgeR)
- Functional enrichment (GSEA)

## Documentation

- **[LLM Setup Guide](docs/LLM_SETUP.md)** - Configure OpenAI/vLLM
- **[Workflow Composer Guide](docs/WORKFLOW_COMPOSER_GUIDE.md)** - Full API reference
- **[Tutorials](docs/TUTORIALS.md)** - Step-by-step guides
- **[Composition Patterns](docs/COMPOSITION_PATTERNS.md)** - 27 workflow examples
- **[Architecture Review](ARCHITECTURE_REVIEW.md)** - Codebase organization

### Quick Links
- [LLM Setup](docs/LLM_SETUP.md)
- [Workflow Tutorials](docs/TUTORIALS.md)
- [Troubleshooting Guide](docs/status/CLEANUP_COMPLETED.md)

## Requirements

- Python >= 3.10
- Conda/Mamba
- Snakemake >= 7.30
- See `environment.yml` for complete list

## Contributing

Contributions are welcome! Please read our contributing guidelines.

## License

MIT License - see LICENSE file for details.

## Citation

If you use BioPipelines in your research, please cite:
```
[Citation to be added]
```

## Contact

For questions and support, please open an issue on GitHub.
# BioPipelines
