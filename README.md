# BioPipelines

A comprehensive, production-ready bioinformatics repository for NGS and genomic data analysis pipelines.

## Features

**10 Production-Ready Pipelines** (8 fully validated, 2 core complete):

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
├── pipelines/          # Analysis pipelines (Snakemake workflows)
│   ├── dna_seq/       # Variant calling with GATK
│   ├── rna_seq/       # Differential expression with DESeq2
│   ├── scrna_seq/     # Single-cell analysis with Scanpy
│   ├── chip_seq/      # Peak calling with MACS2
│   ├── atac_seq/      # Accessibility analysis
│   ├── methylation/   # Bisulfite sequencing
│   ├── hic/           # 3D genome analysis
│   ├── long_read/     # Long-read SV detection
│   ├── metagenomics/  # Taxonomic profiling
│   └── structural_variants/  # SV calling
├── src/                # Python package (pip install -e .)
│   └── biopipelines/  # Reusable modules
├── scripts/            # Utility scripts (download, submit, build)
├── data/               # Data directory (gitignored)
│   ├── raw/           # Input FASTQ files
│   ├── processed/     # Intermediate files
│   ├── references/    # Genomes, indexes, annotations
│   └── results/       # Final outputs
├── docs/               # Documentation
│   ├── tutorials/     # Step-by-step guides
│   ├── pipelines/     # Pipeline documentation
│   └── status/        # Development status
├── logs/               # Job logs organized by type
├── tests/              # Test suite
└── notebooks/          # Jupyter notebooks for exploration
```

## Pipelines

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

- **[Architecture Review](ARCHITECTURE_REVIEW.md)** - Codebase organization and structure
- **[Tutorials](docs/tutorials/)** - Step-by-step pipeline guides
- **[Pipeline Status](docs/status/)** - Development and validation status
- **[Infrastructure](docs/infrastructure/)** - HPC and cloud setup guides
- **[API Reference](docs/api/)** - Python module documentation

### Quick Links
- [DNA-seq Tutorial](docs/tutorials/dna_seq_tutorial.md)
- [RNA-seq Tutorial](docs/tutorials/rna_seq_tutorial.md)
- [scRNA-seq Tutorial](docs/tutorials/scrna_seq_tutorial.md)
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
