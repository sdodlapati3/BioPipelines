# BioPipelines Scripts

This directory contains unified scripts for data download and pipeline submission.

## Core Scripts

### `download_data.py` - Unified Data Download
Replaces 25+ redundant download scripts with a single CLI tool.

**Prerequisites:**
```bash
# Activate conda environment (contains required packages)
conda activate biopipelines
```

**Usage:**
```bash
# Download ChIP-seq from ENCODE
./download_data.py chipseq --accession ENCSR000EUA --output data/raw/chip_seq/

# Download RNA-seq test data
./download_data.py rnaseq --test --output data/raw/rna_seq/

# Download methylation data
./download_data.py methylation --accession ENCSR000AKP --output data/raw/methylation/

# Download Hi-C data
./download_data.py hic --test --output data/raw/hic/

# See all options
./download_data.py --help
./download_data.py chipseq --help  # Pipeline-specific help
```

**Supported Pipelines:**
- `chipseq` - ChIP-seq from ENCODE
- `rnaseq` - RNA-seq from ENCODE
- `atacseq` - ATAC-seq from ENCODE
- `methylation` - Whole-genome bisulfite sequencing
- `hic` - Hi-C chromatin interaction data
- `metagenomics` - Metagenomic sequencing data
- `longread` - Long-read sequencing (PacBio, Oxford Nanopore)
- `scrna` - Single-cell RNA-seq

---

### `submit_pipeline.sh` - Unified Pipeline Submission
Replaces 18 submit scripts with a single configurable tool.

**Usage:**
```bash
# Submit with defaults (32G, 8 cores, 6 hours)
./submit_pipeline.sh --pipeline chip_seq

# Submit with custom resources
./submit_pipeline.sh --pipeline methylation --mem 48G --cores 16 --time 08:00:00

# Submit with simple configuration (faster, less comprehensive)
./submit_pipeline.sh --pipeline dna_seq --config simple

# Dry run (show what would be submitted)
./submit_pipeline.sh --pipeline rna_seq --dry-run

# Rerun incomplete jobs
./submit_pipeline.sh --pipeline atac_seq --rerun
```

**Pipeline-Specific Defaults:**

| Pipeline      | Memory | Cores | Time     |
|--------------|--------|-------|----------|
| atac_seq     | 32G    | 8     | 06:00:00 |
| chip_seq     | 32G    | 8     | 06:00:00 |
| dna_seq      | 32G    | 8     | 06:00:00 |
| rna_seq      | 32G    | 8     | 06:00:00 |
| scrna_seq    | 64G    | 16    | 08:00:00 |
| methylation  | 48G    | 12    | 08:00:00 |
| hic          | 64G    | 16    | 10:00:00 |
| long_read    | 64G    | 16    | 12:00:00 |
| metagenomics | 128G   | 32    | 12:00:00 |
| sv           | 48G    | 12    | 08:00:00 |

**Options:**
```
--pipeline NAME       Pipeline to run (required)
--config TYPE         simple|full (default: full)
--partition NAME      SLURM partition (default: cpuspot)
--mem SIZE           Memory allocation (e.g., 32G, 64G)
--cores NUM          CPU cores
--time DURATION      Time limit (HH:MM:SS)
--rerun              Rerun incomplete jobs
--dry-run            Show what would be submitted
--help               Show help message
```

---

## Migration Guide

### Old Scripts → New Commands

**Download Scripts:**
```bash
# OLD: download_chipseq_encode.py
# NEW:
./download_data.py chipseq --accession ENCSR000EUA --output data/raw/chip_seq/

# OLD: download_test_datasets.py
# NEW:
./download_data.py rnaseq --test --output data/raw/rna_seq/

# OLD: download_methylation_test.py
# NEW:
./download_data.py methylation --test --output data/raw/methylation/
```

**Submit Scripts:**
```bash
# OLD: submit_chip_seq.sh
# NEW:
./submit_pipeline.sh --pipeline chip_seq

# OLD: submit_methylation_simple.sh
# NEW:
./submit_pipeline.sh --pipeline methylation --config simple

# OLD: submit_rna_seq.sh (with hardcoded 48G)
# NEW:
./submit_pipeline.sh --pipeline rna_seq --mem 48G
```

### Deprecated Scripts

Old scripts are preserved in `scripts/deprecated/` for backward compatibility but will be removed in v0.2.0 (January 2026).

**Timeline:**
- **Current (v0.1.x)**: Old scripts work, deprecation warnings shown
- **v0.2.0 (Jan 2026)**: Old scripts removed, new scripts only

---

## Directory Structure

```
scripts/
├── download_data.py          # Unified download CLI
├── submit_pipeline.sh         # Unified submission script
├── README.md                  # This file
├── deprecated/
│   ├── README.md              # Migration guide
│   ├── download_*.py          # Old download scripts (25+)
│   └── submit_*.sh            # Old submit scripts (18)
├── download_references.sh     # Reference genome downloads (kept - different purpose)
├── download_annotations.sh    # Annotation downloads (kept)
└── build_*.sh                 # Index building scripts (kept)
```

---

## Benefits of New Scripts

### Consolidation
- **Before**: 25+ download scripts, 18 submit scripts (43 total)
- **After**: 2 unified scripts
- **Reduction**: 95% fewer scripts to maintain

### Flexibility
- **Configurable resources**: Adjust memory, cores, time per job
- **Multiple configs**: Simple (fast) vs full (comprehensive)
- **Dry-run mode**: Test before submitting
- **Rerun support**: Continue from failed steps

### Consistency
- **Unified interface**: Same pattern for all pipelines
- **Clear documentation**: Built-in help messages
- **Logging**: Organized in logs/slurm/active/
- **Error handling**: Validation before submission

### Maintainability
- **Single codebase**: Updates apply to all pipelines
- **Version control**: Easier to track changes
- **Testing**: One script to test, not 43
- **Documentation**: Centralized usage information

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'requests'"
Solution: Activate the conda environment first:
```bash
conda activate biopipelines
./download_data.py --help
```

### "Permission denied"
Solution: Make scripts executable:
```bash
chmod +x scripts/download_data.py scripts/submit_pipeline.sh
```

### "Pipeline-specific Snakefile not found"
Solution: Ensure you're running from the repository root:
```bash
cd ~/BioPipelines
./scripts/submit_pipeline.sh --pipeline chip_seq
```

### "SLURM job fails immediately"
Solution: Check logs in `logs/slurm/active/`:
```bash
ls -lt logs/slurm/active/  # Find latest log
tail -50 logs/slurm/active/<pipeline>_<timestamp>.err
```

---

## Contributing

When adding new pipelines:

1. **Add to submit_pipeline.sh**: Add pipeline name to case statement, set defaults
2. **Add to download_data.py**: Create subcommand, add to help text
3. **Update this README**: Add examples and defaults table entry
4. **Test both scripts**: Ensure --help works and dry-run succeeds

---

## Questions?

See:
- **Full architecture**: `ARCHITECTURE_REVIEW.md` (repository root)
- **Pipeline tutorials**: `docs/tutorials/` (pipeline-specific guides)
- **Development status**: `DEVELOPMENT_STATUS.md` (feature roadmap)
