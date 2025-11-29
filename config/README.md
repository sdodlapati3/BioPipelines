# BioPipelines Configuration Directory

This directory contains global configuration files used across all BioPipelines.

## Configuration Files

### `defaults.yaml`
Global default settings for all pipelines, including:
- Reference genome locations
- Common tool parameters (FastQC, fastp, BWA, STAR, etc.)
- Quality control thresholds
- Preprocessing settings
- Conda environment specifications

**Usage**: Automatically loaded by pipelines. Override specific settings in pipeline-specific `config.yaml` files.

### `slurm.yaml`
SLURM cluster configuration and resource allocation defaults:
- Default memory, CPU cores, and time limits
- Pipeline-specific resource overrides
- Job notification settings
- Log file locations

**Usage**: Used by `scripts/submit_pipeline.sh` to set SLURM parameters. Override with command-line flags:
```bash
./scripts/submit_pipeline.sh --pipeline chip_seq --mem 64G --cores 16 --time 12:00:00
```

## Configuration Hierarchy

1. **Global defaults** (`config/defaults.yaml`) - Lowest priority
2. **SLURM defaults** (`config/slurm.yaml`) - For cluster submission
3. **Pipeline-specific** (`pipelines/<name>/config.yaml`) - Medium priority
4. **Command-line arguments** - Highest priority

Example:
```bash
# Uses defaults from config/defaults.yaml and config/slurm.yaml
./scripts/submit_pipeline.sh --pipeline dna_seq

# Overrides memory setting from slurm.yaml
./scripts/submit_pipeline.sh --pipeline dna_seq --mem 128G

# Pipeline's config.yaml can override specific tool parameters
```

## Customization

### For Your Environment

Edit paths in `defaults.yaml`:
```yaml
paths:
  base_dir: "/scratch/${USER}/BioPipelines"  # Change to your scratch directory
  home_dir: "~/BioPipelines"                  # Change to your home location
```

Edit SLURM settings in `slurm.yaml`:
```yaml
cluster:
  partition: "cpuspot"           # Change to your cluster's partition name
  gpu_partition: "h100quadflex"  # GPU partition for model inference
  account: "my_account"          # Add your account if required
```

### For Specific Pipelines

Create or edit pipeline-specific configs:
```bash
nano pipelines/chip_seq/config.yaml
```

Override specific parameters:
```yaml
# pipelines/chip_seq/config.yaml
samples:
  - sample1
  - sample2

# Override global defaults
qc:
  fastqc:
    threads: 4  # Override default of 2

peak_calling:
  macs2:
    q_value: 0.01  # More stringent than default 0.05
```

## Reference Genome Setup

To use a reference genome:

1. Download and index the genome (see `scripts/download_references.sh`)
2. Update `config/defaults.yaml` with paths:
```yaml
references:
  human:
    hg38:
      genome: "data/references/genomes/human/hg38.fa"
      star_index: "data/references/indexes/star_hg38"
```

3. Reference in pipeline-specific configs:
```yaml
# pipelines/rna_seq/config.yaml
reference:
  genome: "hg38"  # Uses path from defaults.yaml
```

## Adding New Pipelines

When creating a new pipeline:

1. Create pipeline directory: `pipelines/<new_pipeline>/`
2. Create config: `pipelines/<new_pipeline>/config.yaml`
3. Reference global settings:
```yaml
# Include global settings (automatically loaded)
# Add pipeline-specific overrides below

samples:
  - sample1
  - sample2

# Use global reference
reference:
  genome: "hg38"  # Pulls from config/defaults.yaml

# Override specific settings
qc:
  min_quality: 30  # More stringent than global default
```

4. Add to `config/slurm.yaml` if it needs special resources:
```yaml
resources:
  new_pipeline:
    mem: "64G"
    cores: 16
    time: "24:00:00"
```

## Environment Variables

Use environment variables in configs:
```yaml
paths:
  base_dir: "/scratch/${USER}/BioPipelines"  # Expands to /scratch/username/BioPipelines
  reference_dir: "${REFERENCE_DIR}"           # Use environment variable if set
```

Set environment variables:
```bash
export REFERENCE_DIR="/shared/references"
./scripts/submit_pipeline.sh --pipeline chip_seq
```

## Best Practices

1. **Don't modify global configs** for one-off runs. Use command-line overrides instead.
2. **Keep sensitive info out of configs**. Use environment variables for credentials.
3. **Document custom settings** in pipeline-specific READMEs.
4. **Version control configs** to track changes over time.
5. **Test config changes** with `--dry-run` before submitting jobs.

## Troubleshooting

### Config Not Loading
```bash
# Check if config file exists
ls config/defaults.yaml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/defaults.yaml'))"
```

### Path Issues
```bash
# Verify paths exist
ls /scratch/${USER}/BioPipelines/data/references/genomes/human/hg38.fa

# Check environment variable expansion
echo $USER
echo /scratch/${USER}/BioPipelines
```

### Override Not Working
- Check configuration hierarchy (command-line > pipeline-specific > global)
- Ensure YAML indentation is correct (use spaces, not tabs)
- Check for typos in parameter names

## Further Reading

- [YAML Syntax Guide](https://yaml.org/spec/1.2/spec.html)
- [Snakemake Configuration](https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html)
- [SLURM Sbatch Options](https://slurm.schedmd.com/sbatch.html)

---

*Last updated: November 2025*
