# BioPipelines Base Container

**Version**: 1.0.0  
**Type**: Foundation container  
**Size**: ~1 GB

## Purpose

Base container providing common bioinformatics tools shared across all pipeline-specific containers. This reduces duplication and ensures consistency.

## Included Tools

- **samtools 1.17**: SAM/BAM/CRAM file manipulation
- **bcftools 1.17**: VCF/BCF file manipulation
- **bedtools 2.30.0**: Genomic interval operations
- **fastqc 0.12.1**: Quality control for sequencing reads
- **multiqc 1.14**: Aggregate QC report generation
- **Python 3.11**: With scientific libraries (numpy, pandas, scipy, matplotlib, biopython, pysam)

## Building

### Docker
```bash
cd containers/base
docker build -t biopipelines/base:1.0.0 .
```

### Singularity (for HPC)
```bash
singularity build base_1.0.0.sif docker-daemon://biopipelines/base:1.0.0
# Or directly from Dockerfile
singularity build base_1.0.0.sif Dockerfile
```

## Usage

### Docker
```bash
# Interactive shell
docker run -it -v $PWD:/analysis biopipelines/base:1.0.0

# Run specific command
docker run biopipelines/base:1.0.0 samtools --version
```

### Singularity
```bash
# Interactive shell
singularity shell base_1.0.0.sif

# Execute command
singularity exec base_1.0.0.sif samtools --version

# Run with bind mounts
singularity exec -B /data:/data base_1.0.0.sif fastqc /data/sample.fastq.gz
```

## Extending

Pipeline-specific containers inherit from this base:

```dockerfile
FROM biopipelines/base:1.0.0

# Add pipeline-specific tools
RUN micromamba install -y -n base \
    star=2.7.10b \
    salmon=1.10.1
```

## AI Agent Integration

This container exposes capabilities through `manifest.json` for AI agent discovery:

```python
from biopipelines.containers import ContainerRegistry

registry = ContainerRegistry("containers/")
base = registry.get_container("biopipelines-base")

# AI agent discovers available tools
tools = base.data["tools"]  # List of all tools
capabilities = base.data["capabilities"]  # High-level capabilities
```

## Maintenance

- **Update frequency**: Quarterly or when critical tool updates available
- **Versioning**: Semantic versioning (MAJOR.MINOR.PATCH)
- **Dependencies**: All pipeline containers depend on this version
