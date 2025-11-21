# BioPipelines Data Download Module

Unified Python package for downloading genomics datasets from multiple public databases.

## Features

- **Multiple data sources**: ENCODE, SRA/ENA, GEO, 1000 Genomes
- **Smart downloads**: Automatic retry, progress reporting, integrity checking
- **Search capability**: Find datasets matching your criteria
- **CLI and Python API**: Use from command line or import in scripts
- **Fast downloads**: Uses ENA FTP (faster than SRA) when available

## Installation

```bash
# Install base package
pip install -e .

# Install with all optional dependencies
pip install -e ".[download]"

# Or install dependencies manually
pip install pysradb requests pandas
conda install -c bioconda sra-tools  # Optional, for SRA downloads
```

## Quick Start

### Command Line Interface

```bash
# Download from SRA/ENA
biopipes-download sra SRR891268 --type atac_seq --output-dir data/raw

# Download from ENCODE
biopipes-download encode ENCFF001NQP --type chip_seq

# Download entire ENCODE experiment
biopipes-download encode-experiment ENCSR000AED --type rna_seq

# Search for datasets
biopipes-download search --source encode --query "H3K4me3" --organism human --limit 10

# List available data sources
biopipes-download list-sources
```

### Python API

```python
from biopipelines.data_download import DataDownloader

# Initialize downloader
downloader = DataDownloader(output_dir="data/raw")

# Download from SRA (uses ENA FTP for speed)
files = downloader.download_sra("SRR891268", dataset_type="atac_seq")
print(f"Downloaded: {files}")

# Download from ENCODE
file_path = downloader.download_encode("ENCFF001NQP", dataset_type="chip_seq")

# Search for datasets
results = downloader.search_datasets(
    query="H3K4me3 ChIP-seq",
    source="encode",
    organism="human",
    limit=5
)

for result in results:
    print(f"{result['accession']}: {result['title']}")
```

## Supported Data Sources

| Source | Description | Download Method |
|--------|-------------|-----------------|
| **ENCODE** | ENCODE Project | REST API + direct download |
| **SRA** | NCBI Sequence Read Archive | pysradb + sra-tools |
| **ENA** | European Nucleotide Archive | Direct FTP (fastest) |
| **GEO** | Gene Expression Omnibus | pysradb |
| **1000 Genomes** | 1000 Genomes Project | Direct FTP |

## Dataset Types

- `dna_seq`: Whole genome sequencing, exome
- `rna_seq`: RNA sequencing
- `chip_seq`: ChIP-seq
- `atac_seq`: ATAC-seq
- `wgs`: Whole genome sequencing
- `exome`: Exome sequencing

## Advanced Usage

### Batch Download from GEO Series

```python
from biopipelines.data_download import DataDownloader

downloader = DataDownloader(output_dir="data/raw/rna_seq")

# Download all samples from a GEO series
files = downloader.download_geo_series("GSE12345", dataset_type="rna_seq")
```

### Search and Download Workflow

```python
from biopipelines.data_download import DataDownloader

downloader = DataDownloader()

# Search for ATAC-seq datasets
results = downloader.search_datasets(
    query="ATAC-seq liver",
    source="sra",
    organism="mouse",
    limit=10
)

# Download top result
if results:
    best_match = results[0]
    accession = best_match['accession']
    files = downloader.download_sra(accession, dataset_type="atac_seq")
```

### Custom Download Function

```python
from biopipelines.data_download import download_from_url
from pathlib import Path

# Download from any URL with retry logic
file_path = download_from_url(
    url="https://example.com/data/sample.fastq.gz",
    output_path=Path("data/raw/sample.fastq.gz"),
    retry_attempts=3,
    timeout=300
)
```

## Integration with Pipelines

### Example: Download and Run Pipeline

```bash
#!/bin/bash
# download_and_run_chipseq.sh

# Search and download ChIP-seq data
biopipes-download search \
    --source encode \
    --query "H3K4me3 K562" \
    --type chip_seq \
    --limit 3

# Download specific experiment
biopipes-download encode-experiment ENCSR000AKP \
    --type chip_seq \
    --output-dir data/raw/chip_seq

# Update pipeline config
cat > pipelines/chip_seq/peak_calling/config.yaml <<EOF
samples:
  - sample1
  - sample2

reference:
  genome: "references/genomes/hg38/hg38.fa"
EOF

# Run pipeline
cd pipelines/chip_seq/peak_calling
sbatch ../../../scripts/submit_chip_seq.sh
```

## API Reference

### DataDownloader

Main class for downloading datasets.

**Methods:**
- `download_sra(accession, dataset_type, use_aspera=False)`: Download from SRA/ENA
- `download_encode(file_id, dataset_type)`: Download from ENCODE
- `download_geo_series(geo_id, dataset_type)`: Download GEO series
- `search_datasets(query, source, dataset_type, organism, limit)`: Search for datasets

### Helper Functions

- `download_from_url(url, output_path, retry_attempts, timeout)`: Download with retry
- `search_sra(query, organism, dataset_type, limit)`: Search SRA
- `search_encode(query, organism, dataset_type, limit)`: Search ENCODE

## Dependencies

**Required:**
- `requests`: HTTP downloads
- `pandas`: Data manipulation

**Optional but recommended:**
- `pysradb`: SRA/GEO access (pip install pysradb)
- `sra-tools`: SRA downloads (conda install -c bioconda sra-tools)
- `aspera-cli`: Faster downloads (conda install -c hcc aspera-cli)

## Troubleshooting

### SRA downloads failing

```bash
# Make sure sra-tools is installed
conda install -c bioconda sra-tools

# Or use ENA (automatic fallback)
# ENA is faster and more reliable
```

### ENCODE 404 errors

Some old ENCODE file IDs may be deprecated. Use the search function to find current datasets:

```bash
biopipes-download search --source encode --query "your_query"
```

### Slow downloads

```bash
# Use Aspera for faster SRA downloads
conda install -c hcc aspera-cli
biopipes-download sra SRR123456 --type rna_seq --aspera

# Or use ENA (default, already fast)
```

## Examples

See `examples/data_download/` for complete examples:
- `download_chipseq_data.py`: Download ChIP-seq datasets
- `download_rnaseq_data.py`: Download RNA-seq datasets
- `batch_download.py`: Batch download multiple datasets
- `search_and_download.py`: Search and selective download

## Contributing

Contributions welcome! To add support for a new data source:

1. Create a new module in `src/biopipelines/data_download/`
2. Implement download and search functions
3. Add to `DataDownloader` class
4. Add tests
5. Update documentation

## License

MIT License - see LICENSE file

## Citation

If you use this module, please cite:
```
BioPipelines Data Download Module
https://github.com/yourusername/BioPipelines
```

## Support

- Documentation: https://biopipelines.readthedocs.io
- Issues: https://github.com/yourusername/BioPipelines/issues
- Discussions: https://github.com/yourusername/BioPipelines/discussions
