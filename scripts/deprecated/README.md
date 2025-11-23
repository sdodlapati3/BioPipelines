# Deprecated Scripts

These scripts have been replaced by unified scripts:
- **download_data.py** - Replaces all download_*.py and download_*.sh scripts
- **submit_pipeline.sh** - Replaces all submit_*.sh scripts

## Migration Guide

### Old → New Download Commands
```bash
# Old
./scripts/download_chipseq_encode.py ENCSR000AKP data/raw/chip_seq/
# New
./scripts/download_data.py chipseq --accession ENCSR000AKP --output data/raw/chip_seq/

# Old  
./scripts/download_methylation_test.py
# New
./scripts/download_data.py methylation --test --test-size small --output data/raw/methylation/
```

### Old → New Submit Commands
```bash
# Old
./scripts/submit_chip_seq.sh
# New
./scripts/submit_pipeline.sh --pipeline chip_seq

# Old
./scripts/submit_methylation_simple.sh
# New  
./scripts/submit_pipeline.sh --pipeline methylation --config simple --mem 48G
```

These deprecated scripts will be removed in v0.2.0 (target: January 2026).

