# CellRanger Installation Instructions

## Why Manual Installation is Required

CellRanger from 10x Genomics requires:
1. Acceptance of End User License Agreement (EULA)
2. Registration on 10x Genomics website
3. Time-limited signed download URLs (expire after ~24 hours)

This cannot be automated in container builds.

## Installation Steps

### Option 1: Download via 10x Genomics Website

1. **Register** at https://www.10xgenomics.com/support/software/cell-ranger/downloads
2. **Accept** the EULA
3. **Download** CellRanger 9.0.0:
   ```bash
   # Get the signed download URL from the website, then:
   cd /tmp
   wget -O cellranger-9.0.0.tar.gz "<YOUR_SIGNED_URL>"
   ```

4. **Install** into the container:
   ```bash
   # Extract to /opt
   sudo tar -xzf cellranger-9.0.0.tar.gz -C /opt/
   
   # Or install directly on the system
   tar -xzf cellranger-9.0.0.tar.gz
   export PATH=/path/to/cellranger-9.0.0:$PATH
   ```

### Option 2: Use STARsolo (Alternative)

STARsolo can process 10x scRNA-seq data without CellRanger:

```bash
# Already included in scrna-seq container
STAR --soloType CB_UMI_Simple \
     --soloCBwhitelist 737K-august-2016.txt \
     --soloUMIlen 12 \
     --soloFeatures Gene GeneFull \
     ...
```

### Option 3: Rebuild Container with CellRanger

If you have the tar.gz file:

```bash
# Place cellranger-9.0.0.tar.gz in containers/scrna-seq/
cd ~/BioPipelines/containers/scrna-seq/

# Edit scrna-seq.def to copy and extract the file
# Then rebuild:
sbatch ~/BioPipelines/scripts/containers/build_scrna_seq_container.slurm
```

## Current Status

- **Container**: Built with STAR, Salmon, Scanpy, Seurat
- **Missing**: CellRanger (manual installation required)
- **Alternative**: Use STARsolo for 10x data processing

## Next Steps

1. Decide if CellRanger is required or if STARsolo is acceptable
2. If CellRanger needed, download manually from 10x Genomics
3. Either install system-wide or rebuild container with CellRanger included
