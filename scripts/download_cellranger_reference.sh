#!/bin/bash
# Download CellRanger Reference Genome for Human
# Required for scRNA-seq pipeline

set -e

REFERENCE_DIR="/scratch/sdodl001/BioPipelines/data/references"
mkdir -p "$REFERENCE_DIR"
cd "$REFERENCE_DIR"

echo "================================================"
echo "Downloading CellRanger GRCh38 Reference (2024-A)"
echo "================================================"
echo "Size: ~12 GB compressed, ~25 GB uncompressed"
echo "This may take 10-20 minutes..."
echo ""

# CellRanger reference from 10x Genomics
REFERENCE_URL="https://cf.10xgenomics.com/supp/cell-exp/refdata-gex-GRCh38-2024-A.tar.gz"
REFERENCE_NAME="refdata-gex-GRCh38-2024-A"

if [ -d "$REFERENCE_NAME" ]; then
    echo "✓ Reference already exists: $REFERENCE_NAME"
    echo "  Size: $(du -sh $REFERENCE_NAME | cut -f1)"
    exit 0
fi

echo "Downloading from: $REFERENCE_URL"
wget --continue --progress=bar:force "$REFERENCE_URL" -O "${REFERENCE_NAME}.tar.gz"

echo ""
echo "Extracting reference genome..."
tar -xzf "${REFERENCE_NAME}.tar.gz"

echo ""
echo "Cleaning up..."
rm "${REFERENCE_NAME}.tar.gz"

echo ""
echo "================================================"
echo "✓ CellRanger reference downloaded successfully!"
echo "================================================"
echo "Location: $REFERENCE_DIR/$REFERENCE_NAME"
echo "Size: $(du -sh $REFERENCE_NAME | cut -f1)"
echo ""
echo "Next: Update config.yaml with reference path and run pipeline"
echo "  sbatch scripts/submit_scrna_seq.sh"
