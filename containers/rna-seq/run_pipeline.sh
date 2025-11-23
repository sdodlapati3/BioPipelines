#!/bin/bash
# Execute RNA-seq pipeline
set -euo pipefail

# Check if Snakefile exists
if [[ -f /opt/biopipelines/rna_seq/Snakefile ]]; then
    SNAKEFILE="/opt/biopipelines/rna_seq/Snakefile"
elif [[ -f /analysis/Snakefile ]]; then
    SNAKEFILE="/analysis/Snakefile"
else
    echo "Error: Snakefile not found"
    echo "Mount your Snakefile to /analysis or include in container"
    exit 1
fi

# Run Snakemake with container-native execution
snakemake \
    -s "$SNAKEFILE" \
    --cores "$THREADS" \
    --directory /analysis \
    --latency-wait 60 \
    --printshellcmds \
    --reason \
    --config \
        input_dir="$INPUT_DIR" \
        output_dir="$OUTPUT_DIR" \
        genome="$GENOME" \
        threads="$THREADS" \
        strandedness="$STRANDEDNESS"
