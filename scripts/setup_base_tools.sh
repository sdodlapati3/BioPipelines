#!/bin/bash
# Install all required bioinformatics tools in base environment
# This avoids conda dependency conflicts in per-rule environments

set -e

echo "Setting up base biopipelines environment with all tools..."

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate biopipelines

# Install all tools at once to let conda resolve dependencies properly
conda install -y -c bioconda -c conda-forge \
    fastqc=0.12.1 \
    multiqc=1.14 \
    fastp=0.23.4 \
    trimmomatic=0.39 \
    trim-galore=0.6.10 \
    cutadapt=4.4 \
    bwa=0.7.17 \
    bowtie2=2.5.0 \
    samtools=1.17 \
    picard=3.0.0 \
    bismark=0.24.2 \
    macs2=2.2.7.1 \
    homer=4.11 \
    deeptools=3.5.4 \
    bedtools=2.31.0 \
    ucsc-bedgraphtobigwig=377 \
    hicexplorer=3.7.2 \
    pairix=0.3.7 \
    cooler=0.9.3 \
    hic2cool=0.8.3

echo "Base environment setup complete!"
conda list | grep -E "fastqc|multiqc|bwa|bowtie2|samtools|bismark|macs2"
