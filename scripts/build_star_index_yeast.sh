#!/bin/bash
#SBATCH --job-name=star_index_yeast
#SBATCH --output=star_index_yeast_%j.out
#SBATCH --error=star_index_yeast_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpuspot

# Build STAR index for yeast genome

# Activate conda environment
source ~/miniconda3/bin/activate ~/envs/biopipelines

# Set variables
GENOME_DIR=~/references/indexes/star_yeast
FASTA=~/references/genomes/yeast/sacCer3.fa
GTF=~/references/annotations/yeast/sacCer3.gtf

# Create output directory
mkdir -p $GENOME_DIR

# Build STAR index
# Using smaller genomeSAindexNbases for small genome (yeast is ~12Mb)
STAR --runMode genomeGenerate \
    --genomeDir $GENOME_DIR \
    --genomeFastaFiles $FASTA \
    --sjdbGTFfile $GTF \
    --sjdbOverhang 50 \
    --runThreadN 8 \
    --genomeSAindexNbases 10

echo "STAR index build complete!"
