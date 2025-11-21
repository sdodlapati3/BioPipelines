#!/bin/bash
#SBATCH --job-name=bowtie2_index_hg38
#SBATCH --output=bowtie2_index_hg38_%j.out
#SBATCH --error=bowtie2_index_hg38_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpuspot

# Build Bowtie2 index for hg38 genome

# Activate conda environment
source ~/miniconda3/bin/activate ~/envs/biopipelines

# Set variables
GENOME_FA=~/references/genomes/hg38/hg38.fa
INDEX_PREFIX=~/references/genomes/hg38/hg38

# Build Bowtie2 index
bowtie2-build --threads 16 $GENOME_FA $INDEX_PREFIX

echo "Bowtie2 index build complete!"
