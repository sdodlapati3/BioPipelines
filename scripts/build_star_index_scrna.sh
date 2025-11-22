#!/bin/bash
#SBATCH --job-name=star_index_build
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/logs/star_index_build_%j.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/logs/star_index_build_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --partition=cpuspot

echo "Job started at: $(date)"
echo "Building STAR genome index for scRNA-seq"

cd /scratch/sdodl001/BioPipelines/data/references

# Remove failed partial index
rm -rf star_index_hg38/*

# Build STAR index
~/envs/biopipelines/bin/STAR \
  --runMode genomeGenerate \
  --runThreadN 16 \
  --genomeDir star_index_hg38 \
  --genomeFastaFiles refdata-gex-GRCh38-2024-A/fasta/genome.fa \
  --sjdbGTFfile genes_GRCh38.gtf \
  --sjdbOverhang 100 \
  --limitGenomeGenerateRAM 48000000000

echo "Job completed at: $(date)"
echo "STAR index size:"
du -sh star_index_hg38/
