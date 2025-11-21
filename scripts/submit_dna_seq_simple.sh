#!/bin/bash
#SBATCH --job-name=dna_seq_pipeline
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# BioPipelines DNA-seq Variant Calling Pipeline

set -e

echo "========================================="
echo "BioPipelines DNA-seq Pipeline"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "========================================="

# Activate conda environment
source ~/miniconda3/bin/activate ~/envs/biopipelines

# Clean conda cache to avoid corruption
conda clean --all -y

# Navigate to pipeline directory
cd ~/BioPipelines/pipelines/dna_seq/variant_calling

# Run Snakemake
snakemake \
    --cores $SLURM_CPUS_PER_TASK \
    --use-conda \
    --conda-frontend conda \
    --latency-wait 60 \
    --printshellcmds \
    --keep-going \
    --rerun-incomplete

echo "========================================="
echo "Pipeline complete!"
echo "End time: $(date)"
echo "========================================="
