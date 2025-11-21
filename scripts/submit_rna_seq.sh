#!/bin/bash
#SBATCH --job-name=rna_seq_pipeline
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# BioPipelines RNA-seq Differential Expression Pipeline - Slurm Job Script

echo "========================================="
echo "BioPipelines RNA-seq Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "========================================="

# Activate conda environment
source ~/miniconda3/bin/activate ~/envs/biopipelines

# Setup cleanup trap to unlock on exit/cancel
trap 'echo "Job interrupted, cleaning up locks..."; cd ~/BioPipelines/pipelines/rna_seq/differential_expression && snakemake --unlock 2>/dev/null; exit 130' INT TERM
trap 'if [ $? -ne 0 ]; then echo "Job failed, cleaning up locks..."; cd ~/BioPipelines/pipelines/rna_seq/differential_expression && snakemake --unlock 2>/dev/null; fi' EXIT

# Clean conda cache to prevent corrupted package issues
echo "Cleaning conda cache..."
conda clean --packages -y 2>/dev/null || true

# Navigate to pipeline directory
cd ~/BioPipelines/pipelines/rna_seq/differential_expression

# Run Snakemake with all available CPUs
echo "Starting Snakemake pipeline with $SLURM_CPUS_PER_TASK cores..."
snakemake \
    --cores $SLURM_CPUS_PER_TASK \
    --use-conda \
    --conda-frontend conda \
    --latency-wait 60 \
    --printshellcmds \
    --keep-going \
    --rerun-incomplete

SNAKEMAKE_EXIT=$?

if [ $SNAKEMAKE_EXIT -ne 0 ]; then
    echo "========================================="
    echo "Pipeline FAILED with exit code $SNAKEMAKE_EXIT"
    echo "========================================="
    exit $SNAKEMAKE_EXIT
fi

echo "========================================="
echo "Pipeline complete!"
echo "========================================="
