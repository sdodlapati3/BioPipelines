#!/bin/bash
#SBATCH --job-name=long_read_sv
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Long-Read Sequencing SV Analysis Pipeline
# ==========================================

# Load conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate BioPipelines environment
conda activate ~/envs/biopipelines

# Set working directory
cd ~/BioPipelines/pipelines/long_read/sv_analysis

# Unlock directory in case of previous failures
echo "Unlocking Snakemake directory..."
snakemake --unlock 2>/dev/null || true

# Print environment info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Conda environment: $CONDA_PREFIX"

# Run Snakemake pipeline (using base environment tools)
snakemake \
    --snakefile Snakefile \
    --configfile config.yaml \
    --cores ${SLURM_CPUS_PER_TASK} \
    --keep-going \
    --rerun-incomplete \
    --printshellcmds \
    2>&1 | tee pipeline_${SLURM_JOB_ID}.log

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "Pipeline completed successfully at: $(date)"
    echo "Results are in: /scratch/sdodl001/BioPipelines/data/results/long_read"
else
    echo "Pipeline failed with exit code: $EXIT_CODE at: $(date)"
    exit $EXIT_CODE
fi
