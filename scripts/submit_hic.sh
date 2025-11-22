#!/bin/bash
#SBATCH --job-name=hic_pipeline
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Hi-C Contact Analysis Pipeline Submission Script
# =================================================

echo "Starting Hi-C Contact Analysis pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/envs/biopipelines

# Navigate to pipeline directory
cd ~/BioPipelines/pipelines/hic/contact_analysis

# Unlock directory in case of previous failures
echo "Unlocking Snakemake directory..."
snakemake --unlock 2>/dev/null || true

# Trap handlers for automatic cleanup
trap 'echo "Job interrupted or failed..."; snakemake --unlock 2>/dev/null; exit 130' INT TERM
trap 'if [ $? -ne 0 ]; then snakemake --unlock 2>/dev/null; fi' EXIT

# Run Snakemake (using base environment tools)
snakemake \
    --cores $SLURM_CPUS_PER_TASK \
    --rerun-incomplete \
    --keep-going \
    --printshellcmds \
    --reason \
    --stats hic_stats.json

exit_code=$?

echo "End time: $(date)"
echo "Exit code: $exit_code"

if [ $exit_code -eq 0 ]; then
    echo "Pipeline completed successfully!"
else
    echo "Pipeline failed with exit code $exit_code"
fi

exit $exit_code
