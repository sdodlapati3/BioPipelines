#!/bin/bash
#SBATCH --job-name=sv_calling
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

echo "Starting Structural Variants Detection Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/envs/biopipelines

# Navigate to pipeline directory
cd ~/BioPipelines/pipelines/structural_variants/sv_calling

# Unlock directory
echo "Unlocking Snakemake directory..."
snakemake --unlock 2>/dev/null || true

# Run Snakemake
snakemake \
    --use-conda \
    --conda-frontend conda \
    --conda-prefix ~/.snakemake/conda \
    --cores $SLURM_CPUS_PER_TASK \
    --rerun-incomplete \
    --keep-going \
    --printshellcmds \
    --reason

exit_code=$?
echo "End time: $(date)"
echo "Exit code: $exit_code"
exit $exit_code
