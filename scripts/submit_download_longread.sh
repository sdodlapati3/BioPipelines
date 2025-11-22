#!/bin/bash
#SBATCH --job-name=download_longread
#SBATCH --output=logs/download_longread_%j.out
#SBATCH --error=logs/download_longread_%j.err
#SBATCH --partition=cpuspot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4:00:00

echo "Downloading long-read test data"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/envs/biopipelines

# Run download script
cd ~/BioPipelines
python scripts/download_long_read_data.py

echo "End time: $(date)"
