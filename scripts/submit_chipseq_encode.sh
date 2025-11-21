#!/bin/bash
#SBATCH --job-name=chipseq_encode
#SBATCH --output=/scratch/sdodl001/chipseq_encode_%j.out
#SBATCH --error=/scratch/sdodl001/chipseq_encode_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00

echo "Starting ChIP-seq download from ENCODE at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/envs/biopipelines

# Run download script
cd ~/BioPipelines
python scripts/download_chipseq_encode.py

DOWNLOAD_EXIT=$?

echo ""
echo "Download finished at $(date)"
echo "Exit code: $DOWNLOAD_EXIT"

exit $DOWNLOAD_EXIT
