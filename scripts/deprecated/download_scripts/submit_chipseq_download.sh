#!/bin/bash
#SBATCH --job-name=chipseq_dl
#SBATCH --output=/scratch/sdodl001/chipseq_dl_%j.out
#SBATCH --error=/scratch/sdodl001/chipseq_dl_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00

echo "Starting ChIP-seq download at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/envs/biopipelines

# Verify SRA tools are available
echo "Checking SRA tools..."
which prefetch
which fasterq-dump
which fastq-dump
echo ""

# Run download script
cd ~/BioPipelines
python scripts/download_chipseq_proper.py

DOWNLOAD_EXIT=$?

echo ""
echo "Download finished at $(date)"
echo "Exit code: $DOWNLOAD_EXIT"

exit $DOWNLOAD_EXIT
