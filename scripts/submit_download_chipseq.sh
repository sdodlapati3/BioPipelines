#!/bin/bash
#SBATCH --job-name=download_chipseq
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpuspot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Exit on error
set -e

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Change to BioPipelines directory
cd ~/BioPipelines

# Load Python module
module load python3

# Load conda and activate base environment if available
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate base
fi

# Install pysradb if not already installed
echo "Checking for pysradb..."
python3 -c "import pysradb" 2>/dev/null || python3 -m pip install pysradb

# Download ChIP-seq data with correct ENCODE accessions
echo "Downloading ChIP-seq data (ENCODE GM12878 H3K4me3)..."
python3 scripts/download_datasets.py

echo "=========================================="
echo "Download completed successfully"
echo "End Time: $(date)"
echo "=========================================="

# List downloaded files
echo "Downloaded files:"
ls -lh data/raw/chip_seq/

exit 0
