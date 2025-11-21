#!/bin/bash
#SBATCH --job-name=setup_kraken2_db
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00

echo "Setting up Kraken2 database for Metagenomics pipeline"
echo "Job ID: $SLURM_JOB_ID"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/sdodl001_odu_edu/envs/biopipelines

# Download pre-built Kraken2 standard-8 database
DB_DIR=/scratch/sdodl001/BioPipelines/data/references/kraken2_db
mkdir -p $DB_DIR
cd $DB_DIR

echo "Downloading Kraken2 Standard-8 database (~8GB)..."
wget -c https://genome-idx.s3.amazonaws.com/kraken/k2_standard_08gb_20240904.tar.gz

echo "Extracting database..."
tar -xzf k2_standard_08gb_20240904.tar.gz

echo "Database ready at: $DB_DIR"
ls -lh $DB_DIR/

echo "Setup complete: $(date)"
