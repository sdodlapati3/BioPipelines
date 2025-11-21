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

echo "Hi-C Pipeline - Job ID: $SLURM_JOB_ID"

# Activate conda environment with all tools
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/sdodl001_odu_edu/envs/biopipelines

# Navigate to pipeline directory
cd ~/BioPipelines/pipelines/hic/contact_analysis

# Run Snakemake WITHOUT conda (use base environment)
snakemake \
    --cores $SLURM_CPUS_PER_TASK \
    --rerun-incomplete \
    --keep-going \
    --printshellcmds

echo "Pipeline complete!"
