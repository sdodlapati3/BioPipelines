#!/bin/bash
#SBATCH --job-name=chip_seq_pipeline
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

echo "ChIP-seq Pipeline - Job ID: $SLURM_JOB_ID"

# Activate conda environment with all tools
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/sdodl001_odu_edu/envs/biopipelines

# Navigate to pipeline directory
cd ~/BioPipelines/pipelines/chip_seq/peak_calling

# Run Snakemake WITHOUT conda (use base environment)
snakemake \
    --cores $SLURM_CPUS_PER_TASK \
    --latency-wait 60 \
    --printshellcmds \
    --keep-going \
    --rerun-incomplete

echo "Pipeline complete!"
