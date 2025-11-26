#!/bin/bash
#SBATCH --job-name=nf_singlecellrnaseq_20251126_074407
#SBATCH --output=/home/sdodl001_odu_edu/BioPipelines/generated_workflows/singlecellrnaseq_20251126_074407/logs/slurm_%j.out
#SBATCH --error=/home/sdodl001_odu_edu/BioPipelines/generated_workflows/singlecellrnaseq_20251126_074407/logs/slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cpuspot

# Load required modules
module load nextflow 2>/dev/null || true
module load singularity 2>/dev/null || true

# Activate conda if available
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate biopipelines 2>/dev/null || true

# Change to workflow directory
cd /home/sdodl001_odu_edu/BioPipelines/generated_workflows/singlecellrnaseq_20251126_074407

# Run Nextflow
nextflow run /home/sdodl001_odu_edu/BioPipelines/generated_workflows/singlecellrnaseq_20251126_074407/main.nf -profile slurm -with-report /home/sdodl001_odu_edu/BioPipelines/generated_workflows/singlecellrnaseq_20251126_074407/logs/report.html -with-timeline /home/sdodl001_odu_edu/BioPipelines/generated_workflows/singlecellrnaseq_20251126_074407/logs/timeline.html -with-dag /home/sdodl001_odu_edu/BioPipelines/generated_workflows/singlecellrnaseq_20251126_074407/logs/dag.png --outdir /home/sdodl001_odu_edu/BioPipelines/generated_workflows/singlecellrnaseq_20251126_074407/results 2>&1 | tee /home/sdodl001_odu_edu/BioPipelines/generated_workflows/singlecellrnaseq_20251126_074407/logs/nextflow_job_20251126_074447.log

echo "Pipeline finished at $(date)"
