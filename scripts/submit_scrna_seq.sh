#!/bin/bash
#SBATCH --job-name=scrna_seq
#SBATCH --output=logs/scrna_seq_%j.out
#SBATCH --error=logs/scrna_seq_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=default

echo "=========================================="
echo "Single-cell RNA-seq Analysis Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on: $(hostname)"
echo ""

# Activate conda environment
source /home/sdodl001_odu_edu/miniconda3/etc/profile.d/conda.sh
conda activate /home/sdodl001_odu_edu/envs/biopipelines

# Change to project directory
cd ~/BioPipelines

# Create log directory
mkdir -p logs/scrna_seq

# Run Snakemake pipeline
echo "Running Snakemake pipeline..."
echo ""

snakemake \
    --snakefile pipelines/scrna_seq/Snakefile \
    --configfile pipelines/scrna_seq/config.yaml \
    --cores $SLURM_CPUS_PER_TASK \
    --rerun-incomplete \
    --printshellcmds \
    --reason \
    --keep-going \
    --latency-wait 60

PIPELINE_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Pipeline completed: $(date)"
echo "Exit code: $PIPELINE_EXIT_CODE"
echo "=========================================="

if [ $PIPELINE_EXIT_CODE -eq 0 ]; then
    echo "✓ scRNA-seq pipeline completed successfully!"
    
    echo ""
    echo "Results location:"
    echo "  - Filtered data: data/results/scrna_seq/filtered/"
    echo "  - Clustering: data/results/scrna_seq/clustering/"
    echo "  - Cell types: data/results/scrna_seq/annotation/"
    echo "  - DEGs: data/results/scrna_seq/differential_expression/"
    echo "  - Plots: data/results/scrna_seq/plots/"
    echo "  - Report: data/results/scrna_seq/scrna_seq_report.html"
    echo "  - MultiQC: data/results/scrna_seq/multiqc_report.html"
    
else
    echo "✗ Pipeline failed with exit code $PIPELINE_EXIT_CODE"
    echo "Check logs at: logs/scrna_seq_${SLURM_JOB_ID}.err"
fi

exit $PIPELINE_EXIT_CODE
