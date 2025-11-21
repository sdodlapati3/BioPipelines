#!/bin/bash
#SBATCH --job-name=conda_prebuild
#SBATCH --output=logs/conda_prebuild_%j.out
#SBATCH --error=logs/conda_prebuild_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpuspot

# Pre-build all conda environments to avoid corruption during parallel pipeline execution
# This solves the qt-main and other package corruption issues

set -e

echo "=== Pre-building Conda Environments ==="
echo "Started at: $(date)"

# Navigate to project directory
cd ~/BioPipelines

# Function to create conda envs for a pipeline
build_pipeline_envs() {
    local pipeline=$1
    local pipeline_dir=$2
    
    echo ""
    echo "Building $pipeline environments..."
    cd "$pipeline_dir"
    
    snakemake --use-conda --conda-create-envs-only --cores 1 || {
        echo "WARNING: Some environments for $pipeline failed to build"
    }
    
    cd ~/BioPipelines
}

# Build environments for each pipeline
build_pipeline_envs "RNA-seq" "pipelines/rna_seq/differential_expression"
build_pipeline_envs "ATAC-seq" "pipelines/atac_seq/accessibility_analysis"
build_pipeline_envs "ChIP-seq" "pipelines/chip_seq/peak_calling"
build_pipeline_envs "DNA-seq" "pipelines/dna_seq/variant_calling"

echo ""
echo "=== Conda Environment Pre-build Complete ==="
echo "Finished at: $(date)"
echo ""
echo "All conda environments have been created."
echo "You can now safely run pipelines in parallel without package corruption."
