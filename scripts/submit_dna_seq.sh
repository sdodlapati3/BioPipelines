#!/bin/bash
#SBATCH --job-name=dna_seq_pipeline
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# BioPipelines DNA-seq Variant Calling Pipeline - Slurm Job Script
# Integrates with GCS for input/output storage

set -e  # Exit on error

echo "========================================="
echo "BioPipelines DNA-seq Pipeline"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "========================================="
echo ""

# Configuration
GCP_PROJECT="rcc-hpc"
GCS_RESULTS_BUCKET="gs://biopipelines-results-${GCP_PROJECT}"
PIPELINE_TYPE="dna_seq"
SAMPLE_NAME="sample1"

# Activate conda environment
echo "🔧 Activating conda environment..."
source ~/miniconda3/bin/activate ~/envs/biopipelines
echo "✅ Environment activated: biopipelines"
echo ""

# Stage data from GCS to local scratch
echo "📦 Staging data from GCS..."
bash ~/BioPipelines/scripts/gcp_stage_data.sh ${PIPELINE_TYPE} ${SAMPLE_NAME}

# Source environment variables
source /mnt/disks/scratch/${SLURM_JOB_ID}/env.sh
echo ""

# Navigate to pipeline directory
cd ~/BioPipelines/pipelines/dna_seq/variant_calling

# Update config to use staged data
echo "🔧 Updating pipeline config for this job..."
cat > config.runtime.yaml <<EOF
samples:
  - ${SAMPLE_NAME}

reference:
  genome: "${BIOPIPELINES_REFERENCES}/genome/hg38.fa"
  known_sites: "${BIOPIPELINES_REFERENCES}/known_sites/dbsnp_155.hg38.vcf.gz"

snpeff_genome: "hg38"

# Override paths for this job
input_dir: "${BIOPIPELINES_INPUT}"
output_dir: "${BIOPIPELINES_OUTPUT}"

align:
  threads: ${SLURM_CPUS_PER_TASK}

variant_calling:
  threads: $((SLURM_CPUS_PER_TASK / 2))
EOF

# Run Snakemake with all available CPUs
echo ""
echo "========================================="
echo "🚀 Starting Snakemake pipeline..."
echo "========================================="
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "Working directory: ${BIOPIPELINES_WORKING}"
echo "Output directory: ${BIOPIPELINES_OUTPUT}"
echo ""

snakemake \
    --configfile config.runtime.yaml \
    --cores $SLURM_CPUS_PER_TASK \
    --use-conda \
    --conda-frontend conda \
    --latency-wait 60 \
    --printshellcmds \
    --keep-going \
    --rerun-incomplete \
    --directory "${BIOPIPELINES_WORKING}"

echo ""
echo "========================================="
echo "✅ Pipeline execution complete!"
echo "========================================="
echo ""

# Upload results to GCS
echo "📤 Uploading results to GCS..."
GCS_OUTPUT="${GCS_RESULTS_BUCKET}/dna_seq/${SLURM_JOB_ID}"
gsutil -m rsync -r "${BIOPIPELINES_OUTPUT}/" "${GCS_OUTPUT}/"

echo "✅ Results uploaded to: ${GCS_OUTPUT}"
echo ""

# Generate summary
TOTAL_TIME=$SECONDS
echo "========================================="
echo "📊 Job Summary"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Pipeline: DNA-seq Variant Calling"
echo "Sample: ${SAMPLE_NAME}"
echo "Total time: $((TOTAL_TIME / 60)) minutes"
echo "Results: ${GCS_OUTPUT}"
echo ""
echo "View results:"
echo "  gsutil ls -lh ${GCS_OUTPUT}/"
echo ""
echo "Download results:"
echo "  gsutil -m rsync -r ${GCS_OUTPUT}/ ./results/"
echo ""

# Cleanup local scratch
echo "🧹 Cleaning up local scratch..."
rm -rf "${BIOPIPELINES_SCRATCH}"
echo "✅ Cleanup complete"
echo ""

echo "========================================="
echo "🎉 Job complete!"
echo "End time: $(date)"
echo "========================================="
