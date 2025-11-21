#!/bin/bash
#
# Stage data from GCS to local compute node for fast processing
# This script is called at the beginning of each Slurm job
#

set -e

# Configuration
GCP_PROJECT="rcc-hpc"
GCS_DATA_BUCKET="gs://biopipelines-data"
GCS_REF_BUCKET="gs://biopipelines-references"

# Job-specific directories
JOB_ID=${SLURM_JOB_ID:-$$}
# Detect scratch location (GCP or cluster-specific)
if [ -d "/mnt/disks/scratch" ]; then
    SCRATCH_BASE="/mnt/disks/scratch"
elif [ -d "/scratch/${USER}" ]; then
    SCRATCH_BASE="/scratch/${USER}/BioPipelines"
else
    SCRATCH_BASE="${HOME}/scratch"
fi
SCRATCH_DIR="${SCRATCH_BASE}/${JOB_ID}"

# Parse arguments
PIPELINE_TYPE=${1:-"dna_seq"}  # dna_seq, rna_seq, chip_seq, atac_seq
SAMPLE_NAME=${2:-"sample1"}

echo "========================================="
echo "📦 BioPipelines Data Staging"
echo "========================================="
echo ""
echo "Job ID: ${JOB_ID}"
echo "Pipeline: ${PIPELINE_TYPE}"
echo "Sample: ${SAMPLE_NAME}"
echo "Scratch directory: ${SCRATCH_DIR}"
echo ""

# Create directory structure
echo "📁 Creating working directories..."
mkdir -p "${SCRATCH_DIR}"/{input,working,output,references,logs}

# Check if scratch directory has enough space
AVAILABLE_SPACE=$(df -BG "${SCRATCH_BASE}" | awk 'NR==2 {print $4}' | sed 's/G//')
REQUIRED_SPACE=50

if [ "${AVAILABLE_SPACE}" -lt "${REQUIRED_SPACE}" ]; then
    echo "⚠️  Warning: Low disk space (${AVAILABLE_SPACE}GB available, ${REQUIRED_SPACE}GB recommended)"
fi

echo "✅ Directories created"
echo ""

# ========================================
# Stage Reference Genome
# ========================================
echo "[1/3] Staging reference genome..."

# Check if reference already exists (shared across jobs)
SHARED_REF_DIR="${SCRATCH_BASE}/shared/references"
mkdir -p "${SHARED_REF_DIR}/genome"

if [ ! -f "${SHARED_REF_DIR}/genome/hg38.fa" ]; then
    echo "  Downloading hg38 genome from GCS..."
    gsutil -m rsync -r ${GCS_REF_BUCKET}/genomes/hg38/ ${SHARED_REF_DIR}/genome/
else
    echo "  ✅ Reference genome already staged (using shared copy)"
fi

# Create symlink to shared reference
ln -s "${SHARED_REF_DIR}/genome" "${SCRATCH_DIR}/references/genome"

# Stage known sites
if [ ! -d "${SHARED_REF_DIR}/known_sites" ]; then
    echo "  Downloading known sites from GCS..."
    mkdir -p "${SHARED_REF_DIR}/known_sites"
    gsutil -m rsync -r ${GCS_REF_BUCKET}/known_sites/ ${SHARED_REF_DIR}/known_sites/
else
    echo "  ✅ Known sites already staged (using shared copy)"
fi

ln -s "${SHARED_REF_DIR}/known_sites" "${SCRATCH_DIR}/references/known_sites"

echo "✅ References staged"
echo ""

# ========================================
# Stage Input Data
# ========================================
echo "[2/3] Staging input data for ${PIPELINE_TYPE}..."

case ${PIPELINE_TYPE} in
    dna_seq)
        echo "  Downloading DNA-seq data..."
        gsutil -m rsync -r ${GCS_DATA_BUCKET}/dna_seq/test/ ${SCRATCH_DIR}/input/
        ;;
    rna_seq)
        echo "  Downloading RNA-seq data..."
        gsutil -m rsync -r ${GCS_DATA_BUCKET}/rna_seq/test/ ${SCRATCH_DIR}/input/
        ;;
    chip_seq)
        echo "  Downloading ChIP-seq data..."
        gsutil -m rsync -r ${GCS_DATA_BUCKET}/chip_seq/test/ ${SCRATCH_DIR}/input/
        ;;
    atac_seq)
        echo "  Downloading ATAC-seq data..."
        gsutil -m rsync -r ${GCS_DATA_BUCKET}/atac_seq/test/ ${SCRATCH_DIR}/input/
        ;;
    *)
        echo "❌ Error: Unknown pipeline type: ${PIPELINE_TYPE}"
        exit 1
        ;;
esac

# Verify files were staged
INPUT_SIZE=$(du -sh "${SCRATCH_DIR}/input" | cut -f1)
echo "✅ Input data staged (${INPUT_SIZE})"
echo ""

# ========================================
# Export Environment Variables
# ========================================
echo "[3/3] Setting environment variables..."

export BIOPIPELINES_SCRATCH="${SCRATCH_DIR}"
export BIOPIPELINES_INPUT="${SCRATCH_DIR}/input"
export BIOPIPELINES_WORKING="${SCRATCH_DIR}/working"
export BIOPIPELINES_OUTPUT="${SCRATCH_DIR}/output"
export BIOPIPELINES_REFERENCES="${SCRATCH_DIR}/references"

# Create environment file for pipeline to source
cat > "${SCRATCH_DIR}/env.sh" <<EOF
# BioPipelines environment for job ${JOB_ID}
export BIOPIPELINES_SCRATCH="${SCRATCH_DIR}"
export BIOPIPELINES_INPUT="${SCRATCH_DIR}/input"
export BIOPIPELINES_WORKING="${SCRATCH_DIR}/working"
export BIOPIPELINES_OUTPUT="${SCRATCH_DIR}/output"
export BIOPIPELINES_REFERENCES="${SCRATCH_DIR}/references"
EOF

echo "✅ Environment configured"
echo ""

# ========================================
# Summary
# ========================================
echo "========================================="
echo "✅ Data Staging Complete!"
echo "========================================="
echo ""
echo "📂 Directory Structure:"
echo "  Input:      ${SCRATCH_DIR}/input"
echo "  Working:    ${SCRATCH_DIR}/working"
echo "  Output:     ${SCRATCH_DIR}/output"
echo "  References: ${SCRATCH_DIR}/references"
echo ""
echo "🔧 Environment file: ${SCRATCH_DIR}/env.sh"
echo ""
echo "💡 Usage in pipeline:"
echo "  source ${SCRATCH_DIR}/env.sh"
echo "  cd \${BIOPIPELINES_WORKING}"
echo ""
echo "📤 After pipeline completes, upload results:"
echo "  gsutil -m rsync -r ${SCRATCH_DIR}/output/ gs://biopipelines-results-${GCP_PROJECT}/${PIPELINE_TYPE}/${JOB_ID}/"
echo ""
echo "🧹 Cleanup:"
echo "  rm -rf ${SCRATCH_DIR}"
echo "========================================="

