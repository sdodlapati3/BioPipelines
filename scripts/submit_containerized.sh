#!/bin/bash
# Submit containerized pipeline to SLURM
# This replaces the old conda-based submission

set -euo pipefail

# Default values
PIPELINE=""
CONTAINER_DIR="/scratch/sdodl001/containers"
VERSION="1.0.0"
PARTITION="cpuspot"
MEM="32G"
CORES=8
TIME="06:00:00"
INPUT_DIR=""
OUTPUT_DIR=""
GENOME=""
DRY_RUN=false

usage() {
    cat << EOF
Submit containerized BioPipelines to SLURM

Usage: $0 --pipeline <name> --input <dir> --output <dir> --genome <ref> [OPTIONS]

Required:
  --pipeline NAME   Pipeline to run (rna-seq, dna-seq, chip-seq, etc.)
  --input DIR       Input directory with data files
  --output DIR      Output directory
  --genome STR      Reference genome (hg38, mm10, etc.)

Options:
  --partition NAME  SLURM partition (default: cpuspot)
  --mem SIZE        Memory allocation (default: 32G)
  --cores NUM       CPU cores (default: 8)
  --time DURATION   Time limit (default: 06:00:00)
  --version VER     Container version (default: 1.0.0)
  --container-dir   Container directory (default: /scratch/sdodl001/containers)
  --dry-run         Show what would be submitted without submitting
  --help            Show this help message

Examples:
  # Submit RNA-seq analysis
  $0 --pipeline rna-seq \\
     --input /data/fastq \\
     --output /data/results \\
     --genome hg38 \\
     --cores 16 --mem 64G

  # Quick DNA-seq variant calling
  $0 --pipeline dna-seq \\
     --input /data/fastq \\
     --output /data/vcf \\
     --genome hg38
EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pipeline) PIPELINE="$2"; shift 2 ;;
        --input) INPUT_DIR="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --genome) GENOME="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --mem) MEM="$2"; shift 2 ;;
        --cores) CORES="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        --container-dir) CONTAINER_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Validate required parameters
if [[ -z "$PIPELINE" ]] || [[ -z "$INPUT_DIR" ]] || [[ -z "$OUTPUT_DIR" ]] || [[ -z "$GENOME" ]]; then
    echo "Error: Missing required parameters"
    usage
fi

# Check container exists
CONTAINER_IMAGE="${CONTAINER_DIR}/${PIPELINE}_${VERSION}.sif"
if [[ ! -f "$CONTAINER_IMAGE" ]]; then
    echo "Error: Container not found: $CONTAINER_IMAGE"
    echo "Run: scripts/containers/build_all.sh --singularity"
    exit 1
fi

# Check input directory
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate job name
JOB_NAME="${PIPELINE}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/slurm/containerized"
mkdir -p "$LOG_DIR"

# Create SLURM batch script
SBATCH_SCRIPT=$(cat << EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CORES
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$LOG_DIR/${JOB_NAME}.out
#SBATCH --error=$LOG_DIR/${JOB_NAME}.err

set -euo pipefail

echo "════════════════════════════════════════"
echo "BioPipelines Containerized Execution"
echo "════════════════════════════════════════"
echo "Pipeline:   $PIPELINE"
echo "Container:  $CONTAINER_IMAGE"
echo "Input:      $INPUT_DIR"
echo "Output:     $OUTPUT_DIR"
echo "Genome:     $GENOME"
echo "Resources:  ${MEM}, ${CORES} cores"
echo "════════════════════════════════════════"

cd \$SLURM_SUBMIT_DIR

# Run containerized pipeline
singularity run \\
    --bind $INPUT_DIR:/data/input:ro \\
    --bind $OUTPUT_DIR:/data/output \\
    --bind \${BIOPIPELINES_GENOME_DIR:-/data/references}:/data/references:ro \\
    $CONTAINER_IMAGE \\
    --input /data/input \\
    --output /data/output \\
    --genome $GENOME \\
    --threads $CORES

EXIT_CODE=\$?

echo "════════════════════════════════════════"
if [[ \$EXIT_CODE -eq 0 ]]; then
    echo "✓ Pipeline completed successfully"
else
    echo "✗ Pipeline failed with exit code \$EXIT_CODE"
fi
echo "════════════════════════════════════════"

exit \$EXIT_CODE
EOF
)

# Show what would be submitted
if [[ "$DRY_RUN" == "true" ]]; then
    echo "=== DRY RUN ==="
    echo "$SBATCH_SCRIPT"
    exit 0
fi

# Submit job
echo "$SBATCH_SCRIPT" | sbatch

echo ""
echo "✓ Job submitted: $JOB_NAME"
echo "  Pipeline:  $PIPELINE"
echo "  Container: $CONTAINER_IMAGE"
echo "  Resources: ${MEM}, ${CORES} cores, ${TIME}"
echo "  Logs:      $LOG_DIR/${JOB_NAME}.{out,err}"
echo ""
echo "Monitor: squeue --me"
echo "Cancel:  scancel <jobid>"
