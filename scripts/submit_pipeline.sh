#!/bin/bash
#
# Unified pipeline submission script for BioPipelines
# Replaces 18 separate submit scripts with one configurable script
#

set -e

# Default values
PIPELINE=""
CONFIG_TYPE="full"
PARTITION="cpuspot"
MEM="32G"
CORES=8
TIME="06:00:00"
DRY_RUN=false
RERUN=false

# Usage function
usage() {
    cat << EOF
Usage: $0 --pipeline <name> [OPTIONS]

Required:
  --pipeline NAME       Pipeline to run (atac_seq, chip_seq, dna_seq, rna_seq,
                       scrna_seq, methylation, hic, long_read, metagenomics, sv)

Options:
  --config TYPE         Configuration type: simple|full (default: full)
  --partition NAME      SLURM partition (default: cpuspot)
  --mem SIZE           Memory allocation (default: 32G)
  --cores NUM          CPU cores (default: 8)
  --time DURATION      Time limit (default: 06:00:00)
  --rerun              Rerun incomplete jobs
  --dry-run            Show what would be submitted without submitting
  --help               Show this help message

Examples:
  # Submit ChIP-seq with defaults
  $0 --pipeline chip_seq

  # Submit with custom resources
  $0 --pipeline methylation --mem 48G --cores 16 --time 08:00:00
EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pipeline)
            PIPELINE="$2"
            shift 2
            ;;
        --config)
            CONFIG_TYPE="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --mem)
            MEM="$2"
            shift 2
            ;;
        --cores)
            CORES="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --rerun)
            RERUN=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate pipeline
if [ -z "$PIPELINE" ]; then
    echo "Error: --pipeline is required"
    usage
fi

# Set pipeline-specific defaults
case $PIPELINE in
    scrna_seq)
        [ "$MEM" = "32G" ] && MEM="64G"
        [ "$CORES" = "8" ] && CORES=16
        [ "$TIME" = "06:00:00" ] && TIME="08:00:00"
        ;;
    metagenomics)
        [ "$MEM" = "32G" ] && MEM="128G"
        [ "$CORES" = "8" ] && CORES=32
        [ "$TIME" = "06:00:00" ] && TIME="12:00:00"
        ;;
    dna_seq|methylation)
        [ "$MEM" = "32G" ] && MEM="48G"
        [ "$CORES" = "8" ] && CORES=16
        [ "$TIME" = "06:00:00" ] && TIME="08:00:00"
        ;;
    hic)
        [ "$MEM" = "32G" ] && MEM="64G"
        [ "$CORES" = "8" ] && CORES=16
        ;;
esac

# Determine pipeline directory
PIPELINE_DIR=""
case $PIPELINE in
    atac_seq|chip_seq|dna_seq|rna_seq|scrna_seq|methylation|hic|long_read|metagenomics)
        PIPELINE_DIR="pipelines/$PIPELINE"
        ;;
    sv)
        PIPELINE_DIR="pipelines/structural_variants"
        ;;
    *)
        echo "Error: Unknown pipeline: $PIPELINE"
        exit 1
        ;;
esac

# Check if pipeline directory exists
if [ ! -d "$PIPELINE_DIR" ]; then
    echo "Error: Pipeline directory not found: $PIPELINE_DIR"
    exit 1
fi

# Check if Snakefile exists
if [ ! -f "$PIPELINE_DIR/Snakefile" ]; then
    echo "Error: Snakefile not found in $PIPELINE_DIR"
    exit 1
fi

# Build snakemake command with scratch conda prefix (avoids NFS locks)
SNAKEMAKE_CMD="snakemake --cores $CORES --use-conda --conda-frontend mamba --conda-prefix /scratch/sdodl001/conda_envs --latency-wait 60"

if [ "$RERUN" = true ]; then
    SNAKEMAKE_CMD="$SNAKEMAKE_CMD --rerun-incomplete"
fi

# Generate job name
JOB_NAME="${PIPELINE}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/slurm/active"
mkdir -p "$LOG_DIR"

# Create sbatch script
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

set -e

echo "================================"
echo "BioPipelines Job Submission"
echo "================================"
echo "Pipeline:    $PIPELINE"
echo "Directory:   $PIPELINE_DIR"
echo "Memory:      $MEM"
echo "Cores:       $CORES"
echo "Conda:       /scratch/sdodl001/conda_envs (scratch - no NFS locks)"
echo "================================"

cd \$SLURM_SUBMIT_DIR

# Initialize conda
eval "\$(/home/sdodl001_odu_edu/miniconda3/bin/conda shell.bash hook)"
conda activate /home/sdodl001_odu_edu/envs/biopipelines

# Clean scratch conda directory completely (removes any corrupted envs)
rm -rf /scratch/sdodl001/conda_envs
mkdir -p /scratch/sdodl001/conda_envs

# Clean and unlock
cd $PIPELINE_DIR
rm -rf .snakemake/conda 2>/dev/null || true
snakemake --unlock 2>/dev/null || true

# Run pipeline
echo "Starting pipeline execution..."
$SNAKEMAKE_CMD

echo "================================"
echo "Pipeline completed!"
echo "================================"
EOF
)

# Show what would be submitted
if [ "$DRY_RUN" = true ]; then
    echo "=== DRY RUN ==="
    echo "$SBATCH_SCRIPT"
    exit 0
fi

# Submit job
echo "$SBATCH_SCRIPT" | sbatch

echo ""
echo "✓ Job submitted: $JOB_NAME"
echo "  Pipeline:  $PIPELINE"
echo "  Resources: ${MEM}, ${CORES} cores, ${TIME}"
echo "  Logs:      $LOG_DIR/${JOB_NAME}.{out,err}"
echo "  Conda:     /scratch/sdodl001/conda_envs (scratch storage)"
