#!/bin/bash
# Submit Snakemake pipeline using containerized tools
# Usage: ./submit_pipeline_with_container.sh <pipeline_name>

set -euo pipefail

PIPELINE=$1
CONTAINER_DIR="/home/sdodl001_odu_edu/BioPipelines/containers/images"
# Convert pipeline name: hyphens in container names, underscores in directory names
PIPELINE_UNDERSCORE="${PIPELINE//-/_}"
PIPELINE_DIR="$HOME/BioPipelines/pipelines/${PIPELINE_UNDERSCORE}"
LOG_DIR="$HOME/BioPipelines/logs/pipeline_runs"
PARTITION="cpuspot"
WORKFLOW_ENGINE="${CONTAINER_DIR}/workflow-engine_1.0.0.sif"
PIPELINE_CONTAINER="${CONTAINER_DIR}/${PIPELINE}_1.0.0.sif"

# Validate pipeline exists
if [[ ! -d "$PIPELINE_DIR" ]]; then
    echo "Error: Pipeline directory not found: $PIPELINE_DIR"
    exit 1
fi

# Check workflow engine exists
if [[ ! -f "$WORKFLOW_ENGINE" ]]; then
    echo "Error: Workflow engine not found: $WORKFLOW_ENGINE"
    echo "Build it with: sbatch scripts/containers/build_workflow_engine.slurm"
    exit 1
fi

# Check pipeline container exists
CONTAINER="${CONTAINER_DIR}/${PIPELINE}_1.0.0.sif"
if [[ ! -f "$CONTAINER" ]]; then
    echo "Error: Container not found: $CONTAINER"
    exit 1
fi

# Check Snakefile exists
if [[ ! -f "$PIPELINE_DIR/Snakefile" ]]; then
    echo "Error: Snakefile not found in $PIPELINE_DIR"
    exit 1
fi

mkdir -p "$LOG_DIR"
JOB_NAME="pipeline_${PIPELINE}_$(date +%Y%m%d_%H%M%S)"

echo "════════════════════════════════════════"
echo "Submitting $PIPELINE pipeline"
echo "Container: $CONTAINER"
echo "Pipeline: $PIPELINE_DIR"
echo "════════════════════════════════════════"

# Create SLURM script
cat <<EOF | sbatch
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=$LOG_DIR/${JOB_NAME}.out
#SBATCH --error=$LOG_DIR/${JOB_NAME}.err

set -euo pipefail

# Ensure singularity is in PATH
export PATH="/cm/shared/applications/slurm/wrapper:\$PATH"

# Activate micromamba environment with snakemake
export MAMBA_EXE="$HOME/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "\$(\$MAMBA_EXE shell hook --shell bash --root-prefix \$MAMBA_ROOT_PREFIX)"
micromamba activate

echo "════════════════════════════════════════"
echo "Pipeline: $PIPELINE"
echo "Started: \$(date)"
echo "════════════════════════════════════════"

cd $PIPELINE_DIR

# Create a config override to specify the container for this pipeline
cat > .snakemake_container_config.yaml <<CONTAINER_CONFIG
default-resources:
  singularity: "$PIPELINE_CONTAINER"
CONTAINER_CONFIG

# Run Snakemake using the containerized profile
snakemake \\
    --profile $HOME/BioPipelines/config/snakemake_profiles/containerized \\
    --configfile .snakemake_container_config.yaml

EXIT_CODE=\$?

echo ""
echo "════════════════════════════════════════"
echo "Pipeline: $PIPELINE"
echo "Finished: \$(date)"
if [[ \$EXIT_CODE -eq 0 ]]; then
    echo "Status: ✓ SUCCESS"
else
    echo "Status: ✗ FAILED (exit code \$EXIT_CODE)"
fi
echo "════════════════════════════════════════"

exit \$EXIT_CODE
EOF

echo ""
echo "✓ Job submitted: $JOB_NAME"
echo "  Log: $LOG_DIR/${JOB_NAME}.out"
echo ""
