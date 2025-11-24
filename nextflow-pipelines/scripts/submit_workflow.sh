#!/bin/bash
# Wrapper script to submit Nextflow workflows with custom job names
# Usage: ./scripts/submit_workflow.sh <workflow.nf> [nextflow args...]

set -euo pipefail

WORKFLOW_FILE="${1:-}"
if [ -z "$WORKFLOW_FILE" ]; then
    echo "ERROR: No workflow file specified"
    echo "Usage: ./scripts/submit_workflow.sh <workflow.nf> [nextflow args...]"
    exit 1
fi

if [ ! -f "$WORKFLOW_FILE" ]; then
    echo "ERROR: Workflow file not found: $WORKFLOW_FILE"
    exit 1
fi

# Extract workflow name from file path
WORKFLOW_NAME=$(basename "$WORKFLOW_FILE" .nf)

# Generate unique run ID with timestamp and random suffix
RUN_ID="${WORKFLOW_NAME}_$(date +%Y%m%d_%H%M%S)_$$_$RANDOM"

# Create logs directory and unique work directory
mkdir -p logs

echo "Submitting workflow: $WORKFLOW_NAME"
echo "Unique run ID: $RUN_ID"

# Submit with custom job name, unique session name, and isolated work directory
# Using per-run work directory AND launchDir ensures true concurrent execution
LAUNCH_DIR="/scratch/sdodl001/BioPipelines/nf_runs/${RUN_ID}"
mkdir -p "$LAUNCH_DIR"

sbatch --job-name="nf_${WORKFLOW_NAME}" \
       --output="logs/${WORKFLOW_NAME}_%j.out" \
       --error="logs/${WORKFLOW_NAME}_%j.err" \
       scripts/submit_nextflow.sh "$WORKFLOW_FILE" \
       -name "$RUN_ID" \
       -w "/scratch/sdodl001/BioPipelines/work/${RUN_ID}" \
       "${@:2}" \
       "$LAUNCH_DIR"
