#!/bin/bash
# Submit all pipelines with containerized execution
# Runs each pipeline that has test data available

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/scratch/sdodl001/BioPipelines/data/raw"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "════════════════════════════════════════════════════"
echo "  BioPipelines - Submit All Pipelines"
echo "════════════════════════════════════════════════════"
echo ""

# Define pipelines and their data directories
declare -A PIPELINES=(
    ["rna-seq"]="rna_seq"
    ["dna-seq"]="dna_seq"
    ["chip-seq"]="chip_seq"
    ["atac-seq"]="atac_seq"
    ["hic"]="hic"
    ["long-read"]="long_read"
    ["scrna-seq"]="scrna_seq"
    ["metagenomics"]="metagenomics"
    ["methylation"]="methylation"
)

declare -a SUBMITTED_JOBS=()

for pipeline in "${!PIPELINES[@]}"; do
    data_subdir="${PIPELINES[$pipeline]}"
    data_path="$DATA_DIR/$data_subdir"
    
    # Check if data exists
    if [[ ! -d "$data_path" ]]; then
        echo -e "${YELLOW}⊘${NC} $pipeline: No data directory"
        continue
    fi
    
    # Check if directory has files
    if [[ -z "$(ls -A "$data_path" 2>/dev/null)" ]]; then
        echo -e "${YELLOW}⊘${NC} $pipeline: Empty data directory"
        continue
    fi
    
    # Check if pipeline directory exists
    if [[ ! -d "$HOME/BioPipelines/pipelines/$data_subdir" ]]; then
        echo -e "${YELLOW}⊘${NC} $pipeline: Pipeline directory not found"
        continue
    fi
    
    # Submit pipeline
    echo -e "${GREEN}→${NC} Submitting $pipeline..."
    if bash "$SCRIPT_DIR/submit_pipeline_with_container.sh" "$pipeline" 2>&1 | grep -q "Job submitted"; then
        echo -e "${GREEN}✓${NC} $pipeline submitted"
        SUBMITTED_JOBS+=("$pipeline")
    else
        echo -e "${RED}✗${NC} $pipeline failed to submit"
    fi
    echo ""
done

echo "════════════════════════════════════════════════════"
echo "Submitted ${#SUBMITTED_JOBS[@]} pipelines:"
for job in "${SUBMITTED_JOBS[@]}"; do
    echo "  • $job"
done
echo "════════════════════════════════════════════════════"
echo ""
echo "Monitor: squeue --me"
echo "Logs: ls -lt ~/BioPipelines/logs/pipeline_runs/ | head"
echo ""
