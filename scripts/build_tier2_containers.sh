#!/bin/bash
#
# Tier 2 Container Build Orchestrator
# =====================================
# Builds domain-specific container modules using SLURM compute nodes
# with fakeroot (no sudo required)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINERS_DIR="${SCRIPT_DIR}/../containers/tier2"
OUTPUT_DIR="/scratch/sdodl001/BioPipelines/containers/tier2"
LOGS_DIR="${SCRIPT_DIR}/../logs/tier2_builds"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOGS_DIR}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=================================================================="
echo "Tier 2 Container Build Orchestrator"
echo "=================================================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Definitions: ${CONTAINERS_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Logs: ${LOGS_DIR}"
echo ""

# Function to submit build job
submit_build() {
    local MODULE=$1
    local PRIORITY=$2
    local DEF_FILE="${CONTAINERS_DIR}/${MODULE}.def"
    local OUTPUT_SIF="${OUTPUT_DIR}/${MODULE}.sif"
    local BUILD_LOG="${LOGS_DIR}/build_${MODULE}_${TIMESTAMP}.log"
    local BUILD_ERR="${LOGS_DIR}/build_${MODULE}_${TIMESTAMP}.err"
    
    if [ ! -f "${DEF_FILE}" ]; then
        echo -e "${RED}❌ Definition file not found: ${DEF_FILE}${NC}"
        return 1
    fi
    
    # Create SLURM job script
    cat > "${LOGS_DIR}/job_${MODULE}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=build_${MODULE}
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpuspot
#SBATCH --output=${BUILD_LOG}
#SBATCH --error=${BUILD_ERR}

set -e

echo "=========================================="
echo "Building: ${MODULE}"
echo "Time: \$(date)"
echo "Node: \$(hostname)"
echo "=========================================="
echo ""

# Load Singularity module if available
module load singularity 2>/dev/null || true

# Build container using fakeroot
echo "Starting build with fakeroot..."
singularity build --fakeroot \\
    ${OUTPUT_SIF} \\
    ${DEF_FILE}

if [ \$? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Build completed successfully"
    echo "=========================================="
    
    # Get container info
    echo ""
    echo "Container information:"
    singularity inspect ${OUTPUT_SIF}
    
    # Run validation tests
    echo ""
    echo "Running validation tests..."
    singularity test ${OUTPUT_SIF}
    
    if [ \$? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✅ Validation passed"
        echo "=========================================="
        
        # Set permissions
        chmod 755 ${OUTPUT_SIF}
        
        # Get final size
        SIZE=\$(du -h ${OUTPUT_SIF} | cut -f1)
        echo ""
        echo "Container size: \${SIZE}"
        echo "Location: ${OUTPUT_SIF}"
        
        exit 0
    else
        echo ""
        echo "=========================================="
        echo "❌ Validation failed"
        echo "=========================================="
        exit 1
    fi
else
    echo ""
    echo "=========================================="
    echo "❌ Build failed"
    echo "=========================================="
    exit 1
fi
EOF
    
    # Submit job
    echo -ne "${BLUE}📦 ${MODULE}${NC} (Priority ${PRIORITY}): "
    JOB_ID=$(sbatch --parsable "${LOGS_DIR}/job_${MODULE}.sh")
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Submitted (Job ${JOB_ID})${NC}"
        echo "${JOB_ID}" >> "${LOGS_DIR}/active_jobs_${TIMESTAMP}.txt"
        return 0
    else
        echo -e "${RED}Failed to submit${NC}"
        return 1
    fi
}

# Module definitions with build priority
# Priority 1: Critical modules needed by most pipelines
# Priority 2: Specialized modules
# Priority 3: Lower priority modules

declare -A MODULES=(
    # Priority 1: Core modules
    ["alignment_short_read"]=1
    ["quantification"]=1
    ["peak_calling"]=1
    
    # Priority 2: Specialized modules
    ["variant_calling"]=2
    ["metagenomics"]=2
    ["longread_tools"]=2
    
    # Priority 3: Advanced modules
    ["scrna"]=3
    ["methylation"]=3
    ["structural_variants"]=3
    ["assembly"]=3
)

# Check which modules to build
if [ $# -eq 0 ]; then
    echo "Usage: $0 [module_name] [module_name] ... | all | priority1 | priority2 | priority3"
    echo ""
    echo "Available modules:"
    for module in "${!MODULES[@]}"; do
        priority="${MODULES[$module]}"
        def_file="${CONTAINERS_DIR}/${module}.def"
        if [ -f "${def_file}" ]; then
            status="${GREEN}✓${NC}"
        else
            status="${YELLOW}○${NC}"
        fi
        echo -e "  ${status} ${module} (Priority ${priority})"
    done
    echo ""
    echo "Examples:"
    echo "  $0 alignment_short_read              # Build single module"
    echo "  $0 all                                # Build all modules"
    echo "  $0 priority1                          # Build priority 1 modules"
    echo "  $0 alignment_short_read quantification # Build specific modules"
    exit 0
fi

# Determine which modules to build
BUILD_LIST=()

if [ "$1" == "all" ]; then
    BUILD_LIST=("${!MODULES[@]}")
elif [ "$1" == "priority1" ]; then
    for module in "${!MODULES[@]}"; do
        if [ "${MODULES[$module]}" -eq 1 ]; then
            BUILD_LIST+=("$module")
        fi
    done
elif [ "$1" == "priority2" ]; then
    for module in "${!MODULES[@]}"; do
        if [ "${MODULES[$module]}" -eq 2 ]; then
            BUILD_LIST+=("$module")
        fi
    done
elif [ "$1" == "priority3" ]; then
    for module in "${!MODULES[@]}"; do
        if [ "${MODULES[$module]}" -eq 3 ]; then
            BUILD_LIST+=("$module")
        fi
    done
else
    # Build specified modules
    BUILD_LIST=("$@")
fi

echo "=================================================================="
echo "Build Plan"
echo "=================================================================="
echo "Modules to build: ${#BUILD_LIST[@]}"
echo ""

# Submit builds
SUCCESS_COUNT=0
FAIL_COUNT=0

for module in "${BUILD_LIST[@]}"; do
    priority="${MODULES[$module]:-0}"
    if [ $priority -eq 0 ]; then
        echo -e "${YELLOW}⚠️  Unknown module: ${module}${NC}"
        ((FAIL_COUNT++))
        continue
    fi
    
    if submit_build "$module" "$priority"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi
done

echo ""
echo "=================================================================="
echo "Submission Summary"
echo "=================================================================="
echo -e "${GREEN}✅ Submitted: ${SUCCESS_COUNT}${NC}"
echo -e "${RED}❌ Failed: ${FAIL_COUNT}${NC}"
echo ""

if [ ${SUCCESS_COUNT} -gt 0 ]; then
    echo "Monitor jobs:"
    echo "  squeue -u \$USER"
    echo ""
    echo "Check logs:"
    echo "  tail -f ${LOGS_DIR}/build_*_${TIMESTAMP}.log"
    echo ""
    echo "View active job IDs:"
    echo "  cat ${LOGS_DIR}/active_jobs_${TIMESTAMP}.txt"
    echo ""
    echo "Check completion status:"
    echo "  sacct -j \$(cat ${LOGS_DIR}/active_jobs_${TIMESTAMP}.txt | tr '\n' ',')"
fi

echo "=================================================================="
