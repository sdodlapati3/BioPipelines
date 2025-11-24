#!/bin/bash
#
# Batch Test All Nextflow Pipelines
# ==================================
# Submits all 10 pipelines for validation testing
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOWS_DIR="${SCRIPT_DIR}/../workflows"
RESULTS_BASE="/scratch/sdodl001/BioPipelines/data/results/nextflow_validation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "==================================================================="
echo "Nextflow Pipeline Batch Validation"
echo "==================================================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Workflows: ${WORKFLOWS_DIR}"
echo "Results:   ${RESULTS_BASE}"
echo ""

# Create results directory
mkdir -p "${RESULTS_BASE}"

# Function to submit a pipeline
submit_pipeline() {
    local pipeline_name=$1
    local workflow_file="${WORKFLOWS_DIR}/${pipeline_name}.nf"
    local results_dir="${RESULTS_BASE}/${pipeline_name}"
    
    if [ ! -f "$workflow_file" ]; then
        echo "⚠️  SKIP: ${pipeline_name} (workflow file not found)"
        return 1
    fi
    
    echo "📤 Submitting: ${pipeline_name}"
    mkdir -p "${results_dir}"
    
    # Submit to SLURM
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=nf_${pipeline_name}
#SBATCH --output=${results_dir}/slurm_%j.out
#SBATCH --error=${results_dir}/slurm_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpuspot

# Use containerized Nextflow (has Java 17+)
WORKFLOW_ENGINE_CONTAINER="/scratch/sdodl001/BioPipelines/containers/workflow-engine.sif"

# Set NXF_HOME to writable location
export NXF_HOME=\${SLURM_TMPDIR:-/tmp}/nextflow_\${SLURM_JOB_ID}
mkdir -p \${NXF_HOME}

echo "Starting ${pipeline_name} at \$(date)"
echo "Using containerized Nextflow from: \${WORKFLOW_ENGINE_CONTAINER}"
echo "NXF_HOME: \${NXF_HOME}"
cd ${SCRIPT_DIR}/..

# Use unique work directory for this job to avoid session locks
WORK_DIR="work_${pipeline_name}_\${SLURM_JOB_ID}"
mkdir -p "\${WORK_DIR}"

singularity exec \${WORKFLOW_ENGINE_CONTAINER} nextflow run ${workflow_file} \\
    -work-dir "\${WORK_DIR}" \\
    -process.executor local \\
    -with-report ${results_dir}/report_${TIMESTAMP}.html \\
    -with-timeline ${results_dir}/timeline_${TIMESTAMP}.html \\
    -with-dag ${results_dir}/dag_${TIMESTAMP}.png

exit_code=\$?

if [ \$exit_code -eq 0 ]; then
    echo "✅ ${pipeline_name} COMPLETED at \$(date)"
    echo "${TIMESTAMP}: SUCCESS" >> ${results_dir}/validation.log
else
    echo "❌ ${pipeline_name} FAILED (exit code: \$exit_code) at \$(date)"
    echo "${TIMESTAMP}: FAILED (\$exit_code)" >> ${results_dir}/validation.log
fi

exit \$exit_code
EOF
    
    echo "   → Job submitted for ${pipeline_name}"
}

# Test all pipelines
echo "Submitting pipeline tests..."
echo ""

# Already validated
echo "=== Already Validated ==="
echo "✅ metagenomics (17 min)"
echo "✅ longread (10 min)"
echo ""

# Need testing/fixing
echo "=== Submitting for Testing ==="

submit_pipeline "rnaseq_simple"
submit_pipeline "rnaseq_multi"
submit_pipeline "chipseq"
submit_pipeline "atacseq"
submit_pipeline "dnaseq"
submit_pipeline "hic"
submit_pipeline "scrnaseq"
submit_pipeline "methylation"

echo ""
echo "==================================================================="
echo "Submission Complete"
echo "==================================================================="
echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo ""
echo "Check results:"
echo "  ls -lh ${RESULTS_BASE}/*/slurm_*.{out,err}"
echo ""
echo "View reports (after completion):"
echo "  firefox ${RESULTS_BASE}/*/report_*.html"
echo ""
echo "==================================================================="
