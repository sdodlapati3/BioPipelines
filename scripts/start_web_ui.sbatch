#!/bin/bash
#SBATCH --job-name=biopipe_ui
#SBATCH --partition=cpuspot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm/web_ui_%j.log
#SBATCH --error=logs/slurm/web_ui_%j.err

# ============================================================================
# BioPipelines Web UI Server
# ============================================================================
#
# Starts the Gradio web interface for workflow generation.
#
# Usage:
#   sbatch scripts/start_web_ui.sbatch
#
# Access:
#   After job starts, create SSH tunnel from your local machine:
#   ssh -L 7860:<node>:7860 <cluster> -N
#   Then open: http://localhost:7860
#
# ============================================================================

set -e

echo "============================================"
echo "BioPipelines Web UI Server"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Create logs directory
mkdir -p logs/slurm

# Setup conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Navigate to project
cd /home/sdodl001_odu_edu/BioPipelines

# Load OpenAI API key if available
if [ -f ".secrets/openai_key" ]; then
    export OPENAI_API_KEY=$(cat .secrets/openai_key)
    echo "✓ OpenAI API key loaded"
fi

# Install/update package
echo "Installing workflow_composer..."
pip install -e . -q

# Configuration
PORT=7860
HOST="0.0.0.0"
UI_TYPE="gradio"  # Options: gradio, flask, api

# Write connection info
INFO_FILE="$HOME/web_ui_info.txt"
cat > "$INFO_FILE" << EOF
BioPipelines Web UI
===================
Job ID: $SLURM_JOB_ID
Node: $(hostname)
Port: $PORT
UI Type: $UI_TYPE
Started: $(date)

To connect:
1. From your LOCAL machine, run:
   ssh -L $PORT:$(hostname):$PORT $(whoami)@<cluster-login-node> -N

2. Open in browser:
   http://localhost:$PORT

To stop:
   scancel $SLURM_JOB_ID
EOF

echo ""
echo "============================================"
echo "Server Info"
echo "============================================"
cat "$INFO_FILE"
echo "============================================"
echo ""

# Start the appropriate UI
case $UI_TYPE in
    gradio)
        echo "Starting Gradio UI on port $PORT..."
        python -m workflow_composer.web.gradio_app --host "$HOST" --port "$PORT"
        ;;
    flask)
        echo "Starting Flask UI on port $PORT..."
        python -m workflow_composer.web.app --host "$HOST" --port "$PORT"
        ;;
    api)
        echo "Starting FastAPI on port $PORT..."
        python -m uvicorn workflow_composer.web.api:app --host "$HOST" --port "$PORT"
        ;;
esac

echo ""
echo "Server stopped at $(date)"
