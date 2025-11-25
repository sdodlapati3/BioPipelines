#!/bin/bash
#SBATCH --job-name=vllm-setup
#SBATCH --partition=h100flex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/vllm_setup_%j.log
#SBATCH --error=logs/slurm/vllm_setup_%j.err
# ============================================================================
# BioPipelines vLLM Setup and Server Job
# ============================================================================
# This script:
# 1. Creates/activates a vLLM conda environment
# 2. Installs vLLM if not present
# 3. Starts the vLLM server with the specified model
# ============================================================================

set -e

# Configuration
MODEL="${1:-mistralai/Mistral-7B-Instruct-v0.3}"
PORT=8000
GPU_MEMORY_UTIL=0.9

# Print job info
echo "=============================================="
echo "BioPipelines vLLM Setup"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Time: $(date)"
echo "=============================================="

# Load CUDA
module purge
module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || {
    echo "Trying to find CUDA..."
    module avail cuda 2>&1 | head -10
}

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi || { echo "ERROR: No GPU available"; exit 1; }

# Setup conda
source ~/miniconda3/bin/activate

# Create vLLM environment if needed
ENV_NAME="vllm_env"
if ! conda env list | grep -q "$ENV_NAME"; then
    echo ""
    echo "Creating vLLM conda environment..."
    conda create -n $ENV_NAME python=3.11 -y
fi

# Activate environment
conda activate $ENV_NAME

# Install vLLM if not present
if ! python -c "import vllm" 2>/dev/null; then
    echo ""
    echo "Installing vLLM..."
    pip install vllm --upgrade
    pip install huggingface_hub --upgrade
fi

# Show installed packages
echo ""
echo "vLLM version:"
python -c "import vllm; print(vllm.__version__)"

# Set HuggingFace cache
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

# Login to HuggingFace if token available
if [[ -f ~/BioPipelines/.secrets/hf_token ]]; then
    export HF_TOKEN=$(cat ~/BioPipelines/.secrets/hf_token)
    echo "HuggingFace token loaded"
fi

# Print connection instructions
echo ""
echo "=============================================="
echo "CONNECTION INSTRUCTIONS"
echo "=============================================="
echo ""
echo "The server will start on port $PORT"
echo ""
echo "From your LOCAL machine, create SSH tunnel:"
echo "  ssh -L ${PORT}:${SLURMD_NODENAME}:${PORT} $(whoami)@hpcslurm-slurm-login-001"
echo ""
echo "Then access vLLM API at:"
echo "  http://localhost:${PORT}/v1"
echo ""
echo "Test with:"
echo "  curl http://localhost:${PORT}/v1/models"
echo ""
echo "Or in Python:"
echo "  export VLLM_BASE_URL=http://localhost:${PORT}"
echo "=============================================="
echo ""

# Start vLLM server
echo "Starting vLLM server with model: $MODEL"
echo "This may take a few minutes to download and load the model..."
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --trust-remote-code \
    --dtype auto
