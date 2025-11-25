#!/bin/bash
#SBATCH --job-name=vllm_server
#SBATCH --partition=h100flex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm/vllm_server_%j.log
#SBATCH --error=logs/slurm/vllm_server_%j.err

# vLLM Server Script
# Uses existing vllm_env conda environment

echo "============================================"
echo "vLLM Server Startup"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Load CUDA module if available
module load cuda 2>/dev/null || true

# GPU info
echo "GPU Information:"
nvidia-smi
echo ""

# Activate conda environment
echo "Activating vllm_env..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_env

# Set HuggingFace cache to home directory
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE"

echo "HF_HOME: $HF_HOME"

# Verify vLLM installation
echo ""
echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"
echo ""

# Configuration
MODEL="mistralai/Mistral-7B-Instruct-v0.3"
PORT=8000
HOST="0.0.0.0"

echo "============================================"
echo "Starting vLLM Server"
echo "============================================"
echo "Model: $MODEL"
echo "Host: $HOST"
echo "Port: $PORT"
echo ""

# Write server info for clients
INFO_FILE="$HOME/vllm_server_info.txt"
cat > "$INFO_FILE" << EOF
vLLM Server Information
=======================
Job ID: $SLURM_JOB_ID
Node: $(hostname)
Host: $HOST
Port: $PORT
Model: $MODEL
Started: $(date)
EOF

echo "Server info written to: $INFO_FILE"
echo ""
echo "To connect from login node, run:"
echo "  ssh -L $PORT:$(hostname):$PORT $(hostname) -N &"
echo "  Then access: http://localhost:$PORT"
echo ""

# Start vLLM server with OpenAI-compatible API
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --dtype auto \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9

echo ""
echo "vLLM server stopped at $(date)"
