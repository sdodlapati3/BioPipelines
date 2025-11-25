#!/bin/bash
#SBATCH --job-name=vllm_fix
#SBATCH --partition=h100flex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm/vllm_fix_%j.log
#SBATCH --error=logs/slurm/vllm_fix_%j.err

# Fix vLLM environment dependencies and start server
echo "============================================"
echo "Fixing vLLM Environment"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_env

# Set HuggingFace cache
export HF_HOME="$HOME/.cache/huggingface"
mkdir -p "$HF_HOME"

# Show current versions
echo "Current versions:"
pip show huggingface-hub transformers | grep -E "Name|Version"
echo ""

# Fix huggingface-hub version (downgrade to <1.0)
echo "Downgrading huggingface-hub to compatible version..."
pip install "huggingface-hub>=0.34.0,<1.0" --quiet

echo ""
echo "Updated versions:"
pip show huggingface-hub transformers | grep -E "Name|Version"
echo ""

# Test import
echo "Testing vLLM import..."
python -c "from vllm.entrypoints.openai.api_server import *; print('vLLM import successful!')" 2>&1

if [ $? -ne 0 ]; then
    echo "Import still failing, trying to reinstall transformers..."
    pip install "transformers>=4.50.0" --quiet
    python -c "from vllm.entrypoints.openai.api_server import *; print('vLLM import successful!')" 2>&1
fi

echo ""
echo "============================================"
echo "Starting vLLM Server"
echo "============================================"

# Configuration
MODEL="mistralai/Mistral-7B-Instruct-v0.3"
PORT=8000
HOST="0.0.0.0"

echo "Model: $MODEL"
echo "Host: $HOST"
echo "Port: $PORT"
echo ""

# Write server info
INFO_FILE="$HOME/vllm_server_info.txt"
cat > "$INFO_FILE" << EOF
vLLM Server Information
=======================
Job ID: $SLURM_JOB_ID
Node: $(hostname)
Port: $PORT
Model: $MODEL
Started: $(date)
EOF

echo "Server info: $INFO_FILE"
echo ""

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --dtype auto \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9

echo "Server stopped at $(date)"
