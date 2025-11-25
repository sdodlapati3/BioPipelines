#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --partition=a100flex
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/vllm_server_%j.log
#SBATCH --error=logs/slurm/vllm_server_%j.err
# ============================================================================
# BioPipelines vLLM Server SLURM Job
# ============================================================================
# This script starts a vLLM server on a GPU node for the Workflow Composer.
#
# Usage:
#   sbatch scripts/llm/run_vllm_server.sbatch [model]
#
# After submission:
#   1. Wait for job to start: squeue -u $USER
#   2. Get node name from queue output
#   3. Create SSH tunnel: ssh -L 8000:NODE_NAME:8000 login-node
#   4. Access vLLM at: http://localhost:8000
# ============================================================================

# Configuration
MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
PORT=8000
GPU_MEMORY_UTIL=0.9
TENSOR_PARALLEL=${SLURM_GPUS_ON_NODE:-1}

# Print job info
echo "=============================================="
echo "BioPipelines vLLM Server"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TENSOR_PARALLEL"
echo "=============================================="

# Load modules (adjust for your cluster)
module purge
module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || echo "CUDA module not found"

# Activate conda environment
source ~/miniconda3/bin/activate biopipelines 2>/dev/null || \
    source ~/anaconda3/bin/activate biopipelines 2>/dev/null || \
    echo "Conda environment activation failed"

# Check GPU availability
nvidia-smi || echo "GPU not available"
echo ""

# Set HuggingFace cache directory to scratch
export HF_HOME="${SCRATCH:-/scratch/$USER}/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"
echo "HuggingFace cache: $HF_HOME"

# Print connection instructions
echo ""
echo "=============================================="
echo "CONNECTION INSTRUCTIONS"
echo "=============================================="
echo "1. From your local machine, create SSH tunnel:"
echo "   ssh -L ${PORT}:${SLURMD_NODENAME}:${PORT} ${SLURM_CLUSTER_NAME:-cluster}"
echo ""
echo "2. Then access vLLM API at:"
echo "   http://localhost:${PORT}/v1"
echo ""
echo "3. Test with curl:"
echo "   curl http://localhost:${PORT}/v1/models"
echo ""
echo "4. Set in Python:"
echo "   export VLLM_BASE_URL=http://localhost:${PORT}"
echo "=============================================="
echo ""

# Start vLLM server
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --trust-remote-code

# Keep running (server runs in foreground)
