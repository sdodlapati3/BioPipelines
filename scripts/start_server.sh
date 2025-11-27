#!/bin/bash
# ============================================================================
# BioPipelines Server Launcher
# ============================================================================
#
# Simple, focused script to start the BioPipelines web server.
# Auto-detects GPU resources and selects appropriate model.
#
# Usage:
#   ./start_server.sh              # Auto-detect (GPU job or cloud)
#   ./start_server.sh --cloud      # Force cloud mode (no GPU)
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PORT="${PORT:-7860}"
VLLM_PORT="${VLLM_PORT:-8000}"
MODE="auto"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --cloud|-c) MODE="cloud"; shift ;;
        --port) PORT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--cloud] [--port PORT]"
            echo ""
            echo "  (default)   Auto-detect GPU and submit SLURM job"
            echo "  --cloud     Run on login node with cloud LLM"
            exit 0 ;;
        *) shift ;;
    esac
done

# ============================================================================
# Auto-detect best partition and model
# ============================================================================
select_gpu_config() {
    # Check available partitions
    if sinfo -h -o "%P %a" 2>/dev/null | grep -q "h100quadflex.*up"; then
        echo "h100quadflex 4 MiniMaxAI/MiniMax-M2 800G 52"
    elif sinfo -h -o "%P %a" 2>/dev/null | grep -q "h100dualflex.*up"; then
        echo "h100dualflex 2 meta-llama/Llama-3.3-70B-Instruct 450G 26"
    elif sinfo -h -o "%P %a" 2>/dev/null | grep -q "h100flex.*up"; then
        echo "h100flex 1 Qwen/Qwen2.5-Coder-32B-Instruct 200G 26"
    elif sinfo -h -o "%P %a" 2>/dev/null | grep -q "t4flex.*up"; then
        echo "t4flex 1 Qwen/Qwen2.5-3B-Instruct 56G 8"
    else
        echo ""
    fi
}

# ============================================================================
# Cloud mode: Run on login node
# ============================================================================
run_cloud() {
    echo "🧬 BioPipelines - Cloud Mode"
    echo ""
    
    cd "$PROJECT_DIR"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ~/envs/biopipelines 2>/dev/null || conda activate base
    
    # Load API keys
    [ -f ".secrets/lightning_key" ] && export LIGHTNING_API_KEY=$(cat .secrets/lightning_key)
    [ -f ".secrets/openai_key" ] && export OPENAI_API_KEY=$(cat .secrets/openai_key)
    [ -f ".secrets/google_api_key" ] && export GOOGLE_API_KEY=$(cat .secrets/google_api_key)
    
    pip install -e . -q 2>/dev/null
    python -m workflow_composer.web.gradio_app --host 0.0.0.0 --port "$PORT"
}

# ============================================================================
# GPU mode: Submit SLURM job
# ============================================================================
run_gpu() {
    local PARTITION="$1"
    local NUM_GPUS="$2"
    local MODEL="$3"
    local MEM="$4"
    local CPUS="$5"
    
    echo "🧬 BioPipelines - GPU Mode"
    echo "   Partition: $PARTITION ($NUM_GPUS GPU)"
    echo "   Model: $MODEL"
    echo ""
    
    mkdir -p "${PROJECT_DIR}/logs"
    
    JOB_ID=$(sbatch --parsable << EOF
#!/bin/bash
#SBATCH --job-name=biopipelines
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=8:00:00
#SBATCH --output=${PROJECT_DIR}/logs/server_%j.out
#SBATCH --error=${PROJECT_DIR}/logs/server_%j.err

echo "Job: \$SLURM_JOB_ID on \$SLURM_NODELIST"
echo "GPUs: ${NUM_GPUS}, Model: ${MODEL}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/envs/biopipelines 2>/dev/null || conda activate base
cd ${PROJECT_DIR}

# Load API keys
[ -f ".secrets/lightning_key" ] && export LIGHTNING_API_KEY=\$(cat .secrets/lightning_key)
[ -f ".secrets/hf_token" ] && export HF_TOKEN=\$(cat .secrets/hf_token)
[ -f ".secrets/hf_token" ] && export HUGGING_FACE_HUB_TOKEN=\$(cat .secrets/hf_token)

# Optimize MoE performance for H100 (use Cutlass instead of DeepGEMM for MoE layers)
export VLLM_MOE_USE_DEEP_GEMM=0

# Start vLLM with model-specific parameters
echo "Starting vLLM..."
VLLM_CMD="python -m vllm.entrypoints.openai.api_server --model ${MODEL} --host 0.0.0.0 --port ${VLLM_PORT} --tensor-parallel-size ${NUM_GPUS} --gpu-memory-utilization 0.90 --max-model-len 32768 --trust-remote-code"

# Add MiniMax-M2 specific parameters
if [[ "${MODEL}" == *"MiniMax"* ]]; then
    VLLM_CMD="\${VLLM_CMD} --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think --enable-auto-tool-choice"
fi

eval \${VLLM_CMD} 2>&1 &
VLLM_PID=\$!

# Wait for vLLM
for i in {1..120}; do
    curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1 && break
    sleep 5
    echo "  Waiting... [\$((i*5))s]"
done
echo "vLLM ready"

# Start Gradio
export VLLM_URL="http://localhost:${VLLM_PORT}/v1"
export USE_LOCAL_LLM="true"
echo ""
echo "=========================================="
echo "Server running on: http://\${SLURM_NODELIST}:${PORT}"
echo ""
echo "To access from your browser, run this on the LOGIN NODE:"
echo "  ssh -N -L 7860:\${SLURM_NODELIST}:${PORT} localhost &"
echo "Then open: http://localhost:7860"
echo "=========================================="
python -m workflow_composer.web.gradio_app --host 0.0.0.0 --port ${PORT}

kill \$VLLM_PID 2>/dev/null
EOF
)
    
    echo "✅ Job $JOB_ID submitted"
    echo ""
    echo "Monitor: tail -f ${PROJECT_DIR}/logs/server_${JOB_ID}.out"
    echo "Cancel:  scancel $JOB_ID"
}

# ============================================================================
# Main
# ============================================================================
if [ "$MODE" = "cloud" ]; then
    run_cloud
    exit 0
fi

# Auto-detect GPU
GPU_CONFIG=$(select_gpu_config)

if [ -z "$GPU_CONFIG" ]; then
    echo "No GPU partitions available, using cloud mode"
    run_cloud
else
    read -r PARTITION NUM_GPUS MODEL MEM CPUS <<< "$GPU_CONFIG"
    run_gpu "$PARTITION" "$NUM_GPUS" "$MODEL" "$MEM" "$CPUS"
fi
