#!/bin/bash
# ============================================================================
# BioPipelines Server - Single Entry Point
# ============================================================================
#
# LOGIC:
#   1. Check if vLLM SLURM job already running
#      - YES → Connect to it, start Gradio (instant)
#      - NO  → Submit vLLM job, wait for model load, then start Gradio
#
#   2. vLLM runs on compute node (GPU), Gradio runs on login node (accessible)
#   3. SSH tunnel connects them: login:8000 → compute:8000
#
# Usage:
#   ./start_server.sh              # Auto: reuse existing or start new
#   ./start_server.sh --small      # Use Llama-70B (faster, 3-5 min load)
#   ./start_server.sh --cloud      # Use cloud LLM (no GPU)
#   ./start_server.sh --stop       # Stop vLLM server
#   ./start_server.sh --status     # Check vLLM status
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PORT="${PORT:-7860}"
VLLM_PORT="${VLLM_PORT:-8000}"
MODE="auto"
MODEL_SIZE="large"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --cloud|-c) MODE="cloud"; shift ;;
        --gpu|-g) MODE="gpu"; shift ;;
        --small|-s) MODE="gpu"; MODEL_SIZE="small"; shift ;;
        --stop) MODE="stop"; shift ;;
        --status) MODE="status"; shift ;;
        --port) PORT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  (default)      Auto: use existing vLLM or cloud fallback"
            echo "  --gpu, -g      Start local vLLM (MiniMax-M2, 45-60 min load)"
            echo "  --small, -s    Start local vLLM (Llama-70B, 3-5 min load)"
            echo "  --cloud, -c    Use cloud LLM only (no GPU)"
            echo "  --status       Check if vLLM is running"
            echo "  --stop         Stop vLLM server"
            echo ""
            exit 0 ;;
        *) shift ;;
    esac
done

cd "$PROJECT_DIR"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/envs/biopipelines 2>/dev/null || conda activate base
mkdir -p "${PROJECT_DIR}/logs"

# ============================================================================
# Helper: Find running vLLM job
# ============================================================================
find_vllm_job() {
    squeue --me -h -o "%i %j %T %N" 2>/dev/null | grep -E "vllm|biopipelines" | grep "RUNNING" | head -1
}

# ============================================================================
# --status: Check vLLM status
# ============================================================================
if [ "$MODE" = "status" ]; then
    VLLM_JOB=$(find_vllm_job)
    if [ -n "$VLLM_JOB" ]; then
        JOB_ID=$(echo "$VLLM_JOB" | awk '{print $1}')
        NODE=$(echo "$VLLM_JOB" | awk '{print $4}')
        echo "✅ vLLM running: Job $JOB_ID on $NODE"
        
        # Check health via direct connection
        ENDPOINT="http://${NODE}:${VLLM_PORT}"
        if curl -s "${ENDPOINT}/health" > /dev/null 2>&1; then
            MODEL=$(curl -s "${ENDPOINT}/v1/models" 2>/dev/null | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
            echo "   Health: OK"
            echo "   Model: $MODEL"
            echo "   Endpoint: $ENDPOINT"
        else
            echo "   Health: Not ready (still loading)"
            echo "   Monitor: tail -f ${PROJECT_DIR}/logs/vllm_${JOB_ID}.out"
        fi
    else
        echo "❌ No vLLM server running"
    fi
    exit 0
fi

# ============================================================================
# --stop: Stop vLLM
# ============================================================================
if [ "$MODE" = "stop" ]; then
    VLLM_JOB=$(find_vllm_job)
    if [ -n "$VLLM_JOB" ]; then
        JOB_ID=$(echo "$VLLM_JOB" | awk '{print $1}')
        scancel $JOB_ID
        echo "✅ Stopped vLLM job: $JOB_ID"
    else
        echo "No vLLM job running"
    fi
    pkill -f "ssh.*${VLLM_PORT}:" 2>/dev/null || true
    exit 0
fi

# ============================================================================
# --cloud: Run without GPU
# ============================================================================
if [ "$MODE" = "cloud" ]; then
    echo "🧬 BioPipelines - Cloud Mode"
    echo ""
    
    [ -f ".secrets/lightning_key" ] && export LIGHTNING_API_KEY=$(cat .secrets/lightning_key)
    [ -f ".secrets/openai_key" ] && export OPENAI_API_KEY=$(cat .secrets/openai_key)
    [ -f ".secrets/google_api_key" ] && export GOOGLE_API_KEY=$(cat .secrets/google_api_key)
    
    pip install -e . -q 2>/dev/null
    
    echo "Starting Gradio with public URL..."
    python -m workflow_composer.web.app --host 0.0.0.0 --port "$PORT" --share
    exit 0
fi

# ============================================================================
# --gpu: Start local vLLM server
# ============================================================================
if [ "$MODE" = "gpu" ]; then
    echo "🧬 BioPipelines - GPU Mode"
    echo ""
    
    # Load secrets
    [ -f ".secrets/hf_token" ] && export HF_TOKEN=$(cat .secrets/hf_token)
    [ -f ".secrets/hf_token" ] && export HUGGING_FACE_HUB_TOKEN=$(cat .secrets/hf_token)
    [ -f ".secrets/lightning_key" ] && export LIGHTNING_API_KEY=$(cat .secrets/lightning_key)
    [ -f ".secrets/openai_key" ] && export OPENAI_API_KEY=$(cat .secrets/openai_key)
    [ -f ".secrets/google_api_key" ] && export GOOGLE_API_KEY=$(cat .secrets/google_api_key)
    
    # Check if vLLM already running
    VLLM_JOB=$(find_vllm_job)
    if [ -n "$VLLM_JOB" ]; then
        JOB_ID=$(echo "$VLLM_JOB" | awk '{print $1}')
        COMPUTE_NODE=$(echo "$VLLM_JOB" | awk '{print $4}')
        echo "vLLM already running: Job $JOB_ID on $COMPUTE_NODE"
        echo "Use --stop first if you want to restart."
        echo ""
    else
        # Select model
        if [ "$MODEL_SIZE" = "small" ]; then
            PARTITION="h100dualflex"
            NUM_GPUS=2
            MODEL="meta-llama/Llama-3.3-70B-Instruct"
            MEM="450G"
            CPUS=26
            LOAD_TIME="3-5 minutes"
        else
            PARTITION="h100quadflex"
            NUM_GPUS=4
            MODEL="MiniMaxAI/MiniMax-M2"
            MEM="800G"
            CPUS=52
            LOAD_TIME="45-60 minutes"
        fi
        
        echo "Starting vLLM server..."
        echo "   Model: $MODEL"
        echo "   Partition: $PARTITION ($NUM_GPUS GPUs)"
        echo "   Load time: $LOAD_TIME"
        echo ""
        
        # Submit job
        JOB_ID=$(sbatch --parsable << EOF
#!/bin/bash
#SBATCH --job-name=vllm-biopipelines
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=24:00:00
#SBATCH --output=${PROJECT_DIR}/logs/vllm_%j.out
#SBATCH --error=${PROJECT_DIR}/logs/vllm_%j.err

echo "================================================"
echo "vLLM Server - Job \$SLURM_JOB_ID on \$SLURM_NODELIST"
echo "Started: \$(date)"
echo "Model: ${MODEL}"
echo "================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/envs/biopipelines 2>/dev/null || conda activate base

export HF_TOKEN=\$(cat ${PROJECT_DIR}/.secrets/hf_token 2>/dev/null || echo "")
export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN
export VLLM_MOE_USE_DEEP_GEMM=0

echo "Starting vLLM..."

if [[ "${MODEL}" == *"MiniMax"* ]]; then
    python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL} \
        --host 0.0.0.0 \
        --port ${VLLM_PORT} \
        --tensor-parallel-size ${NUM_GPUS} \
        --gpu-memory-utilization 0.90 \
        --max-model-len 32768 \
        --trust-remote-code \
        --tool-call-parser minimax_m2 \
        --reasoning-parser minimax_m2_append_think \
        --enable-auto-tool-choice
else
    python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL} \
        --host 0.0.0.0 \
        --port ${VLLM_PORT} \
        --tensor-parallel-size ${NUM_GPUS} \
        --gpu-memory-utilization 0.90 \
        --max-model-len 32768 \
        --trust-remote-code
fi
EOF
)
        
        echo "Submitted job: $JOB_ID"
        echo ""
        
        # Wait for job to start
        echo "Waiting for job to start..."
        for i in $(seq 1 40); do
            STATE=$(squeue -j $JOB_ID -h -o "%T" 2>/dev/null)
            if [ "$STATE" = "RUNNING" ]; then
                break
            elif [ -z "$STATE" ]; then
                echo "❌ Job failed to start"
                cat "${PROJECT_DIR}/logs/vllm_${JOB_ID}.err" 2>/dev/null | tail -10
                exit 1
            fi
            if [ $((i % 2)) -eq 0 ]; then
                echo "  Job state: $STATE [$((i*15))s]"
            fi
            sleep 15
        done
        
        COMPUTE_NODE=$(squeue -j $JOB_ID -h -o "%N")
        echo "✅ Running on: $COMPUTE_NODE"
        echo ""
    fi
    
    # Now continue to start Gradio (fall through to auto mode logic)
    MODE="auto"
fi

# ============================================================================
# Main: Auto mode - smart detection
# ============================================================================
echo "🧬 BioPipelines"
echo ""

# Load API keys for cloud fallback
[ -f ".secrets/lightning_key" ] && export LIGHTNING_API_KEY=$(cat .secrets/lightning_key)
[ -f ".secrets/openai_key" ] && export OPENAI_API_KEY=$(cat .secrets/openai_key)
[ -f ".secrets/google_api_key" ] && export GOOGLE_API_KEY=$(cat .secrets/google_api_key)
[ -f ".secrets/hf_token" ] && export HF_TOKEN=$(cat .secrets/hf_token)
[ -f ".secrets/hf_token" ] && export HUGGING_FACE_HUB_TOKEN=$(cat .secrets/hf_token)

# Check for existing vLLM job
VLLM_JOB=$(find_vllm_job)

if [ -n "$VLLM_JOB" ]; then
    # Found running vLLM job
    JOB_ID=$(echo "$VLLM_JOB" | awk '{print $1}')
    COMPUTE_NODE=$(echo "$VLLM_JOB" | awk '{print $4}')
    VLLM_ENDPOINT="http://${COMPUTE_NODE}:${VLLM_PORT}"
    
    echo "Found vLLM: Job $JOB_ID on $COMPUTE_NODE"
    
    # Check if it's ready
    if curl -s --connect-timeout 3 "${VLLM_ENDPOINT}/v1/models" 2>/dev/null | grep -q "id"; then
        MODEL_NAME=$(curl -s --connect-timeout 3 "${VLLM_ENDPOINT}/v1/models" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)
        echo "✅ vLLM ready! Model: $MODEL_NAME"
        export VLLM_URL="${VLLM_ENDPOINT}/v1"
        export USE_LOCAL_LLM="true"
    else
        echo "⏳ vLLM loading... using cloud LLM as fallback"
        echo "   Monitor: tail -f ${PROJECT_DIR}/logs/vllm_${JOB_ID}.err"
        export VLLM_URL="${VLLM_ENDPOINT}/v1"
        export USE_LOCAL_LLM="false"  # Will auto-switch when ready
    fi
else
    # No vLLM job running - use cloud only
    echo "No local vLLM running - using cloud LLM"
    echo ""
    echo "To start local vLLM (MiniMax-M2 456B):"
    echo "  ./start_server.sh --gpu"
    echo ""
    export USE_LOCAL_LLM="false"
fi

# ============================================================================
# Start Gradio with smart LLM selection
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════╗"
echo "║          🧬 BioPipelines                   ║"
echo "╠════════════════════════════════════════════╣"
echo "║  Server: http://localhost:${PORT}          ║"
if [ "$USE_LOCAL_LLM" = "true" ]; then
echo "║  LLM: Local vLLM (${COMPUTE_NODE})         ║"
else
echo "║  LLM: Cloud (OpenAI/Gemini fallback)       ║"
fi
echo "╚════════════════════════════════════════════╝"
echo ""

pip install -e . -q 2>/dev/null
python -m workflow_composer.web.app --host 0.0.0.0 --port ${PORT} --share
