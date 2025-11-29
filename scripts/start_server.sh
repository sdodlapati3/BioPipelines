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
PORT="${PORT:-8080}"
VLLM_PORT="${VLLM_PORT:-8000}"
MODE="auto"
MODEL_SIZE="large"

# ============================================================================
# Helper: Find available port or kill existing process
# ============================================================================
find_available_port() {
    local base_port=${1:-8080}
    local max_tries=10
    
    for i in $(seq 0 $max_tries); do
        local port=$((base_port + i))
        
        # Check if port is in use
        local pid=$(lsof -t -i :$port 2>/dev/null | head -1)
        
        if [ -z "$pid" ]; then
            # Port is free
            echo $port
            return 0
        fi
        
        # Port is in use - check what's using it
        local cmd=$(ps -p $pid -o args= 2>/dev/null || echo "")
        
        # If it's our workflow_composer or gradio, kill it
        if [[ "$cmd" == *"workflow_composer"* ]] || [[ "$cmd" == *"gradio"* ]]; then
            echo "Killing existing BioPipelines server (PID $pid) on port $port..." >&2
            kill -9 $pid 2>/dev/null || true
            sleep 2
            
            # Also kill any child processes (frpc tunnel)
            pkill -9 -f "frpc.*$port" 2>/dev/null || true
            sleep 1
            
            # Verify it's dead
            if ! lsof -i :$port >/dev/null 2>&1; then
                echo $port
                return 0
            fi
        fi
        
        # If first port is taken by something else, try next port
        if [ $i -eq 0 ]; then
            echo "Port $port in use by another process, trying next..." >&2
        fi
    done
    
    # Fallback - return base port and let it fail with clear error
    echo $base_port
}

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --cloud|-c) MODE="cloud"; shift ;;
        --gpu|-g) MODE="gpu"; shift ;;
        --small|-s) MODE="gpu"; MODEL_SIZE="small"; shift ;;
        --single|-1) MODE="gpu"; MODEL_SIZE="single"; shift ;;
        --stop) MODE="stop"; shift ;;
        --status) MODE="status"; shift ;;
        --port) PORT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  (default)      Auto: use existing vLLM or cloud fallback"
            echo "  --gpu, -g      Start local vLLM with 4x H100 (MiniMax-M2, 15-25 min)"
            echo "  --small, -s    Start local vLLM with 2x H100 (Qwen3-Coder-30B, 2-4 min)"
            echo "  --single, -1   Start local vLLM with 1x H100 (DeepSeek-Coder-V2-Lite, 1-2 min)"
            echo "  --cloud, -c    Use cloud LLM only (no GPU)"
            echo "  --status       Check if vLLM is running"
            echo "  --stop         Stop vLLM server"
            echo ""
            echo "Model comparison:"
            echo "  MiniMax-M2 (4 GPU)        - Best reasoning, tool-use, 456B MoE"
            echo "  Qwen3-Coder-30B (2 GPU)   - Best for coding/agentic tasks, 30B MoE"
            echo "  DeepSeek-Coder-V2 (1 GPU) - Fast coding, 16B MoE"
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
# Helper: Cleanup any existing Gradio server
# ============================================================================
cleanup_existing_server() {
    local port=$1
    echo "Checking for existing servers on port $port..." >&2
    
    # Kill any existing BioPipelines/Gradio processes
    local pids=$(pgrep -f "workflow_composer.web.app" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Killing existing BioPipelines server processes: $pids" >&2
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    
    # Kill any frpc tunnel processes
    pids=$(pgrep -f "frpc.*$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Killing existing Gradio tunnel processes: $pids" >&2
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
    
    # Double-check port is free
    local port_pid=$(lsof -t -i :$port 2>/dev/null | head -1)
    if [ -n "$port_pid" ]; then
        echo "Port $port still in use by PID $port_pid, force killing..." >&2
        kill -9 $port_pid 2>/dev/null || true
        sleep 2
    fi
}

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
        
        # Check health via srun (avoids firewall issues from login node)
        if srun --jobid=${JOB_ID} --overlap curl -s http://localhost:8000/health > /dev/null 2>&1; then
            MODEL=$(srun --jobid=${JOB_ID} --overlap curl -s http://localhost:8000/v1/models 2>/dev/null | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
            echo "   Health: OK"
            echo "   Model: $MODEL"
            echo "   Endpoint: localhost:8000 (on compute node)"
        else
            echo "   Health: Not ready (still loading)"
            echo "   Monitor: tail -f ${PROJECT_DIR}/logs/vllm_${JOB_ID}.err"
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
    
    # Priority: GitHub Models (free) > Google Gemini (free) > OpenAI (paid)
    [ -f ".secrets/github_token" ] && export GITHUB_TOKEN=$(cat .secrets/github_token)
    [ -f ".secrets/google_api_key" ] && export GOOGLE_API_KEY=$(cat .secrets/google_api_key)
    [ -f ".secrets/openai_key" ] && export OPENAI_API_KEY=$(cat .secrets/openai_key)
    
    pip install -e . -q 2>/dev/null
    
    # Clean up any existing server first
    cleanup_existing_server $PORT
    
    # Find available port (should be $PORT after cleanup)
    PORT=$(find_available_port $PORT)
    
    echo "Starting Gradio on port $PORT with public URL..."
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
    # Priority: GitHub Models (free) > Google Gemini (free) > OpenAI (paid)
    [ -f ".secrets/hf_token" ] && export HF_TOKEN=$(cat .secrets/hf_token)
    [ -f ".secrets/hf_token" ] && export HUGGING_FACE_HUB_TOKEN=$(cat .secrets/hf_token)
    [ -f ".secrets/github_token" ] && export GITHUB_TOKEN=$(cat .secrets/github_token)
    [ -f ".secrets/google_api_key" ] && export GOOGLE_API_KEY=$(cat .secrets/google_api_key)
    [ -f ".secrets/openai_key" ] && export OPENAI_API_KEY=$(cat .secrets/openai_key)
    
    # Check if vLLM already running
    VLLM_JOB=$(find_vllm_job)
    if [ -n "$VLLM_JOB" ]; then
        JOB_ID=$(echo "$VLLM_JOB" | awk '{print $1}')
        COMPUTE_NODE=$(echo "$VLLM_JOB" | awk '{print $4}')
        echo "vLLM already running: Job $JOB_ID on $COMPUTE_NODE"
        echo "Use --stop first if you want to restart."
        echo ""
    else
        # Select model and configuration
        # MoE models (MiniMax-M2, DeepSeek) need container due to GLIBC 2.29 requirement
        # System has GLIBC 2.28 (Rocky Linux 8.10)
        
        CONTAINER_IMAGE="/scratch/sdodl001/BioPipelines/containers/vllm-openai.sif"
        USE_CONTAINER=false
        
        if [ "$MODEL_SIZE" = "single" ]; then
            # DeepSeek-Coder-V2-Lite: Fast coding model for 1x H100
            # 16B MoE, fits in single 80GB GPU
            PARTITION="h100flex"
            NUM_GPUS=1
            MODEL="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
            MEM="100G"
            CPUS=16
            LOAD_TIME="1-2 minutes"
        elif [ "$MODEL_SIZE" = "small" ]; then
            # Qwen3-Coder-30B-A3B: Best coding model for 2x H100
            # MoE with only 3B active params = fast inference
            # State-of-the-art for agentic coding tasks
            PARTITION="h100dualflex"
            NUM_GPUS=2
            MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
            MEM="200G"
            CPUS=26
            LOAD_TIME="2-4 minutes"
        else
            # MiniMax-M2: Most capable reasoning model (needs 4x H100)
            # 456B MoE with strong tool-use capabilities
            PARTITION="h100quadflex"
            NUM_GPUS=4
            MODEL="MiniMaxAI/MiniMax-M2"
            MEM="800G"
            CPUS=52
            LOAD_TIME="15-25 minutes"
            USE_CONTAINER=true
        fi
        
        echo "Starting vLLM server..."
        echo "   Model: $MODEL"
        echo "   Partition: $PARTITION ($NUM_GPUS GPUs)"
        echo "   Container: $USE_CONTAINER"
        echo "   Load time: $LOAD_TIME"
        echo ""
        
        # Check if container exists for MoE models
        if [ "$USE_CONTAINER" = true ] && [ ! -f "$CONTAINER_IMAGE" ]; then
            echo "⚠️  Container not found. Building vLLM container..."
            echo "   This is a one-time setup (~10 minutes)."
            echo ""
            mkdir -p "${PROJECT_DIR}/containers/images"
            
            # Pull official vLLM container
            singularity pull --force "$CONTAINER_IMAGE" docker://vllm/vllm-openai:latest
            
            if [ ! -f "$CONTAINER_IMAGE" ]; then
                echo "❌ Failed to build container. Falling back to Llama-3.3-70B..."
                MODEL="meta-llama/Llama-3.3-70B-Instruct"
                USE_CONTAINER=false
                MEM="450G"
                CPUS=26
            fi
        fi
        
        # Submit job
        if [ "$USE_CONTAINER" = true ]; then
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
echo "vLLM Server (Container) - Job \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Started: \$(date)"
echo "Model: ${MODEL}"
echo "================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

export HF_TOKEN=\$(cat ${PROJECT_DIR}/.secrets/hf_token 2>/dev/null || echo "")
export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN
# Use HOME cache where model is already partially downloaded (215GB)
export HF_HOME=\$HOME/.cache/huggingface

echo "Starting vLLM in container..."
echo "HF_HOME: \$HF_HOME"
echo "Container: ${CONTAINER_IMAGE}"

# Run singularity with vLLM
singularity exec --nv --bind /scratch:/scratch --bind \$HOME:\$HOME --env HF_TOKEN=\$HF_TOKEN --env HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN --env HF_HOME=\$HF_HOME --env VLLM_MOE_USE_DEEP_GEMM=0 ${CONTAINER_IMAGE} python3 -m vllm.entrypoints.openai.api_server --model ${MODEL} --host 0.0.0.0 --port ${VLLM_PORT} --tensor-parallel-size ${NUM_GPUS} --gpu-memory-utilization 0.90 --max-model-len 32768 --trust-remote-code --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think --enable-auto-tool-choice
EOF
)
        else
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

echo "Starting vLLM..."

python -m vllm.entrypoints.openai.api_server \\
    --model ${MODEL} \\
    --host 0.0.0.0 \\
    --port ${VLLM_PORT} \\
    --tensor-parallel-size ${NUM_GPUS} \\
    --gpu-memory-utilization 0.90 \\
    --max-model-len 32768 \\
    --trust-remote-code
EOF
)
        fi
        
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
# Priority: GitHub Models (free) > Google Gemini (free) > OpenAI (paid)
[ -f ".secrets/github_token" ] && export GITHUB_TOKEN=$(cat .secrets/github_token)
[ -f ".secrets/google_api_key" ] && export GOOGLE_API_KEY=$(cat .secrets/google_api_key)
[ -f ".secrets/openai_key" ] && export OPENAI_API_KEY=$(cat .secrets/openai_key)
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
    
    # Check if it's ready (use srun to check from compute node - avoids firewall issues)
    if srun --jobid=${JOB_ID} --overlap curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "id"; then
        MODEL_NAME=$(srun --jobid=${JOB_ID} --overlap curl -s http://localhost:8000/v1/models 2>/dev/null | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
        echo "✅ vLLM ready! Model: $MODEL_NAME"
        export VLLM_URL="http://localhost:8000/v1"
        export USE_LOCAL_LLM="true"
    else
        echo "⏳ vLLM loading... using cloud LLM as fallback"
        echo "   Monitor: tail -f ${PROJECT_DIR}/logs/vllm_${JOB_ID}.err"
        export VLLM_URL="http://localhost:8000/v1"
        export USE_LOCAL_LLM="false"  # Will auto-switch when ready
    fi
else
    # No vLLM job running - check if we should start one automatically
    echo "No local vLLM running"
    echo ""
    
    # Check if GPUs are available (quick sinfo check)
    GPU_AVAILABLE=$(sinfo -p h100quadflex,h100dualflex -h -o "%A" 2>/dev/null | head -1 | cut -d'/' -f1)
    
    if [ -n "$GPU_AVAILABLE" ] && [ "$GPU_AVAILABLE" != "0" ]; then
        echo "GPUs available! Starting vLLM automatically..."
        echo "Run './start_server.sh --cloud' to skip and use cloud LLM instead."
        echo ""
        
        # Start vLLM (reuse the --gpu logic)
        CONTAINER_IMAGE="/scratch/sdodl001/BioPipelines/containers/vllm-openai.sif"
        PARTITION="h100quadflex"
        NUM_GPUS=4
        MODEL="MiniMaxAI/MiniMax-M2"
        MEM="800G"
        CPUS=52
        
        if [ -f "$CONTAINER_IMAGE" ]; then
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

export HF_TOKEN=\$(cat ${PROJECT_DIR}/.secrets/hf_token 2>/dev/null || echo "")
export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN
export HF_HOME=\$HOME/.cache/huggingface

singularity exec --nv --bind /scratch:/scratch --bind \$HOME:\$HOME \
    --env HF_TOKEN=\$HF_TOKEN --env HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN \
    --env HF_HOME=\$HF_HOME --env VLLM_MOE_USE_DEEP_GEMM=0 \
    ${CONTAINER_IMAGE} python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL} --host 0.0.0.0 --port ${VLLM_PORT} \
    --tensor-parallel-size ${NUM_GPUS} --gpu-memory-utilization 0.90 \
    --max-model-len 32768 --trust-remote-code \
    --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think \
    --enable-auto-tool-choice
EOF
)
            echo "Submitted vLLM job: $JOB_ID"
            echo "Model will take 15-25 minutes to load."
            echo ""
            echo "Starting Gradio now with cloud LLM..."
            echo "Once vLLM is ready, restart: ./start_server.sh"
            echo ""
        fi
    else
        echo "No GPUs available - using cloud LLM"
        echo ""
        echo "To start local vLLM when GPUs are free:"
        echo "  ./start_server.sh --gpu"
        echo ""
    fi
    
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

# Clean up any existing server first
cleanup_existing_server $PORT

# Find available port (should be $PORT after cleanup)
PORT=$(find_available_port $PORT)
echo "Using port: $PORT"
echo ""

# If we have a local vLLM job, run Gradio on the same compute node
# This allows direct localhost:8000 access to vLLM (no firewall issues)
if [ -n "$VLLM_JOB" ] && [ -n "$JOB_ID" ]; then
    echo "Running Gradio on compute node ${COMPUTE_NODE} (same node as vLLM)..."
    echo "This allows direct localhost access to the model."
    echo ""
    
    # Export all needed env vars for the srun command
    export VLLM_URL="http://localhost:8000/v1"
    export USE_LOCAL_LLM="true"
    
    srun --jobid=${JOB_ID} --overlap \
        --export=ALL,VLLM_URL="${VLLM_URL}",USE_LOCAL_LLM="true",OPENAI_API_KEY="${OPENAI_API_KEY}",GOOGLE_API_KEY="${GOOGLE_API_KEY}",LIGHTNING_API_KEY="${LIGHTNING_API_KEY}" \
        bash -c "cd ${PROJECT_DIR} && source ~/miniconda3/etc/profile.d/conda.sh && conda activate ~/envs/biopipelines && python -m workflow_composer.web.app --host 0.0.0.0 --port ${PORT} --share"
else
    python -m workflow_composer.web.app --host 0.0.0.0 --port ${PORT} --share
fi
