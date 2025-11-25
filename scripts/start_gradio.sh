#!/bin/bash
# ============================================================================
# BioPipelines Web UI - Login Node Launcher
# ============================================================================
#
# Start the Gradio web interface on the login node.
#
# Usage:
#   ./scripts/start_gradio.sh
#   ./scripts/start_gradio.sh --port 8080
#   ./scripts/start_gradio.sh --share
#
# ============================================================================

set -e

# Default settings
PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"
SHARE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --share)
            SHARE="--share"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--port PORT] [--host HOST] [--share]"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║        🧬 BioPipelines - Gradio Web UI                           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Activate conda
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

# Go to project directory
cd "$PROJECT_DIR"

# Load OpenAI API key
if [ -f ".secrets/openai_key" ]; then
    export OPENAI_API_KEY=$(cat .secrets/openai_key)
    echo "✓ OpenAI API key loaded"
else
    echo "⚠ No OpenAI key found at .secrets/openai_key"
fi

# Check if package is installed
if ! python -c "import workflow_composer" 2>/dev/null; then
    echo "Installing workflow_composer..."
    pip install -e . -q
fi

echo ""
echo "Starting Gradio server..."
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  URL:  http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the server
python -m workflow_composer.web.gradio_app --host "$HOST" --port "$PORT" $SHARE
