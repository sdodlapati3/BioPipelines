#!/bin/bash
# ============================================================================
# BioPipelines Web UI Launcher
# ============================================================================
# 
# Launches the BioPipelines web interface for workflow generation.
#
# Usage:
#   ./scripts/launch_web_ui.sh [OPTIONS]
#
# Options:
#   --gradio      Launch Gradio UI (default, recommended)
#   --flask       Launch Flask UI (legacy)
#   --api         Launch FastAPI REST API
#   --all         Launch all services
#   --port PORT   Port to use (default: 7860 for Gradio, 5000 for Flask, 8080 for API)
#   --share       Create public Gradio link (Gradio only)
#   --debug       Enable debug mode
#
# Examples:
#   ./scripts/launch_web_ui.sh                    # Start Gradio UI
#   ./scripts/launch_web_ui.sh --api              # Start REST API
#   ./scripts/launch_web_ui.sh --gradio --share   # Start with public link
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Defaults
UI_TYPE="gradio"
PORT=""
SHARE=""
DEBUG=""
HOST="0.0.0.0"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gradio)
            UI_TYPE="gradio"
            shift
            ;;
        --flask)
            UI_TYPE="flask"
            shift
            ;;
        --api)
            UI_TYPE="api"
            shift
            ;;
        --all)
            UI_TYPE="all"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --share)
            SHARE="--share"
            shift
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        -h|--help)
            head -30 "$0" | tail -25
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║        🧬 BioPipelines - Web Interface Launcher                  ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check Python environment
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Check if package is installed
if ! python -c "import workflow_composer" 2>/dev/null; then
    echo -e "${YELLOW}Installing workflow_composer in development mode...${NC}"
    pip install -e . -q
fi

# Load environment variables
if [ -f ".env" ]; then
    echo -e "${GREEN}Loading .env file...${NC}"
    export $(grep -v '^#' .env | xargs)
fi

# Check for API keys
if [ -f ".secrets/openai_key" ]; then
    export OPENAI_API_KEY=$(cat .secrets/openai_key)
    echo -e "${GREEN}✓ OpenAI API key loaded${NC}"
fi

# Launch based on UI type
case $UI_TYPE in
    gradio)
        PORT=${PORT:-7860}
        echo -e "${GREEN}Starting Gradio UI on http://$HOST:$PORT${NC}"
        echo ""
        python -m workflow_composer.web.gradio_app --host "$HOST" --port "$PORT" $SHARE $DEBUG
        ;;
    
    flask)
        PORT=${PORT:-5000}
        echo -e "${GREEN}Starting Flask UI on http://$HOST:$PORT${NC}"
        echo ""
        python -m workflow_composer.web.app --host "$HOST" --port "$PORT" $DEBUG
        ;;
    
    api)
        PORT=${PORT:-8080}
        echo -e "${GREEN}Starting FastAPI on http://$HOST:$PORT${NC}"
        echo -e "${GREEN}API Docs: http://$HOST:$PORT/docs${NC}"
        echo ""
        
        if [ -n "$DEBUG" ]; then
            uvicorn workflow_composer.web.api:app --host "$HOST" --port "$PORT" --reload
        else
            uvicorn workflow_composer.web.api:app --host "$HOST" --port "$PORT"
        fi
        ;;
    
    all)
        echo -e "${GREEN}Starting all services...${NC}"
        echo ""
        
        # Start API in background
        echo -e "${BLUE}Starting FastAPI on port 8080...${NC}"
        uvicorn workflow_composer.web.api:app --host "$HOST" --port 8080 &
        API_PID=$!
        
        # Start Gradio (foreground)
        echo -e "${BLUE}Starting Gradio UI on port 7860...${NC}"
        python -m workflow_composer.web.gradio_app --host "$HOST" --port 7860 $SHARE
        
        # Cleanup
        kill $API_PID 2>/dev/null || true
        ;;
esac

echo -e "${GREEN}Server stopped.${NC}"
