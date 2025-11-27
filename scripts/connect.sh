#!/bin/bash
# ============================================================================
# BioPipelines Connection Helper
# ============================================================================
#
# Run this script on your LOCAL MACHINE to connect to the running server.
#
# Usage:
#   ./connect.sh                    # Auto-detect running job
#   ./connect.sh <compute_node>     # Connect to specific node
#
# ============================================================================

# Configuration - EDIT THESE for your setup
LOGIN_NODE="hpcslurm-slurm-login-001.odu.edu"
USERNAME="${USER:-your_username}"
LOCAL_PORT="${LOCAL_PORT:-7860}"
REMOTE_PORT="${REMOTE_PORT:-7860}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🔗 BioPipelines Connection Helper"
echo ""

# Get compute node from argument or try to detect
if [ -n "$1" ]; then
    COMPUTE_NODE="$1"
else
    echo "Looking for running BioPipelines job..."
    # Try to get the node from squeue (only works if run on login node)
    COMPUTE_NODE=$(ssh -o ConnectTimeout=5 ${USERNAME}@${LOGIN_NODE} \
        "squeue --me -h -o '%N' -n biopipelines 2>/dev/null" 2>/dev/null | head -1)
    
    if [ -z "$COMPUTE_NODE" ]; then
        echo ""
        echo "Could not auto-detect compute node."
        echo ""
        echo "Usage: $0 <compute_node>"
        echo "Example: $0 hpcslurm-nsh100quadflex-0"
        exit 1
    fi
fi

echo "Compute node: ${COMPUTE_NODE}"
echo "Creating SSH tunnel..."
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Tunnel: localhost:${LOCAL_PORT} → ${COMPUTE_NODE}:${REMOTE_PORT}${NC}"
echo -e "${GREEN}║                                                                  ║${NC}"
echo -e "${GREEN}║  Open in browser: ${YELLOW}http://localhost:${LOCAL_PORT}${GREEN}                    ║${NC}"
echo -e "${GREEN}║                                                                  ║${NC}"
echo -e "${GREEN}║  Press Ctrl+C to disconnect                                      ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Create the tunnel
ssh -L ${LOCAL_PORT}:${COMPUTE_NODE}:${REMOTE_PORT} ${USERNAME}@${LOGIN_NODE}
