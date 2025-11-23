#!/bin/bash
# Build all BioPipelines containers
# Usage: ./build_all.sh [--push] [--singularity]

set -euo pipefail

# Configuration
REGISTRY="${REGISTRY:-biopipelines}"
VERSION="${VERSION:-1.0.0}"
BUILD_SINGULARITY=false
PUSH_TO_REGISTRY=false

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH_TO_REGISTRY=true
            shift
            ;;
        --singularity)
            BUILD_SINGULARITY=true
            shift
            ;;
        --help)
            cat << EOF
Build BioPipelines containers

Usage: $0 [OPTIONS]

Options:
  --push           Push Docker images to registry
  --singularity    Convert to Singularity images for HPC
  --help           Show this help message

Environment Variables:
  REGISTRY         Docker registry (default: biopipelines)
  VERSION          Container version (default: 1.0.0)
  SINGULARITY_DIR  Singularity output directory (default: /scratch/sdodl001/containers)

Examples:
  # Build all containers
  ./build_all.sh

  # Build and convert to Singularity
  ./build_all.sh --singularity

  # Build and push to registry
  REGISTRY=ghcr.io/sdodlapa ./build_all.sh --push
EOF
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}BioPipelines Container Builder${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo "Registry: $REGISTRY"
echo "Version:  $VERSION"
echo "Singularity: $BUILD_SINGULARITY"
echo "Push: $PUSH_TO_REGISTRY"
echo -e "${GREEN}═══════════════════════════════════════${NC}"

cd "$(dirname "$0")/../../containers"

# Build base container first (all others depend on it)
echo -e "\n${YELLOW}[1/2] Building base container...${NC}"
docker build -t ${REGISTRY}/base:${VERSION} base/
docker tag ${REGISTRY}/base:${VERSION} ${REGISTRY}/base:latest

if [[ "$PUSH_TO_REGISTRY" == "true" ]]; then
    echo "Pushing base container..."
    docker push ${REGISTRY}/base:${VERSION}
    docker push ${REGISTRY}/base:latest
fi

# Build RNA-seq container
echo -e "\n${YELLOW}[2/2] Building RNA-seq container...${NC}"
docker build -t ${REGISTRY}/rna-seq:${VERSION} rna-seq/
docker tag ${REGISTRY}/rna-seq:${VERSION} ${REGISTRY}/rna-seq:latest

if [[ "$PUSH_TO_REGISTRY" == "true" ]]; then
    echo "Pushing RNA-seq container..."
    docker push ${REGISTRY}/rna-seq:${VERSION}
    docker push ${REGISTRY}/rna-seq:latest
fi

# Convert to Singularity if requested
if [[ "$BUILD_SINGULARITY" == "true" ]]; then
    SINGULARITY_DIR="${SINGULARITY_DIR:-/scratch/sdodl001/containers}"
    echo -e "\n${YELLOW}Converting to Singularity images...${NC}"
    mkdir -p "$SINGULARITY_DIR"
    
    for container in base rna-seq; do
        echo "Converting ${container}..."
        singularity build \
            "${SINGULARITY_DIR}/${container}_${VERSION}.sif" \
            "docker-daemon://${REGISTRY}/${container}:${VERSION}"
        
        # Create latest symlink
        ln -sf "${container}_${VERSION}.sif" "${SINGULARITY_DIR}/${container}_latest.sif"
    done
    
    echo -e "${GREEN}✓ Singularity images saved to: $SINGULARITY_DIR${NC}"
fi

echo -e "\n${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}✓ All containers built successfully${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"

# Show image sizes
echo -e "\n${YELLOW}Container sizes:${NC}"
docker images ${REGISTRY}/* --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

if [[ "$BUILD_SINGULARITY" == "true" ]]; then
    echo -e "\n${YELLOW}Singularity images:${NC}"
    ls -lh "${SINGULARITY_DIR}"/*.sif
fi
