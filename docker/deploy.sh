#!/bin/bash
# BioPipelines deployment helper script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check dependencies
check_deps() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    log_info "Dependencies OK"
}

# Start development environment
dev_up() {
    log_info "Starting development environment..."
    cd "$SCRIPT_DIR"
    docker compose -f docker-compose.dev.yml up -d
    log_info "PostgreSQL: localhost:5432"
    log_info "Redis: localhost:6379"
    log_info "Run API locally with: python -m uvicorn workflow_composer.api.app:create_app --reload"
}

# Stop development environment
dev_down() {
    log_info "Stopping development environment..."
    cd "$SCRIPT_DIR"
    docker compose -f docker-compose.dev.yml down
}

# Start full stack
prod_up() {
    log_info "Starting full stack..."
    cd "$SCRIPT_DIR"
    
    if [ ! -f .env ]; then
        log_warn ".env file not found, copying from .env.example"
        cp .env.example .env
        log_warn "Please edit .env with your configuration"
    fi
    
    docker compose up -d
    log_info "API: http://localhost:8000"
    log_info "Health: http://localhost:8000/health"
}

# Stop full stack
prod_down() {
    log_info "Stopping full stack..."
    cd "$SCRIPT_DIR"
    docker compose down
}

# Show status
status() {
    cd "$SCRIPT_DIR"
    docker compose ps
}

# Show logs
logs() {
    cd "$SCRIPT_DIR"
    docker compose logs -f "$@"
}

# Build images
build() {
    log_info "Building Docker images..."
    cd "$PROJECT_ROOT"
    docker build -f docker/Dockerfile.api -t biopipelines-api:latest .
    docker build -f docker/Dockerfile.worker -t biopipelines-worker:latest .
    log_info "Images built successfully"
}

# Run tests in Docker
test() {
    log_info "Running tests..."
    cd "$SCRIPT_DIR"
    docker compose -f docker-compose.dev.yml up -d
    cd "$PROJECT_ROOT"
    
    export DATABASE_URL=postgresql://biopipelines:dev_password@localhost:5432/biopipelines_dev
    export REDIS_URL=redis://localhost:6379/0
    
    python -m pytest tests/ -v --tb=short
}

# Show help
help() {
    echo "BioPipelines Deployment Script"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  dev-up      Start development environment (PostgreSQL + Redis)"
    echo "  dev-down    Stop development environment"
    echo "  up          Start full production stack"
    echo "  down        Stop full production stack"
    echo "  status      Show container status"
    echo "  logs        Show logs (optionally specify service)"
    echo "  build       Build Docker images"
    echo "  test        Run tests with Docker services"
    echo "  help        Show this help message"
}

# Main
case "${1:-help}" in
    dev-up)    check_deps; dev_up ;;
    dev-down)  dev_down ;;
    up)        check_deps; prod_up ;;
    down)      prod_down ;;
    status)    status ;;
    logs)      shift; logs "$@" ;;
    build)     check_deps; build ;;
    test)      check_deps; test ;;
    help|*)    help ;;
esac
