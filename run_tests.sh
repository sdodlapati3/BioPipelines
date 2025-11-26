#!/bin/bash
#
# BioPipelines Testing Startup Script
# ====================================
# 
# Comprehensive setup and test execution script for November 26 validation.
#
# Usage:
#   ./run_tests.sh              # Full setup and testing
#   ./run_tests.sh validate     # Validate environment only
#   ./run_tests.sh quick        # Quick test run
#   ./run_tests.sh manual       # Show manual checklist
#   ./run_tests.sh diagnose     # Diagnosis tests only
#

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$WORKSPACE_ROOT/test_run_${TIMESTAMP}.log"

# Functions
print_header() {
    echo -e "\n${BOLD}${BLUE}========================================${NC}"
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "${BOLD}${BLUE}========================================${NC}\n"
}

print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

# Main execution
main() {
    local command="${1:-all}"
    
    print_header "BioPipelines Testing Suite - $TIMESTAMP"
    
    cd "$WORKSPACE_ROOT"
    
    case "$command" in
        validate)
            validate_environment
            ;;
        quick)
            validate_environment
            run_quick_tests
            ;;
        manual)
            show_manual_checklist
            ;;
        diagnose)
            run_diagnosis_tests
            ;;
        all|"")
            full_test_run
            ;;
        *)
            print_error "Unknown command: $command"
            print_info "Usage: ./run_tests.sh [validate|quick|manual|diagnose|all]"
            exit 1
            ;;
    esac
}

validate_environment() {
    print_header "ENVIRONMENT VALIDATION"
    
    if command -v python3 &> /dev/null; then
        print_status "Python3 found: $(python3 --version)"
    else
        print_error "Python3 not found"
        exit 1
    fi
    
    # Run validation script
    if [[ -f "validate_environment.py" ]]; then
        python3 validate_environment.py --quick
    else
        print_warning "validate_environment.py not found"
    fi
}

run_quick_tests() {
    print_header "QUICK TEST RUN"
    
    if command -v pytest &> /dev/null; then
        print_info "Running unit tests (timeout: 60s)..."
        timeout 60 pytest tests/test_workflow_composer.py -v --tb=short 2>&1 | tee -a "$LOG_FILE" || true
        print_info "Quick tests completed"
    else
        print_error "pytest not found. Install with: pip install pytest"
        return 1
    fi
}

run_diagnosis_tests() {
    print_header "DIAGNOSIS TESTS"
    
    if command -v pytest &> /dev/null; then
        print_info "Running diagnosis tests..."
        timeout 120 pytest tests/test_diagnosis.py -v --tb=short 2>&1 | tee -a "$LOG_FILE" || true
        print_info "Diagnosis tests completed"
    else
        print_error "pytest not found"
        return 1
    fi
}

full_test_run() {
    print_header "FULL TEST EXECUTION"
    
    # Check prerequisites
    print_info "Checking prerequisites..."
    
    if ! command -v pytest &> /dev/null; then
        print_error "pytest not installed"
        print_info "Install with: pip install pytest"
        exit 1
    fi
    
    if ! command -v nextflow &> /dev/null; then
        print_warning "Nextflow not in PATH (some tests may skip)"
    fi
    
    # Validate environment
    print_info "Validating environment..."
    python3 validate_environment.py --quick 2>&1 | tee -a "$LOG_FILE" || true
    
    # Run test runner
    if [[ -f "test_runner.py" ]]; then
        print_info "Running comprehensive tests..."
        python3 test_runner.py --all --save 2>&1 | tee -a "$LOG_FILE" || true
    else
        print_warning "test_runner.py not found, using pytest directly"
        pytest tests/ -v --tb=short 2>&1 | tee -a "$LOG_FILE" || true
    fi
    
    # Show results
    print_header "TEST EXECUTION COMPLETE"
    print_info "Full log saved to: $LOG_FILE"
    tail -30 "$LOG_FILE"
}

show_manual_checklist() {
    print_header "MANUAL TESTING CHECKLIST"
    
    if [[ -f "test_runner.py" ]]; then
        python3 test_runner.py --manual 2>&1 | tee -a "$LOG_FILE"
    else
        print_warning "test_runner.py not found"
        
        # Basic checklist
        cat << 'EOF'

FRONTEND TESTING (Gradio UI):
  [ ] Start: python -m workflow_composer.web
  [ ] Test Workflow tab
  [ ] Test Diagnosis tab
  [ ] Test Results tab
  [ ] Test Data Discovery tab

INTENT PARSING:
  [ ] Query: "RNA-seq human" → RNA_SEQ_DE intent
  [ ] Query: "ChIP-seq H3K27ac" → CHIP_SEQ intent
  [ ] Query: "mouse brain scRNA-seq" → SCRNA_SEQ intent

TOOL SELECTION:
  [ ] RNA-seq → nf-core/rnaseq
  [ ] ChIP-seq → nf-core/chipseq
  [ ] ATAC-seq → nf-core/atacseq

WORKFLOW GENERATION:
  [ ] Generated workflow valid
  [ ] nextflow lint passes
  [ ] Parameters correct

EXECUTION:
  [ ] SLURM job submits
  [ ] Job ID returned
  [ ] Job appears in squeue

DIAGNOSIS:
  [ ] OOM error detected
  [ ] File-not-found detected
  [ ] Container error detected
  [ ] Suggestions generated

RESULTS:
  [ ] MultiQC report renders
  [ ] Output files located
  [ ] Visualizations display

DATA DISCOVERY:
  [ ] ENCODE search works
  [ ] GEO search works
  [ ] Ensembl search works

EOF
    fi
}

# Show usage
usage() {
    cat << 'EOF'

BioPipelines Testing Startup Script
===================================

USAGE:
  ./run_tests.sh [COMMAND]

COMMANDS:
  validate    Validate environment only
  quick       Quick test run (unit tests)
  manual      Show manual testing checklist
  diagnose    Run diagnosis tests only
  all         Full test execution (default)

EXAMPLES:
  ./run_tests.sh              # Full test suite
  ./run_tests.sh validate     # Check setup
  ./run_tests.sh quick        # Rapid validation
  ./run_tests.sh manual       # Testing guide

OUTPUT:
  Results logged to: test_run_*.log
  JSON report generated: test_results_*.json

EOF
}

# Make script executable and run
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
