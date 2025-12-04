# BioPipelines Makefile
# Convenience commands for development and testing

.PHONY: help test test-quick test-full evaluate analyze baseline report ci clean

# Default Python environment
PYTHON ?= python3
PYTHONPATH ?= $(PWD)/src

help:
	@echo "BioPipelines Development Commands"
	@echo "=================================="
	@echo ""
	@echo "Testing Commands:"
	@echo "  make test        - Run quick evaluation tests"
	@echo "  make test-quick  - Run quick evaluation tests (alias)"
	@echo "  make test-full   - Run full evaluation (~110+ tests)"
	@echo "  make evaluate    - Run full evaluation with HTML report"
	@echo ""
	@echo "Analysis Commands:"
	@echo "  make analyze     - Analyze error patterns from latest run"
	@echo "  make baseline    - Create new baseline from current run"
	@echo "  make report      - Generate HTML report from latest results"
	@echo ""
	@echo "CI Commands:"
	@echo "  make ci          - Run CI-friendly test (returns exit codes)"
	@echo "  make ci-strict   - Run strict CI test"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make clean       - Clean up generated reports"
	@echo "  make install     - Install dependencies"
	@echo ""

# Testing Commands
test: test-quick

test-quick:
	@echo "Running quick evaluation..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ci_test.py --quick

test-full:
	@echo "Running full evaluation..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ci_test.py

evaluate:
	@echo "Running full evaluation with report..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/unified_evaluation_runner.py

# Analysis Commands
analyze:
	@echo "Analyzing error patterns..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/error_pattern_analyzer.py

baseline:
	@echo "Creating new baseline..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ci_test.py --create-baseline

report:
	@echo "Generating HTML report..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/unified_evaluation_runner.py --no-compare

# CI Commands
ci:
	@echo "Running CI tests..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ci_test.py --quick

ci-strict:
	@echo "Running strict CI tests..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ci_test.py --strict

# Utility Commands
clean:
	@echo "Cleaning up reports..."
	rm -f reports/evaluation/evaluation_*.json
	rm -f reports/evaluation/report_*.html
	rm -f reports/evaluation/error_analysis_*.md
	rm -f reports/evaluation/error_analysis_*.json
	@echo "Done. Baseline preserved."

install:
	pip install -r requirements-composer.txt
	pip install pytest pytest-asyncio

# Development helpers
lint:
	@echo "Running linters..."
	ruff check src/ tests/
	mypy src/agents/ --ignore-missing-imports

format:
	@echo "Formatting code..."
	ruff format src/ tests/

# Run specific test category
test-category:
	@if [ -z "$(CATEGORY)" ]; then \
		echo "Usage: make test-category CATEGORY=data_discovery"; \
		exit 1; \
	fi
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/unified_evaluation_runner.py --category $(CATEGORY)
