# BioPipelines Makefile
# Convenience commands for development and testing

.PHONY: help test test-quick test-full test-advanced evaluate analyze baseline report ci clean

# Default Python environment
PYTHON ?= python3
PYTHONPATH ?= $(PWD)/src:$(PWD)/tests

help:
	@echo "BioPipelines Development Commands"
	@echo "=================================="
	@echo ""
	@echo "Testing Commands:"
	@echo "  make test           - Run quick evaluation tests"
	@echo "  make test-quick     - Run quick evaluation tests (alias)"
	@echo "  make test-full      - Run full evaluation (~110+ tests)"
	@echo "  make test-advanced  - Run advanced evaluation with all features"
	@echo "  make test-adversarial - Run adversarial/security tests"
	@echo "  make evaluate       - Run full evaluation with HTML report"
	@echo ""
	@echo "Analysis Commands:"
	@echo "  make analyze        - Analyze error patterns from latest run"
	@echo "  make baseline       - Create new baseline from current run"
	@echo "  make report         - Generate HTML report from latest results"
	@echo "  make dashboard      - Show terminal metrics dashboard"
	@echo "  make trends         - Show historical trends"
	@echo ""
	@echo "Synthetic Testing:"
	@echo "  make generate-tests - Generate synthetic test cases"
	@echo "  make augment-tests  - Augment existing tests"
	@echo ""
	@echo "CI Commands:"
	@echo "  make ci             - Run CI-friendly test (returns exit codes)"
	@echo "  make ci-strict      - Run strict CI test"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make clean          - Clean up generated reports"
	@echo "  make install        - Install dependencies"
	@echo ""

# Testing Commands
test: test-quick

test-quick:
	@echo "Running quick evaluation..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ci_test.py --quick

test-full:
	@echo "Running full evaluation..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ci_test.py

test-advanced:
	@echo "Running advanced evaluation..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/advanced_evaluation_runner.py

test-adversarial:
	@echo "Running adversarial tests..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/advanced_evaluation_runner.py --adversarial --category adversarial

evaluate:
	@echo "Running full evaluation with report..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/unified_evaluation_runner.py

evaluate-html:
	@echo "Running evaluation with HTML report..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/advanced_evaluation_runner.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/metrics_dashboard.py --html-report reports/evaluation_report.html

# Analysis Commands
analyze:
	@echo "Analyzing error patterns..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/error_pattern_analyzer.py

baseline:
	@echo "Creating new baseline..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ci_test.py --create-baseline

report:
	@echo "Generating HTML report..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/metrics_dashboard.py --html-report reports/latest_report.html

dashboard:
	@echo "Showing metrics dashboard..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/metrics_dashboard.py --last-run

trends:
	@echo "Showing historical trends..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/metrics_dashboard.py --trends

# Synthetic Testing
generate-tests:
	@echo "Generating synthetic tests..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c "from evaluation.synthetic_test_generator import TemplateGenerator; g = TemplateGenerator(); print(f'Generated {len(g.generate_all(per_category=20))} tests')"

augment-tests:
	@echo "Augmenting tests with variations..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -c "from evaluation.synthetic_test_generator import DataAugmentor; a = DataAugmentor(); print('Augmentor ready')"

# CI Commands
ci:
	@echo "Running CI tests..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ci_test.py --quick

ci-strict:
	@echo "Running strict CI tests..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/ci_test.py --strict

ci-full:
	@echo "Running full CI tests with advanced metrics..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/advanced_evaluation_runner.py --max-tests 200

# Utility Commands
clean:
	@echo "Cleaning up reports..."
	rm -f reports/evaluation/evaluation_*.json
	rm -f reports/evaluation/report_*.html
	rm -f reports/evaluation/error_analysis_*.md
	rm -f reports/evaluation/error_analysis_*.json
	rm -f reports/evaluations/*.json
	rm -f reports/evaluations/*.html
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
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/advanced_evaluation_runner.py --category $(CATEGORY)

# Run tests for specific difficulty tier
test-tier:
	@if [ -z "$(TIER)" ]; then \
		echo "Usage: make test-tier TIER=3"; \
		exit 1; \
	fi
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/advanced_evaluation_runner.py --difficulty $(TIER)

# Focus on previously failed tests
test-failures:
	@echo "Running previously failed tests..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/advanced_evaluation_runner.py --focus-failures

# Compare two evaluation runs
compare-runs:
	@if [ -z "$(RUN1)" ] || [ -z "$(RUN2)" ]; then \
		echo "Usage: make compare-runs RUN1=eval_20240101_120000 RUN2=eval_20240102_120000"; \
		exit 1; \
	fi
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/metrics_dashboard.py --compare $(RUN1) $(RUN2)
