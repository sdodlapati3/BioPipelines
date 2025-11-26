# BioPipelines Testing & Validation - November 26, 2025

## 📋 Overview

This document serves as the central hub for tomorrow's comprehensive testing of the BioPipelines query flow from natural language input → workflow generation → execution → monitoring → diagnosis and results collection.

**Target Date:** November 26, 2025  
**Estimated Duration:** ~2 hours (full suite)  
**Success Criteria:** All 10 components ≥95% passing, end-to-end workflow <15 minutes

---

## 📚 Documentation Files

### Primary Testing Documents
1. **[TESTING_PLAN_20251126.md](./TESTING_PLAN_20251126.md)** - Comprehensive test plan
   - 10 test categories with detailed test cases
   - Expected outputs and acceptance criteria
   - Test data specifications
   - Regression testing guidelines
   - Success/failure criteria

2. **[TEST_QUICK_REFERENCE.md](./TEST_QUICK_REFERENCE.md)** - Quick reference guide
   - Command cheatsheet
   - Rapid test execution commands
   - Troubleshooting section
   - Performance SLAs
   - ~2 hour estimated timeline

### Test Execution Tools
3. **[test_runner.py](./test_runner.py)** - Automated test runner
   ```bash
   python test_runner.py --all -v              # Run all tests
   python test_runner.py --components frontend # Specific component
   python test_runner.py --manual              # Show checklist
   ```

4. **[validate_environment.py](./validate_environment.py)** - Environment validator
   ```bash
   python validate_environment.py              # Full validation
   python validate_environment.py --llm        # LLM check only
   python validate_environment.py --slurm      # SLURM check only
   python validate_environment.py --quick      # Quick checks
   ```

5. **[run_tests.sh](./run_tests.sh)** - Bash startup script
   ```bash
   ./run_tests.sh validate                     # Validate environment
   ./run_tests.sh quick                        # Quick tests
   ./run_tests.sh manual                       # Show manual checklist
   ./run_tests.sh all                          # Full suite
   ```

---

## 🚀 Quick Start for Tomorrow

### Step 1: Pre-Testing Setup (5 min)
```bash
cd /home/sdodl001_odu_edu/BioPipelines

# Validate environment
python validate_environment.py --quick

# Fix any issues (see troubleshooting in TEST_QUICK_REFERENCE.md)
```

### Step 2: Automated Testing (30-45 min)
```bash
# Run comprehensive test suite
python test_runner.py --all --save

# Or use the startup script
./run_tests.sh all
```

### Step 3: Manual Testing (45-60 min)
```bash
# Show manual checklist
python test_runner.py --manual

# Follow items in: TEST_QUICK_REFERENCE.md "Quick Start" section
```

### Step 4: Documentation (5 min)
Document results in `test_results_20251126.md` template

---

## 🎯 10 Testing Components

All components must achieve ≥95% pass rate:

| # | Component | File | Priority | Time |
|---|-----------|------|----------|------|
| 1 | **Frontend** - Gradio UI tabs | Manual | Important | 15 min |
| 2 | **Intent Parsing** - NL → intent | test_workflow_composer.py | Critical | 5 min |
| 3 | **Tool Selection** - Intent → tool | test_integration.py | Critical | 5 min |
| 4 | **Module Mapping** - Tool → modules | test_workflow_composer.py | Important | 10 min |
| 5 | **Workflow Generation** - Generate DSL2 | test_workflow_composer.py | Critical | 10 min |
| 6 | **Execution** - SLURM submission | test_integration.py | Important | 15 min |
| 7 | **Monitoring** - Real-time status | test_integration.py | Important | 10 min |
| 8 | **Diagnosis** - Error detection | test_diagnosis.py | Critical | 10 min |
| 9 | **Results** - Output scanning | test_results.py | Critical | 10 min |
| 10 | **Data Discovery** - Search ENCODE/GEO/Ensembl | test_data_discovery.py | Important | 15 min |

**Critical Path (must work):** 2, 3, 5, 8, 9

---

## 📊 Test Execution Matrix

### Automated Tests (via pytest)
```bash
# Unit tests for each component
pytest tests/test_workflow_composer.py -v      # Intent, Tool Selection, Modules, WF Gen
pytest tests/test_diagnosis.py -v              # Diagnosis
pytest tests/test_results.py -v                # Results
pytest tests/test_data_discovery.py -v         # Data Discovery

# Integration tests
pytest tests/test_integration.py -v            # Cross-component validation

# Performance benchmarks
pytest tests/test_integration.py::TestPerformance -v
```

### Manual Tests
- Frontend UI tab switching
- SLURM job submission
- Real-time monitoring
- Workflow compilation

---

## 📈 Performance SLAs

All timings must be <2x these targets:

| Component | Target | Max Acceptable |
|-----------|--------|----------------|
| Intent Parsing | 200ms | 400ms |
| Tool Selection | 100ms | 200ms |
| Module Mapping | 300ms | 600ms |
| Workflow Generation | 1000ms | 2000ms |
| Diagnosis (pattern matching) | 500ms | 1000ms |
| Data Discovery (3 sources) | 30s | 60s |
| **Full end-to-end workflow** | 10 min | 15 min |

---

## ✅ Success Criteria

### PASS (Deploy) ✅
- [ ] All 10 components ≥95% passing
- [ ] No crashes or unhandled exceptions
- [ ] Appropriate error messages for all failures
- [ ] Performance within SLA for all components
- [ ] Results scientifically accurate (validated)
- [ ] End-to-end workflow <15 minutes

### WARN (Mostly Ready) ⚠️
- [ ] 90-95% pass rate (minor issues)
- [ ] Performance 1.5-2x SLA
- [ ] Non-critical features have workarounds

### FAIL (Hold) ❌
- [ ] <90% pass rate
- [ ] Critical path broken
- [ ] Performance >2x SLA
- [ ] Security issues
- [ ] Data loss or corruption

---

## 🔧 Common Commands for Tomorrow

### Environment Check
```bash
# Full validation
python validate_environment.py

# Quick check
python validate_environment.py --quick

# LLM endpoints
python validate_environment.py --llm

# SLURM cluster
python validate_environment.py --slurm
```

### Test Execution
```bash
# Run all tests
python test_runner.py --all -v

# Run specific component
python test_runner.py --components intent,diagnosis -v

# Run specific suite
python test_runner.py --suite integration -v

# Show manual checklist
python test_runner.py --manual

# Save results to JSON
python test_runner.py --all --save
```

### Startup Script
```bash
# Full test run
./run_tests.sh

# Validate only
./run_tests.sh validate

# Quick tests
./run_tests.sh quick

# Manual checklist
./run_tests.sh manual
```

### Direct Testing
```bash
# Component-specific tests
pytest tests/test_workflow_composer.py::TestImports -v
pytest tests/test_diagnosis.py::TestErrorDiagnosisAgent -v
pytest tests/test_data_discovery.py::TestDataDiscovery -v

# Integration suite
pytest tests/test_integration.py::TestComposerIntegration -v
pytest tests/test_integration.py::TestDiagnosisIntegration -v
pytest tests/test_integration.py::TestResultsIntegration -v

# Full test run
pytest tests/ -v --tb=short
```

---

## 🐛 Troubleshooting Quick Guide

### Issue: "LLM server not available"
```bash
# Start Ollama
ollama serve &
ollama pull llama3:8b

# Or set OpenAI key
export OPENAI_API_KEY="sk-..."
```

### Issue: "SLURM commands not found"
```bash
# Check if on login node
sinfo
# If fails, may need to load module
module load slurm
```

### Issue: "Test data missing"
```bash
# Create minimal test data
mkdir -p tests/test_data
python -c "import gzip; open('tests/test_data/test.txt', 'w').write('test')"
```

### Issue: "pytest not found"
```bash
pip install pytest pytest-cov pytest-timeout
```

### Issue: "Nextflow syntax error"
```bash
# Lint generated workflow
nextflow lint generated_workflows/main.nf

# Check config
nextflow config generated_workflows/main.nf > /dev/null
```

**See TEST_QUICK_REFERENCE.md for more troubleshooting.**

---

## 📝 Test Result Template

Save results as `test_results_20251126.md`:

```markdown
# Test Results - November 26, 2025

**Date:** 2025-11-26  
**Tester:** [Your Name]  
**Environment:** [SLURM cluster info]  
**Total Time:** [Duration]

## Component Results

| Component | Status | Issues | Time |
|-----------|--------|--------|------|
| Frontend | PASS/FAIL | - | 15m |
| Intent | PASS/FAIL | - | 5m |
| Tool Sel | PASS/FAIL | - | 5m |
| Modules | PASS/FAIL | - | 10m |
| WF Gen | PASS/FAIL | - | 10m |
| Execute | PASS/FAIL | - | 15m |
| Monitor | PASS/FAIL | - | 10m |
| Diagnose | PASS/FAIL | - | 10m |
| Results | PASS/FAIL | - | 10m |
| Discovery | PASS/FAIL | - | 15m |

## Issues Found

[Document any failures or warnings]

## Performance

[Include actual timings vs SLA]

## Recommendation

**READY/HOLD** - [Brief summary]
```

---

## 📞 Key Resources

### Test Data
- Small test datasets: `/data/test_samples/`
- Generated workflows: `generated_workflows/`
- Test results expected: `data/results/`

### Configuration
- Default config: `config/defaults.yaml`
- SLURM profile: `config/slurm.yaml`
- Composer settings: `config/composer.yaml`

### Documentation
- Architecture: `docs/ARCHITECTURE_AND_GUIDE.md`
- API Reference: `docs/API_REFERENCE.md`
- Quick Start: `docs/QUICK_START_CONTAINERS.md`
- Workflow Composer: `docs/WORKFLOW_COMPOSER_GUIDE.md`

### Existing Tests
- Unit: `tests/test_*.py`
- Integration: `tests/test_integration.py`
- Test data: `tests/test_data/`

---

## 🎓 Testing Best Practices

1. **Test in order:** Frontend → Intent → Tools → Modules → WF Gen → Execution → Monitoring → Diagnosis → Results → Discovery
2. **Document everything:** Save logs and screenshots
3. **Test edge cases:** Vague queries, typos, invalid inputs
4. **Validate outputs:** Check against known baselines
5. **Monitor resources:** CPU, memory, disk space
6. **Clean up:** Remove test runs before final summary

---

## 📅 Tomorrow's Schedule

```
09:00 - Environment validation (validate_environment.py)
09:15 - Frontend testing (manual + automated)
09:45 - Intent parsing tests
10:00 - Tool selection tests
10:15 - Module mapping tests
10:30 - Workflow generation (lint + verify)
11:00 - Break / Issue triage
11:15 - SLURM execution test
11:45 - Real-time monitoring
12:15 - Lunch break
13:00 - Diagnosis testing
13:30 - Results processing
14:00 - Data discovery tests
14:45 - Documentation & summary
15:15 - Go/No-go decision
```

---

## ✨ Key Success Factors

✅ **All scripts executable and properly configured**  
✅ **Environment pre-validated before running full suite**  
✅ **Test data prepared and available**  
✅ **LLM endpoints configured (Ollama or API keys)**  
✅ **SLURM cluster accessible and responsive**  
✅ **Disk space available (>50GB recommended)**  
✅ **Clear documentation of any issues**  
✅ **Performance metrics tracked**  

---

## 🚀 Ready to Start?

```bash
# Begin tomorrow's testing
cd /home/sdodl001_odu_edu/BioPipelines

# 1. Validate environment
python validate_environment.py --quick

# 2. Run automated tests
python test_runner.py --all -v

# 3. Check manual items
python test_runner.py --manual

# 4. Document results
# ... Save results to test_results_20251126.md
```

---

**Good luck with testing tomorrow! 🚀**

For questions or issues, refer to:
- [TESTING_PLAN_20251126.md](./TESTING_PLAN_20251126.md) - Detailed test specifications
- [TEST_QUICK_REFERENCE.md](./TEST_QUICK_REFERENCE.md) - Quick commands and troubleshooting
- [docs/](./docs/) - Full documentation

