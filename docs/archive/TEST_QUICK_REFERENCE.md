# BioPipelines Testing Quick Reference - Nov 26, 2025

## ⚡ Quick Start

### Environment Setup
```bash
# Navigate to workspace
cd /home/sdodl001_odu_edu/BioPipelines

# Activate Python environment
conda activate biopipelines  # or your environment

# Install in dev mode if not already
pip install -e .

# Verify LLM server available
curl http://localhost:11434/api/tags  # Ollama
# or check OpenAI/Anthropic API keys
echo $OPENAI_API_KEY
```

### Run All Tests
```bash
# Show manual checklist
python test_runner.py --manual

# Run all automated tests
python test_runner.py --all -v

# Run specific component
python test_runner.py --components diagnosis -v

# Save results
python test_runner.py --all --save
```

---

## 📋 Testing Components

### 1️⃣ FRONTEND (15 min)
```bash
# Start UI
python -m workflow_composer.web

# Or if that doesn't work
cd src && python -c "from workflow_composer.web import create_app; app = create_app(); app.launch()"

# Then navigate to: http://localhost:7860
```

**Quick Test:**
- [x] Tab 1: Type "RNA-seq human" → Submit
- [x] Tab 2: Upload `.nextflow.log` file from `tests/test_data/`
- [x] Tab 3: Browse to `data/results/`
- [x] Tab 4: Search "human ChIP-seq"

### 2️⃣ INTENT PARSING (5 min)
```bash
pytest tests/test_workflow_composer.py::TestImports -v
```

**Expected Outputs:**
| Query | Analysis Type | Organism | Confidence |
|-------|--------------|----------|-----------|
| "RNA-seq human" | RNA_SEQ_DE | hsa | 0.95+ |
| "ChIP-seq H3K27ac" | CHIP_SEQ | None | 0.90+ |
| "mouse brain scRNA-seq" | SCRNA_SEQ | mmu | 0.95+ |

### 3️⃣ TOOL SELECTION (5 min)
```bash
pytest tests/test_integration.py::TestComposerIntegration -v
```

**Expected Mapping:**
```
RNA-seq          → nf-core/rnaseq
ChIP-seq         → nf-core/chipseq
ATAC-seq         → nf-core/atacseq
scRNA-seq        → nf-core/scrnaseq
WGS              → nf-core/sarek
Metagenomics     → nf-core/mag
```

### 4️⃣ MODULE MAPPING (10 min)
```bash
pytest tests/test_workflow_composer.py::TestConfig -v
pytest tests/test_workflow_composer.py::TestDataDownloader -v
```

**Check:**
- [ ] Modules load from nf-core
- [ ] Default parameters sensible
- [ ] Version conflicts detected

### 5️⃣ WORKFLOW GENERATION (10 min)
```bash
# Test with a generated workflow
cd generated_workflows/
nextflow lint chipseqpeakcalling_20251125_201817/main.nf

# Try to preview config
nextflow config chipseqpeakcalling_20251125_201817/main.nf 2>&1 | head -20
```

**Verify:**
- [ ] No syntax errors in main.nf
- [ ] Config file valid YAML
- [ ] All imports resolve

### 6️⃣ EXECUTION (15 min)
```bash
# Submit test job
cd generated_workflows/workflow_20251125_092924/
sbatch submit.sh

# Check status
squeue -u $USER

# Tail logs
tail -f .nextflow.log
```

**Monitor:**
- [ ] Job ID returned
- [ ] Job appears in squeue
- [ ] CPU/memory allocation appropriate

### 7️⃣ MONITORING (10 min)
```bash
# Watch job progress
watch -n 5 'squeue -u $USER'

# Check logs in real-time
tail -f .nextflow.log | grep -E "ERROR|WARN|Completed"

# Get job stats
sstat -j <JOBID> --format=JobID,MaxRSS,Elapsed
```

**Verify:**
- [ ] Status updates every 30s
- [ ] CPU/memory tracking works
- [ ] Log tail readable

### 8️⃣ DIAGNOSIS (10 min)
```bash
pytest tests/test_diagnosis.py::TestErrorDiagnosisAgent -v
```

**Test Errors:**
```bash
# OOM Error
echo "slurmstepd: error: Detected 1 oom-kill event(s)" | python -c \
  "from workflow_composer.diagnosis import ErrorDiagnosisAgent; agent = ErrorDiagnosisAgent(); import asyncio; print(asyncio.run(agent.diagnose(input())))"

# File Not Found
echo "ERROR: File not found: /data/reference.fa" | python -c \
  "from workflow_composer.diagnosis import ErrorDiagnosisAgent; agent = ErrorDiagnosisAgent(); import asyncio; print(asyncio.run(agent.diagnose(input())))"
```

**Expected:**
- [ ] OOM → OUT_OF_MEMORY category
- [ ] File error → FILE_NOT_FOUND category
- [ ] Suggestions generated
- [ ] Risk levels assigned

### 9️⃣ RESULTS (10 min)
```bash
pytest tests/test_results.py -v

# Or manually check outputs
ls -la data/results/*/
head -20 data/results/*/multiqc_report.html
```

**Verify:**
- [ ] MultiQC HTML renders
- [ ] File metadata captured
- [ ] Visualizations display
- [ ] Export works

### 🔟 DATA DISCOVERY (15 min)
```bash
pytest tests/test_data_discovery.py::TestDataDiscovery -v
```

**Manual Tests:**
```python
from workflow_composer.data.discovery import DataDiscovery

discovery = DataDiscovery()

# ENCODE search
results = discovery.search("human ChIP-seq H3K27ac", sources=['encode'], max_results=3)
print(results)

# GEO search
results = discovery.search("mouse brain RNA-seq", sources=['geo'], max_results=3)
print(results)

# Multi-source
results = discovery.search("reference genome GRCh38", sources=['encode','geo','ensembl'], max_results=5)
print(results)
```

---

## 🐛 Troubleshooting

### LLM Server Not Available
```bash
# Check Ollama
ollama serve &
ollama pull llama3:8b

# Or set API key
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."
```

### SLURM Issues
```bash
# Check cluster
sinfo

# Check permissions
id
groups

# Check account
sacctmgr list user $USER

# Check partition limits
sinfo -p debug,standard
```

### Test Data Missing
```bash
# Create minimal test data
mkdir -p tests/test_data
mkdir -p data/results/test_sample/

# Create test FASTQ (1000 reads)
python -c "
import gzip
with gzip.open('tests/test_data/test_R1.fastq.gz', 'wt') as f:
    for i in range(1000):
        f.write(f'@read_{i}\nACGTACGTACGT\n+\nIIIIIIIIIIII\n')
"
```

### Performance Too Slow
```bash
# Check system resources
free -h
df -h
top -b -n 1

# Check network
ping -c 3 8.8.8.8
```

---

## 📊 Success Criteria

### ✅ PASS
- [ ] All 10 components ≥95% passing
- [ ] No crashes or exceptions
- [ ] All timings <2x SLA
- [ ] Results scientifically reasonable
- [ ] Full workflow <15 min end-to-end

### ⚠️ WARN
- [ ] 90-95% pass rate
- [ ] Performance 1.5-2x SLA
- [ ] Minor cosmetic issues

### ❌ FAIL
- [ ] <90% pass rate
- [ ] Critical path broken
- [ ] Performance >2x SLA
- [ ] Data loss or corruption
- [ ] Security issues

---

## 📝 Documentation

### Test Results Template
Save as `test_results_20251126.md`:

```markdown
# Test Results - November 26, 2025

**Date:** 2025-11-26  
**Tester:** [Name]  
**Environment:** SLURM / [Details]  
**Duration:** [Time]

## Component Results

| Component | Status | Issues | Notes |
|-----------|--------|--------|-------|
| Frontend | PASS | - | All tabs working |
| Intent | PASS | - | Accuracy 95%+ |
| Tool Selection | PASS | - | All major types mapped |
| Module Mapping | PASS | - | Parameters validated |
| Workflow Gen | PASS | - | Syntax verified |
| Execution | PASS | - | SLURM working |
| Monitoring | PASS | - | Real-time updates |
| Diagnosis | PASS | - | 4/4 error types detected |
| Results | PASS | - | Visualizations working |
| Discovery | PASS | - | Multi-source search <30s |

## Issues Found

### Critical
- [None]

### High Priority
- [None]

### Medium Priority
- [None]

### Low Priority
- [None]

## Performance Metrics

| Component | Time | SLA | Status |
|-----------|------|-----|--------|
| Intent Parsing | 85ms | 200ms | ✓ |
| Tool Selection | 45ms | 100ms | ✓ |
| Module Mapping | 250ms | 300ms | ✓ |
| Workflow Gen | 800ms | 1000ms | ✓ |
| Diagnosis | 120ms | 500ms | ✓ |
| Discovery (3 sources) | 22s | 30s | ✓ |

## Recommendation

**✅ READY TO DEPLOY**

All tests passing. Performance within SLA. Recommend production deployment.
```

---

## 🔗 Useful Commands

```bash
# Show test structure
find tests -name "*.py" -type f | sort

# Run with coverage
pytest tests/ --cov=src/workflow_composer --cov-report=html

# Run specific test class
pytest tests/test_diagnosis.py::TestErrorDiagnosisAgent -v

# Run with markers
pytest tests/ -m "not slow" -v

# Parallel testing (if supported)
pytest tests/ -n 4

# Debug mode
pytest tests/test_diagnosis.py -vvv -s

# Generate test report
pytest tests/ --html=report.html --self-contained-html

# Check configuration
python -c "from workflow_composer.config import load_config; print(load_config())"

# List available analysis types
python -c "from workflow_composer.core import AnalysisType; print([t.value for t in AnalysisType])"

# List available tools
python -c "from workflow_composer.core import ToolSelector; selector = ToolSelector(); print(selector.tools)"

# Check installed packages
pip list | grep -E "nextflow|nf-core|workflow"
```

---

## ⏱️ Estimated Timing

| Component | Time | Notes |
|-----------|------|-------|
| Frontend | 15 min | Manual + automated |
| Intent | 5 min | Quick test |
| Tool Selection | 5 min | Quick test |
| Module Mapping | 10 min | Parameter checking |
| Workflow Gen | 10 min | Lint + verify |
| Execution | 15 min | Submit + monitor |
| Monitoring | 10 min | Status tracking |
| Diagnosis | 10 min | Error patterns |
| Results | 10 min | Output scanning |
| Discovery | 15 min | API searches |
| **TOTAL** | **~2 hours** | Buffer for issues |

---

## 🎯 Priorities

### Must Test (Critical Path)
1. Intent Parsing → ✅
2. Tool Selection → ✅
3. Workflow Generation → ✅
4. Diagnosis → ✅
5. Results → ✅

### Should Test (Important)
6. Module Mapping → ⚠️
7. Data Discovery → ⚠️
8. Frontend → ⚠️

### Nice to Test (Nice-to-Have)
9. Execution → 🔄
10. Monitoring → 🔄

---

## 📞 Support

If issues arise:
1. Check logs: `.nextflow.log`, `slurm-*.out`
2. Review error message: look in `DIAGNOSIS` section
3. Check system resources: `free -h`, `df -h`
4. Verify API connections: curl, telnet
5. Escalate if: critical path broken, data corrupted

---

**Good luck with testing tomorrow! 🚀**

