# BioPipelines Query Flow Testing Plan - November 26, 2025

## Overview
Complete end-to-end validation of the query flow from natural language intent → workflow generation → execution → monitoring → diagnosis and results.

---

## 1. FRONTEND TESTING - Gradio UI Tabs

### 1.1 Tab Switching & Navigation
- [ ] **Workflow Tab**: Opens without errors
  - [ ] Form inputs render correctly (query text area, parameters)
  - [ ] Submit button is clickable
  - [ ] Output area displays generated workflow
  
- [ ] **Diagnosis Tab**: Tab switching works
  - [ ] Uploads log file correctly
  - [ ] Displays diagnosis results with error category
  - [ ] Shows suggested fixes
  - [ ] Auto-fix suggestions are clickable
  
- [ ] **Results Tab**: Shows output files
  - [ ] File browser renders directory structure
  - [ ] File selection works
  - [ ] Visualization (plots/HTML reports) displays correctly
  
- [ ] **Data Discovery Tab**: Search interface works
  - [ ] Query input accepts natural language
  - [ ] Source selection works (ENCODE/GEO/Ensembl)
  - [ ] Search results display with metadata

### 1.2 UI State Management
- [ ] [ ] Switching tabs preserves previous state
- [ ] [ ] Form data persists during tab navigation
- [ ] [ ] Error messages display appropriately

---

## 2. INTENT PARSING TESTING

### 2.1 Natural Language → Structured Intent Conversion

#### Test Cases:
```
Query: "Run RNA-seq analysis on human samples"
Expected Intent:
  - analysis_type: RNA_SEQ_DE
  - organism: human
  - organism_code: hsa
  - confidence: high
  
Query: "ChIP-seq for H3K27ac in mouse liver"
Expected Intent:
  - analysis_type: CHIP_SEQ
  - organism: mouse
  - target: H3K27ac
  - tissue: liver
  - confidence: high

Query: "ATAC-seq accessibility analysis"
Expected Intent:
  - analysis_type: ATAC_SEQ
  - organism: null (to be inferred)
  - confidence: medium

Query: "single-cell RNA-seq clustering"
Expected Intent:
  - analysis_type: SCRNA_SEQ
  - confidence: high
```

### 2.2 Intent Parser Validation
- [ ] All major analysis types are recognized (>90% accuracy)
- [ ] Organism names are mapped correctly (human→hsa, mouse→mmu, etc.)
- [ ] Tissue/cell-type extraction works
- [ ] Confidence scores are reasonable (0.7-1.0 for unambiguous queries)
- [ ] Edge cases handled gracefully (vague queries, typos)

### 2.3 Multi-Part Queries
- [ ] Queries with multiple datasets handled
- [ ] Batch vs. single-sample distinction works
- [ ] Parameter extraction (e.g., "50bp reads", "paired-end")

---

## 3. TOOL SELECTION LOGIC

### 3.1 Tool Selector Tests

#### Mapping Validation:
| Query | Expected Tool | Expected Modules |
|-------|--------------|-----------------|
| RNA-seq | NF-core/rnaseq | STAR, Salmon, DESeq2 |
| ChIP-seq | NF-core/chipseq | BOWTIE2, MACS2, HOMER |
| ATAC-seq | NF-core/atacseq | BOWTIE2, MACS2 |
| scRNA-seq | NF-core/scrnaseq | Cellranger, Seurat |
| WGS | NF-core/sarek | BWA, HaplotypeCaller, VEP |
| Metagenomics | NF-core/mag | Kraken2, MetaSpades |

- [ ] Correct tool selected for each analysis type
- [ ] Fallback selection for ambiguous queries
- [ ] Multiple tools considered when applicable
- [ ] Tool availability verified

### 3.2 Edge Cases
- [ ] Invalid analysis type → graceful error
- [ ] Multiple applicable tools → best match selected
- [ ] Tool version conflicts → resolution or warning

---

## 4. MODULE MAPPING - nf-core Resolution

### 4.1 Module Discovery
- [ ] nf-core modules catalog accessible
- [ ] Versions correctly listed (e.g., rnaseq/3.14.0)
- [ ] Module dependencies resolved
- [ ] Parameter schemas loaded

### 4.2 Module Parameter Mapping
- [ ] RNA-seq parameters:
  - [ ] `--aligner`: STAR vs. Hisat2
  - [ ] `--pseudo_aligner`: Salmon configuration
  - [ ] `--featurecounts_group_type`: Gene/exon
  
- [ ] ChIP-seq parameters:
  - [ ] `--narrow_peak`: Yes/No
  - [ ] `--macs_gsize`: Human/mouse defaults
  
- [ ] Universal parameters:
  - [ ] `--outdir`: Output directory
  - [ ] `--max_cpus`, `--max_memory`: Resource limits

### 4.3 Default Value Validation
- [ ] nf-core defaults applied when not specified
- [ ] Parameter conflicts detected and resolved
- [ ] Help text accurately represents options

---

## 5. WORKFLOW GENERATION - Nextflow DSL2 Output

### 5.1 Generated Workflow Structure
```
For "Run RNA-seq"
Expected files:
├── main.nf (main workflow)
├── nextflow.config (configuration)
├── params.yaml (parameters)
└── modules/ (if custom modules)
```

- [ ] main.nf syntax valid (can be linted with `nextflow lint`)
- [ ] Configuration file properly formatted
- [ ] Parameter file has correct schema (YAML)
- [ ] All imports resolve correctly

### 5.2 Workflow Logic Validation
- [ ] Process definitions correct
- [ ] Input/output channels properly connected
- [ ] Conditional logic (e.g., skip QC) works
- [ ] Publication logic sends outputs to correct directory

### 5.3 Parameter Passing
- [ ] User-provided parameters override defaults
- [ ] Sample CSV/metadata parsed correctly
- [ ] File paths validated
- [ ] Resource specs (cpus, memory) set appropriately

### 5.4 Syntax & Compilation
```bash
# Test that workflow compiles
nextflow config generated_workflow/main.nf > /dev/null
nextflow lint generated_workflow/main.nf
```

- [ ] No syntax errors
- [ ] All warnings reviewed
- [ ] Config hierarchy correct (CLI > params > defaults)

---

## 6. EXECUTION - SLURM Submission

### 6.1 Submission Tests
- [ ] Workflow submits to SLURM without errors
- [ ] Job ID returned and captured
- [ ] SLURM directives properly set (-N, -n, --mem, -t)

### 6.2 Job Configuration
- [ ] Expected queue selected (standard/debug/gpu)
- [ ] Time limit appropriate for analysis type
- [ ] Memory allocation matches process requirements
- [ ] CPU count reasonable (not exceeding node limits)

### 6.3 Error Handling
- [ ] Queue full → retry logic works
- [ ] Dependency errors → clear error message
- [ ] Authentication issues → proper handling

---

## 7. MONITORING - Real-time Job Status

### 7.1 Status Polling
```
Expected sequence:
SUBMITTED → RUNNING → COMPLETED/FAILED
```

- [ ] Job status accurately reported
- [ ] Status updates every 30 seconds (or configured interval)
- [ ] Historical status preserved
- [ ] Timestamps correct

### 7.2 Progress Tracking
- [ ] Process-level progress visible (e.g., STAR_ALIGN: 3/5)
- [ ] CPU/memory usage displayed from SLURM
- [ ] Estimated time remaining calculated
- [ ] Progress bar or percentage updates

### 7.3 Log Streaming
- [ ] Tail of .nextflow.log displayed in real-time
- [ ] Error lines highlighted
- [ ] Color coding for severity (ERROR/WARNING/INFO)
- [ ] Search/filter capability for large logs

### 7.4 Timeout Handling
- [ ] Long-running jobs monitored correctly
- [ ] No false timeout errors
- [ ] User notification before timeout

---

## 8. DIAGNOSIS - Error Detection & Auto-Fix

### 8.1 Error Detection
Test with known failure scenarios:

#### Test 1: Out-of-Memory (OOM) Error
```
Log contains:
"slurmstepd: error: Detected 1 oom-kill event(s)"
```
- [ ] Detected as OUT_OF_MEMORY category
- [ ] Risk level: HIGH
- [ ] Suggested fix: increase --max_memory

#### Test 2: File Not Found
```
Log contains:
"ERROR: File not found: /data/reference.fa"
```
- [ ] Detected as FILE_NOT_FOUND category
- [ ] Risk level: MEDIUM
- [ ] Suggested fix: verify path or download reference

#### Test 3: Container Pull Failure
```
Log contains:
"ERROR: Could not pull image docker://nfcore/rnaseq:3.14.0"
```
- [ ] Detected as CONTAINER_ERROR category
- [ ] Suggested fix: retry or use alternative registry

#### Test 4: Permission Denied
```
Log contains:
"Permission denied (os error 13)"
```
- [ ] Detected as PERMISSION_DENIED category
- [ ] Suggested fix: chmod or change directory

### 8.2 Pattern Matching Performance
- [ ] Pattern matching completes in <500ms for typical logs
- [ ] No false positives in normal output
- [ ] Handles multi-line error messages

### 8.3 Auto-Fix Suggestions
- [ ] Fix suggestions rank by risk level (SAFE → HIGH)
- [ ] Multiple suggestions for ambiguous errors
- [ ] Commands are safe to execute
- [ ] Explanations clear and actionable

---

## 9. RESULTS - Output Scanning & Visualization

### 9.1 Output Detection
For RNA-seq workflow, expected files:
```
results/
├── qc/
│   ├── multiqc_report.html ✓
│   └── fastqc/ ✓
├── star_align/
│   └── sample_*.bam ✓
├── salmon/
│   └── sample*/quant.sf ✓
├── deseq2/
│   └── *.tsv (DE results) ✓
└── logs/
    └── .nextflow.log ✓
```

- [ ] All expected files located
- [ ] File types correctly identified
- [ ] Metadata extracted (sample names, counts, etc.)

### 9.2 Visualization Generation
- [ ] MultiQC HTML report viewable
- [ ] Plot images render (FastQC, PCA, volcano)
- [ ] Table data (gene counts, DE results) sortable
- [ ] Download links functional

### 9.3 File Format Support
- [ ] BAM files: header validated, index generated
- [ ] FASTQ files: read count obtained
- [ ] TSV files: proper parsing and display
- [ ] PNG/SVG plots: correct rendering

### 9.4 Results Summary
- [ ] Sample count accurate
- [ ] Key metrics highlighted (mapping rate, number of DE genes)
- [ ] Quality assessment displayed
- [ ] PDF report generation

---

## 10. DATA DISCOVERY - ENCODE/GEO/Ensembl Searches

### 10.1 ENCODE Adapter
```
Query: "human ChIP-seq H3K27ac"
Expected:
  - Returns experiments with H3K27ac targets
  - Filters to human Homo sapiens
  - Results include FASTQ download links
```

- [ ] Query parsing correct
- [ ] API responses parsed accurately
- [ ] Result ranking by relevance
- [ ] Download URLs accessible

### 10.2 GEO Adapter
```
Query: "mouse brain RNA-seq"
Expected:
  - Returns GEO datasets (GSE*)
  - Filters to mouse Mus musculus
  - Brain tissue samples
  - SRA/FASTQ download available
```

- [ ] Query term construction correct
- [ ] Results metadata complete
- [ ] Large dataset handling (>10k results)

### 10.3 Ensembl Adapter
```
Query: "human reference genome GRCh38"
Expected:
  - Returns reference assemblies
  - Latest human genome version
  - FTP download links for FASTA/GTF
```

- [ ] Species detection accurate
- [ ] Assembly version correct
- [ ] File availability verified

### 10.4 Multi-Source Search Performance
- [ ] All sources searched in parallel (time <30s)
- [ ] Results deduplicated (no duplicates across sources)
- [ ] Result aggregation preserves metadata
- [ ] Result ranking by source relevance

### 10.5 Edge Cases
- [ ] Invalid organism → clear error message
- [ ] Network timeout → graceful fallback
- [ ] Empty results → handled correctly
- [ ] Special characters in query → properly escaped

---

## Testing Checklist Summary

### Pre-Testing Setup
- [ ] Environment variables set (LLM endpoints, API keys)
- [ ] nf-core modules available/cached
- [ ] SLURM cluster accessible
- [ ] Test data downloaded
- [ ] Test accounts created (if needed)

### Unit Tests
```bash
pytest tests/test_workflow_composer.py -v
pytest tests/unit/test_llm_adapters.py -v
pytest tests/test_diagnosis.py -v
pytest tests/test_results.py -v
pytest tests/test_data_discovery.py -v
```

### Integration Tests
```bash
pytest tests/test_integration.py -v
pytest tests/test_integration.py::TestComposerIntegration -v
pytest tests/test_integration.py::TestDiagnosisIntegration -v
pytest tests/test_integration.py::TestResultsIntegration -v
pytest tests/test_integration.py::TestDiscoveryIntegration -v
```

### Manual Testing
- [ ] Launch Gradio UI: `python -m workflow_composer.web`
- [ ] Go through each tab manually
- [ ] Submit test queries
- [ ] Monitor job execution
- [ ] Validate diagnosis suggestions
- [ ] Review result outputs

### Performance Benchmarking
- [ ] Intent parsing: <200ms per query
- [ ] Tool selection: <100ms
- [ ] Module mapping: <300ms
- [ ] Workflow generation: <1000ms
- [ ] Pattern matching: <500ms for typical logs
- [ ] Multi-source search: <30s all sources

---

## Test Data & Samples

### RNA-seq Test Sample
```bash
# Small test data
INPUT: sample_R1.fastq.gz (10M reads, ~3MB)
       sample_R2.fastq.gz
EXPECTED: DE results, QC report
TIME: ~5-10 minutes
```

### ChIP-seq Test Sample
```bash
# Small ChIP-seq data
INPUT: chip_R1.fastq.gz (5M reads)
       chip_R2.fastq.gz
       input_R1.fastq.gz
       input_R2.fastq.gz
EXPECTED: Peak calls, bigWig files
TIME: ~5 minutes
```

---

## Success Criteria

✅ **PASS if:**
- [ ] All 10 test categories have ≥95% tests passing
- [ ] No crashes or unhandled exceptions
- [ ] Appropriate error messages for failures
- [ ] Performance within SLA (see above)
- [ ] Results scientifically accurate (validate with existing benchmarks)
- [ ] End-to-end workflow completes in <15 minutes

❌ **FAIL if:**
- [ ] Any critical path broken (e.g., workflow generation fails)
- [ ] Diagnosis misses >10% of errors
- [ ] Performance >2x SLA
- [ ] Security issues (exposed credentials, etc.)
- [ ] Data loss or corruption

---

## Regression Testing

Compare with previous runs:
- [ ] Workflow generation output identical (excluding timestamps)
- [ ] Results statistics match known baseline
- [ ] Diagnosis suggestions consistent
- [ ] Performance not degraded by >10%

---

## Documentation for Tomorrow

### Test Execution Log Template
```
Date: 2025-11-26
Tester: [Your Name]
Environment: [SLURM cluster info]

FRONTEND TESTS:
- [ ] Tab switching: PASS/FAIL
- [ ] Error display: PASS/FAIL
...

ISSUES FOUND:
1. [Issue description]
   - Impact: [High/Medium/Low]
   - Workaround: [if available]

PERFORMANCE METRICS:
- Intent parsing: XXXms
- Module mapping: XXXms
...

RECOMMENDATION: [DEPLOY/HOLD]
```

---

## Additional Resources

- **NF-core documentation**: https://nf-core.co/docs/
- **Nextflow DSL2**: https://www.nextflow.io/docs/latest/dsl2.html
- **Test data repository**: `/data/test_samples/`
- **SLURM queue info**: `sinfo` / `squeue`
- **Workflow examples**: `generated_workflows/`

