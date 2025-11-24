# Parallel Validation & Translation Plan

**Date**: November 24, 2025, 18:30 UTC  
**Strategy**: Fix + Validate + Translate simultaneously  
**Goal**: 8-10 working pipelines by Week 4

---

## Current Reality Check ✅

### Successes (2/10):
- ✅ **Metagenomics**: 17 min, COMPLETED
- ✅ **Long-read**: 10 min, COMPLETED

### Failures (7/10):
- 🔴 **Hi-C**: 21 min, FAILED
- 🔴 **ATAC-seq**: 18 min, FAILED
- 🔴 **DNA-seq**: 24 min, FAILED
- 🔴 **ChIP-seq**: Multiple failures (last 52 min)
- 🔴 **scRNA-seq**: 13 min, FAILED (known whitelist issue)
- 🔴 **Methylation**: 9 min, FAILED

### Not Tested (1/10):
- 📝 **RNA-seq**: Ready to test

---

## Immediate Action Plan (Next 2 Hours)

### Step 1: Diagnose Common Failure Pattern (15 min)

**Hypothesis**: All failures are quick (<25 min) suggesting early-stage errors (not compute intensive failures)

**Actions**:
```bash
# Check one failed workflow in detail
cd /home/sdodl001_odu_edu/BioPipelines/nextflow-pipelines/workflows

# Look at chipseq.nf (failed most recently)
cat chipseq.nf | head -30

# Check if it has test data parameters
grep -n "params\." chipseq.nf | head -10

# See what the workflow expects
grep -n "input:" chipseq.nf
```

**Expected Issue**: Missing test data paths or incorrect parameter definitions

---

### Step 2: Create Simple Test Framework (30 min)

**Create**: `scripts/validate_pipeline.sh`

```bash
#!/bin/bash
# Pipeline validation test runner

PIPELINE=$1
TEST_DATA=$2

if [ -z "$PIPELINE" ]; then
    echo "Usage: $0 <pipeline_name> [test_data_path]"
    echo "Example: $0 rnaseq /data/raw/rna_seq"
    exit 1
fi

# Set defaults
WORKFLOWS_DIR="/home/sdodl001_odu_edu/BioPipelines/nextflow-pipelines/workflows"
RESULTS_DIR="/home/sdodl001_odu_edu/BioPipelines/data/results/nextflow_${PIPELINE}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Validating ${PIPELINE} pipeline ==="
echo "Workflow: ${WORKFLOWS_DIR}/${PIPELINE}.nf"
echo "Results: ${RESULTS_DIR}"
echo "Timestamp: ${TIMESTAMP}"

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Run with basic params
cd /home/sdodl001_odu_edu/BioPipelines/nextflow-pipelines

nextflow run "workflows/${PIPELINE}.nf" \
  --outdir "${RESULTS_DIR}" \
  -profile slurm \
  -with-report "${RESULTS_DIR}/report_${TIMESTAMP}.html" \
  -with-timeline "${RESULTS_DIR}/timeline_${TIMESTAMP}.html" \
  -with-dag "${RESULTS_DIR}/dag_${TIMESTAMP}.html" \
  ${TEST_DATA:+--reads "$TEST_DATA"}

# Capture exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ${PIPELINE} VALIDATED"
    echo "$TIMESTAMP: VALIDATED" >> "${RESULTS_DIR}/validation_log.txt"
else
    echo "🔴 ${PIPELINE} FAILED (exit code: $EXIT_CODE)"
    echo "$TIMESTAMP: FAILED ($EXIT_CODE)" >> "${RESULTS_DIR}/validation_log.txt"
fi

exit $EXIT_CODE
```

---

### Step 3: Fix Known Issues (45 min)

#### Issue 1: scRNA-seq Whitelist (KNOWN)
**Solution**: Add whitelist to workflow params

```bash
# Check where whitelist actually is
find /home/sdodl001_odu_edu/BioPipelines/containers -name "*737K*" -o -name "*whitelist*"

# If found, add to scrnaseq.nf params
# If not found, download:
cd /home/sdodl001_odu_edu/BioPipelines/data/references
mkdir -p 10x_whitelists
cd 10x_whitelists
wget https://github.com/10XGenomics/cellranger/raw/master/lib/python/cellranger/barcodes/737K-august-2016.txt
```

**Update workflows/scrnaseq.nf**:
```groovy
params {
    whitelist = '/home/sdodl001_odu_edu/BioPipelines/data/references/10x_whitelists/737K-august-2016.txt'
    // ... other params
}
```

#### Issue 2: Missing Test Data Paths (SUSPECTED)
**Check each workflow for hardcoded paths**:

```bash
# Audit all workflows for test data assumptions
for workflow in workflows/*.nf; do
    echo "=== $(basename $workflow) ==="
    grep -n "params\." "$workflow" | grep -E "reads|genome|input" | head -5
done
```

#### Issue 3: ChIP-seq Repeated Failures
**Investigate**: Why does it run longer but still fail?

```bash
# Check ChIP-seq workflow
cat workflows/chipseq.nf

# Look for process failures
# Common issue: peak calling needs control samples
```

---

### Step 4: Test RNA-seq (Highest Priority) (30 min)

**Why RNA-seq first**:
- Most common pipeline (foundational)
- Well-tested tools (STAR, featureCounts)
- Good reference point for others

**Test Commands**:
```bash
cd /home/sdodl001_odu_edu/BioPipelines/nextflow-pipelines

# Check if test data exists
ls -lh /home/sdodl001_odu_edu/BioPipelines/data/raw/rna_seq/

# If exists, run simple version
./scripts/validate_pipeline.sh rnaseq_simple '/data/raw/rna_seq/*_R{1,2}.fastq.gz'

# Monitor
watch -n 30 'squeue -u $USER'
```

---

## Systematic Debugging Approach

### For Each Failed Pipeline:

1. **Read the workflow file**:
   ```bash
   cat workflows/<pipeline>.nf | less
   ```

2. **Check params block**:
   ```bash
   grep -A20 "params {" workflows/<pipeline>.nf
   ```

3. **Identify required inputs**:
   ```bash
   grep "Channel.fromPath\|Channel.fromFilePairs" workflows/<pipeline>.nf
   ```

4. **Check if test data exists**:
   ```bash
   ls -lh /home/sdodl001_odu_edu/BioPipelines/data/raw/<pipeline_type>/
   ```

5. **Review container**:
   ```bash
   singularity inspect /home/sdodl001_odu_edu/BioPipelines/containers/images/<pipeline>_1.0.0.sif
   ```

6. **Fix and retest**:
   ```bash
   nextflow run workflows/<pipeline>.nf \
     --reads '<correct_path>' \
     --outdir /data/results/nextflow_<pipeline> \
     -profile slurm \
     -resume  # Resume from last successful step
   ```

---

## Translation Strategy (Parallel Track)

### As Pipelines Validate, Document Differences:

**Create**: `docs/SNAKEMAKE_TO_NEXTFLOW_TRANSLATION.md`

For each validated pipeline:

1. **Compare structure**:
   - Snakemake: `pipelines/<pipeline>/Snakefile`
   - Nextflow: `workflows/<pipeline>.nf`

2. **Document differences**:
   - Parameter handling
   - Process definitions
   - Container integration
   - Output structure

3. **Create migration notes**:
   - What changed?
   - Why changed?
   - How to run equivalent analysis?

### Example Template:

```markdown
## Pipeline: RNA-seq

### Snakemake Version
- File: `pipelines/rna_seq/Snakefile`
- Workflow steps: QC → Align → Count → DESeq2
- Container: `rna-seq_1.0.0.sif`
- Runtime: ~2 hours (50 samples)

### Nextflow Version
- File: `workflows/rnaseq_simple.nf`
- Workflow steps: [Same as Snakemake]
- Container: [Same]
- Runtime: ~1.8 hours (better parallelization)

### Key Differences:
1. **Input handling**: Nextflow uses channels, Snakemake uses wildcards
2. **Process isolation**: Nextflow processes are more independent
3. **Resume capability**: Nextflow work/ directory vs Snakemake .snakemake/
4. **Parameter syntax**: `params.reads` vs `config['reads']`

### Migration Notes:
- Users can use same test data
- Same container, same tools, same versions
- Outputs go to same location
- MultiQC reports compatible

### Validation Result:
✅ PASSED - Outputs match Snakemake (verified Nov 24, 2025)
```

---

## Daily Progress Tracking

### Today (Nov 24 Evening):
- [ ] Debug common failure pattern
- [ ] Create validation script
- [ ] Fix scRNA-seq whitelist
- [ ] Test RNA-seq simple
- [ ] Document findings

### Tomorrow (Nov 25):
- [ ] Fix 3-4 failed pipelines based on debugging
- [ ] Re-test all fixed pipelines
- [ ] Start Methylation test
- [ ] Begin translation documentation

### This Week Goal:
- [ ] 6-8 pipelines validated
- [ ] Common issues documented
- [ ] Validation framework working
- [ ] Translation pattern established

---

## Success Metrics (Updated)

### Phase 1 Week 1 (Current):
- **Target**: 4-5 validated pipelines
- **Current**: 2/10 (20%)
- **Needed**: 2-3 more this week

### Phase 1 Week 2:
- **Target**: 7-8 validated pipelines  
- **Strategy**: Fix patterns from Week 1

### Phase 1 Week 3-4:
- **Target**: 10/10 validated pipelines
- **Strategy**: Polish and document

---

## Parallel Implementation of New Features

### While Validating Pipelines:

**Track A: Validation** (described above)

**Track B: Architecture Prep** (Phase 2 preparation)
1. Design Tier 2 module containers (alignment, variant_calling, etc.)
2. Create microservice templates
3. Plan container build infrastructure

**Track C: Documentation**
1. Update PIPELINE_VALIDATION_TRACKER.md
2. Create SNAKEMAKE_TO_NEXTFLOW_TRANSLATION.md
3. Document lessons learned

**Track D: Testing Framework**
1. Create validation scripts
2. Set up automated testing
3. Build comparison tools (Nextflow vs Snakemake outputs)

---

## Questions to Answer Through Testing

1. **Why did 7/10 fail quickly?**
   - Missing test data?
   - Incorrect parameters?
   - Container issues?

2. **What's the pattern?**
   - Do all failures share common error?
   - Is it infrastructure or workflow logic?

3. **How to prevent in future?**
   - Better testing before launch?
   - Validation checklist?
   - Automated pre-flight checks?

4. **What works well?**
   - Metagenomics and Long-read succeeded
   - What did they do right?
   - Can we replicate that pattern?

---

## Next Immediate Actions (Right Now)

```bash
# 1. Check one failed workflow in detail (ChIP-seq)
cd /home/sdodl001_odu_edu/BioPipelines/nextflow-pipelines
cat workflows/chipseq.nf

# 2. Check what Metagenomics did right (it worked!)
cat workflows/metagenomics.nf

# 3. Compare the two
diff workflows/metagenomics.nf workflows/chipseq.nf

# 4. Identify the pattern
# Then we'll know how to fix the others
```

**Ready to start debugging?** Let's check those workflows and find the common issue!
