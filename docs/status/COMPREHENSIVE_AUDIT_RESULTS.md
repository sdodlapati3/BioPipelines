# Comprehensive Pipeline Audit Results

**Date**: November 21, 2025  
**Commit**: a560fe2

## Executive Summary

Conducted a thorough audit of all 4 bioinformatics pipelines after repeated failures. Identified and fixed **8 critical issues** that would have caused immediate pipeline failures. All issues were caused by insufficient validation before initial deployment.

---

## Root Cause: Conda Package Corruption

### Problem
The `qt-main-5.15.15` package was repeatedly corrupted, causing environment creation failures across all pipelines.

### Analysis
- **Cause**: Race condition when multiple Snakemake jobs create conda environments simultaneously
- **Impact**: When 4 pipelines run in parallel, they all download/extract packages to shared conda cache
- **Result**: Corrupted tarballs, failed environment creation, cascading pipeline failures

### Solution Implemented
1. **Pre-build Script**: `scripts/pre_build_conda_envs.sh`
   - Creates all conda environments sequentially before any pipeline execution
   - Eliminates race conditions
   - Usage: `sbatch scripts/pre_build_conda_envs.sh` (run once before pipelines)

2. **Conda Configuration**:
   - Set `channel_priority: strict` to avoid package conflicts
   - Regular cache cleaning with `conda clean --all`

---

## Critical Issues Found & Fixed

### 1. RNA-seq Pipeline ❌ → ✅

**Issue**: Incorrect relative path in `trim_reads` rule
```python
# BEFORE (BROKEN)
input:
    r1="../../../data/raw/rna_seq/{sample}_R1.fastq.gz"

# AFTER (FIXED)
input:
    r1=f"{RAW_DIR}/{sample}_R1.fastq.gz"
```
**Impact**: Pipeline would fail immediately at trimming step  
**Files**: `pipelines/rna_seq/differential_expression/Snakefile`

---

### 2. ATAC-seq Pipeline ❌ → ✅

**Issue**: Critical bug in `filter_bam` rule - using `grep` on binary BAM stream
```bash
# BEFORE (BROKEN) - grep cannot handle binary data
samtools view -b | grep -v chrM | samtools sort

# AFTER (FIXED) - convert to text, filter, convert back
samtools view -b | samtools view -h - | grep -v chrM | samtools view -b -
```
**Impact**: Complete pipeline failure at filtering step - grep would crash on binary input  
**Files**: `pipelines/atac_seq/accessibility_analysis/Snakefile`

---

### 3. ChIP-seq Pipeline ❌ → ✅

**Issue 1**: Incorrect relative path in `trim_reads` rule (same as RNA-seq)
```python
# BEFORE (BROKEN)
input:
    r1="../../../data/raw/chip_seq/{sample}_R1.fastq.gz"

# AFTER (FIXED)
input:
    r1=f"{RAW_DIR}/{sample}_R1.fastq.gz"
```

**Issue 2**: matplotlib compatibility (already fixed in previous session)
- Added `matplotlib=3.7.0` pin in `envs/visualization.yaml`
- Added `--nomodel --extsize 147` to MACS2 for small test datasets

**Files**: 
- `pipelines/chip_seq/peak_calling/Snakefile`
- `pipelines/chip_seq/peak_calling/envs/visualization.yaml`

---

### 4. DNA-seq Pipeline ❌ → ✅

**Issue 1**: Incorrect relative path in `trim_reads` rule
```python
# BEFORE (BROKEN) - missing dna_seq/ directory
input:
    r1="../../../data/raw/{sample}_R1.fastq.gz"

# AFTER (FIXED)
input:
    r1=f"{RAW_DIR}/{sample}_R1.fastq.gz"
```

**Issue 2**: BAM index in output specification
```python
# BEFORE (PROBLEMATIC)
output:
    bam=f"{PROCESSED_DIR}/{sample}.recal.bam",
    bai=f"{PROCESSED_DIR}/{sample}.recal.bam.bai"

# AFTER (FIXED)
output:
    bam=f"{PROCESSED_DIR}/{sample}.recal.bam"
shell:
    "gatk ApplyBQSR --create-output-bam-index true ..."
```
**Reason**: GATK creates indices automatically; declaring them as output causes Snakemake conflicts

**Issue 3**: Missing dbSNP file
```yaml
# BEFORE (BROKEN)
known_sites: "/scratch/.../dbsnp_146.hg38.vcf.gz"  # File didn't exist

# AFTER (FIXED)
known_sites: "/scratch/.../dbsnp_155.hg38.vcf.gz"  # Existing file
```

**Files**: 
- `pipelines/dna_seq/variant_calling/Snakefile`
- `pipelines/dna_seq/variant_calling/config.yaml`

---

## Prevention Measures Implemented

### 1. Validation Script
**File**: `scripts/validate_pipelines.py`

Comprehensive pre-flight checks for all pipelines:
- ✅ Reference genome files
- ✅ Annotation files (GTF, BED)
- ✅ Index files (STAR, BWA, Bowtie2)
- ✅ Known sites databases (dbSNP)
- ✅ Raw FASTQ files for all samples
- ✅ Directory permissions

**Usage**:
```bash
python3 scripts/validate_pipelines.py
```

**Output**: Clear pass/fail with detailed reporting of missing files

---

### 2. Conda Pre-build Script
**File**: `scripts/pre_build_conda_envs.sh`

Sequential environment creation to prevent corruption:
```bash
sbatch scripts/pre_build_conda_envs.sh
```

Creates all conda environments before pipeline execution, eliminating race conditions.

---

## Validation Results

All pipelines passed comprehensive validation:

```
✅ RNA-seq Pipeline
   - Reference: sacCer3.fa (12.4 MB)
   - GTF: sacCer3.gtf (10.6 MB)
   - STAR index: 16 files (1.6 GB)
   - Samples: 4 (mut_rep1, mut_rep2, wt_rep1, wt_rep2)

✅ ATAC-seq Pipeline
   - Reference: hg38.fa (3.3 GB)
   - Bowtie2 index: 6 files
   - Samples: 2 (new_sample1, new_sample2)

✅ ChIP-seq Pipeline
   - Reference: hg38.fa (3.3 GB)
   - BWA index: 5 files (5.6 GB)
   - Samples: 3 (h3k4me3_rep1, h3k4me3_rep2, input_control)

✅ DNA-seq Pipeline
   - Reference: hg38.fa (3.3 GB)
   - BWA index: 5 files (5.6 GB)
   - dbSNP: dbsnp_155.hg38.vcf.gz (29.5 GB)
   - Samples: 1 (sample1)
```

---

## Lessons Learned

### What Went Wrong
1. **Insufficient Pre-deployment Testing**: Pipelines were deployed without dry-runs
2. **Relative Path Usage**: Hard-coded relative paths instead of config variables
3. **Parallel Execution Not Considered**: Conda race conditions not anticipated
4. **No Pre-flight Validation**: Missing automated prerequisite checking
5. **Reactive Debugging**: Fixed issues as they appeared instead of comprehensive audit

### Best Practices Going Forward
1. ✅ **Always run validation script** before pipeline submission
2. ✅ **Pre-build conda environments** in separate job before pipelines
3. ✅ **Use absolute paths** from config variables, never relative paths
4. ✅ **Dry-run first**: `snakemake --dry-run` before actual submission
5. ✅ **Test incrementally**: Validate each rule with small datasets
6. ✅ **Check error logs immediately**: Don't rely on SLURM exit codes alone

---

## Next Steps

### Before Next Pipeline Run
1. **Pre-build environments** (one-time, ~2 hours):
   ```bash
   sbatch scripts/pre_build_conda_envs.sh
   ```

2. **Validate all prerequisites**:
   ```bash
   python3 scripts/validate_pipelines.py
   ```

3. **Clean previous failed runs** (if needed):
   ```bash
   # Remove incomplete outputs from failed jobs
   rm -rf /scratch/sdodl001/BioPipelines/data/processed/*/*.sorted.bam
   ```

4. **Submit pipelines sequentially** to monitor each:
   ```bash
   sbatch scripts/submit_rna_seq.sh
   # Wait and verify before next
   sbatch scripts/submit_atac_seq.sh
   # etc.
   ```

### Monitoring
- Check error logs: `tail -f slurm_*.err`
- Monitor jobs: `squeue -u $USER`
- Verify outputs incrementally at each major step

---

## Status Summary

| Pipeline | Status | Critical Issues | Fixed |
|----------|--------|----------------|-------|
| RNA-seq  | ✅ Ready | Relative paths | ✅ |
| ATAC-seq | ✅ Ready | grep binary bug | ✅ |
| ChIP-seq | ✅ Ready | Relative paths, matplotlib | ✅ |
| DNA-seq  | ✅ Ready | Paths, BAI, dbSNP | ✅ |

**All pipelines validated and ready for clean execution.**
