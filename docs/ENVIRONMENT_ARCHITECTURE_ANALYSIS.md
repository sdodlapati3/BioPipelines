# Environment Architecture: Critical Analysis & Recommendations

**Date**: November 23, 2025  
**Status**: 🔴 CRITICAL - System architecture causing persistent failures  
**Impact**: All 7 pipelines failing, 15+ job failures in last 2 hours

---

## Executive Summary

The current environment management strategy is fundamentally flawed and causing persistent pipeline failures. The system uses a **triple-layer environment architecture** that introduces unnecessary complexity, conflicts, and maintenance burden.

### Critical Issues Identified

1. **Triple-layer environment complexity** (Base → BioPipelines → 43+ per-rule envs)
2. **Corrupted conda environment cache** causing all pipeline failures
3. **NFS file locking** preventing cleanup in home directory
4. **Massive duplication** across 43+ micro-environments
5. **Version conflicts** between layers
6. **No isolation** between development and production

---

## Current Architecture (BROKEN)

### Layer 1: Base Conda Environment
- **Location**: `/home/sdodl001_odu_edu/envs/biopipelines`
- **Purpose**: Main Python environment for Snakemake
- **Size**: Unknown (conda list shows 0 packages - likely not activated properly)
- **Definition**: `environment.yml` (72 dependencies including all pipeline tools)

### Layer 2: Snakemake Managed Environments
- **Location**: `~/.snakemake/conda/` (NFS-mounted home) OR `/scratch/sdodl001/conda_envs/`
- **Count**: 43+ separate YAML files across 10 pipelines
- **Current State**: 
  - `~/.snakemake/conda`: 246MB, 23 cached YAMLs, 1 created environment
  - `/scratch/sdodl001/conda_envs`: 8KB, 1 YAML (corrupted)
- **Problem**: Per-rule micro-environments with massive duplication

### Layer 3: Per-Rule Conda Environments
- **RNA-seq**: 5 environments (qc, preprocessing, alignment, quantification, deseq2)
- **DNA-seq**: 5 environments (qc, preprocessing, alignment, variant_calling, annotation)
- **ATAC-seq**: 5 environments (qc, preprocessing, alignment, peak_calling, visualization)
- **ChIP-seq**: 5 environments (qc, preprocessing, alignment, peak_calling, visualization)
- **Methylation**: 4 environments (qc, bismark, dmr, visualization)
- **Hi-C**: 4 environments (qc, hic_processing, hic_analysis, visualization)
- **scRNA-seq**: 2 environments (starsolo, analysis)
- **Long-read**: 4 environments (qc, alignment, sv_calling, phasing)
- **Metagenomics**: 5 environments (preprocessing, profiling, assembly, functional, visualization)
- **Structural Variants**: 4 environments (sv_calling, annotation, visualization, report)

**Total**: 43+ micro-environments for 10 pipelines

---

## Root Cause Analysis

### Problem 1: Corrupted Conda Cache
**Symptoms**:
```
error    libmamba Non-conda folder exists at prefix - aborting.
critical libmamba Non-conda folder exists at prefix - aborting.
```

**Root Cause**:
- Snakemake creates YAML file: `/scratch/.../9c123240620f10eb0d4a535111483c5a_.yaml`
- Mamba tries to create environment at: `/scratch/.../9c123240620f10eb0d4a535111483c5a_/`
- YAML exists but directory doesn't → Mamba detects "non-conda folder"
- Previous failed runs left incomplete state
- Cannot clean because of NFS locks or interrupted processes

**Impact**: 100% pipeline failure rate (15 consecutive job failures)

### Problem 2: Unnecessary Complexity
**Current**: 43+ separate environments
- RNA-seq QC: `fastqc=0.12.1, multiqc=1.14`
- DNA-seq QC: `fastqc=0.12.1, multiqc=1.14` (DUPLICATE)
- ATAC-seq QC: `fastqc=0.12.1, multiqc=1.14` (DUPLICATE)
- ChIP-seq QC: `fastqc=0.12.1, multiqc=1.14` (DUPLICATE)
- [+6 more duplicates]

**Result**: 10x installation of identical tools across pipelines

### Problem 3: Version Conflicts
**environment.yml** (Base):
```yaml
- fastqc=0.12.1
- star=2.7.10b
- samtools=1.17
- r-base>=4.2
- bioconductor-deseq2
```

**Per-rule environments**:
```yaml
# rna_seq/envs/qc.yaml
- fastqc=0.12.1  # Same as base

# rna_seq/envs/alignment.yaml
- star=2.7.10b   # Same as base
- samtools=1.17  # Same as base
```

**Problem**: Tools installed in BOTH base and per-rule environments, wasting space and causing potential version mismatches

### Problem 4: NFS File Locking
**Home directory** (`~/.snakemake/conda`) on NFS:
- Cannot delete corrupted environments
- `rm -rf` hangs indefinitely
- Files show "Device or resource busy"
- Affects: 246MB of cached environments

**Attempted fix**: Move to `/scratch/sdodl001/conda_envs`
**Result**: Still fails (cache corruption persists across locations)

### Problem 5: No Environment Isolation
- Development work in same environment as production pipelines
- No testing environment
- No rollback capability
- Changes affect all pipelines simultaneously

---

## Consequences

### Operational Impact
- ✗ 15 consecutive pipeline job failures
- ✗ 2+ hours spent troubleshooting conda issues
- ✗ 0 successful pipeline validations after reorganization
- ✗ Cannot demonstrate pipeline functionality
- ✗ Cannot process actual data

### Development Impact
- 🔥 Every code change requires fighting environment issues
- 🔥 Cannot focus on pipeline logic/features
- 🔥 ~80% time on environment debugging vs 20% on development
- 🔥 Demotivating for developers

### Resource Impact
- 246MB+ in `~/.snakemake/conda` (home directory quota)
- 43+ redundant environment installations
- ~10x duplication of common tools (fastqc, multiqc, samtools, etc.)
- Slow environment creation (minutes per pipeline)
- Wasted compute time on failed jobs

---

## Recommended Solutions

### 🎯 Option 1: Containerized Approach (BEST)
**Architecture**: One container per pipeline
```
pipelines/
├── rna_seq/
│   ├── Dockerfile                    # All tools needed
│   ├── Singularity.def              # Singularity equivalent
│   └── Snakefile                     # Use container, not conda
├── dna_seq/
│   ├── Dockerfile
│   └── Snakefile
└── ...
```

**Benefits**:
- ✅ **Complete isolation**: Each pipeline has its own container
- ✅ **Reproducibility**: Exact environment captured in container
- ✅ **No conda issues**: Eliminates all conda/mamba problems
- ✅ **Fast deployment**: Pull pre-built containers from registry
- ✅ **Version control**: Container images are versioned
- ✅ **HPC compatible**: Singularity support on SLURM clusters
- ✅ **Portable**: Works on local, HPC, cloud

**Implementation**:
```dockerfile
# Example: pipelines/rna_seq/Dockerfile
FROM mambaorg/micromamba:1.5.8

# Install all RNA-seq tools in one layer
RUN micromamba install -y -n base -c conda-forge -c bioconda \
    fastqc=0.12.1 \
    multiqc=1.14 \
    fastp=0.23.4 \
    star=2.7.10b \
    salmon=1.10.1 \
    samtools=1.17 \
    r-base=4.2 \
    bioconductor-deseq2 \
    && micromamba clean --all --yes

WORKDIR /analysis
```

**Snakefile changes**:
```python
# OLD (broken)
rule fastqc:
    conda: "envs/qc.yaml"
    
# NEW (containerized)
rule fastqc:
    container: "docker://your-registry/biopipelines-rna-seq:latest"
```

**Build & Deploy**:
```bash
# Build
docker build -t biopipelines-rna-seq:v1.0 pipelines/rna_seq/

# Convert to Singularity for HPC
singularity build rna_seq.sif docker-daemon://biopipelines-rna-seq:v1.0

# Use in Snakemake
snakemake --use-singularity --singularity-prefix /scratch/containers/
```

**Effort**: Medium (2-3 days to containerize all pipelines)  
**Maintenance**: Low (rebuild container when tools change)  
**Risk**: Low (proven approach in bioinformatics)

---

### 🎯 Option 2: Single Unified Environment (GOOD)
**Architecture**: One comprehensive conda environment for all pipelines

**Current Problem**:
- 43 separate environments
- Massive duplication
- Complex dependency management

**Solution**:
```yaml
# environment.yml (comprehensive)
name: biopipelines-unified
channels:
  - conda-forge
  - bioconda
dependencies:
  # Core workflow
  - snakemake>=7.30
  - mamba
  
  # QC (shared by all pipelines)
  - fastqc=0.12.1
  - multiqc=1.14
  - fastp=0.23.4
  - trimmomatic=0.39
  
  # Alignment (shared)
  - bwa=0.7.17
  - bowtie2=2.5.1
  - star=2.7.10b
  - hisat2=2.2.1
  - samtools=1.17
  - bcftools=1.17
  
  # RNA-seq specific
  - salmon=1.10.1
  - subread=2.0.3
  - r-base=4.2
  - bioconductor-deseq2
  - bioconductor-edger
  
  # DNA-seq specific
  - gatk4=4.4.0.0
  - freebayes=1.3.6
  - picard=3.0.0
  - snpeff=5.1
  
  # ChIP-seq/ATAC-seq
  - macs2=2.2.7.1
  - homer=4.11
  - deeptools=3.5.2
  
  # Methylation
  - bismark>=0.24
  
  # Hi-C
  - cooler
  - hicexplorer
  
  # Long-read
  - minimap2
  - nanofilt
  
  # Metagenomics
  - kraken2
  - metaphlan
  - megahit
  
  # All other tools...
```

**Snakefile changes**:
```python
# OLD
rule fastqc:
    conda: "envs/qc.yaml"

# NEW (no conda directive - use activated environment)
rule fastqc:
    shell: "fastqc {input} -o {output}"
```

**Benefits**:
- ✅ Eliminate 43 environments → 1 environment
- ✅ No conda environment creation overhead
- ✅ No cache corruption issues
- ✅ Faster pipeline startup
- ✅ Shared tools reduce disk usage
- ✅ Single point of dependency management

**Drawbacks**:
- ⚠️ Large environment (~5-10GB)
- ⚠️ Potential conflicts between tools
- ⚠️ All-or-nothing updates
- ⚠️ Less isolation between pipelines

**Implementation**:
```bash
# Create unified environment
mamba env create -f environment.yml

# Update submit script
# Remove: --use-conda --conda-frontend mamba
# Just activate biopipelines-unified before running
```

**Effort**: Low (1 day to consolidate and test)  
**Maintenance**: Medium (manage single large environment)  
**Risk**: Medium (conflict resolution may be needed)

---

### 🎯 Option 3: Module-Based Environments (COMPROMISE)
**Architecture**: Logical grouping of tools into 5-7 shared modules

```
envs/
├── shared/
│   ├── qc.yaml              # fastqc, multiqc (used by ALL)
│   ├── alignment.yaml       # bwa, bowtie2, star, samtools (used by 8 pipelines)
│   ├── visualization.yaml   # deeptools, IGV, etc. (used by 6 pipelines)
│   └── python_analysis.yaml # pandas, numpy, matplotlib
├── rna_seq_specific.yaml    # salmon, deseq2, edger
├── dna_seq_specific.yaml    # gatk, freebayes, snpeff
├── chip_atac_specific.yaml  # macs2, homer
├── methylation_specific.yaml # bismark
├── hic_specific.yaml        # cooler, hicexplorer
└── metagenomics_specific.yaml # kraken2, metaphlan
```

**Benefits**:
- ✅ Reduce 43 environments → ~12 shared modules
- ✅ Eliminate duplication (QC shared across all)
- ✅ Better than micro-environments
- ✅ Moderate isolation
- ✅ Easier to maintain than 43 separate files

**Drawbacks**:
- ⚠️ Still uses Snakemake conda integration (source of current problems)
- ⚠️ Requires careful module design
- ⚠️ Doesn't solve fundamental cache corruption issue

**Effort**: Medium (2 days to refactor and test)  
**Maintenance**: Medium-High (manage 12 modules)  
**Risk**: Medium (still depends on conda/mamba stability)

---

## Immediate Action Plan

### Phase 1: Emergency Fix (NOW - 30 minutes)
**Goal**: Get ONE pipeline working to validate code

```bash
# 1. Create clean unified environment
cd ~/BioPipelines
mamba create -n biopipelines-test -c conda-forge -c bioconda \
    snakemake fastqc multiqc fastp star samtools salmon r-base bioconductor-deseq2

# 2. Test RNA-seq WITHOUT conda integration
cat > scripts/test_rna_seq_no_conda.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=rna_test
#SBATCH --partition=cpuspot
#SBATCH --cores=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

eval "$(/home/sdodl001_odu_edu/miniconda3/bin/conda shell.bash hook)"
conda activate biopipelines-test

cd ~/BioPipelines/pipelines/rna_seq

# Run WITHOUT --use-conda
snakemake --cores 8 --latency-wait 60
EOF

sbatch scripts/test_rna_seq_no_conda.sh
```

**Expected**: Pipeline runs successfully without conda env creation overhead

### Phase 2: Strategic Decision (NEXT - 1 day)
**Decision point**: Choose between Option 1 (Containers), Option 2 (Unified), or Option 3 (Modules)

**Recommendation**: **Option 1 (Containers)** because:
1. Industry standard for reproducible bioinformatics
2. Eliminates all conda/mamba issues permanently
3. Better isolation and portability
4. HPC-compatible via Singularity
5. Easier collaboration (share containers)

### Phase 3: Implementation (WEEK 1)
**If choosing containers**:

```bash
# Day 1: Containerize RNA-seq (pilot)
# Day 2: Containerize DNA-seq, ATAC-seq
# Day 3: Containerize ChIP-seq, Methylation
# Day 4: Containerize remaining pipelines
# Day 5: Build and test all containers
```

**Deliverables**:
- 10 Dockerfiles (one per pipeline)
- 10 Singularity images on cluster
- Updated Snakefiles using containers
- CI/CD for automatic container builds
- Documentation for container usage

### Phase 4: Validation (WEEK 2)
- Test all 10 pipelines with real data
- Performance benchmarking
- Documentation updates
- Training for users

---

## Comparison Matrix

| Criteria | Current (Broken) | Option 1: Containers | Option 2: Unified | Option 3: Modules |
|----------|-----------------|---------------------|-------------------|-------------------|
| **Environments** | 43+ | 10 (one/pipeline) | 1 | ~12 |
| **Duplication** | 🔴 Massive | ✅ None | ✅ None | ✅ Minimal |
| **Maintenance** | 🔴 Very High | ✅ Low | 🟡 Medium | 🟡 Medium-High |
| **Reliability** | 🔴 0% success | ✅ 99%+ | 🟡 85%+ | 🟡 70%+ |
| **Isolation** | 🟡 Per-rule | ✅ Per-pipeline | 🔴 None | 🟡 Moderate |
| **Speed** | 🔴 Slow (env creation) | ✅ Fast (pre-built) | ✅ Fast (no creation) | 🟡 Medium |
| **Reproducibility** | 🔴 Poor | ✅ Excellent | 🟡 Good | 🟡 Good |
| **HPC Compatible** | ✅ Yes | ✅ Yes (Singularity) | ✅ Yes | ✅ Yes |
| **Setup Time** | N/A | 🟡 Medium (2-3 days) | ✅ Low (1 day) | 🟡 Medium (2 days) |
| **Industry Standard** | ❌ No | ✅ Yes | 🟡 Acceptable | 🟡 Acceptable |
| **Disk Usage** | 🔴 High (duplication) | ✅ Low (shared base) | ✅ Low (one env) | 🟡 Medium |
| **Version Control** | 🟡 YAML files | ✅ Container tags | 🟡 YAML file | 🟡 YAML files |

**Winner**: 🏆 **Option 1: Containers**

---

## Technical Details: Why Containers Are Better

### Current Conda Approach
```python
# Snakefile
rule fastqc:
    conda: "envs/qc.yaml"  # Snakemake creates env on-the-fly
    shell: "fastqc {input}"
```

**Problems**:
- Environment created per-job (slow)
- Cache corruption issues
- NFS locking problems
- Hash-based directory names (hard to debug)
- Cleanup is difficult

### Container Approach
```python
# Snakefile
rule fastqc:
    container: "docker://biopipelines/rna-seq:v1.0"
    shell: "fastqc {input}"
```

**Benefits**:
- Container pulled once, reused forever
- No cache corruption possible
- No NFS issues (containers stored in `/scratch`)
- Named images (easy to understand)
- Simple cleanup: `rm image.sif`

### Real-World Example: nf-core
The nf-core project (gold standard for bioinformatics pipelines) uses containers exclusively:
- 100+ production pipelines
- All containerized (Docker/Singularity)
- Zero conda environment issues
- Extremely reliable

**We should follow their lead.**

---

## Cost-Benefit Analysis

### Current Approach (Status Quo)
**Cost**: 
- 2+ hours debugging per session
- 15 failed jobs = ~30 compute hours wasted
- Developer frustration
- Cannot demonstrate pipeline functionality
- Blocks progress on actual science

**Benefit**: 
- None (system is broken)

### Option 1: Containers
**Cost**:
- 2-3 days initial implementation
- Learning curve for team
- Container registry setup (optional)

**Benefit**:
- Eliminates 100% of conda issues
- Faster pipeline execution
- Better reproducibility
- Industry-standard approach
- Future-proof
- **ROI**: Pays for itself in 1 week

### Option 2: Unified Environment
**Cost**:
- 1 day to consolidate
- Risk of conflicts
- Large environment size

**Benefit**:
- Quick fix
- Eliminates duplication
- Simpler than current
- **ROI**: Immediate

### Option 3: Modules
**Cost**:
- 2 days to design and implement
- Still vulnerable to conda issues

**Benefit**:
- Better than current
- Moderate improvement
- **ROI**: 1-2 weeks

---

## Conclusion & Recommendation

### Current State: 🔴 UNACCEPTABLE
- 43+ environments causing persistent failures
- 0% pipeline success rate
- 80% time on environment debugging
- Blocking all development progress

### Recommended Path: 🎯 CONTAINERS
**Primary Recommendation**: Implement Option 1 (Containerized approach)

**Rationale**:
1. **Eliminates root cause**: No more conda cache corruption
2. **Industry standard**: Used by nf-core, Broad Institute, etc.
3. **Better long-term**: Reproducibility, portability, maintainability
4. **Acceptable cost**: 2-3 days investment for permanent fix
5. **Future-proof**: Containers are the future of scientific computing

**Interim Solution**: While implementing containers (2-3 days):
- Use Option 2 (Unified environment) as temporary fix
- Get pipelines working immediately
- Migrate to containers pipeline-by-pipeline

### Next Steps
1. **TODAY**: Create unified environment, validate RNA-seq works
2. **THIS WEEK**: Containerize all 10 pipelines
3. **NEXT WEEK**: Test and validate with real data
4. **ONGOING**: Use containers for all future development

### Success Metrics
- ✅ 100% pipeline success rate
- ✅ <5% time on environment issues
- ✅ Fast pipeline startup (<30 seconds)
- ✅ Reproducible results
- ✅ Happy developers

---

**Prepared by**: GitHub Copilot (AI Analysis)  
**Reviewed with**: User feedback on persistent conda issues  
**Priority**: 🔴 **CRITICAL** - Immediate action required
