# BioPipelines Architecture Review & Reorganization Plan

**Date**: November 23, 2025  
**Status**: ✅ ALL PHASES COMPLETE - Production Ready

**Progress**:
- ✅ **Phase 1 Complete**: Root directory cleaned (400+ → 21 items, 95% reduction)
- ✅ **Phase 2 Complete**: Scripts consolidated (43 → 2 core scripts, 95% reduction)
- ✅ **Phase 3 Complete**: Pipeline standardization (10 pipelines flattened to consistent 2-level structure)
- ✅ **Phase 4 Complete**: Module integration (reusable Snakemake rule infrastructure created)
- ✅ **Phase 5 Complete**: Directory consolidation (professional config/ management system)

**Project Transformation**: From cluttered, inconsistent codebase → Clean, professional, production-ready infrastructure

See: `docs/status/CLEANUP_COMPLETED.md`, `docs/status/PHASE2_COMPLETE.md`, `docs/status/PHASE3_COMPLETE.md`, and `docs/status/PHASES_4_5_COMPLETE.md`

## Executive Summary

BioPipelines has grown organically with **336 SLURM log files**, **25 download scripts**, **18 submit scripts**, and scattered orphaned files. While functionally working (8/10 pipelines complete), the codebase needs significant cleanup and reorganization to improve maintainability, reduce duplication, and establish clear architectural patterns.

---

## Current Architecture Assessment

### ✅ Strengths

1. **Clear Pipeline Separation**: Each pipeline in `pipelines/` has its own subdirectory
2. **Consistent Snakemake Structure**: All pipelines use Snakefile + config.yaml pattern
3. **Modular Python Package**: `src/biopipelines/` provides reusable utilities
4. **Comprehensive Documentation**: Multiple tutorials and troubleshooting guides
5. **Working Pipelines**: 8/10 pipelines successfully validated

### ❌ Critical Issues

#### 1. **Root Directory Pollution (HIGH PRIORITY)**
- **336 SLURM log files** (*.err, *.out) cluttering root directory
- **12+ markdown status files** (overlapping documentation)
- **6 orphaned FASTQ files** in root (should be in data/raw/)
- **5 log files** from various operations

**Impact**: Makes navigation difficult, git status messy, increases confusion

#### 2. **Script Redundancy (HIGH PRIORITY)**
```
ChIP-seq Downloads (5 variants):
├── download_chipseq_control.py
├── download_chipseq_encode_direct.sh
├── download_chipseq_encode.py
├── download_chipseq_no_ssl.py
└── download_chipseq_proper.py

Test Data Downloads (9 variants):
├── download_hic_test.py
├── download_metagenomics_test.py
├── download_methylation_test.py
├── download_public_test_data.sh
├── download_rrbs_test.py
├── download_scrna_test_data.py
├── download_test_datasets.py
├── download_test_data.sh
└── download_validated_test_data.sh

Submit Scripts (4 "_simple" variants):
├── submit_chip_seq.sh vs submit_chip_seq_simple.sh
├── submit_dna_seq.sh vs submit_dna_seq_simple.sh
├── submit_hic.sh vs submit_hic_simple.sh
└── submit_methylation.sh vs submit_methylation_simple.sh
```

**Impact**: 
- Confusion about which script to use
- Duplicated maintenance effort
- Version drift between variants
- No single source of truth

#### 3. **Inconsistent Directory Nesting**
```
pipelines/
├── atac_seq/accessibility_analysis/     # 3 levels
├── chip_seq/peak_calling/               # 3 levels
├── rna_seq/differential_expression/     # 3 levels
└── scrna_seq/                           # 2 levels (inconsistent)
    ├── Snakefile                        # Top-level Snakefile
    └── single_cell_analysis/Snakefile   # Nested Snakefile (duplicate?)
```

**Impact**: Inconsistent paths, confusion about which Snakefile to run

#### 4. **Underutilized src/ Module**
- `src/biopipelines/` has ~3,500 lines of Python code
- **Zero imports** from src/ in any Snakefile
- Modules exist but aren't integrated with pipelines

**Impact**: Duplicated logic, missed reuse opportunities

#### 5. **Empty/Placeholder Directories**
- `config/` - Empty
- `notebooks/` - Has subdirs but unclear usage
- `tests/` - Has structure but no integration
- `benchmarks/` - Unclear purpose

#### 6. **Documentation Fragmentation**
12 markdown files in root:
- `COMPREHENSIVE_AUDIT_RESULTS.md`
- `DEVELOPMENT_STATUS.md`
- `PIPELINE_STATUS.md`
- `PRIORITY_STATUS.md`
- `PIPELINE_ROADMAP.md`
- `NEXT_STEPS.md`
- `TODO.md`
- `GCP_ARCHITECTURE_CORRECTED.md`
- `GCP_QUICK_REFERENCE.md`
- `GIT_SETUP.md`
- `PULL_ON_CLUSTER.md`
- `README.md`

**Impact**: Overlapping information, hard to find current status

---

## Proposed Reorganization

### Phase 1: Immediate Cleanup (HIGH PRIORITY)

#### A. Move SLURM Logs (Critical)
```bash
# Create archive directory
mkdir -p logs/slurm/archive/

# Move all old logs
mv *_pipeline_*.{err,out} logs/slurm/archive/ 2>/dev/null
mv slurm_*.{err,out} logs/slurm/archive/ 2>/dev/null
mv *_download_*.{err,out} logs/slurm/archive/ 2>/dev/null
mv *.{err,out} logs/slurm/archive/ 2>/dev/null

# Future: Update submit scripts to write to logs/slurm/
```

**Result**: Clean root directory, organized logs

#### B. Move Orphaned Data Files
```bash
# Move FASTQ files
mv *.fastq.gz data/raw/archive/ 2>/dev/null

# Move log files
mv *.log logs/archive/ 2>/dev/null

# Remove downloaded files that should be in data/
# (or move to appropriate data/raw subdirectories)
```

#### C. Consolidate Documentation
```bash
# Create docs/status/ directory
mkdir -p docs/status/

# Move status documents
mv DEVELOPMENT_STATUS.md docs/status/
mv PIPELINE_STATUS.md docs/status/
mv PRIORITY_STATUS.md docs/status/
mv PIPELINE_ROADMAP.md docs/status/
mv COMPREHENSIVE_AUDIT_RESULTS.md docs/status/

# Move technical docs
mkdir -p docs/infrastructure/
mv GCP_ARCHITECTURE_CORRECTED.md docs/infrastructure/
mv GCP_QUICK_REFERENCE.md docs/infrastructure/
mv GIT_SETUP.md docs/infrastructure/
mv PULL_ON_CLUSTER.md docs/infrastructure/

# Consolidate TODOs
cat TODO.md NEXT_STEPS.md > docs/TODO_CONSOLIDATED.md
rm TODO.md NEXT_STEPS.md
```

**Keep in root**: README.md, LICENSE, environment.yml, pyproject.toml

### Phase 2: Script Consolidation (HIGH PRIORITY)

#### A. Unified Download System
Create `scripts/download_data.py` with subcommands:
```python
# Single entry point for all downloads
python scripts/download_data.py chipseq --accession ENCSR... --output data/raw/chip_seq/
python scripts/download_data.py rnaseq --sra SRR... --output data/raw/rna_seq/
python scripts/download_data.py test --pipeline chipseq --size small
```

**Replace**:
- 5 ChIP-seq download variants → 1 unified script with options
- 9 test download variants → 1 script with --test flag
- Separate scripts for each data type → subcommands

**Delete** after migration:
- `download_chipseq_*.py` (all 4 variants)
- `download_*_test.py` (all test variants)
- `download_test_*.sh` (all test variants)

#### B. Unified Submit System
Create `scripts/submit_pipeline.sh`:
```bash
# Single submit script with parameters
./scripts/submit_pipeline.sh --pipeline chipseq --config simple --mem 32G --cores 8
./scripts/submit_pipeline.sh --pipeline methylation --config full --mem 48G --cores 16
```

**Delete** "_simple" variants:
- `submit_chip_seq_simple.sh`
- `submit_dna_seq_simple.sh`
- `submit_hic_simple.sh`
- `submit_methylation_simple.sh`

Use config parameter: `--config simple` vs `--config full`

#### C. Download Script Cleanup
Keep these core scripts:
```
scripts/
├── download_data.py              # NEW: Unified download CLI
├── download_references.sh        # Reference genomes/indexes
├── download_annotations.sh       # GTF/GFF files
└── download_snpeff_db.sh        # SnpEff databases
```

Delete redundant:
- `download_datasets.py` (merge into download_data.py)
- `download_public_test_data.sh` (use --test flag)
- `download_validated_test_data.sh` (use --test flag)
- All pipeline-specific test downloaders

### Phase 3: Pipeline Standardization (MEDIUM PRIORITY)

#### A. Flatten Pipeline Structure
Current (inconsistent):
```
pipelines/atac_seq/accessibility_analysis/Snakefile
pipelines/scrna_seq/Snakefile  # Different!
```

Proposed (consistent):
```
pipelines/
├── atac_seq/
│   ├── Snakefile
│   ├── config.yaml
│   └── envs/
├── chip_seq/
│   ├── Snakefile
│   ├── config.yaml
│   └── envs/
├── dna_seq/
│   ├── Snakefile
│   ├── config.yaml
│   └── envs/
└── [...]
```

**Benefits**:
- Consistent paths: `pipelines/atac_seq/Snakefile` for all
- Simpler navigation
- Clear 2-level structure

#### B. Remove Redundant Nesting
- Move `pipelines/atac_seq/accessibility_analysis/` → `pipelines/atac_seq/`
- Move `pipelines/chip_seq/peak_calling/` → `pipelines/chip_seq/`
- Move `pipelines/rna_seq/differential_expression/` → `pipelines/rna_seq/`
- Consolidate `pipelines/scrna_seq/` (remove dual Snakefiles)

**scRNA-seq special case**: Keep modular Python scripts in `scripts/` subdirectory

### Phase 4: Module Integration (MEDIUM PRIORITY)

#### A. Connect src/ to Pipelines
Add to all Snakefiles:
```python
import sys
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from biopipelines.preprocessing import FastqValidator
from biopipelines.core import SampleSheet
```

#### B. Refactor Common Patterns
Identify repeated Snakefile patterns:
- FastQC/MultiQC rules → `biopipelines.core.qc_rules`
- Trimming rules → `biopipelines.preprocessing.trim_rules`
- Alignment rules → `biopipelines.alignment.align_rules`

### Phase 5: Directory Consolidation (LOW PRIORITY)

#### A. Populate Empty Directories
```
config/
├── slurm.yaml           # SLURM defaults (mem, time, partition)
├── environments.yaml    # Conda env specs
└── pipelines/           # Pipeline-specific configs (if needed)
```

#### B. Remove Unused Directories
- `cache/` - unclear purpose, consider removing if unused
- `SURVIVOR/` - tool should be in `tools/` or installed via conda

#### C. Tools Management
Current: `tools/manta-1.6.0.centos6_x86_64/`

Better: Install via conda/bioconda when possible
- Keep `tools/` for truly custom/compiled tools only
- Document in README which tools need manual installation

---

## Proposed Final Structure

```
BioPipelines/
├── README.md                    # Main documentation
├── LICENSE
├── pyproject.toml
├── environment.yml
│
├── pipelines/                   # All analysis pipelines (2-level)
│   ├── atac_seq/
│   │   ├── Snakefile
│   │   ├── config.yaml
│   │   ├── envs/               # Conda environments
│   │   └── scripts/            # Pipeline-specific scripts
│   ├── chip_seq/
│   ├── dna_seq/
│   ├── hic/
│   ├── long_read/
│   ├── metagenomics/
│   ├── methylation/
│   ├── rna_seq/
│   ├── scrna_seq/
│   └── structural_variants/
│
├── src/                        # Python package (pip install -e .)
│   └── biopipelines/
│       ├── __init__.py
│       ├── alignment/
│       ├── core/               # Sample sheets, validation, common functions
│       ├── data_download/      # Unified download CLI
│       ├── expression/
│       ├── peak_calling/
│       ├── preprocessing/
│       ├── variant_calling/
│       └── visualization/
│
├── scripts/                    # Standalone utility scripts
│   ├── download_data.py        # Unified download CLI ⭐ NEW
│   ├── submit_pipeline.sh      # Unified submit script ⭐ NEW
│   ├── download_references.sh
│   ├── download_annotations.sh
│   ├── build_star_index.sh
│   ├── build_bwa_index.sh
│   └── setup_*.sh
│
├── config/                     # Global configuration
│   ├── slurm.yaml
│   └── environments.yaml
│
├── data/                       # Data directory (gitignored)
│   ├── raw/
│   │   ├── atac_seq/
│   │   ├── chip_seq/
│   │   ├── [...]/
│   │   └── archive/            # Old test files
│   ├── processed/
│   ├── references/
│   └── results/
│
├── logs/                       # All log files
│   ├── slurm/
│   │   ├── archive/            # ⭐ Move all 336 .err/.out files here
│   │   └── active/             # Current job logs
│   ├── downloads/
│   └── builds/
│
├── docs/                       # Documentation
│   ├── status/                 # ⭐ Consolidated status docs
│   ├── infrastructure/         # ⭐ GCP, Git, cluster setup
│   ├── pipelines/              # Per-pipeline documentation
│   ├── tutorials/              # User guides
│   └── api/                    # API documentation
│
├── notebooks/                  # Jupyter notebooks
│   ├── exploratory/
│   ├── quality_control/
│   └── visualization/
│
├── tests/                      # Test suite
│   ├── unit/
│   ├── integration/
│   └── test_data/
│
├── tools/                      # Custom/compiled tools only
│   └── README.md               # Document what goes here
│
└── benchmarks/                 # Performance benchmarks
    └── results/
```

---

## Implementation Priority

### 🔴 High Priority (Do First)
1. **Move SLURM logs** → `logs/slurm/archive/` (immediate)
2. **Move orphaned files** → appropriate data/ subdirectories
3. **Consolidate documentation** → `docs/status/` and `docs/infrastructure/`
4. **Create unified download script** → `scripts/download_data.py`
5. **Delete redundant download scripts** (after migration)

### 🟡 Medium Priority (Do Soon)
6. **Create unified submit script** → `scripts/submit_pipeline.sh`
7. **Delete "_simple" submit variants**
8. **Flatten pipeline directories** (remove intermediate nesting)
9. **Fix scRNA-seq dual Snakefile issue**

### 🟢 Low Priority (Future)
10. **Integrate src/ modules** into Snakefiles
11. **Populate config/ directory**
12. **Clean up tools/ directory**
13. **Add CI/CD testing**

---

## Metrics

### Current State
- **Root files**: 400+ files (logs, data, docs)
- **Download scripts**: 25 scripts (5 ChIP-seq variants, 9 test variants)
- **Submit scripts**: 18 scripts (4 with "_simple" duplicates)
- **Pipeline structure**: Inconsistent (2-3 levels)
- **src/ integration**: 0% (no imports from src/)

### Target State
- **Root files**: ~10 files (README, LICENSE, configs)
- **Download scripts**: 4 scripts (unified + core utilities)
- **Submit scripts**: 9 scripts (1 per pipeline, no variants)
- **Pipeline structure**: Consistent 2-level
- **src/ integration**: 50%+ (shared rules, utilities)

---

## Backward Compatibility

### Breaking Changes
1. Pipeline paths change: `pipelines/atac_seq/accessibility_analysis/` → `pipelines/atac_seq/`
2. Download script names change
3. Submit script interface changes

### Migration Strategy
1. Create `MIGRATION.md` documenting all path changes
2. Add symlinks for 1 release cycle to old paths
3. Update all documentation
4. Add deprecation warnings to old scripts

---

## Next Steps

1. **Review this document** with team
2. **Get approval** for cleanup plan
3. **Create backup branch** before major changes
4. **Execute Phase 1** (cleanup) immediately
5. **Execute Phase 2** (consolidation) in next sprint
6. **Document changes** in CHANGELOG.md

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking active jobs | High | Test in separate branch, announce maintenance window |
| Lost data files | High | Create comprehensive backup, use git mv for tracking |
| Script confusion | Medium | Keep symlinks, update docs, add migration guide |
| Development disruption | Medium | Coordinate with team, do in phases |

---

## Benefits

After reorganization:
- ✅ **Clean repository**: Professional appearance, easy navigation
- ✅ **Reduced maintenance**: Single source of truth for scripts
- ✅ **Better onboarding**: Consistent structure, clear documentation
- ✅ **Improved reliability**: Less duplication = fewer version conflicts
- ✅ **Scalability**: Clear patterns for adding new pipelines
- ✅ **Professionalism**: Production-ready codebase structure

---

**Recommendation**: Proceed with Phase 1 (cleanup) immediately, as it has no breaking changes and immediate benefits.
