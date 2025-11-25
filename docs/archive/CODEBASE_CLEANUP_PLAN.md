# BioPipelines Codebase Cleanup Plan

**Date:** November 25, 2025  
**Purpose:** Clean and lean codebase before LLM integration

---

## Executive Summary

The codebase has evolved significantly and accumulated redundancy. This document identifies files to **DELETE**, **MERGE**, or **ARCHIVE** to create a clean foundation for LLM integration.

### Current State
- **Total Size:** ~2.7GB (mostly .snakemake conda artifacts)
- **Python Code:** ~11,000 lines across 2 packages
- **Documentation:** 40+ markdown files (many redundant)
- **Scripts:** 25+ bash scripts (many deprecated)
- **Modules:** 71 Nextflow modules (unified) + 62 legacy duplicates

---

## 🔴 CRITICAL: Files/Directories to DELETE

### 1. Duplicate Module Directory (PRIORITY 1)
```
nextflow-modules/          # 62 modules - DUPLICATES of nextflow-pipelines/modules/
```
**Reason:** Already merged into `nextflow-pipelines/modules/` with nf-core style structure.  
**Action:** Delete entire directory after confirming merge.

### 2. Nextflow Work Directories (PRIORITY 1)
```
nextflow-pipelines/work_*   # 24 test work directories
```
**Reason:** Temporary test artifacts, taking up space.  
**Action:** Delete all `work_*` directories.

### 3. Root-level Cleanup Files (PRIORITY 2)
```
merge_modules.sh            # One-time migration script (already executed)
build_conda_test.sh         # Obsolete test script
test_compute_node.sh        # Move to scripts/testing/
wget-log                    # Download log artifact
```

### 4. Snakemake Cache (PRIORITY 3)
```
.snakemake/                 # 2.2GB - conda environment cache
```
**Reason:** Can be rebuilt, but contains installed envs.  
**Action:** Consider partial cleanup of unused envs.

### 5. Cache Directory
```
cache/                      # 8.2MB - orphaned h5ad cache file
```
**Action:** Delete or configure proper cache location.

---

## 🟡 Documentation to CONSOLIDATE

### Root-Level Docs (Keep 2, Archive Rest)

| File | Action | Reason |
|------|--------|--------|
| `README.md` | **KEEP** | Main project readme |
| `LICENSE` | **KEEP** | Legal requirement |
| `ARCHITECTURE_REVIEW.md` | ARCHIVE | Historical, info in docs/ |
| `CONTAINER_IMPLEMENTATION_SUMMARY.md` | ARCHIVE | Superseded by docs/CONTAINER_ARCHITECTURE.md |
| `PIPELINE_STATUS_FINAL.md` | ARCHIVE | Historical status |
| `PREFLIGHT_SUMMARY.txt` | DELETE | One-time check output |
| `REORGANIZATION_SUMMARY.md` | DELETE | Historical, no longer relevant |
| `SESSION_SUMMARY.md` | DELETE | Session notes |
| `codebase_assessment.md` | ARCHIVE | Superseded by this document |

### docs/ Directory Consolidation

**KEEP (Essential):**
```
docs/
├── WORKFLOW_COMPOSER_GUIDE.md     # User guide
├── API_REFERENCE.md               # API docs
├── TUTORIALS.md                   # Quick tutorials
├── COMPOSITION_PATTERNS.md        # Workflow patterns
├── CONTAINER_ARCHITECTURE.md      # Container docs
├── QUICK_START_CONTAINERS.md      # Getting started
├── GCP_HPC_SETUP.md               # Infrastructure
├── tutorials/                     # Pipeline tutorials (10 files)
└── infrastructure/                # Setup guides
```

**ARCHIVE (Historical):**
```
docs/archive/
├── ARCHITECTURE_PLAN_REVIEW.md
├── AI_WORKFLOW_COMPOSER_ARCHITECTURE.md  # Now in WORKFLOW_COMPOSER_GUIDE
├── CELLRANGER_INSTALLATION.md
├── CONTAINER_STRATEGY_PIVOT.md
├── CRITICAL_EVALUATION.md
├── DYNAMIC_CONTAINER_STRATEGY.md
├── DYNAMIC_PIPELINE_REQUIREMENTS.md
├── ENVIRONMENT_ARCHITECTURE_ANALYSIS.md
├── IMPLEMENTATION_GAP_ANALYSIS.md
├── MODULE_LIBRARY_SUMMARY.md
├── NEXTFLOW_ARCHITECTURE_PLAN.md
├── NEXTFLOW_IMPLEMENTATION_COMPLETE.md
├── PROGRESS_REPORT_20251125.md
├── PROGRESS_SESSION_20241125.md
├── TIER2_CONTAINER_DESIGN.md
├── TODO_CONSOLIDATED.md           # Outdated todos
└── status/                        # All status files (11 files)
```

**DELETE:**
```
docs/api/                   # Empty directory
docs/pipelines/             # Empty directory
```

---

## 🟢 Scripts to REORGANIZE

### Current Structure (Messy)
```
scripts/
├── Various .sh and .py files (25+)
├── containers/             # Container build scripts
└── deprecated/             # Already archived
```

### Proposed Clean Structure
```
scripts/
├── README.md
├── run_all_pipelines.sh           # Main entry point
├── submit_pipeline_with_container.sh
├── containers/
│   ├── build_all_containers.sh    # KEEP (delete build_all.sh duplicate)
│   ├── build_*_container.slurm    # 12 individual builds
│   └── check_build_status.sh
├── data/
│   ├── download_test_data.sh      # Consolidated download script
│   └── gcp_stage_data.sh
├── indexes/
│   ├── build_star_index.sh
│   ├── build_bwa_index.sh
│   └── build_bowtie2_index_hg38.sh
├── testing/
│   ├── test_compute_node.sh       # Move from root
│   ├── test_containers_direct.sh
│   └── preflight_check.sh
└── deprecated/                    # Already exists (keep as archive)
```

### Scripts to DELETE
```
scripts/submit_all_pipelines.sh         # Superseded by run_all_pipelines.sh
scripts/pre_build_conda_envs.sh         # No longer needed (containerized)
scripts/quick_start.sh                  # Unclear purpose
scripts/run_nextflow.sh                 # Redundant
scripts/containers/build_all.sh         # Duplicate of build_all_containers.sh
scripts/containers/rebuild_remaining.sh # One-time script
scripts/containers/test_rna_seq*.sh     # One-time tests
```

---

## 🔵 Python Packages Analysis

### Package 1: `src/biopipelines/` (4,217 lines)
**Purpose:** Snakemake-oriented utilities for data processing

| Module | Lines | Purpose | Status |
|--------|-------|---------|--------|
| `core/` | 548 | Logging, config, snakemake rules | Active |
| `data_download/` | 1,978 | SRA/ENCODE/HiC downloaders | Active |
| `alignment/` | 108 | Alignment utilities | Limited use |
| `expression/` | 256 | Expression analysis | Limited use |
| `peak_calling/` | 164 | Peak calling utilities | Limited use |
| `preprocessing/` | 111 | QC preprocessing | Limited use |
| `variant_calling/` | 231 | Variant calling | Limited use |
| `visualization/` | 338 | Plotting utilities | Active |
| `containers/` | 275 | Container registry | Active |

### Package 2: `src/workflow_composer/` (6,800 lines)
**Purpose:** AI-driven workflow composition (NEW - for LLM integration)

| Module | Lines | Purpose | Status |
|--------|-------|---------|--------|
| `core/` | 2,271 | Intent parsing, tool selection, workflow generation | Active |
| `llm/` | 1,402 | LLM adapters (Ollama, OpenAI, Anthropic, HuggingFace) | Active |
| `data/` | 514 | Data downloading | Overlaps with biopipelines |
| `viz/` | 474 | Workflow visualization | Active |
| `web/` | 711 | Flask web UI | Active |
| `monitor/` | 436 | Workflow monitoring | Active |
| Other | 992 | CLI, composer, config | Active |

### RECOMMENDATION: Merge Overlapping Functionality

**Option A: Keep Both Packages (Separate Concerns)**
- `biopipelines`: Snakemake-focused, data utilities
- `workflow_composer`: Nextflow-focused, AI composition

**Option B: Consolidate into Single Package** (Recommended)
- Merge `biopipelines.data_download` → `workflow_composer.data`
- Keep `biopipelines.visualization` → `workflow_composer.viz`
- Archive unused `biopipelines` modules

---

## 📁 Proposed Final Structure

```
BioPipelines/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements-composer.txt
├── environment.yml
│
├── config/
│   ├── defaults.yaml
│   ├── composer.yaml
│   ├── slurm.yaml
│   └── nextflow/
│
├── containers/                     # Container definitions
│   ├── base/
│   ├── rna-seq/
│   ├── ... (12 total)
│   └── images/                     # Built images
│
├── nextflow-pipelines/             # Nextflow infrastructure
│   ├── modules/                    # 71 unified modules
│   ├── workflows/                  # 10 workflow definitions
│   ├── config/
│   └── README.md
│
├── pipelines/                      # Snakemake pipelines
│   ├── atac_seq/
│   ├── chip_seq/
│   ├── ... (10 total)
│   └── results/
│
├── src/
│   └── workflow_composer/          # Main Python package
│       ├── core/
│       ├── llm/
│       ├── data/
│       ├── viz/
│       ├── web/
│       └── monitor/
│
├── scripts/
│   ├── containers/
│   ├── data/
│   ├── indexes/
│   ├── testing/
│   └── deprecated/
│
├── docs/
│   ├── (6-8 essential docs)
│   ├── tutorials/
│   ├── infrastructure/
│   └── archive/
│
├── examples/
│   └── generated/
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── data/
│   ├── raw/
│   ├── references/
│   ├── results/
│   └── tool_catalog/
│
└── logs/
    └── archive/
```

---

## 🚀 Cleanup Execution Script

```bash
#!/bin/bash
# cleanup_codebase.sh - Execute cleanup plan

cd /home/sdodl001_odu_edu/BioPipelines

# 1. Remove duplicate modules directory
echo "Removing duplicate nextflow-modules..."
rm -rf nextflow-modules/

# 2. Remove Nextflow work directories
echo "Removing Nextflow work directories..."
rm -rf nextflow-pipelines/work_*

# 3. Remove one-time scripts
echo "Removing one-time scripts..."
rm -f merge_modules.sh
rm -f build_conda_test.sh
rm -f wget-log

# 4. Clean cache
echo "Cleaning cache..."
rm -rf cache/

# 5. Archive root-level docs
echo "Archiving root-level docs..."
mkdir -p docs/archive
mv ARCHITECTURE_REVIEW.md docs/archive/
mv CONTAINER_IMPLEMENTATION_SUMMARY.md docs/archive/
mv PIPELINE_STATUS_FINAL.md docs/archive/
mv REORGANIZATION_SUMMARY.md docs/archive/
mv SESSION_SUMMARY.md docs/archive/
mv codebase_assessment.md docs/archive/
rm -f PREFLIGHT_SUMMARY.txt

# 6. Archive historical docs
mv docs/ARCHITECTURE_PLAN_REVIEW.md docs/archive/
mv docs/AI_WORKFLOW_COMPOSER_ARCHITECTURE.md docs/archive/
mv docs/CELLRANGER_INSTALLATION.md docs/archive/
mv docs/CONTAINER_STRATEGY_PIVOT.md docs/archive/
mv docs/CRITICAL_EVALUATION.md docs/archive/
mv docs/DYNAMIC_CONTAINER_STRATEGY.md docs/archive/
mv docs/DYNAMIC_PIPELINE_REQUIREMENTS.md docs/archive/
mv docs/ENVIRONMENT_ARCHITECTURE_ANALYSIS.md docs/archive/
mv docs/IMPLEMENTATION_GAP_ANALYSIS.md docs/archive/
mv docs/MODULE_LIBRARY_SUMMARY.md docs/archive/
mv docs/NEXTFLOW_ARCHITECTURE_PLAN.md docs/archive/
mv docs/NEXTFLOW_IMPLEMENTATION_COMPLETE.md docs/archive/
mv docs/PROGRESS_REPORT_20251125.md docs/archive/
mv docs/PROGRESS_SESSION_20241125.md docs/archive/
mv docs/TIER2_CONTAINER_DESIGN.md docs/archive/
mv docs/TODO_CONSOLIDATED.md docs/archive/
mv docs/status/* docs/archive/
rmdir docs/status

# 7. Remove empty directories
rm -rf docs/api docs/pipelines

# 8. Reorganize scripts
mkdir -p scripts/data scripts/indexes scripts/testing
mv scripts/gcp_stage_data.sh scripts/data/
mv scripts/build_*_index*.sh scripts/indexes/
mv test_compute_node.sh scripts/testing/
mv scripts/preflight_check.sh scripts/testing/
mv scripts/test_containers_direct.sh scripts/testing/

# 9. Remove redundant scripts
rm -f scripts/submit_all_pipelines.sh
rm -f scripts/pre_build_conda_envs.sh
rm -f scripts/containers/build_all.sh
rm -f scripts/containers/rebuild_remaining.sh

# 10. Remove old biopipelines package (optional - review first)
# rm -rf src/biopipelines/

echo "Cleanup complete!"
```

---

## Summary of Changes

| Category | Before | After | Reduction |
|----------|--------|-------|-----------|
| Root MD files | 8 | 2 | -75% |
| docs/ files | 24 | 8 | -67% |
| Nextflow modules dirs | 2 | 1 | -50% |
| Work directories | 24 | 0 | -100% |
| Python packages | 2 | 1-2 | -0-50% |
| Scripts | 25+ | 15 | -40% |

**Estimated Disk Savings:** ~500MB (excluding .snakemake)

---

## Next Steps After Cleanup

1. ✅ Run cleanup script
2. ✅ Verify all functionality still works
3. ✅ Update pyproject.toml if consolidating packages
4. ✅ Update imports in tests
5. ✅ Commit cleaned codebase
6. 🚀 Proceed with LLM integration
