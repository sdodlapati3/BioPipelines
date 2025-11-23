# BioPipelines Codebase Assessment

## Architecture Overview
- **Execution Model**: Host-based Snakemake orchestration with containerized tools
- **Container Storage**: ~/BioPipelines/containers/images/ (12 containers, ~20GB total)
- **Data Storage**: /scratch/sdodl001/BioPipelines/data/ (raw/processed/results)
- **Workflow Engine**: Micromamba-managed Snakemake 8.20.5 on compute nodes

## Current State (Post-Containerization)
✅ **Active & Essential:**
- `scripts/run_all_pipelines.sh` - Production runner for all 10 pipelines
- `scripts/submit_pipeline_with_container.sh` - Core submission script
- `scripts/containers/build_*_container.slurm` - 12 container build scripts
- `test_compute_node.sh` - Environment verification tool
- All `pipelines/*/Snakefile` - Workflow definitions
- All `containers/*/\*.def` - Container definitions

## Redundancy Analysis

### 1. DUPLICATE SUBMISSION SCRIPTS (CLEANUP NEEDED)
```
scripts/submit_all_pipelines.sh          # OLD - missing structural-variants
scripts/run_all_pipelines.sh             # ACTIVE - includes all 10 pipelines
scripts/submit_containerized.sh          # UNKNOWN PURPOSE - needs review
scripts/submit_pipeline.sh               # LEGACY - without container support?
```
**Recommendation**: Keep only `run_all_pipelines.sh` and `submit_pipeline_with_container.sh`

### 2. DEPRECATED DOWNLOAD SCRIPTS
```
scripts/deprecated/download_*            # 5 scripts
scripts/deprecated/download_scripts/*    # 4 scripts
```
**Status**: Already archived in deprecated/ folder - OK

### 3. LEGACY SUBMISSION SCRIPTS
```
scripts/deprecated/submit_scripts/*      # 10 pipeline-specific scripts
scripts/deprecated/submit_*_simple.sh    # 4 simplified submission scripts
```
**Status**: Already archived - OK

### 4. CONDA ENVIRONMENT SCRIPTS (OBSOLETE)
```
scripts/pre_build_conda_envs.sh          # No longer needed - fully containerized
```
**Recommendation**: DELETE or move to deprecated/

### 5. CONTAINER BUILD DUPLICATES
```
scripts/containers/build_all_containers.sh
scripts/containers/build_all.sh
```
**Status**: Similar purpose, need to consolidate or choose one

### 6. TEST SCRIPTS
```
scripts/containers/test_base.sh
scripts/containers/test_rna_seq.sh
scripts/containers/test_rna_seq_real_data.sh
test_compute_node.sh                     # Root level - should move to scripts/
```
**Recommendation**: Move test_compute_node.sh to scripts/testing/

## Proposed Clean Architecture

### Core Scripts (Keep)
```
scripts/
├── run_all_pipelines.sh                  # Main entry point
├── submit_pipeline_with_container.sh     # Core submission logic
├── containers/
│   ├── build_all_containers.sh           # Master build script
│   ├── build_*_container.slurm           # Individual container builds (12)
│   └── check_build_status.sh
└── testing/
    ├── test_compute_node.sh
    └── test_*.sh
```

### Scripts to Remove/Archive
```
DELETE:
- scripts/submit_all_pipelines.sh         # Superseded by run_all_pipelines.sh
- scripts/pre_build_conda_envs.sh         # No longer used
- scripts/build_*_index*.sh               # Reference building (keep in docs, not scripts)

REVIEW & DECIDE:
- scripts/submit_containerized.sh         # Unknown usage
- scripts/submit_pipeline.sh              # May be legacy
- scripts/containers/build_all.sh vs build_all_containers.sh (consolidate)
```

## Data Organization

### Current Structure
```
data/
├── raw/              # Input datasets (10 pipelines)
├── processed/        # Intermediate outputs
└── results/          # Final outputs
```
**Status**: Clean and well-organized

### Cache Directory
```
cache/
└── scratch-*-*.h5ad  # One orphaned cache file
```
**Recommendation**: DELETE or configure proper cache location

## Documentation Status

### Essential Docs (Good)
- `docs/CONTAINER_ARCHITECTURE.md`
- `docs/QUICK_START_CONTAINERS.md`
- `README.md`

### Status Files (Useful)
- `PIPELINE_STATUS_FINAL.md`
- `CONTAINER_IMPLEMENTATION_SUMMARY.md`
- `PREFLIGHT_SUMMARY.txt`

### Redundant/Outdated (Review)
- `REORGANIZATION_SUMMARY.md`
- `ARCHITECTURE_REVIEW.md`
- Multiple TODO files in docs/

## Logs Directory
```
logs/
├── pipeline_runs/    # Active pipeline logs
├── build_*.err       # Container build logs (77 files - historical)
└── scrna_seq_*.err   # Old pipeline runs
```
**Recommendation**: Archive old logs (>7 days) to logs/archive/

## Configuration Files

### Active
- `config/defaults.yaml`
- `config/snakemake_profiles/containerized/config.yaml`
- `pyproject.toml`
- `environment.yml`

**Status**: All necessary and up-to-date

## Priority Cleanup Actions

1. **HIGH PRIORITY**
   - Delete `scripts/submit_all_pipelines.sh` (superseded)
   - Delete or archive `scripts/pre_build_conda_envs.sh`
   - Move `test_compute_node.sh` to `scripts/testing/`
   - Archive old build logs (logs/build_*.err)

2. **MEDIUM PRIORITY**
   - Consolidate container build scripts (build_all vs build_all_containers)
   - Review and remove/document: submit_containerized.sh, submit_pipeline.sh
   - Clean cache/ directory

3. **LOW PRIORITY**
   - Archive old TODO/status docs to docs/archive/
   - Add .gitignore entries for large logs
   - Document reference building scripts or move to docs/

## Size Analysis
```
Total containers: ~20GB (in home, excluded from git)
Raw data: ~200GB (in /scratch)
Logs: ~50MB (growing)
.snakemake/: ~2GB conda artifacts (consider cleaning periodically)
```

## Final Recommendations

**Immediate Actions (Post-Pipeline Validation):**
1. Verify all 10 pipelines complete successfully
2. Delete redundant submission scripts
3. Archive historical logs
4. Move test scripts to proper directory

**Maintain Clean Codebase:**
- Regular log cleanup (weekly)
- Document any new scripts immediately
- Keep deprecated/ folder for reference but don't accumulate
- Review .snakemake/ conda cache periodically

