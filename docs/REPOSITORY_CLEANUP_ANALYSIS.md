# Repository Cleanup Analysis

## Executive Summary

After comprehensive analysis, the repository has **significant redundancy and cleanup opportunities**. Total potential space savings: **~2.5GB+** (excluding container images).

---

## Current Repository State

### Directory Structure Overview

```
BioPipelines/                    # Root
├── containers/                  # ✅ KEEP - Active container definitions
├── config/                      # ✅ KEEP - Configuration files
├── data/                        # ✅ KEEP - Data links (symlinks to scratch)
├── deprecated/                  # ⚠️ REVIEW - Old code (308K)
├── docs/                        # ⚠️ CLEANUP - Many redundant docs
│   └── archive/                 # 🗑️ DELETE - 34 old status files (600K)
├── examples/                    # ✅ KEEP - Usage examples
├── generated_workflows/         # 🗑️ DELETE - Temporary outputs (13M)
├── htmlcov/                     # 🗑️ DELETE - Coverage reports (9.3M)
├── logs/                        # 🗑️ CLEAN - Old build logs (127M)
├── nextflow-pipelines/          # ✅ KEEP - Active Nextflow implementation
├── notebooks/                   # ⚠️ EMPTY - No content, placeholder only
├── pipelines_snakemake_archived/ # 🗑️ DELETE/ARCHIVE - Old Snakemake (272M)
├── scripts/                     # ⚠️ CLEANUP - Has deprecated subfolder
├── src/                         # ✅ KEEP - Main source code
├── SURVIVOR/                    # 🗑️ DELETE - External tool, should be submodule (25M)
├── tests/                       # ✅ KEEP - Test suite
├── tools/                       # 🗑️ DELETE - Manta binary (62M)
├── benchmarks/                  # ⚠️ EMPTY - Just results subfolder
├── .snakemake/                  # 🗑️ DELETE - Snakemake cache (2.2G)
├── .nextflow/                   # ⚠️ CLEAN - Can regenerate (44K)
├── .coverage                    # 🗑️ DELETE - Test artifact
└── Various hidden files         # ⚠️ REVIEW
```

---

## Detailed Cleanup Recommendations

### 1. HIGH PRIORITY - Large Directories to Delete

| Directory | Size | Action | Reason |
|-----------|------|--------|--------|
| `.snakemake/` | 2.2GB | DELETE | Build cache, can regenerate |
| `pipelines_snakemake_archived/` | 272MB | ARCHIVE/DELETE | Superseded by Nextflow |
| `logs/` | 127MB | CLEAN | Keep last 7 days only |
| `tools/` | 62MB | DELETE | Manta should be in containers |
| `SURVIVOR/` | 25MB | REMOVE | External repo, use submodule |
| `generated_workflows/` | 13MB | DELETE | Temporary test outputs |
| `htmlcov/` | 9.3MB | DELETE | Test coverage artifacts |

**Total: ~2.7GB savings**

### 2. MEDIUM PRIORITY - Documentation Cleanup

**`docs/archive/` - 34 files to DELETE:**
```
All historical status/progress files should be deleted:
- PHASE2_COMPLETE.md
- PHASE3_COMPLETE.md
- PHASES_4_5_COMPLETE.md
- PROJECT_COMPLETE.md
- PROGRESS_REPORT_*.md
- PIPELINE_STATUS*.md
- SESSION_SUMMARY.md
- CLEANUP_COMPLETED.md
- REORGANIZATION_SUMMARY.md
- CODEBASE_CLEANUP_PLAN.md
- TODO_CONSOLIDATED.md
- etc.
```

These are development artifacts, not user documentation.

**`docs/` main folder - REORGANIZE:**

Keep (Active Documentation):
- `README.md` (in each pipeline folder)
- `API_REFERENCE.md`
- `CONTAINER_ARCHITECTURE.md`
- `WORKFLOW_COMPOSER_GUIDE.md`
- `TUTORIALS.md`
- `WEB_INTERFACE.md`
- `QUICK_START_CONTAINERS.md`
- `GCP_HPC_SETUP.md`
- `GCP_STORAGE_ARCHITECTURE.md`
- `LLM_SETUP.md`
- `LIGHTNING_AI_INTEGRATION.md`

Archive or Delete:
- `SYSTEM_FLOW_ANALYSIS.md` - Internal analysis
- `FRAMEWORK_ARCHITECTURE_REVIEW.md` - Internal analysis
- `COMPONENT_WALKTHROUGH.md` - Developer notes
- `COMPOSITION_PATTERNS.md` - Developer notes
- Various design docs after implementation

### 3. LOW PRIORITY - Code Cleanup

**`deprecated/` directory:**
```
deprecated/
├── alignment/          # Empty module placeholders
├── containers/         # Old registry.py
├── core/              # Old core modules
├── data_download/     # Superseded
├── expression/        # Empty
├── peak_calling/      # Empty
├── preprocessing/     # Empty
├── variant_calling/   # Empty
└── visualization/     # Empty
```
**Recommendation:** DELETE entirely - all functionality moved to `src/workflow_composer/`

**`scripts/deprecated/` directory:**
```
scripts/deprecated/
├── download_*.py      # Old data download scripts
├── submit_*.sh        # Old job submission scripts
└── backups/           # Old backups
```
**Recommendation:** DELETE - replaced by new scripts

### 4. Empty Directories to Remove

```
notebooks/
├── exploratory/       # EMPTY
├── quality_control/   # EMPTY
└── visualization/     # EMPTY

benchmarks/
└── results/          # EMPTY
```

### 5. Files to Add to .gitignore

Already properly ignored:
- `.snakemake/`
- `SURVIVOR/`
- `containers/images/`
- `.nextflow*`
- `htmlcov/`
- `.coverage`

Should be added:
- `generated_workflows/`
- `logs/` (or just `logs/*.out` `logs/*.err`)

---

## Recommended Cleanup Commands

### Phase 1: Safe Deletes (Build Artifacts)

```bash
cd /home/sdodl001_odu_edu/BioPipelines

# Remove Snakemake cache (2.2GB)
rm -rf .snakemake/

# Remove coverage artifacts
rm -rf htmlcov/ .coverage

# Remove nextflow cache
rm -rf .nextflow/

# Remove generated workflows (test outputs)
rm -rf generated_workflows/

# Clean old logs (keep last 7 days)
find logs/ -name "*.out" -o -name "*.err" -mtime +7 -delete
```

### Phase 2: Archive Old Code

```bash
# Create archive tarball
tar -czvf archive_20251126.tar.gz \
    deprecated/ \
    pipelines_snakemake_archived/ \
    scripts/deprecated/ \
    docs/archive/ \
    SURVIVOR/ \
    tools/

# Move to archive location
mv archive_20251126.tar.gz /scratch/sdodl001/archives/

# Remove archived directories
rm -rf deprecated/
rm -rf pipelines_snakemake_archived/
rm -rf scripts/deprecated/
rm -rf docs/archive/
rm -rf SURVIVOR/
rm -rf tools/
```

### Phase 3: Clean Empty Directories

```bash
# Remove empty notebook directories
rm -rf notebooks/

# Remove empty benchmarks
rm -rf benchmarks/
```

### Phase 4: Update .gitignore

```bash
cat >> .gitignore << 'EOF'

# Generated outputs
generated_workflows/

# Build logs (keep directory structure)
logs/*.out
logs/*.err
logs/*/*.out
logs/*/*.err
EOF
```

---

## Proposed Final Structure

```
BioPipelines/
├── config/                      # Configuration files
│   ├── composer.yaml
│   ├── defaults.yaml
│   ├── ensemble.yaml
│   ├── slurm.yaml
│   └── nextflow/
├── containers/                  # Container definitions
│   ├── base/
│   ├── rna-seq/
│   ├── chip-seq/
│   ├── ... (10 pipelines)
│   ├── tier2/
│   └── workflow-engine/
├── data/                        # Symlinks to scratch
├── docs/                        # Documentation
│   ├── API_REFERENCE.md
│   ├── CONTAINER_ARCHITECTURE.md
│   ├── QUICK_START_CONTAINERS.md
│   ├── TUTORIALS.md
│   ├── WEB_INTERFACE.md
│   ├── WORKFLOW_COMPOSER_GUIDE.md
│   ├── infrastructure/
│   └── tutorials/
├── examples/                    # Usage examples
│   └── ai_agent_usage.py
├── logs/                        # Runtime logs (gitignored)
├── nextflow-pipelines/          # Nextflow implementation
│   ├── modules/
│   ├── workflows/
│   └── config/
├── scripts/                     # Utility scripts
│   ├── containers/
│   ├── data/
│   ├── llm/
│   └── ...
├── src/                         # Main source code
│   └── workflow_composer/
├── tests/                       # Test suite
├── environment.yml
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Space Impact Summary

| Category | Current Size | After Cleanup | Savings |
|----------|--------------|---------------|---------|
| .snakemake | 2.2GB | 0 | 2.2GB |
| pipelines_snakemake_archived | 272MB | 0 | 272MB |
| logs | 127MB | ~20MB | 107MB |
| tools | 62MB | 0 | 62MB |
| SURVIVOR | 25MB | 0 | 25MB |
| generated_workflows | 13MB | 0 | 13MB |
| htmlcov | 9.3MB | 0 | 9.3MB |
| docs/archive | ~600KB | 0 | 600KB |
| deprecated | 308KB | 0 | 308KB |
| **TOTAL** | ~2.7GB | ~20MB | **~2.7GB** |

---

## Decision Required

Before proceeding with cleanup:

1. **Archive vs Delete**: Should we create a backup tarball before deleting?
2. **Snakemake pipelines**: Permanently remove or keep in separate branch?
3. **SURVIVOR**: Install as git submodule or container-only?
4. **Logs retention**: How many days of logs to keep?

---

*Analysis completed: 2025-11-26*
