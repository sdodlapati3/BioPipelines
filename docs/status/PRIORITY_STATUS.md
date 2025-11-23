# Priority Task Status (Updated: 2025-11-21 23:59)

## Completed ✅

### Priority 1: Metagenomics Pipeline
- **Status**: Implementation complete, conda environment issues
- **Data**: 2.6GB downloaded (SRR7889849)
- **Issue**: HUMAn functional.yaml conda conflicts persist
- **Action**: Consider using mamba or separate pip install for humann
- **Jobs**: 269 (failed), 271 (failed), 272 (failed), 276 (failing)

### Priority 2: scRNA-seq Pipeline  
- **Status**: Data acquired via alternative method (10x direct)
- **Data**: 5.2GB PBMC 1k cells (v3 chemistry)
  - Files: pbmc_1k_v3_fastqs/ (6 FASTQ files)
  - L001 + L002: R1 (719MB + 714MB), R2 (1.7GB + 1.7GB), I1 (245MB + 243MB)
- **Blocker**: NCBI SSL certificate error resolved by using 10x Genomics website
- **Next**: Need to create scRNA-seq Snakefile (CellRanger, Seurat, scanpy)

### Priority 3: Long-Read Sequencing Pipeline
- **Status**: COMPLETE ✅
- **Commit**: 5cf6615 (46 total commits today)
- **Implementation**:
  - config.yaml: ONT/PacBio platform parameters
  - Snakefile: 8 rules (minimap2, sniffles2, cutesv, survivor, whatshap, nanoplot)
  - 4 conda environments (alignment, sv_calling, phasing, qc)
  - submit_long_read.sh: SLURM script (16 CPUs, 64GB, 12 hours)
  - download_long_read_data.py: E. coli ONT test data downloader
- **Features**:
  - Dual SV callers (Sniffles2 + CuteSV)
  - SURVIVOR merger for multi-sample consensus
  - WhatsHap haplotype phasing
  - NanoPlot read QC
  - Platform-specific minimap2 presets
- **Next**: Download test data and validate pipeline

## In Progress 🔄

### Hi-C Pipeline (Job 270)
- **Status**: Running 39+ minutes (alignment stage)
- **Data**: 27.8GB ENCODE K562 cells
- **Expected**: 1-2 hours total runtime
- **Output**: Contact matrices, TADs, chromatin loops

### Methylation Pipeline (Job 277)  
- **Status**: Running 1 minute
- **Data**: 115GB WGBS (SRR5329161: 74GB R1 + 41GB R2)
- **Fix**: Pinned libdeflate=1.18 for htslib compatibility
- **Expected**: 6-8 hours runtime
- **Previous**: Job 273 (failed 40s), Job 275 (failed)

## Blocked/Issues ❌

### Metagenomics Conda Dependencies
- **Problem**: humann requires metaphlan >=3.1, which requires phylophlan, which requires biopython >=1.73
  - Circular dependency with other conda packages
  - LibMambaUnsatisfiableError persists despite version pinning
- **Attempts**:
  - Relaxed versions (>=4.0): Failed
  - Exact versions (=3.6, =4.0, =1.79): Failed
- **Solutions to try**:
  1. Use mamba instead of conda (faster resolver)
  2. Install humann via pip after conda base env
  3. Use docker/singularity container
  4. Create separate environment for humann only
  5. Use older bioconda snapshot

## Next Priority 4: Structural Variants Pipeline

Still to be implemented after long-read validation.

## Summary

**Today's Progress**:
- ✅ 46 commits
- ✅ ChIP-seq breakthrough (91,162 peaks after 17+ attempts)
- ✅ Long-read pipeline fully implemented
- ✅ 10x scRNA-seq data downloaded (5.2GB)
- ✅ Hi-C data downloaded and processing (27.8GB)
- ✅ WGBS data downloaded and processing (115GB)
- 🔄 2 pipelines running (Hi-C 39m, Methylation 1m)
- ❌ 1 blocker (metagenomics conda)

**Pipelines Status**:
1. RNA-seq: ✅ COMPLETE (3,497 DEGs)
2. DNA-seq: ✅ COMPLETE (1.45M variants)
3. ATAC-seq: ✅ COMPLETE (143K peaks)
4. ChIP-seq: ✅ COMPLETE (91,162 peaks)
5. Hi-C: 🔄 TESTING (Job 270, 39m)
6. Methylation: 🔄 TESTING (Job 277, 1m, 115GB)
7. Metagenomics: ❌ CONDA BLOCKED
8. scRNA-seq: ✅ DATA READY, need Snakefile
9. Long-read: ✅ IMPLEMENTED, need test data
10. Structural Variants: 📋 PLANNED
