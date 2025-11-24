# Tier 2 Container Module Design
**Created**: November 24, 2025  
**Purpose**: Define domain-specific pre-built container modules for bioinformatics workflows  
**Target**: 10 modules covering 95% of common workflows  

---

## Design Principles

### Size & Build Strategy
- **Target size per module**: 3-8 GB (compressed)
- **Build location**: SLURM compute nodes using fakeroot
- **Build time**: 15-45 minutes per module
- **Storage**: Shared cache `/scratch/containers/tier2/`
- **Update frequency**: Monthly for security patches, quarterly for tool versions

### Layer Optimization
```dockerfile
# Optimal layer structure for caching
1. Base OS + system libraries (from Tier 1)
2. Language runtimes (Python, R, Perl)
3. Common dependencies (samtools, bedtools, htslib)
4. Domain-specific tools (heaviest first)
5. Validation scripts
6. Environment configuration
```

### Quality Standards
- ✅ All tools must pass version checks
- ✅ Import tests for Python/R packages
- ✅ Security scan with no critical vulnerabilities
- ✅ Documentation with usage examples
- ✅ Reproducible builds (pinned versions)

---

## Module 1: Alignment Short-Read (`alignment_short_read`)

### Purpose
Mapping short-read sequencing data (RNA-seq, ChIP-seq, ATAC-seq, DNA-seq)

### Tools Included
| Tool | Version | Size | Purpose |
|------|---------|------|---------|
| STAR | 2.7.11a | 2.1 GB | RNA-seq alignment, splice-aware |
| Bowtie2 | 2.5.3 | 45 MB | ChIP-seq, ATAC-seq, DNA-seq alignment |
| BWA | 0.7.18 | 8 MB | DNA-seq alignment, variant calling |
| Salmon | 1.10.3 | 180 MB | RNA-seq quantification, pseudo-alignment |
| samtools | 1.19.2 | 15 MB | BAM processing |
| sambamba | 1.0.1 | 6 MB | Fast BAM sorting/filtering |

### Dependencies (from Tier 1)
- Python 3.11 (for STAR scripts)
- htslib 1.19
- zlib, bzip2, xz compression libraries

### Container Size
- **Uncompressed**: ~8 GB
- **Compressed (SIF)**: ~3.5 GB

### Environment Variables
```bash
# Tool paths
export STAR_PATH=/opt/alignment/star/bin
export BOWTIE2_PATH=/opt/alignment/bowtie2/bin
export BWA_PATH=/opt/alignment/bwa/bin
export SALMON_PATH=/opt/alignment/salmon/bin

# Reference genomes (mount point)
export GENOME_DIR=/references
export STAR_INDEX_DIR=/references/star_indexes
export BOWTIE2_INDEX_DIR=/references/bowtie2_indexes

# Performance tuning
export OMP_NUM_THREADS=8  # Override in Nextflow
```

### Validation Tests
```bash
# Version checks
STAR --version
bowtie2 --version
bwa
salmon --version
samtools --version

# Functional tests (mini test data)
STAR --genomeDir /test/star_mini --readFilesIn /test/R1.fq.gz
bowtie2 -x /test/bt2_mini -U /test/R1.fq.gz
bwa mem /test/bwa_mini /test/R1.fq.gz
salmon quant -i /test/salmon_mini -l A -1 /test/R1.fq.gz -o /test/out
```

### Build Time Estimate
- **Download dependencies**: 5 min
- **Compile tools**: 15 min
- **Install binaries**: 5 min
- **Validation tests**: 3 min
- **Total**: ~28 minutes

---

## Module 2: Variant Calling (`variant_calling`)

### Purpose
Identify genetic variants (SNPs, indels, structural variants)

### Tools Included
| Tool | Version | Size | Purpose |
|------|---------|------|---------|
| GATK | 4.5.0.0 | 650 MB | Best-practices variant calling |
| FreeBayes | 1.3.7 | 12 MB | Haplotype-based variant detection |
| BCFtools | 1.19 | 18 MB | VCF manipulation |
| VEP | 111.0 | 1.2 GB | Variant effect prediction |
| SnpEff | 5.2 | 450 MB | Functional annotation |
| SnpSift | 5.2 | 25 MB | VCF filtering/annotation |

### Dependencies
- Java 17 (for GATK, SnpEff)
- Python 3.11 (for VEP)
- Perl 5.36 (for VEP)
- htslib, tabix

### Container Size
- **Uncompressed**: ~8 GB
- **Compressed (SIF)**: ~3.2 GB

### Environment Variables
```bash
# Java configuration for GATK
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
export GATK_LOCAL_JAR=/opt/gatk/gatk.jar

# VEP cache (mount point)
export VEP_CACHE_DIR=/references/vep_cache
export VEP_PLUGINS=/opt/vep/plugins

# SnpEff data
export SNPEFF_DATA=/references/snpeff_data
```

### Validation Tests
```bash
gatk --version
freebayes --version
bcftools --version
vep --help
snpEff -version

# Functional test
gatk HaplotypeCaller --help
vep --offline --check_existing
```

### Build Time Estimate
- **Total**: ~35 minutes (large VEP/GATK downloads)

---

## Module 3: Peak Calling (`peak_calling`)

### Purpose
Identify enriched regions in ChIP-seq, ATAC-seq, CUT&RUN data

### Tools Included
| Tool | Version | Size | Purpose |
|------|---------|------|---------|
| MACS2 | 2.2.9.1 | 15 MB | ChIP-seq peak calling (narrow/broad) |
| MACS3 | 3.0.1 | 18 MB | Enhanced peak calling, subpeaks |
| HOMER | 4.11.1 | 280 MB | Motif analysis, peak annotation |
| deepTools | 3.5.5 | 120 MB | Coverage tracks, QC metrics |
| bedtools | 2.31.1 | 12 MB | Genomic interval operations |

### Dependencies
- Python 3.11 + NumPy/SciPy (for MACS, deepTools)
- Perl (for HOMER)
- R 4.3 (for HOMER visualization)

### Container Size
- **Uncompressed**: ~3 GB
- **Compressed (SIF)**: ~1.2 GB

### Environment Variables
```bash
export MACS2_PATH=/opt/peaks/macs2/bin
export MACS3_PATH=/opt/peaks/macs3/bin
export HOMER_PATH=/opt/peaks/homer/bin
export DEEPTOOLS_PATH=/opt/peaks/deeptools/bin
```

### Build Time Estimate
- **Total**: ~18 minutes

---

## Module 4: Assembly & Scaffolding (`assembly`)

### Purpose
De novo genome/transcriptome assembly, scaffolding

### Tools Included
| Tool | Version | Size | Purpose |
|------|---------|------|---------|
| SPAdes | 4.0.0 | 125 MB | Genome assembly (Illumina) |
| Trinity | 2.15.1 | 580 MB | Transcriptome assembly (RNA-seq) |
| MEGAHIT | 1.2.9 | 8 MB | Fast metagenome assembly |
| QUAST | 5.2.0 | 95 MB | Assembly quality assessment |
| BUSCO | 5.7.1 | 450 MB | Completeness evaluation |

### Dependencies
- Python 3.11 (SPAdes, QUAST, BUSCO)
- Java 11 (Trinity)
- Boost, zlib (SPAdes)

### Container Size
- **Uncompressed**: ~6 GB
- **Compressed (SIF)**: ~2.5 GB

### Build Time Estimate
- **Total**: ~25 minutes

---

## Module 5: Quantification (`quantification`)

### Purpose
Gene/transcript expression quantification (RNA-seq, scRNA-seq)

### Tools Included
| Tool | Version | Size | Purpose |
|------|---------|------|---------|
| featureCounts | 2.0.6 | 8 MB | Read counting (gene-level) |
| HTSeq | 2.0.5 | 12 MB | Read counting (flexible) |
| RSEM | 1.3.3 | 45 MB | Transcript-level quantification |
| Kallisto | 0.51.0 | 6 MB | Fast pseudo-alignment quantification |
| StringTie | 2.2.2 | 8 MB | Transcript assembly & quantification |

### Dependencies
- Python 3.11 (HTSeq)
- R 4.3 (featureCounts - Subread package)
- htslib, samtools

### Container Size
- **Uncompressed**: ~2.5 GB
- **Compressed (SIF)**: ~1.1 GB

### Build Time Estimate
- **Total**: ~15 minutes

---

## Module 6: Single-Cell RNA-seq (`scrna`)

### Purpose
scRNA-seq processing, QC, clustering, differential expression

### Tools Included
| Tool | Version | Size | Purpose |
|------|---------|------|---------|
| STARsolo | 2.7.11a | 2.1 GB | scRNA-seq alignment & counting |
| Cell Ranger | 8.0.0 | 3.2 GB | 10x Genomics official pipeline |
| Seurat | 5.0.1 | 450 MB | R-based analysis framework |
| Scanpy | 1.10.0 | 280 MB | Python-based analysis |
| Velocyto | 0.17.17 | 95 MB | RNA velocity analysis |

### Dependencies
- Python 3.11 (Scanpy, Velocyto)
- R 4.3 (Seurat)
- HDF5, loom libraries

### Container Size
- **Uncompressed**: ~12 GB (Cell Ranger is large)
- **Compressed (SIF)**: ~5.5 GB

### Build Time Estimate
- **Total**: ~40 minutes (Cell Ranger download + compile)

---

## Module 7: Long-Read Tools (`longread_tools`)

### Purpose
Long-read sequencing data (PacBio, ONT) processing

### Tools Included
| Tool | Version | Size | Purpose |
|------|---------|------|---------|
| Minimap2 | 2.28 | 12 MB | Long-read alignment |
| NanoFilt | 2.8.0 | 5 MB | ONT read filtering |
| NanoPlot | 1.42.0 | 8 MB | ONT QC visualization |
| Canu | 2.2 | 125 MB | Long-read assembly |
| Flye | 2.9.3 | 18 MB | Fast long-read assembly |
| Medaka | 1.11.3 | 380 MB | ONT consensus polishing |

### Dependencies
- Python 3.11 (NanoFilt, NanoPlot, Medaka)
- zlib, minimap2 libraries

### Container Size
- **Uncompressed**: ~4 GB
- **Compressed (SIF)**: ~1.8 GB

### Build Time Estimate
- **Total**: ~22 minutes

---

## Module 8: Methylation Analysis (`methylation`)

### Purpose
Bisulfite sequencing, WGBS, RRBS methylation analysis

### Tools Included
| Tool | Version | Size | Purpose |
|------|---------|------|---------|
| Bismark | 0.24.2 | 85 MB | BS-seq alignment & methylation calling |
| MethylDackel | 0.6.1 | 4 MB | Fast methylation extraction |
| BSseeker3 | 1.1.0 | 25 MB | Alternative BS-seq aligner |
| DSS | 2.50.0 | 45 MB | Differential methylation (R) |
| methylKit | 1.28.0 | 38 MB | Methylation analysis (R) |

### Dependencies
- Python 3.11 (BSseeker3)
- R 4.3 (DSS, methylKit)
- Bowtie2 (from Tier 2 alignment module)

### Container Size
- **Uncompressed**: ~2.8 GB
- **Compressed (SIF)**: ~1.2 GB

### Build Time Estimate
- **Total**: ~18 minutes

---

## Module 9: Metagenomics (`metagenomics`)

### Purpose
Taxonomic profiling, metagenome assembly, functional annotation

### Tools Included
| Tool | Version | Size | Purpose |
|------|---------|------|---------|
| Kraken2 | 2.1.3 | 18 MB | Fast taxonomic classification |
| Bracken | 2.9 | 6 MB | Species abundance estimation |
| MetaPhlAn4 | 4.1.0 | 95 MB | Marker-based profiling |
| HUMAnN | 3.8 | 180 MB | Functional profiling |
| Centrifuge | 1.0.4 | 25 MB | Memory-efficient classification |

### Dependencies
- Python 3.11 (Bracken, MetaPhlAn, HUMAnN)
- Perl (Kraken2)
- Bowtie2 (for MetaPhlAn)

### Container Size
- **Uncompressed**: ~3.5 GB
- **Compressed (SIF)**: ~1.5 GB

### Build Time Estimate
- **Total**: ~20 minutes

---

## Module 10: Structural Variants (`structural_variants`)

### Purpose
Detect large genomic rearrangements, CNVs, SVs

### Tools Included
| Tool | Version | Size | Purpose |
|------|---------|------|---------|
| Manta | 1.6.0 | 45 MB | SV discovery (Illumina) |
| DELLY | 1.2.6 | 12 MB | Integrated SV caller |
| LUMPY | 0.3.1 | 8 MB | Probabilistic SV detection |
| CNVkit | 0.9.10 | 35 MB | Copy number variation (WES/WGS) |
| SURVIVOR | 1.0.7 | 3 MB | SV merging & comparison |

### Dependencies
- Python 3.11 (CNVkit)
- samtools, bcftools
- htslib

### Container Size
- **Uncompressed**: ~2.5 GB
- **Compressed (SIF)**: ~1.1 GB

### Build Time Estimate
- **Total**: ~16 minutes

---

## Summary Table

| Module | Size (GB) | Build Time | Primary Use Cases | Priority |
|--------|-----------|------------|-------------------|----------|
| alignment_short_read | 3.5 | 28 min | RNA/DNA/ChIP-seq alignment | ⭐⭐⭐ |
| variant_calling | 3.2 | 35 min | SNP/indel calling, annotation | ⭐⭐⭐ |
| peak_calling | 1.2 | 18 min | ChIP-seq, ATAC-seq peaks | ⭐⭐ |
| assembly | 2.5 | 25 min | Genome/transcriptome assembly | ⭐⭐ |
| quantification | 1.1 | 15 min | RNA-seq expression | ⭐⭐⭐ |
| scrna | 5.5 | 40 min | 10x scRNA-seq analysis | ⭐⭐ |
| longread_tools | 1.8 | 22 min | PacBio/ONT processing | ⭐ |
| methylation | 1.2 | 18 min | Bisulfite-seq analysis | ⭐ |
| metagenomics | 1.5 | 20 min | Microbiome profiling | ⭐ |
| structural_variants | 1.1 | 16 min | SV/CNV detection | ⭐ |
| **TOTAL** | **22.6 GB** | **237 min** | | |

**Storage Efficiency**: 22.6 GB for 10 modules vs 150-200 GB for per-user tool installations

---

## Build Order Strategy

### Phase 1: Core Modules (Week 2)
Build order based on current pipeline dependencies:
1. **alignment_short_read** (needed by 6 pipelines)
2. **quantification** (needed by RNA-seq)
3. **peak_calling** (needed by ChIP-seq, ATAC-seq)

### Phase 2: Specialized Modules (Week 3)
4. **variant_calling** (needed by DNA-seq)
5. **metagenomics** (already validated)
6. **longread_tools** (already validated)

### Phase 3: Advanced Modules (Week 4)
7. **scrna** (scRNA-seq failing, complex)
8. **methylation** (failing pipeline)
9. **structural_variants** (Hi-C, complex workflows)
10. **assembly** (lower priority, no current pipeline)

---

## Build Infrastructure Requirements

### SLURM Job Template
```bash
#!/bin/bash
#SBATCH --job-name=build_tier2_{MODULE}
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpuspot
#SBATCH --output=logs/build_tier2_{MODULE}_%j.out
#SBATCH --error=logs/build_tier2_{MODULE}_%j.err

# Build using fakeroot (no sudo required)
singularity build --fakeroot \
  /scratch/containers/tier2/{MODULE}.sif \
  containers/tier2/{MODULE}.def

# Validation
singularity exec /scratch/containers/tier2/{MODULE}.sif \
  /opt/validation/test_suite.sh

# Deploy to shared cache if tests pass
if [ $? -eq 0 ]; then
  chmod 755 /scratch/containers/tier2/{MODULE}.sif
  echo "✅ {MODULE} built and validated successfully"
else
  echo "❌ {MODULE} validation failed"
  exit 1
fi
```

### Parallel Build Strategy
```bash
# Build 4 modules in parallel (avoid overloading)
sbatch build_tier2_alignment_short_read.sh
sbatch build_tier2_quantification.sh
sbatch build_tier2_peak_calling.sh
sbatch build_tier2_variant_calling.sh

# Wait for completion, then build next batch
```

---

## Next Steps

1. **Create Singularity Definition Files** (Task 2)
   - Start with `alignment_short_read.def`
   - Template structure with validation
   - Optimized layer caching

2. **Implement Build Orchestration** (Task 3)
   - Automated build script
   - Parallel execution
   - Validation framework

3. **Test with Current Pipelines** (Task 5)
   - Migrate RNA-seq to use Tier 2 containers
   - Benchmark performance vs current approach
   - Validate reproducibility

---

**Status**: Design complete, ready for implementation  
**Next**: Create Singularity .def files starting with alignment_short_read module
