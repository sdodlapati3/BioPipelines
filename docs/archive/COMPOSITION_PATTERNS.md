# Nextflow Module Composition Patterns

**Date:** November 25, 2024  
**Purpose:** Demonstrate how to chain 63 modules into complete bioinformatics workflows

This document provides 20 real-world workflow compositions using the BioPipelines module library. Each pattern shows module chaining, data flow, and parameter configuration.

---

## Table of Contents

1. [RNA-seq Bulk Analysis](#1-rna-seq-bulk-analysis)
2. [ChIP-seq Peak Calling](#2-chip-seq-peak-calling)
3. [ATAC-seq Analysis](#3-atac-seq-analysis)
4. [Whole Genome Variant Calling](#4-whole-genome-variant-calling)
5. [Somatic Variant Calling](#5-somatic-variant-calling)
6. [Bisulfite Sequencing (BS-seq)](#6-bisulfite-sequencing-bs-seq)
7. [scRNA-seq with Seurat](#7-scrna-seq-with-seurat)
8. [scRNA-seq with Cell Ranger + Scanpy](#8-scrna-seq-with-cell-ranger--scanpy)
9. [Metagenomics Taxonomic Profiling](#9-metagenomics-taxonomic-profiling)
10. [Metagenomic Assembly and Annotation](#10-metagenomic-assembly-and-annotation)
11. [Long-read Genome Assembly](#11-long-read-genome-assembly)
12. [Structural Variant Detection](#12-structural-variant-detection)
13. [Hi-C Chromatin Interaction](#13-hi-c-chromatin-interaction)
14. [RNA-seq De Novo Assembly](#14-rna-seq-de-novo-assembly)
15. [Alternative Splicing Analysis](#15-alternative-splicing-analysis)
16. [MeDIP-seq Methylation](#16-medip-seq-methylation)
17. [Small RNA-seq Analysis](#17-small-rna-seq-analysis)
18. [Differential Peak Analysis](#18-differential-peak-analysis)
19. [Multi-sample Variant Calling](#19-multi-sample-variant-calling)
20. [Prokaryotic Genome Annotation](#20-prokaryotic-genome-annotation)

---

## 1. RNA-seq Bulk Analysis

**Goal:** Quantify gene expression and identify differentially expressed genes

**Workflow:** FastQC → STAR → featureCounts → DESeq2

### Complete Pipeline

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// Import modules
include { FASTQC } from './modules/qc/fastqc.nf'
include { STAR_INDEX; STAR_ALIGN } from './modules/alignment/star.nf'
include { FEATURECOUNTS } from './modules/quantification/featurecounts.nf'
include { DESEQ2_DIFFERENTIAL } from './modules/analysis/deseq2.nf'
include { MULTIQC } from './modules/qc/multiqc.nf'

// Parameters
params.reads = "data/raw/*_R{1,2}.fastq.gz"
params.genome = "data/references/genome.fa"
params.gtf = "data/references/genes.gtf"
params.outdir = "results/rnaseq"
params.samplesheet = "data/samplesheet.csv"

workflow {
    // Input channels
    reads_ch = Channel.fromFilePairs(params.reads, checkIfExists: true)
    genome = file(params.genome)
    gtf = file(params.gtf)
    samplesheet = file(params.samplesheet)
    
    // QC raw reads
    FASTQC(reads_ch)
    
    // Build STAR index
    STAR_INDEX(genome, gtf)
    
    // Align reads
    STAR_ALIGN(reads_ch, STAR_INDEX.out.index, gtf)
    
    // Count features
    FEATURECOUNTS(
        STAR_ALIGN.out.bam,
        gtf,
        'gene_id',
        false  // not paired-end counting
    )
    
    // Collect all counts
    count_matrix = FEATURECOUNTS.out.counts.collect()
    
    // Differential expression
    DESEQ2_DIFFERENTIAL(
        count_matrix,
        samplesheet,
        'condition',
        'control',
        'treatment'
    )
    
    // Aggregate QC
    qc_files = FASTQC.out.zip
        .mix(STAR_ALIGN.out.log)
        .mix(FEATURECOUNTS.out.summary)
        .collect()
    
    MULTIQC(qc_files)
}
```

### Key Features
- **QC First:** FastQC checks raw read quality
- **Splice-aware Alignment:** STAR handles RNA-seq splicing
- **Gene Counting:** featureCounts quantifies expression
- **Statistical Analysis:** DESeq2 identifies DE genes
- **Comprehensive Reporting:** MultiQC aggregates all metrics

### Expected Outputs
- `results/rnaseq/qc/fastqc/*.html` - QC reports
- `results/rnaseq/alignment/star/*.bam` - Aligned reads
- `results/rnaseq/quantification/*.counts` - Count matrices
- `results/rnaseq/deseq2/DE_results.csv` - Differential expression
- `results/rnaseq/multiqc/multiqc_report.html` - Aggregate QC

---

## 2. ChIP-seq Peak Calling

**Goal:** Identify transcription factor binding sites or histone modifications

**Workflow:** FastQC → BWA → MACS2 → HOMER → deepTools

### Complete Pipeline

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { FASTQC } from './modules/qc/fastqc.nf'
include { BWA_INDEX; BWA_MEM } from './modules/alignment/bwa.nf'
include { SAMTOOLS_SORT; SAMTOOLS_INDEX } from './modules/utilities/samtools.nf'
include { MACS2_CALLPEAK } from './modules/peaks/macs2.nf'
include { HOMER_FINDMOTIFSGENOME; HOMER_ANNOTATEPEAKS } from './modules/peaks/homer.nf'
include { DEEPTOOLS_BAMCOVERAGE; DEEPTOOLS_PLOTHEATMAP } from './modules/visualization/deeptools.nf'
include { MULTIQC } from './modules/qc/multiqc.nf'

params.treatment = "data/chip/*_treatment_R{1,2}.fastq.gz"
params.control = "data/chip/*_input_R{1,2}.fastq.gz"
params.genome = "data/references/genome.fa"
params.outdir = "results/chipseq"

workflow {
    // Channels
    treatment_ch = Channel.fromFilePairs(params.treatment)
    control_ch = Channel.fromFilePairs(params.control)
    genome = file(params.genome)
    
    // QC
    FASTQC(treatment_ch.mix(control_ch))
    
    // Index genome
    BWA_INDEX(genome)
    
    // Align treatment and control
    BWA_MEM(treatment_ch, BWA_INDEX.out.index)
    BWA_MEM(control_ch, BWA_INDEX.out.index)
    
    // Sort and index BAMs
    treatment_bam = BWA_MEM.out.bam
        .map { id, bam -> tuple(id, bam) }
    
    SAMTOOLS_SORT(treatment_bam)
    SAMTOOLS_INDEX(SAMTOOLS_SORT.out.bam)
    
    // Peak calling
    treatment_sorted = SAMTOOLS_SORT.out.bam
    control_sorted = SAMTOOLS_SORT.out.bam  // control BAMs
    
    MACS2_CALLPEAK(
        treatment_sorted,
        control_sorted,
        'narrow',  // or 'broad' for histone marks
        'hs'       // genome size
    )
    
    // Motif discovery
    HOMER_FINDMOTIFSGENOME(
        MACS2_CALLPEAK.out.peaks,
        genome,
        200  // region size
    )
    
    // Annotate peaks
    HOMER_ANNOTATEPEAKS(
        MACS2_CALLPEAK.out.peaks,
        genome,
        file(params.gtf)
    )
    
    // Visualization
    DEEPTOOLS_BAMCOVERAGE(
        SAMTOOLS_SORT.out.bam.join(SAMTOOLS_INDEX.out.bai),
        'bigwig',
        50  // bin size
    )
    
    // Aggregate QC
    MULTIQC(
        FASTQC.out.zip
            .mix(MACS2_CALLPEAK.out.qc)
            .collect()
    )
}
```

### Key Features
- **Control Subtraction:** MACS2 uses input control
- **Motif Discovery:** HOMER finds enriched motifs
- **Peak Annotation:** Links peaks to nearest genes
- **Visualization:** deepTools creates genome browser tracks

---

## 3. ATAC-seq Analysis

**Goal:** Map open chromatin regions

**Workflow:** FastQC → Bowtie2 → MACS2 → HOMER

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { FASTP_PE } from './modules/trimming/fastp.nf'
include { BOWTIE2_BUILD; BOWTIE2_ALIGN } from './modules/alignment/bowtie2.nf'
include { SAMTOOLS_SORT; SAMTOOLS_INDEX } from './modules/utilities/samtools.nf'
include { PICARD_MARKDUPLICATES } from './modules/utilities/picard.nf'
include { MACS2_CALLPEAK } from './modules/peaks/macs2.nf'
include { HOMER_ANNOTATEPEAKS } from './modules/peaks/homer.nf'

params.reads = "data/atac/*_R{1,2}.fastq.gz"
params.genome = "data/references/genome.fa"
params.outdir = "results/atacseq"

workflow {
    reads_ch = Channel.fromFilePairs(params.reads)
    
    // Quality trim (ATAC-seq specific: remove adapters)
    FASTP_PE(reads_ch)
    
    // Align with Bowtie2 (faster for ATAC-seq)
    BOWTIE2_BUILD(file(params.genome))
    BOWTIE2_ALIGN(FASTP_PE.out.reads, BOWTIE2_BUILD.out.index)
    
    // Remove duplicates
    SAMTOOLS_SORT(BOWTIE2_ALIGN.out.bam)
    SAMTOOLS_INDEX(SAMTOOLS_SORT.out.bam)
    
    bam_bai = SAMTOOLS_SORT.out.bam.join(SAMTOOLS_INDEX.out.bai)
    PICARD_MARKDUPLICATES(bam_bai)
    
    // Call peaks (no control needed for ATAC-seq)
    MACS2_CALLPEAK(
        PICARD_MARKDUPLICATES.out.bam,
        Channel.empty(),  // no control
        'narrow',
        'hs'
    )
    
    // Annotate accessible regions
    HOMER_ANNOTATEPEAKS(
        MACS2_CALLPEAK.out.peaks,
        file(params.genome),
        file(params.gtf)
    )
}
```

---

## 4. Whole Genome Variant Calling

**Goal:** Identify germline variants (SNPs and indels)

**Workflow:** FastQC → BWA → GATK (MarkDuplicates → BQSR → HaplotypeCaller) → bcftools filter

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { FASTQC } from './modules/qc/fastqc.nf'
include { BWA_INDEX; BWA_MEM } from './modules/alignment/bwa.nf'
include { GATK_MARKDUPLICATES; GATK_BASERECALIBRATOR; GATK_APPLYBQSR; GATK_HAPLOTYPECALLER } from './modules/variant_calling/gatk.nf'
include { BCFTOOLS_FILTER; BCFTOOLS_STATS } from './modules/utilities/bcftools.nf'

params.reads = "data/wgs/*_R{1,2}.fastq.gz"
params.genome = "data/references/genome.fa"
params.known_sites = "data/references/dbsnp.vcf.gz"
params.outdir = "results/variants"

workflow {
    reads_ch = Channel.fromFilePairs(params.reads)
    genome = file(params.genome)
    known_sites = file(params.known_sites)
    
    // QC
    FASTQC(reads_ch)
    
    // Align
    BWA_INDEX(genome)
    BWA_MEM(reads_ch, BWA_INDEX.out.index)
    
    // Mark duplicates
    GATK_MARKDUPLICATES(BWA_MEM.out.bam)
    
    // Base quality recalibration
    GATK_BASERECALIBRATOR(
        GATK_MARKDUPLICATES.out.bam,
        genome,
        known_sites
    )
    
    GATK_APPLYBQSR(
        GATK_MARKDUPLICATES.out.bam,
        GATK_BASERECALIBRATOR.out.table,
        genome
    )
    
    // Variant calling
    GATK_HAPLOTYPECALLER(
        GATK_APPLYBQSR.out.bam,
        genome,
        Channel.empty()  // no intervals
    )
    
    // Filter variants
    BCFTOOLS_FILTER(
        GATK_HAPLOTYPECALLER.out.vcf,
        "QUAL>30 && DP>10"
    )
    
    // Statistics
    BCFTOOLS_STATS(BCFTOOLS_FILTER.out.vcf)
}
```

---

## 5. Somatic Variant Calling

**Goal:** Identify tumor-specific mutations

**Workflow:** BWA → GATK → VarScan somatic → LoFreq → Manta (SVs)

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { BWA_MEM } from './modules/alignment/bwa.nf'
include { GATK_MARKDUPLICATES } from './modules/variant_calling/gatk.nf'
include { VARSCAN_SOMATIC } from './modules/variant_calling/varscan.nf'
include { LOFREQ_SOMATIC } from './modules/variant_calling/lofreq.nf'
include { MANTA_SOMATIC } from './modules/structural_variants/manta.nf'

params.tumor = "data/tumor/*_R{1,2}.fastq.gz"
params.normal = "data/normal/*_R{1,2}.fastq.gz"
params.genome = "data/references/genome.fa"
params.outdir = "results/somatic"

workflow {
    tumor_ch = Channel.fromFilePairs(params.tumor)
        .map { id, reads -> tuple("tumor", reads) }
    normal_ch = Channel.fromFilePairs(params.normal)
        .map { id, reads -> tuple("normal", reads) }
    
    genome = file(params.genome)
    index = file("${params.genome}.bwt")  // assume pre-built
    
    // Align tumor and normal
    BWA_MEM(tumor_ch, index)
    BWA_MEM(normal_ch, index)
    
    // Mark duplicates
    tumor_bam = BWA_MEM.out.bam.filter { it[0] == "tumor" }
    normal_bam = BWA_MEM.out.bam.filter { it[0] == "normal" }
    
    GATK_MARKDUPLICATES(tumor_bam)
    GATK_MARKDUPLICATES(normal_bam)
    
    // Combine tumor-normal pairs
    paired_bams = GATK_MARKDUPLICATES.out.bam
        .groupTuple()
        .map { id, bams -> 
            tuple("tumor_vs_normal", bams[0], bams[0] + ".bai", 
                  "normal", bams[1], bams[1] + ".bai")
        }
    
    // Somatic variant calling
    VARSCAN_SOMATIC(paired_bams, genome)
    LOFREQ_SOMATIC(paired_bams, genome)
    
    // Structural variants
    MANTA_SOMATIC(
        paired_bams,
        genome,
        file("${genome}.fai"),
        Channel.empty()  // no target BED
    )
}
```

---

## 6. Bisulfite Sequencing (BS-seq)

**Goal:** Map DNA methylation at single-base resolution

**Workflow:** Trim Galore → Bismark → Methylation Extraction → DMR calling

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { TRIMGALORE_PE } from './modules/trimming/trimgalore.nf'
include { BISMARK_GENOME_PREPARATION; BISMARK_ALIGN; BISMARK_DEDUPLICATE; BISMARK_METHYLATION_EXTRACTOR } from './modules/methylation/bismark.nf'

params.reads = "data/bsseq/*_R{1,2}.fastq.gz"
params.genome = "data/references/genome.fa"
params.outdir = "results/methylation"

workflow {
    reads_ch = Channel.fromFilePairs(params.reads)
    genome = file(params.genome)
    
    // Trim adapters (critical for BS-seq)
    TRIMGALORE_PE(reads_ch)
    
    // Prepare bisulfite-converted genome
    BISMARK_GENOME_PREPARATION(genome)
    
    // Align BS-seq reads
    BISMARK_ALIGN(
        TRIMGALORE_PE.out.reads,
        BISMARK_GENOME_PREPARATION.out.index,
        false  // not PBAT
    )
    
    // Remove duplicates (important for BS-seq)
    BISMARK_DEDUPLICATE(BISMARK_ALIGN.out.bam)
    
    // Extract methylation calls
    BISMARK_METHYLATION_EXTRACTOR(
        BISMARK_DEDUPLICATE.out.bam,
        BISMARK_GENOME_PREPARATION.out.index,
        true,  // paired-end
        false  // not comprehensive
    )
}
```

---

## 7. scRNA-seq with Seurat

**Goal:** Analyze single-cell gene expression, identify cell types

**Workflow:** Cell Ranger → Seurat QC → Clustering → Marker genes

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { CELLRANGER_COUNT } from './modules/scrna/cellranger.nf'
include { SEURAT_QC_NORMALIZE; SEURAT_CLUSTER } from './modules/scrna/seurat.nf'

params.fastq_dir = "data/scrna/fastqs"
params.transcriptome = "data/references/refdata-gex-GRCh38"
params.outdir = "results/scrna"

workflow {
    // Sample info
    samples_ch = Channel.of(
        tuple("sample1", file("${params.fastq_dir}/sample1")),
        tuple("sample2", file("${params.fastq_dir}/sample2"))
    )
    
    // Cell Ranger quantification
    CELLRANGER_COUNT(
        samples_ch,
        file(params.transcriptome)
    )
    
    // QC and normalization
    SEURAT_QC_NORMALIZE(
        CELLRANGER_COUNT.out.filtered_matrix,
        200,   // min_features
        2500,  // max_features
        5      // max_mito_percent
    )
    
    // Clustering
    SEURAT_CLUSTER(
        SEURAT_QC_NORMALIZE.out.seurat_object,
        0.5  // resolution
    )
}
```

---

## 8. scRNA-seq with Cell Ranger + Scanpy

**Goal:** Alternative Python-based scRNA-seq analysis

**Workflow:** Cell Ranger → Scanpy → Leiden clustering

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { CELLRANGER_COUNT } from './modules/scrna/cellranger.nf'
include { SCANPY_ANALYSIS } from './modules/scrna/scanpy.nf'

params.fastq_dir = "data/scrna"
params.transcriptome = "data/references/refdata-gex-GRCh38"
params.outdir = "results/scrna_scanpy"

workflow {
    samples_ch = Channel.fromPath("${params.fastq_dir}/*", type: 'dir')
        .map { dir -> tuple(dir.name, dir) }
    
    CELLRANGER_COUNT(samples_ch, file(params.transcriptome))
    
    // Scanpy analysis
    SCANPY_ANALYSIS(
        CELLRANGER_COUNT.out.filtered_matrix,
        200,  // min_genes
        3,    // min_cells
        5,    // max_mito_percent
        1.0   // resolution
    )
}
```

---

## 9. Metagenomics Taxonomic Profiling

**Goal:** Identify microbial species composition

**Workflow:** FastQC → Kraken2 → Bracken → MetaPhlAn

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { FASTQC } from './modules/qc/fastqc.nf'
include { KRAKEN2_CLASSIFY; BRACKEN } from './modules/metagenomics/kraken2.nf'
include { METAPHLAN_PROFILE; METAPHLAN_MERGE } from './modules/metagenomics/metaphlan.nf'

params.reads = "data/metagenome/*_R{1,2}.fastq.gz"
params.kraken2_db = "data/databases/kraken2_standard"
params.metaphlan_db = "data/databases/metaphlan"
params.outdir = "results/metagenomics"

workflow {
    reads_ch = Channel.fromFilePairs(params.reads)
    
    // QC
    FASTQC(reads_ch)
    
    // Kraken2 classification
    KRAKEN2_CLASSIFY(
        reads_ch,
        file(params.kraken2_db)
    )
    
    // Bracken abundance estimation
    BRACKEN(
        KRAKEN2_CLASSIFY.out.report,
        file(params.kraken2_db),
        150,  // read_length
        'S'   // species level
    )
    
    // MetaPhlAn profiling
    METAPHLAN_PROFILE(
        reads_ch,
        file(params.metaphlan_db)
    )
    
    // Merge all samples
    METAPHLAN_MERGE(
        METAPHLAN_PROFILE.out.profile.map { it[1] }.collect()
    )
}
```

---

## 10. Metagenomic Assembly and Annotation

**Goal:** Assemble and annotate metagenomic contigs

**Workflow:** MEGAHIT → Prokka → Kraken2 (contigs)

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { MEGAHIT_ASSEMBLE } from './modules/metagenomics/megahit.nf'
include { PROKKA_ANNOTATE } from './modules/annotation/prokka.nf'
include { KRAKEN2_CLASSIFY } from './modules/metagenomics/kraken2.nf'

params.reads = "data/metagenome/*_R{1,2}.fastq.gz"
params.kraken2_db = "data/databases/kraken2"
params.outdir = "results/metagenomic_assembly"

workflow {
    reads_ch = Channel.fromFilePairs(params.reads)
    
    // Assemble metagenome
    MEGAHIT_ASSEMBLE(reads_ch)
    
    // Annotate contigs
    PROKKA_ANNOTATE(
        MEGAHIT_ASSEMBLE.out.contigs,
        'Bacteria',  // kingdom
        Channel.empty(),  // no genus
        Channel.empty()   // no species
    )
    
    // Classify contigs taxonomically
    contig_ch = MEGAHIT_ASSEMBLE.out.contigs
        .map { id, contigs -> tuple(id, [contigs]) }
    
    KRAKEN2_CLASSIFY(
        contig_ch,
        file(params.kraken2_db)
    )
}
```

---

## 11. Long-read Genome Assembly

**Goal:** Assemble genome from PacBio or Nanopore reads

**Workflow:** Flye → Racon → Prokka/Augustus

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { FLYE_ASSEMBLE } from './modules/assembly/flye.nf'
include { RACON_ALIGN; RACON_POLISH } from './modules/polishing/racon.nf'
include { PROKKA_ANNOTATE } from './modules/annotation/prokka.nf'
include { QUALIMAP_BAMQC } from './modules/qc/qualimap.nf'

params.reads = "data/nanopore/*.fastq.gz"
params.genome_size = "5m"  // 5 Mb
params.read_type = "nano-raw"  // or "pacbio-raw", "nano-hq"
params.outdir = "results/long_read_assembly"

workflow {
    reads_ch = Channel.fromPath(params.reads)
        .map { file -> tuple(file.simpleName, file) }
    
    // Initial assembly
    FLYE_ASSEMBLE(
        reads_ch,
        params.read_type,
        params.genome_size
    )
    
    // Polish with Racon (3 rounds)
    round1_ch = FLYE_ASSEMBLE.out.assembly
        .combine(reads_ch, by: 0)
    
    RACON_ALIGN(round1_ch)
    RACON_POLISH(
        FLYE_ASSEMBLE.out.assembly,
        reads_ch.map { it[1] },
        RACON_ALIGN.out.paf,
        1  // round number
    )
    
    // Additional polishing rounds (simplified)
    polished_assembly = RACON_POLISH.out.consensus
    
    // Annotate
    PROKKA_ANNOTATE(
        polished_assembly,
        'Bacteria',
        Channel.empty(),
        Channel.empty()
    )
}
```

---

## 12. Structural Variant Detection

**Goal:** Identify large genomic rearrangements

**Workflow:** BWA → Manta → DELLY

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { BWA_MEM } from './modules/alignment/bwa.nf'
include { SAMTOOLS_SORT; SAMTOOLS_INDEX } from './modules/utilities/samtools.nf'
include { MANTA_GERMLINE } from './modules/structural_variants/manta.nf'
include { DELLY_CALL; DELLY_FILTER } from './modules/structural_variants/delly.nf'

params.reads = "data/wgs/*_R{1,2}.fastq.gz"
params.genome = "data/references/genome.fa"
params.exclude = "data/references/exclude.bed"
params.outdir = "results/structural_variants"

workflow {
    reads_ch = Channel.fromFilePairs(params.reads)
    genome = file(params.genome)
    genome_fai = file("${params.genome}.fai")
    exclude = file(params.exclude)
    
    // Align
    index = file("${params.genome}.bwt")
    BWA_MEM(reads_ch, index)
    
    // Sort and index
    SAMTOOLS_SORT(BWA_MEM.out.bam)
    SAMTOOLS_INDEX(SAMTOOLS_SORT.out.bam)
    
    bam_bai = SAMTOOLS_SORT.out.bam.join(SAMTOOLS_INDEX.out.bai)
    
    // Manta SV calling
    MANTA_GERMLINE(
        bam_bai,
        genome,
        genome_fai,
        Channel.empty()  // no target regions
    )
    
    // DELLY SV calling
    DELLY_CALL(bam_bai, genome, exclude)
    DELLY_FILTER(DELLY_CALL.out.bcf, 'germline')
}
```

---

## 13. Hi-C Chromatin Interaction

**Goal:** Map 3D genome organization

**Workflow:** HiC-Pro → Juicer → TAD calling

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { HICPRO_PROCESS; HICPRO_MATRIX } from './modules/hic/hicpro.nf'
include { JUICER_TOOLS_HIC; JUICER_ARROWHEAD; JUICER_HICCUPS } from './modules/hic/juicer.nf'

params.reads = "data/hic/*_R{1,2}.fastq.gz"
params.genome = "data/references/genome.fa"
params.config = "config/hicpro.config"
params.chrom_sizes = "data/references/chrom.sizes"
params.outdir = "results/hic"

workflow {
    reads_ch = Channel.fromFilePairs(params.reads)
        .map { id, reads -> tuple(id, reads[0], reads[1]) }
    
    genome = file(params.genome)
    config = file(params.config)
    chrom_sizes = file(params.chrom_sizes)
    
    // HiC-Pro processing
    genome_index = file("${params.genome}.bwt")
    
    HICPRO_PROCESS(reads_ch, genome, chrom_sizes, config)
    
    // Build contact matrices
    HICPRO_MATRIX(
        HICPRO_PROCESS.out.valid_pairs,
        chrom_sizes,
        [5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
    )
    
    // Convert to .hic format
    merged_nodups = HICPRO_PROCESS.out.merged_nodups
    
    JUICER_TOOLS_HIC(
        merged_nodups,
        chrom_sizes,
        [5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
    )
    
    // Call TADs
    JUICER_ARROWHEAD(
        JUICER_TOOLS_HIC.out.hic_file,
        10000  // resolution
    )
    
    // Call loops
    JUICER_HICCUPS(
        JUICER_TOOLS_HIC.out.hic_file,
        10000
    )
}
```

---

## 14. RNA-seq De Novo Assembly

**Goal:** Assemble transcriptome without reference

**Workflow:** Trinity → BLAST annotation → RSEM quantification

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { TRINITY_DENOVO } from './modules/assembly/trinity.nf'
include { MAKEBLASTDB; BLASTX } from './modules/sequence_search/blast.nf'
include { RSEM_PREPARE_REFERENCE; RSEM_CALCULATE_EXPRESSION } from './modules/quantification/rsem.nf'

params.reads = "data/rnaseq/*_R{1,2}.fastq.gz"
params.protein_db = "data/databases/uniprot_sprot.fasta"
params.outdir = "results/denovo_rnaseq"

workflow {
    reads_ch = Channel.fromFilePairs(params.reads)
    
    // De novo assembly
    TRINITY_DENOVO(
        reads_ch,
        '10G',  // max memory
        32      // threads
    )
    
    // Annotate with BLAST
    protein_db = file(params.protein_db)
    MAKEBLASTDB(protein_db, 'uniprot', 'prot')
    
    trinity_fasta = TRINITY_DENOVO.out.assembly
        .map { id, fasta -> tuple(id, fasta) }
    
    BLASTX(trinity_fasta, MAKEBLASTDB.out.blast_db)
    
    // Quantify with RSEM
    RSEM_PREPARE_REFERENCE(
        TRINITY_DENOVO.out.assembly.map { it[1] },
        'trinity_ref'
    )
    
    RSEM_CALCULATE_EXPRESSION(
        reads_ch,
        RSEM_PREPARE_REFERENCE.out.index,
        false,  // not STAR
        true    // paired-end
    )
}
```

---

## 15. Alternative Splicing Analysis

**Goal:** Detect differential isoform usage

**Workflow:** HISAT2 → StringTie → Ballgown/DEXSeq

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { HISAT2_BUILD; HISAT2_ALIGN } from './modules/alignment/hisat2.nf'
include { STRINGTIE_ASSEMBLE; STRINGTIE_MERGE } from './modules/assembly/stringtie.nf'

params.reads = "data/rnaseq/*_R{1,2}.fastq.gz"
params.genome = "data/references/genome.fa"
params.gtf = "data/references/genes.gtf"
params.outdir = "results/splicing"

workflow {
    reads_ch = Channel.fromFilePairs(params.reads)
    genome = file(params.genome)
    gtf = file(params.gtf)
    
    // Build HISAT2 index
    HISAT2_BUILD(genome, gtf)
    
    // Align reads
    HISAT2_ALIGN(reads_ch, HISAT2_BUILD.out.index, gtf)
    
    // Assemble transcripts per sample
    STRINGTIE_ASSEMBLE(
        HISAT2_ALIGN.out.bam,
        gtf,
        true  // ballgown output
    )
    
    // Merge all assemblies
    all_gtfs = STRINGTIE_ASSEMBLE.out.gtf.map { it[1] }.collect()
    STRINGTIE_MERGE(all_gtfs, gtf)
}
```

---

## 16. MeDIP-seq Methylation

**Goal:** Identify differentially methylated regions from immunoprecipitation

**Workflow:** BWA → MEDIPS → DMR calling

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { BWA_MEM } from './modules/alignment/bwa.nf'
include { SAMTOOLS_SORT; SAMTOOLS_INDEX } from './modules/utilities/samtools.nf'
include { MEDIPS_COVERAGE; MEDIPS_DMR } from './modules/methylation/medips.nf'

params.reads = "data/medip/*_R{1,2}.fastq.gz"
params.genome = "data/references/genome.fa"
params.bsgenome = "BSgenome.Hsapiens.UCSC.hg38"
params.outdir = "results/medip"

workflow {
    reads_ch = Channel.fromFilePairs(params.reads)
    genome = file(params.genome)
    
    // Align
    index = file("${params.genome}.bwt")
    BWA_MEM(reads_ch, index)
    
    // Sort and index
    SAMTOOLS_SORT(BWA_MEM.out.bam)
    SAMTOOLS_INDEX(SAMTOOLS_SORT.out.bam)
    
    bam_bai = SAMTOOLS_SORT.out.bam.join(SAMTOOLS_INDEX.out.bai)
    
    // MEDIPS analysis
    MEDIPS_COVERAGE(
        bam_bai,
        params.bsgenome,
        50  // window size
    )
    
    // Group samples for DMR calling
    condition1 = MEDIPS_COVERAGE.out.medips_object
        .filter { it[0].contains('treatment') }
    condition2 = MEDIPS_COVERAGE.out.medips_object
        .filter { it[0].contains('control') }
    
    MEDIPS_DMR(
        condition1.first(),
        condition2.first(),
        0.01  // FDR threshold
    )
}
```

---

## 17. Small RNA-seq Analysis

**Goal:** Quantify miRNAs and other small RNAs

**Workflow:** Cutadapt → Bowtie → miRNA annotation

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { CUTADAPT_SE } from './modules/trimming/cutadapt.nf'
include { BOWTIE_BUILD; BOWTIE_ALIGN } from './modules/alignment/bowtie.nf'
include { HTSEQ_COUNT } from './modules/quantification/htseq.nf'

params.reads = "data/smallrna/*.fastq.gz"
params.genome = "data/references/genome.fa"
params.mirna_gff = "data/references/mirna.gff3"
params.outdir = "results/smallrna"

workflow {
    reads_ch = Channel.fromPath(params.reads)
        .map { file -> tuple(file.simpleName, file) }
    
    // Trim adapters (critical for small RNA)
    CUTADAPT_SE(reads_ch)
    
    // Align with Bowtie (no gaps needed)
    genome = file(params.genome)
    BOWTIE_BUILD(genome)
    BOWTIE_ALIGN(CUTADAPT_SE.out.reads, BOWTIE_BUILD.out.index)
    
    // Count miRNAs
    mirna_gff = file(params.mirna_gff)
    HTSEQ_COUNT(
        BOWTIE_ALIGN.out.bam,
        mirna_gff,
        'miRNA',
        'union',
        false  // not reverse
    )
}
```

---

## 18. Differential Peak Analysis

**Goal:** Compare ChIP-seq peaks between conditions

**Workflow:** MACS2 (per sample) → SICER differential

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { BWA_MEM } from './modules/alignment/bwa.nf'
include { BEDTOOLS_BAMTOBED } from './modules/utilities/bedtools.nf'
include { SICER_CALL; SICER_DIFF } from './modules/peaks/sicer.nf'

params.condition1 = "data/chip/treatment*.fastq.gz"
params.condition2 = "data/chip/control*.fastq.gz"
params.genome = "hg38"
params.outdir = "results/differential_peaks"

workflow {
    cond1_ch = Channel.fromPath(params.condition1)
        .map { file -> tuple('treatment', file) }
    cond2_ch = Channel.fromPath(params.condition2)
        .map { file -> tuple('control', file) }
    
    // Align all samples
    index = file("data/references/genome.fa.bwt")
    BWA_MEM(cond1_ch.mix(cond2_ch), index)
    
    // Convert to BED
    BEDTOOLS_BAMTOBED(BWA_MEM.out.bam)
    
    // Split by condition
    bed_cond1 = BEDTOOLS_BAMTOBED.out.bed.filter { it[0] == 'treatment' }
    bed_cond2 = BEDTOOLS_BAMTOBED.out.bed.filter { it[0] == 'control' }
    
    // Differential peak calling with SICER
    SICER_DIFF(
        bed_cond1.first(),
        bed_cond2.first(),
        params.genome,
        150,  // fragment_size
        200,  // window_size
        600   // gap_size
    )
}
```

---

## 19. Multi-sample Variant Calling

**Goal:** Joint genotyping across multiple samples

**Workflow:** BWA → GATK HaplotypeCaller (GVCF mode) → GenotypeGVCFs

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { BWA_MEM } from './modules/alignment/bwa.nf'
include { GATK_MARKDUPLICATES; GATK_HAPLOTYPECALLER; GATK_GENOTYPEGVCFS } from './modules/variant_calling/gatk.nf'

params.samples = "data/multisample/samples.csv"
params.genome = "data/references/genome.fa"
params.outdir = "results/joint_calling"

workflow {
    // Read sample sheet
    samples_ch = Channel.fromPath(params.samples)
        .splitCsv(header: true)
        .map { row -> tuple(row.sample_id, [file(row.r1), file(row.r2)]) }
    
    genome = file(params.genome)
    index = file("${params.genome}.bwt")
    
    // Align all samples
    BWA_MEM(samples_ch, index)
    
    // Mark duplicates
    GATK_MARKDUPLICATES(BWA_MEM.out.bam)
    
    // Call variants in GVCF mode per sample
    GATK_HAPLOTYPECALLER(
        GATK_MARKDUPLICATES.out.bam,
        genome,
        Channel.empty(),  // no intervals
        true              // emit GVCF
    )
    
    // Collect all GVCFs
    all_gvcfs = GATK_HAPLOTYPECALLER.out.gvcf.map { it[1] }.collect()
    
    // Joint genotyping
    GATK_GENOTYPEGVCFS(
        all_gvcfs,
        genome,
        Channel.empty()  // no intervals
    )
}
```

---

## 20. Prokaryotic Genome Annotation

**Goal:** Complete bacterial genome annotation

**Workflow:** SPAdes → Prokka → BLAST functional annotation

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { SPADES_ASSEMBLE } from './modules/assembly/spades.nf'
include { PROKKA_ANNOTATE } from './modules/annotation/prokka.nf'
include { MAKEBLASTDB; BLASTP } from './modules/sequence_search/blast.nf'

params.reads = "data/bacterial/*_R{1,2}.fastq.gz"
params.protein_db = "data/databases/nr.faa"
params.genus = "Escherichia"
params.species = "coli"
params.outdir = "results/prokaryotic_annotation"

workflow {
    reads_ch = Channel.fromFilePairs(params.reads)
    
    // Assemble genome
    SPADES_ASSEMBLE(
        reads_ch,
        'isolate'  // mode: isolate, sc, meta, rna
    )
    
    // Annotate with Prokka
    PROKKA_ANNOTATE(
        SPADES_ASSEMBLE.out.contigs,
        'Bacteria',
        params.genus,
        params.species
    )
    
    // Functional annotation with BLAST
    protein_db = file(params.protein_db)
    MAKEBLASTDB(protein_db, 'nr_db', 'prot')
    
    proteins = PROKKA_ANNOTATE.out.proteins
    
    BLASTP(proteins, MAKEBLASTDB.out.blast_db)
}
```

---

## Summary Table

| Workflow | Key Modules | Primary Analysis | Typical Runtime |
|----------|-------------|------------------|-----------------|
| RNA-seq Bulk | STAR, featureCounts, DESeq2 | Differential expression | 2-4 hours |
| ChIP-seq | BWA, MACS2, HOMER | Peak calling, motifs | 1-3 hours |
| ATAC-seq | Bowtie2, MACS2, HOMER | Open chromatin | 1-2 hours |
| WGS Variants | BWA, GATK | Germline variants | 4-8 hours |
| Somatic Variants | GATK, VarScan, Manta | Tumor mutations | 6-12 hours |
| BS-seq | Bismark | DNA methylation | 4-8 hours |
| scRNA-seq | Cell Ranger, Seurat | Cell clustering | 2-6 hours |
| Metagenomics | Kraken2, MetaPhlAn | Taxonomic profiling | 1-3 hours |
| Metagenomic Assembly | MEGAHIT, Prokka | Assembly, annotation | 4-12 hours |
| Long-read Assembly | Flye, Racon | Genome assembly | 6-24 hours |
| Structural Variants | Manta, DELLY | Large rearrangements | 2-6 hours |
| Hi-C | HiC-Pro, Juicer | 3D genome | 4-12 hours |

---

## Best Practices

### 1. Parameter Management
Use configuration files:
```groovy
// nextflow.config
params {
    // Input/Output
    reads = "data/*_R{1,2}.fastq.gz"
    outdir = "results"
    
    // Resources
    max_cpus = 32
    max_memory = 128.GB
    
    // Container settings
    containers {
        rnaseq = "/path/to/rna-seq.sif"
        dnaseq = "/path/to/dna-seq.sif"
    }
}
```

### 2. Resource Allocation
```groovy
process {
    withName: STAR_ALIGN {
        cpus = 16
        memory = 64.GB
        time = 4.h
    }
    withName: GATK_HAPLOTYPECALLER {
        cpus = 4
        memory = 32.GB
        time = 12.h
    }
}
```

### 3. Error Handling
```groovy
process {
    errorStrategy = 'retry'
    maxRetries = 3
    
    withName: 'ASSEMBLY.*' {
        errorStrategy = 'ignore'  // Continue if assembly fails
    }
}
```

### 4. Executor Configuration
```groovy
executor {
    name = 'slurm'
    queueSize = 100
}

process {
    executor = 'slurm'
    queue = 'general'
    clusterOptions = '--account=myproject'
}
```

---

## Next Steps

1. **Test workflows** with real data
2. **Optimize parameters** for your specific needs
3. **Add quality controls** at each step
4. **Document results** with MultiQC
5. **Scale to production** with executor configuration

For questions or contributions, see the main BioPipelines documentation.

---

# New Analysis Types (Extended Patterns)

The following patterns cover emerging analysis types: spatial transcriptomics, long-read RNA-seq, and multi-omics integration.

---

## 21. Spatial Transcriptomics (10x Visium)

**Goal:** Analyze spatially-resolved gene expression from tissue sections

**Workflow:** Space Ranger → Scanpy/Squidpy → Spatial clustering → Cell type deconvolution

### Complete Pipeline

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// Import modules
include { SPACERANGER_COUNT } from './modules/spatial/spaceranger.nf'
include { SCANPY_SPATIAL } from './modules/scrna/scanpy.nf'
include { SQUIDPY_SPATIAL_ANALYSIS } from './modules/spatial/squidpy.nf'

// Parameters
params.fastq_dir = "data/visium/fastqs"
params.image = "data/visium/tissue_image.tif"
params.slide = "V19T26-123"
params.area = "A1"
params.transcriptome = "data/references/refdata-gex-GRCh38"
params.outdir = "results/spatial"

workflow {
    // Sample channels
    samples_ch = Channel.of(
        tuple("tissue_section1", file("${params.fastq_dir}/section1"))
    )
    
    image = file(params.image)
    transcriptome = file(params.transcriptome)
    
    // Space Ranger quantification
    SPACERANGER_COUNT(
        samples_ch,
        transcriptome,
        image,
        params.slide,
        params.area
    )
    
    // Spatial analysis with Scanpy
    SCANPY_SPATIAL(
        SPACERANGER_COUNT.out.filtered_matrix,
        SPACERANGER_COUNT.out.spatial_folder,
        200,   // min_genes
        5,     // max_mito_percent
        2000   // n_top_genes
    )
    
    // Spatial statistics with Squidpy
    SQUIDPY_SPATIAL_ANALYSIS(
        SCANPY_SPATIAL.out.adata,
        'leiden',       // cluster_key
        100,            // n_neighbors_spatial
        ['Moran_I', 'co_occurrence', 'neighborhood_enrichment']
    )
}
```

### Key Features
- **Tissue Imaging:** Integrates H&E images with gene expression
- **Spatial QC:** Filters spots by gene count and mitochondrial content
- **Spatial Statistics:** Moran's I, co-occurrence analysis
- **Visualization:** Interactive spatial plots

### Expected Outputs
- `results/spatial/spaceranger/` - Quantification matrices and spatial coordinates
- `results/spatial/scanpy/spatial_clusters.h5ad` - Clustered AnnData object
- `results/spatial/squidpy/spatial_stats.csv` - Spatial statistics
- `results/spatial/figures/` - Spatial visualizations

---

## 22. Slide-seq Analysis

**Goal:** Analyze near-cellular resolution spatial transcriptomics

**Workflow:** STAR alignment → Bead deconvolution → Spatial clustering

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { FASTQC } from './modules/qc/fastqc.nf'
include { STAR_INDEX; STAR_ALIGN } from './modules/alignment/star.nf'
include { PUCK_CALLER } from './modules/spatial/slideseq.nf'
include { SCANPY_SPATIAL_SLIDESEQ } from './modules/scrna/scanpy.nf'

params.reads = "data/slideseq/*_R{1,2}.fastq.gz"
params.puck_coords = "data/slideseq/puck_coordinates.csv"
params.genome = "data/references/genome.fa"
params.gtf = "data/references/genes.gtf"
params.outdir = "results/slideseq"

workflow {
    reads_ch = Channel.fromFilePairs(params.reads)
    genome = file(params.genome)
    gtf = file(params.gtf)
    puck_coords = file(params.puck_coords)
    
    // QC
    FASTQC(reads_ch)
    
    // Build index and align
    STAR_INDEX(genome, gtf)
    STAR_ALIGN(reads_ch, STAR_INDEX.out.index, gtf)
    
    // Assign reads to beads
    PUCK_CALLER(
        STAR_ALIGN.out.bam,
        puck_coords,
        10  // bead_diameter_um
    )
    
    // Spatial analysis
    SCANPY_SPATIAL_SLIDESEQ(
        PUCK_CALLER.out.count_matrix,
        puck_coords,
        50,   // min_genes
        1.0   // resolution
    )
}
```

---

## 23. Long-read RNA-seq (Nanopore Direct RNA)

**Goal:** Analyze full-length transcripts with RNA modifications

**Workflow:** minimap2 → Transcript assembly → Isoform quantification → m6A detection

### Complete Pipeline

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { NANOPLOT } from './modules/qc/nanoplot.nf'
include { MINIMAP2_INDEX; MINIMAP2_ALIGN } from './modules/alignment/minimap2.nf'
include { SAMTOOLS_SORT; SAMTOOLS_INDEX } from './modules/utilities/samtools.nf'
include { STRINGTIE_LONG } from './modules/assembly/stringtie.nf'
include { FLAIR_CORRECT; FLAIR_COLLAPSE; FLAIR_QUANTIFY } from './modules/long_read/flair.nf'
include { M6ANET_INFERENCE } from './modules/modification/m6anet.nf'

params.reads = "data/drna/*.fastq.gz"
params.genome = "data/references/genome.fa"
params.gtf = "data/references/genes.gtf"
params.outdir = "results/long_read_rna"

workflow {
    reads_ch = Channel.fromPath(params.reads)
        .map { file -> tuple(file.simpleName, file) }
    
    genome = file(params.genome)
    gtf = file(params.gtf)
    
    // QC with NanoPlot
    NANOPLOT(reads_ch)
    
    // Align with minimap2 (splice-aware mode)
    MINIMAP2_INDEX(genome)
    MINIMAP2_ALIGN(
        reads_ch,
        MINIMAP2_INDEX.out.index,
        'splice',  // preset for RNA-seq
        gtf
    )
    
    // Sort and index BAM
    SAMTOOLS_SORT(MINIMAP2_ALIGN.out.bam)
    SAMTOOLS_INDEX(SAMTOOLS_SORT.out.bam)
    
    bam_bai = SAMTOOLS_SORT.out.bam.join(SAMTOOLS_INDEX.out.bai)
    
    // Transcript assembly with StringTie
    STRINGTIE_LONG(bam_bai, gtf)
    
    // FLAIR for isoform analysis
    FLAIR_CORRECT(
        reads_ch,
        MINIMAP2_ALIGN.out.bam,
        genome,
        gtf
    )
    
    FLAIR_COLLAPSE(
        FLAIR_CORRECT.out.corrected,
        genome,
        gtf
    )
    
    FLAIR_QUANTIFY(
        FLAIR_COLLAPSE.out.isoforms,
        reads_ch.map { it[1] }.collect()
    )
    
    // RNA modification detection (m6A)
    M6ANET_INFERENCE(
        SAMTOOLS_SORT.out.bam,
        genome,
        reads_ch.map { it[1] }  // raw signal data
    )
}
```

### Key Features
- **Full-length Transcripts:** No PCR amplification artifacts
- **Isoform Discovery:** FLAIR identifies novel isoforms
- **RNA Modifications:** Direct detection of m6A and other modifications
- **No Fragmentation:** Better quantification of long transcripts

### Expected Outputs
- `results/long_read_rna/nanoplot/` - Read quality statistics
- `results/long_read_rna/flair/isoforms.gtf` - Novel isoform annotations
- `results/long_read_rna/flair/counts.tsv` - Isoform-level quantification
- `results/long_read_rna/m6anet/m6a_sites.csv` - Predicted modification sites

---

## 24. PacBio Iso-Seq Analysis

**Goal:** Full-length isoform sequencing with high accuracy

**Workflow:** ccs → lima → isoseq3 → minimap2 → SQANTI3

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { CCS } from './modules/long_read/pacbio.nf'
include { LIMA_DEMUX } from './modules/long_read/lima.nf'
include { ISOSEQ3_REFINE; ISOSEQ3_CLUSTER } from './modules/long_read/isoseq.nf'
include { MINIMAP2_ALIGN } from './modules/alignment/minimap2.nf'
include { SQANTI3_QC } from './modules/long_read/sqanti.nf'

params.subreads = "data/isoseq/*.subreads.bam"
params.primers = "data/isoseq/primers.fasta"
params.genome = "data/references/genome.fa"
params.gtf = "data/references/genes.gtf"
params.outdir = "results/isoseq"

workflow {
    subreads_ch = Channel.fromPath(params.subreads)
        .map { file -> tuple(file.simpleName, file) }
    
    primers = file(params.primers)
    genome = file(params.genome)
    gtf = file(params.gtf)
    
    // Generate CCS reads (HiFi)
    CCS(subreads_ch, 3)  // min_passes = 3
    
    // Demultiplex
    LIMA_DEMUX(CCS.out.ccs_bam, primers)
    
    // Refine (remove polyA, concatemers)
    ISOSEQ3_REFINE(LIMA_DEMUX.out.demuxed, primers)
    
    // Cluster to get isoforms
    ISOSEQ3_CLUSTER(ISOSEQ3_REFINE.out.flnc)
    
    // Align to genome
    MINIMAP2_ALIGN(
        ISOSEQ3_CLUSTER.out.polished,
        genome,
        'splice:hq',  // high-quality splice mode
        gtf
    )
    
    // Quality control with SQANTI3
    SQANTI3_QC(
        ISOSEQ3_CLUSTER.out.polished_fasta,
        gtf,
        genome,
        'IsoSeq'
    )
}
```

---

## 25. Multi-omics Integration (RNA-seq + ATAC-seq)

**Goal:** Integrate transcriptomic and chromatin accessibility data

**Workflow:** Parallel RNA/ATAC processing → Joint analysis → Regulatory network inference

### Complete Pipeline

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// RNA-seq modules
include { STAR_ALIGN } from './modules/alignment/star.nf'
include { FEATURECOUNTS } from './modules/quantification/featurecounts.nf'

// ATAC-seq modules
include { BOWTIE2_ALIGN } from './modules/alignment/bowtie2.nf'
include { MACS2_CALLPEAK } from './modules/peaks/macs2.nf'

// Integration modules
include { DESEQ2_DIFFERENTIAL } from './modules/analysis/deseq2.nf'
include { CHROMVAR_MOTIF_ACTIVITY } from './modules/epigenomics/chromvar.nf'
include { MOFA_INTEGRATE } from './modules/integration/mofa.nf'

params.rna_reads = "data/multiomics/rna/*_R{1,2}.fastq.gz"
params.atac_reads = "data/multiomics/atac/*_R{1,2}.fastq.gz"
params.genome = "data/references/genome.fa"
params.gtf = "data/references/genes.gtf"
params.outdir = "results/multiomics"

workflow {
    rna_reads = Channel.fromFilePairs(params.rna_reads)
    atac_reads = Channel.fromFilePairs(params.atac_reads)
    genome = file(params.genome)
    gtf = file(params.gtf)
    
    // ========== RNA-seq Branch ==========
    // Align RNA-seq
    star_index = file("data/references/star_index")
    STAR_ALIGN(rna_reads, star_index, gtf)
    
    // Quantify
    FEATURECOUNTS(STAR_ALIGN.out.bam, gtf, 'gene_id', true)
    
    // Differential expression
    count_matrix = FEATURECOUNTS.out.counts.collect()
    samplesheet = file("data/multiomics/samplesheet.csv")
    
    DESEQ2_DIFFERENTIAL(count_matrix, samplesheet, 'condition', 'control', 'treatment')
    
    // ========== ATAC-seq Branch ==========
    // Align ATAC-seq
    bowtie_index = file("data/references/bowtie2_index")
    BOWTIE2_ALIGN(atac_reads, bowtie_index)
    
    // Call peaks
    MACS2_CALLPEAK(BOWTIE2_ALIGN.out.bam, Channel.empty(), 'narrow', 'hs')
    
    // Motif activity
    CHROMVAR_MOTIF_ACTIVITY(
        BOWTIE2_ALIGN.out.bam.collect(),
        MACS2_CALLPEAK.out.peaks.collect(),
        'JASPAR2020'
    )
    
    // ========== Integration ==========
    // Prepare data matrices
    rna_matrix = FEATURECOUNTS.out.counts
    atac_matrix = CHROMVAR_MOTIF_ACTIVITY.out.deviation_scores
    
    // Multi-omics factor analysis
    MOFA_INTEGRATE(
        rna_matrix,
        atac_matrix,
        samplesheet,
        15  // num_factors
    )
}
```

### Key Features
- **Parallel Processing:** RNA and ATAC processed independently
- **Peak-Gene Linking:** Associates accessible regions with nearby genes
- **Latent Factor Discovery:** MOFA finds shared biological signals
- **Regulatory Inference:** Links TF motifs to gene expression changes

### Expected Outputs
- `results/multiomics/rna/deseq2/` - Differential expression results
- `results/multiomics/atac/peaks/` - Called peaks
- `results/multiomics/chromvar/` - Motif activity scores
- `results/multiomics/mofa/factors.h5` - Integrated latent factors

---

## 26. Single-cell Multiome (CITE-seq)

**Goal:** Analyze joint RNA and surface protein expression in single cells

**Workflow:** Cell Ranger → CITE-seq-Count → Seurat/totalVI

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { CELLRANGER_MULTI } from './modules/scrna/cellranger.nf'
include { CITE_SEQ_COUNT } from './modules/scrna/citeseq.nf'
include { SEURAT_MULTIMODAL } from './modules/scrna/seurat.nf'
include { TOTALVI_INTEGRATE } from './modules/scrna/scvi.nf'

params.fastq_dir = "data/citeseq/fastqs"
params.feature_ref = "data/citeseq/feature_reference.csv"
params.transcriptome = "data/references/refdata-gex-GRCh38"
params.outdir = "results/citeseq"

workflow {
    sample = "sample1"
    fastq_dir = file(params.fastq_dir)
    feature_ref = file(params.feature_ref)
    transcriptome = file(params.transcriptome)
    
    // Cell Ranger multi for RNA + antibody
    CELLRANGER_MULTI(
        sample,
        fastq_dir,
        transcriptome,
        feature_ref,
        'Antibody Capture'
    )
    
    // Alternative: CITE-seq-Count for antibody tags
    CITE_SEQ_COUNT(
        file("${params.fastq_dir}/*_R2.fastq.gz"),
        feature_ref,
        CELLRANGER_MULTI.out.barcodes
    )
    
    // Seurat multimodal analysis
    SEURAT_MULTIMODAL(
        CELLRANGER_MULTI.out.rna_matrix,
        CELLRANGER_MULTI.out.adt_matrix,
        200,    // min_genes
        0.5,    // resolution
        'WNN'   // weighted nearest neighbor
    )
    
    // totalVI for joint embedding
    TOTALVI_INTEGRATE(
        CELLRANGER_MULTI.out.rna_matrix,
        CELLRANGER_MULTI.out.adt_matrix,
        20,     // n_latent
        400     // n_epochs
    )
}
```

---

## 27. 10x Multiome (RNA + ATAC)

**Goal:** Analyze gene expression and chromatin accessibility from the same cells

**Workflow:** Cell Ranger ARC → Signac → Joint clustering

```nextflow
#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { CELLRANGER_ARC_COUNT } from './modules/scrna/cellranger_arc.nf'
include { SIGNAC_CREATE; SIGNAC_LSI; SIGNAC_PEAKS } from './modules/scrna/signac.nf'
include { SEURAT_WNN } from './modules/scrna/seurat.nf'
include { CICERO_COACCESS } from './modules/scrna/cicero.nf'

params.fastq_dir = "data/multiome/fastqs"
params.reference = "data/references/refdata-cellranger-arc-GRCh38"
params.outdir = "results/multiome"

workflow {
    sample = "sample1"
    fastq_dir = file(params.fastq_dir)
    reference = file(params.reference)
    
    // Cell Ranger ARC
    CELLRANGER_ARC_COUNT(
        sample,
        fastq_dir,
        reference
    )
    
    // Create Signac object for ATAC
    SIGNAC_CREATE(
        CELLRANGER_ARC_COUNT.out.atac_fragments,
        CELLRANGER_ARC_COUNT.out.peaks,
        CELLRANGER_ARC_COUNT.out.rna_matrix
    )
    
    // LSI dimension reduction for ATAC
    SIGNAC_LSI(
        SIGNAC_CREATE.out.signac_obj,
        50  // n_components
    )
    
    // Weighted nearest neighbor clustering
    SEURAT_WNN(
        CELLRANGER_ARC_COUNT.out.rna_matrix,
        SIGNAC_LSI.out.lsi_embeddings,
        0.5  // resolution
    )
    
    // Co-accessibility with Cicero
    CICERO_COACCESS(
        SIGNAC_CREATE.out.signac_obj,
        SEURAT_WNN.out.clusters,
        file(params.genome_annotation)
    )
}
```

---

## Extended Summary Table

| Workflow | Key Modules | Primary Analysis | Typical Runtime |
|----------|-------------|------------------|-----------------|
| **Spatial (Visium)** | Space Ranger, Scanpy, Squidpy | Spatial clustering, statistics | 2-4 hours |
| **Slide-seq** | STAR, Scanpy | Near-cellular spatial | 3-5 hours |
| **Long-read RNA (Nanopore)** | minimap2, FLAIR, m6Anet | Isoforms, modifications | 4-8 hours |
| **Iso-Seq** | isoseq3, SQANTI3 | Full-length transcripts | 6-12 hours |
| **Multi-omics (RNA+ATAC)** | STAR, Bowtie2, MOFA | Integrated factors | 4-8 hours |
| **CITE-seq** | Cell Ranger, Seurat | RNA + protein | 3-6 hours |
| **10x Multiome** | Cell Ranger ARC, Signac | RNA + ATAC same cell | 4-8 hours |

---

## Analysis Type Quick Reference

### Spatial Transcriptomics Keywords
- "visium", "spatial", "tissue section", "spot deconvolution", "squidpy"
- "slide-seq", "puck", "bead array", "near-cellular"
- "xenium", "in situ", "subcellular"

### Long-read RNA-seq Keywords
- "nanopore rna", "direct rna", "drna", "isoseq", "full-length"
- "isoform quantification", "novel isoform", "alternative splicing"
- "m6a", "rna modification", "epitranscriptome"

### Multi-omics Keywords
- "integration", "multi-omics", "multimodal", "joint embedding"
- "cite-seq", "totalvi", "multiome", "wnn"
- "rna-atac", "chromatin-expression", "regulatory network"

---

For questions or contributions, see the main BioPipelines documentation.
