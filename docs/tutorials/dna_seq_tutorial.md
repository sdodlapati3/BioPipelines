# DNA-seq Variant Calling Pipeline Tutorial

## Table of Contents
1. [Introduction](#introduction)
2. [Biological Background](#biological-background)
3. [Pipeline Overview](#pipeline-overview)
4. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
5. [Understanding the Output](#understanding-the-output)
6. [Running the Pipeline](#running-the-pipeline)

## Introduction

This tutorial will guide you through the DNA-seq variant calling pipeline, designed to identify genetic variations (mutations, SNPs, indels) from whole genome or whole exome sequencing data.

### What You'll Learn
- What DNA sequencing is and why we use it
- How to process raw sequencing reads into variant calls
- What each tool does and why it's important
- How to interpret the results

### Prerequisites
- Basic understanding of DNA and genetics
- Familiarity with command line basics
- Access to the BioPipelines environment

## Biological Background

### What is DNA Sequencing?

**DNA sequencing** is the process of determining the order of nucleotides (A, T, G, C) in a DNA molecule. Modern sequencing technologies can read millions of short DNA fragments in parallel.

### Why Variant Calling?

**Variant calling** identifies differences between a sample's genome and a reference genome. These variants can be:
- **SNPs (Single Nucleotide Polymorphisms)**: Single base changes (e.g., A→G)
- **Indels**: Small insertions or deletions
- **Structural variants**: Large-scale genomic changes

### Clinical Applications
- Disease gene discovery
- Cancer genomics (identifying mutations)
- Pharmacogenomics (drug response prediction)
- Population genetics studies

## Pipeline Overview

### The Complete Workflow

```
┌────────────────────────────────────────────────────────────────┐
│                  DNA-Seq Variant Calling Pipeline               │
└────────────────────────────────────────────────────────────────┘

Raw FASTQ Files (sample_R1.fastq.gz, sample_R2.fastq.gz)
    │
    ├──► [1] QUALITY CONTROL (FastQC + MultiQC)
    │         └─► HTML reports with quality metrics
    │
    ├──► [2] READ TRIMMING (fastp)
    │         └─► Clean FASTQ files
    │
    ├──► [3] ALIGNMENT TO REFERENCE (BWA-MEM)
    │         └─► Mapped reads in BAM format
    │              
    │              FASTQ Format:           BAM Format:
    │              @read_1                 Read aligned to
    │              ATCGATCG                chr1:12345
    │              +                       with CIGAR string
    │              IIIIIIII                showing matches
    │
    ├──► [4] MARK DUPLICATES (Picard)
    │         └─► Deduplicated BAM
    │              (PCR duplicates flagged)
    │
    ├──► [5] BASE QUALITY RECALIBRATION (GATK BQSR)
    │         └─► Recalibrated BAM
    │              (Quality scores adjusted)
    │
    ├──► [6] VARIANT CALLING (GATK HaplotypeCaller)
    │         └─► Raw VCF file
    │              
    │              VCF Format:
    │              #CHROM POS REF ALT QUAL FILTER INFO
    │              chr1   100 A   G   30   PASS   DP=50
    │              chr1   200 C   T   45   PASS   DP=60
    │
    ├──► [7] VARIANT FILTERING (GATK VariantFiltration)
    │         └─► Filtered VCF
    │              (Low quality variants marked)
    │
    └──► [8] ANNOTATION (SnpEff)
              └─► Annotated VCF + summary report
                   (Functional impact predicted)

Final Output:
    • filtered_variants.vcf.gz  (1-2M variants)
    • annotated_variants.vcf.gz (with gene/effect info)
    • MultiQC reports (quality metrics)
```

**Key File Formats**:

| Format | Description | Example Content |
|--------|-------------|-----------------|
| **FASTQ** | Raw sequencing reads with quality scores | `@read1\nATCG\n+\nIIII` |
| **BAM** | Binary alignment (mapped reads) | Compressed; view with `samtools view` |
| **VCF** | Variant Call Format (mutations) | Tab-delimited: position, ref, alt alleles |

### Time Estimates
- Small dataset (exome): 2-4 hours
- Whole genome: 8-24 hours (depends on coverage)

## Step-by-Step Walkthrough

### Step 1: Quality Control with FastQC

**Purpose**: Assess the quality of raw sequencing reads before processing.

**What it does**:
- Checks base quality scores across reads
- Identifies adapter contamination
- Detects over-represented sequences
- Measures GC content distribution

**Command breakdown**:
```bash
fastqc -t 4 -o output_dir input_R1.fastq.gz input_R2.fastq.gz
```
- `-t 4`: Use 4 threads for parallel processing
- `-o`: Output directory for reports
- Input files: Paired-end reads (R1 and R2)

**Output**: HTML reports with quality metrics and graphs

**What to look for**:
- ✅ Per base quality scores >30 (Phred score)
- ✅ No adapter contamination
- ⚠️ If quality is poor, more aggressive trimming may be needed

---

### Step 2: Read Trimming with fastp

**Purpose**: Remove low-quality bases and adapter sequences.

**Why it's important**:
- Low-quality bases cause alignment errors
- Adapters are artificial sequences from library prep
- Improves downstream analysis accuracy

**Command breakdown**:
```bash
fastp \
    -i input_R1.fastq.gz \
    -I input_R2.fastq.gz \
    -o output_R1.fastq.gz \
    -O output_R2.fastq.gz \
    --thread 4 \
    --qualified_quality_phred 20 \
    --length_required 50
```

**Parameters explained**:
- `-i/-I`: Input paired-end reads
- `-o/-O`: Output trimmed reads
- `--qualified_quality_phred 20`: Remove bases with quality <20
- `--length_required 50`: Discard reads shorter than 50bp after trimming
- `--thread 4`: Use 4 CPU cores

**Output**:
- Trimmed FASTQ files
- JSON report with statistics (bases trimmed, reads filtered)

---

### Step 3: Alignment with BWA-MEM

**Purpose**: Map reads to a reference genome to find their genomic locations.

**The Algorithm**:
BWA-MEM uses the Burrows-Wheeler Transform to quickly find where each read best matches the reference genome.

**Command breakdown**:
```bash
bwa mem -t 8 \
    -R '@RG\tID:sample1\tSM:sample1\tPL:ILLUMINA' \
    reference.fa \
    input_R1.fastq.gz input_R2.fastq.gz \
    | samtools sort -@ 4 -o output.bam
```

**Parameters explained**:
- `-t 8`: Use 8 threads
- `-R`: Read group information (required for GATK):
  - `ID`: Unique identifier for this sequencing run
  - `SM`: Sample name
  - `PL`: Sequencing platform (ILLUMINA)
- `reference.fa`: Reference genome (hg38 for human)
- Pipe `|` to samtools to immediately sort the output

**Output**: 
- **BAM file**: Binary format storing aligned reads
- Each read has a position, quality score, and CIGAR string (describes alignment)

**Key concepts**:

**MAPQ score** (Mapping Quality): Phred-scaled probability that mapping is wrong
- MAPQ=60: 99.9999% confident (1 in 1,000,000 chance of wrong position)
- MAPQ=40: 99.99% confident
- MAPQ=20: 99% confident
- MAPQ=0: Reads map to multiple locations (ambiguous)

**CIGAR string**: Compact representation of alignment
```
Read:      ATCG--ATCG (with 2bp deletion)
Reference: ATCGAAATCG

CIGAR: 4M2D4M
       ││ │ │
       ││ │ └─ 4 Matches
       ││ └─── 2 Deletions
       │└───── 4 Matches
       └────── Start
```

**Visual Example**:
```
Reference genome (chr1):
Position: 1000      1010      1020
          |---------|---------|
          ATCGATCGATCGATCGATCG

Read alignment:
Read1: chr1:1001  MAPQ=60  CIGAR=10M
       ATCGATCGAT (perfect match)
       ||||||||||
       ATCGATCGAT

Read2: chr1:1005  MAPQ=20  CIGAR=5M1I5M  
       ATCG[A]ATCGAT (1bp insertion)
       ||||^|||||
       ATCG-ATCG
       
BAM stores: chromosome, position, CIGAR, sequence, quality
```

---

### Step 4: Mark Duplicates with Picard

**Purpose**: Identify PCR duplicates that arise during library preparation.

**Why remove duplicates?**
- PCR duplicates are artificial copies of the same DNA molecule
- They can cause overestimation of coverage
- Can create false variant calls
- We mark (not remove) them so they're ignored in variant calling

**Visual Concept**:
```
Original DNA fragment:  ────ATCG────
                             
Library prep with PCR:  ────ATCG──── (original)
                        ────ATCG──── (PCR copy 1)
                        ────ATCG──── (PCR copy 2)
                        ────ATCG──── (PCR copy 3)

After sequencing, all 4 reads map to same position.
Picard marks 3 as duplicates, keeps 1 for analysis.

Real coverage = 1x (not 4x)
```

**Command breakdown**:
```bash
gatk MarkDuplicates \
    -I input.bam \
    -O output.dedup.bam \
    -M metrics.txt \
    --CREATE_INDEX true
```

**Parameters explained**:
- `-I`: Input BAM file
- `-O`: Output BAM with duplicates marked
- `-M`: Metrics file showing duplication rate
- `--CREATE_INDEX`: Create BAM index (.bai) for fast access

**Output**:
- Deduplicated BAM file
- Metrics showing % duplicates (typically 5-20% is normal)

**What's a good duplication rate?**
- <10%: Excellent library quality
- 10-20%: Normal
- >30%: May indicate over-amplification

---

### Step 5: Base Quality Score Recalibration (BQSR)

**Purpose**: Correct systematic errors in base quality scores from the sequencer.

**Why it matters**:
- Sequencers sometimes over/under-estimate quality scores
- BQSR learns error patterns and adjusts scores
- Improves variant calling accuracy

**Visual concept**:
```
Before BQSR:              After BQSR:
Quality scores may        Corrected scores
be systematically         better reflect
inflated/deflated         true error rates

Read: ATCGATCG            Read: ATCGATCG
Qual: 40404040            Qual: 38383838
      ^^^^^^^^                  ^^^^^^^^
      May be too high           More accurate
```

**Two-step process**:

**Step 5a: BaseRecalibrator**
```bash
gatk BaseRecalibrator \
    -R reference.fa \
    -I input.dedup.bam \
    --known-sites dbsnp.vcf.gz \
    -O recal_data.table
```

**Parameters explained**:
- `-R`: Reference genome
- `-I`: Input BAM
- `--known-sites`: VCF of known variants (dbSNP) - used to distinguish real variants from errors
- `-O`: Recalibration table

**⚠️ Important: Chromosome Naming Compatibility**

Your reference genome and dbSNP file **must use the same chromosome naming convention**:

| Reference Style | dbSNP File Required | Example |
|----------------|---------------------|---------|
| **UCSC** (chr1, chr2, ...) | Needs `chr` prefix | `Homo_sapiens_assembly38.dbsnp138.vcf.gz` |
| **NCBI** (1, 2, ...) | No `chr` prefix | `dbsnp_155.hg38.vcf.gz` |

**Common Error**:
```
ERROR: Input files reference and features have incompatible contigs:
No overlapping contigs found.
  reference contigs = [chr1, chr2, chr3, ...]
  features contigs = [NC_000001.11, NC_000002.12, ...]
```

**Fix**: Use matching dbSNP version. For hg38 with `chr` prefixes:
```yaml
# config.yaml
reference:
  genome: "/path/to/hg38.fa"  # Uses chr1, chr2, ...
  known_sites: "/path/to/Homo_sapiens_assembly38.dbsnp138.vcf.gz"  # Also uses chr prefixes
```

**Step 5b: ApplyBQSR**
```bash
gatk ApplyBQSR \
    -R reference.fa \
    -I input.dedup.bam \
    --bqsr-recal-file recal_data.table \
    -O output.recal.bam
```

Applies the recalibration model to adjust quality scores.

**Output**: Recalibrated BAM file ready for variant calling

---

### Step 6: Variant Calling with GATK HaplotypeCaller

**Purpose**: Identify genetic variants from aligned reads.

**How it works**:
1. Identifies regions with potential variants (active regions)
2. Reassembles reads in these regions
3. Calls SNPs and indels using a Bayesian model
4. Assigns quality scores to each variant

**Command breakdown**:
```bash
gatk HaplotypeCaller \
    -R reference.fa \
    -I input.recal.bam \
    -O output.raw.vcf.gz \
    --native-pair-hmm-threads 4
```

**Parameters explained**:
- `-R`: Reference genome
- `-I`: Recalibrated BAM
- `-O`: Output VCF (Variant Call Format) - compressed
- `--native-pair-hmm-threads`: Parallel processing for the HMM algorithm

**Output**: Raw VCF file with all variant calls

**VCF fields explained**:
- **CHROM**: Chromosome
- **POS**: Position on chromosome
- **REF**: Reference allele
- **ALT**: Alternate allele (the variant)
- **QUAL**: Quality score (Phred-scaled)
- **FILTER**: PASS or reason for filtering
- **FORMAT/INFO**: Detailed genotype information

---

### Step 7: Variant Filtering

**Purpose**: Remove low-quality variant calls.

**Why filter?**
- Not all variant calls are real
- Sequencing errors can look like variants
- Low-quality variants can lead to false discoveries

**Filtering Strategy**:

```
Raw VCF (all variant calls)
    │
    ├─► Quality Depth (QD) < 2.0 ────► FAIL (low_QD)
    │
    ├─► Fisher Strand (FS) > 60.0 ───► FAIL (strand_bias)
    │
    ├─► Mapping Quality (MQ) < 40 ───► FAIL (low_MQ)
    │
    ├─► Strand Odds Ratio (SOR) > 3 ─► FAIL (strand_bias)
    │
    └─► All filters passed ──────────► PASS (high confidence)
    
Filtered VCF (high-confidence variants only)
```

**Filter Thresholds Explained**:

| Filter | Threshold | Meaning | Why Important |
|--------|-----------|---------|---------------|
| **QD** (Quality by Depth) | < 2.0 | Low quality per read covering position | Removes calls with weak evidence |
| **FS** (Fisher Strand) | > 60.0 | Variants only on one strand | Indicates strand bias (technical artifact) |
| **MQ** (Mapping Quality) | < 40.0 | Reads mapped poorly | Removes variants in repeat regions |
| **SOR** (Strand Odds Ratio) | > 3.0 | Imbalanced strand representation | Another strand bias metric |

**Visual Example**:
```
Position 12345:
Raw call: A→G, QUAL=25, QD=1.8, FS=65

Decision tree:
  QD=1.8 < 2.0?  YES ──► Mark as "low_QD"
  FS=65 > 60?    YES ──► Mark as "strand_bias"
  
Result: FILTER column = "low_QD;strand_bias" (FAIL)
- Filtering improves specificity (reduces false positives)

**Command breakdown**:
```bash
gatk VariantFiltration \
    -V input.raw.vcf.gz \
    -O output.filtered.vcf.gz \
    --filter-expression "QD < 2.0 || FS > 60.0 || MQ < 40.0" \
    --filter-name "basic_snp_filter"
```

**Filter criteria explained**:
- **QD < 2.0**: Quality by Depth - variant quality normalized by depth
- **FS > 60.0**: Fisher Strand - strand bias (variants should appear on both strands)
- **MQ < 40.0**: Mapping Quality - how well reads align
- **MQRankSum < -12.5**: Mapping quality difference between ref and alt alleles
- **ReadPosRankSum < -8.0**: Position of variants within reads

**Output**: Filtered VCF with PASS/FAIL annotations

---

### Step 8: Variant Annotation with SnpEff

**Purpose**: Add biological context to variants.

**What annotations include**:
- Gene name
- Effect on protein (synonymous, missense, nonsense)
- Predicted impact (HIGH, MODERATE, LOW, MODIFIER)
- Known disease associations
- Population frequency

**Command breakdown**:
```bash
snpEff -v hg38 input.filtered.vcf.gz \
    -stats stats.html | \
    bgzip > output.annotated.vcf.gz
```

**Parameters explained**:
- `-v`: Verbose output
- `hg38`: Genome version (human reference build 38)
- `-stats`: HTML report with annotation summary
- `bgzip`: Compress output

**Output**: 
- Annotated VCF with detailed information
- HTML summary report

**Annotation fields**:
- **ANN**: Annotation (gene, effect, impact)
- **LOF**: Loss of function variants
- **NMD**: Nonsense-mediated decay

---

## Understanding the Output

### Key Output Files

#### 1. **sample1.annotated.vcf.gz**
Final variant calls with annotations.

**Important fields**:
```
##INFO=<ID=ANN,Description="Functional annotations">
#CHROM  POS     ID      REF     ALT     QUAL    FILTER  INFO    FORMAT  SAMPLE
chr1    12345   .       A       G       1234.5  PASS    ANN=...  GT:AD:DP  0/1:25,30:55
```

- **GT (Genotype)**: 0/0=homozygous ref, 0/1=heterozygous, 1/1=homozygous alt
- **AD (Allele Depth)**: Reads supporting ref, alt
- **DP (Depth)**: Total read depth

#### 2. **MultiQC Report**
Aggregated QC metrics across all samples.

#### 3. **Picard Metrics**
Duplication rates, insert size distributions.

### Interpreting Results

**High-impact variants to prioritize**:
1. **Stop gained/lost**: Creates/removes stop codons
2. **Frameshift**: Changes protein reading frame
3. **Splice site**: Affects RNA splicing
4. **Missense**: Changes amino acid (check prediction tools)

**Filtering recommendations**:
- QUAL > 30: Good quality
- DP > 10: Adequate coverage
- GQ > 20: Good genotype quality

---

## Running the Pipeline

### Quick Start

1. **Prepare your data**:
```bash
cd ~/BioPipelines/pipelines/dna_seq/variant_calling
```

2. **Edit config.yaml**:
```yaml
samples:
  - sample1
  - sample2

reference:
  genome: "/path/to/hg38.fa"
  known_sites: "/path/to/dbsnp.vcf.gz"
```

3. **Submit the job**:
```bash
sbatch ~/BioPipelines/scripts/submit_dna_seq.sh
```

4. **Monitor progress**:
```bash
squeue --me
tail -f slurm_*.err
```

### Troubleshooting

**Common issues**:

1. **"MissingOutputException"**
   - Check disk space
   - Verify input files exist
   - Look at error logs

2. **Low variant calling rate**
   - Check coverage (should be >20x for WES, >30x for WGS)
   - Verify reference genome version matches

3. **High duplication rate**
   - May be normal for targeted sequencing
   - Check library quality

### Expected Runtime
- **Whole Exome**: 2-4 hours
- **Whole Genome (30x)**: 12-24 hours

---

## Additional Resources

### Further Reading
- [GATK Best Practices](https://gatk.broadinstitute.org/hc/en-us/articles/360035535932)
- [VCF Format Specification](https://samtools.github.io/hts-specs/VCFv4.2.pdf)
- [SnpEff Documentation](https://pcingola.github.io/SnpEff/)

### Tools Used
- **FastQC**: Quality control
- **fastp**: Read trimming
- **BWA**: Alignment
- **GATK**: Variant calling suite
- **SnpEff**: Variant annotation

---

## Glossary

- **BAM**: Binary Alignment Map - compressed SAM format
- **FASTQ**: Text format for storing sequences and quality scores
- **Phred Score**: Quality score (Q30 = 99.9% accuracy)
- **Read Depth**: Number of reads covering a position
- **VCF**: Variant Call Format - standard for storing variants
- **Indel**: Insertion or deletion mutation
- **SNP**: Single Nucleotide Polymorphism

---

*Last updated: November 2025*
