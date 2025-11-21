# ChIP-seq Peak Calling Pipeline Tutorial

## Table of Contents
1. [Introduction](#introduction)
2. [Biological Background](#biological-background)
3. [Pipeline Overview](#pipeline-overview)
4. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
5. [Understanding the Output](#understanding-the-output)
6. [Running the Pipeline](#running the-pipeline)

## Introduction

This tutorial covers ChIP-seq (Chromatin Immunoprecipitation followed by sequencing), a technique to identify where proteins bind to DNA genome-wide.

### What You'll Learn
- What ChIP-seq is and why it's used
- How to process ChIP-seq data to find binding sites
- How to identify significant peaks
- How to perform motif analysis

### Prerequisites
- Basic understanding of DNA-protein interactions
- Familiarity with genomics concepts
- Access to BioPipelines environment

## Biological Background

### What is ChIP-seq?

**ChIP-seq** maps protein-DNA interactions across the entire genome. It can identify:
- Transcription factor binding sites
- Histone modifications
- Chromatin remodeling proteins

### The ChIP-seq Process

```
1. Cross-link proteins to DNA (formaldehyde)
2. Fragment DNA (sonication)
3. Immunoprecipitate with specific antibody
4. Reverse cross-links and purify DNA
5. Sequence the enriched fragments
6. Map reads to genome
7. Find "peaks" (enriched regions)
```

### Types of ChIP-seq Experiments

**1. Transcription Factor ChIP-seq**
- Sharp peaks (100-300 bp)
- Identifies regulatory elements
- Examples: CTCF, p53, STAT3

**2. Histone Modification ChIP-seq**
- Broad peaks (1-100 kb)
- Marks active/repressive chromatin
- Examples: H3K4me3 (promoters), H3K27ac (enhancers)

### Why ChIP-seq?

**Applications**:
- Map transcription factor binding sites
- Identify enhancers and promoters
- Study epigenetic regulation
- Understand gene regulation
- Disease mechanisms (cancer, development)

### Key Concept: Peak vs. Background

**Peak**: Region with significantly more reads than expected by chance
- Indicates protein binding
- Height correlates with binding strength

**Background**: Random DNA fragments in input control
- Accounts for sequencing bias
- Open chromatin regions

## Pipeline Overview

### The Complete Workflow

```
Raw Reads (FASTQ)
    ↓
1. Quality Control (FastQC)
    ↓
2. Read Trimming (fastp)
    ↓
3. Alignment (BWA/Bowtie2)
    ↓
4. Remove Duplicates (Picard)
    ↓
5. Peak Calling (MACS2)
    ↓
6. Peak Annotation
    ↓
7. Motif Analysis (HOMER)
    ↓
8. Visualization (BigWig)
    ↓
Final Peak Lists + Motifs
```

### Key Differences from Other Seq Types
- Requires **input control** (no antibody)
- Focuses on **enriched regions** (peaks), not individual bases
- Single-end sequencing often sufficient
- Shorter read lengths acceptable (50bp)

### Time Estimates
- 3 samples + input: 2-4 hours
- Peak calling is fast (<15 min per sample)

## Step-by-Step Walkthrough

### Step 1: Quality Control with FastQC

**Purpose**: Assess read quality.

**ChIP-seq specific checks**:
- ✅ Sequence duplication levels can be higher than RNA-seq
- ✅ Some bias at read start is normal (fragmentation bias)
- ⚠️ Check for adapter contamination

**Command**:
```bash
fastqc -t 4 -o qc_output/ sample.fastq.gz input.fastq.gz
```

**What to look for**:
- Quality scores >28
- Sequence length distribution
- GC content (should match genome)

---

### Step 2: Read Trimming with fastp

**Purpose**: Remove adapters and low-quality bases.

**ChIP-seq considerations**:
- Can be more aggressive with quality trimming
- Shorter reads (>25bp) still useful

**Command**:
```bash
fastp \
    -i input.fastq.gz \
    -o trimmed.fastq.gz \
    --qualified_quality_phred 20 \
    --length_required 25 \
    --thread 4
```

**Output**: Clean reads for alignment

---

### Step 3: Alignment with Bowtie2

**Purpose**: Map reads to reference genome.

**Why Bowtie2 for ChIP-seq?**
- Fast for single-end reads
- Handles short reads well
- No need for splice-aware alignment

**Command breakdown**:
```bash
bowtie2 \
    -x reference_index \
    -U input.fastq.gz \
    -p 8 \
    --very-sensitive \
    | samtools sort -@ 4 -o output.bam
```

**Parameters explained**:
- `-x`: Bowtie2 index of reference genome
- `-U`: Single-end reads (use `-1/-2` for paired-end)
- `-p 8`: Use 8 threads
- `--very-sensitive`: More accurate alignment
- Pipe to samtools to sort immediately

**Alternative: BWA**
```bash
bwa mem -t 8 reference.fa input.fastq.gz \
    | samtools sort -@ 4 -o output.bam
```

**Good alignment rates**:
- >80% aligned
- <10% multimapping (MAPQ=0)

**Output**: Sorted BAM file

---

### Step 4: Mark Duplicates with Picard

**Purpose**: Identify PCR duplicates.

**Why remove duplicates in ChIP-seq?**
- PCR amplification creates artificial copies
- Can inflate peak signals
- Especially important for low-input ChIP

**Command**:
```bash
gatk MarkDuplicates \
    -I input.bam \
    -O deduplicated.bam \
    -M metrics.txt \
    --REMOVE_DUPLICATES true
```

**Parameters**:
- `--REMOVE_DUPLICATES true`: Actually remove (vs. just mark)

**ChIP-seq duplication rates**:
- 10-30%: Normal
- >50%: Poor library complexity, consider more cells

**Output**: Deduplicated BAM

---

### Step 5: Peak Calling with MACS2

**Purpose**: Identify enriched regions (peaks) where protein binds.

**MACS2 Algorithm**:
1. Model fragment size from read distribution
2. Shift reads to fragment centers
3. Calculate local enrichment (ChIP vs. input)
4. Call peaks using Poisson p-value
5. Filter by FDR (q-value)

**Command breakdown**:
```bash
macs2 callpeak \
    -t ChIP.bam \
    -c input.bam \
    -f BAM \
    -g hs \
    -n sample1 \
    --outdir peaks \
    -q 0.05
```

**Parameters explained**:
- `-t`: Treatment (ChIP) BAM file
- `-c`: Control (input) BAM file
- `-f BAM`: Input format
- `-g hs`: Genome size (hs=human, mm=mouse, or provide number)
- `-n`: Sample name prefix
- `-q 0.05`: FDR cutoff (q-value threshold)

**For broad peaks (histones)**:
```bash
macs2 callpeak \
    -t ChIP.bam \
    -c input.bam \
    -g hs \
    -n sample1 \
    --broad \
    --broad-cutoff 0.05
```

**Understanding MACS2 output**:

**1. _peaks.narrowPeak** (transcription factors):
```
chr1  1000  1300  peak1  250  .  10.5  25.2  -1  50
```
Columns:
- chr, start, end: Peak location
- name: Peak ID
- score: Integer score
- strand: Not used (.)
- signalValue: Fold enrichment
- pValue: -log10(p-value)
- qValue: -log10(q-value)
- peak: Relative position of summit

**2. _peaks.broadPeak** (histone marks):
Similar format but for wide regions

**3. _summits.bed**:
Single base position of peak summit (highest point)

**4. _peaks.xls**:
Tab-delimited detailed information

---

### Step 6: Peak Annotation

**Purpose**: Determine genomic context of peaks (promoter, enhancer, intergenic).

**Using ChIPseeker (R)**:
```R
library(ChIPseeker)

# Load peaks
peaks <- readPeakFile("sample1_peaks.narrowPeak")

# Annotate
peakAnno <- annotatePeak(
  peaks,
  tssRegion = c(-3000, 3000),
  TxDb = TxDb.Hsapiens.UCSC.hg38.knownGene
)

# View distribution
plotAnnoPie(peakAnno)
```

**Annotation categories**:
- **Promoter**: Within 3kb of TSS (transcription start site)
- **5' UTR**: Before coding sequence
- **3' UTR**: After coding sequence
- **Exon/Intron**: Within gene body
- **Intergenic**: Between genes

**Output visualization**:
- Pie chart of genomic distribution
- Distance to TSS distribution
- Peak-gene associations

---

### Step 7: Motif Analysis with HOMER

**Purpose**: Find DNA sequence motifs enriched in peaks.

**Why motifs matter**:
- Identify transcription factor binding sequence
- Validate ChIP experiment (should find expected motif)
- Discover co-factor binding sites

**Command breakdown**:
```bash
findMotifsGenome.pl \
    peaks.bed \
    hg38 \
    motif_output/ \
    -size 200 \
    -len 8,10,12 \
    -p 8
```

**Parameters explained**:
- `peaks.bed`: Peak coordinates
- `hg38`: Genome version
- `motif_output/`: Output directory
- `-size 200`: Region size around peak summit
- `-len 8,10,12`: Motif lengths to search
- `-p 8`: Threads

**HOMER algorithm**:
1. Extract sequences around peak summits
2. Search for over-represented patterns
3. Compare to background regions
4. Match to known motif databases

**Output files**:

**1. knownResults.html**:
Matches to known transcription factor motifs
```
Motif Name          Match   P-value     % Targets  % Background
CTCF               ✓       1e-50       45%        5%
```

**2. homerResults.html**:
De novo discovered motifs (novel patterns)

**3. motifX.logo.png**:
Visual representation of motif (sequence logo)

**Interpreting results**:
- **P-value < 1e-10**: Highly significant
- **% Targets > 30%**: Found in many peaks
- **Low % Background**: Specific to peaks

---

### Step 8: Visualization with BigWig

**Purpose**: Create genome browser tracks for visualization.

**Why BigWig?**
- Compressed format
- Fast random access
- Compatible with UCSC, IGV browsers

**Command to create BigWig**:
```bash
bamCoverage \
    -b sample.bam \
    -o sample.bw \
    --binSize 10 \
    --normalizeUsing RPKM \
    -p 8
```

**Parameters**:
- `-b`: Input BAM
- `-o`: Output BigWig
- `--binSize 10`: Resolution (10bp bins)
- `--normalizeUsing RPKM`: Normalization method
- `-p 8`: Threads

**Normalization options**:
- **RPKM**: Reads Per Kilobase per Million
- **CPM**: Counts Per Million
- **BPM**: Bins Per Million

**Visualizing in IGV**:
1. Load reference genome (hg38)
2. Load BigWig track
3. Load peak BED file
4. Navigate to gene of interest

**What to look for**:
- Peaks at expected locations (e.g., promoters)
- Consistent signal across replicates
- Specificity (peaks in ChIP, not input)

---

## Understanding the Output

### Key Output Files

#### 1. **peaks.narrowPeak**
Primary result - list of peaks with coordinates

**Important columns**:
```
chr1  12000  12300  peak1  850  .  15.2  85.5  42.1  150
```
- **Position**: chr1:12000-12300
- **Score**: 850 (for ranking)
- **Fold enrichment**: 15.2x over background
- **-log10(pvalue)**: 85.5
- **-log10(qvalue)**: 42.1
- **Peak summit**: 150bp from start

#### 2. **summits.bed**
Exact peak centers - useful for motif analysis

#### 3. **peaks.xls**
Detailed spreadsheet with all information

#### 4. **BigWig tracks**
For genome browser visualization

### Quality Control Metrics

**1. Number of peaks**:
- Transcription factors: 1,000-50,000 peaks
- Histone marks: 10,000-100,000 peaks
- Too few (<500): Poor ChIP efficiency
- Too many (>100,000 for TF): High background

**2. FRiP (Fraction of Reads in Peaks)**:
```bash
# Calculate FRiP
reads_in_peaks=$(samtools view -c -L peaks.bed sample.bam)
total_reads=$(samtools view -c sample.bam)
frip=$(echo "$reads_in_peaks / $total_reads" | bc -l)
```

**Good FRiP**:
- Transcription factors: >5%
- Histone marks: >10%
- Low FRiP: Poor enrichment

**3. Signal-to-noise ratio**:
Look at peak enrichment scores (fold change)
- Good: >5-10x enrichment
- Excellent: >20x

### Biological Interpretation

**Example: CTCF ChIP-seq**

Findings:
- 45,000 peaks
- FRiP = 15%
- Top motif: CTCF consensus (p=1e-150)
- 60% peaks at promoters/boundaries

**Interpretation**:
✅ Strong enrichment (high FRiP)
✅ Expected motif found
✅ Peaks at known CTCF sites (insulators)
→ High-quality experiment

**Downstream analysis**:
1. Overlap with other datasets (e.g., ATAC-seq)
2. Associate peaks with nearby genes
3. Functional enrichment of target genes
4. Compare conditions (differential binding)

---

## Running the Pipeline

### Quick Start

1. **Prepare config.yaml**:
```yaml
samples:
  - h3k4me3_rep1
  - h3k4me3_rep2

input_control: "input_control"

reference:
  genome: "/path/to/hg38.fa"
  blacklist: "/path/to/hg38-blacklist.v2.bed"

peak_calling:
  q_value: 0.05
  broad_peaks: false  # true for histones like H3K27me3
```

2. **Organize data** (paired-end format):
```bash
data/raw/chip_seq/
├── h3k4me3_rep1_R1.fastq.gz
├── h3k4me3_rep1_R2.fastq.gz
├── h3k4me3_rep2_R1.fastq.gz
├── h3k4me3_rep2_R2.fastq.gz
├── input_control_R1.fastq.gz
└── input_control_R2.fastq.gz
```

**Note**: Pipeline now supports paired-end data (recommended). File naming must follow `{sample}_R1.fastq.gz` and `{sample}_R2.fastq.gz` pattern.

3. **Submit pipeline**:
```bash
cd ~/BioPipelines/pipelines/chip_seq/peak_calling
sbatch ~/BioPipelines/scripts/submit_chip_seq.sh
```

4. **Monitor**:
```bash
squeue --me
tail -f slurm_*.err
```

### Experimental Design

**Input control is essential**:
- Sequence DNA without IP
- Same library prep and sequencing
- Accounts for:
  - Open chromatin bias
  - Mappability issues
  - Copy number variations

**Biological replicates**:
- Minimum: 2 replicates
- Better: 3-4 replicates
- For differential binding: 3+ per condition

**Sequencing depth**:
- Transcription factors: 20-40 million reads
- Histone marks: 40-60 million reads
- Input: Same as ChIP samples

### Troubleshooting

**Too few peaks**:
- Poor ChIP efficiency (antibody quality)
- Over-digestion (small fragments)
- Low sequencing depth
- Try lowering q-value threshold

**Too many peaks**:
- High background
- Non-specific antibody
- Contamination
- Increase q-value cutoff

**Wrong motif enriched**:
- Antibody cross-reactivity
- Co-factor binding
- Sample swap

**Low FRiP score**:
- Poor enrichment
- Suboptimal ChIP protocol
- Consider re-doing experiment

---

## Advanced Topics

### Differential Binding Analysis

**Using DiffBind (R)**:
```R
library(DiffBind)

# Create sample sheet
samples <- data.frame(
  SampleID = c("WT1", "WT2", "KO1", "KO2"),
  Condition = c("WT", "WT", "KO", "KO"),
  bamReads = c("WT1.bam", "WT2.bam", "KO1.bam", "KO2.bam"),
  Peaks = c("WT1_peaks.bed", "WT2_peaks.bed", "KO1_peaks.bed", "KO2_peaks.bed")
)

# Load data
dba <- dba(sampleSheet=samples)

# Count reads
dba <- dba.count(dba)

# Differential analysis
dba <- dba.contrast(dba, categories=DBA_CONDITION)
dba <- dba.analyze(dba)

# Get results
results <- dba.report(dba)
```

### Multi-modal Integration

Combine with other data types:
- ChIP-seq + RNA-seq: TF binding → gene expression
- ChIP-seq + ATAC-seq: Binding at accessible chromatin
- ChIP-seq + Hi-C: 3D genome organization

---

## Additional Resources

### Further Reading
- [ENCODE ChIP-seq Guidelines](https://www.encodeproject.org/chip-seq/)
- [MACS2 Documentation](https://github.com/macs3-project/MACS)
- [HOMER Motif Analysis](http://homer.ucsd.edu/homer/motif/)

### Tools Used
- **FastQC/fastp**: QC and trimming
- **Bowtie2/BWA**: Alignment
- **MACS2**: Peak calling
- **HOMER**: Motif analysis
- **deepTools**: Visualization

---

## Glossary

- **ChIP**: Chromatin Immunoprecipitation
- **Peak**: Region of enrichment
- **Summit**: Highest point of a peak
- **FRiP**: Fraction of Reads in Peaks
- **Motif**: DNA sequence pattern
- **q-value**: FDR-adjusted p-value
- **Blacklist**: Regions to exclude (repetitive, high background)

---

*Last updated: November 2025*
