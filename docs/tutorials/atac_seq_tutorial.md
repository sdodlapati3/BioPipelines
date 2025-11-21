# ATAC-seq Chromatin Accessibility Pipeline Tutorial

## Table of Contents
1. [Introduction](#introduction)
2. [Biological Background](#biological-background)
3. [Pipeline Overview](#pipeline-overview)
4. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
5. [Understanding the Output](#understanding-the-output)
6. [Running the Pipeline](#running-the-pipeline)

## Introduction

This tutorial covers ATAC-seq (Assay for Transposase-Accessible Chromatin using sequencing), a technique to map genome-wide chromatin accessibility and identify regulatory elements.

### What You'll Learn
- What ATAC-seq is and how it works
- How to process ATAC-seq data
- How to identify open chromatin regions
- How to analyze nucleosome positioning

### Prerequisites
- Basic understanding of chromatin structure
- Familiarity with genomics concepts
- Access to BioPipelines environment

## Biological Background

### What is Chromatin Accessibility?

**Chromatin** is DNA wrapped around histone proteins. Its structure determines which genes can be accessed by transcription factors:

```
Closed Chromatin (Heterochromatin)
 - Tightly packed
 - Genes turned OFF
 - Inaccessible to TFs

Open Chromatin (Euchromatin)
 - Loosely packed
 - Genes can be turned ON
 - Accessible to TFs
```

**Accessible regions** = Regulatory elements:
- Promoters
- Enhancers
- Insulators
- Silencers

### How ATAC-seq Works

**The Tn5 transposase**:
- Inserts sequencing adapters into accessible DNA
- Cannot access DNA wrapped around nucleosomes
- Creates a "footprint" of open chromatin

**The process**:
```
1. Isolate nuclei from cells
2. Treat with Tn5 transposase + adapters
3. Tn5 "tagments" (cuts + tags) accessible DNA
4. PCR amplify tagged fragments
5. Sequence fragments
6. Map reads to genome
7. Open regions = high read density
```

### ATAC-seq vs. Other Methods

| Method | Measures | Cells Needed | Time |
|--------|----------|--------------|------|
| **ATAC-seq** | Accessibility | 50,000 | 3 hours |
| **DNase-seq** | Accessibility | 10 million | 2 days |
| **ChIP-seq** | Protein binding | 10 million | 3 days |
| **MNase-seq** | Nucleosomes | 1 million | 2 days |

**ATAC-seq advantages**:
- Fast protocol
- Low cell numbers
- Captures nucleosome positions
- Identifies TF footprints

### Applications

- Map active regulatory elements
- Study development and differentiation
- Compare cell types
- Identify disease mechanisms
- Single-cell chromatin accessibility

## Pipeline Overview

### The Complete Workflow

```
┌────────────────────────────────────────────────────────────────┐
│              ATAC-seq Chromatin Accessibility Pipeline          │
└────────────────────────────────────────────────────────────────┘

Raw FASTQ Files (paired-end reads from Tn5 fragments)
    │
    ├──► [1] QUALITY CONTROL (FastQC + MultiQC)
    │         └─► Check read quality, adapter content
    │
    ├──► [2] ADAPTER TRIMMING (fastp - Nextera adapters)
    │         └─► Remove Tn5 adapter sequences
    │              
    │              Before:  ──adapter──DNA──adapter──
    │              After:   ────────DNA────────
    │
    ├──► [3] ALIGNMENT (Bowtie2 - sensitive mode)
    │         └─► Map fragments to genome
    │              
    │              Chromatin:
    │              ═════════╦══════════════╦═══════════
    │                       ║              ║
    │                   Open regions   Nucleosome
    │                   (many reads)   (few reads)
    │
    ├──► [4] POST-ALIGNMENT FILTERING
    │         ├─► Remove unmapped reads
    │         ├─► Remove mitochondrial reads (chrM)
    │         ├─► Remove duplicates
    │         └─► Keep properly paired reads
    │
    ├──► [5] FRAGMENT SIZE ANALYSIS
    │         └─► Calculate insert size distribution
    │              
    │              Expected pattern:
    │              Reads │     ╱╲
    │                    │    ╱  ╲     Nucleosomal
    │              Count │   ╱    ╲   ╱╲  ladder
    │                    │__╱______╲_╱__╲______
    │                       <147  295  442 bp
    │                       NFR   1N   2N
    │
    ├──► [6] PEAK CALLING (MACS2)
    │         └─► Identify accessible regions
    │              (peaks = open chromatin)
    │
    └──► [7] VISUALIZATION (deepTools)
              ├─► Create bigWig tracks
              └─► Generate heatmaps/profiles

Final Output:
    • peaks.narrowPeak (50K-150K accessible regions)
    • bigWig tracks (for genome browser visualization)
    • Fragment size distribution (QC metric)
    • MultiQC report (quality summary)
```

**Key ATAC-seq Concepts**:

| Concept | Description | Biological Meaning |
|---------|-------------|-------------------|
| **NFR** (Nucleosome-Free Region) | Fragments <147 bp | Open chromatin (active regulatory elements) |
| **Mono-nucleosomal** | Fragments ~147-294 bp | DNA wrapped once around histone octamer |
| **Di-nucleosomal** | Fragments ~295-442 bp | DNA wrapped around two nucleosomes |
| **Peak** | Region with high read density | Accessible chromatin region (promoter/enhancer) |
| **Mitochondrial reads** | Reads mapping to chrM | Technical artifact (remove these) |

**Visual: Tn5 Transposase Action**:
```
Closed chromatin (inaccessible):
    ═══════════════════════════════
    Cannot insert adapters

Open chromatin (accessible):
    ───────────────────────────────
         ↓ Tn5 transposase
    ──A──     ──A──     ──A──
      │         │         │
    Many Tn5 insertions → Many sequencing reads → Peak called
```

**Pipeline Dependencies Fix**:
⚠️ **Important**: This pipeline requires `deeptools>=3.5.6` for compatibility with modern matplotlib versions.

**Known Issue Fixed**: Earlier versions (deeptools 3.5.2) failed with:
```
AttributeError: module 'matplotlib.cm' has no attribute 'register_cmap'
```

**Solution**: Upgraded to deeptools 3.5.6 which uses current matplotlib API.
3. Alignment (Bowtie2)
    ↓
4. Filter Mitochondrial Reads
    ↓
5. Remove Duplicates
    ↓
6. Shift Reads (+4/-5 for Tn5)
    ↓
7. Peak Calling (MACS2)
    ↓
8. Fragment Size Analysis
    ↓
9. TSS Enrichment
    ↓
10. Footprinting (optional)
    ↓
Final Peaks + QC Metrics
```

### ATAC-seq Specific Features

**1. Nextera adapter trimming**:
Special adapters used by Tn5

**2. Mitochondrial reads**:
Very high in ATAC-seq (mitochondria are accessible)
- Can be 20-80% of reads
- Must filter out

**3. Fragment size distribution**:
Reveals nucleosome structure:
- <100bp: Nucleosome-free
- 180-247bp: Mono-nucleosome
- 315-473bp: Di-nucleosome

**4. Tn5 offset**:
Reads must be shifted (+4 forward, -5 reverse)
Accounts for Tn5 binding

### Time Estimates
- 3 samples: 2-4 hours
- Most time in alignment and peak calling

## Step-by-Step Walkthrough

### Step 1: Quality Control with FastQC

**Purpose**: Assess read quality.

**ATAC-seq specific checks**:
- ✅ High duplication is normal (open chromatin sampled repeatedly)
- ✅ Nextera adapter content
- ⚠️ Mitochondrial contamination not visible here

**Command**:
```bash
fastqc -t 4 -o qc_output/ sample_R1.fastq.gz sample_R2.fastq.gz
```

**What to expect**:
- Quality scores >30
- Adapter content present
- Bimodal length distribution (nucleosome/nucleosome-free)

---

### Step 2: Adapter Trimming with fastp

**Purpose**: Remove Nextera transposase adapters.

**Why Nextera-specific?**
ATAC-seq uses Nextera adapters:
```
Adapter sequence: CTGTCTCTTATACACATCT
```

**Command**:
```bash
fastp \
    -i input_R1.fastq.gz \
    -I input_R2.fastq.gz \
    -o trimmed_R1.fastq.gz \
    -O trimmed_R2.fastq.gz \
    --adapter_sequence CTGTCTCTTATACACATCT \
    --adapter_sequence_r2 CTGTCTCTTATACACATCT \
    --qualified_quality_phred 20 \
    --length_required 25 \
    --thread 4
```

**Parameters**:
- `--adapter_sequence`: Nextera adapter
- `--length_required 25`: Minimum length (short fragments OK)

**Output**: Trimmed paired-end reads

---

### Step 3: Alignment with Bowtie2

**Purpose**: Map reads to reference genome.

**Why Bowtie2?**
- Fast paired-end alignment
- Handles short fragments well
- Standard for ATAC-seq

**Command**:
```bash
bowtie2 \
    -x reference_index \
    -1 R1.fastq.gz \
    -2 R2.fastq.gz \
    -p 8 \
    --very-sensitive \
    --maxins 2000 \
    | samtools sort -@ 4 -o output.bam
```

**Parameters explained**:
- `-1/-2`: Paired-end reads
- `--maxins 2000`: Maximum insert size (captures large fragments)
- `--very-sensitive`: Thorough alignment
- `-p 8`: Use 8 threads

**Expected alignment**:
- >85% alignment rate
- Most reads properly paired

---

### Step 4: Filter Mitochondrial Reads

**Purpose**: Remove highly abundant mitochondrial DNA reads.

**Why filter mitochondria?**
- Mitochondrial DNA is very accessible (circular, no nucleosomes)
- Can represent 20-80% of reads in ATAC-seq libraries
- Obscures nuclear chromatin signal
- Not biologically interesting for most analyses

**Visual Concept**:
```
Before filtering:
    Total reads = 10 million
    ├─► Nuclear chromatin: 2M reads (20%)
    └─► Mitochondrial DNA: 8M reads (80%) ← Problem!

After filtering (remove chrM):
    Usable reads = 2 million
    └─► All nuclear chromatin (100%)
    
Now peak calling focuses on real regulatory elements.
```

**Quality Metrics**:
| Mitochondrial % | Quality | Interpretation |
|-----------------|---------|----------------|
| <10% | Excellent | High-quality nuclei prep |
| 10-30% | Good | Acceptable for analysis |
| 30-50% | Fair | May reduce sensitivity |
| >50% | Poor | Consider re-prep or deeper sequencing |

**Command**:
```bash
samtools view -b -h \
    -@ 8 \
    sample.bam \
    $(samtools view -H sample.bam | grep '^@SQ' | cut -f 2 | \
      grep -v 'chrM' | sed 's/SN://') \
    > sample.noMT.bam
```

**Simpler approach**:
```bash
samtools view -b sample.bam \
    chr1 chr2 chr3 ... chr22 chrX chrY \
    > sample.noMT.bam
```

**Check mitochondrial percentage**:
```bash
mito=$(samtools view -c sample.bam chrM)
total=$(samtools view -c sample.bam)
pct=$(echo "$mito / $total * 100" | bc -l)
echo "Mitochondrial reads: $pct%"
```

**Acceptable mito %**:
- <30%: Excellent
- 30-50%: Good
- 50-80%: Acceptable
- >80%: Poor quality, consider redoing

---

### Step 5: Remove Duplicates

**Purpose**: Remove PCR duplicates.

**ATAC-seq duplication considerations**:
- Higher duplication than DNA-seq is normal
- Open chromatin regions sampled repeatedly
- True biological duplicates vs. PCR duplicates

**Command**:
```bash
picard MarkDuplicates \
    -I input.bam \
    -O deduplicated.bam \
    -M metrics.txt \
    --REMOVE_DUPLICATES true
```

**Duplication rates**:
- 10-40%: Normal
- >60%: Low library complexity

**Output**: Deduplicated BAM

---

### Step 6: Tn5 Offset Correction

**Purpose**: Adjust read positions for Tn5 binding.

**Why shift reads?**
Tn5 binds as a dimer:
```
Forward strand: +4bp shift
Reverse strand: -5bp shift
```
This centers reads on the actual cut site.

**Command using deepTools**:
```bash
alignmentSieve \
    -b input.bam \
    -o shifted.bam \
    --ATACshift
```

**Manual shifting with custom script**:
```python
# Shift reads in BAM
for read in bam:
    if read.is_forward:
        read.pos += 4
    else:
        read.pos -= 5
```

---

### Step 7: Peak Calling with MACS2

**Purpose**: Identify open chromatin regions.

**MACS2 for ATAC-seq**:
Uses different parameters than ChIP-seq:
- Shift reads (covered above)
- Set extension size
- No input control needed

**Command**:
```bash
macs2 callpeak \
    -t sample.shifted.bam \
    -f BAMPE \
    -n sample \
    -g hs \
    --outdir peaks \
    --shift -100 \
    --extsize 200 \
    --nomodel \
    -q 0.05 \
    --keep-dup all
```

**Parameters explained**:
- `-f BAMPE`: Paired-end BAM
- `--shift -100`: Shift adjustment
- `--extsize 200`: Extension size
- `--nomodel`: Don't build shifting model
- `--keep-dup all`: Keep all duplicate reads (after filtering)

**Alternative: GENRICH**
Specifically designed for ATAC-seq:
```bash
Genrich \
    -t sample.bam \
    -o peaks.narrowPeak \
    -j  # ATAC-seq mode
```

**Output**:
- **narrowPeak file**: BED format with peak coordinates
- **Summit positions**: Exact location of highest signal
- **Peak scores**: Confidence metric for each peak

**Peak Calling Visual Concept**:
```
Genome browser view:

Chromosome 1: ──────────────────────────────────────────

Read coverage:
             ╱╲              ╱╲╱╲
        ____╱  ╲________    ╱    ╲_____╱╲_____
              │               │         │
              Peak 1          Peak 2    Peak 3
              (promoter)      (enhancer)(weak)
              
              High             High     Low
              confidence       confidence   confidence
              (many reads)     (many reads) (few reads)

MACS2 calls peaks where read density significantly exceeds background.
```

**Peak Interpretation**:

| Peak Count | Typical Cell Type | Biological Meaning |
|------------|-------------------|--------------------|
| **50K-80K** | Differentiated cells | Specific cell identity |
| **80K-150K** | Stem cells / progenitors | Open chromatin landscape |
| **<30K** | May indicate poor library | Re-evaluate quality |
| **>200K** | May indicate noise | Adjust p-value threshold |

**Peak Locations**:
```
Typical peak distribution in human genome:

  30-40%  ──► Promoters (near TSS)
  40-50%  ──► Intergenic (enhancers)
  20-30%  ──► Introns / gene bodies
  <5%     ──► Exons
  
This distribution suggests functional regulatory elements.
```

**Peak Width**:
- **Narrow peaks (200-500bp)**: TF binding sites, promoters
- **Broad peaks (500-1000bp)**: Active enhancers
- **Very broad (>2kb)**: May indicate poor resolution

**narrowPeak Format**:
```
chr1  1000  1500  peak_1  100  .  5.2  10.5  8.3  250
│     │     │     │       │    │  │    │     │    │
│     │     │     │       │    │  │    │     │    └─ Summit offset
│     │     │     │       │    │  │    │     └────── q-value
│     │     │     │       │    │  │    └──────────── p-value  
│     │     │     │       │    │  └───────────────── Signal value
│     │     │     │       │    └──────────────────── Strand
│     │     │     │       └───────────────────────── Score (0-1000)
│     │     │     └───────────────────────────────── Peak name
│     │     └─────────────────────────────────────── End position
│     └───────────────────────────────────────────── Start position
└─────────────────────────────────────────────────── Chromosome
```

---

### Step 8: Fragment Size Analysis

**Purpose**: Assess nucleosome structure from fragment lengths.

**Why fragment sizes matter?**
Different fragment sizes reveal chromatin structure:

**Visual Explanation**:
```
DNA wrapped around nucleosomes:

    NFR          Mono-nuc       Di-nuc         Tri-nuc
    <147bp       147-294bp      295-442bp      443-590bp
    
    ────         ═══╗           ═══╗═══╗       ═══╗═══╗═══╗
    open         ───║───        ───║───║───    ───║───║───║───
                 ═══╝           ═══╝═══╝       ═══╝═══╝═══╝
                 
Fragment Size Distribution (expected):

  Read │     
 Count │    ╱╲                High peak at <147bp
       │   ╱  ╲               (nucleosome-free)
       │  ╱    ╲    ╱╲        
       │ ╱      ╲  ╱  ╲       Smaller peaks at
       │╱        ╲╱    ╲╱     ~200, ~400, ~600bp
       └─────────────────────── Fragment Size (bp)
         50  147  294  442  590
```

**Interpretation Guide**:

| Pattern | Meaning | Quality |
|---------|---------|---------|
| **Strong <147bp peak** | Many NFRs (open chromatin) | ✓ Good ATAC signal |
| **Periodic peaks** (~200, 400, 600bp) | Nucleosome ladder visible | ✓ High-quality library |
| **Single broad peak** | Poor nucleosome resolution | ✗ May indicate degraded DNA |
| **No <147bp peak** | Few accessible regions | ✗ Possible technical issue |

**Fragment Size Categories**:
```
<100bp     : Sub-nucleosomal / TF-protected fragments
100-147bp  : Nucleosome-free regions (NFR) ← Regulatory elements
147-294bp  : Mono-nucleosome (1N)
295-442bp  : Di-nucleosome (2N)
443-590bp  : Tri-nucleosome (3N)
>590bp     : Higher-order chromatin structures
```

**Compute fragment sizes**:
```bash
picard CollectInsertSizeMetrics \
    -I sample.bam \
    -O insert_sizes.txt \
    -H insert_size_histogram.pdf
```

**Plotting with Python**:
```python
import pysam
import matplotlib.pyplot as plt

fragments = []
bam = pysam.AlignmentFile("sample.bam")
for read in bam:
    if read.is_proper_pair and read.is_read1:
        fragments.append(abs(read.template_length))

plt.hist(fragments, bins=100, range=(0, 1000))
plt.xlabel("Fragment size (bp)")
plt.ylabel("Count")
plt.title("ATAC-seq Fragment Size Distribution")
```

**Expected distribution**:
- **Peak 1** at ~50-100bp (NFR)
- **Peak 2** at ~200bp (mono-nucleosome)
- **Peak 3** at ~400bp (di-nucleosome)

**Good quality**:
- Clear periodicity
- Strong NFR peak
- Nucleosome peaks visible

**Poor quality**:
- No periodicity
- Flat distribution
- Suggests degraded or poor quality

---

### Step 9: TSS Enrichment Score

**Purpose**: Measure enrichment at transcription start sites (TSS).

**Why TSS enrichment?**
- TSS regions are typically open (active promoters)
- High-quality ATAC-seq shows strong TSS signal
- Key QC metric

**Computing TSS enrichment**:
```bash
# Using deepTools
computeMatrix reference-point \
    -S sample.bw \
    -R genes_tss.bed \
    --referencePoint TSS \
    -b 2000 -a 2000 \
    -o matrix.gz

# Plot heatmap
plotHeatmap \
    -m matrix.gz \
    -o tss_heatmap.png \
    --sortRegions descend
```

**TSS Enrichment Score**:
```
TSS score = (TSS signal) / (flanking region signal)
```

**Interpreting TSS score**:
- >7: Excellent
- 5-7: Good
- 3-5: Acceptable
- <3: Poor quality

**Visual check**:
- Strong signal at TSS
- V-shaped plot (depleted immediately upstream/downstream)
- Nucleosomes flanking TSS

---

### Step 10: TF Footprinting (Advanced)

**Purpose**: Identify transcription factor binding within accessible regions.

**How footprinting works**:
```
Accessible region (peak)
    ┌──────────────────────┐
    │ ██████  ____  ██████ │
    │        TF bound       │
    └──────────────────────┘
         ↑
    Footprint (protected from Tn5)
```

TF binding protects DNA from Tn5, creating a "footprint".

**Tools for footprinting**:

**1. HINT (Hierarchical INtegration of TF footprints)**:
```bash
rgt-hint footprinting \
    --atac-seq \
    --organism hg38 \
    peaks.bed \
    sample.bam \
    --output-location footprints/
```

**2. TOBIAS**:
```bash
TOBIAS ATACorrect \
    --bam sample.bam \
    --genome genome.fa \
    --peaks peaks.bed \
    --outdir corrected/

TOBIAS FootprintScores \
    --signal corrected/sample_corrected.bw \
    --regions peaks.bed \
    --output scores.bw

TOBIAS BINDetect \
    --motifs motif_database.txt \
    --signals scores.bw \
    --genome genome.fa \
    --peaks peaks.bed \
    --outdir bindetect/
```

**Output**:
- Footprint positions
- Enriched TF motifs
- Binding scores

---

## Understanding the Output

### Key Output Files

#### 1. **peaks.narrowPeak**
Open chromatin regions

**Example**:
```
chr1  10000  10500  peak1  450  .  8.5  45.2  28.1  250
```

**Typical ATAC-seq peaks**:
- 50,000-150,000 peaks
- Enriched at promoters (~30-40%)
- Also at distal enhancers

#### 2. **Fragment Size Distribution**
Shows nucleosome structure

**Good quality indicators**:
- Clear NFR peak
- Visible nucleosome periodicity
- Ratio of NFR:mono-nucleosome >1.5

#### 3. **TSS Enrichment Plot**
Signal around transcription start sites

**Pattern**:
```
Signal
  ^
  |     /\
  |    /  \
  |   /    \
  |  /  TSS \
  | /        \
  +--------------> Position
   -2kb  TSS  +2kb
```

#### 4. **BigWig Tracks**
For visualization in genome browsers

### Quality Control Metrics Summary

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| **Alignment rate** | >85% | 75-85% | <75% |
| **Mitochondrial %** | <30% | 30-50% | >50% |
| **Duplicate rate** | <30% | 30-50% | >50% |
| **TSS enrichment** | >7 | 5-7 | <5 |
| **# Peaks** | 50k-150k | 30k-50k | <30k |
| **NFR/mono ratio** | >1.5 | 1.0-1.5 | <1.0 |

### Biological Interpretation

**Peak annotation**:
```R
library(ChIPseeker)
peaks <- readPeakFile("sample_peaks.narrowPeak")
peakAnno <- annotatePeak(peaks, TxDb=txdb)
plotAnnoPie(peakAnno)
```

**Expected distribution**:
- Promoters: 30-40%
- Intergenic: 30-40%
- Intron: 20-30%
- Exon: <5%

**Motif enrichment**:
Identify active TFs:
```bash
findMotifsGenome.pl \
    peaks.bed hg38 motifs/ \
    -size 200
```

**Differential accessibility**:
Compare cell types or conditions:
```R
library(DiffBind)
# Similar to ChIP-seq differential analysis
```

---

## Running the Pipeline

### Quick Start

1. **Prepare config.yaml**:
```yaml
samples:
  - sample1
  - sample2
  - sample3

reference:
  genome: "/path/to/hg38.fa"
  tss_bed: "/path/to/hg38_tss.bed"
  blacklist: "/path/to/hg38-blacklist.bed"

trimming:
  nextera_adapter: "CTGTCTCTTATACACATCT"
  min_length: 25

peak_calling:
  genome_size: "2.7e9"
  shift: -100
  extsize: 200
```

2. **Organize data**:
```bash
data/raw/atac_seq/
├── sample1_R1.fastq.gz
├── sample1_R2.fastq.gz
├── sample2_R1.fastq.gz
├── sample2_R2.fastq.gz
└── ...
```

3. **Submit pipeline**:
```bash
cd ~/BioPipelines/pipelines/atac_seq/accessibility_analysis
sbatch ~/BioPipelines/scripts/submit_atac_seq.sh
```

4. **Monitor**:
```bash
squeue --me
tail -f slurm_*.err
```

### Experimental Design

**Cell numbers**:
- Standard ATAC-seq: 50,000 cells
- Low-input: 5,000-50,000 cells
- Single-cell: 500-5,000 cells

**Biological replicates**:
- Minimum: 2 replicates
- Recommended: 3-4 replicates

**Sequencing depth**:
- Standard: 50-100 million reads per sample
- Minimum: 25 million reads
- More depth → detect weaker peaks

**Read length**:
- 50-75bp paired-end sufficient
- Longer reads don't improve analysis

### Troubleshooting

**High mitochondrial percentage (>50%)**:
- Poor cell lysis
- Nuclei not washed properly
- Cell type specific (cardiac muscle naturally high)

**Low TSS enrichment (<5)**:
- Poor library quality
- Over-digestion (too much Tn5)
- Cell death/apoptosis

**Few peaks (<30,000)**:
- Low sequencing depth
- Condensed chromatin (heterochromatin-rich cells)
- Poor Tn5 efficiency

**No nucleosome periodicity**:
- Degraded samples
- Over-digestion
- Wrong fragment size selection

**Diffuse signal (no clear peaks)**:
- High background
- Too much Tn5
- Low cell numbers

---

## Advanced Topics

### Single-cell ATAC-seq

Uses same principles but:
- Barcoding for cell identification
- Sparse data (lower coverage per cell)
- Specialized analysis (ArchR, Signac)

### Integration with Other Data

**ATAC + RNA-seq**:
Correlate accessibility with expression
```R
# Genes with open promoters should be expressed
atac_promoters <- accessible_genes
rna_expressed <- expressed_genes
overlap <- intersect(atac_promoters, rna_expressed)
```

**ATAC + ChIP-seq**:
TF binding at accessible sites
```bash
# Intersect ChIP peaks with ATAC peaks
bedtools intersect -a chip_peaks.bed -b atac_peaks.bed
```

### Allele-Specific Accessibility

Detect imprinting or allele-specific regulation using variants.

---

## Additional Resources

### Further Reading
- [ATAC-seq Guidelines (ENCODE)](https://www.encodeproject.org/atac-seq/)
- [Buenrostro et al. 2013 (Original paper)](https://www.nature.com/articles/nmeth.2688)
- [ATAC-seq Data Analysis](https://informatics.fas.harvard.edu/atac-seq-guidelines.html)

### Tools Used
- **FastQC/fastp**: QC and trimming
- **Bowtie2**: Alignment
- **MACS2/Genrich**: Peak calling
- **deepTools**: Visualization and QC
- **TOBIAS/HINT**: Footprinting

---

## Glossary

- **ATAC-seq**: Assay for Transposase-Accessible Chromatin using sequencing
- **Tn5**: Transposase enzyme that tagments DNA
- **NFR**: Nucleosome-Free Region
- **TSS**: Transcription Start Site
- **Footprint**: Protected region where TF binds
- **Tagmentation**: Simultaneous fragmentation and tagging
- **Open chromatin**: Accessible to regulatory proteins

---

*Last updated: November 2025*
