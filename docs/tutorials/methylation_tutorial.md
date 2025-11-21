# DNA Methylation Analysis Tutorial (WGBS/RRBS)

## Table of Contents
1. [Introduction](#introduction)
2. [Biological Background](#biological-background)
3. [Pipeline Overview](#pipeline-overview)
4. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
5. [Output Interpretation](#output-interpretation)
6. [Running the Pipeline](#running-the-pipeline)
7. [Troubleshooting](#troubleshooting)

---

## Introduction

DNA methylation is a crucial epigenetic modification that regulates gene expression without altering the DNA sequence. This pipeline analyzes whole-genome bisulfite sequencing (WGBS) or reduced representation bisulfite sequencing (RRBS) data to identify methylation patterns and differential methylation regions (DMRs).

### What You'll Learn
- How bisulfite conversion works and why it's used
- Understanding methylation context (CpG, CHG, CHH)
- Interpreting M-bias plots and conversion efficiency
- Identifying differentially methylated regions (DMRs)
- Quality control metrics for bisulfite sequencing

### Prerequisites
- Basic understanding of epigenetics and DNA methylation
- Familiarity with FASTQ file format
- WGBS or RRBS sequencing data (paired-end or single-end)

---

## Biological Background

### DNA Methylation Basics

DNA methylation involves adding a methyl group (CH₃) to cytosine bases, typically at CpG dinucleotides:

```
    Unmethylated                  Methylated
    
      H                             CH₃
      |                              |
      N                              N
     / \                            / \
    C   N                          C   N
    ||  |                          ||  |
    N—C—N                          N—C—N
         \                              \
         Cytosine                       5-methylcytosine
```

**Key Concepts:**
- **CpG islands**: Regions rich in CpG dinucleotides, often at gene promoters
- **Hypermethylation**: Increased methylation, often associated with gene silencing
- **Hypomethylation**: Decreased methylation, often associated with gene activation
- **Context**: CpG (major), CHG, CHH (minor contexts in mammals)

### Bisulfite Conversion

Bisulfite treatment converts unmethylated cytosines to uracil, while methylated cytosines remain unchanged:

```
Bisulfite Treatment Process:

Original DNA:     5'- C  C  mC  C  G  C -3'
                     |  |  |   |  |  |
                  3'- G  G  G   G  C  G -5'
                     
After BS:         5'- U  U  mC  U  G  U -3'
                     |  |  |   |  |  |
                  3'- G  G  G   G  C  G -5'
                     
After PCR:        5'- T  T  C   T  G  T -3'
                     |  |  |   |  |  |
                  3'- A  A  G   A  C  A -5'

Legend: C = cytosine, mC = methylated cytosine (5mC), U = uracil, T = thymine
```

**Bisulfite Conversion Workflow:**
1. DNA denaturation (single-stranded)
2. Sodium bisulfite treatment
3. Unmethylated C → U conversion
4. Alkaline desulfonation
5. PCR amplification (U → T)

**Why This Works:**
- Methylated cytosines are protected from conversion
- Sequencing reveals methylation by comparing to reference (C in reads = methylated)
- Conversion efficiency >99% required for accurate results

### WGBS vs RRBS

| Feature | WGBS | RRBS |
|---------|------|------|
| Coverage | Whole genome (~28M CpGs in human) | ~3-4M CpGs (10-15% of genome) |
| Resolution | Single-base | Single-base |
| Cost | High (100-300M reads) | Lower (20-50M reads) |
| Enrichment | None | MspI restriction (CpG-rich regions) |
| Best For | Comprehensive profiling | CpG islands, promoters, enhancers |
| Read Depth | 10-30× | 30-100× at covered sites |

**RRBS Strategy:**
```
MspI Recognition: C^CGG (cuts at ^)

Genome → MspI digest → Size selection (40-220bp)
                           ↓
                    CpG-rich fragments
                           ↓
                    Bisulfite conversion
                           ↓
                    Sequencing & analysis
```

---

## Pipeline Overview

### Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      METHYLATION ANALYSIS PIPELINE                   │
│                      (WGBS/RRBS Bisulfite Sequencing)                │
└─────────────────────────────────────────────────────────────────────┘

Input: Raw FASTQ files (SE or PE)
  ↓
┌─────────────────────────────────────┐
│ 1. Prepare Bismark Index            │  ← One-time setup per genome
│    • Convert genome to bisulfite    │
│    • Create Bowtie2 index           │
│    Time: ~2-4 hours (human genome)  │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 2. FastQC (Raw Reads)               │
│    • Base quality scores            │
│    • GC content                     │
│    • Adapter contamination          │
│    Time: ~5-10 min/sample           │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 3. Trim Galore (Bisulfite Mode)    │
│    • Adapter removal                │
│    • Quality trimming (Q≥20)        │
│    • RRBS-specific trimming         │
│    • Min length filter (≥50bp)      │
│    Time: ~10-20 min/sample          │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 4. Bismark Alignment                │
│    • Bisulfite-aware mapping        │
│    • Handles C→T conversions        │
│    • Multi-threading support        │
│    Time: ~2-6 hours/sample (WGBS)   │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 5. Deduplication                    │
│    • Remove PCR duplicates          │
│    • Keep unique molecules          │
│    Time: ~30-60 min/sample          │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 6. Methylation Extraction           │
│    • Extract CpG methylation        │
│    • Extract CHG/CHH contexts       │
│    • Generate bedGraph tracks       │
│    • M-bias plot generation         │
│    Time: ~1-2 hours/sample          │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 7. Bismark Report                   │
│    • Alignment statistics           │
│    • Conversion efficiency          │
│    • Methylation summary            │
│    • M-bias plots                   │
│    Time: ~5 min/sample              │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 8. DMR Calling (metilene)           │
│    • Statistical testing            │
│    • Group comparisons              │
│    • Minimum difference filter      │
│    Time: ~1-3 hours/comparison      │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 9. MultiQC Report                   │
│    • Aggregate all metrics          │
│    • Interactive visualizations     │
│    Time: ~2-5 min                   │
└─────────────────────────────────────┘
  ↓
Output Files:
  • Deduplicated BAM files
  • Coverage files (.cov.gz)
  • bedGraph tracks (methylation)
  • DMR regions (BED format)
  • MultiQC HTML report
```

### Pipeline Features

**Adaptive Processing:**
- Automatically detects single-end vs paired-end
- RRBS mode with optional MspI fill-in removal
- Handles directional and non-directional libraries

**Quality Control:**
- Conversion efficiency monitoring (target: >99%)
- M-bias detection and correction
- Duplicate rate assessment
- Mapping efficiency tracking

**Analysis Capabilities:**
- Single-base resolution methylation
- Context-specific analysis (CpG, CHG, CHH)
- Differential methylation regions (DMRs)
- Statistical significance testing

---

## Step-by-Step Walkthrough

### Step 1: Prepare Bismark Index

**What It Does:**
Creates a bisulfite-converted reference genome for alignment.

**Process:**
```bash
# Bismark converts reference to 4 strands:
# 1. C→T converted forward strand
# 2. G→A converted forward strand (equivalent to C→T reverse complement)
# 3. C→T converted reverse strand
# 4. G→A converted reverse strand

Original:     5'- ATCGATCG -3'
             3'- TAGCTAGC -5'

BS Forward:   5'- ATTGATTG -3'  (C→T)
             3'- TAACTAGC -5'  (G→A)

BS Reverse:   5'- ATCGATCG -3'  (G→A)
             3'- TAAATAAA -5'  (C→T)
```

**Configuration:**
```yaml
reference:
  genome: /path/to/hg38.fa
  bismark_index: /path/to/bismark_index/  # Will be created
```

**Expected Output:**
- Bisulfite_Genome/CT_conversion/ (C→T converted)
- Bisulfite_Genome/GA_conversion/ (G→A converted)
- Bowtie2 index files for each conversion

**Quality Checks:**
✓ Index files created successfully
✓ No errors in bismark_genome_preparation.log
✓ Index size appropriate (~12GB for hg38)

---

### Step 2: FastQC on Raw Reads

**What It Does:**
Quality assessment of raw sequencing data before processing.

**Key Metrics:**
```
Expected for Bisulfite Data:

1. Per Base Sequence Quality:
   Quality scores should be ≥30 for most bases
   Some decline at read ends is normal

2. GC Content:
   WGBS: Lower than genomic average (~38% instead of 42%)
   Reason: Unmethylated C→T conversion reduces GC%
   
3. Sequence Duplication:
   RRBS: High duplication expected (10-40%)
   WGBS: Moderate duplication (5-20%)
   Reason: Limited complexity from size selection (RRBS)

4. Adapter Content:
   Should be minimal (<5% at read ends)
   High adapter: trimming needed
```

**What to Look For:**
- ✓ Good: Mean quality score ≥28
- ⚠️ Warning: Quality drops below Q20 after position 100
- ✗ Fail: Mean quality score <20 (re-sequence recommended)

**Example Interpretation:**
```
Sample: brain_wgbs_rep1
  Total Sequences: 150,234,567
  Sequence Length: 150bp
  %GC: 38%  ← Expected for BS-seq (lower than genomic)
  Mean Quality: 35.2  ← Excellent
  Adapter Content: 2.1%  ← Acceptable
```

---

### Step 3: Trim Galore (Bisulfite Mode)

**What It Does:**
Removes adapters and low-quality bases, with special handling for bisulfite data.

**Trim Galore Bisulfite Features:**
```
Standard Trimming:
  --quality 20           # Trim bases <Q20
  --length 50            # Discard reads <50bp
  --adapter auto-detect  # Detect Illumina adapters

RRBS-Specific (--rrbs flag):
  Remove 2bp from 5' end (MspI fill-in artifacts)
  
  5'- [NN]CCGG-sequence-rest -3'
      ^^
      Remove these (from end-repair)

Non-directional (--non_directional flag):
  Trim 6bp from 5' end (random priming artifacts)
  Used for non-directional libraries
```

**Configuration Options:**
```yaml
trimming:
  quality: 20                    # Q-score cutoff
  min_length: 50                 # Minimum read length
  clip_r1: 10                    # Hard clip from 5' end (R1)
  clip_r2: 10                    # Hard clip from 5' end (R2)
  rrbs: false                    # Enable RRBS mode
  non_directional: false         # Non-directional library
```

**Output Files:**
- `sample_R1.trimmed.fastq.gz` (trimmed reads)
- `sample_R2.trimmed.fastq.gz` (if PE)
- `sample_trimming_report.txt` (statistics)

**Expected Results:**
```
Trim Galore Report:
  Total reads processed: 150,234,567
  Reads with adapters: 12,345,678 (8.2%)
  Quality trimmed: 45,123,456 (30.0%)
  Reads too short: 2,345,678 (1.6%)
  Reads written: 147,888,889 (98.4%)  ← Retention rate
```

**Quality Checks:**
✓ Retention rate >90% (good library quality)
⚠️ Retention rate 70-90% (acceptable)
✗ Retention rate <70% (poor quality, investigate)

---

### Step 4: Bismark Alignment

**What It Does:**
Aligns bisulfite-converted reads to bisulfite-converted reference genome.

**Bismark Alignment Strategy:**
```
Read:      5'- ATTGATTGATGTTGG -3'
Reference: 5'- ATCGATCGATGCGGG -3'
                ^  ^    ^  ^
                C→T conversions match

Bismark tries 4 alignments:
1. Read vs C→T forward reference
2. Read complement vs C→T forward reference  
3. Read vs G→A reverse reference
4. Read complement vs G→A reverse reference

Best unique alignment is reported.
```

**Alignment Parameters:**
```yaml
alignment:
  bowtie2: true              # Use Bowtie2 (recommended)
  multicore: 4               # Parallel processing (4× faster)
  score_min: "L,0,-0.2"      # Min alignment score
  non_directional: false     # Library type
  pbat: false                # Post-bisulfite adapter tagging
  local: false               # End-to-end alignment (default)
```

**Score_min Explanation:**
```
L,0,-0.2 means:
  Minimum score = 0 + (-0.2 × read_length)
  
For 150bp read:
  Min score = 0 + (-0.2 × 150) = -30
  
This allows ~15 mismatches (each mismatch ≈ -6 penalty)
Balances sensitivity (find BS-converted reads) vs specificity
```

**Output Files:**
- `sample_bismark_bt2.bam` (aligned reads)
- `sample_bismark_bt2_SE_report.txt` (or PE_report.txt)

**Expected Alignment Statistics:**
```
Bismark Alignment Report:
  Sequences analyzed: 147,888,889
  Unique alignment: 105,234,567 (71.2%)  ← Good for WGBS
  No alignment: 28,456,789 (19.2%)
  Ambiguous: 14,197,533 (9.6%)
  
  Mapping efficiency: 71.2%
  
  CT/CT alignments: 52,617,284 (50.0%)
  CT/GA alignments: 52,617,283 (50.0%)
```

**Quality Checks:**
✓ Mapping rate >60% for WGBS, >50% for RRBS
✓ Balanced CT/CT and CT/GA counts (directional library)
⚠️ Mapping rate 40-60% (acceptable but suboptimal)
✗ Mapping rate <40% (investigate: quality, reference mismatch)

**Strand Balance Interpretation:**
```
Directional Library (expected):
  CT/CT ≈ 50%  (original top strand)
  CT/GA ≈ 50%  (original bottom strand)

Non-directional Library (expected):
  CT/CT ≈ 25%
  CT/GA ≈ 25%
  GA/CT ≈ 25%
  GA/GA ≈ 25%
```

---

### Step 5: Deduplication

**What It Does:**
Removes PCR duplicate molecules that arise during library amplification.

**Why Duplicates Matter:**
```
True Biological Signal vs PCR Duplicates:

Molecule 1: ═══════════> (reads 5 cytosines)
Molecule 2: ═══════════> (reads 5 cytosines)
Molecule 3: ═══════════> (reads 5 cytosines)

PCR amplification of Molecule 1:
  Copy 1a: ═══════════>
  Copy 1b: ═══════════>  ← Duplicate (remove)
  Copy 1c: ═══════════>  ← Duplicate (remove)

Without deduplication:
  Methylation estimate biased by over-represented molecules
  
With deduplication:
  Each unique molecule counted once
  Accurate methylation quantification
```

**Deduplication Strategy:**
```
Single-End:
  Duplicates = same start position + same strand
  
Paired-End:
  Duplicates = same R1 start + same R2 start + same strand
  (More stringent, fewer false positives)

Position-based deduplication:
  chr1:12345 (forward) + chr1:12567 (reverse) = unique pair
  chr1:12345 (forward) + chr1:12567 (reverse) = duplicate!
```

**Expected Results:**
```
Deduplication Report:
  Total reads: 105,234,567
  Duplicates removed: 21,046,913 (20.0%)  ← Typical for WGBS
  Unique reads: 84,187,654 (80.0%)

Duplication rate by context:
  WGBS: 15-25% (lower complexity)
  RRBS: 30-50% (higher due to size selection)
```

**Quality Checks:**
✓ Duplication <40% (good library complexity)
⚠️ Duplication 40-60% (acceptable for RRBS)
✗ Duplication >60% (poor library, over-amplification)

---

### Step 6: Methylation Extraction

**What It Does:**
Determines methylation status for each cytosine in each context (CpG, CHG, CHH).

**Extraction Process:**
```
For each read aligned:

1. Compare read to reference:
   Reference: 5'- ATCGATCG -3'
   Read:      5'- ATTGATTG -3'
              Positions 2,6 are C in reference
              
2. Determine methylation:
   Position 2: Read has C → Methylated
   Position 6: Read has T → Unmethylated (BS-converted)

3. Context determination:
   Reference: 5'- ATCGATCG -3'
   Position 2: TCG → CpG context
   Position 6: TCG → CpG context
   
4. Output format:
   chr1  12345  +  CG  1  0  (1 methylated, 0 unmethylated)
   chr1  12348  +  CG  0  1  (0 methylated, 1 unmethylated)
```

**M-bias Analysis:**
```
M-bias Plot: Methylation level by read position

100% |                 
 90% |  ●●●●●●●●●●●●●●●●●●●●●●●  ← Expected pattern
 80% |                            (stable methylation)
 70% | 
     +─────────────────────────────>
       0   25   50   75  100  125 150
              Read Position

Problematic M-bias:
100% |     
 90% | ●●                          ← Position bias!
 80% |   ●●●●                      (ignore first 10bp)
 70% |       ●●●●●●●●●●●●●●●●●
     +─────────────────────────────>
       0   25   50   75  100  125 150

Action: Set --ignore 10 to exclude biased positions
```

**Configuration:**
```yaml
extraction:
  min_depth: 10              # Minimum coverage for calling
  ignore: 5                  # Ignore 5bp from 5' end
  ignore_3prime: 5           # Ignore 5bp from 3' end
  merge_non_cpg: false       # Separate CHG/CHH contexts
  comprehensive: true        # Extract all contexts
```

**Output Files:**
```
CpG_context_sample.txt.gz           # CpG methylation calls
CHG_context_sample.txt.gz           # CHG methylation calls  
CHH_context_sample.txt.gz           # CHH methylation calls
sample.bismark.cov.gz               # Coverage file
sample.bedGraph.gz                  # Genome browser track
sample.M-bias.txt                   # M-bias report
```

**Coverage File Format:**
```
# Format: chr  start  end  %methylation  count_methylated  count_unmethylated
chr1  12345  12346  85.5  100  17
chr1  12389  12390  92.3  120  10
chr1  12456  12457  5.2   5    91

Interpretation:
  Position chr1:12345 has 85.5% methylation
  Based on 100 methylated + 17 unmethylated reads = 117 total
```

**Quality Metrics:**
```
Extraction Summary:
  Total C's analyzed: 85,234,567,890
  CpG methylation: 75.2%  ← Expected ~70-80% in mammals
  CHG methylation: 1.2%   ← Expected <2% in mammals
  CHH methylation: 0.8%   ← Expected <1% in mammals
  
  Conversion rate: 99.6%  ← Critical metric!
  (calculated from CHH: 99.2% unmethylated = 99.2% conversion)
```

**Quality Checks:**
✓ Conversion rate >99.0% (excellent)
✓ CpG methylation 60-80% (normal mammalian)
✓ CHG/CHH <2% (normal mammalian)
⚠️ Conversion rate 95-99% (acceptable but lower confidence)
✗ Conversion rate <95% (incomplete BS conversion, retry)
✗ CHG/CHH >5% (incomplete conversion or plant sample)

---

### Step 7: Bismark Report

**What It Does:**
Generates comprehensive HTML report with all QC metrics and visualizations.

**Report Sections:**

**1. Alignment Summary:**
```
Mapping Efficiency: 71.2%
Unique alignments: 105,234,567
Ambiguous: 14,197,533
No alignment: 28,456,789

Strand distribution:
  CT/CT: 50.0% (original top strand)
  CT/GA: 50.0% (original bottom strand)
  
Interpretation: Balanced directional library ✓
```

**2. Cytosine Methylation:**
```
Context        Total C's    % Methylated    % Unmethylated
CpG            1,234,567    75.2%           24.8%
CHG              345,678     1.2%           98.8%
CHH            8,765,432     0.8%           99.2%

Global methylation: 75.2% (CpG context)
```

**3. M-bias Plots:**
```
Shows methylation % across read positions
Used to detect position-specific biases
Informs --ignore parameters if needed
```

**4. Duplication Rate:**
```
Sequences analyzed: 105,234,567
Duplicates removed: 21,046,913 (20.0%)
Sequences remaining: 84,187,654

Acceptable range: 10-40% for WGBS
```

**5. Conversion Efficiency:**
```
Calculated from CHH context:
CHH unmethylated: 99.2%
Conversion rate: 99.2% ✓

Minimum acceptable: 99.0%
```

---

### Step 8: DMR Calling (metilene)

**What It Does:**
Identifies differentially methylated regions (DMRs) between sample groups using statistical testing.

**DMR Analysis Strategy:**
```
Group A (e.g., tumor):  ████████████████ 85% methylation
Group B (e.g., normal): ████            30% methylation
                        ────────────────
DMR: 55% difference, statistically significant

metilene algorithm:
1. Segment genome into regions
2. Calculate mean methylation per group
3. Perform Mann-Whitney U test
4. Correct for multiple testing (FDR)
5. Filter by minimum difference & CpG count
```

**Configuration:**
```yaml
dmr_calling:
  min_diff: 0.1              # Minimum 10% methylation difference
  min_cpg: 10                # Minimum 10 CpGs in region
  max_dist: 300              # Maximum 300bp between CpGs
  threads: 8                 # Parallel processing

comparisons:
  - comparison: "tumor_vs_normal"
    group1: ["tumor_rep1", "tumor_rep2", "tumor_rep3"]
    group2: ["normal_rep1", "normal_rep2", "normal_rep3"]
```

**DMR Types:**
```
Hypermethylated DMRs:
  Group A > Group B
  Often: tumor > normal (gene silencing)
  Example: Tumor suppressor gene promoters
  
Hypomethylated DMRs:
  Group A < Group B  
  Often: tumor < normal (gene activation)
  Example: Oncogene promoters, repeat elements
```

**Output Format:**
```
# metilene output
chr  start   end     q-value  mean_diff  CpGs  group1_mean  group2_mean
chr1 12345   12789   0.001    0.456      15    0.823        0.367
chr1 45678   46123   0.003    -0.389     12    0.234        0.623
chr2 89012   89567   0.002    0.512      18    0.891        0.379

Interpretation:
Row 1: Hypermethylated in group1 (45.6% higher)
Row 2: Hypomethylated in group1 (38.9% lower) 
Row 3: Hypermethylated in group1 (51.2% higher)
```

**DMR Annotation:**
```
Typical DMR Distribution:
  Promoters: 30-40%  ← Most functionally relevant
  Gene bodies: 20-30%
  Intergenic: 20-30%
  Enhancers: 10-20%

Expected DMR counts:
  Tumor vs Normal: 5,000-50,000 DMRs
  Tissue comparison: 10,000-100,000 DMRs
  Cell type: 1,000-10,000 DMRs
```

**Quality Checks:**
✓ DMR count reasonable for comparison type
✓ Mean difference >10% (biologically relevant)
✓ Q-value <0.05 (FDR-corrected)
✓ Multiple CpGs per region (≥10)
⚠️ Too few DMRs (<100): increase sensitivity
⚠️ Too many DMRs (>100,000): apply stricter filters
✗ No DMRs: check sample groups, coverage, biological variance

---

### Step 9: MultiQC Report

**What It Does:**
Aggregates QC metrics from all samples and steps into a single interactive HTML report.

**MultiQC Sections:**

**1. General Statistics Table:**
```
Sample          Total Reads  % Aligned  % Dup  CpG Meth%  Conversion%
brain_wgbs_r1   150.2M       71.2%      20.0%  75.2%      99.6%
brain_wgbs_r2   148.7M       69.8%      21.3%  74.8%      99.5%
liver_wgbs_r1   145.3M       72.1%      19.2%  78.3%      99.7%
liver_wgbs_r2   147.9M       70.5%      20.8%  77.9%      99.6%
```

**2. FastQC Results:**
- Per-base quality plots (all samples overlaid)
- GC content distribution
- Sequence length distribution
- Adapter content

**3. Bismark Alignment:**
- Mapping efficiency comparison
- Strand distribution
- Conversion efficiency across samples

**4. Duplication Rates:**
- Bar chart of duplication rates
- Compare across samples
- Identify outliers

**5. Methylation Summary:**
- CpG methylation levels per sample
- Context distribution (CpG vs CHG vs CHH)
- Coverage statistics

**Interpreting MultiQC:**
```
Look for:
✓ Consistent metrics across biological replicates
✓ High conversion rates (>99%) in all samples
✓ Similar mapping rates within experiment type
✓ Expected methylation patterns (CpG >> CHG/CHH)

Red flags:
✗ Outlier samples (investigate or exclude)
✗ Low conversion rate (<99%)
✗ High adapter content (re-trim)
✗ Bimodal quality distributions (mixed libraries)
```

---

## Output Interpretation

### Key Output Files

**1. Deduplicated BAM Files**
```
sample.deduplicated.bam
sample.deduplicated.bam.bai

Use for:
- IGV visualization
- Custom analysis scripts
- Re-extraction with different parameters
```

**2. Coverage Files (.cov.gz)**
```
Format: chr  start  end  %meth  count_meth  count_unmeth

Use for:
- DMR analysis
- Regional methylation quantification
- Custom statistical tests
```

**3. bedGraph Tracks**
```
Format: chr  start  end  %methylation

Use for:
- UCSC Genome Browser visualization
- Integration with ChIP-seq or ATAC-seq
- Publication-quality figures
```

**4. DMR Files (BED)**
```
Format: chr  start  end  q-value  mean_diff  ...

Use for:
- Functional enrichment analysis
- Motif analysis in DMRs
- Pathway analysis
- Integration with gene expression
```

### Biological Interpretation

**CpG Island Methylation:**
```
Promoter CpG Islands:
  High methylation → Gene silencing
  Low methylation → Gene expression
  
Example:
  BRCA1 promoter DMR (hypermethylated in tumor)
  → BRCA1 gene silenced
  → Impaired DNA repair
  → Cancer progression
```

**Gene Body Methylation:**
```
Positive correlation with gene expression
  Moderately methylated gene bodies = active transcription
  
Mechanism:
  - Suppresses cryptic transcription start sites
  - Regulates alternative splicing
  - Prevents transposon activation
```

**Enhancer Methylation:**
```
Tissue-specific enhancers:
  Active tissue: Low methylation
  Inactive tissue: High methylation
  
Example:
  Liver-specific enhancers unmethylated in liver
  Same enhancers methylated in brain
```

**Global Hypomethylation:**
```
Observed in:
- Cancer (genome instability)
- Aging (epigenetic drift)
- Environmental stress

Consequences:
- Repeat element reactivation
- Chromosomal instability
- Aberrant gene expression
```

---

## Running the Pipeline

### 1. Download Test Data

```bash
# Use the provided downloader script
cd ~/BioPipelines
python scripts/download_methylation_test.py

# This will download:
# - 2 WGBS replicates from ENCODE
# - Stored in data/raw/methylation/
```

### 2. Configure the Pipeline

Edit `pipelines/methylation/bisulfite_analysis/config.yaml`:

```yaml
# Update sample names from downloaded data
samples:
  - brain_wgbs_rep1
  - brain_wgbs_rep2
  - liver_wgbs_rep1
  - liver_wgbs_rep2

# Define comparisons for DMR analysis
comparisons:
  - comparison: "brain_vs_liver"
    group1: ["brain_wgbs_rep1", "brain_wgbs_rep2"]
    group2: ["liver_wgbs_rep1", "liver_wgbs_rep2"]

# Adjust parameters if needed (defaults are usually good)
trimming:
  quality: 20
  min_length: 50
  rrbs: false  # Set to true for RRBS data

alignment:
  multicore: 4
  score_min: "L,0,-0.2"

extraction:
  min_depth: 10
  ignore: 5
  ignore_3prime: 5

dmr_calling:
  min_diff: 0.1
  min_cpg: 10
  max_dist: 300
```

### 3. Submit the Job

```bash
# Submit to SLURM
sbatch scripts/submit_methylation.sh

# Check job status
squeue -u $USER

# Monitor progress
tail -f slurm-*.out

# Check for errors
tail -f slurm-*.err
```

### 4. Expected Runtime

```
For human WGBS (30× coverage, ~150M reads):
  Index preparation: 2-4 hours (one-time)
  Per sample:
    - FastQC: 10 min
    - Trimming: 15 min
    - Alignment: 4-6 hours
    - Deduplication: 45 min
    - Extraction: 90 min
    - Report: 5 min
  DMR calling: 2-3 hours
  MultiQC: 5 min
  
Total: ~8-10 hours per sample + DMR time

For RRBS (50M reads):
  Per sample: ~2-3 hours
  Total: ~3-4 hours per sample + DMR time
```

### 5. Monitor Resource Usage

```bash
# Check memory and CPU usage
sacct -j <job_id> --format=JobID,MaxRSS,MaxVMSize,Elapsed,CPUTime

# Expected resources:
# Memory: 32-64GB (alignment step)
# CPUs: 16 cores (with multicore=4)
# Time: 24 hours for WGBS
```

---

## Troubleshooting

### Issue 1: Low Mapping Rate (<60%)

**Symptoms:**
- Bismark alignment <60% for WGBS or <50% for RRBS
- Many reads in "no alignment" category

**Possible Causes:**
1. Wrong reference genome
2. Adapter contamination
3. Poor library quality
4. Incorrect library type parameter

**Solutions:**
```bash
# Check reference genome matches your organism
grep "^>" data/references/hg38.fa | head

# Re-run FastQC for adapter contamination
fastqc data/raw/methylation/sample.fastq.gz

# Try more relaxed alignment parameters
# In config.yaml:
alignment:
  score_min: "L,0,-0.4"  # More permissive

# Check if library is non-directional
# If CT/GA ≈ CT/CT ≈ GA/CT ≈ GA/GA, set:
alignment:
  non_directional: true
```

### Issue 2: Low Conversion Rate (<99%)

**Symptoms:**
- Conversion efficiency <99% in Bismark report
- High CHH/CHG methylation (>2%)

**Possible Causes:**
1. Incomplete bisulfite conversion
2. Contamination with unconverted DNA
3. Poor bisulfite treatment

**Solutions:**
```bash
# This is a WET LAB issue - requires re-sequencing
# Check CHH methylation per sample:
grep "CHH" data/results/methylation/bismark_reports/sample_report.txt

# If only one sample affected:
# - Exclude from analysis
# - Request re-sequencing

# If all samples affected:
# - Contact sequencing facility
# - May need new bisulfite conversion
```

### Issue 3: No DMRs Found

**Symptoms:**
- metilene output file is empty or has very few DMRs
- No significant differences detected

**Possible Causes:**
1. Insufficient biological difference
2. Low coverage
3. Too stringent thresholds
4. Wrong sample groupings

**Solutions:**
```bash
# Relax DMR calling thresholds
# In config.yaml:
dmr_calling:
  min_diff: 0.05      # Lower from 0.1
  min_cpg: 5          # Lower from 10
  max_dist: 500       # Increase from 300

# Check coverage in samples
samtools depth sample.deduplicated.bam | awk '{sum+=$3} END {print sum/NR}'

# Should be ≥10× for DMR calling
# If lower, consider:
# - Merging replicates
# - Lowering min_depth in extraction

# Verify sample groups make biological sense
# Check PCA plot of global methylation to confirm grouping
```

### Issue 4: M-bias Detected

**Symptoms:**
- M-bias plot shows declining/increasing methylation at read ends
- Warning in Bismark report about position bias

**Possible Causes:**
1. End-repair bias (RRBS)
2. Adapter sequence bias
3. Sequencing quality issues

**Solutions:**
```bash
# Add --ignore flags to exclude biased positions
# In config.yaml:
extraction:
  ignore: 10          # Ignore first 10bp
  ignore_3prime: 10   # Ignore last 10bp

# For RRBS, use --rrbs flag in trimming:
trimming:
  rrbs: true

# Re-run extraction step only:
snakemake --snakefile pipelines/methylation/bisulfite_analysis/Snakefile \\
          --configfile pipelines/methylation/bisulfite_analysis/config.yaml \\
          --use-conda \\
          --forcerun extract_methylation
```

### Issue 5: High Duplication Rate (>60%)

**Symptoms:**
- >60% duplicates removed
- Low library complexity

**Possible Causes:**
1. Over-amplification during PCR
2. Low input DNA amount
3. Size selection too narrow (RRBS)

**Possible Solutions:**
```bash
# RRBS: High duplication expected (30-50%), may be acceptable
# WGBS: >40% suggests library issues

# Check if biological replicates are similar:
# If all samples high: may be expected for the protocol
# If only one sample: technical issue, consider excluding

# For future libraries:
# - Reduce PCR cycles
# - Increase input DNA amount
# - Widen size selection range (RRBS)

# Can still proceed with analysis if conversion rate is good
```

### Issue 6: Job Killed or Out of Memory

**Symptoms:**
- SLURM job shows "OUT_OF_MEMORY" or "CANCELLED"
- Job terminated during alignment step

**Solutions:**
```bash
# Increase memory in submit script
# Edit scripts/submit_methylation.sh:
#SBATCH --mem=128G  # Increase from 64G

# Or reduce multicore value (uses less memory)
# In config.yaml:
alignment:
  multicore: 2  # Reduce from 4

# Can also split large samples:
# Use seqtk to split FASTQ files:
seqtk sample -s100 sample.fastq.gz 0.5 > sample_half1.fastq
seqtk sample -s200 sample.fastq.gz 0.5 > sample_half2.fastq
# Analyze separately, then merge results
```

### Issue 7: Snakemake Locked

**Symptoms:**
- Error: "Directory cannot be locked"
- Pipeline won't start

**Solutions:**
```bash
# Unlock the directory
cd pipelines/methylation/bisulfite_analysis
snakemake --unlock

# Or clean up locks manually
rm -rf .snakemake/locks

# Then resubmit:
sbatch scripts/submit_methylation.sh
```

---

## Advanced Topics

### Integrating with Other Data Types

**Methylation + RNA-seq:**
```
Question: Do hypermethylated promoters show reduced expression?

Analysis:
1. Identify promoter DMRs
2. Overlap with gene TSS (±2kb)
3. Correlate methylation with RNA-seq expression
4. Expected: Negative correlation for promoters
```

**Methylation + ChIP-seq:**
```
Question: Are H3K4me3 marks anti-correlated with methylation?

Analysis:
1. Calculate methylation in H3K4me3 peaks
2. Compare to methylation in non-peak regions
3. Expected: Lower methylation in active promoters
```

**Methylation + ATAC-seq:**
```
Question: Is chromatin accessibility related to methylation?

Analysis:
1. Calculate methylation in ATAC peaks
2. Compare to methylation in closed chromatin
3. Expected: Lower methylation in accessible regions
```

### Custom Analysis Scripts

**Extract promoter methylation:**
```bash
# Get promoters (TSS ±2kb)
bedtools slop -i genes.bed -g chrom.sizes -l 2000 -r 0 > promoters.bed

# Calculate mean methylation
bedtools map -a promoters.bed -b sample.bismark.cov.gz \\
             -c 4 -o mean > promoter_methylation.txt
```

**Identify tissue-specific DMRs:**
```bash
# Intersect DMRs with tissue-specific enhancers
bedtools intersect -a dmrs.bed -b tissue_enhancers.bed -wa -wb \\
                   > tissue_specific_dmrs.bed
```

---

## Citation

If you use this pipeline, please cite:

**Bismark:**
Krueger F, Andrews SR. Bismark: a flexible aligner and methylation caller for Bisulfite-Seq applications. Bioinformatics. 2011;27(11):1571-2.

**Trim Galore:**
Krueger F. Trim Galore: A wrapper around Cutadapt and FastQC. 2015.

**metilene:**
Jühling F, et al. metilene: fast and sensitive calling of differentially methylated regions from bisulfite sequencing data. Genome Res. 2016;26(2):256-62.

---

## Support

For issues or questions:
1. Check the [Bismark documentation](https://github.com/FelixKrueger/Bismark)
2. Review the [WGBS best practices guide](https://www.encodeproject.org/data-standards/wgbs/)
3. Open an issue on the BioPipelines GitHub repository

**Last updated:** November 2025
