# Hi-C 3D Genome Analysis Tutorial

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

Hi-C is a genome-wide chromosome conformation capture technique that maps three-dimensional genome organization. This pipeline analyzes Hi-C sequencing data to identify topologically associating domains (TADs), chromatin loops, and A/B compartments.

### What You'll Learn
- How Hi-C captures 3D genome architecture
- Understanding contact matrices and resolution
- Interpreting TADs, loops, and compartments
- Quality control metrics for Hi-C experiments
- Normalization strategies (ICE, KR, VC)

### Prerequisites
- Basic understanding of 3D genome organization
- Familiarity with FASTQ and paired-end sequencing
- Hi-C sequencing data (paired-end required)

---

## Biological Background

### 3D Genome Organization

DNA is not randomly organized in the nucleus—it has hierarchical 3D structure:

```
Hierarchical Genome Organization:

1. Chromosome Territories (Mb scale)
   ┌─────────────────────────────────┐
   │ Chr 1                            │
   │  ╭──╮   ╭───╮      ╭──╮        │
   │  │  │   │   │      │  │         │
   │  ╰──╯   ╰───╯      ╰──╯         │
   └─────────────────────────────────┘

2. A/B Compartments (1-10 Mb)
   A: Open chromatin, gene-rich, active
   B: Closed chromatin, gene-poor, inactive
   
   ████████░░░░░░░░████████░░░░░░░░
   A regions      B regions

3. TADs - Topologically Associating Domains (200kb-1Mb)
   Self-interacting chromatin regions
   
   ╔═══════════╗  ╔═══════════╗
   ║  TAD 1    ║  ║  TAD 2    ║
   ║    ↔↔↔    ║  ║    ↔↔↔    ║
   ╚═══════════╝  ╚═══════════╝
   
4. Chromatin Loops (10-1000kb)
   Enhancer-promoter interactions
   
        Enhancer ─────╮
                      │ Loop
         Promoter ────╯
              ↓
         Active Gene

5. DNA-Protein Interactions (bp-kb scale)
   CTCF, Cohesin, Transcription factors
```

### How Hi-C Works

Hi-C uses proximity ligation to capture spatial interactions:

```
Hi-C Protocol Steps:

1. Crosslink DNA and Proteins
   ═══════════════    ═══════════════
   Locus A            Locus B
   (physically close in nucleus)

2. Restriction Enzyme Digestion (e.g., MboI: ^GATC)
   ═══════^GATC      GATC^═══════
   
3. Biotin Fill-in of Overhangs
   ═══════^GATC*     *CTAG^═══════
         (*biotin-dCTP marked)

4. Proximity Ligation (blunt-end ligation)
   ═══════GATC*─*GATC═══════
   Ligation Junction (chimeric molecule)
   
5. Reverse Crosslink, Fragment, Pull down Biotin
   Enrich for ligation products

6. Paired-End Sequencing
   Read 1 ──────>    Ligation      <────── Read 2
               Junction
   (from Locus A)              (from Locus B)

7. Map Reads to Genome
   Read 1 → chr1:12345 (Locus A)
   Read 2 → chr1:98765 (Locus B)
   
   Contact: chr1:12345 <─> chr1:98765
```

**Key Insight:**
Physically close DNA regions produce paired-end reads from different genomic loci. Read pair frequency indicates interaction frequency.

### Contact Matrix

Hi-C data is represented as a contact matrix:

```
Contact Matrix (symmetric, chr1 self-interactions):

Position  0kb   50kb  100kb 150kb 200kb 250kb 300kb
0kb      [100]  [50]  [20]  [5]   [2]   [1]   [0]
50kb      [50] [100]  [50]  [20]  [5]   [2]   [1]
100kb     [20]  [50] [100]  [50]  [20]  [5]   [2]
150kb     [5]   [20]  [50] [100]  [50]  [20]  [5]
200kb     [2]   [5]   [20]  [50] [100]  [50]  [20]
250kb     [1]   [2]   [5]   [20]  [50] [100]  [50]
300kb     [0]   [1]   [2]   [5]   [20]  [50] [100]

Observations:
- Diagonal = highest signal (self-interactions)
- Signal decays with distance
- Off-diagonal peaks = loops
```

**Visual Representation:**
```
     0kb       100kb      200kb      300kb
0kb   ████      ▓▓░        ░          
      ████      ▓▓░        ░          
100kb ▓▓░       ████       ▓▓░        ░
      ▓▓░       ████       ▓▓░        ░
200kb ░         ▓▓░        ████       ▓▓
      ░         ▓▓░        ████       ▓▓
300kb           ░          ▓▓         ████
                ░          ▓▓         ████

Legend: ████ = high contact, ▓▓ = medium, ░ = low
```

### TADs (Topologically Associating Domains)

TADs are self-interacting genomic regions with boundaries:

```
TAD Structure in Contact Matrix:

     TAD 1              TAD 2              TAD 3
   ╔════════╗         ╔════════╗         ╔════════╗
   ║████████║         ║        ║         ║        ║
   ║████████║         ║        ║         ║        ║
   ║████████║         ║        ║         ║        ║
   ║████████║ ░░░░░░░ ║████████║ ░░░░░░░ ║████████║
   ╚════════╝         ║████████║         ║████████║
                      ║████████║         ║████████║
                      ║████████║         ║████████║
                      ╚════════╝         ╚════════╝
                      
Features:
- High intra-TAD contacts (dark squares on diagonal)
- Low inter-TAD contacts (light areas between TADs)
- Sharp boundaries at TAD edges
```

**TAD Boundary Features:**
- Enriched for CTCF binding sites
- Enriched for cohesin complexes
- Often coincide with active promoters
- Conserved across cell types (but not absolute)

**TAD Biological Functions:**
- Constrain enhancer-promoter interactions
- Prevent aberrant gene activation
- Organize chromatin into regulatory domains
- Disruption linked to developmental disorders and cancer

### Chromatin Loops

Loops are specific long-range interactions, often enhancer-promoter:

```
Loop Detection in Contact Matrix:

Position
    ▓▓▓▓
    ▓▓▓▓
    ▓▓▓▓
    ▓▓▓▓     ←───┐
             ████│  Loop Peak
    ▓▓▓▓     ████│  (off-diagonal)
    ▓▓▓▓     └───┘
    ▓▓▓▓
    ▓▓▓▓
         Position

Genomic View:

Chr1: ═══╦═══════════════════════╦═══
         ║                       ║
      Enhancer                Promoter
         ╚═══════════════════════╝
               Loop (300kb)
                  ↓
              Active Gene
```

**Loop Types:**
1. **Enhancer-Promoter Loops**
   - Gene activation
   - Tissue-specific
   - Disrupted in disease

2. **CTCF-mediated Loops**
   - Structural (not regulatory)
   - Form TAD boundaries
   - Convergent CTCF motifs

3. **Polycomb Loops**
   - Gene repression
   - Developmental genes
   - PRC1/PRC2 complexes

### A/B Compartments

Large-scale organization into active (A) and inactive (B) compartments:

```
Compartment Detection via PCA:

Contact Matrix → PCA → Eigenvector 1 (PC1)

PC1 > 0: A compartment (active)
PC1 < 0: B compartment (inactive)

Chromosome View:

PC1:  ████░░░░░░████████░░░░████░░░░░█████
      ++++------+++++++++----+++++-----+++++
      AAAA BBBB  AAAAAAA BBBB AAAAA BBB AAAAA

Characteristics:
A Compartment:
  - Gene-rich
  - Open chromatin
  - Active histone marks (H3K4me3, H3K27ac)
  - High transcription
  - Nuclear center
  
B Compartment:
  - Gene-poor
  - Closed chromatin
  - Repressive marks (H3K9me3, H3K27me3)
  - Low transcription
  - Nuclear periphery
```

**Compartment Dynamics:**
- Change during differentiation
- Cell-type specific
- Associated with lamina-associated domains (LADs)
- A→B switch silences genes; B→A activates

### Resolution Considerations

Hi-C resolution determines what structures can be detected:

```
Resolution Ladder:

1 Mb resolution:
  - Detect: Compartments, large TADs
  - Cannot detect: Loops, precise TAD boundaries
  - Reads needed: 10-50M valid pairs

100 kb resolution:
  - Detect: TADs, compartments
  - Cannot detect: Most loops
  - Reads needed: 50-200M valid pairs

10 kb resolution:
  - Detect: Loops, precise TAD boundaries
  - Reads needed: 200M-1B valid pairs

5 kb resolution:
  - Detect: High-resolution loops
  - Needed for: Regulatory element analysis
  - Reads needed: 1-5B valid pairs

Trade-off:
  Higher resolution = More specific features
  BUT requires exponentially more sequencing
```

---

## Pipeline Overview

### Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HI-C ANALYSIS PIPELINE                       │
│                   (3D Genome Organization Analysis)                  │
└─────────────────────────────────────────────────────────────────────┘

Input: Paired-end FASTQ files (R1, R2)
  ↓
┌─────────────────────────────────────┐
│ 1. FastQC (Raw Reads)               │
│    • Quality scores                 │
│    • Adapter content                │
│    • GC distribution                │
│    Time: ~10 min/sample             │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 2. Trim Reads (fastp)               │
│    • Adapter removal                │
│    • Quality trimming (Q≥20)        │
│    • Length filtering (≥50bp)       │
│    Time: ~15-30 min/sample          │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 3. Align Reads (Bowtie2)            │
│    • Map R1 and R2 independently    │
│    • Very sensitive mode            │
│    • No discordant/mixed pairs      │
│    • Sort by read name              │
│    Time: ~2-4 hours/sample          │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 4. Parse Pairs (pairtools)          │
│    • Detect ligation junctions      │
│    • Classify pair types            │
│    • Sort by position               │
│    • Remove duplicates              │
│    • Generate statistics            │
│    Time: ~1-2 hours/sample          │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 5. Make Contact Matrix (cooler)     │
│    • Create multi-resolution .mcool │
│    • Resolutions: 5kb to 1Mb        │
│    • ICE normalization              │
│    Time: ~30-60 min/sample          │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 6. Call TADs (HiCExplorer)          │
│    • Insulation score calculation   │
│    • Detect domain boundaries       │
│    • Output TAD coordinates         │
│    Time: ~30-60 min/sample          │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 7. Call Loops (chromosight)         │
│    • Pattern recognition            │
│    • Statistical significance       │
│    • FDR correction                 │
│    Time: ~1-2 hours/sample          │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 8. Call Compartments (PCA)          │
│    • Calculate eigenvectors         │
│    • Assign A/B compartments        │
│    • Generate bigWig tracks         │
│    Time: ~15-30 min/sample          │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 9. Plot Matrix (HiCExplorer)        │
│    • Visualize contact matrix       │
│    • Multiple normalizations        │
│    • Annotate features              │
│    Time: ~10-20 min/sample          │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 10. MultiQC Report                  │
│     • Aggregate QC metrics          │
│     • Interactive visualizations    │
│     Time: ~5 min                    │
└─────────────────────────────────────┘
  ↓
Output Files:
  • Multi-resolution contact matrices (.mcool)
  • TAD domains and boundaries (BED)
  • Chromatin loops (BEDPE)
  • A/B compartments (BED, bigWig)
  • Contact matrix plots (PNG)
  • MultiQC report (HTML)
```

### Pipeline Features

**Pair Classification:**
- Valid pairs (cis/trans, near/far)
- Invalid pairs (unmapped, low quality)
- Duplicate pairs (PCR/optical)
- Detailed statistics

**Multi-Resolution Analysis:**
- 8 resolutions: 5kb, 10kb, 25kb, 50kb, 100kb, 250kb, 500kb, 1Mb
- Appropriate resolution auto-selected for each analysis
- Balanced for sensitivity vs specificity

**Normalization Methods:**
- ICE (Iterative Correction and Eigenvalue)
- KR (Knight-Ruiz)
- VC (Vanilla Coverage)
- VC_SQRT (Square root Vanilla Coverage)

---

## Step-by-Step Walkthrough

### Step 1: FastQC on Raw Reads

**What It Does:**
Initial quality assessment of Hi-C sequencing data.

**Hi-C Specific Patterns:**
```
Expected for Hi-C Data:

1. Per Base Quality:
   Should be high (Q≥30) across read length
   Some decline at ends acceptable

2. GC Content:
   Should match genomic GC% (~42% for human)
   Bimodal distribution may indicate contamination

3. Duplication Level:
   Higher than genomic sequencing (20-40%)
   Expected: Limited complexity in Hi-C libraries

4. Adapter Content:
   Should be low (<5% at read ends)
   Illumina adapter sequences most common

5. Kmer Content:
   Enzyme recognition site may be enriched
   Example: GATC for MboI-digested libraries
```

**Quality Checks:**
✓ Mean quality >Q28
✓ GC content 40-45% (human)
✓ Duplication <50%
✓ Adapter content <10%

---

### Step 2: Trim Reads (fastp)

**What It Does:**
Remove adapters and low-quality bases from paired-end reads.

**Hi-C Trimming Strategy:**
```
Why Aggressive Trimming is OK:

Hi-C reads only need to uniquely map
  → Don't need full read length
  → Quality trumps length

Typical trimming:
  Original: 150bp reads
  After trim: 50-140bp (variable)
  
As long as ≥30bp remains:
  → Can uniquely map to genome
  → Preserve contact information
```

**Configuration:**
```yaml
trimming:
  quality: 20              # Trim bases <Q20
  min_length: 50           # Minimum read length
  adapter_auto_detect: true
```

**Expected Results:**
```
fastp Summary:
  Total read pairs: 200,000,000
  Reads passing: 195,000,000 (97.5%)
  Reads too short: 5,000,000 (2.5%)
  
  R1 mean length: 138bp (originally 150bp)
  R2 mean length: 135bp (originally 150bp)
  
  Adapters removed: 15% of reads
```

**Quality Checks:**
✓ Retention rate >90%
✓ Mean length >50bp
✓ Adapter removal <30%

---

### Step 3: Align Reads (Bowtie2)

**What It Does:**
Independently maps R1 and R2 to the reference genome.

**Why Independent Mapping?**
```
Hi-C reads are chimeric (ligation products):

Normal paired-end:
  R1 ────> <──── R2  (same fragment, 300-500bp apart)
  
Hi-C paired-end:
  R1 ────> (chr1:12345)
  R2 <──── (chr3:98765)  (different loci, Mb-Gb apart!)

Therefore:
  - Cannot use insert size constraints
  - Map R1 and R2 independently
  - Pair information reconstructed later
```

**Bowtie2 Parameters:**
```yaml
alignment:
  mode: "--very-sensitive"  # Stringent to reduce mismapping
  no_discordant: true       # No rescue of discordant pairs
  no_mixed: true            # No unpaired alignments in PE mode
  mapq: 30                  # Minimum quality score
```

**Alignment Strategy:**
```
Bowtie2 --very-sensitive:
  -D 20 -R 3 -N 0 -L 20 -i S,1,0.50
  
  -D 20: Try 20 seed extensions
  -R 3: Re-seed 3 times  
  -N 0: No mismatches in seed
  -L 20: Seed length 20bp
  -i S,1,0.50: Seed interval function

Result: High sensitivity, low mismapping
```

**Expected Results:**
```
Alignment Summary:
  Total read pairs: 195,000,000
  
  R1 mapped: 175,500,000 (90.0%)
  R2 mapped: 175,500,000 (90.0%)
  
  Both R1 and R2 mapped: 166,725,000 (85.5%)
  → These become Hi-C pairs
  
  Unmapped: ~10% (expected for Hi-C)
    - Ligation junctions span alignment
    - Chimeric reads
    - Low quality regions
```

**Quality Checks:**
✓ Mapping rate >80% for each mate
✓ Both mates mapped >70%
⚠️ Mapping <70%: check quality, reference
✗ Mapping <50%: investigate thoroughly

---

### Step 4: Parse Pairs (pairtools)

**What It Does:**
Processes aligned reads into valid Hi-C pairs with extensive QC.

**pairtools Workflow:**
```
1. pairtools parse:
   BAM → .pairs format
   - Detect ligation junctions
   - Classify pair types
   - Assign fragments to restriction sites
   
2. pairtools sort:
   Sort by chromosome and position
   
3. pairtools dedup:
   Remove PCR/optical duplicates
   Keep one representative per unique pair

Output: Clean valid pairs ready for matrix construction
```

**Pair Classification:**
```
Valid Pairs (keep for analysis):
  
  1. Cis pairs (same chromosome):
     ═══R1═══     ═══R2═══
     chr1:1000    chr1:5000
     
     Sub-types:
     - Near (<20kb): ~10-20% of cis
     - Far (>20kb): ~80-90% of cis
  
  2. Trans pairs (different chromosomes):
     ═══R1═══     ═══R2═══
     chr1:1000    chr5:8000

Invalid Pairs (exclude):
  
  1. Unmapped: R1 or R2 failed alignment
  2. Multi-mapped: Ambiguous mapping
  3. Same fragment: R1 and R2 on same restriction fragment
     (likely undigested or self-ligation)
  4. Duplicates: Identical genomic positions
```

**Restriction Fragment Assignment:**
```
MboI Recognition: ^GATC (cuts at ^)

Genome:  ═══^GATC═════════^GATC═══
         Fragment 1    Fragment 2

R1 maps to Fragment 1, R2 maps to Fragment 2
→ Valid inter-fragment pair
→ Represents true contact

R1 and R2 both map to Fragment 1
→ Invalid intra-fragment pair
→ Likely undigested DNA or self-circle
→ Exclude from analysis
```

**Expected Statistics:**
```
pairtools stats:
  Total pairs: 166,725,000
  
  Valid pairs: 108,371,250 (65%)
    ├─ Cis >20kb: 65,022,750 (60% of valid)
    ├─ Cis <20kb: 21,674,250 (20% of valid)
    └─ Trans: 21,674,250 (20% of valid)
  
  Invalid pairs: 33,345,000 (20%)
    ├─ Same fragment: 16,672,500 (10%)
    ├─ Low MAPQ: 8,336,250 (5%)
    └─ Other: 8,336,250 (5%)
  
  Duplicates: 25,008,750 (15%)

Key Metrics:
  Valid pairs%: 65% ✓ (good)
  Cis/Trans ratio: 4:1 ✓ (expected)
  Duplicate rate: 15% ✓ (acceptable)
```

**Quality Interpretation:**
```
Valid Pairs % (most important):
  >60%: Excellent library
  40-60%: Good library
  20-40%: Acceptable (low depth)
  <20%: Poor library (investigate)

Cis/Trans Ratio:
  Human: ~4:1 to 5:1 (cis > trans)
  Too high (>10:1): possible contamination
  Too low (<2:1): poor ligation

Duplicate Rate:
  10-30%: Normal for Hi-C
  >50%: Over-amplification or low complexity

Cis Long-Range % (cis >20kb / all cis):
  >70%: Excellent (good ligation efficiency)
  50-70%: Good
  <50%: Poor (many self-ligations)
```

---

### Step 5: Make Contact Matrix (cooler)

**What It Does:**
Converts valid pairs into multi-resolution contact matrices.

**Cooler Format:**
```
Cooler stores Hi-C data efficiently:

Structure:
  .cool file (single resolution)
  .mcool file (multiple resolutions, used here)

Contents:
  - Bin table: Genomic bins (e.g., chr1:0-5000)
  - Pixel table: Contacts between bins
  - Normalization vectors: ICE, KR weights

Advantages:
  - Fast random access
  - Space efficient (sparse matrix)
  - Standard format (widely supported)
```

**Multi-Resolution Strategy:**
```
Pipeline creates 8 resolutions:

5kb:     High-resolution loops, precise boundaries
10kb:    Loops, regulatory interactions
25kb:    TADs, loop clusters
50kb:    TADs, broad loops
100kb:   TADs, compartments (optimal)
250kb:   Large TADs, compartments
500kb:   Compartments, large-scale structure
1Mb:     Chromosomal territories, broad compartments

Usage:
  Loop calling: 5-10kb
  TAD calling: 25-100kb
  Compartments: 100kb-1Mb
```

**ICE Normalization:**
```
Why Normalize?

Raw contact matrix has biases:
  - Restriction site density
  - GC content
  - Mappability
  - Distance-dependent decay

ICE (Iterative Correction):
  1. Assume each bin should have equal visibility
  2. Iteratively balance rows and columns
  3. Converges to bias-corrected matrix

Before ICE:
  Bin A: 1000 contacts (high GC, many sites)
  Bin B: 100 contacts (low GC, few sites)
  
After ICE:
  Bin A: 500 contacts (down-weighted)
  Bin B: 150 contacts (up-weighted)
  
Result: Biases removed, true contacts visible
```

**Expected Output:**
```
Cooler Creation:
  Input: 108,371,250 valid pairs
  
  Contact matrices created:
    sample_5kb.cool:    5,000bp bins
    sample_10kb.cool:   10,000bp bins
    sample_25kb.cool:   25,000bp bins
    sample_50kb.cool:   50,000bp bins
    sample_100kb.cool:  100,000bp bins
    sample_250kb.cool:  250,000bp bins
    sample_500kb.cool:  500,000bp bins
    sample_1000kb.cool: 1,000,000bp bins
  
  Combined into: sample.mcool
  
  ICE normalization: Applied to all resolutions
  KR normalization: Applied to all resolutions
```

**Quality Checks:**
✓ All resolutions created successfully
✓ ICE/KR convergence achieved
✓ No extreme outlier bins
✓ Distance decay profile looks smooth

---

### Step 6: Call TADs (HiCExplorer)

**What It Does:**
Identifies topologically associating domains using insulation score method.

**Insulation Score Method:**
```
Principle:
  TAD boundaries have reduced contact across them
  → "Insulate" adjacent TADs from each other

Calculation:
  For each bin, measure contacts crossing it:
  
  Window = 500kb (typical)
  
     TAD 1      Boundary      TAD 2
  ════════════╱          ╲════════════
              ╲          ╱
               ↓ Low contacts ↓
  
  Insulation score = sum of contacts in square window
  Boundary = local minimum of insulation score

Visual:
  High Insulation ─────┐      ┌─────
                       │      │
  Low Insulation       ╰──────╯
                        ↑
                     Boundary
```

**TAD Calling Process:**
```
1. Calculate insulation score per bin (100kb resolution)
2. Smooth with median filter
3. Identify local minima (boundaries)
4. Define TADs between boundaries
5. Statistical significance testing
6. Filter by FDR threshold (0.05)

Parameters:
  Resolution: 100kb (optimal for TADs)
  Window: 60-120kb (typical TAD size range)
  FDR: 0.05 (5% false discovery rate)
```

**Expected Results:**
```
TAD Calling Summary:
  Total TADs identified: 3,247
  
  TAD size distribution:
    Min: 40 kb
    Median: 880 kb
    Max: 5.2 Mb
    
  Boundaries: 3,248
  
  TAD strength (insulation dip):
    Mean: -0.45
    Strong boundaries (< -0.6): 892
    Weak boundaries (> -0.3): 654
```

**TAD Interpretation:**
```
Strong Boundaries:
  - Deep insulation minima
  - Coincide with CTCF/cohesin sites
  - Conserved across cell types
  - Likely structural (not dynamic)

Weak Boundaries:
  - Shallow insulation minima
  - May be cell-type specific
  - Often regulatory (tissue-specific)
  - More variable across replicates

TAD Size Variation:
  Small TADs (<500kb):
    - Often gene-rich
    - Multiple regulatory elements
    - Active chromatin
    
  Large TADs (>2Mb):
    - Often gene-poor
    - Inactive chromatin
    - Fewer regulatory interactions
```

**Quality Checks:**
✓ TAD count 2,000-5,000 (human genome)
✓ Median size 500kb-1.5Mb
✓ Boundaries enriched for CTCF (if ChIP available)
⚠️ <1,000 TADs: low resolution or quality
⚠️ >10,000 TADs: over-calling, adjust threshold
✗ No TADs: insufficient valid pairs

---

### Step 7: Call Loops (chromosight)

**What It Does:**
Detects chromatin loops using pattern recognition and statistical testing.

**Loop Detection Principle:**
```
Loop Signature in Contact Matrix:

Without loop:
     ▓▓▓▓
     ▓▓▓▓
     ▓▓▓▓  (smooth diagonal decay)
     ▓▓▓▓

With loop:
     ▓▓▓▓
     ▓▓▓▓    ████ ← Peak (loop)
     ▓▓▓▓    ████
     ▓▓▓▓

Chromosight detects:
  - Bright pixel off-diagonal
  - Flanked by depleted regions
  - Statistically significant vs background
```

**Loop Calling Algorithm:**
```
1. Kernel Convolution:
   Apply loop-detection kernel to matrix
   Enhances loop-like patterns
   
2. Background Estimation:
   Calculate expected contact frequency
   Account for distance decay
   
3. Statistical Testing:
   Compare observed vs expected
   Generate p-values
   
4. FDR Correction:
   Control false discovery rate
   Threshold at 0.1 (10% FDR)
   
5. Filtering:
   - Min distance: 15kb (avoid noise)
   - Max distance: 2Mb (typical loop range)
   - Min enrichment: 2-fold
```

**Expected Results:**
```
Loop Calling Summary:
  Total loops detected: 8,456
  
  Loop distance distribution:
    <50kb: 2,537 (30%)   → Enhancer-promoter
    50-200kb: 3,382 (40%) → Regulatory
    200kb-1Mb: 2,114 (25%) → Structural
    >1Mb: 423 (5%)        → Long-range

  Loop strength:
    Mean enrichment: 3.2-fold
    Strong loops (>5×): 1,691
    
  Overlap with CTCF peaks (if available):
    Both anchors have CTCF: 65%
    One anchor has CTCF: 25%
    No CTCF: 10%
```

**Loop Categories:**
```
1. Enhancer-Promoter Loops (<50kb):
   Enhancer ═════╮
                 │ Short loop
   Promoter ═════╯
   
   Features:
   - H3K27ac at both anchors
   - Tissue-specific
   - Mediated by Mediator complex

2. CTCF-CTCF Loops (50kb-1Mb):
   CTCF→ ════════╮
                 │ Convergent
   ←CTCF ════════╯
   
   Features:
   - Convergent CTCF motifs
   - Cohesin-mediated
   - Form TAD boundaries
   - Conserved across tissues

3. Polycomb Loops (variable):
   PRC2 ═════════╮
                 │ Repressive
   PRC2 ═════════╯
   
   Features:
   - H3K27me3 at anchors
   - Developmental genes
   - Dynamic during differentiation
```

**Quality Checks:**
✓ Loop count 5,000-15,000 (human, GM12878)
✓ Enrichment >2-fold for most loops
✓ CTCF enrichment at anchors (if available)
⚠️ <1,000 loops: low resolution or quality
⚠️ >50,000 loops: over-calling, increase FDR
✗ No loops: insufficient valid pairs or resolution

---

### Step 8: Call Compartments (PCA)

**What It Does:**
Identifies A (active) and B (inactive) compartments using principal component analysis.

**PCA on Contact Matrix:**
```
Principle:
  Regions in same compartment interact more
  
  A-A contacts: High
  B-B contacts: High
  A-B contacts: Low

PCA Captures This Pattern:
  
  1. Calculate correlation matrix from contacts
     (each bin vs every other bin)
  
  2. Perform PCA → Eigenvectors
     PC1 = primary pattern (compartments)
     PC2 = secondary pattern
     PC3+ = noise
  
  3. PC1 values:
     Positive = A compartment
     Negative = B compartment

PC1 Profile (chr1):
  +0.5 ████████░░░░░░░░░████████░░░░░░░█████
  +0.0 ────────┼──────────────────┼─────────
  -0.5 ░░░░░░░░████████████░░░░░░███░░░░░░░

  Legend: ████ = A compartment, ░░░░ = B compartment
```

**Compartment Calculation:**
```
Resolution: 100kb (optimal for compartments)

Steps:
1. Load contact matrix at 100kb
2. Calculate observed/expected ratio
3. Compute correlation matrix
4. PCA → Extract PC1
5. Orient PC1 (positive = gene-rich)
6. Threshold at 0 for A/B assignment
7. Generate bigWig track for visualization

Output:
  - A compartment: PC1 > 0
  - B compartment: PC1 < 0
  - Transition zones: PC1 ≈ 0
```

**Expected Results:**
```
Compartment Calling:
  Total 100kb bins: 28,734
  
  A compartment: 15,604 bins (54%)
  B compartment: 13,130 bins (46%)
  
  Compartment size:
    A blocks: Median 1.2 Mb
    B blocks: Median 1.8 Mb
  
  PC1 explained variance: 38%
  (captures primary organization pattern)
```

**Compartment Characteristics:**
```
A Compartment Features:
  - Gene density: High (>10 genes/Mb)
  - GC content: Higher (~44%)
  - Chromatin: Open (DNase, ATAC peaks)
  - Histone marks: H3K4me3, H3K27ac, H3K36me3
  - Transcription: Active
  - Replication: Early
  - Nuclear position: Interior

B Compartment Features:
  - Gene density: Low (<5 genes/Mb)
  - GC content: Lower (~40%)
  - Chromatin: Closed
  - Histone marks: H3K9me3, H3K27me3
  - Transcription: Silent
  - Replication: Late
  - Nuclear position: Periphery (lamina)
```

**Validation:**
```
If other data available:

1. RNA-seq:
   A compartment genes: Higher expression
   B compartment genes: Lower expression
   Expected: Strong correlation

2. ChIP-seq (H3K4me3):
   A compartment: High signal
   B compartment: Low signal

3. ATAC-seq:
   A compartment: High accessibility
   B compartment: Low accessibility

4. Lamina-associated domains (LADs):
   B compartment: Overlap with LADs
   A compartment: Depleted for LADs
```

---

### Step 9: Plot Matrix (HiCExplorer)

**What It Does:**
Generates publication-quality visualizations of contact matrices.

**Plot Configuration:**
```yaml
Visualization settings:
  Resolution: 100kb (good balance)
  Region: chr1:1-10000000 (10Mb region)
  Normalization: KR (balanced)
  Transform: log1p (log(x+1))
  Colormap: RdYlBu_r (red-yellow-blue)
  
Annotations (optional):
  - TAD boundaries (vertical/horizontal lines)
  - Loop anchors (dots)
  - Gene positions
  - Compartment track
```

**Visual Elements:**
```
Hi-C Contact Matrix Plot:

           chr1 Position (Mb)
           0   2   4   6   8   10
      0   [▓▓▓][░░░][░░░][░░░][░░░]
      2   [░░░][▓▓▓][▒▒▒][░░░][░░░]  ← TAD structure
chr1  4   [░░░][▒▒▒][▓▓▓][▒▒▒][░░░]    visible
      6   [░░░][░░░][▒▒▒][▓▓▓][▒▒▒]
      8   [░░░][░░░][░░░][▒▒▒][▓▓▓]
     10   [░░░][░░░][░░░][░░░][▒▒▒]

Legend:
  ▓▓▓ = High contact (diagonal)
  ▒▒▒ = Medium contact (TAD interior)
  ░░░ = Low contact (inter-TAD)
  
Features:
  - Checkerboard: Compartments
  - Triangles on diagonal: TADs
  - Bright off-diagonal pixels: Loops
```

**Interpretation Guide:**
```
Pattern 1: Strong Diagonal
  ▓▓▓▓▓▓▓▓
  ▓▓▓▓▓▓▓▓  Interpretation: Good quality
  ▓▓▓▓▓▓▓▓  Short-range contacts abundant
  
Pattern 2: Checkerboard (Compartments)
  ▓▓░░▓▓░░
  ▓▓░░▓▓░░  Interpretation: Clear compartments
  ░░▓▓░░▓▓  A-A and B-B high, A-B low
  ░░▓▓░░▓▓

Pattern 3: Triangular Domains (TADs)
  ▓▓▓▓
  ▓▓▓▓▓▓▓▓  Interpretation: TAD structure
  ▓▓▓▓▓▓▓▓  High intra-TAD contacts
      ▓▓▓▓
      ▓▓▓▓

Pattern 4: Dots Off-Diagonal (Loops)
  ▓▓  ▓▓
  ▓▓  ██    Interpretation: Chromatin loop
      ▓▓    Enhancer-promoter interaction
```

---

### Step 10: MultiQC Report

**What It Does:**
Aggregates all QC metrics into comprehensive HTML report.

**Report Sections:**

**1. General Statistics:**
```
Sample      Pairs    Valid%  Cis/Trans  Dup%  TADs  Loops  A%
GM12878_r1  200M     65.2%   4.2        16%   3247  8456   54%
GM12878_r2  195M     64.8%   4.1        17%   3189  8234   53%
```

**2. FastQC Results:**
- Quality scores (all samples)
- GC distribution
- Duplication levels

**3. pairtools Statistics:**
- Valid pair percentages
- Pair type distribution
- Cis vs trans ratios
- Duplicate rates

**4. Contact Matrix Metrics:**
- Distance decay profiles
- Resolution statistics
- Normalization convergence

**5. Feature Calling:**
- TAD counts and sizes
- Loop counts and distances
- Compartment percentages

**Interpreting MultiQC:**
```
Look For:
✓ Consistent metrics across replicates
✓ Valid pairs >60%
✓ Cis/Trans ratio 4-5:1
✓ Duplicate rate <30%
✓ TAD counts 2000-5000
✓ Loop counts 5000-15000

Red Flags:
✗ Valid pairs <40% (poor library)
✗ Cis/Trans <2:1 or >10:1 (artifacts)
✗ Duplicate rate >50% (over-amplification)
✗ Outlier samples (investigate/exclude)
✗ Very different TAD/loop counts between replicates
```

---

## Output Interpretation

### Key Output Files

**1. Multi-Resolution Contact Matrix (.mcool)**
```
sample.mcool contains:
  - 8 resolutions (5kb to 1Mb)
  - Raw and normalized counts
  - ICE/KR balancing weights

Use with:
  - HiGlass (web-based viewer)
  - Juicebox (Java application)
  - cooler Python library
  - HiCExplorer tools
```

**2. TAD Files**
```
sample_domains.bed:
  chr1  1000000  1800000  TAD_1  0.89
  chr1  1800000  2600000  TAD_2  0.92
  
  Columns: chr, start, end, name, score

sample_boundaries.bed:
  chr1  1800000  1800001  Boundary_1  -0.67
  
  Score = insulation dip depth (negative)
```

**3. Loop Files (BEDPE)**
```
sample_loops.bedpe:
  chr1  1200000  1201000  chr1  1850000  1851000  8.5  0.001
  
  Format: chr1, start1, end1, chr2, start2, end2, score, q-value
  
  Anchor1 (enhancer) → Anchor2 (promoter)
  Score = enrichment fold-change
  Q-value = FDR-corrected p-value
```

**4. Compartment Files**
```
sample_compartments.bed:
  chr1  0       500000   A   0.45
  chr1  500000  1500000  B  -0.32
  chr1  1500000 2000000  A   0.38
  
sample_compartments.bw:
  BigWig track of PC1 values
  (continuous signal for genome browsers)
```

### Biological Interpretation

**TAD Disruption in Disease:**
```
Example: Limb malformation

Normal:
  TAD 1 [Enh - Gene A]  │  TAD 2 [Enh - Gene B]
                     Boundary (CTCF)

Deletion removes boundary:
  TAD 1+2 [Enh ────┐
                   ├──> Gene B (ectopic activation!)
         Gene A ←──┘

Result:
  - Gene B activated in wrong tissue
  - Developmental abnormality
  - Disease phenotype
```

**Loop Changes in Cancer:**
```
Normal Cell:
  Enhancer ─────(loop)───── Tumor Suppressor
                            ↓ ON

Cancer Cell (CTCF mutation):
  Enhancer     X     Tumor Suppressor
                     ↓ OFF
  
  Lost loop → Gene silencing → Cancer progression
```

**Compartment Switching:**
```
Stem Cell → Differentiated Cell:

Gene X locus:
  Stem: B compartment (silent, H3K27me3)
        ░░░░░░░░
  
  Diff: A compartment (active, H3K4me3)
        ████████
  
Mechanism: B→A switch activates lineage genes
```

---

## Running the Pipeline

### 1. Download Test Data

```bash
# Use the provided downloader
cd ~/BioPipelines
python scripts/download_hic_test.py

# Downloads GM12878 Hi-C data from ENCODE
# Stored in data/raw/hic/
```

### 2. Configure Pipeline

Edit `pipelines/hic/contact_analysis/config.yaml`:

```yaml
# Sample names
samples:
  - GM12878_rep1
  - GM12878_rep2

# Protocol information
protocol:
  enzyme: "MboI"              # Restriction enzyme used
  ligation_site: "GATCGATC"   # Expected junction
  recognition_site: "GATC"
  type: "in_situ"             # or "dilution"

# Resolutions
resolutions:
  - 5000
  - 10000
  - 25000
  - 50000
  - 100000
  - 250000
  - 500000
  - 1000000

# Analysis parameters
trimming:
  quality: 20
  min_length: 50

alignment:
  quality: 30      # MAPQ threshold

filtering:
  max_molecule_size: 1000    # Remove large molecules
  remove_duplicates: true

tad_calling:
  resolution: 100000
  min_depth: 60000
  max_depth: 120000
  step: 10000
  fdr: 0.05

loop_calling:
  resolution: 10000
  min_distance: 15000
  max_distance: 2000000
  fdr: 0.1

compartments:
  resolution: 100000
  pca_components: 3
```

### 3. Submit Job

```bash
# Submit to SLURM
sbatch scripts/submit_hic.sh

# Monitor
squeue -u $USER
tail -f slurm-*.out
```

### 4. Expected Runtime

```
For human Hi-C (200M read pairs):
  FastQC: 15 min
  Trimming: 30 min
  Alignment: 3-4 hours
  Parse pairs: 1.5 hours
  Make cooler: 45 min
  TAD calling: 45 min
  Loop calling: 1.5 hours
  Compartments: 20 min
  Plotting: 15 min
  MultiQC: 5 min
  
Total: ~8-10 hours per sample

For deep Hi-C (1B pairs):
  Total: ~30-40 hours per sample
```

### 5. Resource Requirements

```bash
# Memory: 128GB (matrix operations)
# CPUs: 16 cores
# Storage: ~100GB per sample
#   - Raw FASTQ: 30GB
#   - BAM: 20GB
#   - Pairs: 15GB
#   - Matrices: 10GB
#   - Results: 5GB
```

---

## Troubleshooting

### Issue 1: Low Valid Pairs (<40%)

**Symptoms:**
- pairtools reports <40% valid pairs
- High same-fragment or invalid pair rates

**Causes:**
1. Poor ligation efficiency
2. Incomplete restriction digestion
3. Low sequencing quality

**Solutions:**
```bash
# Check pair distribution
grep "pair_types" logs/pairtools/*.stats

# Look for:
# - High "same_fragment" rate: poor digestion/ligation
# - High "trans" rate (>30%): possible contamination
# - Low "cis_long" rate: poor ligation

# If same_fragment >20%:
# → WET LAB issue: incomplete digestion
# → Check enzyme activity in future preps

# If quality issues:
# → Re-trim more aggressively
# → Increase MAPQ threshold to 40

# Config adjustments:
alignment:
  quality: 40  # More stringent
filtering:
  max_molecule_size: 500  # More stringent
```

### Issue 2: No TADs Detected

**Symptoms:**
- TAD calling produces empty file
- Or very few TADs (<500)

**Causes:**
1. Insufficient valid pairs (<20M)
2. Wrong resolution
3. Too stringent threshold

**Solutions:**
```bash
# Check valid pair count
grep "valid" logs/pairtools/*.stats

# Need ≥50M valid pairs for reliable TAD calling

# If low pairs: Cannot fix in analysis
# → Need more sequencing

# If sufficient pairs, relax parameters:
tad_calling:
  resolution: 100000  # Try 50000 if high depth
  min_depth: 40000    # Lower from 60000
  max_depth: 200000   # Increase from 120000
  fdr: 0.10           # Relax from 0.05
```

### Issue 3: Too Many Loops (>50,000)

**Symptoms:**
- Loop calling reports >50,000 loops
- Many weak loops with low enrichment

**Causes:**
1. Over-calling due to low threshold
2. Noise in high-depth data
3. Artifacts

**Solutions:**
```bash
# Increase stringency:
loop_calling:
  fdr: 0.05          # Stricter (from 0.1)
  min_distance: 20000 # Ignore short-range noise

# Post-filter by enrichment:
awk '$7 > 3.0' sample_loops.bedpe > sample_loops_filtered.bedpe
# (keep only loops with >3-fold enrichment)
```

### Issue 4: Poor Cis/Trans Ratio

**Symptoms:**
- Cis/Trans ratio <2:1 (expect 4-5:1)
- Or Cis/Trans ratio >10:1

**Low Ratio (<2:1):**
```bash
Indicates: Poor ligation or contamination

Possible causes:
- Ligation too short (trans-ligations)
- Contamination with non-Hi-C library
- Wrong species reference

Check:
1. Review protocol (ligation time, temperature)
2. Verify reference genome matches sample
3. Check for contamination with other libraries
```

**High Ratio (>10:1):**
```bash
Indicates: Possible artifacts or quality issues

Possible causes:
- Over-digestion (small fragments)
- Preferential amplification of cis pairs
- Mapping artifacts

Check:
1. Examine fragment size distribution
2. Review restriction digest efficiency
3. Check for PCR bias
```

### Issue 5: ICE Normalization Fails

**Symptoms:**
- Cooler reports "ICE did not converge"
- Missing or incomplete normalized matrix

**Causes:**
1. Bins with zero counts
2. Extreme outliers
3. Insufficient valid pairs

**Solutions:**
```bash
# Pre-filter low-coverage bins:
cooler balance --mad-max 5 \\
               --min-nnz 10 \\
               --min-count 50 \\
               sample.cool

# Or use VC normalization instead:
# In config.yaml:
matrix:
  normalization: ["VC", "VC_SQRT"]  # Skip ICE/KR
  
# VC (Vanilla Coverage) is more robust
# but less effective at removing biases
```

### Issue 6: Memory Issues

**Symptoms:**
- Job killed with OUT_OF_MEMORY
- Usually during matrix operations

**Solutions:**
```bash
# Increase memory allocation:
# Edit scripts/submit_hic.sh:
#SBATCH --mem=256G  # Increase from 128G

# Or process chromosomes separately:
# For each chromosome:
for chr in {1..22} X; do
  cooler cload pairix \\
    --assembly hg38 \\
    chrom.sizes:10000 \\
    sample.pairs.gz \\
    sample_${chr}.cool \\
    --chr1 chr${chr} --chr2 chr${chr}
done

# Merge later:
cooler merge sample.cool sample_chr*.cool
```

### Issue 7: Snakemake Locked

**Symptoms:**
- "Directory cannot be locked"

**Solutions:**
```bash
cd pipelines/hic/contact_analysis
snakemake --unlock

# Or:
rm -rf .snakemake/locks
sbatch scripts/submit_hic.sh
```

---

## Advanced Topics

### Comparing Hi-C Experiments

**Differential TADs:**
```bash
# Use HiCExplorer:
hicDifferentialTAD \\
  --targetMatrix tumor.cool \\
  --controlMatrix normal.cool \\
  --tadDomains normal_tads.bed \\
  --outFileNames diff_tads.bed \\
  --pvalue 0.05
```

**Differential Loops:**
```bash
# Compare loop files:
bedtools intersect -a tumor_loops.bedpe -b normal_loops.bedpe -v \\
  > tumor_specific_loops.bedpe

bedtools intersect -a normal_loops.bedpe -b tumor_loops.bedpe -v \\
  > normal_specific_loops.bedpe
```

### Integration with Other Data

**Hi-C + RNA-seq:**
```
Question: Do genes in A compartments have higher expression?

Analysis:
1. Assign genes to compartments
2. Compare expression A vs B
3. Expected: A > B (strong enrichment)
```

**Hi-C + ChIP-seq:**
```
Question: Are loop anchors enriched for specific TFs?

Analysis:
1. Extract loop anchor regions
2. Intersect with ChIP peaks
3. Calculate enrichment vs genome background
```

**Hi-C + ATAC-seq:**
```
Question: Are TAD boundaries open chromatin?

Analysis:
1. Expand boundaries ±5kb
2. Intersect with ATAC peaks
3. Expected: Strong enrichment
```

### Custom Visualizations

**HiGlass (web viewer):**
```bash
# Start HiGlass server:
higlass-manage start

# Add track:
higlass-manage ingest --uid sample \\
                     --filetype cooler \\
                     sample.mcool

# Access: http://localhost:8888
```

**Python Analysis:**
```python
import cooler
import matplotlib.pyplot as plt

# Load matrix
c = cooler.Cooler('sample.mcool::/resolutions/10000')

# Extract region
matrix = c.matrix(balance=True).fetch('chr1:1000000-3000000')

# Plot
plt.matshow(np.log1p(matrix), cmap='RdYlBu_r')
plt.colorbar()
plt.savefig('chr1_region.png')
```

---

## Citation

**pairtools:**
Open2C et al. Pairtools: From sequencing data to chromosome contact maps. bioRxiv 2023.

**cooler:**
Abdennur N, Mirny LA. Cooler: scalable storage for Hi-C data and other genomically labeled arrays. Bioinformatics. 2020.

**HiCExplorer:**
Wolff J, et al. Galaxy HiCExplorer 3: a web server for reproducible Hi-C, capture Hi-C and single-cell Hi-C data analysis. Nucleic Acids Res. 2020.

**chromosight:**
Matthey-Doret C, et al. Computer vision for pattern detection in chromosome contact maps. Nat Commun. 2020.

---

## Support

For issues:
1. [4DN Data Portal documentation](https://www.4dnucleome.org)
2. [HiC-Pro user guide](https://nservant.github.io/HiC-Pro/)
3. [Open2C forum](https://open2c.github.io)
4. BioPipelines GitHub repository

**Last updated:** November 2025
