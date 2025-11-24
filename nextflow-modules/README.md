# Nextflow Module Library Structure
**Created**: November 24, 2025  
**Purpose**: Modular DSL2 process library leveraging Tier 2 containers  

---

## Directory Structure

```
nextflow-modules/
├── README.md                       # This file
├── alignment/                      # Short-read alignment processes
│   ├── star.nf                    # STAR (RNA-seq splice-aware)
│   ├── bowtie2.nf                 # Bowtie2 (ChIP/ATAC/DNA-seq)
│   ├── bwa.nf                     # BWA (variant calling)
│   └── salmon.nf                  # Salmon (pseudo-alignment)
├── quantification/                 # Expression quantification
│   ├── featurecounts.nf           # featureCounts (gene-level)
│   ├── htseq.nf                   # HTSeq (flexible counting)
│   ├── rsem.nf                    # RSEM (transcript-level)
│   └── stringtie.nf               # StringTie (assembly+quant)
├── peak_calling/                   # ChIP/ATAC peak detection
│   ├── macs2.nf                   # MACS2 (narrow/broad peaks)
│   ├── macs3.nf                   # MACS3 (enhanced calling)
│   └── homer.nf                   # HOMER (motif analysis)
├── variant_calling/                # SNP/indel calling
│   ├── gatk.nf                    # GATK HaplotypeCaller
│   ├── freebayes.nf               # FreeBayes caller
│   ├── vep.nf                     # Variant Effect Predictor
│   └── snpeff.nf                  # SnpEff annotation
├── assembly/                       # De novo assembly
│   ├── spades.nf                  # SPAdes (genome)
│   ├── trinity.nf                 # Trinity (transcriptome)
│   └── megahit.nf                 # MEGAHIT (metagenome)
├── scrna/                          # Single-cell RNA-seq
│   ├── starsolo.nf                # STARsolo (10x)
│   ├── cellranger.nf              # Cell Ranger
│   └── seurat.nf                  # Seurat R analysis
├── longread/                       # Long-read tools
│   ├── minimap2.nf                # Minimap2 alignment
│   ├── nanoplot.nf                # ONT QC
│   └── flye.nf                    # Flye assembly
├── methylation/                    # Bisulfite-seq
│   ├── bismark.nf                 # Bismark aligner
│   └── methyldackel.nf            # MethylDackel extractor
├── metagenomics/                   # Microbiome analysis
│   ├── kraken2.nf                 # Kraken2 classifier
│   ├── bracken.nf                 # Bracken abundance
│   └── metaphlan.nf               # MetaPhlAn4 profiler
├── structural_variants/            # SV detection
│   ├── manta.nf                   # Manta SV caller
│   ├── delly.nf                   # DELLY caller
│   └── cnvkit.nf                  # CNVkit CNV caller
└── qc/                             # Quality control
    ├── fastqc.nf                  # FastQC
    ├── multiqc.nf                 # MultiQC aggregation
    └── trimgalore.nf              # Trim Galore adapter trimming
```

---

## Design Principles

### 1. Container Reference Strategy
Each module references a **Tier 2 domain-specific container**:

```groovy
process STAR_ALIGN {
    container "/scratch/containers/tier2/alignment_short_read.sif"
    // Process definition
}
```

**Benefits**:
- Single container for multiple related tools
- Reduced I/O overhead (one image mount)
- Consistent environment across processes
- Faster startup times

### 2. Parameterization
All modules support dynamic configuration:

```groovy
params {
    // Resource allocation
    star_cpus = 8
    star_memory = '32 GB'
    star_time = '2h'
    
    // Tool parameters
    star_index = '/references/star_indexes/hg38'
    star_extra_args = '--outSAMattributes Standard'
}

process STAR_ALIGN {
    cpus params.star_cpus
    memory params.star_memory
    time params.star_time
    
    script:
    """
    STAR --genomeDir ${params.star_index} ${params.star_extra_args} ...
    """
}
```

### 3. Version Management
Container versions are tracked in module metadata:

```groovy
/*
 * STAR Alignment Module (Tier 2 Container)
 * ==========================================
 * Container: alignment_short_read.sif v1.0.0
 * Tool: STAR 2.7.11a
 * Last Updated: 2025-11-24
 */
```

When containers update:
1. Build new version: `alignment_short_read_v1.1.0.sif`
2. Update module to reference new container
3. Keep old version for reproducibility (30-day TTL)

### 4. Process Tagging
All processes use sample tagging for monitoring:

```groovy
process STAR_ALIGN {
    tag "${sample_id}"
    // Enables: "STAR_ALIGN (sample1)" in logs
}
```

### 5. Output Management
Consistent publishDir patterns:

```groovy
publishDir "${params.outdir}/${tool}/${sample_id}", mode: 'copy'
// Results: outdir/star/sample1/sample1_Aligned.bam
```

---

## Module Usage Examples

### Example 1: RNA-seq Pipeline with Modular Components

```groovy
#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { FASTQC } from './nextflow-modules/qc/fastqc.nf'
include { TRIM_GALORE } from './nextflow-modules/qc/trimgalore.nf'
include { STAR_ALIGN } from './nextflow-modules/alignment/star.nf'
include { FEATURECOUNTS } from './nextflow-modules/quantification/featurecounts.nf'
include { MULTIQC } from './nextflow-modules/qc/multiqc.nf'

workflow RNASEQ {
    // Input channel
    reads_ch = Channel.fromFilePairs("${params.reads}/*_R{1,2}.fastq.gz")
    
    // QC
    FASTQC(reads_ch)
    
    // Trim adapters
    TRIM_GALORE(reads_ch)
    
    // Align with STAR (uses Tier 2 container)
    STAR_ALIGN(
        TRIM_GALORE.out.reads,
        params.star_index
    )
    
    // Quantify
    FEATURECOUNTS(
        STAR_ALIGN.out.bam,
        params.gtf
    )
    
    // Aggregate QC
    MULTIQC(
        FASTQC.out.zip.mix(
            STAR_ALIGN.out.log,
            FEATURECOUNTS.out.summary
        ).collect()
    )
}
```

### Example 2: ChIP-seq Pipeline

```groovy
include { BOWTIE2_ALIGN } from './nextflow-modules/alignment/bowtie2.nf'
include { MACS2_CALLPEAK } from './nextflow-modules/peak_calling/macs2.nf'
include { HOMER_ANNOTATE } from './nextflow-modules/peak_calling/homer.nf'

workflow CHIPSEQ {
    reads_ch = Channel.fromFilePairs("${params.reads}/*_R{1,2}.fastq.gz")
    
    // Align (uses Tier 2 alignment container)
    BOWTIE2_ALIGN(reads_ch, params.bowtie2_index)
    
    // Call peaks (uses Tier 2 peak_calling container)
    MACS2_CALLPEAK(BOWTIE2_ALIGN.out.bam, params.control_bam)
    
    // Annotate (same container)
    HOMER_ANNOTATE(MACS2_CALLPEAK.out.peaks, params.genome)
}
```

### Example 3: Variant Calling Pipeline

```groovy
include { BWA_MEM } from './nextflow-modules/alignment/bwa.nf'
include { GATK_HAPLOTYPECALLER } from './nextflow-modules/variant_calling/gatk.nf'
include { VEP_ANNOTATE } from './nextflow-modules/variant_calling/vep.nf'

workflow VARIANT_CALLING {
    reads_ch = Channel.fromFilePairs("${params.reads}/*_R{1,2}.fastq.gz")
    
    // Align (Tier 2 alignment container)
    BWA_MEM(reads_ch, params.bwa_index)
    
    // Call variants (Tier 2 variant_calling container)
    GATK_HAPLOTYPECALLER(BWA_MEM.out.bam, params.reference_fasta)
    
    // Annotate (same container)
    VEP_ANNOTATE(GATK_HAPLOTYPECALLER.out.vcf, params.vep_cache)
}
```

---

## Module Testing Strategy

### Unit Tests
Each module should have a test workflow:

```groovy
// tests/alignment/star_test.nf
include { STAR_ALIGN } from '../../alignment/star.nf'

workflow TEST_STAR {
    // Use minimal test data
    reads = Channel.of(
        ['test_sample', [
            file('/test/data/sample_R1.fastq.gz'),
            file('/test/data/sample_R2.fastq.gz')
        ]]
    )
    
    index = file('/test/data/star_mini_index')
    
    STAR_ALIGN(reads, index)
}
```

Run test:
```bash
nextflow run tests/alignment/star_test.nf -profile test
```

### Integration Tests
Test module combinations:

```groovy
// tests/integration/rnaseq_mini.nf
workflow TEST_RNASEQ_PIPELINE {
    // Test TRIM -> STAR -> featureCounts chain
}
```

---

## Migration from Current Pipelines

### Step 1: Identify Tool Usage
Analyze current pipeline to find tool calls:
```bash
grep -E "STAR|bowtie2|featureCounts" workflows/rnaseq_simple.nf
```

### Step 2: Replace with Module Imports
Before:
```groovy
process STAR_ALIGN {
    container 'docker://quay.io/biocontainers/star:2.7.11a--h0033a41_0'
    script:
    """
    STAR --genomeDir ${index} ...
    """
}
```

After:
```groovy
include { STAR_ALIGN } from '../nextflow-modules/alignment/star.nf'
```

### Step 3: Adjust Parameters
Update config to use module parameters:
```groovy
params {
    star_index = '/scratch/.../star_index_hg38'
    star_cpus = 8
    star_memory = '32 GB'
}
```

### Step 4: Test Migration
```bash
nextflow run workflows/rnaseq_modular.nf -resume
```

---

## Container Updates Workflow

### When to Update Containers
- **Security patches**: Monthly (automated via CI/CD in Phase 3)
- **Tool versions**: Quarterly (major releases)
- **Bug fixes**: As needed

### Update Process
1. **Build new container version**:
   ```bash
   cd containers/tier2
   # Update alignment_short_read.def (bump STAR 2.7.11a -> 2.7.12a)
   sbatch build_alignment_short_read_v1.1.0.sh
   ```

2. **Test new container**:
   ```bash
   # Run module tests with new container
   nextflow run tests/alignment/star_test.nf \
     --container /scratch/containers/tier2/alignment_short_read_v1.1.0.sif
   ```

3. **Update module reference**:
   ```groovy
   // nextflow-modules/alignment/star.nf
   process STAR_ALIGN {
       container "/scratch/containers/tier2/alignment_short_read_v1.1.0.sif"
   }
   ```

4. **Keep old version for reproducibility** (30-day TTL)

---

## AI Agent Integration (Phase 3)

### Dynamic Module Assembly
AI agents will compose workflows by selecting appropriate modules:

```python
# AI agent selects modules based on user request
workflow_components = {
    'qc': ['fastqc', 'multiqc'],
    'alignment': ['star'],
    'quantification': ['featurecounts']
}

# Generate Nextflow workflow dynamically
nf_code = generate_workflow(workflow_components)
```

### Parameter Optimization
AI agents tune parameters based on data characteristics:

```python
# Detect input data type and size
if is_paired_end and read_length > 100:
    star_params = '--outFilterMultimapNmax 20 --alignSJoverhangMin 8'
elif is_single_end:
    star_params = '--outFilterMultimapNmax 10'
```

### Container Selection
AI agents choose appropriate Tier 2/3 containers:

```python
# Determine required tools
required_tools = ['star', 'featurecounts']

# Check Tier 2 availability
if all(tool in tier2_modules['alignment_short_read'] for tool in required_tools):
    container = 'tier2/alignment_short_read.sif'
else:
    # Build Tier 3 custom container
    container = build_tier3_container(required_tools)
```

---

## Status & Next Steps

### Completed ✅
- Module directory structure defined
- Design principles documented
- Example modules created (STAR, Bowtie2)
- Usage patterns established

### In Progress 🔄
- Creating additional modules (BWA, Salmon, featureCounts)
- Writing module unit tests
- Documenting migration process

### Pending 📋
- Complete all 10 domain module libraries
- Implement automated testing framework
- Create migration guide for each current pipeline
- Build CI/CD for container updates (Phase 3)

---

**Next Action**: Create remaining Tier 2 container definitions and corresponding module files
