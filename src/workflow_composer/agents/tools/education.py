"""
Education Tools
===============

Tools for explaining concepts, comparing samples, and teaching bioinformatics.
"""

import logging
from typing import Any, Dict, List, Optional

from .base import ToolResult

logger = logging.getLogger(__name__)


# =============================================================================
# EXPLAIN_CONCEPT
# =============================================================================

EXPLAIN_CONCEPT_PATTERNS = [
    r"(?:what is|explain|describe|tell me about|how does)\s+(.+?)(?:\s+work|\s+mean|\?|$)",
    r"(?:help me understand|teach me about|define)\s+(.+)",
]


# Knowledge base for bioinformatics concepts
CONCEPT_KNOWLEDGE = {
    "fastqc": {
        "title": "FastQC",
        "category": "Quality Control",
        "description": """FastQC is a quality control tool for high throughput sequence data. 
It provides a modular set of analyses to check raw sequence data for problems before downstream analysis.""",
        "key_points": [
            "Checks per-base sequence quality",
            "Identifies adapter contamination",
            "Detects overrepresented sequences",
            "Measures GC content distribution",
            "Reports sequence duplication levels"
        ],
        "when_to_use": "Always run FastQC on raw FASTQ files before starting any analysis pipeline.",
        "related": ["MultiQC", "Trimmomatic", "fastp"]
    },
    
    "bwa": {
        "title": "BWA (Burrows-Wheeler Aligner)",
        "category": "Alignment",
        "description": """BWA is a software package for mapping low-divergent sequences against a large reference genome.
It's widely used for DNA sequencing alignment.""",
        "key_points": [
            "BWA-MEM is best for reads >70bp",
            "Creates SAM/BAM alignment files",
            "Requires indexed reference genome",
            "Supports paired-end reads",
            "Handles chimeric alignments (split reads)"
        ],
        "when_to_use": "Use BWA for DNA-seq alignment, especially for WGS, WES, and panel sequencing.",
        "related": ["Bowtie2", "HISAT2", "minimap2"]
    },
    
    "star": {
        "title": "STAR (Spliced Transcripts Alignment to a Reference)",
        "category": "Alignment",
        "description": """STAR is an ultrafast RNA-seq aligner that handles spliced alignments.
It's the recommended aligner for RNA sequencing data.""",
        "key_points": [
            "Handles splice junctions automatically",
            "Very fast but memory-intensive",
            "Can output gene counts directly",
            "Supports 2-pass alignment for novel junctions",
            "Creates chimeric output for fusion detection"
        ],
        "when_to_use": "Use STAR for all RNA-seq alignment tasks.",
        "related": ["HISAT2", "Salmon", "kallisto"]
    },
    
    "gatk": {
        "title": "GATK (Genome Analysis Toolkit)",
        "category": "Variant Calling",
        "description": """GATK is a collection of tools for variant discovery and genotyping.
Developed by the Broad Institute, it's the gold standard for germline variant calling.""",
        "key_points": [
            "HaplotypeCaller for germline variants",
            "Mutect2 for somatic variants",
            "Includes BQSR for base quality recalibration",
            "VQSR for variant quality score recalibration",
            "Best Practices workflows available"
        ],
        "when_to_use": "Use GATK for high-confidence variant calling from DNA sequencing.",
        "related": ["bcftools", "DeepVariant", "Strelka2"]
    },
    
    "deseq2": {
        "title": "DESeq2",
        "category": "Differential Expression",
        "description": """DESeq2 is an R package for differential gene expression analysis.
It uses negative binomial distribution to model count data.""",
        "key_points": [
            "Handles low-count genes appropriately",
            "Automatic normalization (size factors)",
            "Shrinkage estimation for fold changes",
            "Multiple testing correction included",
            "Works with raw counts (not FPKM/TPM)"
        ],
        "when_to_use": "Use DESeq2 for RNA-seq differential expression between conditions.",
        "related": ["edgeR", "limma-voom", "sleuth"]
    },
    
    "macs2": {
        "title": "MACS2 (Model-based Analysis of ChIP-Seq)",
        "category": "Peak Calling",
        "description": """MACS2 identifies transcription factor binding sites or histone modification 
regions from ChIP-seq and ATAC-seq data.""",
        "key_points": [
            "Models shift size from data",
            "Supports narrow and broad peak calling",
            "Handles paired-end data",
            "Outputs BED/narrowPeak format",
            "Calculates FDR for peaks"
        ],
        "when_to_use": "Use MACS2 for ChIP-seq and ATAC-seq peak calling.",
        "related": ["HOMER", "SEACR", "Genrich"]
    },
    
    "salmon": {
        "title": "Salmon",
        "category": "Quantification",
        "description": """Salmon is a fast transcript-level quantification tool.
It uses quasi-mapping for speed and bias correction for accuracy.""",
        "key_points": [
            "Alignment-free quantification",
            "Very fast (minutes for typical samples)",
            "Built-in GC and sequence bias correction",
            "Outputs TPM and counts",
            "Can be used with tximport for gene-level analysis"
        ],
        "when_to_use": "Use Salmon when you need fast transcript quantification without full alignment.",
        "related": ["kallisto", "RSEM", "featureCounts"]
    },
    
    "vcf": {
        "title": "VCF (Variant Call Format)",
        "category": "File Formats",
        "description": """VCF is a text file format for storing genetic variation data.
It's the standard format for variant calls.""",
        "key_points": [
            "Header section with metadata",
            "One variant per line",
            "CHROM, POS, ID, REF, ALT columns required",
            "QUAL and FILTER for quality info",
            "INFO and FORMAT for annotations",
            "Can be compressed with bgzip"
        ],
        "when_to_use": "VCF files are output by all variant callers and input to annotation tools.",
        "related": ["BCF", "gVCF", "MAF"]
    },
    
    "bam": {
        "title": "BAM (Binary Alignment Map)",
        "category": "File Formats",
        "description": """BAM is the binary, compressed version of SAM format for storing aligned sequences.
It's indexed for fast random access.""",
        "key_points": [
            "Binary compressed format",
            "Requires .bai index for random access",
            "Contains alignment information",
            "Can be viewed with samtools",
            "CRAM is a more compressed alternative"
        ],
        "when_to_use": "BAM files are the standard output from alignment tools.",
        "related": ["SAM", "CRAM", "FASTQ"]
    },
    
    "fpkm": {
        "title": "FPKM/TPM (Expression Units)",
        "category": "Quantification",
        "description": """FPKM and TPM are normalized expression units that account for gene length and sequencing depth.""",
        "key_points": [
            "FPKM: Fragments Per Kilobase per Million mapped",
            "TPM: Transcripts Per Million",
            "TPM is preferred (sums to 1M per sample)",
            "Both account for gene length",
            "Don't use for differential expression (use raw counts)"
        ],
        "when_to_use": "Use TPM/FPKM for comparing expression levels across genes within a sample.",
        "related": ["counts", "CPM", "RPKM"]
    },
    
    "chip-seq": {
        "title": "ChIP-seq (Chromatin Immunoprecipitation Sequencing)",
        "category": "Assay Types",
        "description": """ChIP-seq identifies DNA binding sites of proteins (transcription factors, histones).
Uses antibodies to pull down protein-DNA complexes.""",
        "key_points": [
            "Requires antibody specific to target protein",
            "Control sample is important (input or IgG)",
            "Narrow peaks for TFs, broad peaks for histones",
            "Typically 10-50M reads per sample",
            "Peak calling identifies enriched regions"
        ],
        "when_to_use": "Use ChIP-seq to map protein-DNA interactions genome-wide.",
        "related": ["ATAC-seq", "CUT&RUN", "CUT&TAG"]
    },
    
    "atac-seq": {
        "title": "ATAC-seq (Assay for Transposase-Accessible Chromatin)",
        "category": "Assay Types",
        "description": """ATAC-seq identifies open chromatin regions using Tn5 transposase.
Open chromatin indicates regulatory regions and active transcription.""",
        "key_points": [
            "Requires fewer cells than ChIP-seq",
            "No antibody needed",
            "Fragment size distribution shows nucleosome pattern",
            "Typically 50-100M reads per sample",
            "Often combined with scATAC for single-cell"
        ],
        "when_to_use": "Use ATAC-seq to profile chromatin accessibility genome-wide.",
        "related": ["ChIP-seq", "DNase-seq", "MNase-seq"]
    },
}


def explain_concept_impl(concept: str) -> ToolResult:
    """
    Explain a bioinformatics concept.
    
    Args:
        concept: The concept to explain
        
    Returns:
        ToolResult with explanation
    """
    if not concept:
        return ToolResult(
            success=False,
            tool_name="explain_concept",
            error="No concept specified",
            message="""❓ **What would you like to learn about?**

I can explain:
- **Tools**: FastQC, BWA, STAR, GATK, DESeq2, MACS2, Salmon
- **File Formats**: VCF, BAM, FASTQ, BED
- **Assays**: ChIP-seq, ATAC-seq, RNA-seq, WGS
- **Metrics**: FPKM, TPM, FDR, p-value
- **Concepts**: Alignment, Variant calling, Peak calling

Just ask! Example: "what is STAR?" or "explain ChIP-seq"
"""
        )
    
    # Normalize concept
    concept_lower = concept.lower().strip()
    
    # Direct match
    if concept_lower in CONCEPT_KNOWLEDGE:
        info = CONCEPT_KNOWLEDGE[concept_lower]
    else:
        # Partial match
        matches = [k for k in CONCEPT_KNOWLEDGE if concept_lower in k or k in concept_lower]
        if matches:
            info = CONCEPT_KNOWLEDGE[matches[0]]
        else:
            # No match - provide general response
            return ToolResult(
                success=True,
                tool_name="explain_concept",
                data={"concept": concept, "matched": False},
                message=f"""🤔 **{concept}**

I don't have a detailed explanation for "{concept}" in my knowledge base.

**What I can tell you:**
This appears to be a bioinformatics term. For more information:
1. Search the tool documentation
2. Check Biostars or SeqAnswers forums
3. Look at relevant papers on PubMed

**Related topics I can explain:**
- {', '.join(list(CONCEPT_KNOWLEDGE.keys())[:10])}

Ask about any of these for detailed information!
"""
            )
    
    # Format explanation
    key_points = "\n".join(f"- {p}" for p in info['key_points'])
    related = ", ".join(info['related'])
    
    message = f"""📚 **{info['title']}**
*Category: {info['category']}*

{info['description']}

**Key Points:**
{key_points}

**When to Use:**
{info['when_to_use']}

**Related Tools/Concepts:**
{related}

---
Need more detail? Ask about any of the related topics!
"""
    
    return ToolResult(
        success=True,
        tool_name="explain_concept",
        data={"concept": concept, "info": info},
        message=message
    )


# =============================================================================
# COMPARE_SAMPLES
# =============================================================================

COMPARE_SAMPLES_PATTERNS = [
    r"compare\s+(?:samples?|conditions?|groups?)\s*(.+)?",
    r"(?:what(?:'s| is))\s+(?:the\s+)?difference\s+between\s+(.+)",
]


def compare_samples_impl(
    sample1: str = None,
    sample2: str = None,
    comparison_type: str = "general",
) -> ToolResult:
    """
    Compare samples or provide guidance on sample comparison.
    
    Args:
        sample1: First sample/condition
        sample2: Second sample/condition
        comparison_type: Type of comparison (general, expression, variants)
        
    Returns:
        ToolResult with comparison guidance
    """
    if not sample1 or not sample2:
        message = """📊 **Sample Comparison Guide**

**To compare samples, I need:**
1. Two sample names or conditions
2. The type of data (expression, variants, peaks)

**Example queries:**
- "compare treated vs control samples"
- "what's the difference between tumor and normal"
- "compare expression in samples A and B"

**Types of comparisons I can help with:**

### Expression Analysis
- Differential gene expression
- Pathway enrichment
- Gene ontology analysis

### Variant Analysis
- Shared vs unique variants
- Allele frequency differences
- Mutation signatures

### ChIP/ATAC Analysis
- Differential peaks
- Motif enrichment differences
- Chromatin state changes

Tell me what you'd like to compare!
"""
    else:
        message = f"""📊 **Comparing: {sample1} vs {sample2}**

**Recommended Analysis Workflow:**

### 1. Quality Check First
- Ensure both samples passed QC
- Check for batch effects
- Verify comparable sequencing depth

### 2. Expression Comparison (if RNA-seq)
```r
# Using DESeq2
dds <- DESeqDataSetFromMatrix(counts, colData, ~ condition)
dds <- DESeq(dds)
res <- results(dds, contrast=c("condition", "{sample1}", "{sample2}"))
```

### 3. Variant Comparison (if DNA-seq)
```bash
# Using bcftools
bcftools isec -p comparison_results {sample1}.vcf {sample2}.vcf
```

### 4. Key Metrics to Report
- Number of differentially expressed genes (padj < 0.05)
- Log2 fold change distribution
- Shared vs unique features

### 5. Visualization
- MA plots
- Volcano plots
- Heatmaps of top differences

**Would you like me to:**
1. Generate a comparison workflow?
2. Explain any analysis step?
3. Help interpret existing results?
"""
    
    return ToolResult(
        success=True,
        tool_name="compare_samples",
        data={"sample1": sample1, "sample2": sample2},
        message=message
    )


# =============================================================================
# GET_HELP
# =============================================================================

GET_HELP_PATTERNS = [
    r"^(?:help|\\?|what can you do|commands?|capabilities)$",
    r"(?:show|list)\s+(?:all\s+)?(?:commands?|capabilities|features)",
]


def get_help_impl() -> ToolResult:
    """
    Show help information about available commands.
    
    Returns:
        ToolResult with help information
    """
    message = """🤖 **BioPipelines AI Assistant**

I can help you with bioinformatics workflows! Here's what I can do:

---

### 📁 Data Discovery
- "scan my data" - Find data files in workspace
- "search for ENCODE K562 ChIP-seq" - Search public databases
- "search TCGA for lung cancer RNA-seq" - Search cancer genomics data
- "describe my files" - Analyze file contents

### 📥 Data Management
- "download ENCODE dataset ENCSR..." - Download from ENCODE
- "cleanup old data" - Remove unnecessary files
- "validate my dataset" - Check data integrity

### 🔧 Workflow Generation
- "create RNA-seq workflow" - Generate analysis pipeline
- "make ChIP-seq pipeline" - Create peak calling workflow
- "list available workflows" - Show workflow templates
- "check reference data" - Verify genome references

### 🚀 Execution
- "run the workflow" - Submit to SLURM
- "check job status" - Monitor running jobs
- "show logs" - View execution logs
- "cancel job 12345" - Stop a job

### 🔍 Analysis & Diagnostics
- "diagnose this error" - Troubleshoot problems
- "analyze my results" - Interpret output files
- "compare samples A vs B" - Comparison guidance

### 📚 Education
- "what is FastQC?" - Explain tools and concepts
- "how does BWA work?" - Learn about algorithms
- "explain ChIP-seq" - Understand assay types

---

**Tips:**
- Be specific about what you need
- I understand natural language
- Ask follow-up questions!

**Example session:**
```
You: scan my data
You: create RNA-seq workflow for the fastq files
You: run it on SLURM
You: check status
```
"""
    
    return ToolResult(
        success=True,
        tool_name="get_help",
        data={},
        message=message
    )
