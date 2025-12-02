"""
Error Guidance Generator
========================

Converts passive error diagnosis into actionable recovery guidance.

Inspired by DeepCode's _generate_error_guidance() pattern which provides:
- Structured recovery steps
- Anti-patterns (what NOT to do)
- Example fixes
- Related documentation

This module extends BioPipelines' CodingAgent with rich error guidance
for bioinformatics-specific errors.

References:
    - DeepCode: workflows/code_implementation_workflow.py
"""

import re
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# Import ErrorType from coding_agent to ensure compatibility
# We'll use a string-based approach for loose coupling
class ErrorCategory(Enum):
    """Error categories for guidance generation."""
    MEMORY = "memory"
    DISK = "disk"
    PERMISSION = "permission"
    NEXTFLOW = "nextflow"
    SNAKEMAKE = "snakemake"
    SLURM = "slurm"
    TOOL = "tool"
    NETWORK = "network"
    SYNTAX = "syntax"
    DATA = "data"           # Data format/integrity issues
    REFERENCE = "reference"  # Reference genome issues
    INDEX = "index"         # Index file issues
    CONTAINER = "container"  # Docker/Singularity issues
    UNKNOWN = "unknown"


@dataclass
class ErrorGuidance:
    """
    Actionable error recovery guidance.
    
    Attributes:
        error_summary: One-line summary of the error
        recovery_steps: Ordered list of recovery actions
        anti_patterns: Things to avoid (common mistakes)
        example_fix: Code example showing the fix
        related_docs: Links to relevant documentation
        estimated_fix_time: Rough estimate (e.g., "5 minutes", "30 minutes")
        requires_human: Whether human intervention is likely needed
    """
    error_summary: str
    recovery_steps: List[str]
    anti_patterns: List[str]
    example_fix: Optional[str] = None
    related_docs: List[str] = field(default_factory=list)
    estimated_fix_time: str = "unknown"
    requires_human: bool = False
    
    def to_markdown(self) -> str:
        """Format as markdown for display."""
        parts = [
            f"## Error: {self.error_summary}",
            "",
            "### Recovery Steps",
        ]
        
        for i, step in enumerate(self.recovery_steps, 1):
            parts.append(f"{i}. {step}")
        
        parts.append("")
        parts.append("### What NOT to Do")
        for ap in self.anti_patterns:
            parts.append(f"- ❌ {ap}")
        
        if self.example_fix:
            parts.append("")
            parts.append("### Example Fix")
            parts.append("```")
            parts.append(self.example_fix.strip())
            parts.append("```")
        
        if self.related_docs:
            parts.append("")
            parts.append("### Related Documentation")
            for doc in self.related_docs:
                parts.append(f"- {doc}")
        
        if self.estimated_fix_time != "unknown":
            parts.append("")
            parts.append(f"⏱️ Estimated fix time: {self.estimated_fix_time}")
        
        if self.requires_human:
            parts.append("")
            parts.append("⚠️ This issue may require human intervention")
        
        return "\n".join(parts)
    
    def to_agent_context(self) -> str:
        """Format for injection into agent context."""
        parts = [
            f"## Error Encountered",
            f"**Summary:** {self.error_summary}",
            "",
            "## Recovery Strategy",
        ]
        parts.extend(self.recovery_steps)
        
        parts.append("")
        parts.append("## What NOT to Do")
        for ap in self.anti_patterns:
            parts.append(f"- {ap}")
        
        if self.example_fix:
            parts.append("")
            parts.append("## Example Fix")
            parts.append("```")
            parts.append(self.example_fix.strip())
            parts.append("```")
        
        return "\n".join(parts)


# =============================================================================
# Extended Bioinformatics Error Patterns
# =============================================================================

BIOINFORMATICS_ERROR_PATTERNS = {
    # STAR Aligner
    "star_quality_mismatch": (
        r"EXITING because of FATAL ERROR in reads input: quality string length is not equal to sequence length",
        ErrorCategory.DATA,
        "FASTQ file corrupted or has mismatched quality scores",
    ),
    "star_version_incompatible": (
        r"STAR.*Genome version.*is INCOMPATIBLE",
        ErrorCategory.INDEX,
        "STAR index built with different STAR version",
    ),
    "star_genome_load": (
        r"EXITING because of FATAL ERROR: could not open genome file",
        ErrorCategory.REFERENCE,
        "STAR genome index files missing or inaccessible",
    ),
    "star_shared_memory": (
        r"EXITING because of FATAL ERROR: Genome could not be loaded into shared memory",
        ErrorCategory.MEMORY,
        "STAR cannot load genome into shared memory",
    ),
    
    # BWA
    "bwa_file_open": (
        r"bwa.*fail to open file|bwa.*No such file",
        ErrorCategory.DATA,
        "BWA cannot open input file",
    ),
    "bwa_index_missing": (
        r"bwa.*Can't find index",
        ErrorCategory.INDEX,
        "BWA index files missing",
    ),
    
    # Samtools
    "samtools_truncated": (
        r"samtools.*truncated file|EOF marker is absent",
        ErrorCategory.DATA,
        "BAM/SAM file is truncated or incomplete",
    ),
    "samtools_header": (
        r"samtools.*invalid BAM binary header|not a BAM file",
        ErrorCategory.DATA,
        "BAM file has corrupted or invalid header",
    ),
    "samtools_reference": (
        r"samtools.*reference.*not found|samtools.*Failed to open reference",
        ErrorCategory.REFERENCE,
        "Reference file missing for samtools operation",
    ),
    
    # GATK
    "gatk_cigar": (
        r"MalformedRead.*CIGAR.*extends past end|Bad CIGAR",
        ErrorCategory.DATA,
        "Read has invalid CIGAR string",
    ),
    "gatk_contig": (
        r"htsjdk.*Contig.*not in sequence dictionary|Contig.*not found",
        ErrorCategory.REFERENCE,
        "Reference contigs don't match between files",
    ),
    "gatk_java_heap": (
        r"java\.lang\.OutOfMemoryError: Java heap space",
        ErrorCategory.MEMORY,
        "Java heap space exhausted",
    ),
    "gatk_gc_overhead": (
        r"java\.lang\.OutOfMemoryError: GC overhead limit exceeded",
        ErrorCategory.MEMORY,
        "Java spending too much time in garbage collection",
    ),
    
    # Salmon/Kallisto
    "salmon_index": (
        r"salmon.*Index appears to be corrupt|salmon.*Index version",
        ErrorCategory.INDEX,
        "Salmon index is corrupted or incompatible version",
    ),
    "kallisto_index": (
        r"kallisto.*Error: index file .* does not exist",
        ErrorCategory.INDEX,
        "Kallisto index file missing",
    ),
    
    # File Format Issues
    "gzip_truncated": (
        r"gzip: stdin: unexpected end of file|gzip:.*not in gzip format",
        ErrorCategory.DATA,
        "Compressed file is truncated or not gzip format",
    ),
    "fastq_format": (
        r"FASTQ file .* appears to be corrupt|invalid FASTQ",
        ErrorCategory.DATA,
        "FASTQ file has format errors",
    ),
    
    # Container Issues
    "singularity_pull": (
        r"FATAL:.*Unable to pull|singularity.*pull failed",
        ErrorCategory.CONTAINER,
        "Failed to pull Singularity container",
    ),
    "docker_not_found": (
        r"docker:.*not found|Cannot connect to the Docker daemon",
        ErrorCategory.CONTAINER,
        "Docker not available or not running",
    ),
    
    # Nextflow Specific
    "nextflow_channel_empty": (
        r"Channel.*is empty|No files matching pattern",
        ErrorCategory.NEXTFLOW,
        "Nextflow channel has no input files",
    ),
    "nextflow_param_missing": (
        r"Missing required parameter|No such variable: params\.",
        ErrorCategory.NEXTFLOW,
        "Required Nextflow parameter not provided",
    ),
    
    # SLURM Specific
    "slurm_memory": (
        r"slurmstepd:.*oom-kill|Exceeded job memory limit|CANCELLED.*OOM",
        ErrorCategory.MEMORY,
        "SLURM job killed due to memory limit",
    ),
    "slurm_time": (
        r"DUE TO TIME LIMIT|TIMEOUT|CANCELLED.*TIME",
        ErrorCategory.SLURM,
        "SLURM job exceeded time limit",
    ),
    "slurm_node": (
        r"Nodes required for job are DOWN|ReqNodeNotAvail",
        ErrorCategory.SLURM,
        "SLURM nodes not available",
    ),
}


def identify_error_category(error_log: str) -> tuple:
    """
    Identify error category from log content.
    
    Returns:
        Tuple of (category: ErrorCategory, pattern_name: str, description: str)
        or (ErrorCategory.UNKNOWN, None, None) if not matched
    """
    for pattern_name, (regex, category, desc) in BIOINFORMATICS_ERROR_PATTERNS.items():
        if re.search(regex, error_log, re.IGNORECASE | re.MULTILINE):
            return category, pattern_name, desc
    
    return ErrorCategory.UNKNOWN, None, None


# =============================================================================
# Guidance Templates
# =============================================================================

GUIDANCE_TEMPLATES: Dict[ErrorCategory, Dict[str, Any]] = {
    ErrorCategory.MEMORY: {
        "recovery_steps": [
            "Check current memory allocation: `squeue -u $USER -o '%i %j %m %M'`",
            "Increase memory in workflow config (see example below)",
            "If still failing, reduce parallelism: `maxForks = 2` or `--max-cpus 8`",
            "For large genomes, consider streaming/chunked processing",
            "Monitor memory usage: `sstat -j $JOBID --format=MaxRSS`",
        ],
        "anti_patterns": [
            "Don't increase memory beyond node limits (check `sinfo -Nel`)",
            "Don't just retry without changing parameters - same failure will occur",
            "Don't run multiple memory-heavy processes in parallel (STAR, BWA-MEM2)",
            "Don't ignore OOM warnings before the final failure",
        ],
        "example_fix": """
// nextflow.config - Memory configuration with retry
process {
    withName: 'STAR_ALIGN' {
        memory = { 32.GB * task.attempt }
        maxRetries = 2
        errorStrategy = { task.exitStatus in [137,140] ? 'retry' : 'finish' }
    }
}
""",
        "related_docs": [
            "https://www.nextflow.io/docs/latest/process.html#memory",
            "https://nf-co.re/docs/usage/configuration#max-resources",
        ],
        "estimated_fix_time": "10-15 minutes",
    },
    
    ErrorCategory.DISK: {
        "recovery_steps": [
            "Check available space: `df -h /path/to/workDir`",
            "Check quota: `quota -s` or `lfs quota -u $USER /path`",
            "Clean Nextflow work directory: `nextflow clean -f`",
            "Set work directory to larger filesystem: `-work-dir /scratch/$USER/work`",
            "Enable automatic cleanup in config: `cleanup = true`",
        ],
        "anti_patterns": [
            "Don't delete work directories during an active run",
            "Don't ignore disk warnings - they lead to data corruption",
            "Don't store large intermediates in home directory",
            "Don't keep all intermediate files for large cohorts",
        ],
        "example_fix": """
// nextflow.config - Disk management
workDir = '/scratch/$USER/nextflow_work'

process {
    // Limit intermediate file retention
    publishDir {
        path: "$params.outdir"
        mode: 'copy'  // Copy, don't symlink
    }
}

// Enable cleanup of successful work directories
cleanup = true
""",
        "related_docs": [
            "https://www.nextflow.io/docs/latest/config.html#config-cleanup",
        ],
        "estimated_fix_time": "5-10 minutes",
    },
    
    ErrorCategory.INDEX: {
        "recovery_steps": [
            "Verify index files exist: `ls -la /path/to/index*`",
            "Check index version matches tool version",
            "Rebuild index with current tool version if needed",
            "Ensure index was built with same reference as being used",
            "Check file permissions on index directory",
        ],
        "anti_patterns": [
            "Don't use indices built with different tool versions",
            "Don't assume downloaded indices match your reference",
            "Don't skip index integrity checks after download",
            "Don't mix references from different sources without validation",
        ],
        "example_fix": """
# Rebuild STAR index (ensure same version as runtime)
STAR --runMode genomeGenerate \\
    --genomeDir /path/to/star_index \\
    --genomeFastaFiles /path/to/genome.fa \\
    --sjdbGTFfile /path/to/genes.gtf \\
    --runThreadN 16

# Verify STAR version matches
STAR --version
""",
        "related_docs": [
            "https://github.com/alexdobin/STAR/blob/master/doc/STARmanual.pdf",
        ],
        "estimated_fix_time": "30-60 minutes (index rebuild)",
    },
    
    ErrorCategory.DATA: {
        "recovery_steps": [
            "Validate file integrity: `gzip -t file.fastq.gz` or `samtools quickcheck file.bam`",
            "Check file sizes match expected: `ls -lh *.fastq.gz`",
            "Re-download corrupted files from source",
            "Run FastQC to identify quality issues",
            "Check for incomplete transfers: `md5sum -c checksums.txt`",
        ],
        "anti_patterns": [
            "Don't assume downloaded files are complete without verification",
            "Don't skip validation on files from external sources",
            "Don't process files that fail integrity checks",
            "Don't mix samples from different sequencing runs without validation",
        ],
        "example_fix": """
# Validate FASTQ files
for f in *.fastq.gz; do
    gzip -t "$f" && echo "OK: $f" || echo "CORRUPT: $f"
done

# Validate BAM files
samtools quickcheck -v *.bam

# Run FastQC for deeper validation
fastqc -t 8 *.fastq.gz -o fastqc_results/
""",
        "related_docs": [
            "https://www.bioinformatics.babraham.ac.uk/projects/fastqc/",
        ],
        "estimated_fix_time": "15-30 minutes",
    },
    
    ErrorCategory.REFERENCE: {
        "recovery_steps": [
            "Verify reference file exists and is readable",
            "Check chromosome naming convention matches (chr1 vs 1)",
            "Ensure reference matches the index that was used",
            "Validate reference with `samtools faidx` to check integrity",
            "Check reference genome version (GRCh38 vs hg38 vs GRCh37)",
        ],
        "anti_patterns": [
            "Don't mix chromosome naming conventions within a pipeline",
            "Don't assume reference file versions match between steps",
            "Don't use references from unknown sources in production",
            "Don't skip .fai index generation for FASTA files",
        ],
        "example_fix": """
# Index reference FASTA
samtools faidx reference.fa

# Create sequence dictionary (for GATK)
gatk CreateSequenceDictionary -R reference.fa

# Verify chromosome names
head -n 1 reference.fa  # Check naming: >chr1 or >1
samtools view -H aligned.bam | grep '@SQ'  # Compare to BAM
""",
        "related_docs": [
            "https://gatk.broadinstitute.org/hc/en-us/articles/360035531652",
        ],
        "estimated_fix_time": "5-15 minutes",
    },
    
    ErrorCategory.CONTAINER: {
        "recovery_steps": [
            "Check container runtime is available: `docker info` or `singularity --version`",
            "Verify image exists: `singularity cache list` or `docker images`",
            "Pull image manually to diagnose issues",
            "Check network connectivity to registry",
            "Verify sufficient disk space for container cache",
        ],
        "anti_patterns": [
            "Don't use 'latest' tag in production - pin specific versions",
            "Don't run containers as root unless required",
            "Don't ignore container security warnings",
            "Don't skip container cache cleanup on shared systems",
        ],
        "example_fix": """
// nextflow.config - Container configuration
singularity {
    enabled = true
    autoMounts = true
    cacheDir = '/scratch/$USER/singularity_cache'
}

// Pull containers manually if needed
// singularity pull docker://quay.io/biocontainers/star:2.7.10b--h9ee0642_0
""",
        "related_docs": [
            "https://www.nextflow.io/docs/latest/singularity.html",
            "https://nf-co.re/docs/usage/configuration#singularity",
        ],
        "estimated_fix_time": "10-20 minutes",
    },
    
    ErrorCategory.NEXTFLOW: {
        "recovery_steps": [
            "Check input file patterns with: `ls -la $params.input`",
            "Validate channel creation with `.view()` operator",
            "Ensure all required params are provided",
            "Check Nextflow version compatibility: `nextflow -version`",
            "Review .nextflow.log for detailed error information",
        ],
        "anti_patterns": [
            "Don't mix DSL1 and DSL2 syntax in the same script",
            "Don't hardcode paths - use params for all inputs",
            "Don't use `.collect()` on large datasets without chunking",
            "Don't ignore deprecation warnings when upgrading Nextflow",
        ],
        "example_fix": """
// Proper input channel handling
Channel
    .fromFilePairs("$params.input/*_{1,2}.fastq.gz", checkIfExists: true)
    .ifEmpty { error "No files matching pattern" }
    .set { reads_ch }

// Debug channel contents
reads_ch.view { "Found: $it" }
""",
        "related_docs": [
            "https://www.nextflow.io/docs/latest/channel.html",
            "https://nf-co.re/docs/contributing/troubleshooting",
        ],
        "estimated_fix_time": "15-30 minutes",
    },
    
    ErrorCategory.SLURM: {
        "recovery_steps": [
            "Check job status: `sacct -j $JOBID --format=JobID,State,ExitCode,MaxRSS,Elapsed`",
            "Review SLURM output: `cat slurm-$JOBID.out`",
            "Check cluster status: `sinfo -Nel`",
            "Verify partition limits: `scontrol show partition`",
            "Adjust resource requests based on actual usage",
        ],
        "anti_patterns": [
            "Don't request more resources than any single node has",
            "Don't set walltime too short for large datasets",
            "Don't submit thousands of jobs simultaneously without throttling",
            "Don't ignore SLURM's email notifications about job issues",
        ],
        "example_fix": """
// nextflow.config - SLURM configuration
process {
    executor = 'slurm'
    queue = 'normal'
    
    clusterOptions = '--account=my_project'
    
    // Dynamic resources based on task
    cpus = { task.attempt < 2 ? 8 : 16 }
    memory = { 32.GB * task.attempt }
    time = { 4.h * task.attempt }
    
    // Retry on SLURM-specific exit codes
    errorStrategy = { task.exitStatus in [140,143,137] ? 'retry' : 'finish' }
    maxRetries = 2
}
""",
        "related_docs": [
            "https://www.nextflow.io/docs/latest/executor.html#slurm",
            "https://slurm.schedmd.com/sacct.html",
        ],
        "estimated_fix_time": "10-20 minutes",
    },
    
    ErrorCategory.UNKNOWN: {
        "recovery_steps": [
            "Review complete error log for context",
            "Search error message in tool documentation",
            "Check BioPipelines troubleshooting guide",
            "Search biostars.org or bioinformatics.stackexchange.com",
            "Submit issue with complete logs if unresolved",
        ],
        "anti_patterns": [
            "Don't ignore preceding warnings before the error",
            "Don't change multiple parameters at once when debugging",
            "Don't delete log files before diagnosis is complete",
            "Don't assume the error message describes the root cause",
        ],
        "example_fix": None,
        "related_docs": [],
        "estimated_fix_time": "varies",
    },
}


# =============================================================================
# Main Guidance Generator
# =============================================================================

def generate_error_guidance(
    error_category: ErrorCategory,
    error_description: str = "",
    context: Optional[Dict[str, Any]] = None,
) -> ErrorGuidance:
    """
    Generate actionable error recovery guidance.
    
    Args:
        error_category: The category of error
        error_description: Description from pattern matching
        context: Additional context (workflow type, tool, etc.)
        
    Returns:
        ErrorGuidance with recovery steps and anti-patterns
    """
    template = GUIDANCE_TEMPLATES.get(
        error_category,
        GUIDANCE_TEMPLATES[ErrorCategory.UNKNOWN]
    )
    
    # Build summary
    if error_description:
        summary = f"{error_category.value.upper()}: {error_description}"
    else:
        summary = f"{error_category.value.upper()} error detected"
    
    # Customize based on context
    recovery_steps = template["recovery_steps"].copy()
    anti_patterns = template["anti_patterns"].copy()
    example_fix = template.get("example_fix")
    
    # Add context-specific guidance
    if context:
        workflow_type = context.get("workflow_type", "nextflow")
        if workflow_type == "snakemake" and error_category != ErrorCategory.NEXTFLOW:
            # Adjust Nextflow-specific advice for Snakemake
            recovery_steps = [
                step.replace("nextflow", "snakemake")
                    .replace("Nextflow", "Snakemake")
                    .replace(".nf", ".smk")
                for step in recovery_steps
            ]
    
    return ErrorGuidance(
        error_summary=summary,
        recovery_steps=recovery_steps,
        anti_patterns=anti_patterns,
        example_fix=example_fix,
        related_docs=template.get("related_docs", []),
        estimated_fix_time=template.get("estimated_fix_time", "unknown"),
        requires_human=error_category in [ErrorCategory.REFERENCE, ErrorCategory.UNKNOWN],
    )


def generate_guidance_from_log(
    error_log: str,
    context: Optional[Dict[str, Any]] = None,
) -> ErrorGuidance:
    """
    Generate guidance directly from error log content.
    
    Combines pattern matching and guidance generation.
    
    Args:
        error_log: Error log content
        context: Additional context
        
    Returns:
        ErrorGuidance based on detected error pattern
    """
    category, pattern_name, description = identify_error_category(error_log)
    
    guidance = generate_error_guidance(
        error_category=category,
        error_description=description or "",
        context=context,
    )
    
    # Add pattern info to metadata if found
    if pattern_name:
        logger.info(f"Matched error pattern: {pattern_name}")
    
    return guidance
