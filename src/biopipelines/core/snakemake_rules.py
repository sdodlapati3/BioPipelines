"""
Reusable Snakemake rules for BioPipelines.

This module provides common rule templates that can be imported and customized
by individual pipelines, reducing code duplication and improving maintainability.

Usage in Snakefile:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
    
    from biopipelines.core.snakemake_rules import create_fastqc_rule, create_fastp_rule
    
    # Create customized rules
    rule fastqc_raw:
        **create_fastqc_rule(
            raw_dir="/path/to/raw",
            qc_dir="/path/to/qc",
            threads=2
        )
"""

def create_fastqc_rule(raw_dir, qc_dir, threads=2, paired_end=True):
    """
    Create a FastQC rule for quality control.
    
    Args:
        raw_dir: Directory containing raw FASTQ files
        qc_dir: Directory for QC output
        threads: Number of threads (default: 2)
        paired_end: Whether samples are paired-end (default: True)
    
    Returns:
        dict: Rule parameters for Snakemake
    """
    if paired_end:
        input_files = lambda w: [f"{raw_dir}/{w.sample}_R1.fastq.gz", 
                                 f"{raw_dir}/{w.sample}_R2.fastq.gz"]
        output_files = {
            "r1": f"{qc_dir}/fastqc/{{sample}}_R1_fastqc.html",
            "r2": f"{qc_dir}/fastqc/{{sample}}_R2_fastqc.html"
        }
    else:
        input_files = f"{raw_dir}/{{sample}}.fastq.gz"
        output_files = f"{qc_dir}/fastqc/{{sample}}_fastqc.html"
    
    return {
        "input": input_files,
        "output": output_files,
        "conda": "envs/qc.yaml",
        "threads": threads,
        "log": f"{qc_dir}/logs/fastqc/{{sample}}.log",
        "shell": f"mkdir -p {qc_dir}/fastqc && fastqc -t {{threads}} {{input}} -o {qc_dir}/fastqc/ 2> {{log}}"
    }


def create_fastp_rule(raw_dir, processed_dir, qc_dir, threads=4, 
                     min_length=50, quality_cutoff=20, paired_end=True):
    """
    Create a fastp rule for adapter trimming and quality filtering.
    
    Args:
        raw_dir: Directory containing raw FASTQ files
        processed_dir: Directory for processed output
        qc_dir: Directory for QC reports
        threads: Number of threads (default: 4)
        min_length: Minimum read length after trimming (default: 50)
        quality_cutoff: Quality score cutoff (default: 20)
        paired_end: Whether samples are paired-end (default: True)
    
    Returns:
        dict: Rule parameters for Snakemake
    """
    if paired_end:
        return {
            "input": {
                "r1": f"{raw_dir}/{{sample}}_R1.fastq.gz",
                "r2": f"{raw_dir}/{{sample}}_R2.fastq.gz"
            },
            "output": {
                "r1": f"{processed_dir}/{{sample}}_R1.trimmed.fastq.gz",
                "r2": f"{processed_dir}/{{sample}}_R2.trimmed.fastq.gz",
                "html": f"{qc_dir}/fastp/{{sample}}.html",
                "json": f"{qc_dir}/fastp/{{sample}}.json"
            },
            "conda": "envs/preprocessing.yaml",
            "threads": threads,
            "log": f"{qc_dir}/logs/trim/{{sample}}.log",
            "shell": f"""
                fastp -i {{input.r1}} -I {{input.r2}} \\
                      -o {{output.r1}} -O {{output.r2}} \\
                      --thread {{threads}} \\
                      --html {{output.html}} \\
                      --json {{output.json}} \\
                      --detect_adapter_for_pe \\
                      --cut_right \\
                      --cut_right_window_size 4 \\
                      --cut_right_mean_quality {quality_cutoff} \\
                      --length_required {min_length} \\
                      2> {{log}}
            """
        }
    else:
        return {
            "input": f"{raw_dir}/{{sample}}.fastq.gz",
            "output": {
                "trimmed": f"{processed_dir}/{{sample}}.trimmed.fastq.gz",
                "html": f"{qc_dir}/fastp/{{sample}}.html",
                "json": f"{qc_dir}/fastp/{{sample}}.json"
            },
            "conda": "envs/preprocessing.yaml",
            "threads": threads,
            "log": f"{qc_dir}/logs/trim/{{sample}}.log",
            "shell": f"""
                fastp -i {{input}} -o {{output.trimmed}} \\
                      --thread {{threads}} \\
                      --html {{output.html}} \\
                      --json {{output.json}} \\
                      --cut_right \\
                      --cut_right_window_size 4 \\
                      --cut_right_mean_quality {quality_cutoff} \\
                      --length_required {min_length} \\
                      2> {{log}}
            """
        }


def create_bwa_alignment_rule(processed_dir, qc_dir, reference, threads=8):
    """
    Create a BWA-MEM alignment rule.
    
    Args:
        processed_dir: Directory for processed files
        qc_dir: Directory for QC/logs
        reference: Path to reference genome FASTA
        threads: Number of threads (default: 8)
    
    Returns:
        dict: Rule parameters for Snakemake
    """
    return {
        "input": {
            "r1": f"{processed_dir}/{{sample}}_R1.trimmed.fastq.gz",
            "r2": f"{processed_dir}/{{sample}}_R2.trimmed.fastq.gz",
            "ref": reference
        },
        "output": {
            "bam": f"{processed_dir}/{{sample}}.sorted.bam",
            "bai": f"{processed_dir}/{{sample}}.sorted.bam.bai"
        },
        "params": {
            "rg": "@RG\\tID:{sample}\\tSM:{sample}\\tPL:ILLUMINA\\tLB:{sample}\\tPU:unit1"
        },
        "conda": "envs/alignment.yaml",
        "threads": threads,
        "log": f"{qc_dir}/logs/align/{{sample}}.log",
        "shell": """
            bwa mem -t {threads} -R '{params.rg}' {input.ref} {input.r1} {input.r2} 2> {log} | \\
            samtools sort -@ {threads} -o {output.bam} - 2>> {log}
            samtools index {output.bam}
        """
    }


def create_bowtie2_alignment_rule(processed_dir, qc_dir, reference_prefix, 
                                   threads=8, max_insert=2000):
    """
    Create a Bowtie2 alignment rule.
    
    Args:
        processed_dir: Directory for processed files
        qc_dir: Directory for QC/logs
        reference_prefix: Path to Bowtie2 index prefix (without .1.bt2 extension)
        threads: Number of threads (default: 8)
        max_insert: Maximum insert size for paired-end (default: 2000)
    
    Returns:
        dict: Rule parameters for Snakemake
    """
    return {
        "input": {
            "r1": f"{processed_dir}/{{sample}}_R1.trimmed.fastq.gz",
            "r2": f"{processed_dir}/{{sample}}_R2.trimmed.fastq.gz"
        },
        "output": {
            "bam": f"{processed_dir}/{{sample}}.sorted.bam",
            "bai": f"{processed_dir}/{{sample}}.sorted.bam.bai"
        },
        "params": {
            "ref": reference_prefix,
            "max_insert": max_insert
        },
        "conda": "envs/alignment.yaml",
        "threads": threads,
        "log": f"{qc_dir}/logs/align/{{sample}}.log",
        "shell": """
            bowtie2 -p {threads} -X {params.max_insert} -x {params.ref} \\
                    -1 {input.r1} -2 {input.r2} 2> {log} | \\
            samtools sort -@ {threads} -o {output.bam} - 2>> {log}
            samtools index {output.bam}
        """
    }


def create_mark_duplicates_rule(processed_dir, qc_dir, remove_duplicates=True):
    """
    Create a Picard MarkDuplicates rule.
    
    Args:
        processed_dir: Directory for processed files
        qc_dir: Directory for QC metrics
        remove_duplicates: Whether to remove duplicates (default: True)
    
    Returns:
        dict: Rule parameters for Snakemake
    """
    remove_flag = "true" if remove_duplicates else "false"
    
    return {
        "input": f"{processed_dir}/{{sample}}.sorted.bam",
        "output": {
            "bam": f"{processed_dir}/{{sample}}.dedup.bam",
            "metrics": f"{qc_dir}/picard/{{sample}}.dedup_metrics.txt"
        },
        "conda": "envs/alignment.yaml",
        "threads": 4,
        "log": f"{qc_dir}/logs/dedup/{{sample}}.log",
        "shell": f"""
            gatk MarkDuplicates \\
                -I {{input}} \\
                -O {{output.bam}} \\
                -M {{output.metrics}} \\
                --REMOVE_DUPLICATES {remove_flag} \\
                --CREATE_INDEX true \\
                2> {{log}}
        """
    }


def create_multiqc_rule(qc_dir, results_dir):
    """
    Create a MultiQC aggregation rule.
    
    Args:
        qc_dir: Directory containing all QC outputs
        results_dir: Directory for final reports
    
    Returns:
        dict: Rule parameters for Snakemake
    """
    return {
        "input": qc_dir,
        "output": f"{results_dir}/multiqc_report.html",
        "conda": "envs/qc.yaml",
        "log": f"{qc_dir}/logs/multiqc.log",
        "shell": f"""
            multiqc {qc_dir} -o {results_dir} -f --no-data-dir 2> {{log}}
        """
    }


# Helper function to generate common pipeline directories
def get_pipeline_directories(base_dir, pipeline_name):
    """
    Generate standard directory structure for a pipeline.
    
    Args:
        base_dir: Base data directory (e.g., /scratch/user/BioPipelines/data)
        pipeline_name: Name of the pipeline (e.g., "chip_seq", "atac_seq")
    
    Returns:
        dict: Dictionary with standard directory paths
    """
    return {
        "RAW_DIR": f"{base_dir}/raw/{pipeline_name}",
        "PROCESSED_DIR": f"{base_dir}/processed/{pipeline_name}",
        "RESULTS_DIR": f"{base_dir}/results/{pipeline_name}",
        "QC_DIR": f"{base_dir}/results/{pipeline_name}/qc",
        "REFERENCE_DIR": f"{base_dir}/references"
    }


# Example usage documentation
USAGE_EXAMPLE = """
Example Snakefile using shared rules:

```python
import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from biopipelines.core.snakemake_rules import (
    create_fastqc_rule,
    create_fastp_rule,
    create_bwa_alignment_rule,
    create_mark_duplicates_rule,
    create_multiqc_rule,
    get_pipeline_directories
)

# Configuration
configfile: "config.yaml"

# Setup directories
DIRS = get_pipeline_directories(
    base_dir="/scratch/user/BioPipelines/data",
    pipeline_name="chip_seq"
)
RAW_DIR = DIRS["RAW_DIR"]
PROCESSED_DIR = DIRS["PROCESSED_DIR"]
RESULTS_DIR = DIRS["RESULTS_DIR"]
QC_DIR = DIRS["QC_DIR"]

SAMPLES = config["samples"]
REFERENCE = config["reference"]["genome"]

# Rules using shared templates
rule fastqc:
    **create_fastqc_rule(
        raw_dir=RAW_DIR,
        qc_dir=QC_DIR,
        threads=2,
        paired_end=True
    )

rule trim_reads:
    **create_fastp_rule(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        qc_dir=QC_DIR,
        threads=4,
        min_length=50,
        quality_cutoff=20,
        paired_end=True
    )

rule align_reads:
    **create_bwa_alignment_rule(
        processed_dir=PROCESSED_DIR,
        qc_dir=QC_DIR,
        reference=REFERENCE,
        threads=8
    )

rule mark_duplicates:
    **create_mark_duplicates_rule(
        processed_dir=PROCESSED_DIR,
        qc_dir=QC_DIR,
        remove_duplicates=True
    )

rule multiqc:
    **create_multiqc_rule(
        qc_dir=QC_DIR,
        results_dir=RESULTS_DIR
    )

# Pipeline-specific rules follow...
rule call_peaks:
    input:
        f"{PROCESSED_DIR}/{{sample}}.dedup.bam"
    output:
        f"{RESULTS_DIR}/peaks/{{sample}}_peaks.narrowPeak"
    # ... peak calling logic
```

Benefits:
- Reduces code duplication across 10 pipelines
- Centralizes rule logic for easier maintenance
- Consistent parameter naming and structure
- Easy to update QC/preprocessing steps globally
- Well-documented and type-hinted
"""
