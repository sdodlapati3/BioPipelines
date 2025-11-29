"""
Error pattern database for bioinformatics pipeline failures.

Contains 50+ regex patterns organized by error category, with associated
fix suggestions and common causes.
"""

from typing import Dict, Optional
from .categories import (
    ErrorCategory, 
    ErrorPattern, 
    FixSuggestion, 
    FixRiskLevel
)


# =============================================================================
# PATTERN DATABASE
# =============================================================================

ERROR_PATTERNS: Dict[ErrorCategory, ErrorPattern] = {
    
    # -------------------------------------------------------------------------
    # FILE NOT FOUND ERRORS
    # -------------------------------------------------------------------------
    ErrorCategory.FILE_NOT_FOUND: ErrorPattern(
        category=ErrorCategory.FILE_NOT_FOUND,
        patterns=[
            r"No such file or directory[:\s]+(.+)",
            r"FileNotFoundError: \[Errno 2\].*'(.+)'",
            r"cannot open file '(.+)'",
            r"Error: Unable to open (.+)",
            r"Input file (.+) does not exist",
            r"Path not found: (.+)",
            r"Failed to open (.+)",
            r"File not found: (.+)",
            r"Error reading file: (.+)",
            r"Unable to locate (.+)",
        ],
        description="A required file or directory was not found",
        common_causes=[
            "Reference genome not downloaded",
            "Incorrect sample sheet paths",
            "Typo in file path",
            "Symlink pointing to deleted file",
            "Input files moved or renamed",
            "Wrong working directory",
        ],
        keywords=["file", "path", "directory", "open", "read", "missing"],
        suggested_fixes=[
            FixSuggestion(
                description="Verify the file path exists",
                command="ls -la {path}",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Create missing directory",
                command="mkdir -p {directory}",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Download missing reference genome",
                command="python -m workflow_composer.data.downloader download-reference {organism} {build}",
                risk_level=FixRiskLevel.LOW,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Check symlink validity",
                command="readlink -f {path}",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
        ],
    ),
    
    # -------------------------------------------------------------------------
    # OUT OF MEMORY ERRORS
    # -------------------------------------------------------------------------
    ErrorCategory.OUT_OF_MEMORY: ErrorPattern(
        category=ErrorCategory.OUT_OF_MEMORY,
        patterns=[
            r"MemoryError",
            r"Out of memory",
            r"Killed.*memory",
            r"slurmstepd: error:.*oom-kill",
            r"exceeded memory limit",
            r"Cannot allocate memory",
            r"std::bad_alloc",
            r"java\.lang\.OutOfMemoryError",
            r"FATAL:.*memory",
            r"memory allocation failed",
            r"mmap: Cannot allocate memory",
            r"ENOMEM",
            r"Insufficient memory",
            # Nextflow-specific OOM patterns
            r"Process requirement exceeds available memory",
            r"req: \d+ GB; avail: \d+ GB",
            r"exit status:\s*137",  # OOM kill exit code
            r"exit code:\s*137",
            r"Command exit status:\s*137",
            r"killed by signal 9",  # SIGKILL from OOM
            r"oom_reaper",
            r"invoked oom-killer",
        ],
        description="The process ran out of available memory",
        common_causes=[
            "Input files too large for allocated memory",
            "Too many parallel processes",
            "Memory leak in tool",
            "SLURM memory limit too low",
            "Large genome/reference",
        ],
        keywords=["memory", "oom", "kill", "allocate", "heap"],
        suggested_fixes=[
            FixSuggestion(
                description="Increase SLURM memory allocation (double current)",
                command="# Edit nextflow.config: memory = '64 GB' instead of '32 GB'",
                risk_level=FixRiskLevel.MEDIUM,
                auto_executable=False,
            ),
            FixSuggestion(
                description="Reduce number of parallel threads",
                command="# Set cpus = 4 instead of 8 in nextflow.config",
                risk_level=FixRiskLevel.MEDIUM,
                auto_executable=False,
            ),
            FixSuggestion(
                description="Retry the job with increased memory",
                command=None,
                risk_level=FixRiskLevel.LOW,
                auto_executable=False,
            ),
        ],
    ),
    
    # -------------------------------------------------------------------------
    # CONTAINER ERRORS
    # -------------------------------------------------------------------------
    ErrorCategory.CONTAINER_ERROR: ErrorPattern(
        category=ErrorCategory.CONTAINER_ERROR,
        patterns=[
            r"FATAL:.*container",
            r"singularity: command not found",
            r"Failed to pull container",
            r"Image not found: (.+\.sif)",
            r"FATAL:.*image file",
            r"Error loading image",
            r"container runtime error",
            r"OCI runtime error",
            r"unable to open image",
            r"Container image.*not found",
            r"failed to resolve reference",
            r"pull access denied",
            r"FATAL:.*Unable to",
            # Additional patterns for container issues
            r"Singularity cannot find",
            r"cannot find the container",
            r"container.*not found",
            r"no such file.*\.sif",
            r"unable to find container",
            r"image.*does not exist",
            r"docker pull.*failed",
        ],
        description="Container (Singularity/Docker) error",
        common_causes=[
            "Container image not built",
            "Singularity module not loaded",
            "Container registry unavailable",
            "Corrupted image file",
            "Permission issues with container cache",
        ],
        keywords=["container", "singularity", "docker", "image", "sif"],
        suggested_fixes=[
            FixSuggestion(
                description="Load Singularity module",
                command="module load singularity",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Build missing container",
                command="sbatch scripts/containers/build_{container}_container.slurm",
                risk_level=FixRiskLevel.LOW,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Pull container from registry",
                command="singularity pull docker://{image}",
                risk_level=FixRiskLevel.LOW,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Check container cache permissions",
                command="ls -la $SINGULARITY_CACHEDIR",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
        ],
    ),
    
    # -------------------------------------------------------------------------
    # PERMISSION ERRORS
    # -------------------------------------------------------------------------
    ErrorCategory.PERMISSION_DENIED: ErrorPattern(
        category=ErrorCategory.PERMISSION_DENIED,
        patterns=[
            r"Permission denied",
            r"EACCES",
            r"Operation not permitted",
            r"Access denied",
            r"cannot write to",
            r"Read-only file system",
            r"cannot create directory",
            r"cannot remove",
        ],
        description="Insufficient permissions to access file or directory",
        common_causes=[
            "File owned by another user",
            "Directory not writable",
            "Scratch space permissions",
            "Quota exceeded",
        ],
        keywords=["permission", "denied", "access", "write", "read"],
        suggested_fixes=[
            FixSuggestion(
                description="Check current permissions",
                command="ls -la {path}",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Fix file permissions",
                command="chmod u+rw {path}",
                risk_level=FixRiskLevel.LOW,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Check disk quota",
                command="quota -s",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
        ],
    ),
    
    # -------------------------------------------------------------------------
    # DEPENDENCY MISSING ERRORS
    # -------------------------------------------------------------------------
    ErrorCategory.DEPENDENCY_MISSING: ErrorPattern(
        category=ErrorCategory.DEPENDENCY_MISSING,
        patterns=[
            r"ModuleNotFoundError: No module named '(.+)'",
            r"ImportError: (.+)",
            r"command not found: (.+)",
            r"Package (.+) is not installed",
            r"Cannot find (.+) in PATH",
            r"Error: Unable to find (.+)",
            r"which: no (.+) in",
            r"(.+): not found",
            r"No module named (.+)",
            # Conda/Mamba dependency resolution failures
            r"LibMambaUnsatisfiableError",
            r"Could not solve for environment specs",
            r"Encountered problems while solving",
            r"The following packages are incompatible",
            r"package .+ requires .+, but none of the providers can be installed",
            r"CreateCondaEnvironmentException",
            r"Could not create conda environment",
            r"ResolvePackageNotFound",
            r"UnsatisfiableError",
            r"PackagesNotFoundError",
            r"EnvironmentCreationError",
            r"conda env create.*failed",
            r"mamba create.*failed",
            r"Solving environment:.*failed",
            # Snakemake environment issues
            r"Creating conda environment.*\n.*Exception",
            r"Failed to create conda environment",
        ],
        description="A required software dependency is missing or incompatible",
        common_causes=[
            "Python package not installed",
            "Tool not in PATH",
            "Module not loaded",
            "Conda environment not activated",
            "Incompatible package versions in environment.yml",
            "Package not available in specified conda channels",
            "Dependency conflict between packages",
            "Old/stale conda cache causing resolution issues",
        ],
        keywords=["module", "import", "package", "command", "found", "conda", "mamba", "environment", "solve", "incompatible"],
        suggested_fixes=[
            FixSuggestion(
                description="Check if tool exists in container",
                command="singularity exec {container} which {tool}",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Load required module",
                command="module load {module}",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Activate conda environment",
                command="conda activate biopipelines",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Install missing Python package",
                command="pip install {package}",
                risk_level=FixRiskLevel.MEDIUM,
                auto_executable=False,
            ),
            # Conda dependency resolution fixes
            FixSuggestion(
                description="Clear conda cache and retry",
                command="conda clean --all --yes",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Remove conflicting Snakemake conda environment",
                command="rm -rf .snakemake/conda/*",
                risk_level=FixRiskLevel.MEDIUM,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Use mamba for faster dependency resolution",
                command="conda install -n base mamba && snakemake --use-conda --conda-frontend mamba",
                risk_level=FixRiskLevel.MEDIUM,
                auto_executable=False,
            ),
            FixSuggestion(
                description="Pin problematic package to specific version in environment.yml",
                command="# Edit environment.yml to pin package versions",
                risk_level=FixRiskLevel.MEDIUM,
                auto_executable=False,
            ),
            FixSuggestion(
                description="Use --use-singularity instead of --use-conda",
                command="snakemake --use-singularity --singularity-args '-B /scratch'",
                risk_level=FixRiskLevel.MEDIUM,
                auto_executable=False,
            ),
        ],
    ),
    
    # -------------------------------------------------------------------------
    # REFERENCE/INDEX MISSING ERRORS
    # -------------------------------------------------------------------------
    ErrorCategory.REFERENCE_MISSING: ErrorPattern(
        category=ErrorCategory.REFERENCE_MISSING,
        patterns=[
            r"Reference genome not found",
            r"Cannot find reference",
            r"FASTA file not found",
            r"GTF/GFF file missing",
            r"Annotation file not found",
            r"genome\.fa.*not found",
            r"reference.*does not exist",
        ],
        description="Reference genome or annotation file is missing",
        common_causes=[
            "Reference not downloaded",
            "Wrong genome build specified",
            "Path misconfigured",
        ],
        keywords=["reference", "genome", "fasta", "gtf", "annotation"],
        suggested_fixes=[
            FixSuggestion(
                description="Download reference genome",
                command="python -m workflow_composer.data.downloader download-reference {organism} {build}",
                risk_level=FixRiskLevel.LOW,
                auto_executable=True,
            ),
            FixSuggestion(
                description="List available references",
                command="ls -la data/references/",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
        ],
    ),
    
    ErrorCategory.INDEX_MISSING: ErrorPattern(
        category=ErrorCategory.INDEX_MISSING,
        patterns=[
            r"Index (.+) not found",
            r"BWA index missing",
            r"STAR index not found",
            r"Bowtie2 index missing",
            r"Genome index does not exist",
            r"\.bwt file not found",
            r"Genome directory not found",
            r"SA file not found",
            r"SAindex not found",
            r"genomeDir.*not found",
        ],
        description="Aligner index files are missing",
        common_causes=[
            "Index not built",
            "Wrong index path",
            "Incomplete index build",
        ],
        keywords=["index", "bwa", "star", "bowtie", "genome"],
        suggested_fixes=[
            FixSuggestion(
                description="Build BWA index",
                command="bwa index {reference}",
                risk_level=FixRiskLevel.LOW,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Download pre-built index",
                command="python -m workflow_composer.data.downloader download-index {aligner} {organism} {build}",
                risk_level=FixRiskLevel.LOW,
                auto_executable=True,
            ),
        ],
    ),
    
    # -------------------------------------------------------------------------
    # SLURM ERRORS
    # -------------------------------------------------------------------------
    ErrorCategory.SLURM_ERROR: ErrorPattern(
        category=ErrorCategory.SLURM_ERROR,
        patterns=[
            r"SLURM.*CANCELLED",
            r"slurmstepd: error",
            r"DUE TO TIME LIMIT",
            r"JOB.*TIMEOUT",
            r"Exceeded job memory limit",
            r"srun: error",
            r"sbatch: error",
            r"PREEMPTED",
            r"NODE_FAIL",
            r"Job cancelled",
        ],
        description="SLURM scheduler error",
        common_causes=[
            "Job exceeded time limit",
            "Job exceeded memory limit",
            "Node failure",
            "Job preemption",
        ],
        keywords=["slurm", "srun", "sbatch", "timeout", "cancel"],
        suggested_fixes=[
            FixSuggestion(
                description="Increase time limit",
                command="# Modify --time=48:00:00 in nextflow.config",
                risk_level=FixRiskLevel.MEDIUM,
                auto_executable=False,
            ),
            FixSuggestion(
                description="Check job history",
                command="sacct -j {job_id} --format=JobID,State,ExitCode,MaxRSS,Elapsed",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Retry job",
                command="sbatch {script}",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
        ],
    ),
    
    # -------------------------------------------------------------------------
    # NETWORK ERRORS
    # -------------------------------------------------------------------------
    ErrorCategory.NETWORK_ERROR: ErrorPattern(
        category=ErrorCategory.NETWORK_ERROR,
        patterns=[
            r"Connection refused",
            r"Connection timed out",
            r"Network is unreachable",
            r"Could not resolve host",
            r"Failed to connect",
            r"SSL certificate problem",
            r"curl: \(\d+\)",
            r"wget: unable to resolve",
            r"Name or service not known",
            r"Connection reset by peer",
        ],
        description="Network connectivity error",
        common_causes=[
            "Server unreachable",
            "Firewall blocking",
            "DNS resolution failure",
            "SSL/TLS issues",
            "Compute nodes without internet",
        ],
        keywords=["network", "connection", "timeout", "dns", "ssl"],
        suggested_fixes=[
            FixSuggestion(
                description="Check network connectivity",
                command="ping -c 3 google.com",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Retry download with extended timeout",
                command="wget --retry-connrefused --waitretry=5 --timeout=60 {url}",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
        ],
    ),
    
    # -------------------------------------------------------------------------
    # TOOL ERRORS
    # -------------------------------------------------------------------------
    ErrorCategory.TOOL_ERROR: ErrorPattern(
        category=ErrorCategory.TOOL_ERROR,
        patterns=[
            r"Error executing process.*>.*'(\w+)'",
            r"Command exited with non-zero status",
            r"Exit status: (\d+)",
            r"Command error:",
            r"failed with exit code",
            r"returned non-zero exit status",
            r"Process.*terminated with status (\d+)",
            r"ERROR ~",
        ],
        description="Bioinformatics tool returned an error",
        common_causes=[
            "Invalid input data",
            "Wrong parameters",
            "Tool bug",
            "Incompatible file format",
        ],
        keywords=["error", "exit", "status", "process", "fail"],
        suggested_fixes=[
            FixSuggestion(
                description="Check process error log",
                command="cat {work_dir}/.command.err",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Check process command",
                command="cat {work_dir}/.command.sh",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Validate input files",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=False,
            ),
        ],
    ),
    
    # -------------------------------------------------------------------------
    # DATA FORMAT ERRORS
    # -------------------------------------------------------------------------
    ErrorCategory.DATA_FORMAT_ERROR: ErrorPattern(
        category=ErrorCategory.DATA_FORMAT_ERROR,
        patterns=[
            r"Invalid FASTQ",
            r"Malformed BAM",
            r"VCF format error",
            r"Invalid BED format",
            r"Truncated file",
            r"Unexpected end of file",
            r"gzip: (.+): not in gzip format",
            r"samtools.*invalid",
            r"bgzip: (.+): invalid",
            r"Corrupt",
        ],
        description="Input data format is invalid or corrupted",
        common_causes=[
            "Corrupted file download",
            "Wrong file format",
            "Incomplete file transfer",
            "Compression issue",
        ],
        keywords=["format", "invalid", "corrupt", "truncate", "gzip"],
        suggested_fixes=[
            FixSuggestion(
                description="Check file integrity",
                command="md5sum {file}",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Check if file is gzipped",
                command="file {file}",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Re-download file",
                risk_level=FixRiskLevel.LOW,
                auto_executable=False,
            ),
        ],
    ),
    
    # -------------------------------------------------------------------------
    # NEXTFLOW SPECIFIC ERRORS
    # -------------------------------------------------------------------------
    ErrorCategory.NEXTFLOW_ERROR: ErrorPattern(
        category=ErrorCategory.NEXTFLOW_ERROR,
        patterns=[
            r"Pipeline failed",
            r"Execution halted",
            r"No processes to run",
            r"Missing required channel",
            r"Unable to parse config",
            r"Script compilation error",
            r"Invalid DSL",
            r"Duplicate process",
            r"Channel.*is empty",
        ],
        description="Nextflow-specific execution error",
        common_causes=[
            "Workflow syntax error",
            "Missing input channel",
            "Configuration issue",
            "DSL version mismatch",
        ],
        keywords=["nextflow", "pipeline", "channel", "process", "dsl"],
        suggested_fixes=[
            FixSuggestion(
                description="Check Nextflow log",
                command="cat .nextflow.log | tail -100",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Validate workflow syntax",
                command="nextflow run {workflow} -validate",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
        ],
    ),
    
    # -------------------------------------------------------------------------
    # CONFIGURATION ERRORS
    # -------------------------------------------------------------------------
    ErrorCategory.CONFIGURATION_ERROR: ErrorPattern(
        category=ErrorCategory.CONFIGURATION_ERROR,
        patterns=[
            r"Invalid configuration",
            r"Config file.*not found",
            r"Missing required parameter",
            r"Unknown config option",
            r"Parse error in config",
            r"Unable to parse params",
        ],
        description="Configuration file error",
        common_causes=[
            "Missing config file",
            "Syntax error in config",
            "Invalid parameter value",
        ],
        keywords=["config", "parameter", "setting", "option"],
        suggested_fixes=[
            FixSuggestion(
                description="Check configuration file",
                command="cat nextflow.config",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
        ],
    ),
    
    # -------------------------------------------------------------------------
    # RESOURCE LIMIT ERRORS (disk, quota, etc.)
    # -------------------------------------------------------------------------
    ErrorCategory.RESOURCE_LIMIT: ErrorPattern(
        category=ErrorCategory.RESOURCE_LIMIT,
        patterns=[
            r"No space left on device",
            r"Disk quota exceeded",
            r"ENOSPC",
            r"cannot create.*No space",
            r"write error: No space",
            r"output file is too large",
            r"Quota exceeded",
            r"disk full",
            r"not enough space",
            r"insufficient disk",
        ],
        description="Disk space or quota limit exceeded",
        common_causes=[
            "Work directory full",
            "Output partition full",
            "User quota exceeded",
            "Too many intermediate files",
        ],
        keywords=["disk", "space", "quota", "full", "limit"],
        suggested_fixes=[
            FixSuggestion(
                description="Check disk usage",
                command="df -h .",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Check user quota",
                command="quota -s 2>/dev/null || echo 'quota command not available'",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Clean Nextflow work directory",
                command="nextflow clean -f",
                risk_level=FixRiskLevel.LOW,
                auto_executable=True,
            ),
            FixSuggestion(
                description="Find large files in work directory",
                command="find work/ -type f -size +1G -exec ls -lh {} \\;",
                risk_level=FixRiskLevel.SAFE,
                auto_executable=True,
            ),
        ],
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_pattern(category: ErrorCategory) -> Optional[ErrorPattern]:
    """Get pattern definition for a category."""
    return ERROR_PATTERNS.get(category)


def get_all_patterns() -> Dict[ErrorCategory, ErrorPattern]:
    """Get all error patterns."""
    return ERROR_PATTERNS


def count_patterns() -> int:
    """Count total number of regex patterns."""
    total = 0
    for pattern_def in ERROR_PATTERNS.values():
        total += len(pattern_def.patterns)
    return total


# Print pattern count when module loads (for debugging)
# print(f"Loaded {count_patterns()} error patterns across {len(ERROR_PATTERNS)} categories")
